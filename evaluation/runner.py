from __future__ import annotations

from typing import List, Dict, Any, Optional
from pydantic import BaseModel

from schema import Query, Chunk, RetrievalResult, RetrievedChunk, Usage
from config import RunConfig, MatrixConfig
import loader
from benchmark import Benchmark
from synthetic_benchmark import SyntheticGenerator
from base import BaseChunker            
from registry import Registry as ChunkerRegistry          
from retrievers_registry import RegistryRetriever         
from metrics_accuracy import Evaluation                  
from latency import Timer, summarize                      
from cost import estimate_cost, aggregate_costs            
from usage import UsageTracker, TrackedEmbedder, TrackedLLMClient 


class RunResult(BaseModel):
    chunker_name: str
    retriever_name: str
    benchmark_type: str                 
    num_queries: int
    num_chunks: int
    index_build_ms: float
    accuracy: Dict[str, float]          
    latency: Dict[str, float]           
    cost: Dict[str, float]              
    errors: List[str] = []              
    failed: bool = False                
    error_message: Optional[str] = None


def load_and_chunk(config: RunConfig) -> List[Dict[str, Any]]:
    result = loader.load(config.corpus_path)

    if not result.needs_chunking:
        return [c.model_dump() for c in result.chunks]

    chunker_registry = ChunkerRegistry(
        pdf_path=str(result.source_path),
        name=config.chunker_name,
        embedder=None,                    # semantic chunker builds its own default embedder
        **config.chunker_params,
    )
    chunker = chunker_registry.get_chunker()
    chunks = chunker.build_chunks()
    # normalize to dicts regardless of whether the chunker returns Chunk objects or dicts
    return [c.model_dump() if hasattr(c, "model_dump") else c for c in chunks]


# ---------------------------------------------------------------------------
# Stage 3: queries (real benchmark, or synthetic if none provided)
# ---------------------------------------------------------------------------

def get_queries(config: RunConfig, chunks: List[Dict[str, Any]],
                llm_client=None) -> (List[Query], str):
    has_real_benchmark = bool(config.benchmark_queries_path and config.benchmark_qrels_path)

    if has_real_benchmark:
        bench = Benchmark(
            qrel_path=config.benchmark_qrels_path,
            question_path=config.benchmark_queries_path,
            chunks=[Chunk(**c) for c in chunks],
        )
        return bench.load(), "real"

    if llm_client is None:
        raise ValueError(
            "No benchmark files provided and no llm_client available to "
            "generate a synthetic benchmark."
        )
    gen = SyntheticGenerator(llm_client=llm_client, chunks=chunks)
    return gen.generate_benchmark_file(), "synthetic"


# ---------------------------------------------------------------------------
# Stage 4: build the retriever (+ its dependencies)
# ---------------------------------------------------------------------------

def build_retriever(config: RunConfig, chunks: List[Dict[str, Any]],
                    tracker: UsageTracker):
    """The wiring step: some retrievers need other retrievers/clients built
    first. Kept explicit and inline rather than hidden in the registry, so
    the dependency order is visible and easy to extend."""

    retriever_registry = RegistryRetriever(name=config.retriever_name)

    if config.retriever_name == "bm25":
        retriever = retriever_registry.get_retriever(chunks=chunks)

    elif config.retriever_name == "dense":
        embedder = _build_tracked_embedder(config, tracker)
        retriever = retriever_registry.get_retriever(chunks=chunks, embedder=embedder)

    elif config.retriever_name == "hybrid":
        embedder = _build_tracked_embedder(config, tracker)
        dense = RegistryRetriever(name="dense").get_retriever(chunks=chunks, embedder=embedder)
        dense.build_index()
        retriever = retriever_registry.get_retriever(
            bm25_chunks=chunks, dense_retriever=dense,
        )

    elif config.retriever_name == "hyde":
        embedder = _build_tracked_embedder(config, tracker)
        llm_client = _build_tracked_llm(config, tracker)
        retriever = retriever_registry.get_retriever(
            llm_client=llm_client, embedder=embedder,
        )

    elif config.retriever_name == "self_rag":
        embedder = _build_tracked_embedder(config, tracker)
        base = RegistryRetriever(name="dense").get_retriever(chunks=chunks, embedder=embedder)
        base.build_index()
        llm_client = _build_tracked_llm(config, tracker)
        retriever = retriever_registry.get_retriever(
            base_retriever=base, judge_fn=llm_client.chat,
        )

    else:
        raise ValueError(f"No wiring defined for retriever '{config.retriever_name}'")

    return retriever


def _build_tracked_embedder(config: RunConfig, tracker: UsageTracker):
    from embeddings.openai_embedder import OpenAIEmbedder
    raw = OpenAIEmbedder(model=config.embedding_model)
    return TrackedEmbedder(raw, tracker)


def _build_tracked_llm(config: RunConfig, tracker: UsageTracker):
    from llm.openai_client import OpenAIClient
    raw = OpenAIClient(model=config.llm_model)
    return TrackedLLMClient(raw, tracker)


# ---------------------------------------------------------------------------
# Stages 5-7: index, query loop, aggregate
# ---------------------------------------------------------------------------

def run_single(config: RunConfig, llm_client_for_synthetic=None) -> RunResult:
    try:
        # 1-2: load + chunk
        chunks = load_and_chunk(config)

        # 3: queries
        queries, benchmark_type = get_queries(config, chunks, llm_client_for_synthetic)

        # 4: build retriever (+ dependencies), with usage tracking wired in
        tracker = UsageTracker()
        retriever = build_retriever(config, chunks, tracker)

        # 5: build index (timed separately - one-time setup cost)
        with Timer() as index_timer:
            retriever.build_index()

        # 6: per-query loop
        per_query_metrics: List[Dict[str, float]] = []
        per_query_latencies: List[float] = []
        per_query_costs: List[float] = []
        total_tokens = 0
        errors: List[str] = []

        cost_model = config.llm_model if config.retriever_name in {"hyde", "self_rag"} \
            else config.embedding_model

        for query in queries:
            try:
                tracker.reset()
                with Timer() as t:
                    raw_results = retriever.retrieve(query.text, top_k=config.top_k)

                usage_snapshot = tracker.snapshot()
                doc_ids = [r.get("doc_name") if isinstance(r, dict) else r.doc_name
                          for r in raw_results]

                ev = Evaluation(retrieved_ids=doc_ids, source_ids=query.gold_doc_ids,
                                k=config.top_k)
                per_query_metrics.append({
                    "hit_at_k": ev.hit_at_k(),
                    "hit_rate_at_k": float(ev.hit_rate_k()),
                    "precision_at_k": ev.precision_at_k(),
                    "recall_at_k": ev.recall_at_k(),
                    "mrr": ev.mrr_at_k(),
                    "ndcg_at_k": ev.ndcg_at_k(query.gold_relevance or None),
                })
                per_query_latencies.append(t.ms)

                cost_result = estimate_cost(usage_snapshot, cost_model or "unknown")
                per_query_costs.append(cost_result["total"])
                total_tokens += (usage_snapshot.prompt_tokens
                                + usage_snapshot.completion_tokens
                                + usage_snapshot.embedding_tokens)

            except Exception as e:
                errors.append(f"{query.query_id}: {e}")
                continue

        # 7: aggregate
        accuracy = _mean_metrics(per_query_metrics)
        latency_summary = summarize(per_query_latencies)
        cost_summary = aggregate_costs(per_query_costs, total_tokens, len(per_query_metrics))

        return RunResult(
            chunker_name=config.chunker_name,
            retriever_name=config.retriever_name,
            benchmark_type=benchmark_type,
            num_queries=len(queries),
            num_chunks=len(chunks),
            index_build_ms=index_timer.ms,
            accuracy=accuracy,
            latency=latency_summary,
            cost=cost_summary,
            errors=errors,
        )

    except Exception as e:
        # a whole-run failure (bad config, load failure, etc.) does not
        # take down the rest of a matrix - it's recorded and skipped
        return RunResult(
            chunker_name=config.chunker_name,
            retriever_name=config.retriever_name,
            benchmark_type="n/a",
            num_queries=0,
            num_chunks=0,
            index_build_ms=0.0,
            accuracy={}, latency={}, cost={},
            failed=True,
            error_message=str(e),
        )


def run_matrix(matrix_config: MatrixConfig, llm_client_for_synthetic=None) -> List[RunResult]:
    results: List[RunResult] = []
    for run_config in matrix_config.expand():
        results.append(run_single(run_config, llm_client_for_synthetic))
    return results


def _mean_metrics(per_query: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_query:
        return {}
    keys = per_query[0].keys()
    return {k: sum(m[k] for m in per_query) / len(per_query) for k in keys}