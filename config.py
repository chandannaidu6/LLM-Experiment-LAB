from __future__ import annotations
from itertools import product
from typing import Dict, Any, List, Optional
from pydantic import BaseModel, Field, field_validator, model_validator



VALID_CHUNKERS = {"fixed_size", "recursive", "semantic", "sentence", "slide"}
VALID_RETRIEVERS = {"bm25", "dense", "hybrid", "hyde", "self_rag"}

VALID_EMBEDDING_PROVIDERS = {"openai", "huggingface", "cohere", "voyage"}
VALID_LLM_PROVIDERS = {"openai", "anthropic", "google", "cohere"}


NEEDS_EMBEDDER = {"dense", "hybrid", "hyde", "semantic"}   # semantic is a CHUNKER that embeds
# retrievers that require an LLM model
NEEDS_LLM = {"hyde", "self_rag"}


PROVIDER_MODEL_PREFIXES = {
    "openai": ("gpt-", "text-embedding-", "o1", "o3"),
    "anthropic": ("claude-",),
    "google": ("gemini-",),
    "huggingface": ("BAAI/", "sentence-transformers/", "intfloat/", "thenlper/"),
    "cohere": ("embed-", "command-", "rerank-"),
    "voyage": ("voyage-",),
}


def _check_provider_model(provider: str, model: Optional[str], kind: str) -> None:
    """Optional cross-check: warn-by-raising if a model clearly doesn't match
    its provider (e.g. provider='anthropic' but model='gpt-4.1')."""
    if not model:
        return
    prefixes = PROVIDER_MODEL_PREFIXES.get(provider)
    if not prefixes:
        return  # unknown provider handled elsewhere; skip cross-check
    if not model.startswith(prefixes):
        raise ValueError(
            f"{kind} model '{model}' does not look like a '{provider}' model "
            f"(expected one of prefixes {prefixes}). If this is intentional, "
            f"add its prefix to PROVIDER_MODEL_PREFIXES."
        )


class RunConfig(BaseModel):
    """A single chunker x retriever experiment."""

    # ---- where the data is ----
    corpus_path: str
    benchmark_queries_path: str
    benchmark_qrels_path: str

    # ---- what to run ----
    chunker_name: str
    retriever_name: str

    # ---- settings (dicts: each chunker/retriever takes different args) ----
    chunker_params: Dict[str, Any] = Field(default_factory=dict)
    retriever_params: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5

    # ---- models (feed the factory + cost.py) ----
    embedding_provider: str = "openai"
    embedding_model: Optional[str] = "text-embedding-3-small"
    llm_provider: str = "openai"
    llm_model: Optional[str] = None

    # ---- output ----
    output_path: str = "results"

    # ---------- field validators ----------
    @field_validator("chunker_name")
    @classmethod
    def _known_chunker(cls, v: str) -> str:
        if v not in VALID_CHUNKERS:
            raise ValueError(
                f"Unknown chunker '{v}'. Available: {', '.join(sorted(VALID_CHUNKERS))}"
            )
        return v

    @field_validator("retriever_name")
    @classmethod
    def _known_retriever(cls, v: str) -> str:
        if v not in VALID_RETRIEVERS:
            raise ValueError(
                f"Unknown retriever '{v}'. Available: {', '.join(sorted(VALID_RETRIEVERS))}"
            )
        return v

    @field_validator("embedding_provider")
    @classmethod
    def _known_embed_provider(cls, v: str) -> str:
        if v not in VALID_EMBEDDING_PROVIDERS:
            raise ValueError(
                f"Unknown embedding_provider '{v}'. "
                f"Available: {', '.join(sorted(VALID_EMBEDDING_PROVIDERS))}"
            )
        return v

    @field_validator("llm_provider")
    @classmethod
    def _known_llm_provider(cls, v: str) -> str:
        if v not in VALID_LLM_PROVIDERS:
            raise ValueError(
                f"Unknown llm_provider '{v}'. "
                f"Available: {', '.join(sorted(VALID_LLM_PROVIDERS))}"
            )
        return v

    @field_validator("top_k")
    @classmethod
    def _positive_k(cls, v: int) -> int:
        if v < 1:
            raise ValueError("top_k must be >= 1")
        return v

    # ---------- model-level validator ----------
    @model_validator(mode="after")
    def _consistency(self) -> "RunConfig":
        # (2) NEEDS_EMBEDDER is the single source of truth, and (3) it now
        # includes the semantic CHUNKER, so check chunker OR retriever.
        needs_embed = (self.retriever_name in NEEDS_EMBEDDER
                       or self.chunker_name in NEEDS_EMBEDDER)
        if needs_embed and not self.embedding_model:
            culprit = self.chunker_name if self.chunker_name in NEEDS_EMBEDDER else self.retriever_name
            raise ValueError(f"'{culprit}' needs an embedding_model.")

        # retrievers that generate need an LLM model
        if self.retriever_name in NEEDS_LLM and not self.llm_model:
            raise ValueError(f"retriever '{self.retriever_name}' needs an llm_model.")

        # (4) optional provider<->model cross-check
        _check_provider_model(self.embedding_provider, self.embedding_model, "embedding")
        if self.retriever_name in NEEDS_LLM:
            _check_provider_model(self.llm_provider, self.llm_model, "llm")

        return self


class MatrixConfig(BaseModel):
    """Lists of chunkers x retrievers -> expands into many RunConfigs."""

    corpus_path: str
    benchmark_queries_path: str
    benchmark_qrels_path: str

    chunker_names: List[str]
    retriever_names: List[str]

    chunker_params: Dict[str, Any] = Field(default_factory=dict)
    retriever_params: Dict[str, Any] = Field(default_factory=dict)
    top_k: int = 5

    embedding_provider: str = "openai"
    embedding_model: Optional[str] = "text-embedding-3-small"
    llm_provider: str = "openai"
    llm_model: Optional[str] = None
    output_path: str = "results"

    def expand(self) -> List[RunConfig]:
        """Cartesian product -> one RunConfig each. Each re-validates."""
        runs: List[RunConfig] = []
        for chunker, retriever in product(self.chunker_names, self.retriever_names):
            runs.append(RunConfig(
                corpus_path=self.corpus_path,
                benchmark_queries_path=self.benchmark_queries_path,
                benchmark_qrels_path=self.benchmark_qrels_path,
                chunker_name=chunker,
                retriever_name=retriever,
                chunker_params=self.chunker_params,
                retriever_params=self.retriever_params,
                top_k=self.top_k,
                embedding_provider=self.embedding_provider,
                embedding_model=self.embedding_model,
                llm_provider=self.llm_provider,
                llm_model=self.llm_model,
                output_path=self.output_path,
            ))
        return runs