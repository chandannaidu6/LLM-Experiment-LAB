from fastapi import APIRouter,HTTPException,Request,Depends
from starlette.concurrency import run_in_threadpool

from api.models import (
    RetrieveRequest,
    HydeRetrieveRequest,
    RetrievedResponse,
    RetrievedDocs,
    RagRequest,
    HydeRagRequest,
    EvaluateRequest,
    RagResponse,
    EvaluationMetrics,
    EvaluateResponse,
    HealthResponse,
    CompareEvaluationResult,
    CompareEvaluateResponse
)

from api.dependencies import normalize_doc_name
from retrievers.bm25 import BM25
from retrievers.dense import DenseRetriever
from reranker.cross_encoder import Reranker
from randomize import Randomize
from evaluation.metrics import Evaluation
from api.dependencies import get_hyde_retriever
from retrievers.hyde import HydeRetriever
import time

router = APIRouter(prefix="/v1",tags=["rag"])

def bm25_retriever(query,bm25_chunks,vocabulary,stats,top_k=5):
    bm = BM25(query,bm25_chunks,vocabulary)
    bm.stats = stats
    return bm.bm25_retriever(top_k=top_k)

def dense_retriever(query,dense,top_k=5):
    return dense.retrieve(query,top_k)

def to_retrieved_docs(chunks):
    results = []
    for c in chunks:
        text = c['text']
        if isinstance(text,list):
            text = " ".join(text)

        pages = c.get("pages")
        if pages is None:
            page_number = c.get("page_number")
            pages = [page_number] if page_number is not None else []

        results.append(
            RetrievedDocs(
                doc_name=c['doc_name'],
                chunk_id=c.get("chunk_id"),
                score=float(c['score']),
                pages=pages,
                text=text,
                rerank_score = c.get("rerank_score"),

            )
        )

    return results

async def rag_answer_async(query,client,chunks,mode="plain"):
    context = "\n\n".join(f"[{i+1}] {c['doc_name']} (chunk {c['chunk_id']}):{c['text']}" for i,c in enumerate(chunks))
    if hasattr(client,"achat"):
        return await client.achat(query,context,mode=mode)
    return await run_in_threadpool(client.chat,query,context,mode)

def unique_in_order(items):
    seen = set()
    out = []
    for x in items:
        if x not in seen:
            seen.add(x)
            out.append(x)

    return out

def build_eval_response(retriever,qid,results,questions,qrels,top_k):
    q = next((q for q in questions if q['id'] == qid),None)
    if q is None:
        raise HTTPException(status_code=404,detail="Question does not exist")
    retrieved_ids = unique_in_order([r['doc_name'] for r in results])
    gold_raw_docs = qrels.get(qid,[])
    gold_docs = [str(d) for d in gold_raw_docs]
    ev = Evaluation(retrieved_ids,gold_docs,k=min(top_k,len(retrieved_ids)))
    primary_gold_doc = gold_docs[0] if gold_docs else None
    rank = gold_rank(results[:top_k],primary_gold_doc) if primary_gold_doc else None
    return EvaluateResponse(
        retriever=retriever,
        question_id=qid,
        gold_doc=gold_docs[0] if gold_docs else None,
        gold_rank = rank,
        retrieved_docs=retrieved_ids,
        metrics=EvaluationMetrics(
            hit_at_k=ev.hit_at_k(),
            hit_rate_at_k=ev.hit_rate_k(),
            precision_at_k=ev.precision_at_k(),
            recall_at_k=ev.recall_at_k()
        )
    )

def find_gold_chunk(question_id,qrels,chunks):
    gold_docs = set(str(x) for x in qrels.get(question_id,[]))
    for c in chunks:
        if str(c.get("doc_name")) in gold_docs:
            return c
    return None

def gold_rank(results,gold_doc):
    ranked = [str(r["doc_name"]) for r in results]
    gold_doc = str(gold_doc)
    return ranked.index(gold_doc)+1 if gold_doc in ranked else None


async def compare_with_and_without_rerank(retriever_name:str,question_id:str,query:str,retriever_fn,retriever_args:tuple,request:Request,top_k:int,rerank_top_k:int | None=None):
    q = next((qq for qq in request.app.state.questions if qq["id"] == question_id),None)
    if q is None:
        raise HTTPException(status_code=404, detail="Question id not available")
    
    rerank_top_k = rerank_top_k or top_k
    gold_docs_raw = request.app.state.qrels.get(question_id,[])
    gold_docs = [str(d) for d in gold_docs_raw]

    baseline_results = await run_in_threadpool(
        retriever_fn,
        query,
        *retriever_args,
        top_k
    )    
    reranked_results = await run_in_threadpool(
        request.app.state.reranker.rerank,
        query,
        baseline_results,
        rerank_top_k or top_k
    )
    baseline_ids = unique_in_order([r["doc_name"] for r in baseline_results])
    reranked_ids = unique_in_order([r["doc_name"] for r in reranked_results])


    baseline_eval = Evaluation(
        baseline_ids,
        gold_docs,
        k = min(top_k,len(baseline_ids))
    )
    reranked_eval = Evaluation(
        reranked_ids,
        gold_docs,
        k = min(top_k,len(reranked_ids))
    )
    return CompareEvaluateResponse(
        retriever=retriever_name,
        question_id=question_id,
        gold_doc=gold_docs,
        top_k=top_k,
        rerank_top_k=rerank_top_k,
        baseline=CompareEvaluationResult(
            retrieved_docs=baseline_ids,
            metrics=EvaluationMetrics(
                hit_at_k=baseline_eval.hit_at_k(),
                hit_rate_at_k=baseline_eval.hit_rate_k(),
                precision_at_k=baseline_eval.precision_at_k(),
                recall_at_k=baseline_eval.recall_at_k(),
            ),
        ),
        reranked=CompareEvaluationResult(
            retrieved_docs = reranked_ids,
            metrics = EvaluationMetrics(
                hit_at_k = reranked_eval.hit_at_k(),
                hit_rate_at_k=reranked_eval.hit_rate_k(),
                precision_at_k=reranked_eval.precision_at_k(),
                recall_at_k=reranked_eval.recall_at_k(),
            )

        )

    )

@router.get("/health",response_model=HealthResponse)
async def health(request:Request):
    return HealthResponse(
        status="ok",
        chunks_loaded=len(request.app.state.chunks)
    )

@router.post("/retrieve/bm25",response_model=RetrievedResponse)
async def retrieve_bm25(payload:RetrieveRequest,request:Request):

    results = await run_in_threadpool(
        bm25_retriever,
        payload.query,
        request.app.state.bm25_chunks,
        request.app.state.vocab,
        request.app.state.stats,
        payload.top_k,
    )
    if getattr(payload,"use_reranker",False):
        results = await run_in_threadpool(
            request.app.state.reranker.rerank,
            payload.query,
            results,
            getattr(payload,"rerank_top_k",payload.top_k)
        )

        
    return RetrievedResponse(
        retriever="bm25",
        query=payload.query,
        top_k=payload.top_k,
        results=to_retrieved_docs(results)
    )

@router.post("/retrieve/dense",response_model=RetrievedResponse)
async def retrieve_dense(payload:RetrieveRequest,request:Request):
    results = await run_in_threadpool(
        dense_retriever,
        payload.query,
        request.app.state.dense_retriever,
        payload.top_k
    )
    if getattr(payload,"use_reranker",False):
        results = await run_in_threadpool(
            request.app.state.reranker.rerank,
            payload.query,
            results,
            getattr(payload,"rerank_top_k",payload.top_k)
        )

    
    return RetrievedResponse(
        retriever="dense",
        query=payload.query,
        top_k=payload.top_k,
        results=to_retrieved_docs(results)
    )

@router.post("/retrieve/hyde",response_model= RetrievedResponse)
async def retrieve_hyde(payload:HydeRetrieveRequest,request:Request,hyde:HydeRetriever = Depends(get_hyde_retriever)):
    results = await run_in_threadpool(
        hyde.retrieve,
        payload.query,
        payload.top_k,
        payload.hyde_mode
    )
    if getattr(payload,"use_reranker",False):
        results = await run_in_threadpool(
            request.app.state.reranker.rerank,
            payload.query,
            results,
            getattr(payload,"rerank_top_k",payload.top_k)
        )
    return RetrievedResponse(
        retriever="hyde",
        query=payload.query,
        top_k=payload.top_k,
        results=to_retrieved_docs(results),
    )

@router.post("/rag/bm25", response_model=RagResponse)
async def rag_bm25(payload: RagRequest, request: Request):
    tracker = request.app.state.mlflow
    mode = "cot" if "cot" in payload.prompt_version else "plain"

    with tracker.start_run(run_name=f"rag-bm25-{payload.prompt_version}"):
        tracker.log_params({
            "endpoint": "rag/bm25",
            "retriever": "bm25",
            "query": payload.query,
            "top_k": payload.top_k,
            "prompt_version": payload.prompt_version,
            "mode": mode,
            "use_reranker": getattr(payload, "use_reranker", False),
            "rerank_top_k": getattr(payload, "rerank_top_k", payload.top_k),
        })

        tracker.log_tags({
            "pipeline": "rag",
            "retrieval_type": "sparse",
        })

        start = time.perf_counter()

        results = await run_in_threadpool(
            bm25_retriever,
            payload.query,
            request.app.state.bm25_chunks,
            request.app.state.vocab,
            request.app.state.stats,
            payload.top_k
        )

        if getattr(payload, "use_reranker", False):
            results = await run_in_threadpool(
                request.app.state.reranker.rerank,
                payload.query,
                results,
                getattr(payload, "rerank_top_k", payload.top_k)
            )

        docs = to_retrieved_docs(results)

        answer = await rag_answer_async(
            payload.query,
            request.app.state.openai_client,
            [d.model_dump() for d in docs],
            mode=mode,
        )

        latency = time.perf_counter() - start

        tracker.log_metrics({
            "num_retrieved_docs": len(docs),
            "latency_seconds": latency,
        })

        return RagResponse(
            retriever="bm25",
            question=payload.query,
            prompt_version=payload.prompt_version,
            answer=answer,
            top_k=payload.top_k,
            retrieved_docs=docs
        )


@router.post("/rag/dense", response_model=RagResponse)
async def rag_dense(payload: RagRequest, request: Request):
    tracker = request.app.state.mlflow
    mode = "cot" if "cot" in payload.prompt_version else "plain"

    with tracker.start_run(run_name=f"rag-dense-{payload.prompt_version}"):
        tracker.log_params({
            "endpoint": "rag/dense",
            "retriever": "dense",
            "query": payload.query,
            "top_k": payload.top_k,
            "prompt_version": payload.prompt_version,
            "mode": mode,
            "use_reranker": getattr(payload, "use_reranker", False),
            "rerank_top_k": getattr(payload, "rerank_top_k", payload.top_k),
        })

        tracker.log_tags({
            "pipeline": "rag",
            "retrieval_type": "dense",
        })

        start = time.perf_counter()

        results = await run_in_threadpool(
            dense_retriever,
            payload.query,
            request.app.state.dense_retriever,
            payload.top_k
        )

        if getattr(payload, "use_reranker", False):
            results = await run_in_threadpool(
                request.app.state.reranker.rerank,
                payload.query,
                results,
                getattr(payload, "rerank_top_k", payload.top_k)
            )

        docs = to_retrieved_docs(results)

        answer = await rag_answer_async(
            payload.query,
            request.app.state.openai_client,
            [d.model_dump() for d in docs],
            mode=mode
        )

        latency = time.perf_counter() - start

        tracker.log_metrics({
            "num_retrieved_docs": len(docs),
            "latency_seconds": latency,
        })

        return RagResponse(
            retriever="dense",
            question=payload.query,
            prompt_version=payload.prompt_version,
            answer=answer,
            top_k=payload.top_k,
            retrieved_docs=docs
        )

@router.post("/rag/hyde", response_model=RagResponse)
async def rag_dense(payload: RagRequest, request: Request, hyde: HydeRetriever = Depends(get_hyde_retriever)):
    tracker = request.app.state.mlflow
    mode = "cot" if "cot" in payload.prompt_version else "plain"

    with tracker.start_run(run_name=f"rag-dense-{payload.prompt_version}"):
        tracker.log_params({
            "endpoint": "rag/hyde",
            "retriever": "hyde",
            "query": payload.query,
            "top_k": payload.top_k,
            "prompt_version": payload.prompt_version,
            "mode": mode,
            "use_reranker": getattr(payload, "use_reranker", False),
            "rerank_top_k": getattr(payload, "rerank_top_k", payload.top_k),
        })

        tracker.log_tags({
            "pipeline": "rag",
            "retrieval_type": "hyde",
        })

        start = time.perf_counter()

        results = await run_in_threadpool(
            hyde.retrieve,
            payload.query,
            payload.top_k,
            getattr(payload,"hyde_mode","short")
        )

        if getattr(payload, "use_reranker", False):
            results = await run_in_threadpool(
                request.app.state.reranker.rerank,
                payload.query,
                results,
                getattr(payload, "rerank_top_k", payload.top_k)
            )

        docs = to_retrieved_docs(results)

        answer = await rag_answer_async(
            payload.query,
            request.app.state.openai_client,
            [d.model_dump() for d in docs],
            mode=mode
        )

        latency = time.perf_counter() - start

        tracker.log_metrics({
            "num_retrieved_docs": len(docs),
            "latency_seconds": latency,
        })

        return RagResponse(
            retriever="hyde",
            question=payload.query,
            prompt_version=payload.prompt_version,
            answer=answer,
            top_k=payload.top_k,
            retrieved_docs=docs
        )

@router.post("/evaluation/bm25",response_model=EvaluateResponse)
async def evaluate_bm25(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        raise HTTPException(status_code=404,detail="Question Id is not available")
    
    tracker = request.app.state.mlflow

    with tracker.start_run(run_name=f"evl-bm25-{payload.question_id}"):
        tracker.log_params({
            "endpoint":"evaluation/bm25",
            "retriever":"bm25",
            "question_id":payload.question_id,
            "top_k":payload.top_k,
            "use_reranker":payload.use_reranker,
            "rerank_top_k":getattr(payload,"rerank_top_k",payload.top_k)
        })
        results = await run_in_threadpool(
            bm25_retriever,
            q['question'],
            request.app.state.bm25_chunks,
            request.app.state.vocab,
            request.app.state.stats,
            payload.top_k
        )
        if getattr(payload,"use_reranker",False):
            results = await run_in_threadpool(
                request.app.state.reranker.rerank,
                q['question'],
                results,
                getattr(payload,"rerank_top_k",payload.top_k)
            )
        response = build_eval_response("bm25", payload.question_id, results, request.app.state.questions, request.app.state.qrels,payload.top_k)

        tracker.log_metrics({
            "hit_at_k":response.metrics.hit_at_k,
            "hit_rate_at_k":float(response.metrics.hit_rate_at_k),
            "precision_at_k":response.metrics.precision_at_k,
            "recall_at_k":response.metrics.recall_at_k
        })

        tracker.log_tags({
            "retriever":"bm25",
            "id":q.get("id"),
            "source_doc":q.get("source_doc")
        })

        return response

@router.post("/evaluation/dense",response_model=EvaluateResponse)
async def evaluate_dense(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        raise HTTPException(status_code=404,detail="question id not available")
    
    
    tracker = request.app.state.mlflow

    with tracker.start_run(run_name=f"evl-dense-{payload.question_id}"):
        tracker.log_params({
            "endpoint":"evaluation/dense",
            "retriever":"dense",
            "question_id":payload.question_id,
            "top_k":payload.top_k,
            "use_reranker":payload.use_reranker,
            "rerank_top_k":getattr(payload,"rerank_top_k",payload.top_k),
            "experiment":getattr(payload,"experiment",False),
            "position":getattr(payload,"position",None),
            "candidate_pool_size":getattr(payload,"candidate_pool_size",30)
        })
        if getattr(payload,"experiment",False):
            gold_chunk = find_gold_chunk(payload.question_id,
                                         request.app.state.qrels,
                                         request.app.state.chunks)
            if gold_chunk is None:
                raise HTTPException(status_code=404,detail="Gold chunk is not found")
            candidate_chunks = Randomize.create_positioned_list(
                gold_chunk=gold_chunk,
                all_chunks=request.app.state.chunks,
                position=getattr(payload,"position","middle"),
                total_docs = getattr(payload,"candidate_pool_size",30)
            )
            print(len(candidate_chunks))
            print(candidate_chunks[0])
            results = await run_in_threadpool(
                dense_retriever,
                q["question"],
                DenseRetriever(candidate_chunks),
                payload.top_k
            )
            if getattr(payload,"use_reranker",False):
                results  = await run_in_threadpool(
                    request.app.state.reranker.rerank,
                    q["question"],
                    results,
                    getattr(payload,"rerank_top_k",payload.top_k)
                )
        else:
            results = await run_in_threadpool(
                dense_retriever,
                q['question'],
                request.app.state.dense_retriever,
                payload.top_k
            )
            if getattr(payload,"use_reranker",False):
                results = await run_in_threadpool(
                    request.app.state.reranker.rerank,
                    q['question'],
                    results,
                    getattr(payload,"rerank_top_k",payload.top_k)
                )
        print("qid:", payload.question_id)
        print("gold_docs:", request.app.state.qrels.get(payload.question_id))
        print("retrieved doc_names:", [r["doc_name"] for r in results])
        response = build_eval_response("dense",payload.question_id,results,request.app.state.questions,request.app.state.qrels,payload.top_k)
        tracker.log_metrics({
            "gold_rank":response.gold_rank,
            "hit_at_k":response.metrics.hit_at_k,
            "hit_rate_at_k":float(response.metrics.hit_rate_at_k),
            "precision_at_k":response.metrics.precision_at_k,
            "recall_at_k":response.metrics.recall_at_k})
        tracker.log_tags({
            "retriever":"dense",
            "id":q.get("id"),
            "source_doc":q.get("source_doc")

        })

        return response

@router.post("/evaluation/bm25/compare",response_model=CompareEvaluateResponse)
async def compare_evaluate_bm25(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        raise HTTPException(status_code=404,detail="question id not available")
    response = await compare_with_and_without_rerank(
        retriever_name="bm25",
        question_id=payload.question_id,
        query=q["question"],
        retriever_fn=bm25_retriever,
        retriever_args=(
        request.app.state.bm25_chunks,
        request.app.state.vocab,
        request.app.state.stats,),
        request=request,
        top_k=payload.top_k,
        rerank_top_k=payload.rerank_top_k
    )
    tracker = request.app.state.mlflow
    with tracker.start_run(run_name=f"evl-bm25-compare-{payload.question_id}"):
        tracker.log_metrics({
            "baseline_hit_at_k":response.baseline.metrics.hit_at_k,
            "baseline_precision_at_k": response.baseline.metrics.precision_at_k,
            "baseline_recall_at_k": response.baseline.metrics.recall_at_k,
            "reranked_hit_at_k": response.reranked.metrics.hit_at_k,
            "reranked_precision_at_k": response.reranked.metrics.precision_at_k,
            "reranked_recall_at_k": response.reranked.metrics.recall_at_k,
            "delta_precision_at_k": response.reranked.metrics.precision_at_k - response.baseline.metrics.precision_at_k,
            "delta_recall_at_k": response.reranked.metrics.recall_at_k - response.baseline.metrics.recall_at_k,
        })
        return response

@router.post("/evaluation/dense/compare",response_model=CompareEvaluateResponse)
async def compare_evaluate_bm25(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        raise HTTPException(status_code=404,detail="question id not available")
    response = await compare_with_and_without_rerank(
        retriever_name="dense",
        question_id=payload.question_id,
        query=q["question"],
        retriever_fn=dense_retriever,
        retriever_args=(request.app.state.dense_retriever,),
        request=request,
        top_k=payload.top_k,
        rerank_top_k=payload.rerank_top_k
    )
    tracker = request.app.state.mlflow
    with tracker.start_run(run_name=f"evl-dense-compare-{payload.question_id}"):
        tracker.log_metrics({
            "baseline_hit_at_k":response.baseline.metrics.hit_at_k,
            "baseline_precision_at_k": response.baseline.metrics.precision_at_k,
            "baseline_recall_at_k": response.baseline.metrics.recall_at_k,
            "reranked_hit_at_k": response.reranked.metrics.hit_at_k,
            "reranked_precision_at_k": response.reranked.metrics.precision_at_k,
            "reranked_recall_at_k": response.reranked.metrics.recall_at_k,
            "delta_precision_at_k": response.reranked.metrics.precision_at_k - response.baseline.metrics.precision_at_k,
            "delta_recall_at_k": response.reranked.metrics.recall_at_k - response.baseline.metrics.recall_at_k,
        })
        return response 






