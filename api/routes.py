from fastapi import APIRouter,HTTPException,Request
from starlette.concurrency import run_in_threadpool

from api.models import (
    RetrieveRequest,
    RetrievedResponse,
    RetrievedDocs,
    RagRequest,
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

from evaluation.metrics import Evaluation

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
        return client.achat(query,context,mode=mode)
    return await run_in_threadpool(client.chat,query,context,mode)

def build_eval_response(retriever,qid,results,questions,top_k):
    q = next((q for q in questions if q['id'] == qid),None)
    if q is None:
        return HTTPException(status_code=404,detail="Question does not exist")
    gold_id = normalize_doc_name(q['source_doc'])
    retrieved_ids = [normalize_doc_name(r['doc_name']) for r in results]
    source_ids = [gold_id]
    ev = Evaluation(retrieved_ids,source_ids,k=min(top_k,len(retrieved_ids)))
    return EvaluateResponse(
        retriever=retriever,
        question_id=qid,
        gold_doc=gold_id,
        retrieved_docs=retrieved_ids,
        metrics=EvaluationMetrics(
            hit_at_k=ev.hit_at_k(),
            hit_rate_at_k=ev.hit_rate_k(),
            precision_at_k=ev.precision_at_k(),
            recall_at_k=ev.recall_at_k()
        )
    )

async def compare_with_and_without_rerank(retriever_name:str,question_id:str,query:str,retriever_fn,retriever_args:tuple,request:Request,top_k:int,rerank_top_k:int | None=None):
    q = next((qq for qq in request.app.state.questions if qq["id"] == question_id),None)
    if q is None:
        raise HTTPException(status_code=404, detail="Question id not available")
    
    rerank_top_k = rerank_top_k or top_k
    gold_doc = normalize_doc_name(q["source_doc"])


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
    baseline_ids = [normalize_doc_name(r["doc_name"]) for r in baseline_results]
    reranked_ids = [normalize_doc_name(r["doc_name"]) for r in reranked_results]

    baseline_eval = Evaluation(
        baseline_ids,
        [gold_doc],
        k = min(top_k,len(baseline_ids))
    )
    reranked_eval = Evaluation(
        reranked_ids,
        [gold_doc],
        k = min(top_k,len(reranked_ids))
    )
    return CompareEvaluateResponse(
        retriever=retriever_name,
        question_id=question_id,
        gold_doc=gold_doc,
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

@router.post("/rag/bm25",response_model=RagResponse)
async def rag_bm25(payload:RagRequest,request:Request):
    results = await run_in_threadpool(
        bm25_retriever,
        payload.query,
        request.app.state.bm25_chunks,
        request.app.state.vocab,
        request.app.state.stats,
        payload.top_k
    )
    if getattr(payload,"use_reranker",False):
        results = await run_in_threadpool(
            request.app.state.reranker.rerank,
            payload.query,
            results,
            getattr(payload,"rerank_top_k",payload.top_k)
        )

    docs = to_retrieved_docs(results)
    mode = "cot" if "cot" in payload.prompt_version else "plain"
    answer = await rag_answer_async(
        payload.query,
        request.app.state.openai_client,
        [d.model_dump() for d in docs],
        mode=mode,
    )

    return RagResponse(
        retriever="bm25",
        question=payload.query,
        prompt_version=payload.prompt_version,
        answer=answer,
        top_k=payload.top_k,
        retrieved_docs=docs
    )

@router.post("/rag/dense",response_model=RagResponse)
async def rag_dense(payload:RagRequest,request:Request):
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
    docs = to_retrieved_docs(results)
    mode = "cot" if "cot" in payload.prompt_version else "plain"
    answer = await rag_answer_async(
        payload.query,
        request.app.state.openai_client,
        [d.model_dump() for d in docs],
        mode=mode
    )
    return RagResponse(
        retriever="dense",
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

    return build_eval_response("bm25", payload.question_id, results, request.app.state.questions, payload.top_k)

@router.post("/evaluation/dense",response_model=EvaluateResponse)
async def evaluate_dense(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        raise HTTPException(status_code=404,detail="question id not available")
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
    return build_eval_response("dense",payload.question_id,results,request.app.state.questions,payload.top_k)

@router.post("/evaluation/bm25/compare",response_model=CompareEvaluateResponse)
async def compare_evaluate_bm25(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        raise HTTPException(status_code=404,detail="question id not available")

    return await compare_with_and_without_rerank(
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

@router.post("/evaluation/dense/compare",response_model=CompareEvaluateResponse)
async def compare_evaluate_bm25(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        raise HTTPException(status_code=404,detail="question id not available")

    return await compare_with_and_without_rerank(
        retriever_name="dense",
        question_id=payload.question_id,
        query=q["question"],
        retriever_fn=dense_retriever,
        retriever_args=(request.app.state.dense_retriever,),
        request=request,
        top_k=payload.top_k,
        rerank_top_k=payload.rerank_top_k
    )






