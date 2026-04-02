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
    HealthResponse
)

# from api.dependencies import normalize_doc_name
from retrievers.bm25 import BM25
from retrievers.dense import DenseRetriever

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
        results.append(
            RetrievedDocs(
                doc_name=c['doc_name'],
                chunk_id=c.get("chunk_id"),
                score=float(c['score']),
                pages=c.get("pages",[]),
                text=text
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
    gold_id = q['source_doc']
    retrieved_ids = [r['doc_name'] for r in results]
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
        return HTTPException(status_code=404,details="Question Id is not available")
    results = await run_in_threadpool(
        bm25_retriever,
        q['question'],
        request.app.state.bm25_chunks,
        request.app.state.vocab,
        request.app.state.stats,
        payload.top_k
    )
    return build_eval_response("bm25", payload.question_id, results, request.app.state.questions, payload.top_k)

@router.post("/evaluation/dense",response_model=EvaluateResponse)
async def evaluate_dense(payload:EvaluateRequest,request:Request):
    q = next((qq for qq in request.app.state.questions if qq['id'] == payload.question_id),None)
    if q is None:
        return HTTPException(status_code=404,details="question id not available")
    results = await run_in_threadpool(
        dense_retriever,
        q['question'],
        request.app.state.dense_retriever,
        payload.top_k
    )
    return build_eval_response("dense",payload.question_id,results,request.app.state.questions,payload.top_k)

    






