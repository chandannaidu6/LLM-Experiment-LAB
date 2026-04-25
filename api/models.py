from pydantic import BaseModel,Field
from typing import List, Optional, Literal

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5
    use_reranker:bool = False
    rerank_top_k:int = 5

class RagRequest(BaseModel):
    query:str
    top_k:int = 5
    prompt_version:str = "v1_plain"
    use_reranker:bool = False
    rerank_top_k:int = 5

class EvaluateRequest(BaseModel):
    question_id:str
    top_k:int = 5
    use_reranker:bool = False
    rerank_top_k:int = 5
    experiment:bool = False
    position:Optional[Literal["start","middle","end"]] = None
    candidate_pool_size:Optional[int] = 30

class RetrievedDocs(BaseModel):
    doc_name:str
    chunk_id: Optional[int] =None
    score: float
    text:str
    pages:List[int] = Field(default_factory= list)
    rerank_score: Optional[float] = None

class RetrievedResponse(BaseModel):
    retriever:Literal["bm25","dense"]
    query:str
    top_k:int
    results:List[RetrievedDocs]

class RagResponse(BaseModel):
    retriever:Literal["bm25","dense"]
    question: str
    prompt_version:str
    answer:str
    top_k:int
    retrieved_docs:List[RetrievedDocs]

class EvaluationMetrics(BaseModel):
    hit_at_k: int
    hit_rate_at_k: bool
    precision_at_k: float
    recall_at_k: float

class EvaluateResponse(BaseModel):
    retriever: Literal["bm25","dense"]
    question_id:str
    gold_doc:Optional[str] = None
    gold_rank:Optional[int] = None
    retrieved_docs:List[str]
    metrics:EvaluationMetrics

class CompareEvaluationResult(BaseModel):
    retrieved_docs:List[str]
    metrics:EvaluationMetrics

class CompareEvaluateResponse(BaseModel):
    retriever: Literal["bm25","dense"]
    question_id:str
    gold_doc:Optional[str] = None
    top_k:int
    rerank_top_k:int
    baseline:CompareEvaluationResult
    reranked:CompareEvaluationResult

class HealthResponse(BaseModel):
    status:str
    chunks_loaded: int

