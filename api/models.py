from pydantic import BaseModel
from typing import List, Optional, Literal

class RetrieveRequest(BaseModel):
    query: str
    top_k: int = 5

class RagRequest(BaseModel):
    query:str
    top_k:int = 5
    prompt_version:str = "v1_plain"

class EvaluateRequest(BaseModel):
    question_id:str
    top_k:int = 5

class RetrievedDocs(BaseModel):
    doc_name:str
    chunk_id: Optional[int] =None
    score: float
    text:str
    pages:list[int]

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
    gold_doc:str
    retrieved_docs:List[str]
    metrics:EvaluationMetrics

class HealthResponse(BaseModel):
    status:str
    chunks_loaded: int

