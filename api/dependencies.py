#Building an app level cache 
import json
from typing import Any
from fastapi import Request
from dotenv import load_dotenv
import os 
from pathlib import Path
from chunkers.slide_chunking import SlideChunker
from preprocessing import preprocessing
from retrievers.bm25 import BM25
from retrievers.dense import DenseRetriever
from llm.openai_client import OpenAIClient
from reranker.cross_encoder import Reranker
from tracking import MLFlow
from retrievers.hyde import HydeRetriever
from retrievers.hybrid import HybridRetriever
from embeddings.openai_embedder import OpenAIEmbedder
from storage.chroma_client import ChromaVector
from cache.redis_semantic_cache import RedisSemanticCache
load_dotenv()

DATA_DIR = Path("Data/beir_scifact")
QUESTIONS_PATH = Path("QA/questions.json")

def normalize_doc_name(name:str)->str:
  return os.path.basename(str(name))

def load_json(path:Path):
    with open(path,"r",encoding="utf-8") as f:
        return json.load(f)
"""
def load_chunks():
    chunks = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        ch = SlideChunker(pdf_path=str(pdf_path))
        pages_data = ch.extract_text()
        sentences = ch.metadata_sentences(pages_data)
        doc_chunks = ch.chunk_sentences(sentences=sentences)
        chunks.extend(doc_chunks)
    return chunks
"""
def bm25_index(chunks):
    pre = preprocessing(chunks=chunks,corpus=None)
    tokenized = pre.tokenized_corpus()
    vocab = pre.vocabulary_construction(tokenized)

    bm_chunks = []

    for chunk,tokens in zip(chunks,tokenized):
        c = dict(chunk)
        c['text'] = tokens
        bm_chunks.append(c)

    bm = BM25("dummy",bm_chunks,vocab)
    stats = bm.IDF()
    return bm_chunks,vocab,stats

def dense_index(chunks):
    dense = DenseRetriever(chunks)
    dense.build_index()
    return dense

def load_questions():
    with open(QUESTIONS_PATH,'r',encoding='utf-8') as f:
        return json.load(f)
    
    
def build_app_state(app):
    chunks = load_json(DATA_DIR/"chunks.json")
    questions = load_json(DATA_DIR/"questions.json")
    qrels = load_json(DATA_DIR/"qrels.json")
    bm25_chunks,vocab,stats = bm25_index(chunks)
    openai_client = OpenAIClient()
    embedder = OpenAIEmbedder()
    vector_store = ChromaVector(collection_name="rag_chunks")
    dense_retriever = DenseRetriever(chunks)
    hyde_retriever = HydeRetriever(llm_client=openai_client,embedder=embedder,vector_store=vector_store)
    hybrid_retriever = HybridRetriever(bm25_chunks,vocab,stats,dense_retriever)
    reranker = Reranker()
    mlflow_tracker = MLFlow()
    semantic_cache = RedisSemanticCache(embed_fn = lambda text:embedder.embed_texts(text)[0],index_name="idx:rag_semantic_cache_v2",key_prefix="rag:cache:",vector_dim=1536,similarity_threshold= 0.95, ttl_seconds = 86400)

    app.state.chunks = chunks
    app.state.questions = questions
    app.state.qrels = qrels
    app.state.bm25_chunks = bm25_chunks
    app.state.vocab = vocab
    app.state.stats = stats
    app.state.openai_client = openai_client
    app.state.embedder = embedder
    app.state.vector_store = vector_store
    app.state.dense_retriever = dense_retriever
    app.state.hyde_retriever = hyde_retriever
    app.state.hybrid_retriever = hybrid_retriever
    app.state.reranker = reranker
    app.state.mlflow = mlflow_tracker
    app.state.semantic_cache = semantic_cache


def get_app_state(request:Request) -> Any:
    return request.app.state

def get_openai_client(request:Request) -> OpenAIClient:
    return request.app.state.openai_client

def get_reranker(request:Request) -> Reranker:
    return request.app.state.reranker

def get_mlflow(request:Request) -> MLFlow:
    return request.app.state.mlflow

def get_hyde_retriever(request: Request) -> HydeRetriever:
    return request.app.state.hyde_retriever

def get_hybrid_retriever(request: Request) -> HybridRetriever:
    return request.app.state.hybrid_retriever

def get_semantic_cache(request:Request) -> RedisSemanticCache:
    return request.app.state.semantic_cache






