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
    dense_retriever = DenseRetriever(chunks)
    reranker = Reranker()
    mlflow_tracker = MLFlow()

    app.state.chunks = chunks
    app.state.questions = questions
    app.state.qrels = qrels
    app.state.bm25_chunks = bm25_chunks
    app.state.vocab = vocab
    app.state.stats = stats
    app.state.dense_retriever = dense_retriever
    app.state.openai_client = OpenAIClient()
    app.state.reranker = reranker
    app.state.mlflow = mlflow_tracker


def get_app_state(request:Request) -> Any:
    return request.app.state

def get_openai_client(request:Request) -> OpenAIClient:
    return request.app.state.openai_client

def get_reranker(request:Request) -> Reranker:
    return request.app.state.reranker

def get_mlflow(request:Request) -> MLFlow:
    return request.app.state.mlflow



