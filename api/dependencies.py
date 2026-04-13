#Building an app level cache 
import os
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

load_dotenv()

DATA_DIR = Path("Data")
QUESTIONS_PATH = Path("QA/questions.json")

def normalize_doc_name(name:str)->str:
  return os.path.basename(str(name))
  
def load_chunks():
    chunks = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        ch = SlideChunker(pdf_path=str(pdf_path))
        pages_data = ch.extract_text()
        sentences = ch.metadata_sentences(pages_data)
        doc_chunks = ch.chunk_sentences(sentences=sentences)
        chunks.extend(doc_chunks)
    return chunks

def bm25_index(chunks):
    pre = preprocessing(chunks=chunks,corpus=None)
    tokenized = pre.tokenized_corpus()
    vocab = pre.vocabulary_construction(tokenized)

    bm_chunks = []

    for chunk,tokens in zip(chunks,tokenized):
        c = dict(chunk)
        c['text'] = tokens
        bm_chunks.append(c)

    bm = BM25("dummy",chunks,vocab)
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
    chunks = load_chunks()
    bm25_chunks,vocab,stats = bm25_index(chunks)
    dense_retriever = DenseRetriever(chunks)
    dense_retriever.build_index()
    questions = load_questions()
    reranker = Reranker()

    app.state.chunks = chunks
    app.state.bm25_chunks = bm25_chunks
    app.state.vocab = vocab
    app.state.stats = stats
    app.state.dense_retriever = dense_retriever
    app.state.questions = questions
    app.state.openai_client = OpenAIClient()
    app.state.reranker = reranker


def get_app_state(request:Request) -> Any:
    return request.app.state

def get_openai_client(request:Request) -> OpenAIClient:
    return request.app.state.openai_client

def get_reranker(request:Request) -> Reranker:
    return request.app.state.reranker



