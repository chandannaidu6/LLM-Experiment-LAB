import numpy as np
import pytest
from retrievers.bm25 import BM25
from retrievers.dense import DenseRetriever,_cosine_sim

class TestEmbedder:
    def __init__(self):
        self.calls = []

    def embed_texts(self,texts):
        self.calls.append('text',texts)
        return np.array([
            [1.0,0.0,0.0],
            [0.0,1.0,0.0],
            [0.0,0.0,1.0]
        ],dtype=np.float32)
    
def test_cosine_sim():
    a = np.array([[1.0,0.0]],dtype= np.float32)
    b = np.array([[1.0,0.0],[0.0,1.0]],dtype=np.float32)
    sims = _cosine_sim(a,b)
    assert sims.shape == (1,2)
    assert sims[0,0] == pytest.approx(1.0,rel=1e-5)
    assert sims[0,1] == pytest.approx(0.0,abs=1e-5)

def test_dense_retriever():
    chunks = [
        {"text":"chunk0","doc_name":"d0.pdf"},
        {"text":"chunk1","doc_name":"d1.pdf"},
        {"text":"chunk2","doc_name":"d2.pdf"},
    ]
    den = DenseRetriever(chunks,embedder=TestEmbedder)
    den.build_index()
    res = den.retrieve("some query",top_k=3)
    assert len(res) == 3
    assert res[0]["doc_name"] == "d0.pdf"
    assert res[0]["score"] > res[1]["score"] > res[2]["score"]

def test_bm25_idf():
    chunks = [
        {"text": ["bm25", "works", "well"]},
        {"text": ["bm25", "bm25", "retrieval"]},
        {"text": ["dense", "retrieval"]},
    ]
    vocab = {
        "bm25": 0,
        "works": 1,
        "well": 2,
        "retrieval": 3,
        "dense": 4,
    }
    bm = BM25.IDF(query="bm25_retriever",chunks=chunks,vocabulary=vocab)
    stats = bm.IDF

    assert stats["tf_matrix"].shape == (3,len(vocab))
    assert stats["df_array"].shape == (len(vocab),)
    assert stats["doc_len"].shape == (3,)
    assert stats["idf_array"].shape == (len(vocab),)
    idf_bm25 = stats["idf_array"][vocab["bm25"]]
    idf_dense = stats["idf_array"][vocab["bm25"]]
    assert idf_dense > idf_bm25




def test_bm25_retriever(monkeypatch):
    chunks = [
        {"text": ["cat", "sat", "mat"], "doc_name": "d0.pdf"},
        {"text": ["dog", "sat", "rug"], "doc_name": "d1.pdf"},
        {"text": ["cat", "dog", "sat"], "doc_name": "d2.pdf"},
    ]
    vocab = {"cat": 0, "dog": 1, "sat": 2, "mat": 3, "rug": 4}
    bm = BM25(query="cat",chunks=chunks,vocabulary=vocab)
    stats = bm.IDF()
    bm.stats = stats


    class DummyPre:
        def __init__(self,chunks,corpus=None):
            self.chunks = chunks

        def tokenized_corpus(self):
            return [["cat"]]
        
    monkeypatch.setattr("retrievers.bm25.preprocessing",DummyPre)
    res = bm.bm25_retriever(top_k=3)
    assert len(res) > 3
    doc_names = [r['doc_name'] for r in res]
    assert "d0.pdf" in doc_names 
    assert "d1.pdf" in doc_names



    


    

