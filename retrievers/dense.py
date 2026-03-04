from typing import List,Dict,Any
import numpy as np
from embeddings.openai_embedder import OpenAIEmbedder

def _cosine_sim(a:np.ndarray,b:np.ndarray)-> np.ndarray:
    a_norm = a/(np.linalg.norm(a,axis=1,keepdims=True) + 1e-8)
    b_norm = b/(np.linalg.norm(b,axis=1,keepdims=True) + 1e-8)

    return a_norm @ b_norm.T


class DenseRetriever:
    def __init__(self,chunks:List[Dict[str,Any]], embedder:OpenAIEmbedder | None = None):
        self.chunks = chunks
        self.embedder = embedder or OpenAIEmbedder()
        self.embeddings = None

    def build_index(self) -> None:
        texts = [c['text'] for c in self.chunks]
        self.embeddings = self.embedder.embed_texts(texts)

    def retrieve(self,query:str,top_k:int = 5,) -> List[Dict[str,Any]]:
        if self.embeddings is None:
            raise RuntimeError("Call build index before retrieve")
        
        q_vec = self.embedder.embed_text(query)
        q_vec = q_vec.reshape(1,-1)

        sims = _cosine_sim(q_vec,self.embeddings)[0]
        top_idx = np.argsort(sims)[::-1][:top_k]
        results:List[Dict[str,Any]] = []

        for idx in top_idx:
            c = self.chunks[idx]
            results.append(
                {
                    "score": float(sims[idx]),
                    "text": c["text"],
                    "doc_name": c.get("doc_name", "Unknown"),
                    "page_number": c.get("page_number"),
                    "chunk_id": idx,

                }
            )

        return results
        



