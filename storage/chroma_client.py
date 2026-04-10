from __future__ import annotations
from typing import List,Dict,Any,Optional

import chromadb

class ChromaVector:
    def __init__(self,collection_name:str = "rag_chunks",host:str = "localhost", port:int = 8001):
        self.collection_name = collection_name
        self.client = chromadb.HttpClient(host=host, port= port)
        self.collection = self.client.get_or_create_collection(name = self.collection_name,
                                                               metadata={"hnsw:space":"cosine"})
        

    def upsert(self,ids:List[str],documents:List[str],embeddings:List[List[float]],metadatas:Optional[List[Dict[str,Any]]] = None) -> None:
        if metadatas is None:
            metadatas = [{} for _ in ids]

        self.collection.upsert(ids=ids,
                    documents=documents,
                    embeddings = embeddings,
                    metadatas=metadatas)
        

    def query(self,query_embeddings:List[float],top_k:int = 5,) -> Dict[str,Any]:
        results = self.collection.query(query_embeddings=[query_embeddings],n_results=top_k,include=["documents","metadatas","distances"])
        return {
            "ids":results["ids"][0],
            "documents":results["documents"][0],
            "metadatas":results["metadatas"][0],
            "distances":results["distances"][0]
        }
    
    def reset_collection(self)->None:
        try:
            self.client.delete_collection(self.collection_name)
        except Exception:
            pass

        self.collection = self.client.get_or_create_collection(
            name = self.collection_name,
            metadata = {"hnsw:space":"cosine"}
        )



 


