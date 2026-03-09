from openai import OpenAI
import os
import numpy as np
from typing import List

class OpenAIEmbedder:
    def __init__(self,model:str="text-embedding-3-small",api_key:str | None = None):
        self.model = model
        self.client = OpenAI(api_key =api_key or os.getenv("OPENAI_API_KEY"))

    def embed_texts(self,texts:List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0,0),dtype = np.float32)
        
        clean_texts = [str(t) for t in texts]
        
        resp = self.client.embeddings.create(model=self.model,input=clean_texts)

        vectors = [item.embedding for item in resp.data]
        return np.array(vectors,dtype=np.float32)
    
    def embed_text(self,text:str) -> np.ndarray:
        arr = self.embed_texts([text])
        return arr[0]
    
