from openai import OpenAI
import os
import numpy as np
from typing import List

class OpenAIEmbedder:
    def __init__(self,model:str="text-embedding-3-small",api_key:str | None = None):
        self.model = model
        self.client = OpenAI(api_key = self.api_key)

    def embed_texts(self,texts:List[str]) -> np.ndarray:
        if not texts:
            return np.zeros((0,0),dtype = np.float32)
        
        resp = self.client.embeddings.create(model=self.model,input=texts)

        vectors = [item.embedding for item in resp.data]
        return np.array(vectors,dtype=np.float32)
    
    def embed_text(self,text:str) -> np.ndarray:
        arr = self.embed_texts([text])
        return arr[0]
    
