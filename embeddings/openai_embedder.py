from openai import OpenAI,RateLimitError
import os
import numpy as np
import time
from typing import List

class OpenAIEmbedder:
    def __init__(self,model:str="text-embedding-3-small",api_key:str | None = None,batch_size:int = 100):
        self.model = model
        self.client = OpenAI(api_key =api_key or os.getenv("OPENAI_API_KEY"))
        self.batch_size = batch_size

    def embed_texts(self,texts:List[str]) -> np.ndarray:
        clean_texts = []
        for t in texts:
            if t is None:
                t = " "
            elif isinstance(t,list):
                text = " ".join(str(x) for x in t).strip()
            else:
                text = str(t).strip()
            if not text:
                text = " "

            clean_texts.append(text)
        all_embeddings = []
        
        for i in range(0,len(clean_texts),self.batch_size):
            batch = clean_texts[i:i+self.batch_size]
            while True:
                try:
                    resp = self.client.embeddings.create(model=self.model,input=batch)
                    all_embeddings.extend([d.embedding for d in resp.data])
                    break

                except RateLimitError as e:
                    time.sleep(1.0)
                    continue
        return np.array(all_embeddings,dtype=np.float32)

    def embed_sentences(self,sentences:List[str])->np.ndarray:
        if not sentences:
            return np.empty((0,0),dtype=np.float32)
        return self.embed_texts(sentences)
    
    def embed_text(self,text:str) -> np.ndarray:
        arr = self.embed_texts([text])
        return arr[0]
    
