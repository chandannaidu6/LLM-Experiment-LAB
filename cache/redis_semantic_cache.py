from __future__ import annotations
from typing import Any,Callable,Optional
import hashlib 
import json
import os
import time
import numpy as  np
import redis
from redis.commands.search.field import TextField, NumericField, VectorField
from redis.commands.search.index_definition import IndexDefinition, IndexType
from redis.commands.search.query import Query

class RedisSemanticCache:
    def __init__(self,embed_fn: Callable[[str],list[float]],redis_url:Optional[str] = None,index_name:str = "idx:rag_semantic_cache_v2",key_prefix:str = "rag:cache:",vector_dim:int = 1536,
        similarity_threshold:float = 0.95,ttl_seconds:int = 86400):
        self.embed_fn = embed_fn
        self.redis_url = redis_url or os.getenv("REDIS_URL","redis://localhost:6379")
        self.index_name = index_name
        self.key_prefix = key_prefix
        self.vector_dim = vector_dim
        self.similarity_threshold = similarity_threshold
        self.ttl_seconds = ttl_seconds
        self.client = redis.Redis.from_url(self.redis_url,decode_responses = False,protocol=2)
        self._ensure_index()

    def ping(self)->bool:
        return bool(self.client.ping())

    def _normalize_query(self,query:str)->str:
        return " ".join(query.lower().split())

    def _hash(self,value:str)->str:
        return hashlib.sha256(value.encode("utf-8")).hexdigest()

    def _key(self,normalize_query:str)->str:
        return f"{self.key_prefix}{self._hash(normalize_query)}"

    def _to_vector_bytes(self,embeddings:list[float])->bytes:
        return np.array(embeddings,dtype=np.float32).tobytes()

    def _decode(self,value:Any)->Any:
        if isinstance(value,bytes):
            return value.decode("utf-8")
        return value

    def _ensure_index(self)->None:
        try:
            self.client.ft(self.index_name).info()
#            print("INDEX already exists:", self.index_name)
            return
        except Exception:
            pass

        schema = (
            TextField("query"),
            TextField("answer"),
            TextField("metadata"),
            NumericField("created_at"),
            VectorField(
                "embedding",
                "HNSW",
                {
                    "TYPE":"FLOAT32",
                    "DIM": self.vector_dim,
                    "DISTANCE_METRIC": "COSINE",
                    "M": 16,
                    "EF_CONSTRUCTION": 200,
                }
            )
        )
        definition = IndexDefinition(
            prefix = [self.key_prefix],
            index_type = IndexType.HASH
        )

        self.client.ft(self.index_name).create_index(schema,definition=definition)
        info = self.client.ft(self.index_name).info()
#        print("INDEX created:", self.index_name)
#        print("INDEX info:", info)

    def lookup(self,query:str)->Optional[dict[str,Any]]:
        normalized_query = self._normalize_query(query)
        query_embedding = self.embed_fn(normalized_query)
        query_vector = self._to_vector_bytes(query_embedding)
#        print("LOOKUP normalized_query:", normalized_query)


        q = (
            Query("*=>[KNN 1 @embedding $vec AS distance]")
            .sort_by("distance")
            .return_fields("query","answer","metadata","created_at","distance")
            .paging(0,1)
            .dialect(2)
        )

        results = self.client.ft(self.index_name).search(q,{"vec":query_vector})
#        print("LOOKUP docs found:", len(results.docs))

        if not results.docs:
            self.client.hincrby("rag:cache:stats","misses",1)
            return None


        doc = results.docs[0]
        distance = float(self._decode(doc.distance))
        similarity = 1.0 - distance
"""        print("LOOKUP matched query:", self._decode(doc.query))
        print("LOOKUP distance:", distance)
        print("LOOKUP similarity:", similarity)
        print("LOOKUP threshold:", self.similarity_threshold)
"""
        if similarity < self.similarity_threshold:
            print("LOOKUP miss: similarity below threshold")

            self.client.hincrby("rag:cache:stats","misses",1)
            return None
        print("LOOKUP hit")

        self.client.hincrby("rag:cache:stats","hits",1)
        metadata_raw = self._decode(doc.metadata)

        return {
            "query": self._decode(doc.query),
            "answer": self._decode(doc.answer),
            "metadata": json.loads(metadata_raw) if metadata_raw else {},
            "created_at": float(self._decode(doc.created_at)),
            "similarity": similarity
        }

    def store(self,query:str,answer:str,metadata:Optional[dict[str,Any]] = None)->None:
        normalized_query = self._normalize_query(query)
        key =  self._key(normalized_query)
        embedding = self.embed_fn(normalized_query)
        embedding_bytes = self._to_vector_bytes(embedding)
"""        print("STORE normalized_query:", normalized_query)
        print("STORE key:", key)
        print("STORE embedding bytes:", len(embedding_bytes), "expected:", self.vector_dim * 4)   # add this
"""

        payload = {
            "query":normalized_query,
            "answer":answer,
            "metadata":json.dumps(metadata or {}),
            "created_at":time.time(),
            "embedding":embedding_bytes
        }
        self.client.hset(key,mapping=payload)
        self.client.expire(key,self.ttl_seconds)
        stored = self.client.hgetall(key)
#        print("STORE exists after hset:", bool(stored))
#        print("STORE fields:", stored.keys())
        info = self.client.ft(self.index_name).info()
        print("num_docs:", info.get("num_docs"), "failures:", info.get("hash_indexing_failures"))
       
    def stats(self)->dict[str,float]:
        raw = self.client.hgetall("rag:cache:stats")
        hits = int(raw.get(b"hits",b"0"))
        misses = int(raw.get(b"misses",b"0"))
        total = hits + misses
        return {
            "hits":hits,
            "misses":misses,
            "total":total,
            "hit_rate": hits/total if total else 0.0
        }

    def clear(self) -> None:
        keys = self.client.keys(f"{self.key_prefix}*")
        if keys:
            self.client.delete(*keys)
        self.client.delete("rag:cache:stats")        

        




        


    

    