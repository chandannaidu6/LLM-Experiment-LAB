from typing import Optional, List

from bm25 import BM25
from dense import DenseRetriever
from hybrid import HybridRetriever
from hyde import HydeRetriever
from self_rag import SelfRAGRetriever


class RegistryRetriever:
    RETRIEVER_REGISTRY = {
        "bm25": BM25,
        "dense": DenseRetriever,
        "hybrid": HybridRetriever,
        "hyde": HydeRetriever,
        "self_rag": SelfRAGRetriever,
    }

    def __init__(self, name: str, **params):
        self.name = name.lower()
        self.params = params

    def get_retriever(self):
        if self.name not in self.RETRIEVER_REGISTRY:
            raise ValueError(
                f"Unknown retriever '{self.name}'. "
                f"Available: {', '.join(self.available_retrievers())}"
            )
        retriever_cls = self.RETRIEVER_REGISTRY[self.name]
        return retriever_cls(**self.params)     

    @classmethod
    def available_retrievers(cls) -> List[str]:
        return list(cls.RETRIEVER_REGISTRY.keys())