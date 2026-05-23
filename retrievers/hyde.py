from embeddings.openai_embedder import OpenAIEmbedder
from storage.chroma_client import ChromaVector
from llm.openai_client import OpenAIClient

class HydeRetriever:
    def __init__(self,llm_client: OpenAIClient | None,embedder:OpenAIEmbedder | None,vector_store:ChromaVector | None):
        self.llm_client = llm_client
        self.embedder = embedder
        self.vector_store = vector_store

    def generate_hypothetical_document(self,query:str,mode:str = "short")->str:
        return self.llm_client.generate_context(query=query,mode=mode)

    def retrieve(self,query:str,top_k:int = 5,mode:str = "short"):
        hypothetical_doc = self.generate_hypothetical_document(query,mode)
        q_vec = self.embedder.embed_text(hypothetical_doc)
        
        if hasattr(q_vec,"tolist"):
            q_vec = q_vec.tolist()

        results = self.vector_store.query(query_embeddings=q_vec,top_k=top_k)
        output = []
        docs = results["documents"]
        metas = results["metadatas"]
        dists = results["distances"]

        for doc,meta,dist in zip(docs,metas,dists):
            meta = meta or {}
            output.append({
                "score": float(1-dist),
                "text":doc,
                "doc_name":meta.get("doc_name","Unknown"),
                "page_number":meta.get("page_number"),
                "chunk_id":meta.get("chunk_id"),

            })
        return output

