from typing import List,Dict,Any
from embeddings.openai_embedder import OpenAIEmbedder
from storage.chroma_client import ChromaVector

class DenseRetriever:
    def __init__(self,chunks:List[Dict[str,Any]], embedder:OpenAIEmbedder | None = None,vector_store:ChromaVector | None = None):
        self.chunks = chunks
        self.embedder = embedder or OpenAIEmbedder()
        self.vector_store = vector_store or ChromaVector(collection_name="rag_chunks")

    def build_index(self) -> None:
        texts = [c['text'] for c in self.chunks]
        embeddings = self.embedder.embed_texts(texts)
        ids,metadatas = [],[]

        for id,chunk in enumerate(self.chunks):
            chunk_id = chunk.get("chunk_id",id)
            ids.append(str(id))
            metadatas.append(
                {
                    "doc_name":chunk.get("doc_name","Unknown"),
                    "page_number":chunk.get("page_number"),
                    "chunk_id":chunk_id
                }
            )
        self.vector_store.upsert(ids=ids,documents=texts,embeddings=embeddings.tolist() if hasattr(embeddings,"tolist") else embeddings,metadatas=metadatas)


    def retrieve(self,query:str,top_k:int = 5,) -> List[Dict[str,Any]]:

        q_vec = self.embedder.embed_text(query)
        if hasattr(q_vec,"tolist"):
            q_vec = q_vec.tolist()

        results = self.vector_store.query(query_embeddings=q_vec,top_k=top_k)

        output: List[Dict[str,any]] = []

        for doc,meta,dist in zip(
            results["documents"],
            results["metadatas"],
            results["distances"]
        ):
            output.append(
                {
                    "score":float(1-dist),
                    "text":doc,
                    "doc_name":meta.get("doc_name","Unknown"),
                    "page_number":meta.get("page_number"),
                    "chunk_id":meta.get("chunk_id")
                }
            )
        return output

        



