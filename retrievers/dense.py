import torch 
import faiss
from transformers import DPRQuestionEncoder, DPRContextEncoder, DPRQuestionEncoderTokenizer, DPRContextEncoderTokenizer

class DPR:
    def __init__(self,query,chunks):
        self.query = query
        self.chunks = chunks

    def dpr_retriever(self):
        question_encoder = DPRQuestionEncoder.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        context_encoder = DPRContextEncoder.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")
        question_tokenizer = DPRQuestionEncoderTokenizer.from_pretrained("facebook/dpr-question_encoder-single-nq-base")
        context_tokenizer = DPRContextEncoderTokenizer.from_pretrained("facebook/dpr-ctx_encoder-single-nq-base")

        inputs = question_tokenizer(self.query,return_tensors="pt")
        query_embedding = question_encoder(**inputs).pooler_output.detach().numpy()
        chunk_texts = [c['text'] for c in self.chunks]
        chunk_embedding = []
        for chunk in chunk_texts:
            inputs = context_tokenizer(chunk,return_tensors='pt')
            embedding = context_encoder(**inputs).pooler_output().numpy()
            chunk_embedding.append(embedding)

        chunk_embedding = torch.tensor(chunk_embedding).squeeze().numpy()
        index = faiss.IndexFlatL2(chunk_embedding.shape[1])
        index.add(chunk_embedding)

        D,I = index.search(query_embedding,k=5)

        return D,I




