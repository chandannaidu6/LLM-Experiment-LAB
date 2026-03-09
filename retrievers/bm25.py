import numpy as np
from typing import List,Dict,Any,Tuple
from preprocessing import preprocessing

class BM25:
    def __init__(self,query,chunks,vocabulary):
        self.query = query
        self.chunks = chunks
        self.vocabulary = vocabulary

    def IDF(self) -> Dict:
        chunk_texts = [c['text'] for c in self.chunks]
        N = len(chunk_texts)
        V = len(self.vocabulary)
        tf_matrix = np.zeros((N,V),dtype=np.float32)
        df_array = np.zeros(V,dtype=np.float32)
        doc_len = np.zeros(N,dtype=np.float32)

        for chunk_id,tokens in enumerate(chunk_texts):
            doc_len[chunk_id] = len(tokens)
            seen_in_chunk = set()
            for word in tokens:
                if word in self.vocabulary:
                    word_id = self.vocabulary[word]
                    tf_matrix[chunk_id][word_id] += 1

                    if word_id not in seen_in_chunk:
                        df_array[word_id] += 1
                        seen_in_chunk.add(word_id)

        avg_dl = np.mean(doc_len)
        idf_array = np.log(((N-df_array+0.5)/(df_array+0.5)) + 1)
        return {
            "N":N,
            "tf_matrix":tf_matrix,
            "df_array":df_array,
            "doc_len":doc_len,
            "avgdl":avg_dl,
            "idf_array":idf_array
        }
    

    def bm25_retriever(self,k1=1.5,b=0.75,top_k=5):
        pre = preprocessing(chunks=[{"text":self.query}],corpus=None)
        query_tokens = pre.tokenized_corpus()[0]
        query_word_ids = [self.vocabulary[word] for word in query_tokens if word in self.vocabulary]
        if not query_word_ids:
            return []
        
        N = self.stats["N"]
        tf_matrix = self.stats["tf_matrix"]
        doc_len = self.stats["doc_len"]
        avgdl = self.stats["avgdl"]
        idf_array = self.stats["idf_array"]

        scores = np.zeros(N,dtype=np.float32)
        for word_id in query_word_ids:
            idf = idf_array[word_id]
            tf = tf_matrix[:,word_id]
            candidates =  tf>0
            len_norm = 1.0 - b + b*(doc_len[candidates]/avgdl)
            numerator = tf[candidates]*(k1+1.0)
            denominator = tf[candidates] + k1 * len_norm
            word_score = idf * (numerator / denominator)
            scores[candidates] += word_score
            
        top_indices = np.argsort(scores)[::-1][:top_k]
        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "text": self.chunks[idx]["text"],
                "doc_name": self.chunks[idx].get("doc_name", "Unknown"),
                "pages": self.chunks[idx].get("pages", []),
                "chunk_id": idx
            })
            
        return results




