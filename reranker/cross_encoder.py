from sentence_transformers import CrossEncoder

class Reranker:
    def __init__(self,model="cross-encoder/ms-marco-MiniLM-L6-v2"):
        self.model = CrossEncoder(model)

    def rerank(self,query,candidates,top_k:int=5):
        if not candidates:
            return []
        pairs = []
        normalized_candidates = []

        for c in candidates:
            text = c.get("text","")
            if isinstance(text,list):
                text = " ".join(str(x) for x in text)
            elif text is None:
                text = ""
            else:
                text = str(text)
            item  = dict(c)
            item["text"] = text
            normalized_candidates.append(item)
            pairs.append((str(query),text))
        scores = self.model.predict(pairs)
        reranked = []
        for candidate,score in zip(normalized_candidates,scores):
            item = dict(candidate)
            item["rerank_score"] = float(score)
            reranked.append(item)

        reranked.sort(key=lambda x:x["rerank_score"],reverse=True)
        return reranked[:top_k]
