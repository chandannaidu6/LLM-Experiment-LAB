class Evaluation:
    def __init__(self,retrieved_ids,source_ids,k,total = None):
        self.retrieved_ids = list(retrieved_ids)[:k]
        self.source_ids =  set(source_ids)
        self.k = k
        self.total = total

    def hit_at_k(self):
        return len(self.source_ids.intersection(self.retrieved_ids))
    
    def hit_rate_k(self):
        return self.hit_at_k()>0
    
    def precision_at_k(self):
        if self.k == 0:
            return 0.0
        
        return self.hit_at_k/self.k
    

    def recall_at_k(self):
        if not self.source_ids:
            return 0.0
        return self.hit_at_k()/len(self.source_ids)
    

