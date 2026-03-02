class Evaluation:
    def __init__(self,retrieved_chunks,ground_truth,k):
        self.retrieved_chunks = retrieved_chunks
        self.ground_truth = ground_truth
        self.k = k

    def hit_rate_k(self):
        for doc in self.retrieved_chunks:
            if any(doc in gt for gt in self.ground_truth):
                return True
            
        return False
    
    def precision_at_k(self):
        hits = 0
        for doc in self.retrieved_chunks:
            if any(doc in gt for gt in self.ground_truth):
                hits+=1

        return hits/self.k
    
        



