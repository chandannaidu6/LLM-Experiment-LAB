import random

class Randomize:
    @staticmethod
    def create_positioned_list(gold_chunk,all_chunks,position="middle",total_docs=20):
        distractors =[
            c for c in all_chunks
            if not (
                c.get('doc_name') == gold_chunk.get('doc_name') and 
                c.get('chunk_id') == gold_chunk.get('chunk_id')
            )
        ]
        n = min(total_docs-1,len(distractors))
        distractors = random.sample(distractors,n)

        if position  == "start":
            return [gold_chunk] + distractors
        elif position == "end":
            return distractors + [gold_chunk]
        
        else:
            mid = len(distractors)//2
            return distractors[:mid] + [gold_chunk] + distractors[mid:]

