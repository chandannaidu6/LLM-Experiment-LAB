import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List,Dict,Any,Tuple


class SlideChunker:
    def __init__(self,pdf_path: str | Path):
        self.pdf_path = pdf_path

    def extract_text(self) -> List[Dict[str,Any]]:
        doc_name = self.pdf_path.split('/')[-1]
        pages_data = []
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                number = page.number
                text = page.get_text()
                pages_data.append({
                    "text":text,
                    "number":number,
                    "title":doc_name
                })

        return pages_data
    
    @staticmethod
    def split_sentences(text:str,min_length=20) -> List[str]:
        sentences = re.split(r'(?<=[.?!])\s+',text)
        return [s.strip() for s in sentences if len(s.strip())>=min_length]
    

    def metadata_sentences(self,data:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        sentences = []
        global_id = 0
        for page in data:
            page_sentences = self.split_sentences(page['text'])
            for sen_text in page_sentences:
                sentences.append({
                    'text':sen_text,
                    'doc_name':page['title'],
                    "page_number":page['number'],
                    'global_idx':global_id
                })
                global_id+=1

        return sentences
    
    @staticmethod
    def chunk_sentences(sentences:List[Dict[str,Any]],chunk_size:int=5,overlap:int=2) -> List[Dict[str,Any]]:
        if not sentences:
            return []
        
        chunks = []
        total = len(sentences)
        start_id = 0
        while start_id<total:
            end_id = min(start_id+chunk_size,total)
            chunk_sents = sentences[start_id:end_id]

            if len(chunk_sents)<2 and chunks:
                chunks[-1]['source_sentences'].extend(chunk_sents),
                chunks[-1]['text'] = " ".join(s['text'] for s in chunks[-1]['source_sentences']),
                chunks[-1]['sentence_range'] = (chunks[-1]['sentence_range'][0],chunk_sents[-1]['global_idx']),
                chunks[-1]['pages'] = sorted(list(set(chunks[-1]['pages'] + [s['page_number'] for s in chunk_sents])))
                break


            chunk = {
                "text": " ".join(s['text'] for s in chunk_sents),
                "doc_name":chunk_sents[0]['doc_name'],
                "sentence_range":(chunk_sents[0]['global_idx'],chunk_sents[-1]['global_idx']),
                "pages": sorted(list(set([s['page_number'] for s in chunk_sents]))),
                "source_sentences":chunk_sents,
                "position":"start" if start_id == 0 else (end_id if end_id>total else "middle")
            }
            chunks.append(chunk)
            start_id += (chunk_size-overlap)

        return chunks

    



    
