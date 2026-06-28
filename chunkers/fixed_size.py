import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List,Dict,Any,Tuple

class FixedSize:
    def __init__(self,pdf_path:str):
        self.pdf_path = pdf_path

    @staticmethod
    def clean_text(text:str) -> str:
        text = text.replace("\r\n","\n").replace("\r","\n")
        text = re.sub(r"[ \t]+"," ",text)
        lines = [line.strip() for line in text.split("\n")]
        text = "\n".join(lines)
        text = re.sub(r"\n{3,}","\n\n",text)
        return text.strip()

    def extract_pages(self)->List[Dict[str,Any]]:
        doc_name = self.pdf_path.split('/')[-1]
        pages_data = []
        with pymupdf.open(self.pdf_path) as doc:
            for page in doc:
                raw_text = page.get_text()
                cleaned_text = self.clean_text(raw_text)

                if not cleaned_text:
                    continue

                pages_data.append({
                    "doc_name":doc_name,
                    "page_number":page.number + 1,
                    "text":cleaned_text
                })

        return pages_data

    @staticmethod
    def words_with_metadata(pages_data:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        words = []
        global_idx = 0
        for page in pages_data:
            for match in re.finditer(r"\S+",page["text"]):
                words.append({
                    "text":match.group,
                    "page_number":page["page_number"],
                    "doc_name":page["doc_name"],
                    "global_idx":global_idx
                })

                global_idx += 1

        return words

    @staticmethod
    def chunk_words(word_units:List[Dict[str,Any]],chunk_size:int=200,overlap:int=30)->List[Dict[str,Any]]:
        if not word_units:
            return []

        if chunk_size <=0:
            raise ValueError("Chunk size must be greater than zero")

        if overlap<0:
            raise ValueError("Overlap size must be >= zero")

        if overlap >= chunk_size:
            raise ValueError("overlap must be smaller than chunk size")

        chunks = []
        total = len(word_units)
        stride = chunk_size-overlap
        start_idx = 0
        chunk_id = 0

        while start_idx<total:
            end_idx = min(start_idx+chunk_size,total)
            chunk_words = word_units[start_idx:end_idx]

            if not chunk_words:
                break

            pages = [w["page_number"] for w in chunk_words]

            if start_idx == 0 and end_idx == total:
                position = "full"
            elif start_idx == 0:
                position = "start"
            elif end_idx == total:
                position = "end"
            else:
                position = "middle"

            chunk = {
                "chunk_id": chunk_id,
                "text": " ".join(w["text"] for w in chunk_words),
                "doc_name": chunk_words[0]["doc_name"],
                "pages": pages,
                "chunker_name": "fixed_size",
                "source_unit": "words",
                "start_idx": chunk_words[0]["global_idx"],
                "end_idx": chunk_words[-1]["global_idx"],
                "word_count": len(chunk_words),
                "position": position,
                "source_words": chunk_words,
            }

            chunks.append(chunk)
            chunk_id += 1
            start_idx += stride

        return chunks


    def build_chunks(self,chunk_size:int = 200,overlap:int = 30)->List[Dict[str,Any]]:
        pages_data = self.extract_pages()
        word_units = self.words_with_metadata(pages_data)
        return self.chunk_words(word_units,chunk_size=chunk_size,overlap=overlap)










