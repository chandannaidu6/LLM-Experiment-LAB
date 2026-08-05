import pymupdf
import numpy as np
import re
from pathlib import Path
from typing import List,Dict,Any,Tuple
from schema import Chunk,Query
import json

class Benchmark:
    QUERY_ALIASES = {
        "_id": "query_id", "id": "query_id", "query_id": "query_id",
        "question": "text", "query": "text", "text": "text"
    }
    def __init__(self,qrel_path:str|Path,chunks,question_path:str|Path):
        self.qrel_path = Path(qrel_path)
        self.chunk_path = Path(chunk_path)
        self.chunks = chunks

    def load(self):
        for p in (self.question_path,self.qrel_path):
            if not p.exists():
                raise FileNotFoundError("File does not exist {p}")
        queries_lookup = self._load_queries(self.question_path) # query_id -> text
        qrels = self._load_qrels(self.qrel_path) # query_id -> {doc_id->relevance}
        corpus_ids = {c.doc_name if c in self.chunks} if self.chunks else None
        queries:List[Query] = []
        skipped_no_text = 0
        missing_gold = 0

        for qid,rel_map in gold.items():
            text = queries_lookup.get(qid)
            if text is None:
                skipped_no_text+=1
            if corpus_ids is not None:
                missing_gold += sum(1 for d in rel_map if d not in corpus_ids)
            queries.append(Query(
                query_id = qid,
                text = text,
                gold_doc_ids = list(rel_map.keys())
                gold_relevance = rel_map
            ))
        if not queries:
            raise ValueError(
                "No queries built - the qrels/questions join produced nothing. "
                "Check that query ids match between the two files."
            )
        print(f"Built {len(queries)} queries; "
              f"skipped {skipped_no_text} (no question text); "
              f"{missing_gold} gold ids missing from corpus.")
        return queries

    @staticmethod
    def _read_file(path:Path):
        text = path.read_text(encoding="utf-8")
        if path.suffix.lower() == ".jsonl":
            return [json.loads(line) for line in text.splitlines() if line.strip()]
        data = json.load(text)
        return data

    def _load_queries(self) -> Dict[str,str]:
        raw  = self._read_file(self.question_path)
        lookup:Dict[str,str] = {}
        for rec in raw:
            mapped = {self.QUERY_ALIASES.get(k,k):v for k,v in rec.items()}
            if "query_id" not in mapped or "text" not in mapped:
                continue
            lookup[str(mapped["query_id"])] = mapped["text"]
        return lookup

    def _load_qrels(self)-> Dict[str,Dict[str,int]]:
        raw = self.read_file(self.qrel_path)
        if isinstance(raw,dict):
            return {
                str(qid):{str(doc):1 for doc in doc_list}
                for qid,doc_list in raw.items()
            }
        #Assuming the shape is something like [{query_id,corpus_id,score}]
        if isinstance(raw,list):
            gold: Dict[str[Dict[str,int]]] = {}
            for row in raw:
                if isinstance(row,dict):
                    qid = row.get("query_id") or row.get("query-id") or row.get("qid")
                    doc = row.get("corpus_id") or row.get("corpus-id") or row.get("doc_id")
                    rel = int(row.get("score",1))
                else:
                    qid,doc,rel = row[0],row[1],int(row[2]) if len(row)>2 else 1
                if rel<=0:
                    continue
                gold.setdefault(str(qid),{})[str(doc)] = rel

            return gold

        raise ValueError(f"Unrecognized qrel format in {self.qrel_path}")



            


        
