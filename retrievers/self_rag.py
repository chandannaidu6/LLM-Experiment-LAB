from __future__ import annotations
import json 
import re
from typing import List, Dict,Any,Callable, Optional

class SelfRAGRetriever:
    def __init__(self,base_retriever:Any,judge_fn:Callable[[str],str],answer_fn:Optional[Callable[[str],str]] = None, min_relevant_chunks:int = 2,retry_multiplier:int = 2, max_retry_top_k:int = 12,enable_query_rewrite:bool = True):
        self.base_retriever = base_retriever
        self.judge_fn = judge_fn
        self.answer_fn = answer_fn or judge_fn
        self.min_relevant_chunks = min_relevant_chunks
        self.retry_multiplier = retry_multiplier
        self.max_retry_top_k = max_retry_top_k
        self.enable_query_rewrite = enable_query_rewrite

        self.last_trace:Dict[str,Any] = {}
        self.last_answer:Optional[str] = None
        self.last_answer_support: Optional[Dict[str,Any]] = None

    def build_index(self) -> None:
        if hasattr(self.base_retriever,"build_index"):
            self.base_retriever.build_index()

    def retrieve(self,query:str,top_k:int = 5) -> List[Dict[str,Any]]:
        trace:Dict[str,Any] = {
            "query":query,
            "top_k_requested":top_k,
            "initial_candidates":[],
            "initial_isrel":[],
            "retry_used":False,
            "retry_query":None,
            "retry_candidates": [],
            "retry_isrel":[],
            "final_chunks":[],
            "answer":None,
            "issup":None
        }
        initial_candidates = self.base_retriever.retrieve(query=query,top_k=top_k)
        trace["initial_candidates"] = initial_candidates

        initial_graded = self._grade_chunks(query,initial_candidates)
        trace["initial_isrel"] = initial_graded
        relevant_chunks = self._keep_relevant(initial_graded)

        if len(relevant_chunks) < self.min_relevant_chunks:
            trace["retry_used"] = True
            retry_query = self._rewrite_query(query,initial_candidates) if self.enable_query_rewrite else query
            trace["retry_query"] = retry_query
            retry_top_k = min(max(top_k*self.retry_multiplier,top_k+1),self.max_retry_top_k)
            retry_candidates = self.base_retriever.retrieve(query=retry_query,top_k=retry_top_k)
            trace["retry_candidates"] = retry_candidates
            retry_graded  = self._grade_chunks(query,retry_candidates)
            retry_relevant = self._keep_relevant(retry_graded)

            if len(retry_relevant) > len(relevant_chunks):
                relevant_chunks = retry_relevant

        final_chunks = relevant_chunks[:top_k]
        trace["final_chunks"] = final_chunks




    def _grade_chunks(self,query:str,chunks:List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        graded = []

        for chunk in chunks:
            prompt = self._build_isrel_prompt(query,chunk)
            raw = self.judge_fn(prompt)
            parsed = self._parse_json(raw,default={
                "label":"Irrelevant",
                "reason":"Parser fallback",
                "confidence":0.0
            })

            graded.append({
                "chunk":chunk,
                "label":str(parsed.get("label","Irrelevant")).strip(),
                "reason":str(parsed.get("reason","")).strip(),
                "confidence":float(parsed.get("confidence",0.0) or 0.0)
            })

            return graded

    def _keep_relevant(self,graded_chunks: List[Dict[str,Any]]) -> List[Dict[str,Any]]:
        kept = []
        for item in graded_chunks:
            label = item.get("label","").strip().lower()
            if label == "relevant":
                chunk = dict(item["chunk"])
                chunk["isrel_reason"] = item.get("reason")
                chunk["isrel_confidence"] = item.get("confidence")
                kept.append(chunk)
        return kept

    def _rewrite_query(self,query:str,graded_chunks:List[Dict[str,Any]])->str:
        rejected_reasons = [
            item.get("reason","")
            for item in graded_chunks
            if item.get("reason","").strip().lower() != "relevant"
        ][:5]
        prompt = f"""
        You are rewriting a retrieval query to improve document search.

        Original query:
        {query}

        Observed problems with retrieved chunks:
        {json.dumps(rejected_reasons, ensure_ascii=False)}

        Rewrite the query to be more specific and retrieval-friendly.
        Return JSON only:
        {{
        "rewritten_query": "..."
        }}
        """.strip()
        raw = self.judge_fn(prompt)
        parsed = self._parse_json(raw,default = {"rewritten_query":query})
        rewritten = str(parsed.get("rewritten_query",query)).strip()
        return rewritten or query

    def _generate_answer(self,query:str,chunks:List[Dict[str,Any]]) -> str:
        if not chunks:
            return "Insufficient support in retrieved contexts"

        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                f"[Chunk {i} | chunk_id={chunk.get('chunk_id')} | "
                f"doc={chunk.get('doc_name')} | page={chunk.get('page_number')}]\n"
                f"{chunk.get('text', '')}"
            )

        prompt = f"""
        Answer the question using only the provided context.
        If the context is insufficient, say so clearly and do not invent facts.

        Question:
        {query}

        Context:
        {chr(10).join(context_blocks)}

        Return JSON only:
        {{
        "answer": "..."
        }}
        """.strip()

        raw = self.answer_fn(prompt)
        parsed = self._parse_json(
            raw,
            default={"answer": "Insufficient support in retrieved context."},
        )
    return str(parsed.get("answer", "")).strip()

    def _grade_answer_support(self,query:str,answer:str,chunks:List[Dict[str,Any]]) -> Dict[str,Any]:
        context_blocks = []
        for i, chunk in enumerate(chunks, start=1):
            context_blocks.append(
                f"[Chunk {i} | chunk_id={chunk.get('chunk_id')}]\n{chunk.get('text', '')}"
            )

        prompt = f"""
        You are grading whether an answer is supported by retrieved context.

        Question:
        {query}

        Answer:
        {answer}

        Context:
        {chr(10).join(context_blocks)}

        Return JSON only:
        {{
        "label": "Supported" or "Unsupported",
        "reason": "...",
        "confidence": 0.0
        }}
""".strip()

        raw = self.judge_fn(prompt)
        return self._parse_json(
            raw,
            default={
                "label": "Unsupported",
                "reason": "Parser fallback",
                "confidence": 0.0,
            },
        )



    def _build_isrel_prompt(self,query:str.chunk:Dict[str,Any])->str:
        return f"""
        You are grading whether a retrieved chunk is relevant to a user query.

        Query:
        {query}

        Chunk metadata:
        doc_name={chunk.get('doc_name')}
        page_number={chunk.get('page_number')}
        chunk_id={chunk.get('chunk_id')}

        Chunk text:
        {chunk.get('text', '')}

        Return JSON only:
        {{
        "label": "Relevant" or "Irrelevant",
        "reason": "...",
        "confidence": 0.0
        }}
        """.strip()

    def _parse_json(self,text:str,default:Dict[str,Any])->Dict[str,Any]:
        if not text:
            return default

        text = text.strip()

        try:
            return json.loads(text)
        except Exception:
            pass

        match = re.search(r"\{.*\}",text,re.DOTALL)

        if match:
            try:
                return json.loads(match.group(0))
            except Exception:
                pass

        return default
