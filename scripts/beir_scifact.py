from pathlib import Path 
from collections import defaultdict
import json
from datasets import load_dataset

OUT_DIR = Path("data/beir_scifact")
OUT_DIR.mkdir(parents=True,exist_ok=True)

def main():
    corpus_ds = load_dataset("BeIR/scifact","corpus")["corpus"]
    queries_ds = load_dataset("BeIR/scifact","queries")["queries"]
    qrels_ds = load_dataset("BeIR/scifact-qrels",split="test")

    chunks = []
    for doc in corpus_ds:
        doc_id = doc["_id"]
        title = (doc.get("title") or "").strip()
        text = (doc.get("text") or "").strip()
        full_text = (title + "\n" + text).strip() if title else text

        chunks.append(
            {
                "doc_name":doc_id,
                "chunk_id":0,
                "title":title,
                "text":full_text,
                "score":0.0
            }
        )

    questions = []
    for q in  queries_ds:
        questions.append(
            {
                "id":q['_id'],
                "question":q["text"],

            }
        )

    qrels = defaultdict(list)
    for row in qrels_ds:
        if int(row["score"]) >0:
            qrels[row["query-id"]].append(row["corpus-id"])

    with open(OUT_DIR/"chunks.json","w",encoding="utf-8") as f:
        json.dump(chunks,f,ensure_ascii=False,indent=2)

    with open(OUT_DIR/"questions.json","w",encoding="utf-8") as f:
        json.dump(questions,f,ensure_ascii=False,indent=2)

    with open(OUT_DIR/"qrels.json","w",encoding="utf-8") as f:
        json.dump(qrels,f,ensure_ascii=False,indent=2)


    print("Wrote:",OUT_DIR)

if __name__ == "__main__":
    main()


    

