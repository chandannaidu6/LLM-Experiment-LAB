import json
from pathlib import Path
from chunkers.slide_chunking import SlideChunker
from preprocessing import preprocessing
from retrievers.bm25 import BM25
from retrievers.dense import DenseRetriever
from llm.openai_client import OpenAIClient
from evaluation.metrics import Evaluation
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = Path("Data")
QUESTIONS_PATH = Path("QA/questions.json")

def load_chunks():
    chunks  = []
    for pdf_path in DATA_DIR.glob("*.pdf"):
        ch = SlideChunker(str(pdf_path))
        pages_data = ch.extract_text()
        sentences = ch.metadata_sentences(pages_data)
        doc_chunks = ch.chunk_sentences(sentences)
        chunks.extend(doc_chunks)

    return chunks

def bm25_index(chunks):
    prep = preprocessing(chunks,corpus=None)
    tokenized = prep.tokenized_corpus()
    vocab = prep.vocabulary_construction(tokenized)

    bm25_chunks = []
    for chunk,tokens in zip(chunks,tokenized):
        c = dict(chunk)
        c["text"] = tokens
        bm25_chunks.append(c)

    dummy_query = "dummy"
    bm = BM25(dummy_query,bm25_chunks,vocab)
    stats = bm.IDF()
    return bm25_chunks,vocab,stats

def dense_index(chunks):
    dense = DenseRetriever(chunks)
    dense.build_index()
    return dense

def load_questions():
    with open(QUESTIONS_PATH,"r") as f:
        return json.load(f)
    
def bm25_retriever(query, bm25_chunks, vocab, stats, top_k=5):
    bm = BM25(query,bm25_chunks,vocab)
    bm.stats = stats
    return bm.bm25_retriever(top_k=top_k)

def dense_retrieve(query,dense_retriever,top_k=5):
    return dense_retriever.retrieve(query=query,top_k=top_k)

def rag_answer(query,context_chunks,mode="plain"):
    context = "\n\n".join(f"[{i+1}] {c['doc_name']} (chunk {c['chunk_id']}):{c['text']}" for i,c in enumerate(context_chunks))
    client = OpenAIClient()
    return client.chat(query,context,mode=mode)

def eval_doc(qid,results,questions,top_k):
    q = next(q for q in questions if q["id"] == qid)
    gold_doc = q["source_doc"]
    retrieved_ids = [r["doc_name"] for r in results]
    source_ids = [gold_doc]
    ev = Evaluation(retrieved_ids, source_ids, k=min(top_k, len(retrieved_ids)))
    print(f"\nEvaluation for {qid} (gold_doc={gold_doc})")
    print("  hit@k      :", ev.hit_at_k())
    print("  hit_rate@k :", ev.hit_rate_k())
    print("  precision@k:", ev.precision_at_k())
    print("  recall@k   :", ev.recall_at_k())

def main():
    print("Loading chunks from PDFs...")
    chunks = load_chunks()
    print(f"Total chunks: {len(chunks)}")

    print("Building BM25 index...")
    bm25_chunks, vocab, stats = bm25_index(chunks)

    print("Building dense index (OpenAI embeddings)...")
    dense_retriever = dense_index(chunks)

    questions = load_questions()

    while True:
        print("\nMenu:")
        print(" 1) BM25 retrieve")
        print(" 2) Dense retrieve")
        print(" 3) RAG answer (BM25 + LLM)")
        print(" 4) RAG answer (Dense + LLM)")
        print(" 5) Evaluate BM25 on a question id")
        print(" 6) Quit")
        choice = input("Select option: ").strip()

        if choice == "1":
            q = input("Enter query: ")
            res = bm25_retriever(q, bm25_chunks, vocab, stats, top_k=5)
            for r in res:
                print(f"[{r['score']:.4f}] {r['doc_name']}: {str(r['text'])[:160]}...")
        elif choice == "2":
            q = input("Enter query: ")
            res = dense_retrieve(q, dense_retriever, top_k=5)
            for r in res:
                print(f"[{r['score']:.4f}] {r['doc_name']}: {r['text'][:160]}...")
        elif choice == "3":
            q = input("Enter query: ")
            res = bm25_retriever(q, bm25_chunks, vocab, stats, top_k=5)
            answer = rag_answer(q, res, mode="plain")
            print("\nRAG answer (BM25, plain):\n", answer)
        elif choice == "4":
            q = input("Enter query: ")
            res = dense_retrieve(q, dense_retriever, top_k=5)
            answer = rag_answer(q, res, mode="cot")
            print("\nRAG answer (Dense, CoT):\n", answer)
        elif choice == "5":
            qid = input("Enter question id (e.g., q16): ").strip()
            q = next((qq for qq in questions if qq["id"] == qid), None)
            if q is None:
                print("Unknown question id.")
                continue
            res = bm25_retriever(q["question"], bm25_chunks, vocab, stats, top_k=5)
            eval_doc(qid, res, questions, top_k=5)
        elif choice == "6":
            break
        else:
            print("Invalid choice.")


if __name__ == "__main__":
    main()







