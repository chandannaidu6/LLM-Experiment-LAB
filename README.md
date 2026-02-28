# LLM-Experiment-LAB

An experiment rig for Retrieval-Augumented Generation(RAG) and LLM agents- compare chunking,retrievers, rerankers and agent loops with real benchmarks.

## Problem 

LLM apps are evrywhere, but most teams still:

1. Pick **chunk sizes, retrievers, and prompts by intuition**, not data.
2. Ship a RAG stack, then discover **hallucinations and missed documents** only from user complaints.
3. Have **no reproducible way** to answer basic questions like:
   - "Is BM25 + fixed chunks actually worse than dense retrieval + semantic chunks?"
   - "What's the quality vs latency vs cost trade-off between configs?"
4. End up rewriting **evaluation scripts** for every project, with no UI and no history of experiments.
This makes RAG and agent systems hard to debug, hard to improve and hard to justify to stakeholders.

## What this project solves

This project is a **web based experiment lab** for retrieval and rag systems:

1. Treats your RAG stack as **system to be benchmarked**, not a single chatbot.
2. Lets you **define experiments** over:
  - Chunking strategies(fixed, semantic, sliding window, recursive)
  - Retrievers(BM25, dense, hybrid, HyDE, self-RAG–style loops)
  - Rerankers and LLM models
3. Runs them on **curated QA benchmarks** or replayed production queries.
4. Produces **side-by-side metrics**: retrieval accuracy, answer quality, latency, and cost.

You get a single place to answer: *“Which RAG + agent config should we actually deploy, and why?”*
