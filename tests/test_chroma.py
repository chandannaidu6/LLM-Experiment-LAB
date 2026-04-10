import chromadb

client = chromadb.HttpClient(host="localhost",port=8001)
collection = client.get_or_create_collection(name="test_chunks",metadata={"similarity":"cosine"})

collection.upsert(
    ids=["c1","c2"],
    documents=[
        "Paris is the capital of France.",
        "Berlin is the capital of Germany."
    ],
    embeddings=[
        [0.1, 0.2, 0.3],
        [0.9, 0.8, 0.7]
    ],
    metadatas=[
        {"doc_name": "doc1", "page_number": 1},
        {"doc_name": "doc2", "page_number": 2},
    ],    
)

results = collection.query(
    query_embeddings=[[0.1,0.2,0.29]],
    n_results = 1,
    include=["documents","metadatas","distances"]
)
print(results)