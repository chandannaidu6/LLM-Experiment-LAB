import redis

client = redis.Redis.from_url("redis://localhost:6379", decode_responses=False)

try:
    client.ft("idx:rag_semantic_cache_v2").dropindex(delete_documents=False)
    print("index dropped")
except Exception as e:
    print("no index to drop:", e)

keys = client.keys("rag:cache:*")
if keys:
    client.delete(*keys)
print("deleted", len(keys), "keys")