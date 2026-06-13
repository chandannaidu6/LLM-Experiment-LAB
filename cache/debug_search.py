# debug2.py
import redis
client = redis.Redis.from_url("redis://localhost:6379", decode_responses=False)
INDEX = "idx:rag_semantic_cache_v2"

kw = client.connection_pool.connection_kwargs
print("connection:", kw.get("host"), kw.get("port"), "db", kw.get("db"))
print("all indexes:", client.execute_command("FT._LIST"))

info = client.ft(INDEX).info()
print("num_docs (script sees):", info.get("num_docs"))
print("definition:", info.get("index_definition"))

print("raw search:", client.execute_command("FT.SEARCH", INDEX, "*", "LIMIT", "0", "10"))