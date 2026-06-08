import chromadb

# 1. Connect to the database you just created
client = chromadb.PersistentClient(path="data_vector_db")

# 2. Get the collection
collection = client.get_collection(name="langchain")

# 3. Peek at the first 3 entries
results = collection.peek(limit=3)

print("--- DATABASE SNAPSHOT ---")
for i in range(len(results['documents'])):
    print(f"\n[ID]: {results['ids'][i]}")
    print(f"[SOURCE]: {results['metadatas'][i]['source']}")
    print(f"[CONTENT PREVIEW]: {results['documents'][i][:200]}...") 
    print("-" * 30)

# 4. Total count check
print(f"\nTotal legal chunks indexed: {collection.count()}")

