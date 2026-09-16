import chromadb


client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_collection(
    name="w6d4_documents"
)

print("Collection name:", collection.name)
print("Number of documents:", collection.count())

print("\nChroma collection successfully loaded.")
print("Similarity search is performed using vector embeddings.")
print("The retrieved results are ranked by vector distance.")
