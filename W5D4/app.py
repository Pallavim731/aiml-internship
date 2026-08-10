import chromadb
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# 1. Load embedding model
# ---------------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# ---------------------------------------------------------
# 2. Create ChromaDB client
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# ---------------------------------------------------------
# 3. Create collection
# ---------------------------------------------------------

collection = client.get_or_create_collection(
    name="w5d3_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


# ---------------------------------------------------------
# 4. Prepare 20 documents
# ---------------------------------------------------------

documents = [
    "Machine learning enables computers to learn patterns from data.",
    "Supervised learning uses labeled training data.",
    "Unsupervised learning discovers patterns in unlabeled data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing allows computers to understand text.",
    "Computer vision enables machines to analyze images and videos.",
    "Python is widely used for machine learning and artificial intelligence.",
    "Scikit-learn provides many classical machine learning algorithms.",
    "TensorFlow is a popular framework for deep learning.",
    "PyTorch is widely used for neural network research and development.",
    "Data preprocessing improves the quality of machine learning input.",
    "Feature engineering creates useful variables from raw data.",
    "Model evaluation measures how well a machine learning model performs.",
    "Precision measures the proportion of predicted positives that are correct.",
    "Recall measures the proportion of actual positives that are identified.",
    "Vector databases store numerical representations of data.",
    "Embeddings represent text as numerical vectors.",
    "ChromaDB is a vector database designed for AI applications.",
    "Retrieval augmented generation combines document retrieval with language models.",
    "Ollama allows large language models to run locally."
]


# ---------------------------------------------------------
# 5. Metadata
# ---------------------------------------------------------

categories = [
    "machine-learning",
    "machine-learning",
    "machine-learning",
    "deep-learning",
    "nlp",
    "computer-vision",
    "programming",
    "machine-learning",
    "deep-learning",
    "deep-learning",
    "data-preprocessing",
    "data-preprocessing",
    "evaluation",
    "evaluation",
    "evaluation",
    "vector-database",
    "embeddings",
    "vector-database",
    "rag",
    "llm"
]


# ---------------------------------------------------------
# 6. Generate embeddings
# ---------------------------------------------------------

print("\nGenerating embeddings...")

embeddings = embedding_model.encode(
    documents
).tolist()

print("Embeddings generated.")


# ---------------------------------------------------------
# 7. Add documents to ChromaDB
# ---------------------------------------------------------

collection.upsert(
    ids=[f"doc_{i+1}" for i in range(len(documents))],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"category": category}
        for category in categories
    ]
)

print(f"\nAdded {len(documents)} documents to ChromaDB.")


# ---------------------------------------------------------
# 8. Verify document count
# ---------------------------------------------------------

print("\nTotal documents:")
print(collection.count())


# ---------------------------------------------------------
# 9. Similarity search using cosine distance
# ---------------------------------------------------------

query = "How do computers learn from data?"

query_embedding = embedding_model.encode(
    [query]
).tolist()

results = collection.query(
    query_embeddings=query_embedding,
    n_results=3
)


print("\n==============================")
print("COSINE SIMILARITY SEARCH")
print("==============================")

for i, document in enumerate(results["documents"][0]):
    print(f"\nResult {i+1}:")
    print(document)

    print(
        "Metadata:",
        results["metadatas"][0][i]
    )

    print(
        "Distance:",
        results["distances"][0][i]
    )


# ---------------------------------------------------------
# 10. Metadata filtering
# ---------------------------------------------------------

print("\n==============================")
print("METADATA FILTERING")
print("==============================")


filtered_results = collection.query(
    query_embeddings=query_embedding,
    n_results=5,
    where={
        "category": "machine-learning"
    }
)


for i, document in enumerate(
    filtered_results["documents"][0]
):

    print(f"\nFiltered Result {i+1}:")
    print(document)

    print(
        "Metadata:",
        filtered_results["metadatas"][0][i]
    )