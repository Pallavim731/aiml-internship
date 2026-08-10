import chromadb
from sentence_transformers import SentenceTransformer


# =========================================================
# 1. Load embedding model
# =========================================================

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)

print("Embedding model loaded.")


# =========================================================
# 2. Create persistent ChromaDB client
# =========================================================

client = chromadb.PersistentClient(
    path="./chroma_db"
)


# =========================================================
# 3. Create collection using cosine distance
# =========================================================

collection = client.get_or_create_collection(
    name="semantic_search_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


# =========================================================
# 4. Prepare 20 documents
# =========================================================

documents = [
    "Machine learning enables computers to learn patterns from data.",
    "Supervised learning uses labeled training data.",
    "Unsupervised learning discovers patterns from unlabeled data.",
    "Deep learning uses neural networks with multiple layers.",
    "Natural language processing allows computers to understand text.",
    "Computer vision enables machines to analyze images and videos.",
    "Python is widely used for machine learning and artificial intelligence.",
    "Scikit-learn provides algorithms for classical machine learning.",
    "TensorFlow is a framework used for deep learning.",
    "PyTorch is widely used for neural network research.",
    "Data preprocessing improves the quality of machine learning data.",
    "Feature engineering creates useful features from raw data.",
    "Model evaluation measures machine learning model performance.",
    "Precision measures the proportion of predicted positives that are correct.",
    "Recall measures the proportion of actual positives that are identified.",
    "Vector databases store numerical representations of data.",
    "Embeddings represent text as numerical vectors.",
    "ChromaDB is a vector database for AI applications.",
    "Retrieval augmented generation combines retrieval with language models.",
    "Ollama allows large language models to run locally."
]


# =========================================================
# 5. Metadata
# =========================================================

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
    "feature-engineering",
    "evaluation",
    "evaluation",
    "evaluation",
    "vector-database",
    "embeddings",
    "vector-database",
    "rag",
    "llm"
]


# =========================================================
# 6. Generate embeddings
# =========================================================

print("\nGenerating embeddings...")

embeddings = embedding_model.encode(
    documents,
    normalize_embeddings=True
).tolist()

print("Embeddings generated.")


# =========================================================
# 7. Store documents in ChromaDB
# =========================================================

collection.upsert(
    ids=[
        f"doc_{i + 1}"
        for i in range(len(documents))
    ],
    documents=documents,
    embeddings=embeddings,
    metadatas=[
        {"category": category}
        for category in categories
    ]
)


print(
    f"\nStored {len(documents)} documents."
)


# =========================================================
# 8. Verify collection
# =========================================================

print(
    "Total documents in collection:",
    collection.count()
)


# =========================================================
# 9. Semantic search function
# =========================================================

def semantic_search(query, top_k=3):

    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist()

    results = collection.query(
        query_embeddings=query_embedding,
        n_results=top_k
    )

    return results


# =========================================================
# 10. Perform semantic search
# =========================================================

query = "How can computers learn from information?"

results = semantic_search(
    query,
    top_k=3
)


print("\n===================================")
print("SEMANTIC SEARCH RESULTS")
print("===================================")

print("Query:", query)


for i, document in enumerate(
    results["documents"][0]
):

    print(f"\nResult {i + 1}")
    print("Document:", document)
    print(
        "Category:",
        results["metadatas"][0][i]["category"]
    )
    print(
        "Cosine distance:",
        results["distances"][0][i]
    )


# =========================================================
# 11. Metadata filtering
# =========================================================

filtered_results = collection.query(
    query_embeddings=embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).tolist(),
    n_results=5,
    where={
        "category": "machine-learning"
    }
)


print("\n===================================")
print("METADATA FILTERED RESULTS")
print("===================================")


for i, document in enumerate(
    filtered_results["documents"][0]
):

    print(f"\nFiltered Result {i + 1}")
    print("Document:", document)
    print(
        "Category:",
        filtered_results["metadatas"][0][i]["category"]
    )