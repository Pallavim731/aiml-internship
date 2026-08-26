import time
import chromadb

from llama_index.core import (
    VectorStoreIndex,
    SimpleDirectoryReader,
    StorageContext,
    Settings
)

from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "data"

OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"


# --------------------------------------------------
# Configure Ollama
# --------------------------------------------------

Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=120.0
)

Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)


# --------------------------------------------------
# Load documents
# --------------------------------------------------

print("=" * 60)
print("W7D3 LLAMAINDEX + CHROMADB")
print("=" * 60)

print("\nLoading documents...")

documents = SimpleDirectoryReader(DATA_DIR).load_data()

print(f"Loaded {len(documents)} documents.")


# --------------------------------------------------
# Create ChromaDB client
# --------------------------------------------------

print("\nConnecting to ChromaDB...")

chroma_client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = chroma_client.get_or_create_collection(
    name="w7d3_documents"
)


# --------------------------------------------------
# Create Chroma Vector Store
# --------------------------------------------------

vector_store = ChromaVectorStore(
    chroma_collection=collection
)

storage_context = StorageContext.from_defaults(
    vector_store=vector_store
)


# --------------------------------------------------
# Create VectorStoreIndex
# --------------------------------------------------

print("\nCreating ChromaDB-backed index...")

start_time = time.perf_counter()

index = VectorStoreIndex.from_documents(
    documents,
    storage_context=storage_context
)

indexing_time = time.perf_counter() - start_time

print(f"Index created successfully.")
print(f"Indexing time: {indexing_time:.4f} seconds")


# --------------------------------------------------
# Query Engine
# --------------------------------------------------

query_engine = index.as_query_engine(
    similarity_top_k=3
)


# --------------------------------------------------
# Ten Questions
# --------------------------------------------------

queries = [
    "What is Artificial Intelligence?",
    "What are the main tasks performed by AI systems?",
    "What is Machine Learning?",
    "What are the three common types of Machine Learning?",
    "What is Deep Learning?",
    "Where is Deep Learning commonly used?",
    "What is Natural Language Processing?",
    "What are some applications of NLP?",
    "What is Retrieval Augmented Generation?",
    "Why are vector databases used in RAG systems?"
]


# --------------------------------------------------
# Run queries
# --------------------------------------------------

results_file = "results/chroma_results.txt"

total_latency = 0

with open(results_file, "w", encoding="utf-8") as file:

    file.write("W7D3 - LLAMAINDEX + CHROMADB QUERY RESULTS\n")
    file.write("=" * 70 + "\n\n")

    for number, query in enumerate(queries, start=1):

        print("\n" + "-" * 60)
        print(f"Query {number}: {query}")

        start_time = time.perf_counter()

        response = query_engine.query(query)

        latency = time.perf_counter() - start_time

        total_latency += latency

        print(f"Answer: {response}")
        print(f"Latency: {latency:.4f} seconds")

        file.write(f"Query {number}: {query}\n")
        file.write(f"Answer: {response}\n")
        file.write(f"Latency: {latency:.4f} seconds\n")
        file.write("-" * 70 + "\n\n")


average_latency = total_latency / len(queries)

print("\n" + "=" * 60)
print("CHROMADB RESULTS")
print("=" * 60)

print(f"Average query latency: {average_latency:.4f} seconds")

with open(results_file, "a", encoding="utf-8") as file:

    file.write("\n")
    file.write("=" * 70 + "\n")
    file.write(
        f"Average query latency: {average_latency:.4f} seconds\n"
    )

print(f"\nResults saved to: {results_file}")