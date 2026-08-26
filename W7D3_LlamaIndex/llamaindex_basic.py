import os
import time

from llama_index.core import VectorStoreIndex, SimpleDirectoryReader, Settings
from llama_index.embeddings.ollama import OllamaEmbedding
from llama_index.llms.ollama import Ollama


# --------------------------------------------------
# Configuration
# --------------------------------------------------

DATA_DIR = "data"

OLLAMA_MODEL = "llama3.2:3b"
EMBEDDING_MODEL = "nomic-embed-text"


# --------------------------------------------------
# Configure Ollama LLM and Embeddings
# --------------------------------------------------

Settings.llm = Ollama(
    model=OLLAMA_MODEL,
    request_timeout=600.0
)

Settings.embed_model = OllamaEmbedding(
    model_name=EMBEDDING_MODEL,
    base_url="http://localhost:11434"
)


# --------------------------------------------------
# Load documents
# --------------------------------------------------

print("=" * 60)
print("W7D3 LLAMAINDEX DOCUMENT INDEXING")
print("=" * 60)

print("\nLoading documents...")

documents = SimpleDirectoryReader(DATA_DIR).load_data()

print(f"Loaded {len(documents)} documents.")


# --------------------------------------------------
# Create VectorStoreIndex
# --------------------------------------------------

print("\nCreating VectorStoreIndex...")

start_time = time.perf_counter()

index = VectorStoreIndex.from_documents(documents)

indexing_time = time.perf_counter() - start_time

print(f"Index created successfully.")
print(f"Indexing time: {indexing_time:.4f} seconds")


# --------------------------------------------------
# Create Query Engine
# --------------------------------------------------

query_engine = index.as_query_engine(
    similarity_top_k=2,
    response_mode="compact"
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
# Run Queries
# --------------------------------------------------

results_file = "results/basic_results.txt"

with open(results_file, "w", encoding="utf-8") as file:

    file.write("W7D3 - LLAMAINDEX BASIC QUERY RESULTS\n")
    file.write("=" * 70 + "\n\n")

    for number, query in enumerate(queries, start=1):

        print("\n" + "-" * 60)
        print(f"Query {number}: {query}")

        start_time = time.perf_counter()

        response = query_engine.query(query)

        latency = time.perf_counter() - start_time

        print(f"Answer: {response}")
        print(f"Latency: {latency:.4f} seconds")

        file.write(f"Query {number}: {query}\n")
        file.write(f"Answer: {response}\n")
        file.write(f"Latency: {latency:.4f} seconds\n")
        file.write("-" * 70 + "\n\n")


print("\nResults saved to:")
print(results_file)