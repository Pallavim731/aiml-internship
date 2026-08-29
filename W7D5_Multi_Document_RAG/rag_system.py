import os
import chromadb

from llama_index.core import (
    Document,
    Settings,
    StorageContext,
    VectorStoreIndex,
)
from llama_index.core.node_parser import SentenceSplitter
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.ollama import Ollama
from llama_index.vector_stores.chroma import ChromaVectorStore


# Get the project folder path
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# Project directories
DATA_DIRECTORY = os.path.join(BASE_DIR, "data")
CHROMA_DIRECTORY = os.path.join(BASE_DIR, "chroma_db")

COLLECTION_NAME = "multi_document_rag"


def configure_models():
    """
    Configure local embedding model and Ollama LLM.
    No OpenAI API key is required.
    """

    Settings.embed_model = HuggingFaceEmbedding(
        model_name="BAAI/bge-small-en-v1.5"
    )

    Settings.llm = Ollama(
        model="llama3.2:3b",
        request_timeout=120.0
    )

    Settings.node_parser = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=50
    )


def load_documents():
    """
    Load all text files from the data folder.
    """

    documents = []

    if not os.path.exists(DATA_DIRECTORY):
        print(f"Data folder not found: {DATA_DIRECTORY}")
        return documents

    for filename in os.listdir(DATA_DIRECTORY):

        if filename.endswith(".txt"):

            file_path = os.path.join(
                DATA_DIRECTORY,
                filename
            )

            with open(
                file_path,
                "r",
                encoding="utf-8"
            ) as file:

                content = file.read()

                documents.append(
                    Document(
                        text=content,
                        metadata={
                            "source": filename
                        }
                    )
                )

                print(f"Loaded: {filename}")

    return documents


def create_index(documents):
    """
    Create a ChromaDB vector store and index documents.
    """

    if not documents:
        raise ValueError(
            "No documents found. "
            "Add .txt files inside the data folder."
        )

    chroma_client = chromadb.PersistentClient(
        path=CHROMA_DIRECTORY
    )

    chroma_collection = chroma_client.get_or_create_collection(
        COLLECTION_NAME
    )

    vector_store = ChromaVectorStore(
        chroma_collection=chroma_collection
    )

    storage_context = StorageContext.from_defaults(
        vector_store=vector_store
    )

    index = VectorStoreIndex.from_documents(
        documents,
        storage_context=storage_context
    )

    return index


def display_sources(response):
    """
    Display the documents used for generating the answer.
    """

    print("\nSources used:")

    for source_node in response.source_nodes:

        source = source_node.node.metadata.get(
            "source",
            "Unknown"
        )

        print(f"- {source}")


def main():

    print("=" * 60)
    print("MULTI-DOCUMENT RAG SYSTEM")
    print("=" * 60)

    print("\nConfiguring local models...")

    configure_models()

    print("Local models configured successfully.")

    print("\nLoading documents...")

    documents = load_documents()

    print(f"\nTotal documents loaded: {len(documents)}")

    if len(documents) == 0:
        print("\nNo documents found.")
        print(f"Please add .txt files to: {DATA_DIRECTORY}")
        return

    print("\nCreating vector index...")

    index = create_index(documents)

    print("Vector index created successfully.")

    query_engine = index.as_query_engine(
        similarity_top_k=3
    )

    print("\nRAG system is ready.")
    print("Type 'exit' to stop.")

    while True:

        question = input("\nAsk a question: ")

        if question.lower() == "exit":
            print("Closing RAG system.")
            break

        response = query_engine.query(question)

        print("\nAnswer:")
        print(response)

        display_sources(response)


if __name__ == "__main__":
    main()