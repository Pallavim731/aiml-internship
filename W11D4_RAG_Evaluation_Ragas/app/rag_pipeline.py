from pathlib import Path

from langchain_ollama import OllamaEmbeddings
from langchain_chroma import Chroma
from langchain_text_splitters import RecursiveCharacterTextSplitter


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "knowledge.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"


def build_vector_store(chunk_size=500, chunk_overlap=50):
    """Load documents, split them into chunks, and store embeddings."""

    text = DATA_FILE.read_text(encoding="utf-8")

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.create_documents([text])

    embeddings = OllamaEmbeddings(
        model="nomic-embed-text"
    )

    vector_store = Chroma(
        collection_name="w11d4_rag",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_DIR)
    )

    # Clear previous documents so each experiment starts cleanly.
    try:
        existing = vector_store.get()
        if existing["ids"]:
            vector_store.delete(ids=existing["ids"])
    except Exception:
        pass

    vector_store.add_documents(chunks)

    print(f"Created {len(chunks)} chunks.")
    print(f"Chunk size: {chunk_size}")
    print(f"Chunk overlap: {chunk_overlap}")

    return vector_store


def retrieve_documents(vector_store, question, k=3):
    """Retrieve the top-k relevant chunks."""

    documents = vector_store.similarity_search(
        question,
        k=k
    )

    return documents


if __name__ == "__main__":
    store = build_vector_store()

    question = "What does RAG stand for?"

    results = retrieve_documents(
        store,
        question,
        k=3
    )

    print("\nQuestion:")
    print(question)

    print("\nRetrieved Context:")
    for index, document in enumerate(results, start=1):
        print(f"\n--- Chunk {index} ---")
        print(document.page_content)