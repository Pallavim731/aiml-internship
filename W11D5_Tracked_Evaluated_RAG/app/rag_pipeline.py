from pathlib import Path

from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter


# Paths
BASE_DIR = Path(__file__).resolve().parent.parent
KNOWLEDGE_FILE = BASE_DIR / "data" / "knowledge.txt"
CHROMA_DIR = BASE_DIR / "chroma_db"


# Ollama embedding model
EMBEDDING_MODEL = "nomic-embed-text"


def build_vector_store():
    """Load the knowledge base, split it into chunks, and store embeddings."""

    # Read knowledge base
    text = KNOWLEDGE_FILE.read_text(encoding="utf-8")

    # Split the document into smaller chunks
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50,
    )

    documents = splitter.create_documents([text])

    # Create Ollama embeddings
    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    # Store documents and embeddings in Chroma
    vector_store = Chroma.from_documents(
        documents=documents,
        embedding=embeddings,
        persist_directory=str(CHROMA_DIR),
        collection_name="w11d5_rag",
    )

    return vector_store


def get_retriever(k=3):
    """Create the vector store and return a retriever."""

    vector_store = build_vector_store()

    return vector_store.as_retriever(
        search_kwargs={"k": k}
    )


def retrieve_context(question, k=3):
    """Retrieve relevant knowledge for a question."""

    retriever = get_retriever(k=k)

    documents = retriever.invoke(question)

    return [document.page_content for document in documents]


if __name__ == "__main__":
    question = "What is RAG?"

    contexts = retrieve_context(question)

    print("\nRetrieved Context:\n")

    for index, context in enumerate(contexts, start=1):
        print(f"--- Context {index} ---")
        print(context)