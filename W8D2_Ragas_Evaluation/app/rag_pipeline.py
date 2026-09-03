from pathlib import Path

from langchain_chroma import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_ollama import OllamaLLM
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document


DATA_FILE = Path("data/knowledge.txt")
CHROMA_DIR = "chroma_db"


def build_rag(chunk_size=500, chunk_overlap=50, k=3):

    # Check dataset
    if not DATA_FILE.exists():
        raise FileNotFoundError(
            f"Dataset not found: {DATA_FILE}"
        )

    text = DATA_FILE.read_text(encoding="utf-8").strip()

    if not text:
        raise ValueError(
            "knowledge.txt is empty. Add your dataset text to data/knowledge.txt"
        )

    print(f"Loaded dataset: {len(text)} characters")

    document = Document(
        page_content=text,
        metadata={"source": "knowledge.txt"}
    )

    # Split document
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap
    )

    chunks = splitter.split_documents([document])

    if not chunks:
        raise ValueError("No chunks were created from the dataset.")

    print(f"Created {len(chunks)} chunks")

    # Embeddings
    print("Loading embedding model...")

    embeddings = HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

    # ChromaDB
    print("Creating ChromaDB vector store...")

    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        collection_name="w8d2_rag",
        persist_directory=CHROMA_DIR
    )

    print("ChromaDB created successfully.")

    retriever = vectorstore.as_retriever(
        search_kwargs={"k": k}
    )

    # Ollama
    llm = OllamaLLM(
        model="llama3.2:3b"
    )

    return retriever, llm


def answer_question(question, chunk_size=500, k=3):

    retriever, llm = build_rag(
        chunk_size=chunk_size,
        chunk_overlap=50,
        k=k
    )

    documents = retriever.invoke(question)

    if not documents:
        raise ValueError(
            "Retriever returned no documents."
        )

    context = "\n\n".join(
        document.page_content
        for document in documents
    )

    prompt = f"""
You are a helpful RAG assistant.

Answer the question using ONLY the provided context.

If the answer cannot be found in the context,
say that the information is not available
in the provided context.

Context:
{context}

Question:
{question}

Answer:
"""

    answer = llm.invoke(prompt)

    return {
        "question": question,
        "answer": answer,
        "contexts": [
            document.page_content
            for document in documents
        ]
    }


if __name__ == "__main__":

    result = answer_question(
        "What is a digital twin?",
        chunk_size=500,
        k=3
    )

    print("\n==============================")
    print("QUESTION")
    print("==============================")
    print(result["question"])

    print("\n==============================")
    print("ANSWER")
    print("==============================")
    print(result["answer"])

    print("\n==============================")
    print("RETRIEVED CONTEXT")
    print("==============================")

    for i, context in enumerate(result["contexts"], 1):
        print(f"\n--- Context {i} ---")
        print(context)