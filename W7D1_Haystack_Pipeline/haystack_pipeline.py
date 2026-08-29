from pathlib import Path

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore
from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentCleaner
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.writers import DocumentWriter
from haystack.components.retrievers.in_memory import InMemoryBM25Retriever


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"


def create_document_store():
    return InMemoryDocumentStore()


def index_documents(document_store):

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if len(pdf_files) < 5:
        raise ValueError(
            f"At least 5 PDF files are required. Found {len(pdf_files)}."
        )

    print(f"Found {len(pdf_files)} PDF documents.")

    converter = PyPDFToDocument()
    cleaner = DocumentCleaner()

    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=5,
        split_overlap=1
    )

    writer = DocumentWriter(
        document_store=document_store
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "converter",
        converter
    )

    pipeline.add_component(
        "cleaner",
        cleaner
    )

    pipeline.add_component(
        "splitter",
        splitter
    )

    pipeline.add_component(
        "writer",
        writer
    )

    pipeline.connect(
        "converter.documents",
        "cleaner.documents"
    )

    pipeline.connect(
        "cleaner.documents",
        "splitter.documents"
    )

    pipeline.connect(
        "splitter.documents",
        "writer.documents"
    )

    pipeline.run(
        {
            "converter": {
                "sources": pdf_files
            }
        }
    )

    print("Documents indexed successfully.")

    print(
        "Total documents/chunks in store:",
        document_store.count_documents()
    )


def create_bm25_retriever(document_store):

    return InMemoryBM25Retriever(
        document_store=document_store,
        top_k=3
    )


def ask_question(retriever, question):

    result = retriever.run(
        {
            "query": question,
            "top_k": 3
        }
    )

    documents = result["documents"]

    print("\n" + "=" * 70)
    print("QUESTION:", question)

    print("\nBM25 Retrieved Documents:")

    if not documents:
        print("No documents retrieved.")
        return

    for i, document in enumerate(
        documents,
        start=1
    ):

        source = document.meta.get(
            "file_path",
            document.meta.get(
                "source",
                "Unknown"
            )
        )

        print(f"\n{i}. Source: {source}")

        print(
            f"   Score: {document.score}"
        )

        preview = document.content[:300].replace(
            "\n",
            " "
        )

        print(
            f"   Preview: {preview}..."
        )


if __name__ == "__main__":

    print("=" * 70)
    print("W7D1 HAYSTACK BM25 PIPELINE")
    print("=" * 70)

    store = create_document_store()

    index_documents(store)

    retriever = create_bm25_retriever(store)

    questions = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is deep learning?",
        "What is natural language processing?",
        "What is computer vision?",
        "What are neural networks?",
        "What is supervised learning?",
        "What is unsupervised learning?",
        "What is the purpose of training data?",
        "How are AI models evaluated?"
    ]

    print("\nRunning 10 questions...")

    for question in questions:

        ask_question(
            retriever,
            question
        )

    print("\n" + "=" * 70)
    print("BM25 PIPELINE COMPLETED")
    print("=" * 70)