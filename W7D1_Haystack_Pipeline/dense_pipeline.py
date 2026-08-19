from pathlib import Path

from haystack import Pipeline
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import (
    DocumentCleaner,
    DocumentSplitter
)
from haystack.components.writers import DocumentWriter

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder
)

from haystack.components.retrievers.in_memory import (
    InMemoryEmbeddingRetriever
)


BASE_DIR = Path(__file__).resolve().parent
PDF_DIR = BASE_DIR / "pdfs"

MODEL = "sentence-transformers/all-MiniLM-L6-v2"


def create_document_store():

    return InMemoryDocumentStore(
        embedding_similarity_function="cosine"
    )


def index_documents(document_store):

    pdf_files = list(PDF_DIR.glob("*.pdf"))

    if len(pdf_files) < 5:
        raise ValueError(
            f"At least 5 PDF documents are required. "
            f"Found {len(pdf_files)}."
        )

    print(f"Found {len(pdf_files)} PDF documents.")

    converter = PyPDFToDocument()

    cleaner = DocumentCleaner()

    splitter = DocumentSplitter(
        split_by="sentence",
        split_length=5,
        split_overlap=1
    )

    document_embedder = SentenceTransformersDocumentEmbedder(
        model=MODEL
    )

    document_embedder.warm_up()

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
        "embedder",
        document_embedder
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
        "embedder.documents"
    )

    pipeline.connect(
        "embedder.documents",
        "writer.documents"
    )

    pipeline.run(
        {
            "converter": {
                "sources": pdf_files
            }
        }
    )

    print("Dense document indexing completed.")

    print(
        "Total documents/chunks:",
        document_store.count_documents()
    )


def create_dense_retrieval_pipeline(document_store):

    text_embedder = SentenceTransformersTextEmbedder(
        model=MODEL
    )

    text_embedder.warm_up()

    retriever = InMemoryEmbeddingRetriever(
        document_store=document_store,
        top_k=3
    )

    pipeline = Pipeline()

    pipeline.add_component(
        "text_embedder",
        text_embedder
    )

    pipeline.add_component(
        "retriever",
        retriever
    )

    pipeline.connect(
        "text_embedder.embedding",
        "retriever.query_embedding"
    )

    return pipeline


def ask_question(pipeline, question):

    result = pipeline.run(
        {
            "text_embedder": {
                "text": question
            }
        }
    )

    documents = result["retriever"]["documents"]

    print("\n" + "=" * 70)

    print(
        "QUESTION:",
        question
    )

    print("\nDense Retrieved Documents:")

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

        print(
            f"\n{i}. Source: {source}"
        )

        print(
            f"   Score: {document.score:.4f}"
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
    print("W7D1 HAYSTACK DENSE RETRIEVAL PIPELINE")
    print("=" * 70)

    store = create_document_store()

    index_documents(store)

    dense_pipeline = create_dense_retrieval_pipeline(
        store
    )

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
            dense_pipeline,
            question
        )

    print("\n" + "=" * 70)
    print("DENSE RETRIEVAL PIPELINE COMPLETED")
    print("=" * 70)