from pathlib import Path

from haystack.components.converters import PyPDFToDocument
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.retrievers.in_memory import (
    InMemoryBM25Retriever,
    InMemoryEmbeddingRetriever,
)
from haystack.document_stores.in_memory import InMemoryDocumentStore

from haystack_integrations.components.embedders.sentence_transformers import (
    SentenceTransformersDocumentEmbedder,
    SentenceTransformersTextEmbedder,
)

from haystack_integrations.components.readers.transformers import (
    TransformersExtractiveReader,
)


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"


class HaystackRAG:

    def __init__(self):

        print("Creating document stores...")

        # Separate stores are used for BM25 and Dense retrieval.
        self.bm25_store = InMemoryDocumentStore()

        self.dense_store = InMemoryDocumentStore(
            embedding_similarity_function="cosine"
        )

        self.bm25_retriever = InMemoryBM25Retriever(
            document_store=self.bm25_store,
            top_k=5
        )

        self.dense_retriever = InMemoryEmbeddingRetriever(
            document_store=self.dense_store,
            top_k=5
        )

        print("Loading extractive reader...")

        self.reader = TransformersExtractiveReader(
            model="deepset/tinyroberta-squad2",
            top_k=3,
            no_answer=True
        )

        self.reader.warm_up()

        print("Loading text embedder...")

        self.text_embedder = SentenceTransformersTextEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        self.text_embedder.warm_up()

        self.index_documents()

    # ------------------------------------------------------
    # INDEX PDF DOCUMENTS
    # ------------------------------------------------------

    def index_documents(self):

        pdf_files = list(DATA_DIR.glob("*.pdf"))

        if not pdf_files:
            raise FileNotFoundError(
                f"No PDF files found in {DATA_DIR}"
            )

        print(f"\nFound {len(pdf_files)} PDF files.")

        converter = PyPDFToDocument()

        splitter = DocumentSplitter(
            split_by="word",
            split_length=300,
            split_overlap=30
        )

        all_documents = []

        for pdf_file in pdf_files:

            print(f"Reading: {pdf_file.name}")

            try:

                conversion_result = converter.run(
                    sources=[pdf_file]
                )

                documents = conversion_result["documents"]

                split_result = splitter.run(
                    documents=documents
                )

                split_documents = split_result["documents"]

                for document in split_documents:
                    document.meta["source_file"] = pdf_file.name

                all_documents.extend(split_documents)

                print(
                    f"  Added {len(split_documents)} chunks"
                )

            except Exception as exc:

                print(
                    f"  ERROR reading {pdf_file.name}: {exc}"
                )

        if not all_documents:

            raise RuntimeError(
                "No valid documents were created from the PDFs."
            )

        print(
            f"\nTotal chunks created: {len(all_documents)}"
        )

        # --------------------------------------------------
        # BM25 STORE
        # --------------------------------------------------

        self.bm25_store.write_documents(
            all_documents
        )

        print(
            f"BM25 index created with "
            f"{len(all_documents)} documents."
        )

        # --------------------------------------------------
        # DENSE STORE
        # --------------------------------------------------

        print("\nCreating document embeddings...")

        document_embedder = SentenceTransformersDocumentEmbedder(
            model="sentence-transformers/all-MiniLM-L6-v2"
        )

        document_embedder.warm_up()

        embedding_result = document_embedder.run(
            documents=all_documents
        )

        embedded_documents = embedding_result["documents"]

        self.dense_store.write_documents(
            embedded_documents
        )

        print(
            f"Dense index created with "
            f"{len(embedded_documents)} documents."
        )

    # ------------------------------------------------------
    # BM25 SEARCH
    # ------------------------------------------------------

    def bm25_search(self, query, top_k=5):

        result = self.bm25_retriever.run(
            query=query,
            top_k=top_k
        )

        return result["documents"]

    # ------------------------------------------------------
    # DENSE SEARCH
    # ------------------------------------------------------

    def dense_search(self, query, top_k=5):

        embedding_result = self.text_embedder.run(
            text=query
        )

        query_embedding = embedding_result["embedding"]

        result = self.dense_retriever.run(
            query_embedding=query_embedding,
            top_k=top_k
        )

        return result["documents"]

    # ------------------------------------------------------
    # READER
    # ------------------------------------------------------

    def read_answer(self, query, documents):

        if not documents:
            return []

        result = self.reader.run(
            query=query,
            documents=documents
        )

        return result["answers"]

    # ------------------------------------------------------
    # BM25 ANSWER
    # ------------------------------------------------------

    def answer_bm25(self, query):

        documents = self.bm25_search(
            query=query,
            top_k=5
        )

        answers = self.read_answer(
            query=query,
            documents=documents
        )

        return {
            "query": query,
            "answers": answers,
            "documents": documents
        }

    # ------------------------------------------------------
    # DENSE ANSWER
    # ------------------------------------------------------

    def answer_dense(self, query):

        documents = self.dense_search(
            query=query,
            top_k=5
        )

        answers = self.read_answer(
            query=query,
            documents=documents
        )

        return {
            "query": query,
            "answers": answers,
            "documents": documents
        }


# ==========================================================
# TEST THE PIPELINE
# ==========================================================

if __name__ == "__main__":

    rag = HaystackRAG()

    query = "What is artificial intelligence?"

    # ------------------------------------------------------
    # BM25
    # ------------------------------------------------------

    print("\n======================================")
    print("BM25 RESULTS")
    print("======================================")

    bm25_result = rag.answer_bm25(query)

    print(f"\nQuery: {query}")

    print("\nRetrieved documents:")

    for i, document in enumerate(
        bm25_result["documents"],
        start=1
    ):

        print(
            f"{i}. "
            f"{document.meta.get('source_file', 'unknown')} "
            f"| score={document.score}"
        )

    print("\nAnswers:")

    for answer in bm25_result["answers"]:

        print(
            f"- {answer.data} "
            f"| score={answer.score}"
        )

    # ------------------------------------------------------
    # DENSE
    # ------------------------------------------------------

    print("\n======================================")
    print("DENSE RESULTS")
    print("======================================")

    dense_result = rag.answer_dense(query)

    print(f"\nQuery: {query}")

    print("\nRetrieved documents:")

    for i, document in enumerate(
        dense_result["documents"],
        start=1
    ):

        print(
            f"{i}. "
            f"{document.meta.get('source_file', 'unknown')} "
            f"| score={document.score}"
        )

    print("\nAnswers:")

    for answer in dense_result["answers"]:

        print(
            f"- {answer.data} "
            f"| score={answer.score}"
        )