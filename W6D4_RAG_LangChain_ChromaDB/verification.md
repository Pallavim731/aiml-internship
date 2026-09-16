# W6D4 RAG Verification

## 1. ChromaDB Collection

- Collection: w6d4_documents
- Documents added: 20
- Embedding model: nomic-embed-text
- Vector database: ChromaDB

Result: PASS

---

## 2. Similarity Search

Query:

How does artificial intelligence learn from data?

Top 3 relevant documents were retrieved using embedding-based similarity search.

Result: PASS

---

## 3. Metadata Filtering

Filter:

category = databases

Only documents belonging to the databases category were retrieved.

Result: PASS

---

## 4. PDF RAG Pipeline

Pipeline:

PDF
→ PDF Loader
→ Text Splitting
→ Embeddings
→ ChromaDB
→ Top-3 Retrieval
→ Context
→ Ollama
→ Answer

Result: PASS

---

## 5. Manual Answer Verification

The generated answer was compared against the retrieved PDF chunks.

Result: PASS

The final answer was supported by information present in the retrieved context.
