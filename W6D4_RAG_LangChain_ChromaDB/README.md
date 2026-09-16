# W6D4 — RAG Pipeline with LangChain + ChromaDB

## Objective

Build a Retrieval Augmented Generation pipeline using:

- LangChain
- ChromaDB
- Ollama
- Local embedding model
- Local LLM

## Architecture

PDF
↓
PyPDFLoader
↓
Text Chunking
↓
Ollama Embeddings
↓
ChromaDB
↓
Top-3 Similarity Retrieval
↓
Context
↓
Ollama LLM
↓
Final Answer

## Technologies

- Python
- LangChain
- ChromaDB
- Ollama
- llama3.2:3b
- nomic-embed-text

## Tasks Completed

- [x] Installed ChromaDB
- [x] Created ChromaDB collection
- [x] Added 20 documents
- [x] Generated embeddings
- [x] Performed similarity search
- [x] Performed metadata filtering
- [x] Loaded PDF
- [x] Split PDF into chunks
- [x] Stored PDF embeddings in ChromaDB
- [x] Retrieved top-3 chunks
- [x] Passed retrieved context to Ollama
- [x] Manually verified answer

## How to Run

```bash
pip install -r requirements.txt
python chroma_setup.py