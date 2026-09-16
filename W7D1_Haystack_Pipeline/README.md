# W7D1: Haystack Pipeline Architecture

## Objective

Built a Haystack document retrieval and extractive question-answering pipeline using five PDF documents.

## Technologies

- Python
- Haystack
- PyPDF
- Sentence Transformers
- BM25
- Dense Embedding Retrieval

## Pipeline Architecture

### BM25

PDF Files
↓
PyPDFToDocument
↓
DocumentCleaner
↓
DocumentSplitter
↓
InMemoryDocumentStore
↓
InMemoryBM25Retriever
↓
ExtractiveReader
↓
Answer

### Dense Retrieval

PDF Files
↓
PyPDFToDocument
↓
DocumentCleaner
↓
DocumentSplitter
↓
SentenceTransformersDocumentEmbedder
↓
InMemoryDocumentStore
↓
SentenceTransformersTextEmbedder
↓
InMemoryEmbeddingRetriever
↓
ExtractiveReader
↓
Answer

## Documents

Five PDF documents were indexed.

## Evaluation

Ten questions were tested using both retrieval approaches.

Precision@1 was calculated manually based on whether the top retrieved document was relevant to the question.

## BM25 vs Dense Retrieval

BM25 uses keyword matching and is effective when query terminology appears directly in the document.

Dense retrieval uses vector embeddings and is better suited for semantic similarity and queries using different wording.

## Conclusion

The experiment compared lexical retrieval and semantic retrieval using the same documents and questions.
