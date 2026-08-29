# W7D2: Haystack Retrieval — BM25 & Dense Retrieval

## Objective

Build a Haystack retrieval system using five PDF documents and compare BM25 retrieval with dense retrieval.

## Technologies

- Python
- Haystack
- PyPDF
- BM25 Retrieval
- Sentence Transformers
- InMemoryDocumentStore

## Pipeline

### BM25

PDF Documents
↓
PyPDFToDocument
↓
InMemoryDocumentStore
↓
InMemoryBM25Retriever
↓
Top-k Documents

### Dense Retrieval

PDF Documents
↓
PyPDFToDocument
↓
Sentence Transformers Document Embeddings
↓
InMemoryDocumentStore
↓
InMemoryEmbeddingRetriever
↓
Top-k Documents

## Dataset

Five PDF documents were indexed.

## Evaluation

Ten questions were used for manual retrieval evaluation.

Precision@3 was calculated for both BM25 and Dense Retrieval.

## Results

| Metric      |       BM25 |      Dense |
| ----------- | ---------: | ---------: |
| Questions   |         10 |         10 |
| Top-k       |          3 |          3 |
| Precision@3 | Add result | Add result |

## Conclusion

BM25 and Dense Retrieval were compared using the same ten questions. BM25 is based mainly on keyword matching, while dense retrieval uses semantic similarity. The better approach was determined based on the observed Precision@3 results.
