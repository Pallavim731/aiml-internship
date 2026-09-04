from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from app.pipeline import HaystackRAG


app = FastAPI(
    title="Haystack Retrieval API",
    description="BM25 and Dense Retrieval API using Haystack",
    version="1.0.0",
)


# Load and index the PDFs when the API starts.
rag = HaystackRAG()


class QueryRequest(BaseModel):
    query: str
    method: str = "bm25"
    top_k: int = 5


@app.get("/")
def root():
    return {
        "message": "Haystack Retrieval API is running",
        "methods": ["bm25", "dense"],
    }


@app.get("/health")
def health():
    return {
        "status": "healthy",
        "documents": rag.bm25_store.count_documents(),
    }


@app.post("/search")
def search(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    if request.top_k < 1 or request.top_k > 20:
        raise HTTPException(
            status_code=400,
            detail="top_k must be between 1 and 20.",
        )

    method = request.method.lower()

    if method == "bm25":

        documents = rag.bm25_search(
            query=query,
            top_k=request.top_k,
        )

    elif method == "dense":

        documents = rag.dense_search(
            query=query,
            top_k=request.top_k,
        )

    else:

        raise HTTPException(
            status_code=400,
            detail="method must be 'bm25' or 'dense'.",
        )

    results = []

    for document in documents:

        results.append(
            {
                "content": document.content,
                "source": document.meta.get(
                    "source_file",
                    "unknown",
                ),
                "score": document.score,
            }
        )

    return {
        "query": query,
        "method": method,
        "top_k": request.top_k,
        "results": results,
    }


@app.post("/answer")
def answer(request: QueryRequest):

    query = request.query.strip()

    if not query:
        raise HTTPException(
            status_code=400,
            detail="Query cannot be empty.",
        )

    method = request.method.lower()

    if method == "bm25":

        result = rag.answer_bm25(query)

    elif method == "dense":

        result = rag.answer_dense(query)

    else:

        raise HTTPException(
            status_code=400,
            detail="method must be 'bm25' or 'dense'.",
        )

    answers = []

    for answer in result["answers"]:

        answers.append(
            {
                "answer": answer.data,
                "score": answer.score,
            }
        )

    return {
        "query": query,
        "method": method,
        "answers": answers,
    }