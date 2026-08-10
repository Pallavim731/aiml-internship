import requests
import chromadb

from pypdf import PdfReader
from sentence_transformers import SentenceTransformer


# ---------------------------------------------------------
# Configuration
# ---------------------------------------------------------

PDF_PATH = "sample.pdf"

OLLAMA_URL = "http://localhost:11434/api/generate"

OLLAMA_MODEL = "llama3.2:3b"


# ---------------------------------------------------------
# 1. Read PDF
# ---------------------------------------------------------

print("Reading PDF...")

reader = PdfReader(PDF_PATH)

text = ""

for page in reader.pages:
    page_text = page.extract_text()

    if page_text:
        text += page_text + "\n"


print("PDF text extracted.")


# ---------------------------------------------------------
# 2. Split text into chunks
# ---------------------------------------------------------

chunk_size = 500

chunks = [
    text[i:i + chunk_size]
    for i in range(0, len(text), chunk_size)
]

print(f"Created {len(chunks)} chunks.")


# ---------------------------------------------------------
# 3. Load embedding model
# ---------------------------------------------------------

print("Loading embedding model...")

embedding_model = SentenceTransformer(
    "all-MiniLM-L6-v2"
)


# ---------------------------------------------------------
# 4. Create ChromaDB collection
# ---------------------------------------------------------

client = chromadb.PersistentClient(
    path="./chroma_db"
)

collection = client.get_or_create_collection(
    name="pdf_documents",
    metadata={
        "hnsw:space": "cosine"
    }
)


# ---------------------------------------------------------
# 5. Generate embeddings
# ---------------------------------------------------------

embeddings = embedding_model.encode(
    chunks
).tolist()


# ---------------------------------------------------------
# 6. Store PDF chunks
# ---------------------------------------------------------

collection.upsert(
    ids=[
        f"pdf_chunk_{i}"
        for i in range(len(chunks))
    ],
    documents=chunks,
    embeddings=embeddings,
    metadatas=[
        {
            "source": PDF_PATH,
            "chunk": i
        }
        for i in range(len(chunks))
    ]
)


print(
    f"Stored {len(chunks)} PDF chunks in ChromaDB."
)


# ---------------------------------------------------------
# 7. Ask question
# ---------------------------------------------------------

question = input(
    "\nAsk a question about the PDF: "
)


# ---------------------------------------------------------
# 8. Embed question
# ---------------------------------------------------------

question_embedding = embedding_model.encode(
    [question]
).tolist()


# ---------------------------------------------------------
# 9. Retrieve top 3 chunks
# ---------------------------------------------------------

results = collection.query(
    query_embeddings=question_embedding,
    n_results=3
)


retrieved_chunks = results["documents"][0]


print("\n==============================")
print("TOP 3 RETRIEVED CHUNKS")
print("==============================")


for i, chunk in enumerate(retrieved_chunks):

    print(f"\nChunk {i+1}:")
    print(chunk)


# ---------------------------------------------------------
# 10. Build context
# ---------------------------------------------------------

context = "\n\n".join(
    retrieved_chunks
)


# ---------------------------------------------------------
# 11. Create RAG prompt
# ---------------------------------------------------------

prompt = f"""
You are an AI assistant answering questions
using only the provided context.

Context:
{context}

Question:
{question}

Instructions:
- Answer using the provided context.
- Do not invent information.
- If the answer is not available in the context,
  say that the information is not available.
"""


# ---------------------------------------------------------
# 12. Send prompt to Ollama
# ---------------------------------------------------------

print("\nSending context to Ollama...")

response = requests.post(
    OLLAMA_URL,
    json={
        "model": OLLAMA_MODEL,
        "prompt": prompt,
        "stream": False
    },
    timeout=120
)


# ---------------------------------------------------------
# 13. Display answer
# ---------------------------------------------------------

if response.status_code == 200:

    result = response.json()

    print("\n==============================")
    print("OLLAMA ANSWER")
    print("==============================")

    print(result["response"])

else:

    print("Ollama request failed.")

    print(
        "Status code:",
        response.status_code
    )

    print(response.text)