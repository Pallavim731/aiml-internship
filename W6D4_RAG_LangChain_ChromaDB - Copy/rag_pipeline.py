from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings, ChatOllama


# --------------------------------------------------
# 1. Load PDF
# --------------------------------------------------

PDF_PATH = "data/sample.pdf"

loader = PyPDFLoader(PDF_PATH)

documents = loader.load()

print(f"Loaded {len(documents)} pages from PDF.")


# --------------------------------------------------
# 2. Split PDF into chunks
# --------------------------------------------------

text_splitter = RecursiveCharacterTextSplitter(
    chunk_size=800,
    chunk_overlap=100
)

chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")


# --------------------------------------------------
# 3. Create embeddings
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# 4. Create Chroma vector store
# --------------------------------------------------

vectorstore = Chroma(
    collection_name="w6d4_pdf_rag",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 5. Add PDF chunks
# --------------------------------------------------

vectorstore.add_documents(chunks)

print("PDF chunks successfully stored in ChromaDB.")


# --------------------------------------------------
# 6. User question
# --------------------------------------------------

question = input("\nAsk a question about the PDF: ")


# --------------------------------------------------
# 7. Retrieve top 3 chunks
# --------------------------------------------------

retrieved_docs = vectorstore.similarity_search(
    question,
    k=3
)

print("\n--- TOP 3 RETRIEVED CHUNKS ---")

for i, doc in enumerate(retrieved_docs, start=1):
    print(f"\nChunk {i}")
    print("Source:", doc.metadata.get("source"))
    print("Page:", doc.metadata.get("page"))
    print("Content:")
    print(doc.page_content[:1000])


# --------------------------------------------------
# 8. Build context
# --------------------------------------------------

context = "\n\n".join(
    doc.page_content for doc in retrieved_docs
)


# --------------------------------------------------
# 9. Create prompt
# --------------------------------------------------

prompt = f"""
You are a helpful RAG assistant.

Answer the question ONLY using the provided context.

If the answer cannot be found in the context,
say "The answer is not available in the provided document."

Context:
{context}

Question:
{question}

Answer:
"""


# --------------------------------------------------
# 10. Ollama LLM
# --------------------------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# --------------------------------------------------
# 11. Generate answer
# --------------------------------------------------

response = llm.invoke(prompt)

print("\n--- FINAL RAG ANSWER ---")
print(response.content)