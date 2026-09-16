from langchain_chroma import Chroma
from langchain_ollama import OllamaEmbeddings
from langchain_core.documents import Document


# --------------------------------------------------
# 1. Embedding model
# --------------------------------------------------

embeddings = OllamaEmbeddings(
    model="nomic-embed-text"
)


# --------------------------------------------------
# 2. Create 20 documents
# --------------------------------------------------

documents = [
    Document(
        page_content="Python is a high-level programming language widely used in artificial intelligence and data science.",
        metadata={"category": "programming", "source": "doc1"}
    ),
    Document(
        page_content="Machine learning allows computers to learn patterns from data without being explicitly programmed.",
        metadata={"category": "machine_learning", "source": "doc2"}
    ),
    Document(
        page_content="Deep learning uses neural networks with multiple layers to learn complex patterns.",
        metadata={"category": "deep_learning", "source": "doc3"}
    ),
    Document(
        page_content="Natural language processing enables computers to process and understand human language.",
        metadata={"category": "nlp", "source": "doc4"}
    ),
    Document(
        page_content="Computer vision enables machines to analyze and understand images and videos.",
        metadata={"category": "computer_vision", "source": "doc5"}
    ),
    Document(
        page_content="Artificial intelligence is used in healthcare, finance, education, transportation, and manufacturing.",
        metadata={"category": "artificial_intelligence", "source": "doc6"}
    ),
    Document(
        page_content="Supervised learning uses labeled training data to learn a mapping between inputs and outputs.",
        metadata={"category": "machine_learning", "source": "doc7"}
    ),
    Document(
        page_content="Unsupervised learning discovers hidden patterns and structures in unlabeled data.",
        metadata={"category": "machine_learning", "source": "doc8"}
    ),
    Document(
        page_content="Reinforcement learning trains agents through rewards and penalties based on their actions.",
        metadata={"category": "machine_learning", "source": "doc9"}
    ),
    Document(
        page_content="Vector databases store numerical representations of data called embeddings.",
        metadata={"category": "databases", "source": "doc10"}
    ),
    Document(
        page_content="Embeddings represent text, images, or other information as vectors in a numerical space.",
        metadata={"category": "embeddings", "source": "doc11"}
    ),
    Document(
        page_content="ChromaDB is a vector database designed for storing and searching embeddings.",
        metadata={"category": "databases", "source": "doc12"}
    ),
    Document(
        page_content="Retrieval augmented generation combines document retrieval with language model generation.",
        metadata={"category": "rag", "source": "doc13"}
    ),
    Document(
        page_content="LangChain provides tools and abstractions for building applications powered by language models.",
        metadata={"category": "langchain", "source": "doc14"}
    ),
    Document(
        page_content="Ollama allows developers to run large language models locally.",
        metadata={"category": "ollama", "source": "doc15"}
    ),
    Document(
        page_content="Cosine similarity measures the angle between two vectors and is commonly used for semantic search.",
        metadata={"category": "similarity", "source": "doc16"}
    ),
    Document(
        page_content="Metadata filtering allows vector database searches to restrict results based on document attributes.",
        metadata={"category": "retrieval", "source": "doc17"}
    ),
    Document(
        page_content="Text splitting divides large documents into smaller chunks that can be embedded and retrieved efficiently.",
        metadata={"category": "rag", "source": "doc18"}
    ),
    Document(
        page_content="A retriever selects relevant documents from a knowledge base based on a user query.",
        metadata={"category": "retrieval", "source": "doc19"}
    ),
    Document(
        page_content="A RAG system can reduce hallucinations by grounding language model responses in retrieved documents.",
        metadata={"category": "rag", "source": "doc20"}
    ),
]


# --------------------------------------------------
# 3. Create Chroma vector store
# --------------------------------------------------

vectorstore = Chroma(
    collection_name="w6d4_documents",
    embedding_function=embeddings,
    persist_directory="./chroma_db"
)


# --------------------------------------------------
# 4. Add documents
# --------------------------------------------------

ids = [f"doc-{i}" for i in range(1, 21)]

vectorstore.add_documents(
    documents=documents,
    ids=ids
)

print("Successfully added 20 documents to ChromaDB.")


# --------------------------------------------------
# 5. Similarity search
# --------------------------------------------------

query = "How does artificial intelligence learn from data?"

results = vectorstore.similarity_search_with_score(
    query,
    k=3
)

print("\n--- COSINE/SIMILARITY SEARCH ---")

for i, (doc, score) in enumerate(results, start=1):
    print(f"\nResult {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)
    print("Score:", score)


# --------------------------------------------------
# 6. Metadata filtering
# --------------------------------------------------

filtered_results = vectorstore.similarity_search(
    "vector databases and embeddings",
    k=5,
    filter={"category": "databases"}
)

print("\n--- METADATA FILTERING ---")

for i, doc in enumerate(filtered_results, start=1):
    print(f"\nResult {i}")
    print("Content:", doc.page_content)
    print("Metadata:", doc.metadata)