from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool

from langchain_ollama import (
    OllamaLLM,
    OllamaEmbeddings
)

from langchain_community.document_loaders import TextLoader

from langchain_text_splitters import (
    RecursiveCharacterTextSplitter
)

from langchain_chroma import Chroma


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

DOCUMENT_PATH = BASE_DIR / "sample.txt"

CHROMA_PATH = BASE_DIR / "chroma_db"

LLM_MODEL = "llama3.2:3b"

EMBEDDING_MODEL = "nomic-embed-text"


# ============================================================
# Create Ollama LLM
# ============================================================

def create_llm():

    return OllamaLLM(
        model=LLM_MODEL
    )


# ============================================================
# Step 1: Basic LangChain Chain
# ============================================================

def build_chain():

    prompt = PromptTemplate(
        input_variables=["question"],
        template=(
            "You are a helpful AI assistant.\n"
            "Answer the following question clearly and concisely.\n\n"
            "Question: {question}\n"
            "Answer:"
        )
    )

    llm = create_llm()

    parser = StrOutputParser()

    chain = (
        prompt
        | llm
        | parser
    )

    return chain


# ============================================================
# Step 2: Test Chain with 5 Inputs
# ============================================================

def test_chain():

    print("\n")
    print("=" * 60)
    print("1. CHAIN TEST - 5 INPUTS")
    print("=" * 60)

    chain = build_chain()

    questions = [
        "What is LangChain?",
        "What is Ollama?",
        "What is ChromaDB?",
        "What are embeddings?",
        "What is RAG?"
    ]

    results = []

    for index, question in enumerate(
        questions,
        start=1
    ):

        print(f"\nTest {index}")
        print("-" * 60)

        print(
            f"Question: {question}"
        )

        try:

            answer = chain.invoke(
                {
                    "question": question
                }
            )

            print(
                f"Answer: {answer}"
            )

            results.append(
                {
                    "test": index,
                    "question": question,
                    "answer": answer
                }
            )

        except Exception as error:

            print(
                f"Error: {error}"
            )

            results.append(
                {
                    "test": index,
                    "question": question,
                    "answer": f"ERROR: {error}"
                }
            )

    return results


# ============================================================
# Step 3: Conversation Memory
# ============================================================

def test_conversation_memory():

    print("\n")
    print("=" * 60)
    print("2. CONVERSATION MEMORY - 5 TURNS")
    print("=" * 60)

    history = []

    turns = [
        "My name is Pallavi.",
        "What is my name?",
        "I am working on a LangChain project.",
        "What project am I working on?",
        "What local LLM are we using?"
    ]

    results = []

    prompt = PromptTemplate(
        input_variables=[
            "history",
            "question"
        ],
        template=(
            "You are a helpful conversational assistant.\n\n"

            "Conversation history:\n"
            "{history}\n\n"

            "Current question:\n"
            "{question}\n\n"

            "Answer:"
        )
    )

    llm = create_llm()

    parser = StrOutputParser()

    chain = (
        prompt
        | llm
        | parser
    )

    for index, user_message in enumerate(
        turns,
        start=1
    ):

        print(f"\nTurn {index}")
        print("-" * 60)

        print(
            f"User: {user_message}"
        )

        if history:

            conversation_history = "\n\n".join(
                [
                    f"User: {user}\n"
                    f"Assistant: {assistant}"
                    for user, assistant in history
                ]
            )

        else:

            conversation_history = (
                "No previous conversation."
            )

        try:

            answer = chain.invoke(
                {
                    "history": conversation_history,
                    "question": user_message
                }
            )

            print(
                f"Assistant: {answer}"
            )

            history.append(
                (
                    user_message,
                    answer
                )
            )

            results.append(
                {
                    "turn": index,
                    "user": user_message,
                    "assistant": answer
                }
            )

        except Exception as error:

            print(
                f"Error: {error}"
            )

            results.append(
                {
                    "turn": index,
                    "user": user_message,
                    "assistant": f"ERROR: {error}"
                }
            )

    return results


# ============================================================
# Step 4: Calculator Tool
# ============================================================

@tool
def calculator(expression: str) -> str:
    """
    Calculate a mathematical expression.
    """

    try:

        result = eval(
            expression,
            {
                "__builtins__": {}
            },
            {}
        )

        return str(result)

    except Exception as error:

        return (
            f"Calculation error: {error}"
        )


# ============================================================
# Step 5: Web Search Stub
# ============================================================

@tool
def web_search_stub(query: str) -> str:
    """
    Simulate a web search.
    """

    return (
        f"Web search stub result for "
        f"'{query}': "
        "This is a simulated web search "
        "result for the Week 6 project."
    )


# ============================================================
# Step 6: Agent / Tool Tests
# ============================================================

def test_agent_tools():

    print("\n")
    print("=" * 60)
    print("3. AGENT TEST - 2 TOOLS / 3 TASKS")
    print("=" * 60)

    tasks = [
        (
            "calculator",
            "25 * 4 + 10"
        ),
        (
            "calculator",
            "144 / 12"
        ),
        (
            "web_search_stub",
            "What is LangChain?"
        )
    ]

    results = []

    for index, (
        tool_name,
        task
    ) in enumerate(
        tasks,
        start=1
    ):

        print(f"\nAgent Task {index}")
        print("-" * 60)

        print(
            f"Tool: {tool_name}"
        )

        print(
            f"Input: {task}"
        )

        try:

            if tool_name == "calculator":

                result = calculator.invoke(
                    task
                )

            else:

                result = web_search_stub.invoke(
                    task
                )

            print(
                f"Result: {result}"
            )

            results.append(
                {
                    "task": index,
                    "tool": tool_name,
                    "input": task,
                    "result": result
                }
            )

        except Exception as error:

            print(
                f"Error: {error}"
            )

            results.append(
                {
                    "task": index,
                    "tool": tool_name,
                    "input": task,
                    "result": f"ERROR: {error}"
                }
            )

    return results


# ============================================================
# Step 7: Load Document
# ============================================================

def load_document():

    print("\n")
    print("=" * 60)
    print("4. DOCUMENT LOADING")
    print("=" * 60)

    print(
        f"Document: {DOCUMENT_PATH}"
    )

    if not DOCUMENT_PATH.exists():

        raise FileNotFoundError(
            f"Document not found: "
            f"{DOCUMENT_PATH}"
        )

    loader = TextLoader(
        str(DOCUMENT_PATH),
        encoding="utf-8"
    )

    documents = loader.load()

    print(
        f"Documents loaded: {len(documents)}"
    )

    return documents


# ============================================================
# Step 8: Split Document
# ============================================================

def split_documents(documents):

    print("\n")
    print("=" * 60)
    print("5. DOCUMENT CHUNKING")
    print("=" * 60)

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )

    chunks = splitter.split_documents(
        documents
    )

    print(
        f"Number of chunks: {len(chunks)}"
    )

    for index, chunk in enumerate(
        chunks[:3],
        start=1
    ):

        print(
            f"\nChunk {index}:"
        )

        print(
            chunk.page_content[:200]
        )

    return chunks


# ============================================================
# Step 9: Create ChromaDB Vector Store
# ============================================================

def create_vector_store(chunks):

    print("\n")
    print("=" * 60)
    print("6. CHROMADB VECTOR STORE")
    print("=" * 60)

    embeddings = OllamaEmbeddings(
        model=EMBEDDING_MODEL
    )

    vector_store = Chroma(
        collection_name="w6d5_documents",
        embedding_function=embeddings,
        persist_directory=str(CHROMA_PATH)
    )

    # Add documents to ChromaDB
    vector_store.add_documents(
        chunks
    )

    print(
        "Documents successfully stored in ChromaDB."
    )

    print(
        f"ChromaDB location: {CHROMA_PATH}"
    )

    return vector_store


# ============================================================
# Step 10: Document Question Answering
# ============================================================

def document_chatbot(vector_store):

    print("\n")
    print("=" * 60)
    print("7. DOCUMENT CHATBOT")
    print("=" * 60)

    retriever = vector_store.as_retriever(
        search_kwargs={
            "k": 3
        }
    )

    prompt = PromptTemplate(
        input_variables=[
            "context",
            "question"
        ],
        template=(
            "You are a document question-answering assistant.\n\n"

            "Use ONLY the information provided "
            "in the context below.\n\n"

            "If the answer cannot be found in "
            "the context, say:\n"
            "'I could not find that information "
            "in the document.'\n\n"

            "Context:\n"
            "{context}\n\n"

            "Question:\n"
            "{question}\n\n"

            "Answer:"
        )
    )

    llm = create_llm()

    parser = StrOutputParser()

    chain = (
        prompt
        | llm
        | parser
    )

    questions = [
        "What is LangChain?",
        "What is ChromaDB used for?",
        "What are embeddings?",
        "What does Ollama do?",
        "What is RAG?"
    ]

    results = []

    for index, question in enumerate(
        questions,
        start=1
    ):

        print(f"\nDocument Question {index}")
        print("-" * 60)

        print(
            f"Question: {question}"
        )

        try:

            retrieved_documents = retriever.invoke(
                question
            )

            context = "\n\n".join(
                [
                    document.page_content
                    for document in retrieved_documents
                ]
            )

            answer = chain.invoke(
                {
                    "context": context,
                    "question": question
                }
            )

            print(
                f"Answer: {answer}"
            )

            results.append(
                {
                    "question": question,
                    "answer": answer,
                    "documents_retrieved": len(
                        retrieved_documents
                    )
                }
            )

        except Exception as error:

            print(
                f"Error: {error}"
            )

            results.append(
                {
                    "question": question,
                    "answer": f"ERROR: {error}",
                    "documents_retrieved": 0
                }
            )

    return results


# ============================================================
# Step 11: Save Evidence
# ============================================================

def save_output(
    chain_results,
    memory_results,
    agent_results,
    document_results
):

    output_directory = (
        BASE_DIR / "outputs"
    )

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory
        / "test_output.txt"
    )

    with open(
        output_file,
        "w",
        encoding="utf-8"
    ) as file:

        file.write(
            "W6D5 DOCUMENT CHATBOT TEST OUTPUT\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        # ----------------------------------------------------
        # Chain
        # ----------------------------------------------------

        file.write(
            "1. LANGCHAIN CHAIN - 5 INPUTS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for result in chain_results:

            file.write(
                f"Test {result['test']}\n"
            )

            file.write(
                f"Question: "
                f"{result['question']}\n"
            )

            file.write(
                f"Answer: "
                f"{result['answer']}\n\n"
            )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        file.write(
            "2. CONVERSATION MEMORY - 5 TURNS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for result in memory_results:

            file.write(
                f"Turn {result['turn']}\n"
            )

            file.write(
                f"User: "
                f"{result['user']}\n"
            )

            file.write(
                f"Assistant: "
                f"{result['assistant']}\n\n"
            )

        # ----------------------------------------------------
        # Agent
        # ----------------------------------------------------

        file.write(
            "3. AGENT - 2 TOOLS / 3 TASKS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for result in agent_results:

            file.write(
                f"Task {result['task']}\n"
            )

            file.write(
                f"Tool: "
                f"{result['tool']}\n"
            )

            file.write(
                f"Input: "
                f"{result['input']}\n"
            )

            file.write(
                f"Result: "
                f"{result['result']}\n\n"
            )

        # ----------------------------------------------------
        # Document Chatbot
        # ----------------------------------------------------

        file.write(
            "4. DOCUMENT CHATBOT\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for index, result in enumerate(
            document_results,
            start=1
        ):

            file.write(
                f"Document Question {index}\n"
            )

            file.write(
                f"Question: "
                f"{result['question']}\n"
            )

            file.write(
                f"Documents Retrieved: "
                f"{result['documents_retrieved']}\n"
            )

            file.write(
                f"Answer: "
                f"{result['answer']}\n\n"
            )

    print("\n")
    print("=" * 60)
    print("EVIDENCE SAVED")
    print("=" * 60)

    print(
        f"Output file: {output_file}"
    )


# ============================================================
# Main
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("W6D5 - DOCUMENT CHATBOT WITH LANGCHAIN")
    print("=" * 60)

    # Step 1
    chain_results = test_chain()

    # Step 2
    memory_results = test_conversation_memory()

    # Step 3
    agent_results = test_agent_tools()

    # Step 4
    documents = load_document()

    # Step 5
    chunks = split_documents(
        documents
    )

    # Step 6
    vector_store = create_vector_store(
        chunks
    )

    # Step 7
    document_results = document_chatbot(
        vector_store
    )

    # Save evidence
    save_output(
        chain_results,
        memory_results,
        agent_results,
        document_results
    )

    print("\n")
    print("=" * 60)
    print("W6D5 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nCompleted:")
    print("✓ LangChain chain")
    print("✓ 5 chain inputs")
    print("✓ 5 conversation turns")
    print("✓ Calculator tool")
    print("✓ Web-search stub")
    print("✓ 3 agent tasks")
    print("✓ Document loading")
    print("✓ Document splitting")
    print("✓ Ollama embeddings")
    print("✓ ChromaDB vector store")
    print("✓ Document retrieval")
    print("✓ Document question answering")
    print("✓ Output evidence")


# ============================================================
# Entry Point
# ============================================================

if __name__ == "__main__":
    main()