from pathlib import Path

from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_core.tools import tool
from langchain_ollama import OllamaLLM


# ============================================================
# Configuration
# ============================================================

BASE_DIR = Path(__file__).resolve().parent

MODEL_NAME = "llama3.2:3b"


# ============================================================
# Create Ollama LLM
# ============================================================

def create_llm():
    """
    Create and return the Ollama language model.
    """

    return OllamaLLM(
        model=MODEL_NAME
    )


# ============================================================
# Step 1: Basic LangChain Chain
# PromptTemplate → Ollama → OutputParser
# ============================================================

def build_chain():
    """
    Build a simple LangChain chain.
    """

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

    output_parser = StrOutputParser()

    chain = prompt | llm | output_parser

    return chain


# ============================================================
# Step 2: Test Chain with 5 Inputs
# ============================================================

def test_chain():
    """
    Test the LangChain chain with 5 questions.
    """

    print("\n")
    print("=" * 60)
    print("CHAIN TEST: 5 INPUTS")
    print("=" * 60)

    chain = build_chain()

    questions = [
        "What is LangChain?",
        "What is Ollama?",
        "What is ChromaDB?",
        "What is a vector database?",
        "What are embeddings?"
    ]

    results = []

    for index, question in enumerate(questions, start=1):

        print(f"\nTest {index}")
        print("-" * 60)

        print(f"Question: {question}")

        try:

            answer = chain.invoke(
                {
                    "question": question
                }
            )

            print(f"Answer: {answer}")

            results.append(
                {
                    "test": index,
                    "question": question,
                    "answer": answer
                }
            )

        except Exception as error:

            print(f"Error: {error}")

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
    """
    Demonstrate conversation history across 5 turns.
    """

    print("\n")
    print("=" * 60)
    print("CONVERSATION MEMORY TEST: 5 TURNS")
    print("=" * 60)

    history = []

    turns = [
        "My name is Pallavi.",
        "What is my name?",
        "I am working on a LangChain project.",
        "What project am I working on?",
        "What local LLM are we using in this project?"
    ]

    results = []

    llm = create_llm()

    prompt = PromptTemplate(
        input_variables=[
            "history",
            "question"
        ],
        template=(
            "You are a helpful conversational AI assistant.\n\n"

            "Here is the conversation history:\n"
            "{history}\n\n"

            "Answer the current user message using "
            "the conversation history when necessary.\n\n"

            "Current user message:\n"
            "{question}\n\n"

            "Answer:"
        )
    )

    memory_chain = (
        prompt
        | llm
        | StrOutputParser()
    )

    for index, user_message in enumerate(turns, start=1):

        print(f"\nTurn {index}")
        print("-" * 60)

        print(f"User: {user_message}")

        if history:

            conversation_history = "\n\n".join(
                [
                    f"User: {user}\nAssistant: {assistant}"
                    for user, assistant in history
                ]
            )

        else:

            conversation_history = "No previous conversation."

        try:

            answer = memory_chain.invoke(
                {
                    "history": conversation_history,
                    "question": user_message
                }
            )

            print(f"Assistant: {answer}")

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

            print(f"Error: {error}")

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

    Example:
        25 * 4 + 10
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

        return f"Calculation error: {error}"


# ============================================================
# Step 5: Web Search Stub Tool
# ============================================================

@tool
def web_search_stub(query: str) -> str:
    """
    Simulate a web search.

    This is a stub and does not access the internet.
    """

    return (
        f"Web search stub result for '{query}': "
        "LangChain is a framework for building "
        "applications powered by language models."
    )


# ============================================================
# Step 6: Test Agent Tools with 3 Tasks
# ============================================================

def test_agent_tools():
    """
    Test the two tools with three tasks.
    """

    print("\n")
    print("=" * 60)
    print("AGENT TEST: 2 TOOLS / 3 TASKS")
    print("=" * 60)

    tasks = [
        {
            "tool": "calculator",
            "input": "25 * 4 + 10"
        },
        {
            "tool": "calculator",
            "input": "144 / 12"
        },
        {
            "tool": "web_search_stub",
            "input": "What is LangChain?"
        }
    ]

    results = []

    for index, task in enumerate(tasks, start=1):

        print(f"\nAgent Task {index}")
        print("-" * 60)

        print(f"Tool: {task['tool']}")
        print(f"Input: {task['input']}")

        try:

            if task["tool"] == "calculator":

                result = calculator.invoke(
                    task["input"]
                )

            elif task["tool"] == "web_search_stub":

                result = web_search_stub.invoke(
                    task["input"]
                )

            else:

                result = "Unknown tool."

            print(f"Result: {result}")

            results.append(
                {
                    "task": index,
                    "tool": task["tool"],
                    "input": task["input"],
                    "result": result
                }
            )

        except Exception as error:

            print(f"Error: {error}")

            results.append(
                {
                    "task": index,
                    "tool": task["tool"],
                    "input": task["input"],
                    "result": f"ERROR: {error}"
                }
            )

    return results


# ============================================================
# Step 7: Save Evidence
# ============================================================

def save_output(
    chain_results,
    memory_results,
    agent_results
):
    """
    Save all Step 10 and Step 11 results.
    """

    output_directory = BASE_DIR / "outputs"

    output_directory.mkdir(
        parents=True,
        exist_ok=True
    )

    output_file = (
        output_directory / "test_output.txt"
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
            "1. LANGCHAIN CHAIN TESTS - 5 INPUTS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for result in chain_results:

            file.write(
                f"Test {result['test']}\n"
            )

            file.write(
                f"Question: {result['question']}\n"
            )

            file.write(
                f"Answer: {result['answer']}\n\n"
            )

        # ----------------------------------------------------
        # Memory
        # ----------------------------------------------------

        file.write(
            "2. CONVERSATION MEMORY TESTS - 5 TURNS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for result in memory_results:

            file.write(
                f"Turn {result['turn']}\n"
            )

            file.write(
                f"User: {result['user']}\n"
            )

            file.write(
                f"Assistant: {result['assistant']}\n\n"
            )

        # ----------------------------------------------------
        # Agent
        # ----------------------------------------------------

        file.write(
            "3. AGENT TESTS - 2 TOOLS / 3 TASKS\n"
        )

        file.write(
            "=" * 60 + "\n\n"
        )

        for result in agent_results:

            file.write(
                f"Task {result['task']}\n"
            )

            file.write(
                f"Tool: {result['tool']}\n"
            )

            file.write(
                f"Input: {result['input']}\n"
            )

            file.write(
                f"Result: {result['result']}\n\n"
            )

    print("\n")
    print("=" * 60)
    print("OUTPUT SAVED")
    print("=" * 60)

    print(
        f"Evidence file: {output_file}"
    )


# ============================================================
# Main Program
# ============================================================

def main():

    print("\n")
    print("=" * 60)
    print("W6D5 - LANGCHAIN DOCUMENT CHATBOT")
    print("STEP 11: AGENT WITH 2 TOOLS")
    print("=" * 60)

    # Step 10 - Chain
    chain_results = test_chain()

    # Step 10 - Conversation Memory
    memory_results = test_conversation_memory()

    # Step 11 - Agent Tools
    agent_results = test_agent_tools()

    # Save all evidence
    save_output(
        chain_results,
        memory_results,
        agent_results
    )

    print("\n")
    print("=" * 60)
    print("STEP 11 COMPLETED SUCCESSFULLY")
    print("=" * 60)

    print("\nCompleted:")
    print("✓ PromptTemplate")
    print("✓ Ollama LLM")
    print("✓ StrOutputParser")
    print("✓ Chain tested with 5 inputs")
    print("✓ Conversation history tested with 5 turns")
    print("✓ Calculator tool")
    print("✓ Web-search stub tool")
    print("✓ 3 agent/tool tasks")
    print("✓ Evidence saved")


# ============================================================
# Program Entry Point
# ============================================================

if __name__ == "__main__":
    main()