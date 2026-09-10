from langchain_core.tools import tool
from langchain_ollama import OllamaLLM


# Tool 1: Web Search Stub
@tool
def web_search(query: str) -> str:
    """Search the web for information. This is a simple demo stub."""
    
    return (
        f"Web search result for '{query}': "
        "Artificial Intelligence is a field of computer science that "
        "creates systems capable of performing tasks that normally require "
        "human intelligence."
    )


# Tool 2: Calculator
@tool
def calculator(expression: str) -> str:
    """Calculate a mathematical expression."""
    
    try:
        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)
    except Exception:
        return "Invalid mathematical expression."


# List of two tools
tools = [web_search, calculator]


# Local Ollama model
llm = OllamaLLM(
    model="llama3.2:3b"
)


def run_agent(task):
    """
    Simple two-tool agent.
    It selects a tool based on the task.
    """

    task_lower = task.lower()

    # Calculator tasks
    if any(word in task_lower for word in [
        "calculate",
        "plus",
        "minus",
        "multiply",
        "divide",
        "times"
    ]):
        expression = (
            task_lower
            .replace("calculate", "")
            .replace("what is", "")
            .strip()
        )

        return "Calculator result: " + calculator.invoke(expression)

    # Web search tasks
    elif any(word in task_lower for word in [
        "search",
        "web",
        "find",
        "information",
        "research"
    ]):
        return "Web Search result: " + web_search.invoke(task)

    # General question
    else:
        return llm.invoke(
            f"Answer this question simply:\n{task}"
        )


if __name__ == "__main__":

    print("W9D4 Two-Tool Agent Test")
    print("=" * 45)

    tasks = [
        "Search the web for information about artificial intelligence.",
        "Calculate 25 * 4.",
        "Find information about machine learning."
    ]

    for i, task in enumerate(tasks, 1):

        print(f"\nTask {i}")
        print("User:", task)

        result = run_agent(task)

        print("Agent:", result)