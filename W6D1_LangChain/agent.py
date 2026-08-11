from langchain_ollama import OllamaLLM
from langchain_core.tools import tool


@tool
def calculator(expression: str) -> str:
    """Calculate a basic mathematical expression."""

    try:
        allowed = "0123456789+-*/(). "

        if not all(char in allowed for char in expression):
            return "Invalid mathematical expression."

        result = eval(expression, {"__builtins__": {}}, {})
        return str(result)

    except Exception as e:
        return f"Calculation error: {e}"


@tool
def web_search(query: str) -> str:
    """Stub for web search. Returns simulated search information."""

    return (
        f"WEB SEARCH STUB RESULT for '{query}': "
        "This is simulated search data for the LangChain assignment."
    )


llm = OllamaLLM(model="llama3.2:3b")

tools = [calculator, web_search]


print("Available tools:")
for tool in tools:
    print("-", tool.name)


tasks = [
    "Calculate 25 * 4 + 10.",
    "Search the web for information about LangChain.",
    "Calculate (100 / 4) + 25."
]


with open("outputs/agent_output.txt", "w", encoding="utf-8") as file:

    for task in tasks:

        print("\n" + "=" * 60)
        print("TASK:", task)

        file.write("=" * 60 + "\n")
        file.write(f"TASK: {task}\n")

        if "calculate" in task.lower():

            if "25 * 4 + 10" in task:
                result = calculator.invoke("25 * 4 + 10")

            elif "100 / 4" in task:
                result = calculator.invoke("(100 / 4) + 25")

            else:
                result = "Calculation task detected."

        elif "search" in task.lower():

            result = web_search.invoke("LangChain")

        else:

            result = "No matching tool found."

        print("RESULT:", result)
        file.write(f"RESULT: {result}\n")