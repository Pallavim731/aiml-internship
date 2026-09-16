from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from langchain_ollama import ChatOllama


# -------------------------------
# MODEL
# -------------------------------

llm = ChatOllama(
    model="llama3.2:3b",
    temperature=0
)


# -------------------------------
# TASK 1: LANGCHAIN CHAIN
# PromptTemplate → Ollama → Parser
# -------------------------------

prompt = PromptTemplate(
    input_variables=["topic"],
    template="""
You are a helpful AI/ML mentor.

Explain the following topic in simple terms:
{topic}

Give a short and clear explanation with one example.
"""
)

parser = StrOutputParser()

chain = prompt | llm | parser


print("\n" + "=" * 60)
print("TASK 1: LANGCHAIN CHAIN")
print("=" * 60)


inputs = [
    "Machine Learning",
    "Large Language Models",
    "Vector Databases",
    "Prompt Engineering",
    "AI Agents"
]

for i, topic in enumerate(inputs, start=1):
    result = chain.invoke({"topic": topic})

    print(f"\nInput {i}: {topic}")
    print("-" * 40)
    print(result)
    from langchain_classic.memory import ConversationBufferMemory


# -------------------------------
# TASK 2: CONVERSATION MEMORY
# -------------------------------

memory = ConversationBufferMemory(
    return_messages=True
)

print("\n" + "=" * 60)
print("TASK 2: CONVERSATION BUFFER MEMORY")
print("=" * 60)


conversation_turns = [
    ("Hi, my name is Pallavi.", "Hello Pallavi! Nice to meet you."),
    ("I am learning AI and Machine Learning.", "That's great! AI and Machine Learning are valuable skills."),
    ("I am currently learning LangChain.", "LangChain is useful for building LLM applications."),
    ("I also use Ollama for local models.", "Ollama is useful for running LLMs locally."),
    ("What technologies am I learning?", "You are learning AI/ML, LangChain and Ollama.")
]


for turn, (user_message, assistant_message) in enumerate(
    conversation_turns, start=1
):

    memory.chat_memory.add_user_message(user_message)
    memory.chat_memory.add_ai_message(assistant_message)

    print(f"\nTurn {turn}")
    print("User:", user_message)
    print("Assistant:", assistant_message)

    print("\nCurrent conversation history:")

    for message in memory.chat_memory.messages:
        print(
            f"{message.type.upper()}: {message.content}"
        )


print("\n" + "-" * 60)
print("FINAL MEMORY CHECK")
print("-" * 60)

print(
    f"Total messages stored: "
    f"{len(memory.chat_memory.messages)}"
)
from langchain.tools import tool


# -------------------------------
# TASK 3: AGENT TOOLS
# -------------------------------

@tool
def web_search_stub(query: str) -> str:
    """
    Simulates a web search and returns mock search results.
    This is a stub and does not access the real internet.
    """

    results = {
        "LangChain": "LangChain is a framework for developing applications powered by language models.",
        "Ollama": "Ollama allows developers to run LLMs locally.",
        "AI agents": "AI agents use language models together with tools to perform tasks."
    }

    for key, value in results.items():
        if key.lower() in query.lower():
            return value

    return f"Stub search result for '{query}': No specific result found."


@tool
def calculator(expression: str) -> str:
    """
    Performs basic arithmetic calculations.
    Example: 25 * 4
    """

    try:
        allowed_characters = "0123456789+-*/(). "

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Invalid mathematical expression."

        result = eval(expression, {"__builtins__": {}}, {})

        return f"Result: {result}"

    except Exception as e:
        return f"Calculation error: {e}"

    from langchain.tools import tool


# -------------------------------
# TASK 3: AGENT TOOLS
# -------------------------------

@tool
def web_search_stub(query: str) -> str:
    """
    Simulates a web search and returns mock search results.
    This is a stub and does not access the real internet.
    """

    results = {
        "LangChain": "LangChain is a framework for developing applications powered by language models.",
        "Ollama": "Ollama allows developers to run LLMs locally.",
        "AI agents": "AI agents use language models together with tools to perform tasks."
    }

    for key, value in results.items():
        if key.lower() in query.lower():
            return value

    return f"Stub search result for '{query}': No specific result found."


@tool
def calculator(expression: str) -> str:
    """
    Performs basic arithmetic calculations.
    Example: 25 * 4
    """

    try:
        allowed_characters = "0123456789+-*/(). "

        if not all(
            character in allowed_characters
            for character in expression
        ):
            return "Invalid mathematical expression."

        result = eval(expression, {"__builtins__": {}}, {})

        return f"Result: {result}"

    except Exception as e:
        return f"Calculation error: {e}"
        # -------------------------------
# RUN 3 AGENT TASKS
# -------------------------------

agent_tasks = [
    "Calculate 125 * 8 + 50.",
    "Search for information about Ollama.",
    "Calculate 450 / 9 and explain what Ollama is."
]


print("\n" + "=" * 60)
print("TASK 3: LANGCHAIN AGENT")
print("=" * 60)


for i, task in enumerate(agent_tasks, start=1):

    print(f"\nAgent Task {i}")
    print("-" * 40)
    print("User:", task)

    result = agent.invoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": task
                }
            ]
        }
    )

    final_message = result["messages"][-1]

    print("Agent:", final_message.content)