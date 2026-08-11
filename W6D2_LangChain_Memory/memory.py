from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


# Ollama local model
llm = OllamaLLM(model="llama3.2:3b")


# Prompt with conversation history
prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful AI assistant. "
        "Use the previous conversation history to answer questions."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])


# Chain
chain = prompt | llm


# Conversation history
history = []


# Five conversation turns
conversations = [
    "My name is Pallavi.",
    "What is my name?",
    "I am studying Information Science and Engineering.",
    "What am I studying?",
    "Can you summarize what you remember from our conversation?"
]


with open("outputs/memory_output.txt", "w", encoding="utf-8") as file:

    for turn, user_input in enumerate(conversations, start=1):

        print("\n" + "=" * 60)
        print(f"TURN {turn}")
        print("USER:", user_input)

        response = chain.invoke({
            "history": history,
            "input": user_input
        })

        print("AI:", response)

        file.write("=" * 60 + "\n")
        file.write(f"TURN {turn}\n")
        file.write(f"USER: {user_input}\n")
        file.write(f"AI: {response}\n\n")

        # Save conversation history
        history.append(
            HumanMessage(content=user_input)
        )

        history.append(
            AIMessage(content=response)
        )