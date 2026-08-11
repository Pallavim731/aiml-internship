from langchain_ollama import OllamaLLM
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import HumanMessage, AIMessage


llm = OllamaLLM(model="llama3.2:3b")

prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        "You are a helpful assistant. Use the conversation history "
        "to answer questions consistently."
    ),
    MessagesPlaceholder(variable_name="history"),
    ("human", "{input}")
])

chain = prompt | llm

history = []

conversations = [
    "My name is Pallavi.",
    "What is my name?",
    "I am learning Artificial Intelligence.",
    "What am I learning?",
    "Can you summarize what you know about me from this conversation?"
]

with open("outputs/memory_output.txt", "w", encoding="utf-8") as file:

    for i, user_input in enumerate(conversations, start=1):

        print("\n" + "=" * 60)
        print(f"TURN {i}")
        print("USER:", user_input)

        response = chain.invoke({
            "history": history,
            "input": user_input
        })

        print("AI:", response)

        file.write("=" * 60 + "\n")
        file.write(f"TURN {i}\n")
        file.write(f"USER: {user_input}\n")
        file.write(f"AI: {response}\n\n")

        history.append(HumanMessage(content=user_input))
        history.append(AIMessage(content=response))