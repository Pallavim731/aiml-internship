from langchain_classic.memory import ConversationBufferMemory
from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# Create conversation memory
memory = ConversationBufferMemory(
    memory_key="history",
    return_messages=False
)

# Prompt includes previous conversation
prompt = PromptTemplate(
    input_variables=["history", "question"],
    template="""
You are a helpful AI assistant.

Previous conversation:
{history}

Current question:
{question}

Answer in simple words:
"""
)

# Local Ollama model
llm = OllamaLLM(
    model="llama3.2:3b"
)

parser = StrOutputParser()

chain = prompt | llm | parser


def chat(question):
    # Get previous conversation
    history = memory.load_memory_variables({})["history"]

    # Generate answer
    answer = chain.invoke({
        "history": history,
        "question": question
    })

    # Save this turn
    memory.save_context(
        {"input": question},
        {"output": answer}
    )

    return answer


if __name__ == "__main__":
    print("W9D4 ConversationBufferMemory Test")
    print("=" * 45)

    questions = [
        "My name is Pallavi.",
        "What is my name?",
        "I am learning artificial intelligence.",
        "What am I learning?",
        "Can you remember my name and what I am learning?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nTurn {i}")
        print("User:", question)

        answer = chat(question)

        print("AI:", answer)

    print("\n" + "=" * 45)
    print("Conversation history:")
    print(memory.load_memory_variables({})["history"])