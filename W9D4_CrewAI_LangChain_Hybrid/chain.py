from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# Prompt
prompt = PromptTemplate(
    input_variables=["question"],
    template="""
You are a helpful AI assistant.

Answer the following question in simple and clear words.

Question: {question}

Answer:
"""
)

# Local Ollama model
llm = OllamaLLM(
    model="llama3.2:3b"
)

# Output parser
parser = StrOutputParser()

# LangChain chain
chain = prompt | llm | parser


def ask_question(question):
    """Send a question through the LangChain chain."""
    return chain.invoke({"question": question})


if __name__ == "__main__":
    print("W9D4 LangChain Chain Test")
    print("=" * 40)

    questions = [
        "What is artificial intelligence?",
        "What is machine learning?",
        "What is a digital twin?",
        "What is LangChain?",
        "What is CrewAI?"
    ]

    for i, question in enumerate(questions, 1):
        print(f"\nTest {i}")
        print("Question:", question)

        answer = ask_question(question)

        print("Answer:", answer)