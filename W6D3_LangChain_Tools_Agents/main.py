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