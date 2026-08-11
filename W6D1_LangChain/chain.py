from langchain_core.prompts import PromptTemplate
from langchain_ollama import OllamaLLM
from langchain_core.output_parsers import StrOutputParser


# 1. Create prompt
prompt = PromptTemplate(
    input_variables=["topic"],
    template="Explain {topic} in simple terms for a beginner."
)

# 2. Create Ollama LLM
llm = OllamaLLM(model="llama3.2:3b")

# 3. Create output parser
parser = StrOutputParser()

# 4. Build LangChain chain
chain = prompt | llm | parser


# 5. Test with 5 inputs
topics = [
    "Artificial Intelligence",
    "Machine Learning",
    "LangChain",
    "Vector Database",
    "Generative AI"
]

with open("outputs_chain.txt", "w", encoding="utf-8") as file:

    for topic in topics:

        print("\n" + "=" * 60)
        print("INPUT:", topic)

        result = chain.invoke({"topic": topic})

        print("OUTPUT:", result)

        file.write("=" * 60 + "\n")
        file.write(f"INPUT: {topic}\n")
        file.write(f"OUTPUT: {result}\n\n")