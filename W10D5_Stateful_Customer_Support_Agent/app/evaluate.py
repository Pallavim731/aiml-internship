"""
Ragas evaluation for the Stateful Customer Support Agent.
"""

from datasets import Dataset
from langchain_ollama import ChatOllama, OllamaEmbeddings

from ragas import evaluate
from ragas.metrics import faithfulness, answer_relevancy
from ragas.llms import LangchainLLMWrapper
from ragas.embeddings import LangchainEmbeddingsWrapper


# Local Ollama LLM used by Ragas.
ollama_model = ChatOllama(
    model="llama3.2:3b",
    base_url="http://localhost:11434",
)

ragas_llm = LangchainLLMWrapper(ollama_model)


# Local Ollama embedding model.
ollama_embeddings = OllamaEmbeddings(
    model="nomic-embed-text",
    base_url="http://localhost:11434",
)

ragas_embeddings = LangchainEmbeddingsWrapper(
    ollama_embeddings
)


def evaluate_response(question, answer, context):
    """Evaluate a customer-support response using Ragas."""

    data = {
        "question": [question],
        "answer": [answer],
        "contexts": [[context]],
    }

    dataset = Dataset.from_dict(data)

    result = evaluate(
        dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings,
    )

    return result


if __name__ == "__main__":
    sample_question = "Where is my order?"

    sample_answer = (
        "I can help you check your order. "
        "Please provide your order number."
    )

    sample_context = (
        "The customer wants help checking the status of an order."
    )

    print("Running Ragas evaluation...")

    result = evaluate_response(
        sample_question,
        sample_answer,
        sample_context,
    )

    print("\nEvaluation result:")
    print(result)