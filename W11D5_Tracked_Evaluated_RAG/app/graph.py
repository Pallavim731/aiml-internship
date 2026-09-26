from typing import TypedDict

from langgraph.graph import StateGraph, START, END

from app.rag_pipeline import retrieve_context
from app.crew_agent import generate_answer


class RAGState(TypedDict):
    question: str
    contexts: list[str]
    answer: str


def retrieve_node(state: RAGState):
    """Retrieve relevant documents for the question."""

    contexts = retrieve_context(
        question=state["question"],
        k=3,
    )

    return {
        "contexts": contexts
    }


def generate_node(state: RAGState):
    """Generate an answer using the retrieved context."""

    answer = generate_answer(
        question=state["question"],
        contexts=state["contexts"],
    )

    return {
        "answer": answer
    }


def build_graph():
    """Build the LangGraph RAG workflow."""

    graph = StateGraph(RAGState)

    # Add processing nodes
    graph.add_node("retrieve", retrieve_node)
    graph.add_node("generate", generate_node)

    # Define workflow
    graph.add_edge(START, "retrieve")
    graph.add_edge("retrieve", "generate")
    graph.add_edge("generate", END)

    return graph.compile()


def run_rag(question: str):
    """Run the complete RAG workflow."""

    rag_graph = build_graph()

    result = rag_graph.invoke(
        {
            "question": question,
            "contexts": [],
            "answer": "",
        }
    )

    return result


if __name__ == "__main__":
    question = "What is RAG?"

    result = run_rag(question)

    print("\nQuestion:")
    print(result["question"])

    print("\nRetrieved Contexts:")
    for index, context in enumerate(result["contexts"], start=1):
        print(f"\n--- Context {index} ---")
        print(context)

    print("\nFinal Answer:")
    print(result["answer"])