"""
Local AI Research Assistant.

This module implements a simple local research workflow using
LangGraph and Ollama.
"""

from pathlib import Path
from typing import TypedDict

from langchain_ollama import ChatOllama
from langgraph.graph import END, START, StateGraph


BASE_DIR = Path(__file__).resolve().parent.parent
DATA_FILE = BASE_DIR / "data" / "research_notes.txt"


class ResearchState(TypedDict, total=False):
    question: str
    context: str
    answer: str


class LocalResearchAssistant:
    """Simple local AI research assistant."""

    def __init__(self, model_name: str = "llama3.2:3b"):
        self.model = ChatOllama(
            model=model_name,
            temperature=0
        )

        self.research_text = self._load_research()

        workflow = StateGraph(ResearchState)

        workflow.add_node("retrieve", self.retrieve)
        workflow.add_node("generate", self.generate)

        workflow.add_edge(START, "retrieve")
        workflow.add_edge("retrieve", "generate")
        workflow.add_edge("generate", END)

        self.graph = workflow.compile()

    def _load_research(self) -> str:
        """Load local research notes."""
        if not DATA_FILE.exists():
            raise FileNotFoundError(
                f"Research file not found: {DATA_FILE}"
            )

        return DATA_FILE.read_text(
            encoding="utf-8"
        )

    def retrieve(self, state: ResearchState) -> ResearchState:
        """Retrieve relevant text using simple keyword matching."""
        question = state["question"].lower()

        paragraphs = self.research_text.split("\n\n")

        matching = []

        for paragraph in paragraphs:
            words = question.split()

            score = sum(
                1
                for word in words
                if len(word) > 3
                and word in paragraph.lower()
            )

            if score > 0:
                matching.append((score, paragraph))

        matching.sort(
            key=lambda item: item[0],
            reverse=True
        )

        context = "\n\n".join(
            paragraph
            for _, paragraph in matching[:3]
        )

        if not context:
            context = self.research_text

        return {
            **state,
            "context": context
        }

    def generate(self, state: ResearchState) -> ResearchState:
        """Generate an answer using the local Ollama model."""
        prompt = f"""
You are a local AI research assistant.

Answer the user's question using only the research context below.

Research context:
{state["context"]}

Question:
{state["question"]}

Give a clear and concise answer.
"""

        response = self.model.invoke(prompt)

        return {
            **state,
            "answer": response.content
        }

    def ask(self, question: str) -> str:
        """Answer a research question."""
        if not question.strip():
            raise ValueError("Question cannot be empty.")

        result = self.graph.invoke(
            {
                "question": question
            }
        )

        return result["answer"]


if __name__ == "__main__":
    assistant = LocalResearchAssistant()

    question = "What is a digital twin?"

    print("Question:", question)
    print("\nAnswer:")
    print(assistant.ask(question))