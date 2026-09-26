from crewai import Agent, Crew, Task, LLM


# Local Ollama LLM
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
)


def generate_answer(question, contexts):
    """Generate an answer using retrieved RAG context."""

    context_text = "\n\n".join(contexts)

    # Create the RAG question-answering agent
    rag_agent = Agent(
        role="RAG Question Answering Agent",
        goal="Answer questions using only the information provided in the retrieved context.",
        backstory=(
            "You are a careful AI assistant. "
            "You answer questions using the supplied knowledge context "
            "and do not invent information."
        ),
        llm=llm,
        verbose=False,
    )

    # Define the task
    answer_task = Task(
        description=f"""
Answer the following question using only the provided context.

Question:
{question}

Retrieved Context:
{context_text}

Instructions:
- Give a clear and concise answer.
- Use only information supported by the context.
- If the context does not contain enough information, say so.
""",
        expected_output="A clear answer supported by the retrieved context.",
        agent=rag_agent,
    )

    # Create and run the Crew
    crew = Crew(
        agents=[rag_agent],
        tasks=[answer_task],
        verbose=False,
    )

    result = crew.kickoff()

    return str(result)


if __name__ == "__main__":
    question = "What is RAG?"

    contexts = [
        "RAG combines information retrieval with text generation. "
        "A RAG system retrieves relevant information from a knowledge "
        "base and provides it to a language model to generate an answer."
    ]

    answer = generate_answer(question, contexts)

    print("\nQuestion:")
    print(question)

    print("\nAnswer:")
    print(answer)