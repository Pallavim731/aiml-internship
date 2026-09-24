"""
CrewAI agents for the Stateful Customer Support Agent.
"""

from crewai import Agent, LLM


# Local Ollama LLM
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434",
)


def create_support_agent():
    """Create the main customer support agent."""

    return Agent(
        role="Customer Support Specialist",
        goal=(
            "Understand the customer's issue and provide a helpful, "
            "clear and professional response."
        ),
        backstory=(
            "You are an experienced customer support specialist. "
            "You carefully use the conversation history so that "
            "customers do not need to repeat information."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )


def create_reviewer_agent():
    """Create an agent that reviews the support response."""

    return Agent(
        role="Customer Support Reviewer",
        goal=(
            "Review customer support responses for accuracy, clarity, "
            "professionalism and whether the customer's issue was addressed."
        ),
        backstory=(
            "You are a quality reviewer for a customer support team. "
            "You check that responses are useful and consistent with "
            "the conversation history."
        ),
        llm=llm,
        verbose=True,
        allow_delegation=False,
    )