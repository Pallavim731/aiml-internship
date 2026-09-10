from crewai import Agent, LLM
from tools import search_tool, code_tool


def create_agents():

    # Local Ollama LLM
    local_llm = LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.2
    )

    # Researcher
    researcher = Agent(
        role="Research Specialist",
        goal=(
            "Research the assigned topic and identify accurate, "
            "relevant information using web search and code execution."
        ),
        backstory=(
            "You are an experienced research specialist. "
            "You collect useful information from reliable web sources, "
            "identify important facts, perform calculations when needed, "
            "and organize findings clearly."
        ),
        tools=[search_tool, code_tool],
        llm=local_llm,
        verbose=True
    )

    # Writer
    writer = Agent(
        role="Technical Writer",
        goal=(
            "Create a clear and well-structured report using "
            "the research findings and calculations."
        ),
        backstory=(
            "You are a skilled technical writer who converts complex "
            "research information into simple, professional and readable reports."
        ),
        llm=local_llm,
        verbose=True
    )

    # Reviewer
    reviewer = Agent(
        role="Research Reviewer",
        goal=(
            "Review the report for accuracy, completeness, "
            "clarity, organization and logical consistency."
        ),
        backstory=(
            "You are a careful research reviewer. "
            "You identify missing information, logical problems "
            "and unclear explanations and provide useful improvements."
        ),
        llm=local_llm,
        verbose=True
    )

    return researcher, writer, reviewer