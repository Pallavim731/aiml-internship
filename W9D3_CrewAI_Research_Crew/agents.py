from crewai import Agent, LLM
from tools import search_tool


def create_agents():

    local_llm = LLM(
        model="ollama/llama3.2:3b",
        base_url="http://localhost:11434",
        temperature=0.2
    )

    researcher = Agent(
        role="Research Specialist",
        goal=(
            "Research the assigned topic and collect accurate, "
            "relevant information from reliable web sources."
        ),
        backstory=(
            "You are an experienced research specialist who "
            "collects useful information, identifies important facts "
            "and organizes research findings clearly."
        ),
        tools=[search_tool],
        llm=local_llm,
        verbose=True
    )

    writer = Agent(
        role="Technical Writer",
        goal=(
            "Create a clear, professional and well-structured "
            "research report using the research findings."
        ),
        backstory=(
            "You are a skilled technical writer who converts "
            "complex research information into simple and readable reports."
        ),
        llm=local_llm,
        verbose=True
    )

    reviewer = Agent(
        role="Research Reviewer",
        goal=(
            "Review the research report for accuracy, completeness, "
            "clarity and logical consistency."
        ),
        backstory=(
            "You are a careful reviewer who checks research reports "
            "and identifies weaknesses and possible improvements."
        ),
        llm=local_llm,
        verbose=True
    )

    return researcher, writer, reviewer