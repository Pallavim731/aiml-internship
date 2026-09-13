from crewai import Agent, Crew, Task, Process
from crewai_tools import SerperDevTool


OLLAMA_MODEL = "ollama/llama3.2:3b"

search_tool = SerperDevTool()


# Local Ollama LLM
OLLAMA_MODEL = "ollama/llama3.2:3b"


def create_agents():

    researcher = Agent(
        role="Researcher",
        goal=(
            "Research the given topic carefully and identify accurate, "
            "relevant, useful and well-organized information."
        ),
        backstory=(
            "You are an experienced research analyst. "
            "You investigate technical topics, identify important facts, "
            "compare different viewpoints and organize findings clearly."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    writer = Agent(
        role="Technical Writer",
        goal=(
            "Transform research findings into a clear, structured and "
            "easy-to-understand technical report."
        ),
        backstory=(
            "You are a professional technical writer who specializes "
            "in converting complex technical information into concise, "
            "readable and well-structured reports."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    reviewer = Agent(
        role="Reviewer",
        goal=(
            "Review the report for accuracy, clarity, completeness, "
            "logical consistency and unsupported claims."
        ),
        backstory=(
            "You are a meticulous senior reviewer. You check technical "
            "reports for factual problems, missing information, poor "
            "structure, repetition and unsupported claims."
        ),
        llm=OLLAMA_MODEL,
        verbose=True,
        allow_delegation=False,
    )

    return researcher, writer, reviewer


def create_tasks(researcher, writer, reviewer):

    research_task = Task(
        description=(
            "Research the following topic: {topic}. "
            "Identify the main concepts, important developments, "
            "advantages, limitations, real-world applications and "
            "future trends. Focus on factual and useful information."
        ),
        expected_output=(
            "A structured research summary containing key facts, "
            "important concepts, applications, advantages, limitations "
            "and future trends related to the topic."
        ),
        agent=researcher,
    )

    writing_task = Task(
        description=(
            "Using the research findings provided by the Researcher, "
            "write a well-structured technical report about {topic}. "
            "Use clear headings and concise paragraphs. "
            "Include important concepts, applications, advantages, "
            "limitations and future trends."
        ),
        expected_output=(
            "A clear and well-organized technical report about the topic "
            "with headings, explanations, examples and a conclusion."
        ),
        agent=writer,
        context=[research_task],
        markdown=True,
    )

    review_task = Task(
        description=(
            "Review the technical report produced about {topic}. "
            "Check accuracy, clarity, completeness, logical flow, "
            "unsupported claims and unnecessary repetition. "
            "Improve weak sections and produce a polished final report."
        ),
        expected_output=(
            "A reviewed and improved final report that is accurate, "
            "well-structured, complete, readable and suitable for submission."
        ),
        agent=reviewer,
        context=[writing_task],
        markdown=True,
    )

    return research_task, writing_task, review_task


def create_crew():

    researcher, writer, reviewer = create_agents()

    research_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer
    )

    crew = Crew(
        agents=[
            researcher,
            writer,
            reviewer
        ],
        tasks=[
            research_task,
            writing_task,
            review_task
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew