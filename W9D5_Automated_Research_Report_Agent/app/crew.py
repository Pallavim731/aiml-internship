from crewai import Agent, Task, Crew, Process
from crewai import LLM


# Local Ollama LLM
llm = LLM(
    model="ollama/llama3.2:3b",
    base_url="http://localhost:11434"
)


researcher = Agent(
    role="AI Researcher",
    goal="Research the given topic and provide useful factual information",
    backstory="You are a careful AI researcher who explains technical topics clearly.",
    llm=llm,
    verbose=False
)


writer = Agent(
    role="Research Report Writer",
    goal="Create a clear and structured research report",
    backstory="You are a technical writer who converts research into simple reports.",
    llm=llm,
    verbose=False
)


reviewer = Agent(
    role="Report Reviewer",
    goal="Check the report for clarity, completeness and relevance",
    backstory="You are a quality reviewer who checks AI-generated reports.",
    llm=llm,
    verbose=False
)


def create_crew(topic):

    research_task = Task(
        description=f"""
Research the following topic:

{topic}

Provide important information, applications,
advantages and relevant points in simple language.
""",
        expected_output="A useful research summary about the topic.",
        agent=researcher
    )

    writing_task = Task(
        description=f"""
Create a structured research report about:

{topic}

Use the research collected by the researcher.

Include:
1. Introduction
2. Important points
3. Applications
4. Advantages
5. Conclusion
""",
        expected_output="A structured research report.",
        agent=writer
    )

    review_task = Task(
        description="""
Review the research report created by the writer.

Check:
- Clarity
- Relevance
- Completeness
- Structure

Give a short review.
""",
        expected_output="A short quality review of the report.",
        agent=reviewer
    )

    crew = Crew(
        agents=[researcher, writer, reviewer],
        tasks=[research_task, writing_task, review_task],
        process=Process.sequential,
        verbose=False
    )

    return crew