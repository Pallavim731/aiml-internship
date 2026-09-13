from crewai import Task


def create_tasks(researcher, writer, reviewer):

    research_task = Task(
        description=(
            "Research the topic 'Applications of Generative AI in Smart Buildings'. "
            "Use the web search tool to find useful and relevant information. "
            "Cover major applications, benefits, challenges and future possibilities."
        ),
        expected_output=(
            "A structured collection of research findings covering "
            "applications, benefits, challenges and future possibilities."
        ),
        agent=researcher
    )

    writing_task = Task(
        description=(
            "Using the research findings, write a professional report about "
            "Applications of Generative AI in Smart Buildings. "
            "Include an introduction, applications, benefits, challenges, "
            "future possibilities and conclusion."
        ),
        expected_output=(
            "A clear and well-organized research report."
        ),
        agent=writer
    )

    review_task = Task(
        description=(
            "Review the research report created by the Technical Writer. "
            "Check its accuracy, completeness, clarity, organization and "
            "logical consistency. Provide strengths, weaknesses and improvements."
        ),
        expected_output=(
            "A concise review containing strengths, weaknesses and "
            "specific suggestions for improvement."
        ),
        agent=reviewer
    )

    return research_task, writing_task, review_task