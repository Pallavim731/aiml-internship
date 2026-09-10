from crewai import Task


def create_tasks(researcher, writer, reviewer):

    # Research Task
    research_task = Task(
        description=(
            "Research the topic: 'Applications of Generative AI in Smart Buildings'. "
            "Use the Web Search Tool to find relevant information. "
            "Identify important applications, benefits, challenges and future possibilities. "
            "Provide clear and useful research findings."
        ),
        expected_output=(
            "A structured list of research findings covering applications, "
            "benefits, challenges and future scope."
        ),
        agent=researcher
    )

    # Code Execution Task
    code_task = Task(
        description=(
            "Use the Code Execution Tool to perform this calculation. "
            "A smart building has 24 rooms. Each room saves 15 percent "
            "energy through Generative AI optimization. "
            "Calculate the average percentage energy saving and explain "
            "the result clearly. The calculation must be performed using "
            "the Code Execution Tool."
        ),
        expected_output=(
            "A numerical calculation performed using the Code Execution Tool "
            "and a short explanation of the result."
        ),
        agent=researcher
    )

    # Writing Task
    writing_task = Task(
        description=(
            "Using the research findings and code execution results, "
            "write a professional report about Applications of Generative AI "
            "in Smart Buildings. Include an introduction, applications, "
            "benefits, challenges, future possibilities and conclusion. "
            "Use clear professional language."
        ),
        expected_output=(
            "A well-structured research report written in clear "
            "professional language."
        ),
        agent=writer
    )

    # Review Task
    review_task = Task(
        description=(
            "Review the research report created by the Technical Writer. "
            "Check its accuracy, completeness, clarity, organization "
            "and logical consistency. Identify strengths, weaknesses "
            "and specific improvements."
        ),
        expected_output=(
            "A review containing strengths, weaknesses and specific "
            "improvements for the research report."
        ),
        agent=reviewer
    )

    return research_task, code_task, writing_task, review_task