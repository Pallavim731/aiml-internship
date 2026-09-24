"""
CrewAI crew for the Stateful Customer Support Agent.
"""

from crewai import Crew, Process

from .agents import create_support_agent, create_reviewer_agent
from .tasks import create_support_task, create_review_task


def create_support_crew(customer_message, conversation_history):
    """Create the CrewAI customer support crew."""

    support_agent = create_support_agent()
    reviewer_agent = create_reviewer_agent()

    support_task = create_support_task(
        support_agent,
        customer_message,
        conversation_history,
    )

    review_task = create_review_task(
        reviewer_agent,
        customer_message,
        conversation_history,
    )

    # The reviewer receives the support agent's output.
    review_task.context = [support_task]

    crew = Crew(
        agents=[
            support_agent,
            reviewer_agent,
        ],
        tasks=[
            support_task,
            review_task,
        ],
        process=Process.sequential,
        verbose=True,
    )

    return crew


def run_support_crew(customer_message, conversation_history):
    """Run the support crew and return the final response."""

    crew = create_support_crew(
        customer_message,
        conversation_history,
    )

    result = crew.kickoff()

    return str(result)