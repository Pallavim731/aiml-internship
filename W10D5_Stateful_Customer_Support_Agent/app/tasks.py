"""
CrewAI tasks for the Stateful Customer Support Agent.
"""

from crewai import Task


def create_support_task(agent, customer_message, conversation_history):
    """Create a task for handling the customer's request."""

    return Task(
        description=f"""
You are handling a customer support request.

Previous conversation:
{conversation_history}

Latest customer message:
{customer_message}

Instructions:
1. Understand the customer's problem.
2. Use the previous conversation when relevant.
3. Do not ask the customer to repeat information already provided.
4. Give a clear and professional answer.
5. If information is missing, ask only for the information needed.
6. Do not invent order details, refunds, policies, or account information.
""",
        expected_output=(
            "A concise, helpful and professional customer support response."
        ),
        agent=agent,
    )


def create_review_task(agent, customer_message, conversation_history):
    """Create a task for reviewing the support response."""

    return Task(
        description=f"""
Review the customer support response produced by the previous task.

Conversation history:
{conversation_history}

Customer's latest message:
{customer_message}

The support response will be provided through the previous task's output.

Check whether the response:
1. Addresses the customer's actual question.
2. Uses the conversation context correctly.
3. Is clear and professional.
4. Avoids unsupported claims.
5. Does not invent customer or order information.

If improvements are needed, provide a corrected response.
Otherwise, return the original response.
""",
        expected_output=(
            "A final customer support response that is clear, accurate "
            "and helpful."
        ),
        agent=agent,
    )