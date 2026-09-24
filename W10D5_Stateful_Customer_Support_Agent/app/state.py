
"""
State definition for the Stateful Customer Support Agent.

LangGraph uses this state object to remember information
between different steps of the customer conversation.
"""

from typing import TypedDict, List


class CustomerSupportState(TypedDict):
    """State maintained throughout the support conversation."""

    customer_id: str
    conversation: List[str]
    current_message: str
    response: str
    issue_type: str
    resolved: bool