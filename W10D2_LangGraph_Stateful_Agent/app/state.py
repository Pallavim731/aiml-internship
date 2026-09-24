from typing import TypedDict


class AgentState(TypedDict, total=False):
    user_input: str
    category: str
    response: str
    human_approval: str