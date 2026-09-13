from typing import TypedDict


class ResearchState(TypedDict, total=False):
    topic: str
    research: str
    report: str
    review: str