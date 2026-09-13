from langgraph.graph import StateGraph, START, END

from app.research_state import ResearchState
from app.crew import create_crew


def research_node(state: ResearchState):
    topic = state["topic"]

    crew = create_crew(topic)
    result = crew.kickoff()

    return {
        "research": str(result)
    }


def report_node(state: ResearchState):
    topic = state["topic"]
    research = state["research"]

    report = f"""# Automated Research Report

## Topic

{topic}

## CrewAI Research

{research}

## Conclusion

The research was generated using a multi-agent CrewAI workflow.
"""

    return {
        "report": report
    }


def review_node(state: ResearchState):
    report = state["report"]

    review = (
        "Review completed successfully.\n"
        f"Report length: {len(report)} characters.\n"
        "The report contains the topic, CrewAI research and conclusion."
    )

    return {
        "review": review
    }


# Create LangGraph workflow
graph_builder = StateGraph(ResearchState)

graph_builder.add_node("research", research_node)
graph_builder.add_node("report", report_node)
graph_builder.add_node("review", review_node)

graph_builder.add_edge(START, "research")
graph_builder.add_edge("research", "report")
graph_builder.add_edge("report", "review")
graph_builder.add_edge("review", END)

research_graph = graph_builder.compile()