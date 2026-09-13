from app.workflow import research_graph


def test_research_workflow():
    result = research_graph.invoke({
        "topic": "Generative AI"
    })

    assert "research" in result
    assert "report" in result
    assert "review" in result

    assert len(result["report"]) > 0
    assert len(result["review"]) > 0