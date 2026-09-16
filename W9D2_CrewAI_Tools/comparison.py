from pathlib import Path

baseline_file = Path("outputs/baseline_report.md")
web_file = Path("outputs/web_research_report.md")
comparison_file = Path("outputs/comparison.md")

# Read web research report
web_report = web_file.read_text(encoding="utf-8")

# Create baseline if it does not already exist
if not baseline_file.exists():
    baseline_content = """# Baseline Research Report

## Topic
Applications of Generative AI in Smart Buildings

## Research
The baseline report was created without using the web search tool.

It focuses on general knowledge about Generative AI applications
in smart buildings, including energy management, HVAC optimization,
predictive maintenance, occupant comfort, security and automation.
"""
    baseline_file.write_text(baseline_content, encoding="utf-8")

comparison = f"""# Baseline vs Web Research Comparison

## Topic
Applications of Generative AI in Smart Buildings

## Baseline Approach

The baseline version uses the LLM's existing knowledge without
external web search.

### Limitations
- No real-time web information
- Fewer directly referenced sources
- Information may be less current
- Limited evidence from external websites

## Web Research Approach

The web-enabled version uses the Serper search tool to retrieve
current information from online sources before generating the report.

### Improvements
- Provides current web-based information
- Adds information from multiple sources
- Improves factual grounding
- Provides more specific real-world applications
- Helps identify recent developments and use cases

## Web Research Output

{web_report}

## Conclusion

Adding the Serper web search tool improves the research workflow by
giving the CrewAI Research Specialist access to current external
information. The Writer can then use those findings to create a more
grounded report, while the Reviewer can check the final result.
"""

comparison_file.write_text(comparison, encoding="utf-8")

print("Comparison created successfully.")
print("Saved to:", comparison_file)