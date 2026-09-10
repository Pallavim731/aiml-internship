from dotenv import load_dotenv

load_dotenv()

from crewai import Crew, Process

from agents import create_agents
from tasks import create_tasks


def run_crew():

    # Create agents
    researcher, writer, reviewer = create_agents()

    # Create tasks
    research_task, code_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer
    )

    # Create crew
    crew = Crew(
        agents=[
            researcher,
            writer,
            reviewer
        ],
        tasks=[
            research_task,
            code_task,
            writing_task,
            review_task
        ],
        process=Process.sequential,
        verbose=True
    )

    # Run crew
    result = crew.kickoff()

    # Display final result
    print("\n" + "=" * 60)
    print("FINAL CREW OUTPUT")
    print("=" * 60)
    print(result)

    # Save output
    import os

    os.makedirs("outputs", exist_ok=True)

    with open(
        "outputs/web_research_report.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(str(result))

    print("\nReport saved to outputs/web_research_report.md")

    return result


if __name__ == "__main__":
    run_crew()