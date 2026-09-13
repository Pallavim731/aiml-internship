from dotenv import load_dotenv
load_dotenv(override=True)

from crewai import Crew, Process
from agents import create_agents
from tasks import create_tasks
from pathlib import Path


def run_crew():
    researcher, writer, reviewer = create_agents()

    research_task, writing_task, review_task = create_tasks(
        researcher,
        writer,
        reviewer
    )

    crew = Crew(
        agents=[
            researcher,
            writer,
            reviewer
        ],
        tasks=[
            research_task,
            writing_task,
            review_task
        ],
        process=Process.sequential,
        verbose=True
    )

    result = crew.kickoff()

    print("\n" + "=" * 60)
    print("FINAL CREW OUTPUT")
    print("=" * 60)
    print(result)

    # Create outputs folder
    output_dir = Path("outputs")
    output_dir.mkdir(exist_ok=True)

    # Save final result
    with open(
        output_dir / "web_research_report.md",
        "w",
        encoding="utf-8"
    ) as f:
        f.write(str(result))

    print("\nReport saved to:")
    print("outputs/web_research_report.md")

    return result


if __name__ == "__main__":
    run_crew()