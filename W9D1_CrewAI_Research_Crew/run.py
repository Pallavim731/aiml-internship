from dotenv import load_dotenv
from app.crew import create_crew


def main():
    load_dotenv()

    topic = input(
        "\nEnter a research topic "
        "(press Enter for 'Generative AI'): "
    ).strip()

    if not topic:
        topic = "Generative AI"

    print("\n" + "=" * 70)
    print("CREWAI MULTI-AGENT RESEARCH CREW")
    print("=" * 70)
    print(f"Topic: {topic}")
    print("=" * 70)

    crew = create_crew()

    result = crew.kickoff(
        inputs={"topic": topic}
    )

    print("\n" + "=" * 70)
    print("FINAL REPORT")
    print("=" * 70)
    print(result)

    with open(
        "outputs/web_research_report.md",
        "w",
        encoding="utf-8"
    ) as file:
        file.write(str(result))

    print("\nReport saved to:")
    print("outputs/web_research_report.md")


if __name__ == "__main__":
    main()