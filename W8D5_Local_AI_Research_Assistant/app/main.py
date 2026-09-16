"""
Command-line interface for the Local AI Research Assistant.
"""

from app.research_assistant import LocalResearchAssistant


def main():
    """Run the research assistant."""
    assistant = LocalResearchAssistant()

    print("======================================")
    print(" Local AI Research Assistant")
    print("======================================")

    question = input("\nEnter your research question: ")

    try:
        answer = assistant.ask(question)

        print("\nAnswer:")
        print(answer)

    except ValueError as exc:
        print(f"Error: {exc}")


if __name__ == "__main__":
    main()