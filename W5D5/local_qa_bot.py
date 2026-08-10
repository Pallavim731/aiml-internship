import requests


OLLAMA_URL = "http://localhost:11434/api/chat"

MODEL = "llama3.2:3b"


SYSTEM_PROMPT = """
You are a helpful local AI assistant.

Answer questions clearly and accurately.
Use simple explanations when possible.
If you are unsure about an answer, say so instead
of inventing information.
"""


def ask_ollama(question, model=MODEL):
    """
    Send a question to Ollama and return the response.
    """

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ],
        "stream": False
    }

    response = requests.post(
        OLLAMA_URL,
        json=payload,
        timeout=120
    )

    response.raise_for_status()

    data = response.json()

    return data["message"]["content"]


def main():

    print("=" * 60)
    print("LOCAL Q&A BOT")
    print("=" * 60)

    print(f"Model: {MODEL}")
    print("Type 'exit' to quit.\n")

    while True:

        question = input("You: ")

        if question.lower() == "exit":
            print("Goodbye!")
            break

        if not question.strip():
            continue

        try:

            answer = ask_ollama(question)

            print("\nAssistant:")
            print(answer)
            print()

        except requests.exceptions.ConnectionError:

            print(
                "\nError: Could not connect to Ollama."
            )

            print(
                "Make sure Ollama is running.\n"
            )

        except requests.exceptions.RequestException as error:

            print(
                f"\nOllama API error: {error}\n"
            )


if __name__ == "__main__":
    main()