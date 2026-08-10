import requests


OLLAMA_URL = "http://localhost:11434/api/chat"

MODELS = [
    "llama3.2:3b",
    "qwen2.5:3b"
]


QUESTIONS = [
    "Explain machine learning to a beginner in 100 words.",

    "What is the difference between supervised and "
    "unsupervised learning? Give two examples.",

    "Explain how a REST API works using a simple "
    "real-world example."
]


def ask_model(model, question):

    payload = {
        "model": model,
        "messages": [
            {
                "role": "system",
                "content": (
                    "You are a helpful AI assistant. "
                    "Give clear and accurate answers."
                )
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

    return response.json()["message"]["content"]


with open(
    "model_comparison.txt",
    "w",
    encoding="utf-8"
) as file:

    file.write("W5D5 MODEL COMPARISON\n")
    file.write("====================\n\n")

    for question_number, question in enumerate(
        QUESTIONS,
        start=1
    ):

        file.write(
            f"QUESTION {question_number}\n"
        )

        file.write(
            f"{question}\n\n"
        )

        for model in MODELS:

            print("\n" + "=" * 60)
            print(f"MODEL: {model}")
            print("=" * 60)

            print("Question:")
            print(question)

            answer = ask_model(
                model,
                question
            )

            print("\nAnswer:")
            print(answer)

            file.write(
                f"MODEL: {model}\n"
            )

            file.write(
                "ANSWER:\n"
            )

            file.write(
                answer + "\n\n"
            )

        file.write("\n" + "-" * 60 + "\n\n")


print(
    "\nComparison completed."
)

print(
    "Results saved to model_comparison.txt"
)