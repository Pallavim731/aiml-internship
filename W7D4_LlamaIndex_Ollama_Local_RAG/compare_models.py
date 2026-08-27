from ollama import chat


SYSTEM_PROMPT = """
You are a helpful AI/ML assistant.

Explain technical concepts in simple language.
Give clear and accurate answers.
Use examples when appropriate.
"""


questions = [
    "What is the difference between supervised and unsupervised learning?",
    "Explain overfitting in machine learning with a simple example.",
    "Why is model evaluation important in machine learning?"
]


models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]


def get_response(model, question):

    response = chat(
        model=model,
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": question
            }
        ]
    )

    return response["message"]["content"]


print("=" * 70)
print("LLAMA3.2:3B VS QWEN2.5:3B COMPARISON")
print("=" * 70)


for question_number, question in enumerate(questions, start=1):

    print("\n")
    print("#" * 70)
    print(f"QUESTION {question_number}")
    print(question)
    print("#" * 70)

    for model in models:

        print(f"\nMODEL: {model}")
        print("-" * 70)

        answer = get_response(model, question)

        print(answer)