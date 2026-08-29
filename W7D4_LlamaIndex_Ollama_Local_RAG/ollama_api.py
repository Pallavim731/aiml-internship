from ollama import chat


SYSTEM_PROMPT = """
You are an AI assistant for an AI/ML engineering student.

Follow these rules:
1. Explain concepts in simple and understandable language.
2. Give practical examples when possible.
3. Keep answers structured and clear.
4. Focus on AI and machine learning concepts.
"""


prompts = [
    "What is artificial intelligence?",
    "What is machine learning?",
    "Explain the difference between AI and machine learning.",
    "What is a neural network?",
    "Give an example of machine learning used in daily life."
]


def ask_ollama(prompt):
    response = chat(
        model="llama3.2:3b",
        messages=[
            {
                "role": "system",
                "content": SYSTEM_PROMPT
            },
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


print("=" * 60)
print("W7D4: OLLAMA LOCAL LLM INFERENCE")
print("=" * 60)


for i, prompt in enumerate(prompts, start=1):

    print(f"\nPROMPT {i}")
    print("-" * 60)
    print("Question:", prompt)

    answer = ask_ollama(prompt)

    print("\nResponse:")
    print(answer)
    print("=" * 60)