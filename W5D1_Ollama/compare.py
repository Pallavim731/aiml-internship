import requests

URL = "http://localhost:11434/api/generate"

questions = [
    "Explain Machine Learning.",
    "What is Docker?",
    "Write a Python function to reverse a string."
]

models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

for question in questions:

    print("=" * 60)
    print("QUESTION:", question)
    print("=" * 60)

    for model in models:

        data = {
            "model": model,
            "prompt": question,
            "stream": False
        }

        response = requests.post(URL, json=data)

        print(f"\nMODEL: {model}\n")
        print(response.json()["response"])
        print("-" * 60)