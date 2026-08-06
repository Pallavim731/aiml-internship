import requests

URL = "http://localhost:11434/api/generate"

questions = [
    "What is Artificial Intelligence?",
    "Explain Deep Learning.",
    "Benefits of Local LLMs."
]

models = [
    "llama3.2:3b",
    "qwen2.5:3b"
]

for question in questions:

    print("="*80)
    print("QUESTION:", question)
    print("="*80)

    for model in models:

        response = requests.post(
            URL,
            json={
                "model": model,
                "prompt": question,
                "stream": False
            }
        )

        print("\nMODEL:", model)
        print(response.json()["response"])