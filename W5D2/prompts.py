import requests

url = "http://localhost:11434/api/generate"

system_prompt = """
You are an AI tutor.
Answer clearly.
Use simple English.
Limit answers to 100 words.
"""

prompts = [
    "What is Machine Learning?",
    "Explain Neural Networks.",
    "Difference between AI and ML.",
    "Why is Python popular?",
    "Explain Prompt Engineering."
]

for i, prompt in enumerate(prompts, start=1):

    response = requests.post(
        url,
        json={
            "model": "llama3.2:3b",
            "prompt": prompt,
            "system": system_prompt,
            "stream": False
        }
    )

    print("="*60)
    print(f"Prompt {i}: {prompt}")
    print("="*60)

    print(response.json()["response"])
    print()