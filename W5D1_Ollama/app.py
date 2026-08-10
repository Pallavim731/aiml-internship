import requests

url = "http://localhost:11434/api/generate"

system_prompt = (
    "You are an AI tutor. "
    "Explain answers simply with examples."
)

prompt = input("Enter your question: ")

data = {
    "model": "llama3.2:3b",
    "prompt": f"System: {system_prompt}\nUser: {prompt}",
    "stream": False
}

response = requests.post(url, json=data)

print("\nAnswer:\n")
print(response.json()["response"])