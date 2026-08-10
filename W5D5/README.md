\# W5D5 - Local Q\&A Bot with Ollama



\## Objective



Build and test a local question-answering application using

Ollama and locally running large language models.



\## Technologies



\- Python

\- Ollama

\- Llama 3.2 3B

\- Qwen 2.5 3B

\- Requests

\- Git

\- GitHub



\## Features



\- Local LLM inference

\- Ollama API integration

\- Custom system prompt

\- Interactive Q\&A

\- Five prompt tests

\- Llama vs Qwen comparison

\- Response quality analysis



\## Architecture



User Question

&#x20;      ↓

Python Local Q\&A Bot

&#x20;      ↓

Ollama API

&#x20;      ↓

Local LLM

&#x20;      ↓

Response



\## Models Tested



\### llama3.2:3b



Used for local Q\&A and five-prompt testing.



\### qwen2.5:3b



Used for comparison against llama3.2:3b using the same

three questions.



\## Testing



Five questions were tested using llama3.2:3b.



Three identical questions were tested against both

llama3.2:3b and qwen2.5:3b.



\## Why Local LLMs?



Local models can provide:



\- Better privacy for sensitive data

\- Offline or reduced-internet dependency

\- Lower API usage costs

\- Greater control over the inference environment



\## Viva Topics



\### Model Parameter



A model parameter is a learned value inside the neural network

that is adjusted during training.



\### Prompt Parameter



A prompt parameter refers to information or configuration supplied

to influence the model's response, such as the user prompt,

system instructions, temperature, or token limits depending on

the API.



\### Quantisation



Quantisation reduces the numerical precision used to represent

model weights. This can reduce memory usage and improve inference

speed, usually with some possible reduction in model quality.



\## Git



Branch:



feat/aiml-W5-PallaviM



Required commit:



feat: ollama setup — local llm inference

