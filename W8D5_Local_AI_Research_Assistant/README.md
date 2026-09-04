\# W8D5 — 2M Capstone: Local AI Research Assistant



\## Overview



This project implements a simple local AI research assistant using LangGraph and Ollama.



The assistant reads local research notes, retrieves relevant information, and uses a locally running language model to generate an answer.



No paid external AI API is required.



\## Architecture



```text

User Question

&#x20;     ↓

LangGraph Workflow

&#x20;     ↓

Retrieve Relevant Context

&#x20;     ↓

Ollama Local LLM

&#x20;     ↓

Generated Research Answer

```



\## Technologies



\* Python

\* LangGraph

\* LangChain

\* Ollama

\* Llama 3.2

\* Pytest

\* Git

\* GitHub



\## Project Structure



```text

W8D5\_Local\_AI\_Research\_Assistant/

│

├── app/

│   ├── \_\_init\_\_.py

│   ├── research\_assistant.py

│   └── main.py

│

├── data/

│   └── research\_notes.txt

│

├── tests/

│   └── test\_assistant.py

│

├── results/

├── requirements.txt

└── README.md

```



\## Setup



Create and activate a Python virtual environment.



Install dependencies:



```powershell

pip install -r requirements.txt

```



Install the local Ollama model:



```powershell

ollama pull llama3.2:3b

```



\## Run



Start the assistant:



```powershell

python app/main.py

```



Enter a question such as:



```text

What is a digital twin?

```



\## Testing



Run:



```powershell

pytest -v

```



\## Code Quality



The project uses:



\* Clear function and class names

\* Python docstrings

\* Input validation

\* Error handling

\* Modular code

\* Automated tests

\* Comments for important processing steps



\## Capstone Result



The prototype demonstrates a local AI research workflow where user questions are processed by a LangGraph pipeline and answered using an Ollama-hosted local language model.



