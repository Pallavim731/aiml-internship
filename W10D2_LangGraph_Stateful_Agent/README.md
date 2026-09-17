\# W10D2 — LangGraph Stateful Agent



\## Objective



Build a LangGraph state machine that can classify user requests, route them using conditional edges, generate a response, and pause for human approval.



\## Workflow



The agent follows this flow:



```text

START

&#x20; ↓

classify

&#x20; ↓

route

&#x20; ↓

respond

&#x20; ↓

human\_review

&#x20; ↓

END

```



\## Features



\* Classifies requests into:



&#x20; \* Technical

&#x20; \* Urgent

&#x20; \* General

\* Uses conditional routing based on classification.

\* Generates a response according to the category.

\* Uses LangGraph `interrupt()` for human-in-the-loop approval.

\* Uses `MemorySaver` to maintain graph state.

\* Resumes execution using human input.

\* Includes pytest tests for classification and routing.



\## Example Inputs



| Input                                   | Category  |

| --------------------------------------- | --------- |

| How can I learn Python?                 | Technical |

| How do I write Java code?               | Technical |

| This is an emergency, I need help       | Urgent    |

| What is artificial intelligence?        | General   |

| Tell me something about smart buildings | General   |



\## Technologies Used



\* Python 3.12

\* LangGraph

\* Pytest

\* LangGraph MemorySaver

\* LangGraph Interrupt



\## Project Structure



```text

W10D2\_LangGraph\_Stateful\_Agent/

│

├── app/

│   ├── state.py

│   ├── graph.py

│   └── main.py

│

├── tests/

│   ├── test\_graph.py

│   └── conftest.py

│

├── requirements.txt

└── README.md

```



\## How to Run



Activate the virtual environment:



```cmd

venv\\Scripts\\activate

```



Run the agent:



```cmd

python -m app.main

```



Enter a question when prompted.



The agent pauses at the human-review stage and asks for:



```text

yes/no

```



The graph then resumes and completes the workflow.



\## Testing



Run:



```cmd

pytest -v

```



The tests verify:



\* Technical classification

\* Urgent classification

\* General classification

\* Conditional routing



\## Human-in-the-Loop



The `human\_review` node uses LangGraph's interrupt mechanism.



The workflow pauses before completion and waits for human approval. After entering `yes` or `no`, the graph resumes using the saved state.



\## Result



The project demonstrates a stateful LangGraph workflow with classification, conditional routing, response generation, and human-in-the-loop interruption.



