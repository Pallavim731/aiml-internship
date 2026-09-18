\# W10D5 — Stateful Customer Support Agent



\## 1. Project Overview



This project implements a \*\*stateful customer support agent\*\* using CrewAI and LangGraph.



The agent can remember previous messages during a conversation and use that information when answering the customer's next question.



The project also includes:



\* \*\*CrewAI\*\* — creates the support and review agents

\* \*\*LangGraph\*\* — manages conversation state and workflow

\* \*\*MLflow\*\* — tracks customer-support runs and metrics

\* \*\*Ragas\*\* — evaluates the quality of generated responses

\* \*\*Ollama\*\* — runs the local Llama 3.2 model

\* \*\*Pytest\*\* — tests the workflow and state management



\## 2. Architecture



```text

Customer Message

&#x20;      |

&#x20;      v

LangGraph State

&#x20;      |

&#x20;      v

CrewAI Support Agent

&#x20;      |

&#x20;      v

CrewAI Reviewer Agent

&#x20;      |

&#x20;      v

Final Response

&#x20;      |

&#x20;      +------> Conversation History

&#x20;      |

&#x20;      +------> MLflow Tracking

&#x20;      |

&#x20;      v

Customer

```



\## 3. Main Components



\### `app/state.py`



Defines the customer-support state.



The state stores:



\* Customer ID

\* Conversation history

\* Current message

\* Agent response

\* Issue type

\* Resolution status



\### `app/agents.py`



Creates two CrewAI agents:



1\. Customer Support Specialist

2\. Customer Support Reviewer



The agents use the local Ollama Llama 3.2 model.



\### `app/tasks.py`



Defines the tasks performed by the support and reviewer agents.



\### `app/crew.py`



Creates and runs the CrewAI crew.



The support agent first prepares a response and the reviewer checks it.



\### `app/workflow.py`



Creates the LangGraph workflow.



It:



1\. Receives the customer message.

2\. Reads previous conversation history.

3\. Detects the issue type.

4\. Runs the CrewAI support workflow.

5\. Stores the new conversation turn.



\### `app/evaluate.py`



Uses Ragas to evaluate the generated response using:



\* Faithfulness

\* Answer relevancy



The evaluation uses local Ollama models.



\### `app/main.py`



Runs the complete interactive customer-support application and records information in MLflow.



\## 4. Issue Detection



The application can identify basic issue categories:



\* Order

\* Refund

\* Payment

\* Account

\* General



For example:



```text

"My order has not arrived."

&#x20;       ↓

Order issue

```



```text

"I was charged twice."

&#x20;       ↓

Payment issue

```



\## 5. Installation



Create and activate a Python virtual environment:



```cmd

python -m venv venv

venv\\Scripts\\activate

```



Install the required packages:



```cmd

pip install -r requirements.txt

```



Make sure Ollama is running and the required models are available.



\## 6. Run the Application



From the project directory:



```cmd

python -m app.main

```



Enter a customer ID when requested.



Example:



```text

Enter customer ID: CUST001

```



Then enter customer messages:



```text

Customer: My order has not arrived yet.

```



The agent processes the request and returns a response.



The conversation history is retained for the next message.



Type:



```text

exit

```



to end the conversation.



\## 7. MLflow



The application records:



\* Customer ID

\* Customer message

\* Issue type

\* Response length

\* Conversation turns

\* Generated response



MLflow uses a local SQLite tracking database.



To start the MLflow interface:



```cmd

mlflow ui

```



Then open the local MLflow address shown in the terminal.



\## 8. Ragas Evaluation



Run:



```cmd

python -m app.evaluate

```



The evaluation checks the generated answer for:



\* Faithfulness

\* Answer relevancy



The project uses Ollama locally for the evaluation LLM and embeddings.



\## 9. Testing



Run all tests:



```cmd

pytest -v

```



The tests check:



\* State structure

\* Order issue detection

\* Refund issue detection

\* Payment issue detection

\* Account issue detection

\* General issue detection

\* Conversation state preservation



\## 10. Project Structure



```text

W10D5\_Stateful\_Customer\_Support\_Agent/

│

├── app/

│   ├── \_\_init\_\_.py

│   ├── agents.py

│   ├── crew.py

│   ├── evaluate.py

│   ├── main.py

│   ├── state.py

│   ├── tasks.py

│   └── workflow.py

│

├── tests/

│   └── test\_workflow.py

│

├── outputs/

│

├── .gitignore

├── pytest.ini

├── requirements.txt

└── README.md

```



\## 11. Technologies Used



| Technology | Purpose                      |

| ---------- | ---------------------------- |

| Python     | Application development      |

| CrewAI     | Multi-agent customer support |

| LangGraph  | Stateful workflow            |

| Ollama     | Local LLM and embeddings     |

| MLflow     | Experiment and run tracking  |

| Ragas      | Response evaluation          |

| Pytest     | Automated testing            |

| Git/GitHub | Version control              |



\## 12. Key Learning



This project demonstrates how a customer-support application can combine multiple AI/ML tools.



LangGraph maintains the conversation state, CrewAI handles the support and review agents, MLflow tracks execution information, and Ragas evaluates response quality.



