# W6D1 — LangChain Fundamentals: Chains & Prompts

## Objective

Implemented basic LangChain concepts using Ollama:

- LangChain chains
- PromptTemplate
- Ollama LLM
- OutputParser
- Conversation history
- Tool-based agent workflow

## Technologies Used

- Python
- LangChain
- Ollama
- llama3.2:3b
- VS Code
- Git/GitHub

## Project Structure

```text
W6D1_LangChain/
│
├── chain.py
├── memory.py
├── agent.py
├── requirements.txt
├── README.md
├── SELF_REVIEW.md
└── outputs/
    ├── chain_output.txt
    ├── memory_output.txt
    └── agent_output.txt
```

## Task 1 — LangChain Chain

Implemented:

```text
PromptTemplate → Ollama LLM → OutputParser
```

Tested the chain with five inputs:

1. Artificial Intelligence
2. Machine Learning
3. LangChain
4. Vector Database
5. Generative AI

## Task 2 — Conversation Memory

Implemented conversation history across five turns.

The conversation history is passed back to the LangChain prompt so that the model can use previous messages when generating responses.

## Task 3 — Agent and Tools

Implemented two tools:

1. Calculator
2. Web Search Stub

Tested the workflow with three tasks.

## Output Evidence

Execution outputs are stored in the `outputs` directory.

### Chain

`outputs/chain_output.txt`

### Memory

`outputs/memory_output.txt`

### Agent

`outputs/agent_output.txt`

## Key Learning

A LangChain chain connects multiple processing components into a reusable workflow.

Memory allows previous conversation messages to be supplied to later prompts.

Agents use tools to perform tasks beyond direct language generation.

## Conclusion

Successfully implemented the Week 6 Day 1 LangChain fundamentals practical tasks using Python, LangChain, and Ollama.
