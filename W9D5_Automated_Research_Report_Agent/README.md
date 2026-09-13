# Automated Research Report Agent

## Overview

This project is an automated research report generation system built using CrewAI and LangGraph.

The system takes a research topic from the user and:

1. Researches the topic using CrewAI agents.
2. Generates a structured research report.
3. Reviews the generated report.
4. Evaluates the report using Ragas.
5. Tracks the workflow using MLflow.
6. Saves the generated report and review in the `outputs` folder.

## Technologies Used

* CrewAI
* LangGraph
* MLflow
* Ragas
* Ollama
* Python
* Pytest

## How to Run

Activate the Python virtual environment and run:

```cmd
python main.py
```

Enter a research topic when prompted.

## Output

The generated files are saved in:

```text
outputs/
```

The project also contains automated tests for validating the LangGraph workflow structure.
