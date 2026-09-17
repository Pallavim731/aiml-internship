# W10D1 - LangGraph Stateful Agent Graphs

## Objective

Build a stateful agent graph using LangGraph with:

- Three main nodes
- Conditional routing
- Stateful execution
- Human-in-the-loop interrupt
- Automated tests

## Architecture

User Input
|
v
Classify
|
v
Route
|
+------ Question
|
+------ Task
|
+------ Complaint
|
+------ General
|
v
Respond
|
v
Human Review
|
v
Final Response

## Nodes

### 1. Classify

Analyzes the user input and assigns a category:

- question
- task
- complaint
- general

### 2. Route

Uses the classification stored in the graph state to select the appropriate route.

### 3. Respond

Generates a response based on the classification.

### 4. Human Review

Uses LangGraph's interrupt mechanism to pause execution and wait for human feedback.

### 5. Final Response

Resumes execution after human input and produces the final result.

## Conditional Routing

The graph uses conditional edges based on the classification.

Example:

question -> respond
task -> respond
complaint -> respond
general -> respond

## Human-in-the-Loop

The graph uses:

- interrupt()
- Command(resume=...)
- MemorySaver()

The workflow pauses before final response, accepts human feedback, and then resumes.

## Test Inputs

1. What is LangGraph?
2. How can I learn Python?
3. Create a research report
4. My application is not working
5. Hello, good morning

## Running the Project

Activate the virtual environment:

```powershell
venv\Scripts\activate
```
