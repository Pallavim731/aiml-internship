# W6D2 — LangChain Memory & Conversation History

## Objective

Implemented conversation memory using LangChain and Ollama.

## Technologies

- Python
- LangChain
- Ollama
- llama3.2:3b

## Implementation

The application maintains conversation history using LangChain message objects.

The workflow is:

User Input
↓
Conversation History
↓
Prompt
↓
Ollama LLM
↓
AI Response
↓
Update History

## Testing

The application was tested across 5 conversation turns.

### Test Conversations

1. User introduces their name.
2. User asks for their name.
3. User provides their field of study.
4. User asks what they are studying.
5. User requests a conversation summary.

## Output

Execution evidence is stored in:

`outputs/memory_output.txt`

## Learning

Conversation memory allows an LLM application to use information from previous turns instead of treating every request as an isolated interaction.