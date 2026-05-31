# Learning Agent

This project is a from-scratch learning journey for building an AI agent.

The goal is to start with a very small CLI agent and gradually evolve it toward a more production-style agent architecture. Each step is intentionally simple enough to understand, but real enough to teach the core ideas behind agents: LLM calls, tool use, memory, tracing, error handling, tests, and guardrails.

## Current Capabilities

- CLI chat interface
- Gemini API integration
- Manual agent loop
- Gemini function/tool calling
- Python-controlled tool execution
- Calculator tool
- Note tools: save, list, read, search
- Chat memory stored in JSON
- Preference memory stored in JSON
- System instruction for stable behavior
- Debug trace in terminal with colors
- Trace logs saved to JSONL
- Tool and LLM error handling
- Step limits for LLM/tool calls
- Local pytest test suite

## Architecture

```txt
User
  |
  v
main.py
  |
  v
agent_loop.py
  |-- memory.py
  |-- llm_client.py
  |-- tool_manager.py
  |     |-- tools/calculator.py
  |     |-- tools/notes.py
  |
  v
trace.py
```

## Agent Flow

For a normal question:

```txt
User message
  -> load chat memory
  -> load preferences
  -> send context to Gemini
  -> Gemini answers directly
  -> save turn to memory
  -> save trace
  -> print answer
```

For a tool-using question:

```txt
User message
  -> load memory and preferences
  -> send context + tool schemas to Gemini
  -> Gemini returns function_call
  -> Python validates and executes the tool
  -> Python appends function_call + function_response
  -> Gemini writes final answer
  -> save turn to memory
  -> save trace
  -> print answer
```

The LLM does not execute tools. It only requests a tool call. Python decides whether the tool exists, validates arguments, executes the function, and sends the result back to the model.

## Files

```txt
main.py              CLI entrypoint
agent_loop.py        Agent orchestration loop
llm_client.py        Gemini client and tool declarations
tool_manager.py      Tool registry and executor
memory.py            Chat memory and preference memory
trace.py             Trace events and JSONL trace logging

tools/calculator.py  Safe calculator tool
tools/notes.py       Note tools

tests/               Local pytest test suite
data/notes/          Runtime note storage
```

## Setup

Create a virtual environment:

```bash
python3 -m venv venv
source venv/bin/activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

Create an environment file:

```bash
cp .env.example .env
```

Add your Gemini API key:

```txt
GEMINI_API_KEY=your_gemini_api_key_here
```

## Run

Normal mode:

```bash
venv/bin/python main.py
```

Debug mode with colored trace:

```bash
venv/bin/python main.py --debug
```

Exit:

```txt
exit
```

or:

```txt
quit
```

## Example Prompts

Calculator:

```txt
what is 12 * 8 % 2 + 4?
```

Notes:

```txt
save a note titled Agent Memory saying Agents can use memory to keep useful context between turns
list my notes
read my note titled Agent Memory
search my notes for memory
```

Preferences:

```txt
remember that I prefer simple examples
what are my preferences?
explain tool calling
```

Debug trace:

```txt
search my notes for memory
```

Expected trace events include:

```txt
user_message_received
memory_loaded
preferences_loaded
llm_request_started
tool_call_requested
tool_executed
tool_result_appended
llm_final_answer_received
memory_saved
```

## Runtime Data

Runtime files are intentionally ignored by Git:

```txt
data/memory.json
data/preferences.json
data/traces.jsonl
data/notes/*.md
```

`data/notes/.gitkeep` is committed only to keep the notes folder in the repo.

## Tests

Run all tests:

```bash
venv/bin/python -m pytest
```

Current tests cover:

- calculator behavior
- note save/list/read/search
- memory save/load
- preference memory
- tool manager success/error handling
- trace JSONL logging
- agent loop direct-answer and tool-call paths
- step limit behavior

## What We Have Learned So Far

- How to call Gemini from Python
- How function calling works
- Why the LLM does not execute tools directly
- How to build a ToolManager
- How to send function responses back to the model
- How to persist short-term chat memory
- How to persist user preferences
- How to add debug traces
- How to log traces to a file
- How to add safety through error handling and step limits
- How to test deterministic parts of an agent without calling the LLM

## Roadmap

### 1. Multi-Step Tool Loop

Current flow supports:

```txt
LLM -> one tool -> LLM final answer
```

Next upgrade:

```txt
LLM -> tool -> LLM -> tool -> LLM final answer
```

This will allow prompts like:

```txt
search my notes for memory and save a summary note
```

### 2. Stronger Guardrails

Add stricter validation around model outputs and tool arguments.

Possible direction:

- Pydantic schemas
- stricter tool argument models
- clearer user-facing validation errors

### 3. Better Retrieval

Current note search is keyword-based.

Next path:

```txt
keyword search
  -> better snippets
  -> embeddings
  -> local vector database
  -> semantic note search / RAG
```

Possible tools:

- ChromaDB
- LanceDB
- SQLite vector extensions

### 4. Better Long-Term Memory

Current memory has:

- recent chat history
- saved preferences

Future memory can include:

- facts about the user
- project-specific memory
- summaries of older conversations
- memory pruning and compaction

### 5. Plan-and-Execute

For complex tasks, add planning:

```txt
Prompt
  -> LLM creates a structured plan
  -> agent executes each step
  -> final answer summarizes results
```

Example:

```txt
Find my notes about memory, calculate how many matches exist, and save a summary.
```

### 6. Reflection / Self-Critique

Before returning an answer, the agent can run a hidden review step:

```txt
Did the answer satisfy the user request?
Were tools used correctly?
Is anything missing?
```

This helps with quality, but should be added only after the basic loop is solid.

### 7. Production Concerns

Later production-style features:

- structured logs
- config management
- retry policies
- tool timeouts
- prompt/version tracking
- eval datasets
- cost tracking
- API or web interface
- authentication and permissions

## Learning Principle

This project intentionally avoids starting with a large framework.

The learning path is:

```txt
Build the loop yourself
Understand the moving parts
Add safety and tests
Then study frameworks with better intuition
```

Frameworks like LangGraph, OpenAI Agents SDK, Pydantic AI, or CrewAI will make more sense after this core loop feels natural.
