# Agent Implementation Plan

This document is the step-by-step build plan for this project.

The goal is to learn agent engineering from scratch by starting with a small local CLI agent and gradually evolving it toward a production-ready agent system. The project should teach the fundamentals first, then add reliability, observability, retrieval, stronger control flow, and deployment concerns.

## Product Goal

Build a personal learning agent that can:

- chat from a CLI, and later from an API or UI
- use tools instead of guessing
- save and search notes
- remember user preferences
- explain its internal steps through traces
- handle errors safely
- grow from a simple one-tool loop into a multi-step production-style agent

The learning principle is:

```txt
Build the loop yourself
Understand every moving part
Add safety and tests
Then add more advanced architecture
```

## Current Architecture

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

## Current Runtime Flow

Direct answer flow:

```txt
User message
  -> CLI receives input
  -> agent loads recent chat memory
  -> agent loads saved preferences
  -> agent sends context + tool schemas to Gemini
  -> Gemini answers directly
  -> agent saves turn to memory
  -> agent saves trace events
  -> CLI prints final answer
```

Tool flow:

```txt
User message
  -> CLI receives input
  -> agent loads recent chat memory
  -> agent loads saved preferences
  -> agent sends context + tool schemas to Gemini
  -> Gemini returns function_call
  -> Python validates tool name and arguments
  -> Python executes tool
  -> Python appends function_call and function_response to context
  -> Gemini writes final answer from tool result
  -> agent saves turn to memory
  -> agent saves trace events
  -> CLI prints final answer
```

Important rule:

```txt
The LLM does not execute tools.
The LLM requests tools.
Python validates and executes tools.
```

## Current Components

### `main.py`

Status: implemented.

Responsibilities:

- CLI chat loop
- `--debug` flag
- colored terminal trace output using Rich
- calls `run_agent()`
- prints final answer

### `agent_loop.py`

Status: implemented.

Responsibilities:

- orchestrates one agent turn
- loads memory and preferences
- calls Gemini
- handles multi-step tool calls
- enforces current step/LLM/tool call limits
- records trace events
- handles LLM/tool errors
- saves final turn and trace

Current limits:

```txt
MAX_LLM_CALLS = 5
MAX_TOOL_CALLS = 3
MAX_STEPS = 6
```

### `llm_client.py`

Status: implemented.

Responsibilities:

- creates Gemini client
- defines model
- defines system instruction
- defines tool declarations
- calls Gemini with tool schemas

Current model:

```txt
gemini-2.5-flash
```

### `tool_manager.py`

Status: implemented.

Responsibilities:

- tool registry
- tool lookup
- argument validation with `inspect.signature(...).bind(...)`
- tool execution
- returns `ToolResult`

Current tools:

- `calculator`
- `save_note`
- `list_notes`
- `read_note`
- `search_notes`
- `remember_preference`
- `recall_preferences`

### `tools/calculator.py`

Status: implemented.

Responsibilities:

- safe math evaluation
- uses `ast`, not raw `eval`
- supports basic arithmetic

### `tools/notes.py`

Status: implemented.

Responsibilities:

- save notes to `data/notes`
- list notes
- read notes
- keyword-search notes

Current limitation:

```txt
search_notes is keyword-based, not semantic.
```

### `memory.py`

Status: implemented.

Responsibilities:

- save recent chat turns to `data/memory.json`
- keep last 10 user/assistant turns
- convert memory into Gemini content format
- save user preferences to `data/preferences.json`
- build preference context for future model calls

Current limitation:

```txt
No token counting.
No summarization of old memory.
No semantic long-term memory.
```

### `trace.py`

Status: implemented.

Responsibilities:

- `TraceEvent`
- `AgentResult`
- save traces to `data/traces.jsonl`

### `tests/`

Status: implemented.

Current coverage:

- calculator behavior
- notes behavior
- memory behavior
- preference memory
- tool manager success/error paths
- trace file logging
- agent loop direct-answer path
- agent loop tool-call path
- agent loop multi-tool path
- step limit behavior

## Implemented Milestones

### Milestone 1: Basic LLM Call

Status: complete.

Learned:

- API key setup
- Gemini client usage
- prompt in, response out

### Milestone 2: CLI Chat

Status: complete.

Learned:

- simple terminal loop
- separating UI from agent logic

### Milestone 3: Manual Tool Execution

Status: complete.

Learned:

- tools are normal Python functions
- agent code controls execution
- LLM should not directly run code

### Milestone 4: Gemini Function Calling

Status: complete.

Learned:

- tool schemas
- `function_call`
- `function_response`
- two-call tool flow

### Milestone 5: Tool Manager

Status: complete.

Learned:

- tool registry
- generic tool execution
- validation before execution

### Milestone 6: Multiple Tools

Status: complete.

Learned:

- different tools have different argument shapes
- Gemini can choose among available tools

### Milestone 7: Notes

Status: complete.

Learned:

- local file tools
- save/list/read/search workflow
- basic retrieval

### Milestone 8: Memory

Status: complete.

Learned:

- chat history memory
- preference memory
- context injection

### Milestone 9: Tracing

Status: complete.

Learned:

- agents need observability
- trace events explain decisions
- terminal traces and JSONL traces serve different needs

### Milestone 10: Error Handling

Status: complete.

Learned:

- API failures should not crash the agent
- tool failures should become structured results
- invalid tool args should be caught before execution

### Milestone 11: Step Limits

Status: complete for current architecture.

Learned:

- every agent loop needs guardrails
- call limits prevent runaway behavior
- limits become more important when multi-step loops are added

### Milestone 12: Tests

Status: complete and ongoing.

Learned:

- deterministic parts should be tested without calling the LLM
- fake Gemini responses can test agent-loop behavior

### Milestone 13: Multi-Step Tool Loop

Status: complete.

Learned:

- agents can run more than one tool in a single user turn
- model responses may request one or multiple tool calls
- each tool result must be appended back into conversation context
- step, LLM call, and tool call limits are necessary guardrails
- multi-step behavior can be tested with fake LLM responses

## Next Implementation Phase

### Phase 1: Token Counting and Context Truncation

Priority: next.

Why:

The prompt will grow from:

- chat memory
- preferences
- tool call history
- tool results
- note search results

Without limits, long context can become expensive, slow, or rejected by the model.

Implementation steps:

1. Add `context_manager.py`.
2. Start with approximate token counting:

   ```txt
   approx_tokens = len(text) // 4
   ```

3. Define budgets:

   ```txt
   MAX_CONTEXT_TOKENS
   MAX_TOOL_RESULT_CHARS
   MAX_MEMORY_MESSAGES
   MAX_NOTE_SEARCH_RESULTS
   ```

4. Truncate large tool results before sending to Gemini.
5. Keep recent memory first.
6. Add trace events:

   ```txt
   context_budget_checked
   context_truncated
   tool_result_truncated
   ```

7. Add tests for truncation.

Acceptance criteria:

- Large notes/tool results do not explode prompt size.
- Trace shows when truncation happened.
- Agent still gives useful answers after truncation.

### Phase 2: Stronger Structured Validation

Priority: after context management.

Why:

Tool schemas guide the model, but Python should still strictly validate data.

Implementation steps:

1. Introduce Pydantic models for tool arguments.
2. Define each tool with:

   ```txt
   name
   description
   args_model
   function
   ```

3. Replace manual tool schema duplication with generated schemas.
4. Validate all tool args through Pydantic.
5. Return clear validation errors.

Acceptance criteria:

- Tool schema and Python validation come from one source of truth.
- Bad args produce clear user and trace errors.
- Tests cover validation failures.

### Phase 3: Better Note Retrieval

Priority: after structured validation.

Current:

```txt
keyword search
```

Next:

```txt
ranked keyword search
better snippets
metadata
```

Implementation steps:

1. Return structured search results internally.
2. Rank exact title matches higher than content matches.
3. Limit result count.
4. Include snippets with surrounding context.
5. Add note metadata:

   ```txt
   title
   filename
   created_at
   updated_at
   ```

Acceptance criteria:

- Search results are easier for the model to use.
- User can ask broad note questions with better answers.

### Phase 4: Semantic Memory and RAG

Priority: after better keyword retrieval.

Why:

Keyword search cannot find meaning-based matches.

Implementation steps:

1. Add embeddings.
2. Choose a local vector store:

   - ChromaDB
   - LanceDB
   - SQLite vector extension

3. Embed note chunks.
4. Add `semantic_search_notes(query)`.
5. Compare keyword vs semantic search.
6. Add tests with mocked embeddings or deterministic fixtures.

Acceptance criteria:

- Agent can find notes by meaning, not exact words.
- Search results include source filenames/snippets.
- Runtime notes remain local and ignored by Git.

### Phase 5: Async Execution

Priority: when tools become slow or external.

Why:

Local tools are fast. External tools are not.

Useful for:

- web search
- API calls
- database queries
- multiple independent tools

Implementation steps:

1. Convert tool manager to support sync and async tools.
2. Add async agent loop option.
3. Use `asyncio` for slow tools.
4. Add timeouts.
5. Add trace timings.

Acceptance criteria:

- Slow tools do not block unnecessarily.
- Tool timeouts return clean errors.
- Trace shows durations.

### Phase 6: Streaming Responses

Priority: after core logic is stable.

Why:

Streaming improves user experience but does not change core reasoning.

Implementation steps:

1. Add streaming support in `llm_client.py`.
2. Stream direct model answers in CLI.
3. For tool calls, stream only final answer after tool execution.
4. Later expose streaming through API/UI.

Acceptance criteria:

- CLI can show long answers progressively.
- Tool flow remains correct.

### Phase 7: Dynamic Tool Auto-Discovery

Priority: when tool count grows.

Current:

```python
self.tools = {
    "calculator": calculate,
    ...
}
```

Target:

```txt
tool modules declare metadata
ToolManager discovers and registers them
Gemini tool schemas are generated from registry
```

Implementation steps:

1. Create a `ToolSpec` abstraction.
2. Each tool exports a spec.
3. ToolManager loads specs from `tools/`.
4. LLM tool declarations are generated automatically.
5. Add tests for discovery.

Acceptance criteria:

- Adding a tool does not require editing multiple files.
- Tool implementation, schema, and validation stay together.

### Phase 8: Plan-and-Execute

Priority: after multi-step loop and validation.

Why:

ReAct-style "decide next step" is good for small tasks but weaker for complex tasks.

Target:

```txt
User task
  -> planner LLM creates structured plan
  -> executor runs steps
  -> final answer summarizes execution
```

Implementation steps:

1. Add planning prompt.
2. Force structured plan output.
3. Validate plan schema.
4. Execute each step through the agent loop.
5. Trace plan creation and execution.

Acceptance criteria:

- Complex tasks are decomposed before execution.
- User can see plan in debug mode.
- Plan failure is handled safely.

### Phase 9: Reflection / Self-Critique

Priority: after plan-and-execute.

Why:

Improves answer quality for complex tasks.

Flow:

```txt
draft answer
  -> hidden critique call
  -> revise or return
```

Implementation steps:

1. Add optional reflection mode.
2. Critique asks:

   ```txt
   Did the answer satisfy the user?
   Were tools used correctly?
   Is anything missing?
   ```

3. Add max one revision to avoid loops.
4. Trace reflection result.

Acceptance criteria:

- Reflection improves complex answers.
- Reflection has strict limits.
- User does not see noisy internal critique unless debug mode is on.

### Phase 10: API and UI

Priority: after core agent is reliable.

Implementation options:

- FastAPI API
- simple web UI
- Streamlit prototype

API shape:

```txt
POST /chat
GET /traces
GET /notes
GET /health
```

Acceptance criteria:

- Agent can be called outside CLI.
- Runtime data remains private.
- Errors return structured API responses.

### Phase 11: Production Hardening

Priority: before shipping.

Production checklist:

- config management
- logging levels
- request IDs
- retry policies
- rate-limit handling
- tool timeouts
- eval dataset
- cost tracking
- prompt versioning
- deployment docs
- security review
- secret management
- backup/export for notes and memory

Acceptance criteria:

- Agent can run predictably for real users.
- Failures are observable.
- Data and secrets are not accidentally committed.

## Suggested Learning Sequence

Follow this order:

```txt
1. Understand current code
2. Run current agent with --debug
3. Read trace events for direct and tool paths
4. Study the multi-step tool loop
5. Add context/token management
6. Add Pydantic validation
7. Improve retrieval
8. Add semantic search/RAG
9. Add async tools
10. Add streaming
11. Add dynamic tool discovery
12. Add plan-and-execute
13. Add reflection
14. Add API/UI
15. Production hardening
```

## Commit Strategy

Use small commits after each milestone.

Suggested format:

```txt
Add multi-step agent loop
Add context budget manager
Add Pydantic tool validation
Improve note search ranking
Add semantic note search
Add async tool execution
Add streaming CLI responses
Add dynamic tool registry
Add planner executor flow
Add reflection review step
```

Before every commit:

```bash
venv/bin/python -m pytest
git status --short
```

## Current Definition of Done

A feature is done when:

- code is readable
- functions have docstrings
- trace events explain important behavior
- tool errors are handled
- tests cover the deterministic parts
- runtime/private data is ignored by Git
- README or this plan is updated if behavior changes

## Immediate Next Task

Implement:

```txt
Token counting and context truncation
```

This is the next best architecture step because the project now has:

- tools
- memory
- trace logging
- error handling
- step limits
- multi-step tool calls
- tests

Multi-step agents grow context quickly, so the next job is making sure memory, notes, and tool results stay inside a controlled context budget.
