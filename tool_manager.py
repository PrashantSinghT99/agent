"""Tool registry and executor.

Each registered tool has three parts:
  - function   : the Python callable that does the real work
  - args_model : Pydantic BaseModel that validates arguments before execution
  - description: human-readable sentence used to build the Gemini declaration

Adding a new tool:
  1. Write the function in tools/
  2. Add its Pydantic model to tools/schemas.py
  3. Add an entry to _TOOL_REGISTRY below
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

from pydantic import BaseModel, ValidationError

from memory import recall_preferences, remember_preference
from tools.calculator import calculate
from tools.notes import list_notes, read_note, save_note, search_notes
from tools.schemas import (
    CalculatorArgs,
    ListNotesArgs,
    ReadNoteArgs,
    RecallPreferencesArgs,
    RememberPreferenceArgs,
    SaveNoteArgs,
    SearchNotesArgs,
    build_gemini_declaration,
)


# ─── Data types ────────────────────────────────────────────────────────────────


@dataclass
class ToolResult:
    """Result of a tool execution.

    Args:
        ok: Whether the tool ran successfully.
        output: Tool output or user-friendly error text.
        error: Debug error details, if any.
    """

    ok: bool
    output: str
    error: str | None = None


@dataclass
class ToolEntry:
    """Registry entry for one tool.

    Args:
        function: Python callable that implements the tool.
        args_model: Pydantic model class used to validate arguments.
        description: Short sentence shown to the LLM in the tool declaration.
    """

    function: Callable
    args_model: type[BaseModel]
    description: str


# ─── Tool registry ─────────────────────────────────────────────────────────────

# To add a tool: add one ToolEntry here.
# No other file needs to change for registration.
_TOOL_REGISTRY: dict[str, ToolEntry] = {
    "calculator": ToolEntry(
        function=calculate,
        args_model=CalculatorArgs,
        description="Evaluate a basic math expression.",
    ),
    "save_note": ToolEntry(
        function=save_note,
        args_model=SaveNoteArgs,
        description="Save a short note for the user.",
    ),
    "list_notes": ToolEntry(
        function=list_notes,
        args_model=ListNotesArgs,
        description="List all saved notes.",
    ),
    "read_note": ToolEntry(
        function=read_note,
        args_model=ReadNoteArgs,
        description="Read the content of a saved note by title.",
    ),
    "search_notes": ToolEntry(
        function=search_notes,
        args_model=SearchNotesArgs,
        description="Search saved notes by keyword and return matching snippets.",
    ),
    "remember_preference": ToolEntry(
        function=remember_preference,
        args_model=RememberPreferenceArgs,
        description="Save a user preference for future conversations.",
    ),
    "recall_preferences": ToolEntry(
        function=recall_preferences,
        args_model=RecallPreferencesArgs,
        description="Recall saved user preferences.",
    ),
}


# ─── Public helpers ────────────────────────────────────────────────────────────


def get_gemini_declarations() -> list[dict]:
    """Build Gemini function declaration dicts for every registered tool.

    Called by ``llm_client.py`` so tool schemas are never hand-written.

    Returns:
        List of Gemini-compatible function declaration dicts.
    """
    return [
        build_gemini_declaration(name, entry.description, entry.args_model)
        for name, entry in _TOOL_REGISTRY.items()
    ]


# ─── ToolManager ──────────────────────────────────────────────────────────────


class ToolManager:
    """Registry and executor for tools the agent is allowed to use."""

    def run(self, tool_name: str, tool_args: dict) -> ToolResult:
        """Validate and run a registered tool.

        Validation order:
          1. Check the tool name exists in the registry.
          2. Parse and validate arguments through the Pydantic model.
          3. Call the tool function with validated data.

        Args:
            tool_name: Name of the tool to run.
            tool_args: Raw argument dict from the LLM function call.

        Returns:
            ToolResult containing output or structured error details.
        """
        entry = _TOOL_REGISTRY.get(tool_name)

        if entry is None:
            available = ", ".join(_TOOL_REGISTRY)
            return ToolResult(
                ok=False,
                output=f"I cannot use unknown tool: {tool_name}.",
                error=f"Available tools: {available}",
            )

        # Pydantic validates types, required fields, and extra fields.
        try:
            validated = entry.args_model.model_validate(tool_args)
        except ValidationError as exc:
            # Format Pydantic errors into a concise user-facing message.
            problems = "; ".join(
                f"{'.'.join(str(loc) for loc in e['loc'])}: {e['msg']}"
                for e in exc.errors()
            )
            return ToolResult(
                ok=False,
                output=f"The tool {tool_name} was called with invalid arguments.",
                error=problems,
            )

        # Tool exceptions become ToolResult errors instead of crashing the agent.
        try:
            result = entry.function(**validated.model_dump())
            return ToolResult(ok=True, output=result)
        except Exception as exc:
            return ToolResult(
                ok=False,
                output=f"The tool {tool_name} failed while running.",
                error=str(exc),
            )

    def list_tools(self) -> list[str]:
        """Return the names of all registered tools."""
        return list(_TOOL_REGISTRY)
