from dataclasses import dataclass
import inspect

from tools.calculator import calculate
from tools.notes import list_notes, read_note, save_note


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


class ToolManager:
    """Registry and executor for tools the agent is allowed to use."""

    def __init__(self):
        """Register available tool functions by name."""
        self.tools = {
            "calculator": calculate,
            "save_note": save_note,
            "list_notes": list_notes,
            "read_note": read_note,
        }

    def run(self, tool_name: str, tool_args: dict) -> ToolResult:
        """Run a registered tool with keyword arguments.

        Args:
            tool_name: Name of the tool to run.
            tool_args: Dictionary of arguments for the tool function.

        Returns:
            ToolResult containing output or error details.
        """
        tool = self.tools.get(tool_name)

        if tool is None:
            available_tools = ", ".join(self.tools)
            return ToolResult(
                ok=False,
                output=f"I cannot use unknown tool: {tool_name}.",
                error=f"Available tools: {available_tools}",
            )

        try:
            inspect.signature(tool).bind(**tool_args)
        except TypeError as e:
            return ToolResult(
                ok=False,
                output=f"The tool {tool_name} was called with invalid arguments.",
                error=str(e),
            )

        try:
            return ToolResult(ok=True, output=tool(**tool_args))
        except Exception as e:
            return ToolResult(
                ok=False,
                output=f"The tool {tool_name} failed while running.",
                error=str(e),
            )

    def list_tools(self) -> list[str]:
        """Return the names of registered tools."""
        return list(self.tools)
