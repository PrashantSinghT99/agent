from tool_manager import ToolManager


def test_tool_manager_runs_registered_tool():
    """ToolManager runs a known tool and returns a successful ToolResult."""
    result = ToolManager().run("calculator", {"expression": "6 * 7"})

    assert result.ok is True
    assert result.output == "42"
    assert result.error is None


def test_tool_manager_handles_unknown_tool():
    """ToolManager returns an error for unknown tools."""
    result = ToolManager().run("unknown", {})

    assert result.ok is False
    assert result.output == "I cannot use unknown tool: unknown."
    assert "Available tools:" in result.error


def test_tool_manager_handles_invalid_args():
    """ToolManager validates tool arguments before calling the tool."""
    result = ToolManager().run("calculator", {"wrong_key": "6 * 7"})

    assert result.ok is False
    assert result.output == "The tool calculator was called with invalid arguments."
    assert "expression" in result.error
