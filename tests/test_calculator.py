from tools.calculator import calculate


def test_calculate_basic_expression():
    """Calculator evaluates supported math expressions."""
    assert calculate("45 * 12") == "540"
    assert calculate("2 ** 5") == "32"


def test_calculate_rejects_unsafe_expression():
    """Calculator returns an error for unsupported Python syntax."""
    result = calculate("__import__('os').system('echo bad')")

    assert result.startswith("Calculator error:")
