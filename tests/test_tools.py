import pytest
from notebook_ai.tools import tool, get_tool_names, get_tool_functions, clear_registry, set_verbose

@pytest.fixture(autouse=True)
def clean_registry():
    """Clear tool registry before each test."""
    clear_registry()
    set_verbose(False)  # Disable print during tests
    yield
    clear_registry()
    set_verbose(True)

def test_tool_registration():
    @tool
    def my_tool(x: int) -> int:
        """Double a number."""
        return x * 2

    assert "my_tool" in get_tool_names()

def test_tool_function_retrieval():
    @tool
    def add(a: int, b: int) -> int:
        """Add two numbers."""
        return a + b

    funcs = get_tool_functions(["add"])
    assert len(funcs) == 1
    assert funcs[0](1, 2) == 3

def test_tool_preserves_function():
    @tool
    def multiply(x: int, y: int) -> int:
        """Multiply two numbers."""
        return x * y

    # Decorated function should still work
    assert multiply(3, 4) == 12

def test_tool_docstring_preserved():
    @tool
    def documented():
        """This is my docstring."""
        pass

    assert documented.__doc__ == "This is my docstring."

def test_get_nonexistent_tool():
    funcs = get_tool_functions(["nonexistent"])
    assert funcs == []
