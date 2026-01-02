import pytest
from notebook_ai.context import safe_repr, resolve_variables, build_system_message

def test_safe_repr_simple():
    assert safe_repr(42) == "42"
    assert safe_repr("hello") == "'hello'"
    assert safe_repr([1, 2, 3]) == "[1, 2, 3]"

def test_safe_repr_truncation():
    long_string = "x" * 5000
    result = safe_repr(long_string, max_length=100)
    assert len(result) < 200
    assert "truncated" in result

def test_safe_repr_error():
    class BadRepr:
        def __repr__(self):
            raise ValueError("nope")

    result = safe_repr(BadRepr())
    assert "repr failed" in result

def test_resolve_variables():
    namespace = {"x": 42, "y": "hello"}
    result = resolve_variables(["x", "y", "z"], namespace)

    assert result["x"] == "42"
    assert result["y"] == "'hello'"
    assert "undefined" in result["z"]

def test_build_system_message_empty():
    msg = build_system_message([], {})
    assert "Jupyter notebook" in msg

def test_build_system_message_with_cells():
    cells = [
        {"cell_num": 1, "source": "x = 1", "output": None},
        {"cell_num": 2, "source": "x + 1", "output": "2"},
    ]
    msg = build_system_message(cells, {})
    assert "x = 1" in msg
    assert "x + 1" in msg
    assert "2" in msg

def test_build_system_message_with_variables():
    msg = build_system_message([], {"data": "[1, 2, 3]"})
    assert "data" in msg
    assert "[1, 2, 3]" in msg
