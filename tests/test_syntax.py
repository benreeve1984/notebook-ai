import pytest
from notebook_ai.syntax import parse_references

def test_parse_single_variable():
    vars, tools = parse_references("What is {data}?", set())
    assert vars == ["data"]
    assert tools == []

def test_parse_multiple_variables():
    vars, tools = parse_references("Compare {df1} with {df2}", set())
    assert vars == ["df1", "df2"]
    assert tools == []

def test_parse_tool_reference():
    vars, tools = parse_references("Use {analyze} on {data}", {"analyze"})
    assert vars == ["data"]
    assert tools == ["analyze"]

def test_parse_no_references():
    vars, tools = parse_references("What is 2 + 2?", set())
    assert vars == []
    assert tools == []

def test_parse_escaped_braces():
    # {{escaped}} should not be parsed
    vars, tools = parse_references("Format: {{not_a_var}} but {real_var}", set())
    assert vars == ["real_var"]
    assert tools == []

def test_parse_mixed():
    vars, tools = parse_references(
        "Analyze {sales} and {inventory} using {summarize} and {plot}",
        {"summarize", "plot"}
    )
    assert set(vars) == {"sales", "inventory"}
    assert set(tools) == {"summarize", "plot"}
