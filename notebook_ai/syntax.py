import re
from typing import Tuple, List

# Pattern matches {word} but not {{escaped}}
VAR_PATTERN = re.compile(r'(?<!\{)\{(\w+)\}(?!\})')

def parse_references(prompt: str, tool_registry: set[str]) -> Tuple[List[str], List[str]]:
    """
    Extract variable and tool references from prompt.

    Args:
        prompt: The user's prompt text
        tool_registry: Set of registered tool names

    Returns:
        Tuple of (variable_names, tool_names)

    Example:
        >>> parse_references("Analyze {df} with {summarize}", {"summarize"})
        (['df'], ['summarize'])
    """
    refs = VAR_PATTERN.findall(prompt)
    variables = [r for r in refs if r not in tool_registry]
    tools = [r for r in refs if r in tool_registry]
    return variables, tools
