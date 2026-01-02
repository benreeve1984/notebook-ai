from typing import Optional

def get_execution_history(shell) -> list[dict]:
    """
    Get executed cells with outputs from IPython history.

    Returns list of:
        {"cell_num": int, "source": str, "output": str | None}
    """
    cells = []

    # Get input history
    for i, source in enumerate(shell.user_ns.get('In', []), start=0):
        if not source or i == 0:  # Skip empty and cell 0
            continue

        output = None
        if 'Out' in shell.user_ns and i in shell.user_ns['Out']:
            output = safe_repr(shell.user_ns['Out'][i])

        cells.append({
            "cell_num": i,
            "source": source.strip(),
            "output": output
        })

    return cells

def safe_repr(obj, max_length: int = 2000) -> str:
    """Safe string representation, truncated if needed."""
    try:
        r = repr(obj)
        if len(r) > max_length:
            return r[:max_length] + f"... [truncated, {len(r)} chars total]"
        return r
    except Exception as e:
        return f"<repr failed: {e}>"

def resolve_variables(var_names: list[str], namespace: dict) -> dict[str, str]:
    """
    Resolve variable names to their string representations.

    Returns dict of {name: repr_string}
    """
    resolved = {}
    for name in var_names:
        if name in namespace:
            resolved[name] = safe_repr(namespace[name])
        else:
            resolved[name] = f"<undefined: {name} not found in namespace>"
    return resolved

def build_system_message(cells: list[dict], variables: dict[str, str], code_only: bool = False) -> str:
    """
    Build the system message for the LLM.

    Args:
        cells: Notebook execution history
        variables: Resolved variable values
        code_only: If True, instruct LLM to return only code without explanation
    """
    if code_only:
        parts = [
            "You are a coding assistant in a Jupyter notebook.",
            "Return ONLY the improved/requested code in a single Python code block.",
            "Do not include any explanation, comments about changes, or surrounding text.",
            "The code should be complete and ready to copy-paste directly into a cell.",
            ""
        ]
    else:
        parts = [
            "You are a helpful assistant working in a Jupyter notebook.",
            "You can see the code that has been executed and help the user analyze their data.",
            "",
            "When providing code improvements or examples:",
            "- Put code in ```python fenced blocks",
            "- Make code complete and runnable (not snippets)",
            "- Keep explanations concise",
            ""
        ]

    # Add cell history
    if cells:
        parts.append("## Notebook execution history:\n")
        for cell in cells:
            parts.append(f"### In [{cell['cell_num']}]:")
            parts.append(f"```python\n{cell['source']}\n```")
            if cell['output']:
                parts.append(f"Out [{cell['cell_num']}]:")
                parts.append(f"```\n{cell['output']}\n```")
            parts.append("")

    # Add explicitly referenced variables
    if variables:
        parts.append("## Variables the user has given you access to:\n")
        for name, value in variables.items():
            parts.append(f"**`{name}`**:")
            parts.append(f"```\n{value}\n```")
            parts.append("")

    return "\n".join(parts)
