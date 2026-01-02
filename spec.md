# Claude Code Task: Build `notebook-ai`

## Overview

Build a Python package that adds a `%%prompt` IPython magic command for LLM-assisted notebook workflows. This is a lightweight alternative to SolveIt - pure Python, no frontend/TypeScript.

**LLM Backend**: Use Claudette (https://github.com/AnswerDotAI/claudette) with Claude Opus 4.5 (`claude-opus-4-5-20250514`).

## Project Structure

Create this exact structure:

```
notebook-ai/
├── notebook_ai/
│   ├── __init__.py           # Package init, expose key imports
│   ├── magic.py              # IPython magic registration
│   ├── context.py            # Build context from notebook state
│   ├── syntax.py             # Parse {var} references from prompts
│   ├── tools.py              # @tool decorator and registry
│   └── display.py            # Output rendering (streaming-ready)
├── tests/
│   ├── __init__.py
│   ├── test_syntax.py
│   ├── test_tools.py
│   └── test_context.py
├── examples/
│   └── demo.ipynb            # Working demo notebook
├── pyproject.toml
├── README.md
├── .gitignore
└── .env.example
```

---

## Detailed Specifications

### 1. `notebook_ai/syntax.py`

Parse `{variable_name}` references from prompt text.

```python
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
```

### 2. `notebook_ai/tools.py`

Tool registration and execution with auto-resolution from namespace.

```python
import inspect
from typing import Callable, Any, get_type_hints
from functools import wraps

_TOOL_REGISTRY: dict[str, Callable] = {}

def tool(func: Callable) -> Callable:
    """
    Register a function as an LLM-callable tool.
    
    Usage:
        @tool
        def analyze(df: "pd.DataFrame") -> str:
            '''Analyze a dataframe and return insights.'''
            return df.describe().to_string()
    """
    _TOOL_REGISTRY[func.__name__] = func
    
    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)
    
    return wrapper

def get_tool_names() -> set[str]:
    """Return set of registered tool names."""
    return set(_TOOL_REGISTRY.keys())

def get_tool_functions(tool_names: list[str]) -> list[Callable]:
    """
    Get the actual function objects for specified tools.
    Used for Claudette's tool parameter.
    """
    return [_TOOL_REGISTRY[name] for name in tool_names if name in _TOOL_REGISTRY]

def clear_registry():
    """Clear tool registry (useful for testing)."""
    _TOOL_REGISTRY.clear()
```

**Important**: Claudette handles tool schemas and execution internally. We just need to pass it the function objects. Claudette will:
- Generate tool schemas from function signatures/docstrings
- Handle the tool call loop
- Execute tools with arguments

For argument resolution (passing actual objects instead of string names), we need a wrapper approach:

```python
def create_namespace_resolver(namespace: dict) -> Callable:
    """
    Create a function that wraps tool execution to resolve 
    string arguments from namespace.
    """
    def resolve_and_execute(func: Callable, **kwargs) -> Any:
        resolved = {}
        for key, value in kwargs.items():
            if isinstance(value, str) and value in namespace:
                resolved[key] = namespace[value]
            else:
                resolved[key] = value
        return func(**resolved)
    return resolve_and_execute
```

### 3. `notebook_ai/context.py`

Build context string from notebook execution history.

```python
from IPython import get_ipython
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
        if f'Out' in shell.user_ns and i in shell.user_ns['Out']:
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

def build_system_message(cells: list[dict], variables: dict[str, str]) -> str:
    """
    Build the system message for the LLM.
    """
    parts = [
        "You are a helpful assistant working in a Jupyter notebook.",
        "You can see the code that has been executed and help the user analyze their data.",
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
```

### 4. `notebook_ai/display.py`

Output rendering, structured for future streaming support.

```python
from IPython.display import display, Markdown, update_display, HTML
import uuid

class ResponseDisplay:
    """
    Handles rendering LLM responses.
    Designed to support streaming in future.
    """
    
    def __init__(self, show_thinking: bool = False):
        self.display_id = str(uuid.uuid4())
        self.content = ""
        self.show_thinking = show_thinking
        self._started = False
    
    def start(self):
        """Initialize display with thinking indicator."""
        self._started = True
        display(Markdown("*Thinking...*"), display_id=self.display_id)
    
    def update(self, chunk: str):
        """Update with new content (for future streaming)."""
        self.content += chunk
        if self._started:
            update_display(Markdown(self.content), display_id=self.display_id)
    
    def show_tool_call(self, tool_name: str, args: dict):
        """Show tool being called (optional transparency)."""
        tool_msg = f"\n\n---\n*🔧 Calling `{tool_name}({args})`*\n---\n\n"
        self.content += tool_msg
        if self._started:
            update_display(Markdown(self.content), display_id=self.display_id)
    
    def show_tool_result(self, tool_name: str, result: str, truncate: int = 500):
        """Show tool result (optional transparency)."""
        if len(result) > truncate:
            result = result[:truncate] + "..."
        result_msg = f"\n*Result from `{tool_name)}`:*\n```\n{result}\n```\n"
        self.content += result_msg
        if self._started:
            update_display(Markdown(self.content), display_id=self.display_id)
    
    def finish(self, content: str):
        """Finalize with complete content."""
        self.content = content
        if self._started:
            update_display(Markdown(content), display_id=self.display_id)
        else:
            display(Markdown(content))
    
    def error(self, message: str):
        """Display error message."""
        error_content = f"**Error:** {message}"
        if self._started:
            update_display(Markdown(error_content), display_id=self.display_id)
        else:
            display(Markdown(error_content))
```

### 5. `notebook_ai/magic.py`

The main magic command implementation using Claudette.

```python
from IPython.core.magic import Magics, magics_class, cell_magic
from IPython import get_ipython

from .context import get_execution_history, resolve_variables, build_system_message
from .syntax import parse_references
from .tools import get_tool_names, get_tool_functions, _TOOL_REGISTRY
from .display import ResponseDisplay

# Claudette imports
from claudette import Chat, models

# Default model
DEFAULT_MODEL = "claude-opus-4-5-20250514"

@magics_class
class PromptMagic(Magics):
    """IPython magic for LLM-assisted notebook workflows."""
    
    def __init__(self, shell):
        super().__init__(shell)
        self.model = DEFAULT_MODEL
        self.show_tool_calls = True  # Show tool execution in output
    
    @cell_magic
    def prompt(self, line, cell):
        """
        Send prompt to LLM with notebook context.
        
        Usage:
            %%prompt
            What patterns do you see in {my_dataframe}?
            
            %%prompt model=claude-sonnet-4-20250514
            Analyze {data} using {my_analysis_tool}
            
        Options:
            model=MODEL_NAME    Use specific model
            quiet               Don't show tool calls
        """
        # Parse options from magic line
        options = self._parse_options(line)
        model = options.get("model", self.model)
        show_tools = "quiet" not in options
        
        # Initialize display
        disp = ResponseDisplay()
        disp.start()
        
        try:
            # Get notebook context
            cells = get_execution_history(self.shell)
            
            # Parse variable and tool references
            var_names, tool_names = parse_references(cell, get_tool_names())
            
            # Resolve variables
            variables = resolve_variables(var_names, self.shell.user_ns)
            
            # Build system message
            system = build_system_message(cells, variables)
            
            # Get tool functions for Claudette
            tools = get_tool_functions(tool_names) if tool_names else None
            
            # Create chat and get response
            chat = Chat(model=model, sp=system, tools=tools)
            
            # Use toolloop for automatic tool execution
            # We need to handle namespace resolution for tool arguments
            if tools:
                response = self._run_with_tools(
                    chat, cell, tool_names, disp, show_tools
                )
            else:
                response = chat(cell)
            
            # Display final response
            disp.finish(response.content if hasattr(response, 'content') else str(response))
            
        except Exception as e:
            disp.error(f"{type(e).__name__}: {e}")
            raise
    
    def _run_with_tools(self, chat, prompt, tool_names, disp, show_tools):
        """
        Run prompt with tool loop, resolving namespace arguments.
        """
        # For namespace resolution, we wrap tool execution
        # Claudette's toolloop handles the iteration
        
        namespace = self.shell.user_ns
        
        # Custom tool execution that resolves namespace references
        def resolve_arg(value):
            if isinstance(value, str) and value in namespace:
                return namespace[value]
            return value
        
        # Monkey-patch the registered tools to resolve arguments
        # This is a pragmatic approach - cleaner would be Claudette middleware
        original_tools = {}
        for name in tool_names:
            if name in _TOOL_REGISTRY:
                original_func = _TOOL_REGISTRY[name]
                original_tools[name] = original_func
                
                def make_resolver(func):
                    def resolved_func(**kwargs):
                        resolved_kwargs = {k: resolve_arg(v) for k, v in kwargs.items()}
                        return func(**resolved_kwargs)
                    resolved_func.__name__ = func.__name__
                    resolved_func.__doc__ = func.__doc__
                    resolved_func.__annotations__ = getattr(func, '__annotations__', {})
                    return resolved_func
                
                _TOOL_REGISTRY[name] = make_resolver(original_func)
        
        try:
            # Run with toolloop
            response = chat.toolloop(prompt, trace_func=self._make_tracer(disp, show_tools))
            return response
        finally:
            # Restore original tools
            for name, func in original_tools.items():
                _TOOL_REGISTRY[name] = func
    
    def _make_tracer(self, disp, show_tools):
        """Create a trace function for toolloop to show progress."""
        if not show_tools:
            return None
        
        def tracer(tool_name, args, result):
            disp.show_tool_call(tool_name, args)
        
        return tracer
    
    def _parse_options(self, line: str) -> dict:
        """Parse magic line options."""
        options = {}
        for part in line.split():
            if "=" in part:
                key, value = part.split("=", 1)
                options[key.strip()] = value.strip()
            else:
                options[part.strip()] = True
        return options


def load_ipython_extension(ipython):
    """Called when user runs %load_ext notebook_ai"""
    ipython.register_magics(PromptMagic)


def unload_ipython_extension(ipython):
    """Called when user runs %unload_ext notebook_ai"""
    pass  # Magics are automatically cleaned up
```

### 6. `notebook_ai/__init__.py`

```python
"""
notebook-ai: LLM-powered notebook assistant

Usage:
    %load_ext notebook_ai
    
    @tool
    def my_function(x):
        '''Description for LLM.'''
        return x * 2
    
    %%prompt
    Analyze {my_data} using {my_function}
"""

from .tools import tool, clear_registry
from .magic import load_ipython_extension, unload_ipython_extension

__version__ = "0.1.0"
__all__ = ["tool", "load_ipython_extension", "unload_ipython_extension"]
```

---

## Configuration Files

### `pyproject.toml`

```toml
[project]
name = "notebook-ai"
version = "0.1.0"
description = "LLM-powered notebook assistant using Claudette"
readme = "README.md"
requires-python = ">=3.10"
license = {text = "MIT"}
authors = [
    {name = "Your Name", email = "your.email@oliverwyman.com"}
]
dependencies = [
    "ipython>=8.0",
    "claudette>=0.1.0",
]

[project.optional-dependencies]
dev = [
    "pytest>=7.0",
    "pytest-cov",
    "jupyter",
    "pandas",  # For demo/testing
]

[build-system]
requires = ["hatchling"]
build-backend = "hatchling.build"

[tool.pytest.ini_options]
testpaths = ["tests"]
python_files = ["test_*.py"]
```

### `.gitignore`

```gitignore
# Python
__pycache__/
*.py[cod]
*$py.class
*.so
.Python
build/
develop-eggs/
dist/
downloads/
eggs/
.eggs/
lib/
lib64/
parts/
sdist/
var/
wheels/
*.egg-info/
.installed.cfg
*.egg

# Virtual environments
.venv/
venv/
ENV/

# IDE
.idea/
.vscode/
*.swp
*.swo

# Jupyter
.ipynb_checkpoints/
*.ipynb_checkpoints

# Environment
.env
.env.local

# Testing
.pytest_cache/
.coverage
htmlcov/

# OS
.DS_Store
Thumbs.db
```

### `.env.example`

```bash
# Copy to .env and fill in your API key
ANTHROPIC_API_KEY=sk-ant-...
```

### `README.md`

```markdown
# notebook-ai

LLM-powered notebook assistant using Claudette and Claude Opus 4.5.

A lightweight alternative to SolveIt - pure Python, no frontend complexity.

## Installation

```bash
# Install from GitHub
pip install "git+https://github.com/your-org/notebook-ai.git"

# Or for development
git clone https://github.com/your-org/notebook-ai.git
cd notebook-ai
pip install -e ".[dev]"
```

## Setup

Set your Anthropic API key:

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
```

Or create a `.env` file (if using python-dotenv).

## Quick Start

```python
# Load the extension
%load_ext notebook_ai

# Import the tool decorator
from notebook_ai import tool

# Define your data
import pandas as pd
sales = pd.DataFrame({
    'product': ['A', 'B', 'C'],
    'revenue': [100, 200, 150]
})

# Define a tool (optional)
@tool
def summarize(df: pd.DataFrame) -> str:
    '''Summarize a dataframe with descriptive statistics.'''
    return df.describe().to_string()

# Use the magic!
%%prompt
What's the total revenue in {sales}? 
Use {summarize} to get more details.
```

## Syntax

### Variable References

Use `{variable_name}` to give the LLM access to a specific variable:

```python
%%prompt
What patterns do you see in {my_dataframe}?
```

The LLM sees the variable's string representation and can reason about it.

### Tool References

Register functions with `@tool`, then reference them with `{function_name}`:

```python
@tool
def calculate_metrics(df: pd.DataFrame) -> dict:
    '''Calculate key business metrics from a dataframe.'''
    return {
        'total': df['value'].sum(),
        'mean': df['value'].mean(),
        'count': len(df)
    }

%%prompt
Analyze {sales_data} using {calculate_metrics}
```

The LLM can call the tool and will receive the actual DataFrame object (not just the name).

### Magic Options

```python
%%prompt model=claude-sonnet-4-20250514
Use a different model for this prompt

%%prompt quiet
Don't show tool execution details
```

## How It Works

1. **Context Building**: The magic collects all executed cells and their outputs
2. **Reference Parsing**: `{var}` patterns are extracted from your prompt
3. **Variable Resolution**: Referenced variables are converted to string representations
4. **Tool Preparation**: `@tool` functions are made available to the LLM
5. **LLM Call**: Claudette sends the prompt with context and tools to Claude
6. **Tool Loop**: If Claude calls tools, they execute with resolved arguments
7. **Response**: Final response is rendered as Markdown in the cell output

## Differences from SolveIt

| Feature | SolveIt | notebook-ai |
|---------|---------|----------------|
| Architecture | Full JupyterLab extension | IPython magic (pure Python) |
| Syntax | `$\`var\`` and `&\`func\`` | `{var}` and `{func}` |
| Cell type | Custom "prompt" cells | Standard code cells with `%%prompt` |
| Frontend | Custom TypeScript | None |
| Complexity | High | Low |

## Development

```bash
# Clone and install
git clone https://github.com/your-org/notebook-ai.git
cd notebook-ai
pip install -e ".[dev]"

# Run tests
pytest

# Run with coverage
pytest --cov=notebook_ai
```

## License

MIT
```

---

## Test Files

### `tests/test_syntax.py`

```python
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
```

### `tests/test_tools.py`

```python
import pytest
from notebook_ai.tools import tool, get_tool_names, get_tool_functions, clear_registry

@pytest.fixture(autouse=True)
def clean_registry():
    """Clear tool registry before each test."""
    clear_registry()
    yield
    clear_registry()

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
```

### `tests/test_context.py`

```python
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
```

---

## Demo Notebook

### `examples/demo.ipynb`

Create a Jupyter notebook with these cells:

**Cell 1 (Markdown):**
```markdown
# notebook-ai Demo

This notebook demonstrates the `%%prompt` magic for LLM-assisted workflows.
```

**Cell 2 (Code):**
```python
# Load the extension
%load_ext notebook_ai
from notebook_ai import tool
```

**Cell 3 (Code):**
```python
# Create some sample data
import pandas as pd

sales = pd.DataFrame({
    'product': ['Widget A', 'Widget B', 'Widget C', 'Widget D'],
    'units': [150, 89, 203, 67],
    'price': [25.99, 45.50, 12.99, 89.99],
    'region': ['North', 'South', 'North', 'East']
})

sales['revenue'] = sales['units'] * sales['price']
sales
```

**Cell 4 (Code):**
```python
# Define a tool
@tool
def top_products(df: pd.DataFrame, n: int = 3) -> str:
    """Return the top N products by revenue."""
    top = df.nlargest(n, 'revenue')[['product', 'revenue']]
    return top.to_string()
```

**Cell 5 (Code):**
```python
%%prompt
What's the total revenue in {sales}? Which region is performing best?
```

**Cell 6 (Code):**
```python
%%prompt
Use {top_products} to find the best performers in {sales}, then explain why they might be successful.
```

**Cell 7 (Markdown):**
```markdown
## Testing Different Models
```

**Cell 8 (Code):**
```python
%%prompt model=claude-sonnet-4-20250514
Give me a quick one-line summary of {sales}.
```

---

## Testing Instructions

After creating all files:

1. **Install the package:**
   ```bash
   cd notebook-ai
   pip install -e ".[dev]"
   ```

2. **Run unit tests:**
   ```bash
   pytest -v
   ```

3. **Set API key:**
   ```bash
   export ANTHROPIC_API_KEY="your-key-here"
   ```

4. **Test in Jupyter:**
   ```bash
   jupyter notebook examples/demo.ipynb
   ```

5. **Run each cell and verify:**
   - Extension loads without error
   - `@tool` decorator works
   - `%%prompt` sends to LLM and returns response
   - `{variable}` references are resolved
   - `{tool}` references enable tool calling
   - Tool results are incorporated into response

---

## Known Limitations to Document

1. **Streaming not yet implemented** - Responses appear all at once
2. **Tool argument resolution** - Arguments must be variable names in namespace, not expressions
3. **Context size** - Large notebooks may exceed context window; consider truncation
4. **History persistence** - Uses IPython's In/Out history, which resets on kernel restart

---

## Implementation Notes

- **Claudette integration**: The `Chat.toolloop()` method handles the tool execution loop. We wrap tools to resolve namespace arguments before execution.

- **trace_func**: Claudette's toolloop accepts a trace function for debugging. We use this to show tool calls in the output. Check Claudette's current API - this may be `trace_func`, `tracer`, or similar.

- **Error handling**: Wrap the main magic in try/except and use `disp.error()` for user-friendly errors.

- **Check Claudette API**: Before implementing, run `from claudette import Chat; help(Chat)` and `help(Chat.toolloop)` to verify the exact method signatures. The API may have changed.
