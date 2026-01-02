import os
import hashlib
import json
from pathlib import Path
from functools import wraps

from IPython.core.magic import Magics, magics_class, cell_magic
from IPython import get_ipython
from dotenv import load_dotenv

from .context import get_execution_history, resolve_variables, build_system_message
from .syntax import parse_references
from .tools import get_tool_names, get_tool_functions, _TOOL_REGISTRY
from .display import ResponseDisplay

# Claudette imports
from claudette import Chat

# Default model
DEFAULT_MODEL = "claude-opus-4-5-20251101"

# Response cache: maps cache_key -> (content, tool_calls_made)
_RESPONSE_CACHE: dict[str, tuple[str, list]] = {}


def _make_cache_key(prompt: str, variables: dict[str, str], tool_names: list[str], model: str) -> str:
    """Create a cache key from prompt, variables, tools, and model."""
    cache_data = {
        'prompt': prompt,
        'variables': variables,
        'tools': sorted(tool_names),
        'model': model
    }
    cache_str = json.dumps(cache_data, sort_keys=True)
    return hashlib.sha256(cache_str.encode()).hexdigest()[:16]


def clear_cache():
    """Clear the response cache."""
    _RESPONSE_CACHE.clear()
    print("✓ Response cache cleared")


def cache_stats() -> dict:
    """Return cache statistics."""
    return {
        'entries': len(_RESPONSE_CACHE),
        'keys': list(_RESPONSE_CACHE.keys())
    }

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
            --no-cache          Force fresh API call, ignore cache
            --code              Return only code, no explanation (for code improvements)
        """
        # Parse options from magic line
        options = self._parse_options(line)
        model = options.get("model", self.model)
        show_tools = "quiet" not in options
        use_cache = "--no-cache" not in options and "no-cache" not in options
        code_only = "--code" in options or "code" in options

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

            # Check cache first
            cache_key = _make_cache_key(cell, variables, tool_names, model)
            if use_cache and cache_key in _RESPONSE_CACHE:
                content, tool_calls_made = _RESPONSE_CACHE[cache_key]
                # Add cached indicator
                cached_content = "*(cached response)*\n\n" + content
                disp.finish(cached_content)
                return

            # Build system message
            system = build_system_message(cells, variables, code_only=code_only)

            # Get tool functions and wrap them with namespace resolution
            if tool_names:
                tools = self._wrap_tools_with_resolver(tool_names)
            else:
                tools = None

            # Create chat and get response
            chat = Chat(model=model, sp=system, tools=tools)

            # Use toolloop for automatic tool execution
            tool_calls_made = []
            if tools:
                response = None
                for output in chat.toolloop(cell):
                    response = output
                    # Check for tool use in the response
                    if hasattr(output, 'content'):
                        for block in output.content:
                            if hasattr(block, 'name') and hasattr(block, 'input'):
                                # This is a tool use block
                                tool_calls_made.append({
                                    'name': block.name,
                                    'args': block.input
                                })

                # Build final content with tool call info
                content = self._extract_content(response)
                if tool_calls_made:
                    tool_section = "\n\n---\n**Tools called:**\n"
                    for tc in tool_calls_made:
                        args_str = ", ".join(f"{k}={repr(v)}" for k, v in tc['args'].items())
                        tool_section += f"- `{tc['name']}({args_str})`\n"
                    tool_section += "---\n\n"
                    content = tool_section + content
            else:
                response = chat(cell)
                content = self._extract_content(response)

            # Cache the response
            _RESPONSE_CACHE[cache_key] = (content, tool_calls_made)

            # Display final response
            disp.finish(content)

        except Exception as e:
            disp.error(f"{type(e).__name__}: {e}")
            raise

    def _wrap_tools_with_resolver(self, tool_names: list[str]) -> list:
        """
        Wrap tool functions with namespace resolution.

        This creates wrapper functions that resolve string variable names
        to actual objects from the notebook namespace before calling the tool.
        """
        namespace = self.shell.user_ns
        wrapped_tools = []

        for name in tool_names:
            if name not in _TOOL_REGISTRY:
                continue

            original_func = _TOOL_REGISTRY[name]

            # Create a wrapper that resolves namespace references
            def make_wrapper(func):
                @wraps(func)
                def wrapper(**kwargs):
                    resolved_kwargs = {}
                    for k, v in kwargs.items():
                        # If the value is a string that matches a variable name, resolve it
                        if isinstance(v, str) and v in namespace:
                            resolved_kwargs[k] = namespace[v]
                        else:
                            resolved_kwargs[k] = v
                    return func(**resolved_kwargs)
                return wrapper

            wrapped_tools.append(make_wrapper(original_func))

        return wrapped_tools if wrapped_tools else None

    def _extract_content(self, response) -> str:
        """Extract text content from a Claudette response."""
        if hasattr(response, 'content'):
            # Handle list of content blocks
            if isinstance(response.content, list):
                text_parts = []
                for block in response.content:
                    if hasattr(block, 'text'):
                        text_parts.append(block.text)
                    elif isinstance(block, str):
                        text_parts.append(block)
                return "\n".join(text_parts)
            return str(response.content)
        return str(response)

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


def _load_env():
    """Load .env file from current directory or parents."""
    # Try to find .env starting from cwd and going up
    cwd = Path.cwd()
    for directory in [cwd] + list(cwd.parents):
        env_file = directory / ".env"
        if env_file.exists():
            load_dotenv(env_file)
            return str(env_file)
        # Stop at home directory
        if directory == Path.home():
            break
    # Also try load_dotenv() default behavior
    load_dotenv()
    return None


def load_ipython_extension(ipython):
    """Called when user runs %load_ext notebook_ai"""
    env_file = _load_env()
    if env_file and not os.environ.get("ANTHROPIC_API_KEY"):
        print(f"Warning: Loaded {env_file} but ANTHROPIC_API_KEY not found")
    ipython.register_magics(PromptMagic)


def unload_ipython_extension(ipython):
    """Called when user runs %unload_ext notebook_ai"""
    pass  # Magics are automatically cleaned up
