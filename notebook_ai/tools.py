from typing import Callable, Any
from functools import wraps

_TOOL_REGISTRY: dict[str, Callable] = {}
_VERBOSE_REGISTRATION: bool = True  # Print when tools are registered


def tool(func: Callable) -> Callable:
    """
    Register a function as an LLM-callable tool.

    Usage:
        @tool
        def analyze(df: str) -> str:
            '''Analyze a dataframe and return insights.'''
            return df.describe().to_string()
    """
    _TOOL_REGISTRY[func.__name__] = func

    if _VERBOSE_REGISTRATION:
        print(f"✓ Tool registered: {func.__name__}")

    @wraps(func)
    def wrapper(*args, **kwargs):
        return func(*args, **kwargs)

    return wrapper


def set_verbose(verbose: bool = True):
    """Enable or disable tool registration messages."""
    global _VERBOSE_REGISTRATION
    _VERBOSE_REGISTRATION = verbose

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
