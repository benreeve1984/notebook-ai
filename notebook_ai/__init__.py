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

from .tools import tool, clear_registry, set_verbose
from .magic import load_ipython_extension, unload_ipython_extension, clear_cache, cache_stats

__version__ = "0.1.0"
__all__ = ["tool", "set_verbose", "clear_cache", "cache_stats", "load_ipython_extension", "unload_ipython_extension"]
