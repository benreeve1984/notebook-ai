from IPython.display import display, Markdown, clear_output

class ResponseDisplay:
    """
    Handles rendering LLM responses.

    Uses clear_output + display for final output to ensure proper
    serialization in saved notebooks (for GitHub rendering etc).
    """

    def __init__(self, show_thinking: bool = False):
        self.content = ""
        self.show_thinking = show_thinking
        self._started = False

    def start(self):
        """Initialize display with thinking indicator."""
        self._started = True
        print("Thinking...", flush=True)

    def update(self, chunk: str):
        """Update with new content (for future streaming)."""
        self.content += chunk
        # For now, accumulate content. Streaming will enhance this.

    def show_tool_call(self, tool_name: str, args: dict):
        """Show tool being called (optional transparency)."""
        tool_msg = f"\n\n---\n*Calling `{tool_name}({args})`*\n---\n\n"
        self.content += tool_msg

    def show_tool_result(self, tool_name: str, result: str, truncate: int = 500):
        """Show tool result (optional transparency)."""
        if len(result) > truncate:
            result = result[:truncate] + "..."
        result_msg = f"\n*Result from `{tool_name}`:*\n```\n{result}\n```\n"
        self.content += result_msg

    def finish(self, content: str):
        """Finalize with complete content. Clears 'Thinking...' and displays result."""
        self.content = content
        if self._started:
            clear_output(wait=True)
        display(Markdown(content))

    def error(self, message: str):
        """Display error message."""
        error_content = f"**Error:** {message}"
        if self._started:
            clear_output(wait=True)
        display(Markdown(error_content))
