from IPython.display import display, Markdown, update_display
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
        tool_msg = f"\n\n---\n*Calling `{tool_name}({args})`*\n---\n\n"
        self.content += tool_msg
        if self._started:
            update_display(Markdown(self.content), display_id=self.display_id)

    def show_tool_result(self, tool_name: str, result: str, truncate: int = 500):
        """Show tool result (optional transparency)."""
        if len(result) > truncate:
            result = result[:truncate] + "..."
        result_msg = f"\n*Result from `{tool_name}`:*\n```\n{result}\n```\n"
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
