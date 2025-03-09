from pathlib import Path
from textual.containers import VerticalScroll
from textual.widgets import Markdown


class ChatHistory(VerticalScroll):
    """Widget displaying the history of the conversation."""

    def __init__(self):
        super().__init__()
        self.current_id = 0

    CSS_PATH = Path("styles", "chat_history.tcss")

    def add_message(self, role: str, message: str = "") -> int:
        """Add a message to the history."""
        identifier = self.current_id
        self.mount(Message(role, identifier, message))
        self.current_id += 1
        return identifier

    def update_message(self, identifier: int, text: str | None):
        """Update the message with the given identifier."""
        self.query_one(f"#message-{identifier}").add_text(text)


class Message(Markdown):
    def __init__(self, role: str, identifier: int, message: str):
        super().__init__(f"**{role}**: {message}", id=f"message-{identifier}")
        self.role = role
        self.message = message

    def add_text(self, text: str | None):
        """Add text to this message."""
        if text is None:
            return
        self.message += text
        self.update(f"**{self.role}**: {self.message}")
