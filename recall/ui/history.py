from pathlib import Path
from textual.containers import VerticalScroll
from textual.widgets import Markdown


class ChatHistory(VerticalScroll):
    """Widget displaying the history of the conversation."""

    def __init__(self):
        super().__init__()
        self.styles.align_vertical = "bottom"
        self.current_id = 0

    def add_message(self, role: str, message: str | None = "") -> int:
        """Add a message to the history."""
        if message is None:
            return
        identifier = self.current_id
        match role:
            case "User":
                message = Prompt(role, identifier, message)
            case "Agent":
                message = Response(role, identifier, message)
            case _:
                message = Message(role, identifier, message)
        self.mount(message)
        message.anchor()
        self.current_id += 1
        return identifier

    def update_message(self, identifier: int, text: str | None):
        """Update the message with the given identifier."""
        if text is None:
            return
        message = self.query_one(f"#message-{identifier}")
        message.add_text(text)
        message.anchor()


class Message(Markdown):
    def __init__(self, role: str, identifier: int, message: str):
        super().__init__(message, id=f"message-{identifier}")
        self.role = role
        self.message = message

    def add_text(self, text: str | None):
        """Add text to this message."""
        if text is None:
            return
        self.message += text
        self.update(self.message)


class Prompt(Message):
    pass


class Response(Message):
    BORDER_TITLE = "Assistant"
