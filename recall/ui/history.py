from pathlib import Path
from textual.containers import VerticalScroll
from textual.widgets import Markdown


class ChatHistory(VerticalScroll):
    """Widget displaying the history of the conversation."""

    def __init__(self):
        super().__init__()

    CSS_PATH = Path("styles", "chat_history.tcss")

    def add_message(self, role: str, message: str):
        self.mount(Message(role, message))


class Message(Markdown):
    def __init__(self, role: str, message: str):
        super().__init__(f"**{role}**: {message}")
