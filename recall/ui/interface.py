from collections.abc import Generator

import textual
from textual.app import App, ComposeResult
from textual.screen import Screen
from textual.widgets import Input, LoadingIndicator, Markdown

from recall import knowledge
from recall.knowledge import Conversation
from recall.models import EmbeddingModel, LanguageModel
from recall.ui.history import ChatHistory
from recall.ui.input import ChatInput


class LoadingScreen(Screen):
    """Loading screen while the dataset loads."""

    def compose(self) -> ComposeResult:
        yield LoadingIndicator()


class Interface(App):
    """The user interface for the chat."""

    def __init__(self, embedding_model: EmbeddingModel, language_model: LanguageModel):
        super().__init__()
        self.convo = Conversation(
            embedding_model=embedding_model, language_model=language_model
        )
        self.chat_history = ChatHistory()
        self.chat_input = ChatInput()

    def on_mount(self) -> None:
        """Initialize the interface.

        Shows the loading screen. Slow-running work is delegated to
        `prepare`.
        """
        self.install_screen(LoadingScreen(), name="loading")
        self.push_screen("loading")
        self.prepare()
        self.set_focus(self.chat_input)

    @textual.work(thread=True)
    async def prepare(self) -> None:
        """Finish initializing the application in a separate thread."""
        self.convo.load_dataset(knowledge.load_dataset("cats.txt"))
        self.call_from_thread(self.pop_screen)

    def compose(self) -> ComposeResult:
        """Get the components for display."""
        yield self.chat_history
        yield self.chat_input

    @textual.on(Input.Submitted)
    async def submit_query(self, event: Input.Submitted):
        """Handle a query from the user."""
        self.chat_history.add_message("User", event.value)
        response = await self.response(event.value).wait()
        self.chat_history.add_message("Agent", response)
        self.chat_input.loading = False

    @textual.work(thread=True)
    async def response(self, query: str) -> Generator[str, None, None]:
        """Get the response to a `query` in a separate thread."""
        identifier = self.call_from_thread(
            lambda: self.chat_history.add_message("Agent")
        )
        for chunk in self.convo.ask(query):
            self.call_from_thread(
                lambda: self.chat_history.update_message(identifier, chunk)
            )
