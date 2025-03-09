from textual.widgets import Input


class ChatInput(Input):
    """The input field for the chat."""

    def __init__(self):
        super().__init__(
            placeholder="Ask a question...",
        )

    def on_input_submitted(self, event: Input.Submitted):
        if not event.value:
            return
        self.loading = True
        self.clear()
