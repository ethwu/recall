from typing import Annotated

from loguru import logger
import typer
from typer import Option

from recall.models import EmbeddingModel, LanguageModel
from recall.ui.interface import Interface

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)


@app.command()
def chat(
    embedding_model: Annotated[
        EmbeddingModel,
        Option("-e", "--embedding-model", help="The embedding model to use."),
    ] = EmbeddingModel.NOMIC_EMBED_TEXT,
    language_model: Annotated[
        LanguageModel,
        Option("-l", "--language-model", help="The language model to use."),
    ] = LanguageModel.PHI4_MINI,
    quiet: Annotated[
        bool,
        Option("-q", "--quiet", help="Suppress logging output."),
    ] = False,
) -> int:
    """Run the chat."""
    logger.remove()
    if not quiet:
        logger.add("recall.log", rotation="1 MB", retention="10 days")
    interface = Interface(
        embedding_model=embedding_model, language_model=language_model
    )
    interface.run()
    return interface.return_code or 0


def main(argv: list[str]) -> int:
    logger.info("started with arguments {!r}", argv)
    app(argv)
    return 0
