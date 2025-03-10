import logging
from os import PathLike
from pathlib import Path
import sys
from typing import Annotated

from loguru import logger
import typer
from typer import Argument, Option

from recall.models import EmbeddingModel, LanguageModel
from recall.ui.interface import Interface

app = typer.Typer(pretty_exceptions_show_locals=False, pretty_exceptions_enable=False)


@app.command()
def chat(
    source: Annotated[Path, Argument(help="The source directory to chat with.")],
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
        logger.add(
            "recall.log", level=logging.DEBUG, rotation="1 GB", retention="15 minutes"
        )
    interface = Interface(
        source, embedding_model=embedding_model, language_model=language_model
    )
    interface.run()
    return interface.return_code or 0


def main(argv: list[str]) -> int:
    logger.info("started with arguments {!r}", argv)
    app(argv)
    return 0
