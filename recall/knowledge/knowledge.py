from __future__ import annotations

from collections.abc import Callable, Generator
from os import PathLike
from pathlib import Path
import pickle
from pickle import PickleError
from typing import TYPE_CHECKING, Any

import ollama
from ollama import Message
from loguru import logger

from recall.knowledge.vectors import VectorDb

if TYPE_CHECKING:
    from recall.knowledge.types import Chunk, Dataset, Embedding

HISTORY_SIZE_THRESHOLD = 128
SAVE_HISTORY_THRESHOLD = 4
RETAIN_EXCHANGES = 12


def pack(path: Path, o: Any) -> None:
    path = Path(path)
    with path.with_suffix(".pickle").open("wb") as f:
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)


def unpack_or_load[T](path: Path | PathLike, callback: Callable[[Path], T]) -> T:
    """Unpack an archive, or generate one if it doesn't exist."""
    path = Path(path)
    pickled_path = path.with_suffix(".pickle")
    if pickled_path.exists():
        try:
            with pickled_path.open("rb") as f:
                r = pickle.load(f)
                return r
        except Exception as e:
            logger.error("Could not load {}; recreating data: {}", pickled_path, e)
    obj = callback(path)
    pack(path, obj)
    return obj


def load_dataset(source: Path) -> Dataset:
    source = Path(source)
    with open(source, "r") as f:
        return f.readlines()


class Conversation:
    """The conversation with the knowledge agent."""

    def __init__(
        self,
        name: str,
        *,
        embedding_model: str,
        language_model: str,
    ):
        self.name = name
        self.vector_db: VectorDb
        self.embedding_model = embedding_model
        self.language_model = language_model

        self.context: list[Message] = unpack_or_load("history", lambda _path: [])
        self.messageno = 0

    def load_dataset(self, dataset_path: PathLike):
        """Load the given dataset."""
        self.vector_db = VectorDb(dataset_path.name, self.embedding_model).with_dir(
            dataset_path
        )
        logger.info("Vector database loaded.")

    def ask(self, question: str) -> Generator[str, None, None]:
        """Get the answer to the given question."""
        logger.info("Answering {!r}.", question)

        question_message = Message(role="user", content=question)
        self.context.append(question_message)

        knowledge = list(self.vector_db.query(question, n=3))
        logger.debug(
            "Retrieved knowledge:\n{}",
            "\n".join(f"- ({similarity}) {chunk}" for chunk, similarity in knowledge),
        )
        instructions = (
            "You are an unbiased research assistant. "
            "Respond to the question using the following context. "
            "If you do not know the answer, do not invent an answer. "
            "Keep your response to a couple sentences. "
            "Do not invent any new information unless specifically requested: "
            # ) + "\n".join([f" - {chunk}" for chunk in knowledge])
        ) + "\n".join([f" - {chunk}" for chunk, _similarity in knowledge])
        logger.debug("Submitting request.")
        stream = ollama.chat(
            model=self.language_model,
            messages=self.context
            + [
                Message(role="system", content=instructions),
                question_message,
            ],
            stream=True,
        )
        response = ""
        for chunk in stream:
            logger.trace("Streaming response: {!r}", chunk)
            response += chunk.message.content
            yield chunk.message.content
        self.context.append(Message(role="assistant", content=response))
        self.messageno += 1
        if self.messageno % SAVE_HISTORY_THRESHOLD:
            pack(Path(self.name + ".history"), self.context)
        if self.messageno > HISTORY_SIZE_THRESHOLD:
            self.compact_history()
            self.messageno %= HISTORY_SIZE_THRESHOLD

    def compact_history(self):
        """Compact the history."""
        logger.info("Compacting the history...")
        compacted = ollama.chat(
            model=self.language_model,
            messages=self.context
            + [
                Message(
                    role="user",
                    content="Summarize the above interactions between yourself (the agent) and the user. "
                    "Retain as much information as possible. "
                    "Keep the summary brief. "
                    "Frame the summary as notes to yourself.",
                ),
            ],
            stream=False,
        )
        compacted.message.role = "system"
        self.context = [
            *self.context[-RETAIN_EXCHANGES * 2 :],
            Message(
                role="system",
                content="You may reference this summary of the history of this conversation:",
            ),
            compacted.message,
        ]
        logger.info("Compacted history:\n{!r}", self.context)
        # Pack the context for future reference.
        pack("history", self.context)
