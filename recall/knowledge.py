from collections.abc import Callable, Generator, Iterable
from os import PathLike
from pathlib import Path
import pickle
from typing import TypedDict

import numpy as np
from numpy.typing import ArrayLike
import ollama
from ollama import Message
from loguru import logger


Dataset = Iterable[str]

Chunk = str
Embedding = ArrayLike

Vector = tuple[Chunk, Embedding]


def unpack_or_load[T](path: Path | PathLike, callback: Callable[[Path], T]) -> T:
    """Unpack an archive, or generate one if it doesn't exist."""
    path = Path(path)
    pickled_path = path.with_suffix(".pickle")
    if pickled_path.exists():
        with pickled_path.open("rb") as f:
            return pickle.load(f)
    else:
        obj = callback(path)
        with pickled_path.open("wb") as f:
            pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        return obj


def load_dataset(source: Path) -> Dataset:
    source = Path(source)
    with open(source, "r") as f:
        return f.readlines()


class Conversation:
    """The conversation with the knowledge agent."""

    def __init__(
        self,
        *,
        embedding_model: str,
        language_model: str,
    ):
        self.vector_db: list[Vector] = []
        self.embedding_model = embedding_model
        self.language_model = language_model

        self.context: list[Message] = unpack_or_load("history", lambda _path: [])

    def embedding(self, text: str) -> Embedding:
        """Get an embedding for the given text."""
        logger.debug("Generating an embedding for {!r}.", text)
        embedding = ollama.embed(model=self.embedding_model, input=text)
        return np.array(
            []
            if embedding is None or "embeddings" not in embedding
            else embedding["embeddings"][0]
        )

    def load_dataset(self, dataset_path: PathLike):
        """Load the given dataset."""

        def load(dataset_path: PathLike):
            dataset = load_dataset(dataset_path)
            dataset_length = len(dataset)
            for i, chunk in enumerate(dataset):
                self.vector_db.append((chunk, self.embedding(chunk)))
                logger.info(
                    "Added chunk {} of {} to the database.", i + 1, dataset_length
                )

        unpack_or_load(Path(dataset_path), load)

    def retrieve(self, query: str, count: int = 3):
        """Get the response for the given query."""
        # Get the embedding for the query.
        query = self.embedding(query)
        return (
            [
                (
                    chunk,
                    # Calculate the similarities.
                    np.dot(query, embedding)
                    / (np.linalg.norm(query) * np.linalg.norm(embedding)),
                )
                for chunk, embedding in self.vector_db
            ].sort(key=lambda x: x[1], reverse=True)
            or []
        )[:count]

    def ask(self, question: str) -> Generator[str, None, None]:
        """Get the answer to the given question."""
        logger.info("Answering {!r}.", question)

        question_message = Message(role="user", content=question)
        self.context.append(question_message)

        knowledge = self.retrieve(question)
        logger.debug(
            "Retrieved knowledge:\n{}",
            "\n".join(
                "- (similarity: {:.2f}) {}".format(similarity, chunk)
                for chunk, similarity in knowledge
            ),
        )
        instructions = (
            "You are an unbiased assistant. "
            "Respond to the question using this context. "
            "Do not make up any new information: "
        ) + "\n".join([f" - {chunk}" for chunk, _similarity in knowledge])
        logger.debug("Submitting request.")
        stream = ollama.chat(
            model=self.language_model,
            messages=self.context
            + [
                {"role": "system", "content": instructions},
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
