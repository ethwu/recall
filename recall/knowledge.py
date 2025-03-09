from collections.abc import Callable, Generator, Iterable
from os import PathLike
from pathlib import Path
import pickle
from pickle import PickleError
from typing import Any

import ollama
from ollama import Message
from loguru import logger

Dataset = Iterable[str]

Chunk = str
Embedding = list[float]

Vector = tuple[Chunk, Embedding]


def cosine_similarity(a, b):
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x**2 for x in a]) ** 0.5
    norm_b = sum([x**2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


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
        except (EOFError, IOError, PickleError) as e:
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
        *,
        embedding_model: str,
        language_model: str,
    ):
        self.vector_db: dict[Chunk, Embedding] = {}
        self.embedding_model = embedding_model
        self.language_model = language_model

        self.context: list[Message] = unpack_or_load("history", lambda _path: [])
        self.messageno = 0

    def embedding(self, text: str) -> Embedding:
        """Get an embedding for the given text."""
        logger.debug("Generating an embedding for {!r}.", text)
        embedding = ollama.embed(model=self.embedding_model, input=text)
        logger.debug("Got {!r}", embedding.embeddings)
        return (
            []
            if embedding is None or "embeddings" not in embedding
            else embedding["embeddings"][0]
        )

    def load_dataset(self, dataset_path: PathLike):
        """Load the given dataset."""

        def load(dataset_path: PathLike):
            dataset = load_dataset(dataset_path)
            dataset_length = len(dataset)
            vector_db = {}
            for i, chunk in enumerate(dataset):
                vector_db[chunk] = self.embedding(chunk)
                logger.info(
                    "Added chunk {} of {} to the database.", i + 1, dataset_length
                )
            return vector_db

        logger.info("Loading dataset from {!r}.", dataset_path)
        self.vector_db = unpack_or_load(Path(dataset_path), load)
        logger.info("Vector database loaded.")

    def retrieve(self, query: str, count: int = 3):
        """Get the response for the given query."""
        # Get the embedding for the query.
        query = self.embedding(query)
        logger.info("{}", len(self.vector_db))
        logger.debug("Got embedding {!r}", query)
        similarities = [
            (chunk, cosine_similarity(query, embedding))
            for chunk, embedding in self.vector_db.items()
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:count]

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
        if self.messageno > 3:
            self.compact_history()
            self.messageno %= 3

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
            Message(
                role="system",
                content="You may reference this summary of the history of this conversation:",
            ),
            compacted.message,
        ]
        logger.info("Compacted history:\n{}", compacted.message.content)
        # Pack the context for future reference.
        pack("history", self.context)
