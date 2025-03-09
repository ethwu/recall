from pathlib import Path
from typing import Iterable

import numpy as np
from numpy.typing import ArrayLike
import ollama
from loguru import logger


Dataset = Iterable[str]

Chunk = str
Embedding = ArrayLike

Vector = tuple[Chunk, Embedding]


def load_dataset(source: Path) -> Dataset:
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

    def embedding(self, text: str) -> Embedding:
        """Get an embedding for the given text."""
        logger.debug("Generating an embedding for {!r}.", text)
        embedding = ollama.embed(model=self.embedding_model, input=text)
        return np.array(
            []
            if embedding is None or "embeddings" not in embedding
            else embedding["embeddings"][0]
        )

    def load_dataset(self, dataset: Dataset):
        """Load the given dataset."""
        dataset_length = len(dataset)
        for i, chunk in enumerate(dataset):
            self.vector_db.append((chunk, self.embedding(chunk)))
            logger.info("Added chunk {} of {} to the database.", i + 1, dataset_length)

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

    def ask(self, question: str) -> str:
        """Get the answer to the given question."""
        logger.info("Answering {!r}.", question)
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
            messages=[
                {"role": "system", "content": instructions},
                {"role": "user", "content": question},
            ],
            stream=True,
        )
        logger.debug("Received response: {!r}", stream)
        return "".join([chunk["message"]["content"] for chunk in stream])
