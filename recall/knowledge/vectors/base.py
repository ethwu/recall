from abc import ABC, abstractmethod
from collections.abc import Iterator
from os import PathLike
from types import TracebackType
from typing import Literal, Self

from recall.knowledge.types import Chunk, EmbedFn, Embedding


Similarity = float


def cosine_similarity(a: Embedding, b: Embedding) -> Similarity:
    dot_product = sum([x * y for x, y in zip(a, b)])
    norm_a = sum([x**2 for x in a]) ** 0.5
    norm_b = sum([x**2 for x in b]) ** 0.5
    return dot_product / (norm_a * norm_b)


class BaseVectorDb(ABC):
    """The vector database holding the reference data."""

    @classmethod
    @abstractmethod
    def from_path(cls, path: PathLike, embed: EmbedFn) -> Self: ...

    def __enter__(self) -> Self:
        return self

    def __exit__(
        self,
        _type: type[BaseException] | None,
        _value: BaseException | None,
        _traceback: TracebackType | None,
    ) -> Literal[False]:
        return False

    def similarities(
        self, given: Embedding, n: int
    ) -> Iterator[tuple[Chunk, Similarity]]:
        """Get the top n vectors ranked by similarity with the given embedding."""
        similarities = [
            (chunk, cosine_similarity(given, embedding)) for chunk, embedding in self
        ]
        similarities.sort(key=lambda x: x[1], reverse=True)
        return similarities[:n]

    @abstractmethod
    def __setitem__(self, chunk: Chunk, embedding: Embedding) -> None: ...

    # @abstractmethod
    # def __getitem__(self, chunk: Chunk) -> Embedding | None: ...

    # @abstractmethod
    # def __iter__(self) -> Iterator[tuple[Chunk, Embedding]]: ...
