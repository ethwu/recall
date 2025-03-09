from collections.abc import Callable, Iterator
from os import PathLike
from pathlib import Path
import pickle
from typing import Any, Self, override

from loguru import logger
from recall.knowledge.types import Chunk, Dataset, EmbedFn, Embedding
from recall.knowledge.vectors.base import BaseVectorDb


def load_dataset(source: PathLike) -> Dataset:
    with open(source, "r") as f:
        return f.readlines()


def pack(path: Path, o: Any) -> None:
    path = Path(path)
    with path.with_suffix(".pickle").open("wb") as f:
        pickle.dump(o, f, pickle.HIGHEST_PROTOCOL)


def unpack_or_load[T](path: PathLike, callback: Callable[[Path], T]) -> T:
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


def load(embed: EmbedFn) -> BaseVectorDb:
    def load_helper(dataset_path: PathLike):
        dataset = load_dataset(dataset_path)
        dataset_length = len(dataset)
        vector_db = DictVectorDb()
        for i, chunk in enumerate(dataset):
            vector_db[chunk] = embed(chunk)
            logger.info("Added chunk {} of {} to the database.", i + 1, dataset_length)
        return vector_db

    return load_helper


class DictVectorDb(BaseVectorDb):
    def __init__(self):
        self.vectors: dict[Chunk, Embedding] = {}

    @override
    @classmethod
    def from_path(cls, path: PathLike, embed: EmbedFn) -> Self:
        logger.info("Loading dataset from {!r}.", path)
        return unpack_or_load(path, load(embed))

    @override
    def __setitem__(self, chunk: Chunk, embedding: Embedding) -> None:
        self.vectors[chunk] = embedding

    @override
    def __iter__(self) -> Iterator[tuple[Chunk, Embedding]]:
        return iter(self.vectors.items())
