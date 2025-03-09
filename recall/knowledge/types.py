from collections.abc import Callable, Iterable
from dataclasses import dataclass


Dataset = Iterable[str]

Chunk = str
Embedding = list[float]

Vector = tuple[Chunk, Embedding]

EmbedFn = Callable[[Chunk], Embedding]


@dataclass
class Document:
    name: str
    content: str
