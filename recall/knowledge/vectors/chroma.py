from collections.abc import Generator, Iterable
import itertools
from os import PathLike
from pathlib import Path
from typing import Self, override
from recall.knowledge.embedding import OllamaEmbeddingFunction
from recall.knowledge.types import Chunk, Dataset, Document, EmbedFn, Embedding
from recall.knowledge.vectors.base import BaseVectorDb
from recall.models import EmbeddingModel

from chromadb import (
    Client,
    Collection,
    EmbeddingFunction,
    PersistentClient,
    QueryResult,
)


def load_dataset(source: PathLike) -> Generator[Document]:
    with open(source, "r") as f:
        for i, line in enumerate(f.readlines()):
            yield Document(f"document {i}", line)


class ChromaVectorDb(BaseVectorDb):
    def __init__(self, identifier: str, model: EmbeddingModel) -> None:
        self.identifier = identifier
        self.cnx: Client
        self.vectors: Collection
        self.embedding_function = OllamaEmbeddingFunction(model)

    def with_dir(self, path: PathLike) -> Self:
        self.__enter__()
        path = Path(path)
        documents = {}
        for document in path.glob("**/*.md"):
            with open(document, "r") as f:
                if not self.vectors.get(ids=[document.name], limit=1, include=["uris"])["uris"]:
                    documents[document.name] = f.read()
        if documents:
            self.vectors.add(
                documents=list(doc for doc in documents.values()),
                ids=list(title for title in documents.keys()),
            )
        return self

    # @override
    @classmethod
    def from_path(cls, path: PathLike, model: EmbeddingModel) -> Self:
        vector_db = ChromaVectorDb(model).__enter__()
        for doc in load_dataset(path):
            vector_db.vectors.add(documents=[doc.content], ids=[doc.name])
        return vector_db

    @override
    def __enter__(self) -> Self:
        self.cnx = PersistentClient(
            path=str(Path(".", self.identifier).with_suffix(".chroma"))
        )
        self.vectors = self.cnx.get_or_create_collection(
            "corpus", embedding_function=self.embedding_function
        )
        return self

    @override
    def __setitem__(self, chunk: Chunk, document: Document) -> None:
        self.vectors.add(embeddings=[document.content], ids=[document.name])

    # def add(self, *documents: str) -> None:
    #     self.vectors.add(documents=documents)

    def query(self, query: str, *, n: int) -> Iterable[float, Chunk]:
        result = self.vectors.query(query_texts=[query], n_results=n)
        return zip(result["documents"][0], result["distances"][0])
