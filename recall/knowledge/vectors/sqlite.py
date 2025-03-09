from collections.abc import Iterator
from os import PathLike
import sqlite3
from types import TracebackType
from typing import Literal, Self, override
from recall.knowledge.types import Chunk, Embedding
from recall.knowledge.vectors import VectorDb


class Sqlite3VectorDb(VectorDb):
    def __init__(self, dbfile: PathLike):
        self.dbfile = dbfile
        self.cnx: sqlite3.Connection

    @override
    def __enter__(self) -> Self:
        self.cnx = sqlite3.connect(self.dbfile)
        cur = self.cnx.cursor()
        cur.execute("""
            CREATE TABLE IF NOT EXISTS corpus (
                text TEXT,
                embedding TEXT
            );
        """)
        self.cnx.commit()
        return self

    @override
    def __exit__(
        self,
        type: type[BaseException] | None,
        value: BaseException | None,
        traceback: TracebackType | None,
    ) -> Literal[False]:
        return self.cnx.__exit__(type, value, traceback)

    def __iter__(self) -> Iterator[tuple[Chunk, Embedding]]:
        cur = self.cnx.cursor()
        cur.execute(""""
        SELECT text, embedding FROM corpus;
        """)
        return iter(cur.fetchall())

    def __repr__(self) -> str:
        return f"VectorDb({self.dbfile!r})"
