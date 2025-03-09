from chromadb import Documents, EmbeddingFunction, Embeddings
from loguru import logger
import ollama
from recall.models import EmbeddingModel


class OllamaEmbeddingFunction(EmbeddingFunction):
    def __init__(self, embedding_model: EmbeddingModel) -> None:
        self.embedding_model = embedding_model.value

    def __call__(self, documents: Documents) -> Embeddings:
        embedding = ollama.embed(self.embedding_model, documents)
        if not embedding or "embeddings" not in embedding:
            logger.warn("Embedding didn't work! Got {!r}", embedding)
            return []
        return embedding["embeddings"]
