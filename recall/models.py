from enum import StrEnum


class EmbeddingModel(StrEnum):
    NOMIC_EMBED_TEXT = "nomic-embed-text"


class LanguageModel(StrEnum):
    MISTRAL = "mistral"
    PHI4_MINI = "phi4-mini"
