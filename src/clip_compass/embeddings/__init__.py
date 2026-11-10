"""CLIP embedding generation and caching."""

from .cache import EmbeddingCache
from .service import embed_images, embed_texts
from .similarity import ClipSimilarityMatrix, image_text_similarity
from .types import ClipModelSpec, EmbeddingBatch

__all__ = [
    "ClipModelSpec",
    "ClipSimilarityMatrix",
    "EmbeddingBatch",
    "EmbeddingCache",
    "embed_images",
    "embed_texts",
    "image_text_similarity",
]
