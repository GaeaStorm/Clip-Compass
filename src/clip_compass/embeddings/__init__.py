"""CLIP embedding generation and caching."""

from .cache import EmbeddingCache
from .service import embed_images
from .types import ClipModelSpec, EmbeddingBatch

__all__ = ["ClipModelSpec", "EmbeddingBatch", "EmbeddingCache", "embed_images"]
