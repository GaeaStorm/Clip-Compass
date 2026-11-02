"""Cache-aware CLIP image embedding orchestration."""

from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np

from .base import ClipEncoder
from .cache import EmbeddingCache
from .types import EmbeddingBatch


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def l2_normalize(vectors: np.ndarray) -> np.ndarray:
    vectors = np.asarray(vectors, dtype=np.float32)
    if vectors.ndim != 2:
        raise ValueError("vectors must be a two-dimensional array")
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    if np.any(norms <= np.finfo(np.float32).eps):
        raise ValueError("cannot normalize a zero-length embedding")
    return vectors / norms


def embed_images(
    image_paths: Sequence[Path],
    *,
    encoder: ClipEncoder,
    cache: EmbeddingCache,
    item_ids: Optional[Sequence[str]] = None,
    batch_size: int = 16,
) -> EmbeddingBatch:
    """Embed images, encoding only cache misses and preserving input order."""

    if batch_size < 1:
        raise ValueError("batch_size must be at least 1")
    paths = tuple(Path(path).expanduser().resolve() for path in image_paths)
    if not paths:
        raise ValueError("at least one image path is required")
    for path in paths:
        if not path.is_file():
            raise FileNotFoundError(path)

    ids = tuple(item_ids) if item_ids is not None else tuple(path.stem for path in paths)
    if len(ids) != len(paths):
        raise ValueError("item_ids must have the same length as image_paths")
    if len(set(ids)) != len(ids):
        raise ValueError("item_ids must be unique")

    digests = tuple(sha256_file(path) for path in paths)
    vectors: List[Optional[np.ndarray]] = [None] * len(paths)
    cache_hits = np.zeros(len(paths), dtype=np.bool_)
    misses_by_digest: Dict[str, List[int]] = {}

    for index, digest in enumerate(digests):
        cached = cache.load(encoder.spec, "image", digest)
        if cached is not None:
            vectors[index] = cached
            cache_hits[index] = True
        else:
            misses_by_digest.setdefault(digest, []).append(index)

    unique_miss_indices = [indices[0] for indices in misses_by_digest.values()]
    for start in range(0, len(unique_miss_indices), batch_size):
        batch_indices = unique_miss_indices[start : start + batch_size]
        encoded = np.asarray(
            encoder.encode_images([paths[index] for index in batch_indices]), dtype=np.float32
        )
        if encoded.ndim != 2 or encoded.shape[0] != len(batch_indices):
            raise ValueError("encoder returned an invalid batch shape")
        if not np.isfinite(encoded).all():
            raise ValueError("encoder returned non-finite values")
        for batch_offset, index in enumerate(batch_indices):
            vector = encoded[batch_offset]
            digest = digests[index]
            cache.store(encoder.spec, "image", digest, vector)
            for duplicate_index in misses_by_digest[digest]:
                vectors[duplicate_index] = vector.copy()

    if any(vector is None for vector in vectors):
        raise AssertionError("embedding orchestration left an item unresolved")
    raw = np.stack([vector for vector in vectors if vector is not None]).astype(np.float32)
    return EmbeddingBatch(
        item_ids=ids,
        raw=raw,
        normalized=l2_normalize(raw),
        cache_hits=cache_hits,
        model_spec=encoder.spec,
        input_sha256=digests,
    )
