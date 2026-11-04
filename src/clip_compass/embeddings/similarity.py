"""Validated CLIP image-text cosine similarity artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np

from .types import EmbeddingBatch


CLIP_SIMILARITY_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClipSimilarityMatrix:
    image_ids: Tuple[str, ...]
    text_ids: Tuple[str, ...]
    texts: Tuple[str, ...]
    scores: np.ndarray
    model_identity: Mapping[str, str]

    def __post_init__(self) -> None:
        if self.scores.shape != (len(self.image_ids), len(self.text_ids)):
            raise ValueError("similarity scores must have shape (image_count, text_count)")
        if len(self.texts) != len(self.text_ids):
            raise ValueError("similarity text values must align with text IDs")
        if not np.isfinite(self.scores).all():
            raise ValueError("similarity scores must be finite")
        if np.any(self.scores < -1.00001) or np.any(self.scores > 1.00001):
            raise ValueError("cosine similarity scores must be in [-1, 1]")

    def metadata(self) -> Dict[str, object]:
        return {
            "schema_version": CLIP_SIMILARITY_SCHEMA_VERSION,
            "image_count": len(self.image_ids),
            "text_count": len(self.text_ids),
            "model_identity": dict(self.model_identity),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        with partial.open("wb") as output:
            np.savez_compressed(
                output,
                image_ids=np.asarray(self.image_ids, dtype=np.str_),
                text_ids=np.asarray(self.text_ids, dtype=np.str_),
                texts=np.asarray(self.texts, dtype=np.str_),
                scores=np.asarray(self.scores, dtype=np.float32),
                metadata_json=np.asarray(
                    json.dumps(self.metadata(), sort_keys=True), dtype=np.str_
                ),
            )
        os.replace(partial, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "ClipSimilarityMatrix":
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            if metadata.get("schema_version") != CLIP_SIMILARITY_SCHEMA_VERSION:
                raise ValueError("unsupported CLIP similarity schema version")
            return cls(
                image_ids=tuple(str(value) for value in artifact["image_ids"].tolist()),
                text_ids=tuple(str(value) for value in artifact["text_ids"].tolist()),
                texts=tuple(str(value) for value in artifact["texts"].tolist()),
                scores=np.asarray(artifact["scores"], dtype=np.float32),
                model_identity={
                    str(key): str(value)
                    for key, value in metadata["model_identity"].items()
                },
            )


def _model_identity(batch: EmbeddingBatch) -> Dict[str, str]:
    spec = batch.model_spec
    return {
        "provider": spec.provider,
        "model_id": spec.model_id,
        "revision": spec.revision,
    }


def image_text_similarity(
    images: EmbeddingBatch, texts: EmbeddingBatch
) -> ClipSimilarityMatrix:
    """Compute cosine similarity after validating cross-modal compatibility."""

    if images.modality != "image":
        raise ValueError("images artifact must have image modality")
    if texts.modality != "text":
        raise ValueError("texts artifact must have text modality")
    image_identity = _model_identity(images)
    text_identity = _model_identity(texts)
    if image_identity != text_identity:
        raise ValueError("image and text embeddings must use the same CLIP model revision")
    if images.embedding_dim != texts.embedding_dim:
        raise ValueError("image and text embedding dimensions must match")
    # einsum avoids a spurious Accelerate/BLAS overflow warning observed on
    # some macOS NumPy builds for otherwise finite unit vectors.
    scores = np.asarray(
        np.einsum("id,jd->ij", images.normalized, texts.normalized),
        dtype=np.float32,
    )
    return ClipSimilarityMatrix(
        image_ids=images.item_ids,
        text_ids=texts.item_ids,
        texts=texts.input_texts,
        scores=scores,
        model_identity=image_identity,
    )
