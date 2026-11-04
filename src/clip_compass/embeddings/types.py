"""Shared embedding types and serialization."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple

import numpy as np


EMBEDDING_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class ClipModelSpec:
    """Everything that determines the numerical meaning of a CLIP vector."""

    provider: str
    model_id: str
    revision: str
    preprocessing_id: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "provider": self.provider,
            "model_id": self.model_id,
            "revision": self.revision,
            "preprocessing_id": self.preprocessing_id,
        }

    @property
    def fingerprint(self) -> str:
        payload = json.dumps(self.as_dict(), sort_keys=True, separators=(",", ":"))
        return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:20]


@dataclass(frozen=True)
class EmbeddingBatch:
    """Raw projected and unit-normalized CLIP features in stable item order."""

    item_ids: Tuple[str, ...]
    raw: np.ndarray
    normalized: np.ndarray
    cache_hits: np.ndarray
    model_spec: ClipModelSpec
    input_sha256: Tuple[str, ...]
    modality: str = "image"
    input_texts: Tuple[str, ...] = ()

    def __post_init__(self) -> None:
        count = len(self.item_ids)
        if self.modality not in {"image", "text"}:
            raise ValueError("embedding modality must be 'image' or 'text'")
        if self.modality == "text":
            if len(self.input_texts) != count or any(
                not value.strip() for value in self.input_texts
            ):
                raise ValueError("text embeddings must retain one non-empty input per item")
        elif self.input_texts:
            raise ValueError("image embeddings must not contain text inputs")
        if self.raw.ndim != 2 or self.raw.shape[0] != count:
            raise ValueError("raw embeddings must have shape (item_count, embedding_dim)")
        if self.normalized.shape != self.raw.shape:
            raise ValueError("normalized embeddings must match raw embedding shape")
        if self.cache_hits.shape != (count,):
            raise ValueError("cache_hits must have shape (item_count,)")
        if len(self.input_sha256) != count:
            raise ValueError("input_sha256 must have one digest per item")
        if not np.isfinite(self.raw).all() or not np.isfinite(self.normalized).all():
            raise ValueError("embeddings must contain only finite values")
        if count:
            norms = np.linalg.norm(self.normalized, axis=1)
            if not np.allclose(norms, 1.0, rtol=1e-5, atol=1e-6):
                raise ValueError("normalized embeddings must have unit L2 norm")

    @property
    def embedding_dim(self) -> int:
        return int(self.raw.shape[1])

    def metadata(self) -> Dict[str, object]:
        return {
            "schema_version": EMBEDDING_SCHEMA_VERSION,
            "model_spec": self.model_spec.as_dict(),
            "model_fingerprint": self.model_spec.fingerprint,
            "item_count": len(self.item_ids),
            "embedding_dim": self.embedding_dim,
            "raw_dtype": str(self.raw.dtype),
            "cache_hit_count": int(self.cache_hits.sum()),
            "modality": self.modality,
        }

    def save(self, path: Path) -> Path:
        """Atomically save a portable, pickle-free compressed NumPy artifact."""

        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        with partial.open("wb") as output:
            np.savez_compressed(
                output,
                item_ids=np.asarray(self.item_ids, dtype=np.str_),
                raw=np.asarray(self.raw, dtype=np.float32),
                normalized=np.asarray(self.normalized, dtype=np.float32),
                cache_hits=np.asarray(self.cache_hits, dtype=np.bool_),
                input_sha256=np.asarray(self.input_sha256, dtype=np.str_),
                input_texts=np.asarray(self.input_texts, dtype=np.str_),
                metadata_json=np.asarray(
                    json.dumps(self.metadata(), sort_keys=True), dtype=np.str_
                ),
            )
        os.replace(partial, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "EmbeddingBatch":
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            spec_data: Mapping[str, object] = metadata["model_spec"]
            model_spec = ClipModelSpec(
                provider=str(spec_data["provider"]),
                model_id=str(spec_data["model_id"]),
                revision=str(spec_data["revision"]),
                preprocessing_id=str(spec_data["preprocessing_id"]),
            )
            return cls(
                item_ids=tuple(str(value) for value in artifact["item_ids"].tolist()),
                raw=np.asarray(artifact["raw"], dtype=np.float32),
                normalized=np.asarray(artifact["normalized"], dtype=np.float32),
                cache_hits=np.asarray(artifact["cache_hits"], dtype=np.bool_),
                model_spec=model_spec,
                input_sha256=tuple(
                    str(value) for value in artifact["input_sha256"].tolist()
                ),
                modality=str(metadata.get("modality", "image")),
                input_texts=(
                    tuple(str(value) for value in artifact["input_texts"].tolist())
                    if "input_texts" in artifact.files
                    else ()
                ),
            )
