"""Content-addressed cache for individual embedding vectors."""

from __future__ import annotations

import hashlib
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Optional

import numpy as np

from .types import ClipModelSpec


CACHE_SCHEMA_VERSION = 1


class CacheCorruptionError(RuntimeError):
    """Raised when a cache entry is incomplete or fails validation."""


@dataclass(frozen=True)
class CacheAddress:
    key: str
    vector_path: Path
    metadata_path: Path


class EmbeddingCache:
    """Store raw projected vectors using image content and model identity."""

    def __init__(self, root: Path):
        self.root = root.expanduser().resolve()

    def address(self, spec: ClipModelSpec, modality: str, input_sha256: str) -> CacheAddress:
        payload = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_fingerprint": spec.fingerprint,
            "modality": modality,
            "input_sha256": input_sha256,
        }
        key = hashlib.sha256(
            json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()
        directory = self.root / spec.fingerprint / modality / key[:2]
        return CacheAddress(
            key=key,
            vector_path=directory / f"{key}.npy",
            metadata_path=directory / f"{key}.json",
        )

    def load(
        self, spec: ClipModelSpec, modality: str, input_sha256: str
    ) -> Optional[np.ndarray]:
        address = self.address(spec, modality, input_sha256)
        vector_exists = address.vector_path.exists()
        metadata_exists = address.metadata_path.exists()
        if not vector_exists and not metadata_exists:
            return None
        if vector_exists != metadata_exists:
            raise CacheCorruptionError(f"Incomplete cache entry: {address.key}")

        metadata = json.loads(address.metadata_path.read_text(encoding="utf-8"))
        expected = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_fingerprint": spec.fingerprint,
            "modality": modality,
            "input_sha256": input_sha256,
        }
        for key, value in expected.items():
            if metadata.get(key) != value:
                raise CacheCorruptionError(
                    f"Cache metadata mismatch for {address.key}: {key}"
                )

        vector = np.load(address.vector_path, allow_pickle=False)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise CacheCorruptionError(f"Invalid vector in cache entry: {address.key}")
        if metadata.get("embedding_dim") != int(vector.shape[0]):
            raise CacheCorruptionError(f"Dimension mismatch in cache entry: {address.key}")
        return np.asarray(vector, dtype=np.float32)

    def store(
        self,
        spec: ClipModelSpec,
        modality: str,
        input_sha256: str,
        vector: np.ndarray,
    ) -> CacheAddress:
        vector = np.asarray(vector, dtype=np.float32)
        if vector.ndim != 1 or not np.isfinite(vector).all():
            raise ValueError("cached vector must be a finite one-dimensional array")

        address = self.address(spec, modality, input_sha256)
        address.vector_path.parent.mkdir(parents=True, exist_ok=True)
        vector_partial = address.vector_path.with_name(f"{address.vector_path.name}.part")
        metadata_partial = address.metadata_path.with_name(
            f"{address.metadata_path.name}.part"
        )
        with vector_partial.open("wb") as output:
            np.save(output, vector, allow_pickle=False)
        metadata: Dict[str, object] = {
            "schema_version": CACHE_SCHEMA_VERSION,
            "model_fingerprint": spec.fingerprint,
            "model_spec": spec.as_dict(),
            "modality": modality,
            "input_sha256": input_sha256,
            "embedding_dim": int(vector.shape[0]),
            "dtype": str(vector.dtype),
        }
        metadata_partial.write_text(
            json.dumps(metadata, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )
        os.replace(vector_partial, address.vector_path)
        os.replace(metadata_partial, address.metadata_path)
        return address
