"""One-dimensional lens functions for CLIP embedding spaces."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Tuple

import numpy as np

from clip_compass.embeddings.types import EmbeddingBatch


LENS_NAMES: Tuple[str, ...] = (
    "l2_raw",
    "pca1_raw",
    "pca1_normalized",
    "eccentricity_normalized",
    "centroid_distance_normalized",
    "density_normalized",
)


def _matrix(values: np.ndarray) -> np.ndarray:
    matrix = np.asarray(values, dtype=np.float64)
    if matrix.ndim != 2 or matrix.shape[0] == 0 or matrix.shape[1] == 0:
        raise ValueError("embeddings must be a non-empty two-dimensional array")
    if not np.isfinite(matrix).all():
        raise ValueError("embeddings must contain only finite values")
    return matrix


def minmax_scale(values: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=np.float64)
    minimum = float(values.min())
    span = float(values.max()) - minimum
    if span <= np.finfo(np.float64).eps:
        return np.zeros_like(values, dtype=np.float64)
    return (values - minimum) / span


def l2_norm(embeddings: np.ndarray) -> np.ndarray:
    return np.linalg.norm(_matrix(embeddings), axis=1)


def pca_first_component(embeddings: np.ndarray) -> np.ndarray:
    matrix = _matrix(embeddings)
    centered = matrix - matrix.mean(axis=0, keepdims=True)
    if matrix.shape[0] == 1 or np.allclose(centered, 0.0):
        return np.zeros(matrix.shape[0], dtype=np.float64)
    _, _, components = np.linalg.svd(centered, full_matrices=False)
    component = components[0]
    pivot = int(np.argmax(np.abs(component)))
    if component[pivot] < 0:
        component = -component
    # np.matmul on some macOS Accelerate builds emits spurious floating-point
    # status warnings for this otherwise finite dot product. Einsum produces
    # the same projection without mutating global NumPy error settings.
    return np.einsum("ij,j->i", centered, component)


def distance_to_centroid(embeddings: np.ndarray) -> np.ndarray:
    matrix = _matrix(embeddings)
    centroid = matrix.mean(axis=0, keepdims=True)
    return np.linalg.norm(matrix - centroid, axis=1)


def _euclidean_distances(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left_sq = np.sum(left * left, axis=1, keepdims=True)
    right_sq = np.sum(right * right, axis=1, keepdims=True).T
    squared = np.maximum(left_sq + right_sq - 2.0 * (left @ right.T), 0.0)
    return np.sqrt(squared)


def eccentricity(embeddings: np.ndarray, *, block_size: int = 1024) -> np.ndarray:
    """Mean Euclidean distance from every point to the complete point cloud."""

    matrix = _matrix(embeddings)
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    result = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], block_size):
        stop = min(start + block_size, matrix.shape[0])
        result[start:stop] = _euclidean_distances(matrix[start:stop], matrix).mean(axis=1)
    return result


def local_density(
    embeddings: np.ndarray, *, neighbors: int = 5, block_size: int = 1024
) -> np.ndarray:
    """Mean distance to k nearest neighbors; lower values indicate denser regions."""

    matrix = _matrix(embeddings)
    if matrix.shape[0] < 2:
        return np.zeros(matrix.shape[0], dtype=np.float64)
    if neighbors < 1:
        raise ValueError("neighbors must be at least 1")
    if block_size < 1:
        raise ValueError("block_size must be at least 1")
    k = min(neighbors, matrix.shape[0] - 1)
    result = np.empty(matrix.shape[0], dtype=np.float64)
    for start in range(0, matrix.shape[0], block_size):
        stop = min(start + block_size, matrix.shape[0])
        distances = _euclidean_distances(matrix[start:stop], matrix)
        row_indices = np.arange(stop - start)
        distances[row_indices, np.arange(start, stop)] = np.inf
        nearest = np.partition(distances, kth=k - 1, axis=1)[:, :k]
        result[start:stop] = nearest.mean(axis=1)
    return result


@dataclass(frozen=True)
class LensResult:
    name: str
    source: str
    raw_values: np.ndarray
    values: np.ndarray
    scaled: bool
    parameters: Dict[str, object]

    def __post_init__(self) -> None:
        if self.name not in LENS_NAMES:
            raise ValueError(f"unknown lens: {self.name}")
        if self.raw_values.ndim != 1 or self.values.shape != self.raw_values.shape:
            raise ValueError("lens values must be matching one-dimensional arrays")
        if not np.isfinite(self.raw_values).all() or not np.isfinite(self.values).all():
            raise ValueError("lens values must be finite")

    def metadata(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "source": self.source,
            "scaled": self.scaled,
            "parameters": self.parameters,
            "item_count": int(self.values.shape[0]),
            "minimum": float(self.values.min()),
            "maximum": float(self.values.max()),
        }

    def save(self, path: Path, item_ids: Tuple[str, ...]) -> Path:
        if len(item_ids) != self.values.shape[0]:
            raise ValueError("item_ids must align with lens values")
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        with partial.open("wb") as output:
            np.savez_compressed(
                output,
                item_ids=np.asarray(item_ids, dtype=np.str_),
                raw_values=np.asarray(self.raw_values, dtype=np.float64),
                values=np.asarray(self.values, dtype=np.float64),
                metadata_json=np.asarray(
                    json.dumps(self.metadata(), sort_keys=True), dtype=np.str_
                ),
            )
        os.replace(partial, path)
        return path

    @classmethod
    def load(cls, path: Path) -> Tuple["LensResult", Tuple[str, ...]]:
        with np.load(path, allow_pickle=False) as artifact:
            metadata = json.loads(str(artifact["metadata_json"].item()))
            result = cls(
                name=str(metadata["name"]),
                source=str(metadata["source"]),
                raw_values=np.asarray(artifact["raw_values"], dtype=np.float64),
                values=np.asarray(artifact["values"], dtype=np.float64),
                scaled=bool(metadata["scaled"]),
                parameters=dict(metadata["parameters"]),
            )
            item_ids = tuple(str(value) for value in artifact["item_ids"].tolist())
        if len(item_ids) != result.values.shape[0]:
            raise ValueError("lens item IDs do not align with lens values")
        return result, item_ids


def compute_lens(
    name: str,
    embeddings: EmbeddingBatch,
    *,
    scale: bool = True,
    neighbors: int = 5,
    block_size: int = 1024,
) -> LensResult:
    """Compute a named lens with an explicit raw/normalized feature source."""

    if name == "l2_raw":
        source = "raw"
        raw_values = l2_norm(embeddings.raw)
        parameters: Dict[str, object] = {}
    elif name == "pca1_raw":
        source = "raw"
        raw_values = pca_first_component(embeddings.raw)
        parameters = {"sign_rule": "largest_absolute_loading_is_positive"}
    elif name == "pca1_normalized":
        source = "normalized"
        raw_values = pca_first_component(embeddings.normalized)
        parameters = {"sign_rule": "largest_absolute_loading_is_positive"}
    elif name == "eccentricity_normalized":
        source = "normalized"
        raw_values = eccentricity(embeddings.normalized, block_size=block_size)
        parameters = {"metric": "euclidean", "block_size": block_size}
    elif name == "centroid_distance_normalized":
        source = "normalized"
        raw_values = distance_to_centroid(embeddings.normalized)
        parameters = {"metric": "euclidean"}
    elif name == "density_normalized":
        source = "normalized"
        raw_values = local_density(
            embeddings.normalized, neighbors=neighbors, block_size=block_size
        )
        parameters = {
            "metric": "euclidean",
            "neighbors": min(neighbors, max(len(embeddings.item_ids) - 1, 0)),
            "block_size": block_size,
        }
    else:
        choices = ", ".join(LENS_NAMES)
        raise ValueError(f"unknown lens {name!r}; expected one of: {choices}")

    raw_values = np.asarray(raw_values, dtype=np.float64)
    values = minmax_scale(raw_values) if scale else raw_values.copy()
    return LensResult(
        name=name,
        source=source,
        raw_values=raw_values,
        values=values,
        scaled=scale,
        parameters=parameters,
    )
