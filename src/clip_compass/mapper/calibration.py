"""Deterministic DBSCAN epsilon calibration."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict

import numpy as np


@dataclass(frozen=True)
class EpsCalibration:
    eps: float
    method: str
    min_samples: int
    metric: str
    sample_count: int
    sample_size_limit: int
    quantile: float
    knee_index: int
    distance_min: float
    distance_median: float
    distance_max: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "eps": self.eps,
            "method": self.method,
            "min_samples": self.min_samples,
            "metric": self.metric,
            "sample_count": self.sample_count,
            "sample_size_limit": self.sample_size_limit,
            "quantile": self.quantile,
            "knee_index": self.knee_index,
            "distance_min": self.distance_min,
            "distance_median": self.distance_median,
            "distance_max": self.distance_max,
        }


def calibrate_dbscan_eps(
    features: np.ndarray,
    *,
    min_samples: int,
    metric: str = "euclidean",
    sample_size: int = 5000,
    fallback_quantile: float = 0.9,
    random_seed: int = 42,
    n_jobs: int = 1,
) -> EpsCalibration:
    """Estimate epsilon from the sorted k-distance curve.

    This adapts the researcher's nearest-neighbor elbow heuristic while using
    DBSCAN's configured ``min_samples`` as k, deterministic subsampling, and an
    explicit quantile fallback when no usable knee exists.
    """

    try:
        from kneed import KneeLocator
        from sklearn.neighbors import NearestNeighbors
    except ImportError as error:
        raise RuntimeError(
            "Mapper epsilon calibration requires the 'mapper' extra: "
            "pip install -e '.[mapper]'"
        ) from error

    values = np.asarray(features, dtype=np.float64)
    if values.ndim != 2 or values.shape[0] < 2:
        raise ValueError("epsilon calibration requires at least two feature vectors")
    if not np.isfinite(values).all():
        raise ValueError("features must contain only finite values")
    if min_samples < 2:
        raise ValueError("min_samples must be at least 2")
    if sample_size < 2:
        raise ValueError("sample_size must be at least 2")
    if not 0.0 < fallback_quantile <= 1.0:
        raise ValueError("fallback_quantile must be in (0, 1]")

    count = min(values.shape[0], sample_size)
    if count < values.shape[0]:
        rng = np.random.default_rng(random_seed)
        indices = np.sort(rng.choice(values.shape[0], size=count, replace=False))
        sample = values[indices]
    else:
        sample = values

    neighbor_count = min(min_samples, count)
    neighbors = NearestNeighbors(
        n_neighbors=neighbor_count,
        metric=metric,
        algorithm="brute" if metric == "cosine" else "auto",
        n_jobs=n_jobs,
    ).fit(sample)
    distances, _ = neighbors.kneighbors(sample)
    k_distances = np.sort(np.asarray(distances[:, -1], dtype=np.float64))
    positive = k_distances[k_distances > np.finfo(np.float64).eps]
    if positive.size == 0:
        raise ValueError("all k-neighbor distances are zero; DBSCAN epsilon is undefined")

    knee = KneeLocator(
        np.arange(k_distances.size),
        k_distances,
        curve="convex",
        direction="increasing",
        interp_method="interp1d",
    ).knee
    knee_index = int(knee) if knee is not None else -1
    candidate = float(k_distances[knee_index]) if knee_index >= 0 else 0.0
    if not np.isfinite(candidate) or candidate <= np.finfo(np.float64).eps:
        candidate = float(np.quantile(positive, fallback_quantile))
        method = "quantile_fallback"
        knee_index = -1
    else:
        method = "k_distance_knee"

    return EpsCalibration(
        eps=candidate,
        method=method,
        min_samples=min_samples,
        metric=metric,
        sample_count=count,
        sample_size_limit=sample_size,
        quantile=fallback_quantile,
        knee_index=knee_index,
        distance_min=float(k_distances.min()),
        distance_median=float(np.median(k_distances)),
        distance_max=float(k_distances.max()),
    )
