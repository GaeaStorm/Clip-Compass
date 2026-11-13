"""Hypothesis-level significance testing and multiple-comparison correction."""

from __future__ import annotations

from typing import List, Optional, Sequence

import numpy as np


def object_rank(object_value: float, control_values: Sequence[float]) -> int:
    """How many random controls the object edit beat (strictly), from 0 to len(controls).

    Ties count against the object, which keeps the test conservative.
    """

    controls = np.asarray(control_values, dtype=np.float64)
    if controls.ndim != 1 or controls.size == 0:
        raise ValueError("control_values must be a non-empty one-dimensional sequence")
    if not (np.isfinite(controls).all() and np.isfinite(object_value)):
        raise ValueError("object and control values must be finite")
    return int(np.count_nonzero(controls < object_value))


def rank_sum_p_value(ranks: Sequence[int], control_counts: Sequence[int]) -> float:
    """Exact one-sided p-value that the object edits beat their controls more than chance.

    Under the null hypothesis the object mask is just another random region, so in
    each image the object is exchangeable with its ``K`` controls and its rank
    (the number of controls it beats) is uniform on ``0..K``. Images are
    independent, so the null distribution of the summed rank is the convolution of
    those uniform distributions. The p-value is the probability that a null sum is
    at least the observed sum.

    Unlike a per-image threshold followed by a majority vote, every image's full
    rank counts: consistent moderate effects add up, and two images that each beat
    all 39 controls already give p = 1/1600.
    """

    if len(ranks) != len(control_counts) or not len(ranks):
        raise ValueError("ranks and control_counts must be non-empty and the same length")
    distribution = np.ones(1, dtype=np.float64)
    observed = 0
    for rank, count in zip(ranks, control_counts):
        count, rank = int(count), int(rank)
        if count < 1 or not 0 <= rank <= count:
            raise ValueError("each rank must lie between 0 and its control count")
        distribution = np.convolve(distribution, np.full(count + 1, 1.0 / (count + 1)))
        observed += rank
    return float(min(1.0, distribution[observed:].sum()))


def benjamini_hochberg(p_values: Sequence[Optional[float]]) -> List[Optional[float]]:
    """Benjamini-Hochberg adjusted p-values (q-values), preserving input order.

    ``None`` entries are tests that were not run (for example, too few testable
    images); they stay ``None`` and do not count toward the number of comparisons.
    """

    indices = [index for index, value in enumerate(p_values) if value is not None]
    adjusted: List[Optional[float]] = [None] * len(p_values)
    if not indices:
        return adjusted
    values = np.asarray([p_values[index] for index in indices], dtype=np.float64)
    if np.any(values < 0.0) or np.any(values > 1.0):
        raise ValueError("p-values must be in [0, 1]")
    total = values.size
    order = np.argsort(values, kind="stable")
    ranked = values[order] * total / np.arange(1, total + 1)
    # Enforce monotonicity from the largest p-value downwards, then cap at 1.
    ranked = np.minimum.accumulate(ranked[::-1])[::-1]
    ranked = np.minimum(ranked, 1.0)
    result = np.empty(total, dtype=np.float64)
    result[order] = ranked
    for slot, index in enumerate(indices):
        adjusted[index] = float(result[slot])
    return adjusted
