"""Hypothesis verdicts from perturbation results, plus the text-only comparison arm."""

from .baseline import auroc, score_prompt_baseline
from .service import aggregate_verdicts
from .stats import benjamini_hochberg, object_rank, rank_sum_p_value
from .types import (
    LABELS,
    STATISTICS,
    BaselineArtifact,
    BaselineScore,
    HypothesisVerdict,
    ImageResult,
    StatisticTest,
    VerdictArtifact,
    VerdictConfig,
)

__all__ = [
    "LABELS",
    "STATISTICS",
    "BaselineArtifact",
    "BaselineScore",
    "HypothesisVerdict",
    "ImageResult",
    "StatisticTest",
    "VerdictArtifact",
    "VerdictConfig",
    "aggregate_verdicts",
    "auroc",
    "benjamini_hochberg",
    "object_rank",
    "rank_sum_p_value",
    "score_prompt_baseline",
]
