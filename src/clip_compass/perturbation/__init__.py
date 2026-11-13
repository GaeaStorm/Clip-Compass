"""Verify explanation hypotheses by masking COCO objects and re-embedding with CLIP."""

from .masks import category_mask, clip_visible_box, control_masks
from .service import verify_hypotheses
from .types import (
    HypothesisResult,
    ImageResult,
    ImageScore,
    PerturbationConfig,
    Tally,
    Verdict,
    VerificationResults,
)
from .verify import decide, format_summary, verified_explanations

__all__ = [
    "HypothesisResult",
    "ImageResult",
    "ImageScore",
    "PerturbationConfig",
    "Tally",
    "Verdict",
    "VerificationResults",
    "category_mask",
    "clip_visible_box",
    "control_masks",
    "decide",
    "format_summary",
    "verified_explanations",
    "verify_hypotheses",
]
