"""Verify explanation hypotheses by masking COCO objects and re-embedding with CLIP."""

from .masks import category_mask, clip_visible_box, control_masks
from .service import verify_hypotheses
from .verify import (
    PerturbationConfig,
    decide,
    format_summary,
    load_results,
    verified_explanations,
)

__all__ = [
    "PerturbationConfig",
    "category_mask",
    "clip_visible_box",
    "control_masks",
    "decide",
    "format_summary",
    "load_results",
    "verified_explanations",
    "verify_hypotheses",
]
