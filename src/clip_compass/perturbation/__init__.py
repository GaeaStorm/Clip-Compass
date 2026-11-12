"""Verify explanation hypotheses by masking COCO objects and re-embedding with CLIP."""

from .masks import category_mask, clip_visible_box, control_masks
from .verify import (
    PerturbationConfig,
    format_summary,
    load_results,
    verified_explanations,
    verify_hypotheses,
)

__all__ = [
    "PerturbationConfig",
    "category_mask",
    "clip_visible_box",
    "control_masks",
    "format_summary",
    "load_results",
    "verified_explanations",
    "verify_hypotheses",
]
