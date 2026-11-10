"""Controlled visual perturbations for explanation verification."""

from .base import PerturbedImages, VisualPerturbationEngine
from .masking import FILL_MODES, MaskingPerturbationEngine, translated_control_mask
from .service import generate_perturbations
from .types import PerturbationArtifact, PerturbationPair, SkippedPerturbation

__all__ = [
    "FILL_MODES",
    "MaskingPerturbationEngine",
    "PerturbationArtifact",
    "PerturbationPair",
    "PerturbedImages",
    "SkippedPerturbation",
    "VisualPerturbationEngine",
    "generate_perturbations",
    "translated_control_mask",
]
