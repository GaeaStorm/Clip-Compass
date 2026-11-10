"""Mapper-node explanation context and hypothesis generation."""

from .context import build_explanation_contexts
from .service import generate_explanations, write_hypothesis_prompts
from .types import (
    EvidenceItem,
    ExplanationArtifact,
    ExplanationContextBundle,
    ExplanationHypothesis,
    NodeExplanation,
    NodeExplanationContext,
)

__all__ = [
    "EvidenceItem",
    "ExplanationArtifact",
    "ExplanationContextBundle",
    "ExplanationHypothesis",
    "NodeExplanation",
    "NodeExplanationContext",
    "build_explanation_contexts",
    "generate_explanations",
    "write_hypothesis_prompts",
]
