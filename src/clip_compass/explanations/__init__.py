"""Mapper-node explanation context and hypothesis generation."""

from .context import build_explanation_contexts, node_centroid
from .service import generate_explanations, write_hypothesis_prompts
from .types import (
    EvidenceItem,
    ExplanationArtifact,
    ExplanationContextBundle,
    ExplanationHypothesis,
    NodeExplanation,
    NodeExplanationContext,
    hypothesis_id,
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
    "hypothesis_id",
    "node_centroid",
    "write_hypothesis_prompts",
]
