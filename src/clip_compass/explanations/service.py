"""Explanation-run orchestration and prompt export."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Dict, List

from .base import ExplanationAgent
from .types import ExplanationArtifact, ExplanationContextBundle


def generate_explanations(
    contexts: ExplanationContextBundle,
    agent: ExplanationAgent,
    *,
    max_hypotheses: int = 3,
) -> ExplanationArtifact:
    if max_hypotheses < 1 or max_hypotheses > 8:
        raise ValueError("max_hypotheses must be between 1 and 8")
    explanations = tuple(
        agent.explain(context, max_hypotheses=max_hypotheses)
        for context in contexts.contexts
    )
    generator = dict(agent.metadata)
    generator["max_hypotheses"] = max_hypotheses
    return ExplanationArtifact(
        graph_sha256=contexts.graph_sha256,
        annotations_sha256=contexts.annotations_sha256,
        embedding_model_fingerprint=contexts.embedding_model_fingerprint,
        generator=generator,
        context_config=contexts.config,
        explanations=explanations,
    )


def write_hypothesis_prompts(artifact: ExplanationArtifact, path: Path) -> Path:
    """Export positive and negative prompts as JSONL accepted by ``encode-texts``."""

    rows: List[Dict[str, object]] = []
    for explanation in artifact.explanations:
        for index, hypothesis in enumerate(explanation.hypotheses):
            prefix = f"{explanation.node_id}-h{index:02d}"
            rows.extend(
                [
                    {
                        "id": f"{prefix}-positive",
                        "text": hypothesis.positive_prompt,
                        "node_id": explanation.node_id,
                        "hypothesis": hypothesis.name,
                        "polarity": "positive",
                    },
                    {
                        "id": f"{prefix}-negative",
                        "text": hypothesis.negative_prompt,
                        "node_id": explanation.node_id,
                        "hypothesis": hypothesis.name,
                        "polarity": "negative",
                    },
                ]
            )
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    with partial.open("w", encoding="utf-8") as output:
        for row in rows:
            output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)
    return path
