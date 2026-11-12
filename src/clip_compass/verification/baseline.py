"""The text-only comparison arm: CLIP prompt scoring without any perturbation.

A hypothesis is accepted when its positive prompt is, on average, more similar
to the node's members than to every other image in the graph. This is the
check the explanation is judged by when nothing is verified visually.
"""

from __future__ import annotations

from pathlib import Path
from typing import List, Optional

import numpy as np

from clip_compass.data.coco_instances import sha256_file
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.explanations.types import ExplanationArtifact
from clip_compass.mapper.graph import MapperGraphArtifact

from .checks import check_graph_inputs, check_image_text_compatibility, prompt_id
from .types import BaselineArtifact, BaselineScore


def auroc(inside: np.ndarray, outside: np.ndarray) -> Optional[float]:
    """Probability a random member outscores a random non-member (ties count half).

    Computed from average ranks (the Mann-Whitney U statistic), so it is exact
    with ties and does not need an n_inside x n_outside comparison matrix.
    """

    if inside.size == 0 or outside.size == 0:
        return None
    combined = np.concatenate([inside, outside])
    unique, inverse, counts = np.unique(combined, return_inverse=True, return_counts=True)
    first_rank = np.cumsum(counts) - counts + 1
    average_rank = first_rank + (counts - 1) / 2.0
    ranks = average_rank[inverse]
    rank_sum = ranks[: inside.size].sum()
    u_statistic = rank_sum - inside.size * (inside.size + 1) / 2.0
    return float(u_statistic / (inside.size * outside.size))


def score_prompt_baseline(
    graph: MapperGraphArtifact,
    image_embeddings: EmbeddingBatch,
    explanations: ExplanationArtifact,
    prompt_embeddings: EmbeddingBatch,
    *,
    graph_path: Path,
    explanations_path: Path,
) -> BaselineArtifact:
    graph_sha256 = check_graph_inputs(graph, image_embeddings, explanations, graph_path)
    check_image_text_compatibility(image_embeddings, prompt_embeddings)
    nodes = {node.id: node for node in graph.nodes}
    prompt_rows = {item_id: row for row, item_id in enumerate(prompt_embeddings.item_ids)}
    images = np.asarray(image_embeddings.normalized, dtype=np.float64)
    prompts = np.asarray(prompt_embeddings.normalized, dtype=np.float64)

    scores: List[BaselineScore] = []
    for explanation in explanations.explanations:
        node = nodes.get(explanation.node_id)
        if node is None:
            raise ValueError(f"explanation refers to unknown node {explanation.node_id}")
        inside_mask = np.zeros(len(graph.item_ids), dtype=np.bool_)
        inside_mask[list(node.member_indices)] = True
        if inside_mask.all():
            raise ValueError(f"node {node.id} contains every image; nothing to compare against")
        for index, hypothesis in enumerate(explanation.hypotheses):
            text_id = prompt_id(node.id, index, "positive")
            if text_id not in prompt_rows:
                raise ValueError(f"missing prompt embedding {text_id!r}")
            similarity = images @ prompts[prompt_rows[text_id]]
            inside = similarity[inside_mask]
            outside = similarity[~inside_mask]
            inside_mean = float(inside.mean())
            outside_mean = float(outside.mean())
            scores.append(
                BaselineScore(
                    node_id=node.id,
                    hypothesis_index=index,
                    hypothesis_name=hypothesis.name,
                    member_count=int(inside.size),
                    outside_count=int(outside.size),
                    inside_mean=inside_mean,
                    outside_mean=outside_mean,
                    auroc=auroc(inside, outside),
                    accepted=inside_mean > outside_mean,
                )
            )
    return BaselineArtifact(
        graph_sha256=graph_sha256,
        explanations_sha256=sha256_file(explanations_path),
        embedding_model_fingerprint=image_embeddings.model_spec.fingerprint,
        scores=tuple(scores),
    )
