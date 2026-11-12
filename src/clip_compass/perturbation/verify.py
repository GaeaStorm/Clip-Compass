"""Verdict rules for perturbation results, plus summary and verified-hypothesis export.

For each node member that contains the hypothesis categories, the object mask and
``controls`` area-matched random masks are filled and re-embedded (see ``service``).
An image *exceeds* on a statistic when the object mask beats its controls
(empirical p <= alpha):

- ``centroid``: increase in distance from the node centroid
- ``prompt``: drop in cosine similarity to the hypothesis's positive prompt

Verdict rules, in order:

1. no categories, or a category unknown to COCO -> untestable
2. no member contains the categories -> untestable
3. categories visible in fewer than ``min_coverage`` of members -> not_supported
4. fewer than ``min_testable`` testable members -> untestable
5. a strict majority of testable members exceed on centroid -> supported;
   only on prompt -> detected_not_explanatory; neither -> not_supported
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence

from clip_compass.data.coco_instances import sha256_file
from clip_compass.explanations.types import (
    ExplanationArtifact,
    NodeExplanation,
    hypothesis_id,
)

SCHEMA_VERSION = 1
FILLS = {"mean": (123, 117, 104), "black": (0, 0, 0)}  # mean = round(255 * CLIP image_mean)
PREDICTED_STATISTICS = {
    "decrease_text_similarity": ("prompt",),
    "move_away_from_node_centroid": ("centroid",),
    "both": ("centroid", "prompt"),
}


@dataclass(frozen=True)
class PerturbationConfig:
    controls: int = 39  # p-values in steps of 1/40, so p can reach 0.05
    seed: int = 0
    fill: str = "mean"
    dilate_px: int = 3
    min_visible_fraction: float = 0.01  # of the 224x224 crop, about half a ViT-B/32 patch
    max_control_overlap: float = 0.25
    alpha: float = 0.05
    min_coverage: float = 0.5
    min_testable: int = 2
    min_identity_cosine: float = 0.999

    def __post_init__(self) -> None:
        if self.fill not in FILLS:
            raise ValueError(f"fill must be one of {sorted(FILLS)}")
        if self.controls < 1 or self.min_testable < 1 or self.dilate_px < 0:
            raise ValueError("controls and min_testable must be positive; dilate_px >= 0")


def decide(
    categories: Sequence[str],
    unknown: Sequence[str],
    predicted_effect: str,
    member_count: int,
    images: Sequence[Dict[str, object]],
    config: PerturbationConfig,
) -> Dict[str, object]:
    """Apply the verdict rules to one hypothesis's scored member images."""

    testable = [image for image in images if image["status"] == "testable"]
    visible = sum(image["status"] in ("testable", "no_control_room") for image in images)
    tallies = {}
    for name in ("centroid", "prompt"):
        exceeding = sum(image["p"][name] <= config.alpha + 1e-12 for image in testable)
        tallies[name] = {
            "exceeding": exceeding,
            "testable": len(testable),
            "passes": 2 * exceeding > len(testable) > 0,
        }

    def result(label: str, reason: Optional[str] = None) -> Dict[str, object]:
        confirmed = None
        if label != "untestable" and len(testable) >= config.min_testable:
            statistics = PREDICTED_STATISTICS[predicted_effect]
            confirmed = all(tallies[name]["passes"] for name in statistics)
        return {
            "label": label,
            "reason": reason,
            "verified": label == "supported",
            "member_count": member_count,
            "visible_members": visible,
            "testable_members": len(testable),
            "tallies": tallies,
            "prediction_confirmed": confirmed,
        }

    if not categories:
        return result("untestable", "empty_categories")
    if unknown:
        return result("untestable", "unknown_categories")
    if all(image["status"] == "absent" for image in images):
        return result("untestable", "no_member_contains_categories")
    if visible < config.min_coverage * member_count:
        return result("not_supported", "low_coverage")
    if len(testable) < config.min_testable:
        return result("untestable", "too_few_testable_members")
    if tallies["centroid"]["passes"]:
        return result("supported")
    if tallies["prompt"]["passes"]:
        return result("detected_not_explanatory")
    return result("not_supported", "no_exceedance")


def load_results(path: Path) -> Dict[str, object]:
    results = json.loads(path.read_text(encoding="utf-8"))
    if results.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported perturbation results schema version")
    return results


def format_summary(results: Dict[str, object]) -> str:
    header = ("hypothesis", "categories", "coverage", "centroid", "prompt", "verdict", "reason")
    rows: List[Sequence[str]] = [header]
    for result in results["hypotheses"]:
        verdict = result["verdict"]
        tallies = verdict["tallies"]
        rows.append(
            (
                result["hypothesis_id"],
                "+".join(result["coco_categories"]) or "-",
                f"{verdict['visible_members']}/{verdict['member_count']}",
                f"{tallies['centroid']['exceeding']}/{tallies['centroid']['testable']}",
                f"{tallies['prompt']['exceeding']}/{tallies['prompt']['testable']}",
                verdict["label"],
                verdict["reason"] or "-",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )


def verified_explanations(
    results: Dict[str, object], explanations_path: Path
) -> Optional[ExplanationArtifact]:
    """Keep only ``supported`` hypotheses, in the original schema; ``None`` if none survive."""

    if sha256_file(explanations_path) != results["inputs"]["explanations_sha256"]:
        raise ValueError("explanations file differs from the one these results verified")
    explanations = ExplanationArtifact.load(explanations_path)
    verified = {h["hypothesis_id"] for h in results["hypotheses"] if h["verdict"]["verified"]}
    nodes = []
    for explanation in explanations.explanations:
        kept = tuple(
            hypothesis
            for index, hypothesis in enumerate(explanation.hypotheses)
            if hypothesis_id(explanation.node_id, index) in verified
        )
        if kept:
            nodes.append(
                NodeExplanation(
                    explanation.node_id, explanation.summary, kept, explanation.response_id
                )
            )
    if not nodes:
        return None
    generator = dict(explanations.generator)
    generator["verification"] = {"method": "image-perturbation", "kept": sorted(verified)}
    return ExplanationArtifact(
        graph_sha256=explanations.graph_sha256,
        annotations_sha256=explanations.annotations_sha256,
        embedding_model_fingerprint=explanations.embedding_model_fingerprint,
        generator=generator,
        context_config=explanations.context_config,
        explanations=tuple(nodes),
    )
