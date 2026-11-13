"""Verdict rules for perturbation results, plus summary and verified-hypothesis export.

For each node member that contains the hypothesis categories, the object mask and
``controls`` area-matched random masks are filled and re-embedded (see ``service``).
An image *exceeds* on a statistic when the object mask beats its controls
(empirical p <= alpha):

- ``centroid``: increase in distance from the node centroid (leave-one-out: the
  image under test is excluded from its own centroid)
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

from pathlib import Path
from typing import List, Mapping, Optional, Sequence, Tuple

from clip_compass.data.coco_instances import sha256_file
from clip_compass.explanations.types import (
    ExplanationArtifact,
    NodeExplanation,
    hypothesis_id,
)

from .types import (
    STATISTICS,
    ImageResult,
    Label,
    PerturbationConfig,
    Statistic,
    Tally,
    Verdict,
    VerificationResults,
)

PREDICTED_STATISTICS: Mapping[str, Tuple[Statistic, ...]] = {
    "decrease_text_similarity": ("prompt",),
    "move_away_from_node_centroid": ("centroid",),
    "both": ("centroid", "prompt"),
}


def decide(
    images: Sequence[ImageResult],
    *,
    categories: Sequence[str],
    unknown: Sequence[str],
    predicted_effect: str,
    member_count: int,
    config: PerturbationConfig,
) -> Verdict:
    """Apply the verdict rules to one hypothesis's scored member images."""

    scores = [image.score for image in images if image.score is not None]
    visible = sum(image.status in ("testable", "no_control_room") for image in images)
    tallies = {}
    for name in STATISTICS:
        exceeding = sum(score.p[name] <= config.alpha + 1e-12 for score in scores)
        tallies[name] = Tally(exceeding, len(scores), 2 * exceeding > len(scores) > 0)

    def result(label: Label, reason: Optional[str] = None) -> Verdict:
        confirmed = None
        if label != "untestable" and len(scores) >= config.min_testable:
            statistics = PREDICTED_STATISTICS[predicted_effect]
            confirmed = all(tallies[name].passes for name in statistics)
        return Verdict(label, reason, member_count, visible, len(scores), tallies, confirmed)

    if not categories:
        return result("untestable", "empty_categories")
    if unknown:
        return result("untestable", "unknown_categories")
    if all(image.status == "absent" for image in images):
        return result("untestable", "no_member_contains_categories")
    if visible < config.min_coverage * member_count:
        return result("not_supported", "low_coverage")
    if len(scores) < config.min_testable:
        return result("untestable", "too_few_testable_members")
    if tallies["centroid"].passes:
        return result("supported")
    if tallies["prompt"].passes:
        return result("detected_not_explanatory")
    return result("not_supported", "no_exceedance")


def format_summary(results: VerificationResults) -> str:
    header = ("hypothesis", "categories", "coverage", "centroid", "prompt", "verdict", "reason")
    rows: List[Sequence[str]] = [header]
    for result in results.hypotheses:
        verdict = result.verdict
        centroid, prompt = verdict.tallies["centroid"], verdict.tallies["prompt"]
        rows.append(
            (
                result.hypothesis_id,
                "+".join(result.coco_categories) or "-",
                f"{verdict.visible_members}/{verdict.member_count}",
                f"{centroid.exceeding}/{centroid.testable}",
                f"{prompt.exceeding}/{prompt.testable}",
                verdict.label,
                verdict.reason or "-",
            )
        )
    widths = [max(len(row[column]) for row in rows) for column in range(len(header))]
    return "\n".join(
        "  ".join(cell.ljust(width) for cell, width in zip(row, widths)).rstrip()
        for row in rows
    )


def verified_explanations(
    results: VerificationResults, explanations_path: Path
) -> Optional[ExplanationArtifact]:
    """Keep only ``supported`` hypotheses, in the original schema; ``None`` if none survive."""

    if sha256_file(explanations_path) != results.inputs["explanations_sha256"]:
        raise ValueError("explanations file differs from the one these results verified")
    explanations = ExplanationArtifact.load(explanations_path)
    verified = {h.hypothesis_id for h in results.hypotheses if h.verdict.verified}
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
