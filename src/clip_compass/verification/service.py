"""Turn per-image perturbation results into one verdict per hypothesis.

Pipeline position: this stage reads the ``verification.json`` written by
``clip-compass perturb run`` (Dhruv's masking engine). That file holds, for each
node member, the object edit's effect and the effects of K area-matched random
controls on two statistics. This module asks, for each hypothesis, whether the
object edits beat their controls across the group more than chance allows.

How it differs from the engine's own majority vote: instead of thresholding
each image at p <= alpha and counting, it uses every image's full rank among its
controls in one exact test (see ``stats.rank_sum_p_value``), then corrects for
testing many hypotheses with Benjamini-Hochberg within each statistic.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Mapping, Optional

import numpy as np

from clip_compass.data.coco_instances import sha256_file
from clip_compass.explanations.types import ExplanationArtifact, hypothesis_id
from clip_compass.perturbation.types import HypothesisResult, VerificationResults

from .stats import benjamini_hochberg, object_rank, rank_sum_p_value
from .types import (
    PREDICTED_STATISTICS,
    STATISTICS,
    HypothesisVerdict,
    ImageResult,
    StatisticTest,
    VerdictArtifact,
    VerdictConfig,
)

VISIBLE_STATUSES = ("testable", "no_control_room")


def _image_results(hypothesis: HypothesisResult) -> List[ImageResult]:
    results = []
    for image in hypothesis.images:
        if image.status != "testable" or image.score is None:
            results.append(ImageResult(hypothesis.hypothesis_id, image.item_id, image.status))
            continue
        score = image.score
        counts = {len(score.controls[statistic]) for statistic in STATISTICS}
        if len(counts) != 1:
            raise ValueError(f"image {image.item_id} has unequal control counts per statistic")
        ranks = {
            statistic: object_rank(score.object[statistic], score.controls[statistic])
            for statistic in STATISTICS
        }
        results.append(
            ImageResult(hypothesis.hypothesis_id, image.item_id, image.status, counts.pop(), ranks)
        )
    return results


def _test(statistic: str, images: List[ImageResult]) -> StatisticTest:
    ranks = [image.ranks[statistic] for image in images]
    counts = [image.control_count for image in images]
    percentiles = [image.percentile(statistic) for image in images]
    return StatisticTest(
        statistic=statistic,
        image_count=len(images),
        mean_percentile=float(np.mean(percentiles)),
        p_value=rank_sum_p_value(ranks, counts),
    )


def _label(
    hypothesis: HypothesisResult,
    images: List[ImageResult],
    tests: Mapping[str, StatisticTest],
    config: VerdictConfig,
) -> tuple:
    member_count = hypothesis.member_count
    visible = sum(image.status in VISIBLE_STATUSES for image in images)
    testable = sum(image.status == "testable" for image in images)
    if not hypothesis.coco_categories:
        return "untestable", "empty_categories"
    if hypothesis.unknown_categories:
        return "untestable", "unknown_categories"
    if not images or all(image.status == "absent" for image in images):
        return "untestable", "no_member_contains_categories"
    # The engine's rule: an object that most of the group doesn't visibly contain
    # cannot be what defines the group.
    if visible < config.min_coverage * member_count:
        return "not_supported", "low_coverage"
    if testable < config.min_testable:
        return "insufficient_evidence", "too_few_testable_members"
    if tests["centroid"].passes(config.alpha):
        return "supported", None
    if tests["prompt"].passes(config.alpha):
        return "detected_not_explanatory", None
    return "not_supported", "no_effect"


def aggregate_verdicts(
    results_path: Path,
    explanations_path: Path,
    *,
    config: Optional[VerdictConfig] = None,
) -> VerdictArtifact:
    """Read ``perturb run`` results and return calibrated per-hypothesis verdicts."""

    config = config or VerdictConfig()
    results = VerificationResults.load(results_path)
    explanations_sha256 = sha256_file(explanations_path)
    if results.inputs["explanations_sha256"] != explanations_sha256:
        raise ValueError("perturbation results were produced from different explanations")
    explanations = ExplanationArtifact.load(explanations_path)
    confidence: Dict[str, float] = {
        hypothesis_id(explanation.node_id, index): hypothesis.confidence
        for explanation in explanations.explanations
        for index, hypothesis in enumerate(explanation.hypotheses)
    }

    hypotheses = list(results.hypotheses)
    images_by_hypothesis = [_image_results(hypothesis) for hypothesis in hypotheses]
    tests_by_hypothesis: List[Dict[str, StatisticTest]] = []
    for images in images_by_hypothesis:
        testable = [image for image in images if image.status == "testable"]
        if len(testable) >= config.min_testable:
            tests_by_hypothesis.append(
                {statistic: _test(statistic, testable) for statistic in STATISTICS}
            )
        else:
            tests_by_hypothesis.append({})

    # Correct within each statistic across every hypothesis that was tested.
    for statistic in STATISTICS:
        q_values = benjamini_hochberg(
            [tests[statistic].p_value if tests else None for tests in tests_by_hypothesis]
        )
        for tests, q_value in zip(tests_by_hypothesis, q_values):
            if tests:
                old = tests[statistic]
                tests[statistic] = StatisticTest(
                    old.statistic, old.image_count, old.mean_percentile, old.p_value, q_value
                )

    verdicts = []
    for hypothesis, images, tests in zip(hypotheses, images_by_hypothesis, tests_by_hypothesis):
        label, reason = _label(hypothesis, images, tests, config)
        predicted = hypothesis.predicted_effect
        confirmed = None
        if tests and label != "untestable":
            confirmed = all(
                tests[name].passes(config.alpha) for name in PREDICTED_STATISTICS[predicted]
            )
        verdicts.append(
            HypothesisVerdict(
                hypothesis_id=hypothesis.hypothesis_id,
                node_id=hypothesis.node_id,
                name=hypothesis.name,
                coco_categories=tuple(hypothesis.coco_categories),
                predicted_effect=predicted,
                confidence=confidence.get(hypothesis.hypothesis_id),
                member_count=hypothesis.member_count,
                visible_members=sum(image.status in VISIBLE_STATUSES for image in images),
                testable_members=sum(image.status == "testable" for image in images),
                tests=tests,
                label=label,
                reason=reason,
                prediction_confirmed=confirmed,
            )
        )

    return VerdictArtifact(
        perturbation_results_sha256=sha256_file(results_path),
        explanations_sha256=explanations_sha256,
        graph_sha256=str(results.inputs["graph_sha256"]),
        model_fingerprint=str(results.model_fingerprint),
        config=config,
        hypotheses=tuple(verdicts),
        images=tuple(image for images in images_by_hypothesis for image in images),
    )
