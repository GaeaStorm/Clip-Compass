"""Serializable verdicts and text-check scores."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Mapping, Optional, Sequence, Tuple

VERDICT_SCHEMA_VERSION = 1
BASELINE_SCHEMA_VERSION = 1

# Statistics written per image by ``perturb run``. Larger values mean the edit
# moved the image further: "centroid" is the increase in distance from the node
# centroid, "prompt" the drop in similarity to the hypothesis's positive prompt.
STATISTICS = ("centroid", "prompt")
PREDICTED_STATISTICS = {
    "decrease_text_similarity": ("prompt",),
    "move_away_from_node_centroid": ("centroid",),
    "both": ("centroid", "prompt"),
}
# supported: hiding the object moves images away from their group (it explains the node).
# detected_not_explanatory: hiding it changes what CLIP sees, but not group membership.
LABELS = (
    "supported",
    "detected_not_explanatory",
    "not_supported",
    "insufficient_evidence",
    "untestable",
)


@dataclass(frozen=True)
class VerdictConfig:
    """Thresholds that turn per-image results into one verdict per hypothesis."""

    alpha: float = 0.05
    min_testable: int = 2
    min_coverage: float = 0.5

    def __post_init__(self) -> None:
        if not 0.0 < self.alpha < 1.0:
            raise ValueError("alpha must be in (0, 1)")
        if self.min_testable < 1:
            raise ValueError("min_testable must be positive")
        if not 0.0 <= self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in [0, 1]")

    def as_dict(self) -> Dict[str, object]:
        return {
            "alpha": self.alpha,
            "min_testable": self.min_testable,
            "min_coverage": self.min_coverage,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "VerdictConfig":
        return cls(
            alpha=float(data["alpha"]),
            min_testable=int(data["min_testable"]),
            min_coverage=float(data["min_coverage"]),
        )


@dataclass(frozen=True)
class ImageResult:
    """One node member's outcome for one hypothesis."""

    hypothesis_id: str
    item_id: str
    status: str
    control_count: int = 0
    ranks: Mapping[str, int] = field(default_factory=dict)

    def percentile(self, statistic: str) -> Optional[float]:
        """Share of random controls the object edit beat on this statistic."""

        if statistic not in self.ranks or not self.control_count:
            return None
        return self.ranks[statistic] / self.control_count

    def as_dict(self) -> Dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "item_id": self.item_id,
            "status": self.status,
            "control_count": self.control_count,
            "ranks": dict(self.ranks),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ImageResult":
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            item_id=str(data["item_id"]),
            status=str(data["status"]),
            control_count=int(data["control_count"]),
            ranks={str(key): int(value) for key, value in dict(data["ranks"]).items()},
        )


@dataclass(frozen=True)
class StatisticTest:
    """Group-level test of one statistic across a hypothesis's testable images."""

    statistic: str
    image_count: int
    mean_percentile: Optional[float]
    p_value: Optional[float]
    q_value: Optional[float] = None

    def passes(self, alpha: float) -> bool:
        return self.q_value is not None and self.q_value <= alpha

    def as_dict(self) -> Dict[str, object]:
        return {
            "statistic": self.statistic,
            "image_count": self.image_count,
            "mean_percentile": self.mean_percentile,
            "p_value": self.p_value,
            "q_value": self.q_value,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "StatisticTest":
        def optional(key: str) -> Optional[float]:
            return None if data.get(key) is None else float(data[key])

        return cls(
            statistic=str(data["statistic"]),
            image_count=int(data["image_count"]),
            mean_percentile=optional("mean_percentile"),
            p_value=optional("p_value"),
            q_value=optional("q_value"),
        )


@dataclass(frozen=True)
class HypothesisVerdict:
    hypothesis_id: str
    node_id: str
    name: str
    coco_categories: Tuple[str, ...]
    predicted_effect: str
    confidence: Optional[float]
    member_count: int
    visible_members: int
    testable_members: int
    tests: Mapping[str, StatisticTest]
    label: str
    reason: Optional[str]
    prediction_confirmed: Optional[bool]

    def __post_init__(self) -> None:
        if self.label not in LABELS:
            raise ValueError(f"label must be one of {LABELS}")

    def as_dict(self) -> Dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "node_id": self.node_id,
            "name": self.name,
            "coco_categories": list(self.coco_categories),
            "predicted_effect": self.predicted_effect,
            "confidence": self.confidence,
            "member_count": self.member_count,
            "visible_members": self.visible_members,
            "testable_members": self.testable_members,
            "tests": {name: test.as_dict() for name, test in self.tests.items()},
            "label": self.label,
            "reason": self.reason,
            "prediction_confirmed": self.prediction_confirmed,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "HypothesisVerdict":
        confidence = data.get("confidence")
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            node_id=str(data["node_id"]),
            name=str(data["name"]),
            coco_categories=tuple(str(value) for value in data["coco_categories"]),
            predicted_effect=str(data["predicted_effect"]),
            confidence=None if confidence is None else float(confidence),
            member_count=int(data["member_count"]),
            visible_members=int(data["visible_members"]),
            testable_members=int(data["testable_members"]),
            tests={
                str(name): StatisticTest.from_mapping(value)
                for name, value in dict(data["tests"]).items()
            },
            label=str(data["label"]),
            reason=None if data.get("reason") is None else str(data["reason"]),
            prediction_confirmed=data.get("prediction_confirmed"),
        )


@dataclass(frozen=True)
class VerdictArtifact:
    perturbation_results_sha256: str
    explanations_sha256: str
    graph_sha256: str
    model_fingerprint: str
    config: VerdictConfig
    hypotheses: Tuple[HypothesisVerdict, ...]
    images: Tuple[ImageResult, ...] = ()

    def summary(self) -> Dict[str, object]:
        counts = {label: 0 for label in LABELS}
        for hypothesis in self.hypotheses:
            counts[hypothesis.label] += 1
        return {"hypothesis_count": len(self.hypotheses), "label_counts": counts}

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": VERDICT_SCHEMA_VERSION,
            "perturbation_results_sha256": self.perturbation_results_sha256,
            "explanations_sha256": self.explanations_sha256,
            "graph_sha256": self.graph_sha256,
            "model_fingerprint": self.model_fingerprint,
            "config": self.config.as_dict(),
            "summary": self.summary(),
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
            "images": [image.as_dict() for image in self.images],
        }

    def save(self, output_dir: Path) -> Mapping[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = _atomic_json(output_dir / "verdicts.json", self.as_dict())
        hypotheses_csv = _atomic_csv(
            output_dir / "hypotheses.csv", [_hypothesis_row(value) for value in self.hypotheses]
        )
        images_csv = _atomic_csv(
            output_dir / "images.csv", [_image_row(value) for value in self.images]
        )
        return {"json": json_path, "hypotheses_csv": hypotheses_csv, "images_csv": images_csv}

    @classmethod
    def load(cls, path: Path) -> "VerdictArtifact":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != VERDICT_SCHEMA_VERSION:
            raise ValueError("unsupported verdict schema version")
        return cls(
            perturbation_results_sha256=str(data["perturbation_results_sha256"]),
            explanations_sha256=str(data["explanations_sha256"]),
            graph_sha256=str(data["graph_sha256"]),
            model_fingerprint=str(data["model_fingerprint"]),
            config=VerdictConfig.from_mapping(data["config"]),
            hypotheses=tuple(HypothesisVerdict.from_mapping(value) for value in data["hypotheses"]),
            images=tuple(ImageResult.from_mapping(value) for value in data["images"]),
        )


@dataclass(frozen=True)
class BaselineScore:
    """Text-only check: does the positive prompt score node members above the rest?"""

    node_id: str
    hypothesis_index: int
    hypothesis_name: str
    member_count: int
    outside_count: int
    inside_mean: float
    outside_mean: float
    auroc: Optional[float]
    accepted: bool

    @property
    def margin(self) -> float:
        return self.inside_mean - self.outside_mean

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "hypothesis_index": self.hypothesis_index,
            "hypothesis_name": self.hypothesis_name,
            "member_count": self.member_count,
            "outside_count": self.outside_count,
            "inside_mean": self.inside_mean,
            "outside_mean": self.outside_mean,
            "margin": self.margin,
            "auroc": self.auroc,
            "accepted": self.accepted,
        }


@dataclass(frozen=True)
class BaselineArtifact:
    graph_sha256: str
    explanations_sha256: str
    embedding_model_fingerprint: str
    scores: Tuple[BaselineScore, ...] = field(default_factory=tuple)

    def summary(self) -> Dict[str, object]:
        return {
            "hypothesis_count": len(self.scores),
            "accepted_count": sum(score.accepted for score in self.scores),
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": BASELINE_SCHEMA_VERSION,
            "graph_sha256": self.graph_sha256,
            "explanations_sha256": self.explanations_sha256,
            "embedding_model_fingerprint": self.embedding_model_fingerprint,
            "summary": self.summary(),
            "scores": [score.as_dict() for score in self.scores],
        }

    def save(self, output_dir: Path) -> Mapping[str, Path]:
        output_dir.mkdir(parents=True, exist_ok=True)
        json_path = _atomic_json(output_dir / "baseline.json", self.as_dict())
        csv_path = _atomic_csv(
            output_dir / "baseline.csv", [score.as_dict() for score in self.scores]
        )
        return {"json": json_path, "csv": csv_path}


def _hypothesis_row(hypothesis: HypothesisVerdict) -> Dict[str, object]:
    row: Dict[str, object] = {
        "hypothesis_id": hypothesis.hypothesis_id,
        "node_id": hypothesis.node_id,
        "name": hypothesis.name,
        "coco_categories": ";".join(hypothesis.coco_categories),
        "predicted_effect": hypothesis.predicted_effect,
        "confidence": hypothesis.confidence,
        "member_count": hypothesis.member_count,
        "visible_members": hypothesis.visible_members,
        "testable_members": hypothesis.testable_members,
        "label": hypothesis.label,
        "reason": hypothesis.reason,
        "prediction_confirmed": hypothesis.prediction_confirmed,
    }
    for statistic in STATISTICS:
        test = hypothesis.tests.get(statistic)
        for key in ("mean_percentile", "p_value", "q_value"):
            row[f"{statistic}_{key}"] = None if test is None else test.as_dict()[key]
    return row


def _image_row(image: ImageResult) -> Dict[str, object]:
    row: Dict[str, object] = {
        "hypothesis_id": image.hypothesis_id,
        "item_id": image.item_id,
        "status": image.status,
        "control_count": image.control_count,
    }
    for statistic in STATISTICS:
        row[f"{statistic}_rank"] = image.ranks.get(statistic)
        row[f"{statistic}_percentile"] = image.percentile(statistic)
    return row


def _atomic_json(path: Path, data: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(
        json.dumps(data, indent=2, sort_keys=True, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    os.replace(partial, path)
    return path


def _atomic_csv(path: Path, rows: Sequence[Mapping[str, object]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    fieldnames = list(rows[0].keys()) if rows else []
    with partial.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: "" if value is None else value for key, value in row.items()})
    os.replace(partial, path)
    return path
