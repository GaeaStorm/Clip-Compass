"""Serializable evidence, hypotheses, and explanation-run artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple


EXPLANATION_CONTEXT_SCHEMA_VERSION = 1
EXPLANATION_ARTIFACT_SCHEMA_VERSION = 1
PREDICTED_EFFECTS = {
    "decrease_text_similarity",
    "move_away_from_node_centroid",
    "both",
}


@dataclass(frozen=True)
class EvidenceItem:
    item_id: str
    image_path: str
    role: str
    captions: Tuple[str, ...]
    category_names: Tuple[str, ...]

    def __post_init__(self) -> None:
        if self.role not in {"target", "contrast"}:
            raise ValueError("evidence role must be 'target' or 'contrast'")
        if not self.item_id or not self.image_path:
            raise ValueError("evidence item ID and image path must not be empty")

    def as_dict(self) -> Dict[str, object]:
        return {
            "item_id": self.item_id,
            "image_path": self.image_path,
            "role": self.role,
            "captions": list(self.captions),
            "category_names": list(self.category_names),
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "EvidenceItem":
        return cls(
            item_id=str(data["item_id"]),
            image_path=str(data["image_path"]),
            role=str(data["role"]),
            captions=tuple(str(value) for value in data["captions"]),
            category_names=tuple(str(value) for value in data["category_names"]),
        )


@dataclass(frozen=True)
class NodeExplanationContext:
    node_id: str
    source_id: str
    component_id: str
    member_count: int
    lens_min: float
    lens_mean: float
    lens_max: float
    neighbor_node_ids: Tuple[str, ...]
    target_category_counts: Mapping[str, int]
    contrast_category_counts: Mapping[str, int]
    evidence: Tuple[EvidenceItem, ...]

    def __post_init__(self) -> None:
        if self.member_count < 1:
            raise ValueError("explanation context must describe a non-empty node")
        if not self.lens_min <= self.lens_mean <= self.lens_max:
            raise ValueError("explanation context has invalid lens statistics")
        target_count = sum(item.role == "target" for item in self.evidence)
        contrast_count = sum(item.role == "contrast" for item in self.evidence)
        if target_count < 1 or contrast_count < 1:
            raise ValueError("explanation context requires target and contrast evidence")
        if len({item.item_id for item in self.evidence}) != len(self.evidence):
            raise ValueError("explanation evidence item IDs must be unique")

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "source_id": self.source_id,
            "component_id": self.component_id,
            "member_count": self.member_count,
            "lens": {
                "min": self.lens_min,
                "mean": self.lens_mean,
                "max": self.lens_max,
            },
            "neighbor_node_ids": list(self.neighbor_node_ids),
            "target_category_counts": dict(self.target_category_counts),
            "contrast_category_counts": dict(self.contrast_category_counts),
            "evidence": [item.as_dict() for item in self.evidence],
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "NodeExplanationContext":
        lens = data["lens"]
        return cls(
            node_id=str(data["node_id"]),
            source_id=str(data["source_id"]),
            component_id=str(data["component_id"]),
            member_count=int(data["member_count"]),
            lens_min=float(lens["min"]),
            lens_mean=float(lens["mean"]),
            lens_max=float(lens["max"]),
            neighbor_node_ids=tuple(str(value) for value in data["neighbor_node_ids"]),
            target_category_counts={
                str(key): int(value)
                for key, value in data["target_category_counts"].items()
            },
            contrast_category_counts={
                str(key): int(value)
                for key, value in data["contrast_category_counts"].items()
            },
            evidence=tuple(
                EvidenceItem.from_mapping(value) for value in data["evidence"]
            ),
        )


@dataclass(frozen=True)
class ExplanationContextBundle:
    graph_sha256: str
    annotations_sha256: str
    embedding_model_fingerprint: str
    config: Mapping[str, object]
    contexts: Tuple[NodeExplanationContext, ...]

    def __post_init__(self) -> None:
        node_ids = tuple(context.node_id for context in self.contexts)
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("explanation contexts must have unique node IDs")

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": EXPLANATION_CONTEXT_SCHEMA_VERSION,
            "graph_sha256": self.graph_sha256,
            "annotations_sha256": self.annotations_sha256,
            "embedding_model_fingerprint": self.embedding_model_fingerprint,
            "config": dict(self.config),
            "contexts": [context.as_dict() for context in self.contexts],
        }

    def save(self, path: Path) -> Path:
        return _atomic_json(path, self.as_dict())

    @classmethod
    def load(cls, path: Path) -> "ExplanationContextBundle":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != EXPLANATION_CONTEXT_SCHEMA_VERSION:
            raise ValueError("unsupported explanation-context schema version")
        return cls(
            graph_sha256=str(data["graph_sha256"]),
            annotations_sha256=str(data["annotations_sha256"]),
            embedding_model_fingerprint=str(data["embedding_model_fingerprint"]),
            config=data["config"],
            contexts=tuple(
                NodeExplanationContext.from_mapping(value)
                for value in data["contexts"]
            ),
        )


@dataclass(frozen=True)
class ExplanationHypothesis:
    name: str
    description: str
    positive_prompt: str
    negative_prompt: str
    coco_categories: Tuple[str, ...]
    predicted_effect: str
    rationale: str
    supporting_item_ids: Tuple[str, ...]
    counterexample_item_ids: Tuple[str, ...]
    confidence: float

    def __post_init__(self) -> None:
        required = (
            self.name,
            self.description,
            self.positive_prompt,
            self.negative_prompt,
            self.rationale,
        )
        if any(not value.strip() for value in required):
            raise ValueError("hypothesis text fields must not be empty")
        if self.predicted_effect not in PREDICTED_EFFECTS:
            raise ValueError("hypothesis has an unsupported predicted effect")
        if not 0.0 <= self.confidence <= 1.0:
            raise ValueError("hypothesis confidence must be in [0, 1]")

    def as_dict(self) -> Dict[str, object]:
        return {
            "name": self.name,
            "description": self.description,
            "positive_prompt": self.positive_prompt,
            "negative_prompt": self.negative_prompt,
            "coco_categories": list(self.coco_categories),
            "predicted_effect": self.predicted_effect,
            "rationale": self.rationale,
            "supporting_item_ids": list(self.supporting_item_ids),
            "counterexample_item_ids": list(self.counterexample_item_ids),
            "confidence": self.confidence,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ExplanationHypothesis":
        return cls(
            name=str(data["name"]),
            description=str(data["description"]),
            positive_prompt=str(data["positive_prompt"]),
            negative_prompt=str(data["negative_prompt"]),
            coco_categories=tuple(str(value) for value in data["coco_categories"]),
            predicted_effect=str(data["predicted_effect"]),
            rationale=str(data["rationale"]),
            supporting_item_ids=tuple(
                str(value) for value in data["supporting_item_ids"]
            ),
            counterexample_item_ids=tuple(
                str(value) for value in data["counterexample_item_ids"]
            ),
            confidence=float(data["confidence"]),
        )


@dataclass(frozen=True)
class NodeExplanation:
    node_id: str
    summary: str
    hypotheses: Tuple[ExplanationHypothesis, ...]
    response_id: str = ""

    def __post_init__(self) -> None:
        if not self.node_id or not self.summary.strip() or not self.hypotheses:
            raise ValueError("node explanation must include a summary and hypotheses")

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "summary": self.summary,
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
            "response_id": self.response_id,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "NodeExplanation":
        return cls(
            node_id=str(data["node_id"]),
            summary=str(data["summary"]),
            hypotheses=tuple(
                ExplanationHypothesis.from_mapping(value)
                for value in data["hypotheses"]
            ),
            response_id=str(data.get("response_id", "")),
        )


@dataclass(frozen=True)
class ExplanationArtifact:
    graph_sha256: str
    annotations_sha256: str
    embedding_model_fingerprint: str
    generator: Mapping[str, object]
    context_config: Mapping[str, object]
    explanations: Tuple[NodeExplanation, ...]

    def __post_init__(self) -> None:
        node_ids = tuple(explanation.node_id for explanation in self.explanations)
        if not node_ids or len(set(node_ids)) != len(node_ids):
            raise ValueError("explanations must have unique node IDs")

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": EXPLANATION_ARTIFACT_SCHEMA_VERSION,
            "graph_sha256": self.graph_sha256,
            "annotations_sha256": self.annotations_sha256,
            "embedding_model_fingerprint": self.embedding_model_fingerprint,
            "generator": dict(self.generator),
            "context_config": dict(self.context_config),
            "explanations": [value.as_dict() for value in self.explanations],
        }

    def save(self, path: Path) -> Path:
        return _atomic_json(path, self.as_dict())

    @classmethod
    def load(cls, path: Path) -> "ExplanationArtifact":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != EXPLANATION_ARTIFACT_SCHEMA_VERSION:
            raise ValueError("unsupported explanation-artifact schema version")
        return cls(
            graph_sha256=str(data["graph_sha256"]),
            annotations_sha256=str(data["annotations_sha256"]),
            embedding_model_fingerprint=str(data["embedding_model_fingerprint"]),
            generator=data["generator"],
            context_config=data["context_config"],
            explanations=tuple(
                NodeExplanation.from_mapping(value) for value in data["explanations"]
            ),
        )


def _atomic_json(path: Path, data: Mapping[str, object]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(
        json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    os.replace(partial, path)
    return path
