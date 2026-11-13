"""Control hypotheses whose right answer is known in advance, built from COCO labels.

The judges (the agent itself, the text check, the masking test) can only be
compared fairly on hypotheses where we already know what a good judge should say.
This module writes three kinds, in the same explanation format the agent produces,
so they run through the unchanged pipeline:

* ``obvious`` (should be accepted): a COCO category present in most of the node's
  members and far more common inside the node than outside it.
* ``common`` (should be rejected): a category present in most members but about as
  common outside the node, so it cannot be what sets the node apart.
* ``swapped`` (should be rejected): another node's real hypothesis, taken from the
  node whose categories are least enriched in this one.

A category counts as present in an image when one of its instances covers at
least ``min_area_fraction`` of the image, so tiny background objects are ignored.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Sequence, Set, Tuple

from clip_compass.data.coco_instances import CocoInstanceSubset, sha256_file
from clip_compass.explanations.types import (
    ExplanationArtifact,
    ExplanationHypothesis,
    NodeExplanation,
    hypothesis_id,
)
from clip_compass.mapper.graph import MapperGraphArtifact

from .types import _atomic_json

CONTROL_SCHEMA_VERSION = 1
EXPECTED = {"obvious": "accept", "common": "reject", "swapped": "reject"}


@dataclass(frozen=True)
class ControlConfig:
    min_area_fraction: float = 0.01
    min_coverage: float = 0.5
    min_enrichment: float = 2.0
    max_common_enrichment: float = 1.25

    def __post_init__(self) -> None:
        if not 0.0 <= self.min_area_fraction < 1.0:
            raise ValueError("min_area_fraction must be in [0, 1)")
        if not 0.0 < self.min_coverage <= 1.0:
            raise ValueError("min_coverage must be in (0, 1]")
        if self.min_enrichment <= 1.0:
            raise ValueError("min_enrichment must be above 1")
        if self.max_common_enrichment < 1.0 or self.max_common_enrichment >= self.min_enrichment:
            raise ValueError("max_common_enrichment must be in [1, min_enrichment)")

    def as_dict(self) -> Dict[str, object]:
        return {
            "min_area_fraction": self.min_area_fraction,
            "min_coverage": self.min_coverage,
            "min_enrichment": self.min_enrichment,
            "max_common_enrichment": self.max_common_enrichment,
        }


@dataclass(frozen=True)
class ControlKey:
    """What a good judge should say about one control hypothesis, and why."""

    hypothesis_id: str
    node_id: str
    kind: str
    expected: str
    categories: Tuple[str, ...]
    inside_rate: float
    outside_rate: float
    source_node_id: Optional[str] = None

    def as_dict(self) -> Dict[str, object]:
        return {
            "hypothesis_id": self.hypothesis_id,
            "node_id": self.node_id,
            "kind": self.kind,
            "expected": self.expected,
            "categories": list(self.categories),
            "inside_rate": self.inside_rate,
            "outside_rate": self.outside_rate,
            "source_node_id": self.source_node_id,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "ControlKey":
        source = data.get("source_node_id")
        return cls(
            hypothesis_id=str(data["hypothesis_id"]),
            node_id=str(data["node_id"]),
            kind=str(data["kind"]),
            expected=str(data["expected"]),
            categories=tuple(str(value) for value in data["categories"]),
            inside_rate=float(data["inside_rate"]),
            outside_rate=float(data["outside_rate"]),
            source_node_id=None if source is None else str(source),
        )


def present_categories(
    annotations: CocoInstanceSubset, item_ids: Sequence[str], min_area_fraction: float
) -> Dict[str, Set[str]]:
    """Category names visibly present in each image (instances above the area floor)."""

    categories = annotations.categories_by_id
    images = {image.id: image for image in annotations.images}
    present: Dict[str, Set[str]] = {}
    for item_id in item_ids:
        image = images[item_id]
        floor = min_area_fraction * image.width * image.height
        present[item_id] = {
            categories[instance.category_id].name
            for instance in annotations.annotations_for(item_id)
            if instance.area >= floor
        }
    return present


def _rates(
    names: Sequence[str],
    inside: Sequence[str],
    outside: Sequence[str],
    present: Mapping[str, Set[str]],
) -> Tuple[float, float]:
    """Share of inside and outside images containing every one of ``names``."""

    def rate(items: Sequence[str]) -> float:
        if not items:
            return 0.0
        return sum(all(name in present[item] for name in names) for item in items) / len(items)

    return rate(inside), rate(outside)


def _enrichment(inside_rate: float, outside_rate: float) -> float:
    if outside_rate == 0.0:
        return float("inf") if inside_rate > 0.0 else 0.0
    return inside_rate / outside_rate


def _control_key(
    node_id: str,
    inside: Sequence[str],
    outside: Sequence[str],
    present: Mapping[str, Set[str]],
    kind: str,
    names: Tuple[str, ...],
    source: Optional[str] = None,
) -> ControlKey:
    inside_rate, outside_rate = _rates(names, inside, outside, present)
    return ControlKey("", node_id, kind, EXPECTED[kind], names, inside_rate, outside_rate, source)


def _supporters(inside: Sequence[str], present: Mapping[str, Set[str]], name: str) -> List[str]:
    return [item for item in inside if name in present[item]]


def _article(word: str) -> str:
    return "an" if word[:1].lower() in "aeiou" else "a"


def _label_hypothesis(
    category: str,
    kind: str,
    inside_count: int,
    member_count: int,
    outside_rate: float,
    supporting: Sequence[str],
) -> ExplanationHypothesis:
    article = _article(category)
    if kind == "obvious":
        description = f"Control: the group is defined by photos containing {article} {category}."
        rationale = (
            f"Label-based control: {category} appears in {inside_count} of {member_count} "
            f"group photos and {outside_rate:.0%} of other photos."
        )
    else:
        description = f"Control: {category} is common in the group but just as common elsewhere."
        rationale = (
            f"Label-based control: {category} appears in {inside_count} of {member_count} "
            f"group photos, but also in {outside_rate:.0%} of other photos."
        )
    return ExplanationHypothesis(
        name=f"Contains {article} {category}" + (" (common)" if kind == "common" else ""),
        description=description,
        positive_prompt=f"a photo containing {article} {category}",
        negative_prompt=f"a photo without {article} {category}",
        coco_categories=(category,),
        predicted_effect="both",
        rationale=rationale,
        supporting_item_ids=tuple(supporting[:5]),
        counterexample_item_ids=(),
        confidence=0.5,
    )


def build_controls(
    graph: MapperGraphArtifact,
    annotations: CocoInstanceSubset,
    *,
    graph_path: Path,
    annotations_path: Path,
    explanations: Optional[ExplanationArtifact] = None,
    config: Optional[ControlConfig] = None,
) -> Tuple[ExplanationArtifact, List[ControlKey]]:
    """Return control hypotheses (as an explanation artifact) and their answer key."""

    config = config or ControlConfig()
    if annotations.item_ids != graph.item_ids:
        raise ValueError("graph and annotation item IDs must align in order")
    if explanations is not None and explanations.graph_sha256 != sha256_file(graph_path):
        raise ValueError("explanations were prepared from a different Mapper graph")
    present = present_categories(annotations, graph.item_ids, config.min_area_fraction)
    all_names = sorted({name for names in present.values() for name in names})
    real = {}
    if explanations is not None:
        real = {
            explanation.node_id: explanation.hypotheses[0]
            for explanation in explanations.explanations
            if explanation.hypotheses
        }

    nodes_out: List[NodeExplanation] = []
    keys: List[ControlKey] = []
    for node in graph.nodes:
        inside = list(node.item_ids)
        outside = [item for item in graph.item_ids if item not in set(inside)]
        if not outside:
            continue
        stats = {name: _rates((name,), inside, outside, present) for name in all_names}
        chosen: List[Tuple[str, ExplanationHypothesis, ControlKey]] = []
        key = partial(_control_key, node.id, inside, outside, present)
        supporters = partial(_supporters, inside, present)

        covered = [name for name in all_names if stats[name][0] >= config.min_coverage]
        obvious = sorted(
            (name for name in covered if _enrichment(*stats[name]) >= config.min_enrichment),
            key=lambda name: (-_enrichment(*stats[name]), -stats[name][0], name),
        )
        if obvious:
            name = obvious[0]
            count = len(supporters(name))
            chosen.append(
                (
                    "obvious",
                    _label_hypothesis(
                        name, "obvious", count, len(inside), stats[name][1], supporters(name)
                    ),
                    key("obvious", (name,)),
                )
            )
        common = sorted(
            (name for name in covered if _enrichment(*stats[name]) <= config.max_common_enrichment),
            key=lambda name: (-stats[name][0], name),
        )
        if common:
            name = common[0]
            count = len(supporters(name))
            chosen.append(
                (
                    "common",
                    _label_hypothesis(
                        name, "common", count, len(inside), stats[name][1], supporters(name)
                    ),
                    key("common", (name,)),
                )
            )
        # Swap in another node's real hypothesis. Skip donors whose objects are enriched
        # here (they might genuinely describe this node too). Among the rest, prefer the
        # one whose objects appear in most of this node's photos: an absent object is
        # rejected for lack of anything to hide, which tells us nothing about a judge.
        donors = []
        for donor_id, hypothesis in real.items():
            names = tuple(hypothesis.coco_categories)
            if donor_id == node.id or not names:
                continue
            inside_rate, outside_rate = _rates(names, inside, outside, present)
            donors.append(
                (_enrichment(inside_rate, outside_rate), inside_rate, donor_id, hypothesis)
            )
        donors = [donor for donor in donors if donor[0] <= config.max_common_enrichment]
        if donors:
            _, _, donor_id, hypothesis = min(
                donors, key=lambda donor: (-donor[1], donor[0], donor[2])
            )
            swapped = ExplanationHypothesis(
                name=f"{hypothesis.name} (swapped from {donor_id})",
                description=hypothesis.description,
                positive_prompt=hypothesis.positive_prompt,
                negative_prompt=hypothesis.negative_prompt,
                coco_categories=hypothesis.coco_categories,
                predicted_effect=hypothesis.predicted_effect,
                rationale=f"Swapped control: written for {donor_id}, assigned to {node.id}.",
                supporting_item_ids=(),
                counterexample_item_ids=(),
                confidence=hypothesis.confidence,
            )
            chosen.append(
                ("swapped", swapped, key("swapped", tuple(hypothesis.coco_categories), donor_id))
            )
        if not chosen:
            continue
        for index, (_, _, control_key) in enumerate(chosen):
            keys.append(
                ControlKey(
                    hypothesis_id(node.id, index),
                    control_key.node_id,
                    control_key.kind,
                    control_key.expected,
                    control_key.categories,
                    round(control_key.inside_rate, 4),
                    round(control_key.outside_rate, 4),
                    control_key.source_node_id,
                )
            )
        nodes_out.append(
            NodeExplanation(
                node.id,
                f"Control hypotheses for {node.id}: " + ", ".join(kind for kind, _, _ in chosen),
                tuple(hypothesis for _, hypothesis, _ in chosen),
                "controls",
            )
        )

    if not nodes_out:
        raise ValueError("no node qualified for any control hypothesis")
    artifact = ExplanationArtifact(
        graph_sha256=sha256_file(graph_path),
        annotations_sha256=sha256_file(annotations_path),
        embedding_model_fingerprint=graph.embedding_model_fingerprint,
        generator={"provider": "controls", "config": config.as_dict(), "expected": dict(EXPECTED)},
        context_config={},
        explanations=tuple(nodes_out),
    )
    return artifact, keys


def save_controls(
    artifact: ExplanationArtifact,
    keys: Sequence[ControlKey],
    explanations_path: Path,
    key_path: Path,
) -> Tuple[Path, Path]:
    """Write the control explanations and an answer key tied to that exact file."""

    artifact.save(explanations_path)
    _atomic_json(
        key_path,
        {
            "schema_version": CONTROL_SCHEMA_VERSION,
            "explanations_sha256": sha256_file(explanations_path),
            "controls": [key.as_dict() for key in keys],
        },
    )
    return explanations_path, key_path


def load_control_key(path: Path) -> Tuple[str, List[ControlKey]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if data.get("schema_version") != CONTROL_SCHEMA_VERSION:
        raise ValueError("unsupported control key schema version")
    return str(data["explanations_sha256"]), [
        ControlKey.from_mapping(value) for value in data["controls"]
    ]
