"""Serializable visual-perturbation experiment artifacts."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Mapping, Tuple


PERTURBATION_SCHEMA_VERSION = 1


@dataclass(frozen=True)
class PerturbationPair:
    """One targeted mask and its area/shape-matched random control."""

    id: str
    node_id: str
    hypothesis_index: int
    hypothesis_name: str
    item_id: str
    source_image_path: str
    source_sha256: str
    category_names: Tuple[str, ...]
    annotation_ids: Tuple[int, ...]
    target_mask_path: str
    control_mask_path: str
    target_image_path: str
    control_image_path: str
    mask_pixel_count: int
    image_pixel_count: int
    control_overlap_pixel_count: int
    control_seed: int
    target_image_sha256: str
    control_image_sha256: str

    def __post_init__(self) -> None:
        if not self.id or not self.node_id or not self.item_id:
            raise ValueError("perturbation identifiers must not be empty")
        if self.hypothesis_index < 0 or not self.hypothesis_name.strip():
            raise ValueError("perturbation hypothesis metadata is invalid")
        if not self.category_names or not self.annotation_ids:
            raise ValueError("a targeted perturbation requires COCO annotations")
        if self.mask_pixel_count < 1 or self.image_pixel_count < self.mask_pixel_count:
            raise ValueError("perturbation mask area is invalid")
        if not 0 <= self.control_overlap_pixel_count <= self.mask_pixel_count:
            raise ValueError("control overlap must be within the mask area")
        if self.control_seed < 0:
            raise ValueError("control seed must be non-negative")

    @property
    def mask_fraction(self) -> float:
        return self.mask_pixel_count / self.image_pixel_count

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "node_id": self.node_id,
            "hypothesis_index": self.hypothesis_index,
            "hypothesis_name": self.hypothesis_name,
            "item_id": self.item_id,
            "source_image_path": self.source_image_path,
            "source_sha256": self.source_sha256,
            "category_names": list(self.category_names),
            "annotation_ids": list(self.annotation_ids),
            "target_mask_path": self.target_mask_path,
            "control_mask_path": self.control_mask_path,
            "target_image_path": self.target_image_path,
            "control_image_path": self.control_image_path,
            "mask_pixel_count": self.mask_pixel_count,
            "image_pixel_count": self.image_pixel_count,
            "mask_fraction": self.mask_fraction,
            "control_overlap_pixel_count": self.control_overlap_pixel_count,
            "control_overlap_fraction": (
                self.control_overlap_pixel_count / self.mask_pixel_count
            ),
            "control_seed": self.control_seed,
            "target_image_sha256": self.target_image_sha256,
            "control_image_sha256": self.control_image_sha256,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "PerturbationPair":
        return cls(
            id=str(data["id"]),
            node_id=str(data["node_id"]),
            hypothesis_index=int(data["hypothesis_index"]),
            hypothesis_name=str(data["hypothesis_name"]),
            item_id=str(data["item_id"]),
            source_image_path=str(data["source_image_path"]),
            source_sha256=str(data["source_sha256"]),
            category_names=tuple(str(value) for value in data["category_names"]),
            annotation_ids=tuple(int(value) for value in data["annotation_ids"]),
            target_mask_path=str(data["target_mask_path"]),
            control_mask_path=str(data["control_mask_path"]),
            target_image_path=str(data["target_image_path"]),
            control_image_path=str(data["control_image_path"]),
            mask_pixel_count=int(data["mask_pixel_count"]),
            image_pixel_count=int(data["image_pixel_count"]),
            control_overlap_pixel_count=int(data["control_overlap_pixel_count"]),
            control_seed=int(data["control_seed"]),
            target_image_sha256=str(data["target_image_sha256"]),
            control_image_sha256=str(data["control_image_sha256"]),
        )


@dataclass(frozen=True)
class SkippedPerturbation:
    node_id: str
    hypothesis_index: int
    item_id: str
    reason: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "node_id": self.node_id,
            "hypothesis_index": self.hypothesis_index,
            "item_id": self.item_id,
            "reason": self.reason,
        }

    @classmethod
    def from_mapping(cls, data: Mapping[str, object]) -> "SkippedPerturbation":
        return cls(
            node_id=str(data["node_id"]),
            hypothesis_index=int(data["hypothesis_index"]),
            item_id=str(data["item_id"]),
            reason=str(data["reason"]),
        )


@dataclass(frozen=True)
class PerturbationArtifact:
    graph_sha256: str
    annotations_sha256: str
    explanations_sha256: str
    embedding_model_fingerprint: str
    engine: Mapping[str, object]
    selection: Mapping[str, object]
    pairs: Tuple[PerturbationPair, ...]
    skipped: Tuple[SkippedPerturbation, ...] = ()

    def __post_init__(self) -> None:
        pair_ids = tuple(pair.id for pair in self.pairs)
        if not pair_ids:
            raise ValueError("a perturbation artifact must contain at least one pair")
        if len(set(pair_ids)) != len(pair_ids):
            raise ValueError("perturbation pair IDs must be unique")

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": PERTURBATION_SCHEMA_VERSION,
            "graph_sha256": self.graph_sha256,
            "annotations_sha256": self.annotations_sha256,
            "explanations_sha256": self.explanations_sha256,
            "embedding_model_fingerprint": self.embedding_model_fingerprint,
            "engine": dict(self.engine),
            "selection": dict(self.selection),
            "summary": {
                "pair_count": len(self.pairs),
                "generated_image_count": 2 * len(self.pairs),
                "skipped_item_count": len(self.skipped),
            },
            "pairs": [pair.as_dict() for pair in self.pairs],
            "skipped": [value.as_dict() for value in self.skipped],
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        partial.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "PerturbationArtifact":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != PERTURBATION_SCHEMA_VERSION:
            raise ValueError("unsupported perturbation-artifact schema version")
        return cls(
            graph_sha256=str(data["graph_sha256"]),
            annotations_sha256=str(data["annotations_sha256"]),
            explanations_sha256=str(data["explanations_sha256"]),
            embedding_model_fingerprint=str(data["embedding_model_fingerprint"]),
            engine=data["engine"],
            selection=data["selection"],
            pairs=tuple(PerturbationPair.from_mapping(value) for value in data["pairs"]),
            skipped=tuple(
                SkippedPerturbation.from_mapping(value) for value in data["skipped"]
            ),
        )
