"""Configuration and typed results for perturbation runs, as stored in ``verification.json``."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass, fields
from pathlib import Path
from typing import Dict, Literal, Mapping, Optional, Tuple

SCHEMA_VERSION = 1
FILLS = {"mean": (123, 117, 104), "black": (0, 0, 0)}  # mean = round(255 * CLIP image_mean)

Statistic = Literal["centroid", "prompt"]
STATISTICS: Tuple[Statistic, ...] = ("centroid", "prompt")
ImageStatus = Literal["absent", "too_small_in_crop", "no_control_room", "testable"]
Label = Literal["supported", "detected_not_explanatory", "not_supported", "untestable"]


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


@dataclass(frozen=True)
class ImageScore:
    """Effect of masking the object versus each control, per statistic (larger = bigger effect)."""

    object: Mapping[Statistic, float]
    controls: Mapping[Statistic, Tuple[float, ...]]
    p: Mapping[Statistic, float]  # share of {object, *controls} at least as large as object


@dataclass(frozen=True)
class ImageResult:
    """One node member: whether it can test the hypothesis, and its score if it was tested."""

    item_id: str
    status: ImageStatus
    visible_crop_fraction: Optional[float] = None
    score: Optional[ImageScore] = None

    def __post_init__(self) -> None:
        if self.score is not None and self.status != "testable":
            raise ValueError("only testable images can have a score")

    def as_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {"item_id": self.item_id, "status": self.status}
        if self.visible_crop_fraction is not None:
            data["visible_crop_fraction"] = self.visible_crop_fraction
        if self.score is not None:
            data.update(asdict(self.score))  # stored flat: object, controls, p
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> ImageResult:
        score = None
        if "p" in data:
            controls = {name: tuple(values) for name, values in data["controls"].items()}
            score = ImageScore(object=data["object"], controls=controls, p=data["p"])
        return cls(data["item_id"], data["status"], data.get("visible_crop_fraction"), score)


@dataclass(frozen=True)
class Tally:
    exceeding: int
    testable: int
    passes: bool  # a strict majority of testable members exceed


@dataclass(frozen=True)
class Verdict:
    label: Label
    reason: Optional[str]
    member_count: int
    visible_members: int
    testable_members: int
    tallies: Mapping[Statistic, Tally]
    prediction_confirmed: Optional[bool]

    @property
    def verified(self) -> bool:
        return self.label == "supported"

    def as_dict(self) -> Dict[str, object]:
        return {**asdict(self), "verified": self.verified}

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> Verdict:
        values = {key: value for key, value in data.items() if key != "verified"}
        values["tallies"] = {name: Tally(**tally) for name, tally in data["tallies"].items()}
        return cls(**values)


@dataclass(frozen=True)
class HypothesisResult:
    hypothesis_id: str
    node_id: str
    name: str
    coco_categories: Tuple[str, ...]
    unknown_categories: Tuple[str, ...]
    predicted_effect: str
    positive_prompt: str
    member_count: int
    images: Tuple[ImageResult, ...]
    verdict: Verdict

    def as_dict(self) -> Dict[str, object]:
        data: Dict[str, object] = {field.name: getattr(self, field.name) for field in fields(self)}
        data["images"] = [image.as_dict() for image in self.images]
        data["verdict"] = self.verdict.as_dict()
        return data

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> HypothesisResult:
        return cls(
            **{
                **data,
                "coco_categories": tuple(data["coco_categories"]),
                "unknown_categories": tuple(data["unknown_categories"]),
                "images": tuple(ImageResult.from_dict(image) for image in data["images"]),
                "verdict": Verdict.from_dict(data["verdict"]),
            }
        )


@dataclass(frozen=True)
class VerificationResults:
    inputs: Mapping[str, str]  # SHA-256 of every input file
    model_fingerprint: str
    config: PerturbationConfig
    min_identity_cosine: Optional[float]  # None when no image was testable
    hypotheses: Tuple[HypothesisResult, ...]

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": SCHEMA_VERSION,
            "inputs": dict(self.inputs),
            "model_fingerprint": self.model_fingerprint,
            "config": asdict(self.config),
            "min_identity_cosine": self.min_identity_cosine,
            "hypotheses": [hypothesis.as_dict() for hypothesis in self.hypotheses],
        }

    @classmethod
    def from_dict(cls, data: Mapping[str, object]) -> VerificationResults:
        if data.get("schema_version") != SCHEMA_VERSION:
            raise ValueError("unsupported perturbation results schema version")
        return cls(
            inputs=dict(data["inputs"]),
            model_fingerprint=data["model_fingerprint"],
            config=PerturbationConfig(**data["config"]),
            min_identity_cosine=data["min_identity_cosine"],
            hypotheses=tuple(HypothesisResult.from_dict(h) for h in data["hypotheses"]),
        )

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        text = json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n"
        partial.write_text(text, encoding="utf-8")
        os.replace(partial, path)
        return path

    @classmethod
    def load(cls, path: Path) -> VerificationResults:
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
