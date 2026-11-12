"""Verify explanation hypotheses by masking their COCO objects and re-embedding with CLIP.

For each node member that contains the hypothesis categories, the object mask and
``controls`` area-matched random masks are filled and re-embedded. An image *exceeds*
on a statistic when the object mask beats its controls (empirical p <= alpha):

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

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from clip_compass.data.coco_instances import CocoInstanceSubset, sha256_file
from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.service import embed_images, embed_texts
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.explanations.context import node_centroid
from clip_compass.explanations.types import (
    ExplanationArtifact,
    NodeExplanation,
    hypothesis_id,
)
from clip_compass.mapper.diagnostics import load_manifest_images
from clip_compass.mapper.graph import MapperGraphArtifact

from .masks import category_mask, clip_visible_box, control_masks, visible_pixels

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


def _write_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    Image.fromarray(rgb).save(partial, format="PNG")  # lossless and metadata-free
    os.replace(partial, path)


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:  # decode exactly as the CLIP encoder does
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _verdict(
    categories, unknown, predicted_effect, member_count, images, config
) -> Dict[str, object]:
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


def verify_hypotheses(
    *,
    graph_path: Path,
    embeddings_path: Path,
    annotations_path: Path,
    explanations_path: Path,
    manifest_path: Path,
    data_root: Path,
    encoder,
    cache: EmbeddingCache,
    output_dir: Path,
    config: Optional[PerturbationConfig] = None,
    batch_size: int = 16,
) -> Dict[str, object]:
    """Mask, embed, and judge every hypothesis; write ``verification.json``."""

    config = config or PerturbationConfig()
    graph = MapperGraphArtifact.load(graph_path)
    embeddings = EmbeddingBatch.load(embeddings_path)
    annotations = CocoInstanceSubset.load(annotations_path)
    explanations = ExplanationArtifact.load(explanations_path)
    manifest = load_manifest_images(manifest_path, data_root)
    hashes = {
        "graph_sha256": sha256_file(graph_path),
        "embeddings_sha256": sha256_file(embeddings_path),
        "annotations_sha256": sha256_file(annotations_path),
        "explanations_sha256": sha256_file(explanations_path),
        "manifest_sha256": sha256_file(manifest_path),
    }
    if explanations.graph_sha256 != hashes["graph_sha256"]:
        raise ValueError("explanations were generated from a different Mapper graph file")
    if explanations.annotations_sha256 != hashes["annotations_sha256"]:
        raise ValueError("explanations were generated from a different annotation file")
    fingerprints = {
        graph.embedding_model_fingerprint,
        embeddings.model_spec.fingerprint,
        explanations.embedding_model_fingerprint,
        encoder.spec.fingerprint,
    }
    if len(fingerprints) != 1:
        raise ValueError("graph, embeddings, explanations, and encoder use different models")
    if not graph.item_ids == embeddings.item_ids == annotations.item_ids:
        raise ValueError("graph, embedding, and annotation item IDs must align in order")
    missing = [item_id for item_id in graph.item_ids if item_id not in manifest]
    if missing:
        raise ValueError(f"manifest is missing item {missing[0]}")

    output_dir = output_dir.expanduser().resolve()
    known = {category.name for category in annotations.categories}
    nodes = {node.id: node for node in graph.nodes}
    fill = FILLS[config.fill]
    style = f"{config.fill}-d{config.dilate_px}"
    pngs: Dict[str, Path] = {}  # variant ID -> PNG path, embedded in one call below
    rgb_cache: Dict[str, np.ndarray] = {}
    hypotheses: List[Dict[str, object]] = []

    # 1. Classify every member image and write its object, control, and identity PNGs.
    for explanation in explanations.explanations:
        node = nodes[explanation.node_id]
        for index, hypothesis in enumerate(explanation.hypotheses):
            identifier = hypothesis_id(node.id, index)
            names = tuple(dict.fromkeys(hypothesis.coco_categories))
            unknown = [name for name in names if name not in known]
            slug = "+".join(sorted(name.replace(" ", "-") for name in names))
            images = []
            for item_id in node.item_ids if names and not unknown else ():
                image = annotations.images[annotations.item_ids.index(item_id)]
                box = clip_visible_box(image.width, image.height)
                crop_pixels = (box[2] - box[0]) * (box[3] - box[1])
                mask = category_mask(annotations, item_id, names, dilate_px=config.dilate_px)
                record: Dict[str, object] = {"item_id": item_id, "status": "absent"}
                images.append(record)
                if mask is None:
                    continue
                record["visible_crop_fraction"] = visible_pixels(mask, box) / crop_pixels
                if record["visible_crop_fraction"] < config.min_visible_fraction:
                    record["status"] = "too_small_in_crop"
                    continue
                controls = control_masks(
                    mask,
                    box,
                    count=config.controls,
                    seed_key=f"{config.seed}|{item_id}|{slug}",
                    max_overlap=config.max_control_overlap,
                )
                if len(controls) < config.controls:
                    record["status"] = "no_control_room"
                    continue
                if item_id not in rgb_cache:
                    rgb_cache[item_id] = _load_rgb(manifest[item_id].image_path)
                    if rgb_cache[item_id].shape[:2] != (image.height, image.width):
                        raise ValueError(f"image {item_id} does not match its annotations")
                    identity = f"{item_id}__identity"
                    pngs[identity] = output_dir / "images" / item_id / f"{identity}.png"
                    _write_png(pngs[identity], rgb_cache[item_id])
                stem = f"{item_id}__{identifier}__{slug}__{style}__"
                variants = [f"{stem}object"] + [
                    f"{stem}ctrl-s{config.seed}-{k:02d}" for k in range(config.controls)
                ]
                for variant, variant_mask in zip(variants, [mask, *controls]):
                    perturbed = rgb_cache[item_id].copy()
                    perturbed[variant_mask] = fill
                    pngs[variant] = output_dir / "images" / item_id / f"{variant}.png"
                    _write_png(pngs[variant], perturbed)
                record.update(status="testable", variants=variants)
            hypotheses.append(
                {
                    "hypothesis_id": identifier,
                    "node_id": node.id,
                    "name": hypothesis.name,
                    "coco_categories": list(names),
                    "unknown_categories": unknown,
                    "predicted_effect": hypothesis.predicted_effect,
                    "positive_prompt": hypothesis.positive_prompt,
                    "member_count": len(node.item_ids),
                    "images": images,
                }
            )

    # 2. Embed every PNG and prompt; the content-addressed cache makes reruns free.
    vectors: Dict[str, np.ndarray] = {}
    prompts: Dict[str, np.ndarray] = {}
    min_identity: Optional[float] = None
    if pngs:
        batch = embed_images(
            list(pngs.values()),
            encoder=encoder,
            cache=cache,
            item_ids=list(pngs),
            batch_size=batch_size,
        )
        vectors = dict(zip(batch.item_ids, batch.normalized.astype(np.float64)))
        texts = embed_texts(
            [h["positive_prompt"] for h in hypotheses],
            encoder=encoder,
            cache=cache,
            item_ids=[h["hypothesis_id"] for h in hypotheses],
        )
        prompts = dict(zip(texts.item_ids, texts.normalized.astype(np.float64)))
        originals = dict(zip(embeddings.item_ids, embeddings.normalized.astype(np.float64)))
        min_identity = min(
            float(vectors[f"{item_id}__identity"] @ originals[item_id]) for item_id in rgb_cache
        )
        if min_identity < config.min_identity_cosine:
            raise ValueError(
                f"unmasked PNGs do not reproduce the stored embeddings "
                f"(min cosine {min_identity:.6f}); check images and model"
            )

    # 3. Score each testable image against its controls, then judge each hypothesis.
    for result in hypotheses:
        node = nodes[result["node_id"]]
        for image in result["images"]:
            if image["status"] != "testable":
                continue
            # Leave the image under test out of its own centroid; otherwise it pulls the
            # centroid towards itself, which weakens the test most in small nodes.
            position = graph.item_ids.index(image["item_id"])
            others = [index for index in node.member_indices if index != position]
            centroid = node_centroid(
                embeddings.normalized, others or node.member_indices
            ).astype(np.float64)
            baseline = vectors[f"{image['item_id']}__identity"]
            points = np.stack([vectors[variant] for variant in image.pop("variants")])
            positive = prompts[result["hypothesis_id"]]
            values = {
                "centroid": np.linalg.norm(points - centroid, axis=1)
                - np.linalg.norm(baseline - centroid),
                "prompt": float(baseline @ positive) - points @ positive,
            }
            image["object"] = {name: float(v[0]) for name, v in values.items()}
            image["controls"] = {name: v[1:].round(8).tolist() for name, v in values.items()}
            image["p"] = {
                name: (1 + int((v[1:] >= v[0]).sum())) / len(v) for name, v in values.items()
            }
        result["verdict"] = _verdict(
            result["coco_categories"],
            result["unknown_categories"],
            result["predicted_effect"],
            result["member_count"],
            result["images"],
            config,
        )

    results = {
        "schema_version": SCHEMA_VERSION,
        "inputs": hashes,
        "model_fingerprint": encoder.spec.fingerprint,
        "config": asdict(config),
        "min_identity_cosine": min_identity,
        "hypotheses": hypotheses,
    }
    path = output_dir / "verification.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(json.dumps(results, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)
    return results


def load_results(path: Path) -> Dict[str, object]:
    results = json.loads(path.read_text(encoding="utf-8"))
    if results.get("schema_version") != SCHEMA_VERSION:
        raise ValueError("unsupported perturbation results schema version")
    return results


def format_summary(results: Dict[str, object]) -> str:
    header = ("hypothesis", "categories", "coverage", "centroid", "prompt", "verdict", "reason")
    rows = [header]
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
