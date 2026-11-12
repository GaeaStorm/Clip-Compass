"""Load inputs, render masked images, embed them, and score each hypothesis."""

from __future__ import annotations

import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Tuple

import numpy as np

from clip_compass.data.coco_instances import CocoInstanceSubset, sha256_file
from clip_compass.embeddings.base import ClipEncoder, ClipTextEncoder
from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.service import embed_images, embed_texts
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.explanations.context import node_centroid
from clip_compass.explanations.types import ExplanationArtifact, hypothesis_id
from clip_compass.mapper.diagnostics import ManifestImage, load_manifest_images
from clip_compass.mapper.graph import MapperGraphArtifact

from .masks import category_mask, clip_visible_box, control_masks, visible_pixels
from .verify import FILLS, SCHEMA_VERSION, PerturbationConfig, decide

Record = Dict[str, object]


class ClipImageTextEncoder(ClipEncoder, ClipTextEncoder, Protocol):
    """A CLIP backend that embeds both images and text, such as HuggingFaceClipEncoder."""


@dataclass(frozen=True)
class _Inputs:
    graph: MapperGraphArtifact
    embeddings: EmbeddingBatch
    annotations: CocoInstanceSubset
    explanations: ExplanationArtifact
    manifest: Mapping[str, ManifestImage]
    hashes: Mapping[str, str]


def verify_hypotheses(
    *,
    graph_path: Path,
    embeddings_path: Path,
    annotations_path: Path,
    explanations_path: Path,
    manifest_path: Path,
    data_root: Path,
    encoder: ClipImageTextEncoder,
    cache: EmbeddingCache,
    output_dir: Path,
    config: Optional[PerturbationConfig] = None,
    batch_size: int = 16,
) -> Record:
    """Mask, embed, and judge every hypothesis; write ``verification.json``."""

    config = config or PerturbationConfig()
    output_dir = output_dir.expanduser().resolve()
    inputs = _load_and_check(
        graph_path=graph_path,
        embeddings_path=embeddings_path,
        annotations_path=annotations_path,
        explanations_path=explanations_path,
        manifest_path=manifest_path,
        data_root=data_root,
        encoder=encoder,
    )
    hypotheses, pngs = _render(inputs, config, output_dir)
    vectors, prompts, min_identity = _embed(
        pngs, hypotheses, inputs, encoder, cache, config, batch_size
    )
    _score(hypotheses, vectors, prompts, inputs, config)
    results: Record = {
        "schema_version": SCHEMA_VERSION,
        "inputs": dict(inputs.hashes),
        "model_fingerprint": encoder.spec.fingerprint,
        "config": asdict(config),
        "min_identity_cosine": min_identity,
        "hypotheses": hypotheses,
    }
    _save_json(output_dir / "verification.json", results)
    return results


def _load_and_check(
    *,
    graph_path: Path,
    embeddings_path: Path,
    annotations_path: Path,
    explanations_path: Path,
    manifest_path: Path,
    data_root: Path,
    encoder: ClipImageTextEncoder,
) -> _Inputs:
    """Load every input and refuse files that were not produced together."""

    inputs = _Inputs(
        graph=MapperGraphArtifact.load(graph_path),
        embeddings=EmbeddingBatch.load(embeddings_path),
        annotations=CocoInstanceSubset.load(annotations_path),
        explanations=ExplanationArtifact.load(explanations_path),
        manifest=load_manifest_images(manifest_path, data_root),
        hashes={
            "graph_sha256": sha256_file(graph_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "annotations_sha256": sha256_file(annotations_path),
            "explanations_sha256": sha256_file(explanations_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
    )
    graph, explanations = inputs.graph, inputs.explanations
    if explanations.graph_sha256 != inputs.hashes["graph_sha256"]:
        raise ValueError("explanations were generated from a different Mapper graph file")
    if explanations.annotations_sha256 != inputs.hashes["annotations_sha256"]:
        raise ValueError("explanations were generated from a different annotation file")
    fingerprints = {
        graph.embedding_model_fingerprint,
        inputs.embeddings.model_spec.fingerprint,
        explanations.embedding_model_fingerprint,
        encoder.spec.fingerprint,
    }
    if len(fingerprints) != 1:
        raise ValueError("graph, embeddings, explanations, and encoder use different models")
    if not graph.item_ids == inputs.embeddings.item_ids == inputs.annotations.item_ids:
        raise ValueError("graph, embedding, and annotation item IDs must align in order")
    missing = [item_id for item_id in graph.item_ids if item_id not in inputs.manifest]
    if missing:
        raise ValueError(f"manifest is missing item {missing[0]}")
    return inputs


def _render(
    inputs: _Inputs, config: PerturbationConfig, output_dir: Path
) -> Tuple[List[Record], Dict[str, Path]]:
    """Classify each member image and write its object, control, and identity PNGs.

    Returns one record per hypothesis and a map from variant ID to PNG path.
    """

    annotations = inputs.annotations
    known = {category.name for category in annotations.categories}
    nodes = {node.id: node for node in inputs.graph.nodes}
    fill = FILLS[config.fill]
    style = f"{config.fill}-d{config.dilate_px}"
    pngs: Dict[str, Path] = {}
    sources: Dict[str, np.ndarray] = {}
    hypotheses: List[Record] = []

    def source(item_id: str) -> np.ndarray:
        """Decode a member image once and write its unmasked identity PNG."""

        if item_id not in sources:
            image = annotations.images[annotations.item_ids.index(item_id)]
            sources[item_id] = _load_rgb(inputs.manifest[item_id].image_path)
            if sources[item_id].shape[:2] != (image.height, image.width):
                raise ValueError(f"image {item_id} does not match its annotations")
            identity = f"{item_id}__identity"
            pngs[identity] = output_dir / "images" / item_id / f"{identity}.png"
            _write_png(pngs[identity], sources[item_id])
        return sources[item_id]

    for explanation in inputs.explanations.explanations:
        node = nodes[explanation.node_id]
        for index, hypothesis in enumerate(explanation.hypotheses):
            identifier = hypothesis_id(node.id, index)
            names = tuple(dict.fromkeys(hypothesis.coco_categories))
            unknown = [name for name in names if name not in known]
            slug = "+".join(sorted(name.replace(" ", "-") for name in names))
            images: List[Record] = []
            for item_id in node.item_ids if names and not unknown else ():
                image = annotations.images[annotations.item_ids.index(item_id)]
                box = clip_visible_box(image.width, image.height)
                crop_pixels = (box[2] - box[0]) * (box[3] - box[1])
                mask = category_mask(annotations, item_id, names, dilate_px=config.dilate_px)
                record: Record = {"item_id": item_id, "status": "absent"}
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
                rgb = source(item_id)
                stem = f"{item_id}__{identifier}__{slug}__{style}__"
                variants = [f"{stem}object"] + [
                    f"{stem}ctrl-s{config.seed}-{k:02d}" for k in range(config.controls)
                ]
                for variant, variant_mask in zip(variants, [mask, *controls]):
                    perturbed = rgb.copy()
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
    return hypotheses, pngs


def _embed(
    pngs: Mapping[str, Path],
    hypotheses: List[Record],
    inputs: _Inputs,
    encoder: ClipImageTextEncoder,
    cache: EmbeddingCache,
    config: PerturbationConfig,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray], Optional[float]]:
    """Embed every PNG and positive prompt, and check unmasked PNGs match the originals.

    The content-addressed cache makes reruns free.
    """

    if not pngs:
        return {}, {}, None
    images = embed_images(
        list(pngs.values()),
        encoder=encoder,
        cache=cache,
        item_ids=list(pngs),
        batch_size=batch_size,
    )
    vectors = dict(zip(images.item_ids, images.normalized.astype(np.float64)))
    texts = embed_texts(
        [hypothesis["positive_prompt"] for hypothesis in hypotheses],
        encoder=encoder,
        cache=cache,
        item_ids=[hypothesis["hypothesis_id"] for hypothesis in hypotheses],
    )
    prompts = dict(zip(texts.item_ids, texts.normalized.astype(np.float64)))
    originals = dict(
        zip(inputs.embeddings.item_ids, inputs.embeddings.normalized.astype(np.float64))
    )
    min_identity = min(
        float(vectors[variant] @ originals[variant.split("__")[0]])
        for variant in pngs
        if variant.endswith("__identity")
    )
    if min_identity < config.min_identity_cosine:
        raise ValueError(
            f"unmasked PNGs do not reproduce the stored embeddings "
            f"(min cosine {min_identity:.6f}); check images and model"
        )
    return vectors, prompts, min_identity


def _score(
    hypotheses: List[Record],
    vectors: Mapping[str, np.ndarray],
    prompts: Mapping[str, np.ndarray],
    inputs: _Inputs,
    config: PerturbationConfig,
) -> None:
    """Score each testable image against its controls, then judge each hypothesis."""

    nodes = {node.id: node for node in inputs.graph.nodes}
    for hypothesis in hypotheses:
        node = nodes[hypothesis["node_id"]]
        centroid = node_centroid(inputs.embeddings.normalized, node.member_indices)
        centroid = centroid.astype(np.float64)
        positive = prompts.get(hypothesis["hypothesis_id"])
        for image in hypothesis["images"]:
            if image["status"] != "testable":
                continue
            baseline = vectors[f"{image['item_id']}__identity"]
            points = np.stack([vectors[variant] for variant in image.pop("variants")])
            values = {
                "centroid": np.linalg.norm(points - centroid, axis=1)
                - np.linalg.norm(baseline - centroid),
                "prompt": float(baseline @ positive) - points @ positive,
            }
            image["object"] = {name: float(value[0]) for name, value in values.items()}
            image["controls"] = {
                name: value[1:].round(8).tolist() for name, value in values.items()
            }
            image["p"] = {
                name: (1 + int((value[1:] >= value[0]).sum())) / len(value)
                for name, value in values.items()
            }
        hypothesis["verdict"] = decide(
            hypothesis["coco_categories"],
            hypothesis["unknown_categories"],
            hypothesis["predicted_effect"],
            hypothesis["member_count"],
            hypothesis["images"],
            config,
        )


def _load_rgb(path: Path) -> np.ndarray:
    from PIL import Image

    with Image.open(path) as image:  # decode exactly as the CLIP encoder does
        return np.asarray(image.convert("RGB"), dtype=np.uint8).copy()


def _write_png(path: Path, rgb: np.ndarray) -> None:
    from PIL import Image

    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    Image.fromarray(rgb).save(partial, format="PNG")  # lossless and metadata-free
    os.replace(partial, path)


def _save_json(path: Path, data: Record) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(json.dumps(data, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(partial, path)
