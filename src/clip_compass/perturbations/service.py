"""Build image-grounded perturbation pairs from explanation hypotheses."""

from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from clip_compass.data.coco_instances import (
    CocoImage,
    CocoInstance,
    CocoInstanceSubset,
    sha256_file,
)
from clip_compass.data.coco_masks import instance_mask
from clip_compass.explanations.types import ExplanationArtifact, ExplanationHypothesis
from clip_compass.mapper.diagnostics import ManifestImage, load_manifest_images
from clip_compass.mapper.graph import MapperGraphArtifact

from .base import VisualPerturbationEngine
from .types import PerturbationArtifact, PerturbationPair, SkippedPerturbation


def _seed_for(global_seed: int, *parts: object) -> int:
    payload = ":".join((str(global_seed), *(str(part) for part in parts)))
    return int.from_bytes(hashlib.sha256(payload.encode("utf-8")).digest()[:8], "big")


def _target_mask(
    annotations: CocoInstanceSubset,
    image: CocoImage,
    hypothesis: ExplanationHypothesis,
) -> Tuple[np.ndarray, Tuple[int, ...], Tuple[str, ...]]:
    categories = annotations.categories_by_id
    requested = set(hypothesis.coco_categories)
    selected: List[CocoInstance] = [
        annotation
        for annotation in annotations.annotations_for(image.id)
        if categories[annotation.category_id].name in requested
    ]
    mask = np.zeros((image.height, image.width), dtype=np.bool_)
    for annotation in selected:
        mask |= instance_mask(annotation, image)
    names = tuple(
        sorted({categories[annotation.category_id].name for annotation in selected})
    )
    return mask, tuple(annotation.id for annotation in selected), names


def _selected_explanations(
    explanations: ExplanationArtifact,
    node_ids: Optional[Sequence[str]],
) -> Iterable[object]:
    selected = set(node_ids) if node_ids else None
    known = {explanation.node_id for explanation in explanations.explanations}
    if selected:
        unknown = sorted(selected - known)
        if unknown:
            raise ValueError(f"unknown explanation node ID: {unknown[0]}")
    return (
        explanation
        for explanation in explanations.explanations
        if selected is None or explanation.node_id in selected
    )


def _atomic_manifest(
    artifact: PerturbationArtifact,
    path: Path,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    with partial.open("w", encoding="utf-8") as output:
        for pair in artifact.pairs:
            common: Dict[str, object] = {
                "source_image_id": pair.item_id,
                "node_id": pair.node_id,
                "hypothesis_index": pair.hypothesis_index,
                "hypothesis_name": pair.hypothesis_name,
                "category_names": list(pair.category_names),
                "captions": [
                    f"{pair.hypothesis_name}: {', '.join(pair.category_names)}"
                ],
            }
            for variant, image_path in (
                ("target", pair.target_image_path),
                ("control", pair.control_image_path),
            ):
                row = dict(common)
                row.update(
                    {
                        "image_id": f"{pair.id}-{variant}",
                        "image_path": image_path,
                        "variant": variant,
                    }
                )
                output.write(json.dumps(row, sort_keys=True) + "\n")
    os.replace(partial, path)
    return path


def _atomic_png(image: object, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    image.save(partial, format="PNG", optimize=True)
    os.replace(partial, path)
    return path


def generate_perturbations(
    graph: MapperGraphArtifact,
    explanations: ExplanationArtifact,
    annotations: CocoInstanceSubset,
    engine: VisualPerturbationEngine,
    *,
    graph_path: Path,
    explanations_path: Path,
    annotations_path: Path,
    manifest_path: Path,
    data_root: Path,
    output_dir: Path,
    scope: str = "all-members",
    seed: int = 42,
    node_ids: Optional[Sequence[str]] = None,
    hypothesis_indices: Optional[Sequence[int]] = None,
    max_items_per_hypothesis: Optional[int] = None,
) -> Tuple[PerturbationArtifact, Mapping[str, Path]]:
    """Generate category masks and matched random controls for node hypotheses."""

    if scope not in {"all-members", "supporting"}:
        raise ValueError("scope must be 'all-members' or 'supporting'")
    if seed < 0:
        raise ValueError("seed must be non-negative")
    if max_items_per_hypothesis is not None and max_items_per_hypothesis < 1:
        raise ValueError("max_items_per_hypothesis must be positive")
    graph_sha256 = sha256_file(graph_path)
    annotations_sha256 = sha256_file(annotations_path)
    if explanations.graph_sha256 != graph_sha256:
        raise ValueError("explanations were prepared from a different Mapper graph")
    if explanations.annotations_sha256 != annotations_sha256:
        raise ValueError("explanations were prepared from different COCO annotations")
    if explanations.embedding_model_fingerprint != graph.embedding_model_fingerprint:
        raise ValueError("explanation and graph embedding fingerprints do not match")
    if graph.item_ids != annotations.item_ids:
        raise ValueError("graph and COCO annotation item IDs must align")

    manifest = load_manifest_images(manifest_path, data_root)
    missing = [item_id for item_id in graph.item_ids if item_id not in manifest]
    if missing:
        raise ValueError(f"manifest is missing graph item ID {missing[0]}")
    nodes = {node.id: node for node in graph.nodes}
    coco_images = {image.id: image for image in annotations.images}
    index_filter = set(hypothesis_indices) if hypothesis_indices is not None else None
    if index_filter and min(index_filter) < 0:
        raise ValueError("hypothesis indices must be non-negative")

    run_root = output_dir.expanduser().resolve()
    image_dir = run_root / "images"
    mask_dir = run_root / "masks"
    image_dir.mkdir(parents=True, exist_ok=True)
    mask_dir.mkdir(parents=True, exist_ok=True)
    resolved_data_root = data_root.expanduser().resolve()
    pairs: List[PerturbationPair] = []
    skipped: List[SkippedPerturbation] = []

    try:
        from PIL import Image
    except ImportError as error:
        raise RuntimeError(
            "Visual perturbations require Pillow: pip install -e '.[coco]'"
        ) from error

    for explanation in _selected_explanations(explanations, node_ids):
        node = nodes.get(explanation.node_id)
        if node is None:
            raise ValueError(f"explanation refers to unknown node {explanation.node_id}")
        for hypothesis_index, hypothesis in enumerate(explanation.hypotheses):
            if index_filter is not None and hypothesis_index not in index_filter:
                continue
            candidates = (
                tuple(hypothesis.supporting_item_ids)
                if scope == "supporting"
                else node.item_ids
            )
            node_members = set(node.item_ids)
            invalid = [item_id for item_id in candidates if item_id not in node_members]
            if invalid:
                raise ValueError(
                    f"hypothesis support item {invalid[0]} is not in node {node.id}"
                )
            matched_count = 0
            for item_id in candidates:
                if (
                    max_items_per_hypothesis is not None
                    and matched_count >= max_items_per_hypothesis
                ):
                    break
                coco_image = coco_images[item_id]
                target_mask, annotation_ids, category_names = _target_mask(
                    annotations, coco_image, hypothesis
                )
                if not target_mask.any():
                    skipped.append(
                        SkippedPerturbation(
                            node_id=node.id,
                            hypothesis_index=hypothesis_index,
                            item_id=item_id,
                            reason="no_requested_coco_category",
                        )
                    )
                    continue
                record: ManifestImage = manifest[item_id]
                if not record.image_path.is_file():
                    raise FileNotFoundError(record.image_path)
                with Image.open(record.image_path) as opened:
                    source = opened.convert("RGB")
                if source.size != (coco_image.width, coco_image.height):
                    raise ValueError(
                        f"image {item_id} size {source.size} does not match COCO metadata "
                        f"{(coco_image.width, coco_image.height)}"
                    )
                pair_id = f"{node.id}-h{hypothesis_index:02d}-{item_id}"
                control_seed = _seed_for(seed, node.id, hypothesis_index, item_id)
                result = engine.perturb(
                    source, target_mask, control_seed=control_seed
                )
                target_image_path = image_dir / f"{pair_id}-target.png"
                control_image_path = image_dir / f"{pair_id}-control.png"
                target_mask_path = mask_dir / f"{pair_id}-target.png"
                control_mask_path = mask_dir / f"{pair_id}-control.png"
                _atomic_png(result.target_image, target_image_path)
                _atomic_png(result.control_image, control_image_path)
                _atomic_png(
                    Image.fromarray(target_mask.astype(np.uint8) * 255),
                    target_mask_path,
                )
                _atomic_png(
                    Image.fromarray(result.control_mask.astype(np.uint8) * 255),
                    control_mask_path,
                )
                source_relative = record.image_path.relative_to(resolved_data_root).as_posix()
                pairs.append(
                    PerturbationPair(
                        id=pair_id,
                        node_id=node.id,
                        hypothesis_index=hypothesis_index,
                        hypothesis_name=hypothesis.name,
                        item_id=item_id,
                        source_image_path=source_relative,
                        source_sha256=sha256_file(record.image_path),
                        category_names=category_names,
                        annotation_ids=annotation_ids,
                        target_mask_path=target_mask_path.relative_to(run_root).as_posix(),
                        control_mask_path=control_mask_path.relative_to(run_root).as_posix(),
                        target_image_path=target_image_path.relative_to(run_root).as_posix(),
                        control_image_path=control_image_path.relative_to(run_root).as_posix(),
                        mask_pixel_count=int(target_mask.sum()),
                        image_pixel_count=int(target_mask.size),
                        control_overlap_pixel_count=(
                            result.control_overlap_pixel_count
                        ),
                        control_seed=control_seed,
                        target_image_sha256=sha256_file(target_image_path),
                        control_image_sha256=sha256_file(control_image_path),
                    )
                )
                matched_count += 1

    artifact = PerturbationArtifact(
        graph_sha256=graph_sha256,
        annotations_sha256=annotations_sha256,
        explanations_sha256=sha256_file(explanations_path),
        embedding_model_fingerprint=explanations.embedding_model_fingerprint,
        engine=engine.metadata,
        selection={
            "scope": scope,
            "seed": seed,
            "node_ids": list(node_ids) if node_ids else "all",
            "hypothesis_indices": (
                list(hypothesis_indices) if hypothesis_indices is not None else "all"
            ),
            "max_items_per_hypothesis": max_items_per_hypothesis,
        },
        pairs=tuple(pairs),
        skipped=tuple(skipped),
    )
    artifact_path = artifact.save(run_root / "perturbations.json")
    embedding_manifest_path = _atomic_manifest(
        artifact, run_root / "embedding-manifest.jsonl"
    )
    return artifact, {
        "artifact": artifact_path,
        "embedding_manifest": embedding_manifest_path,
        "run_root": run_root,
    }
