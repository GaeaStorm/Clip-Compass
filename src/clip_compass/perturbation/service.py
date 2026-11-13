"""Load inputs, render masked images, embed them, and score each hypothesis."""

from __future__ import annotations

import os
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Protocol, Sequence, Tuple

import numpy as np

from clip_compass.data.coco_instances import CocoImage, CocoInstanceSubset, sha256_file
from clip_compass.embeddings.base import ClipEncoder, ClipTextEncoder
from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.service import embed_images, embed_texts
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.explanations.context import node_centroid
from clip_compass.explanations.types import (
    ExplanationArtifact,
    ExplanationHypothesis,
    hypothesis_id,
)
from clip_compass.mapper.diagnostics import ManifestImage, load_manifest_images
from clip_compass.mapper.graph import MapperGraphArtifact, MapperNode

from .masks import category_mask, clip_visible_box, control_masks, visible_pixels
from .types import (
    FILLS,
    STATISTICS,
    HypothesisResult,
    ImageResult,
    ImageScore,
    PerturbationConfig,
    VerificationResults,
)
from .verify import decide


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
    nodes: Mapping[str, MapperNode]
    images: Mapping[str, CocoImage]


@dataclass(frozen=True)
class _Pending:
    """A hypothesis whose member images are classified and rendered, but not yet scored."""

    hypothesis_id: str
    node: MapperNode
    hypothesis: ExplanationHypothesis
    categories: Tuple[str, ...]
    unknown: Tuple[str, ...]
    images: Tuple[ImageResult, ...]
    variants: Mapping[str, Tuple[str, ...]]  # testable item ID -> (object, *control) IDs


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
) -> VerificationResults:
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
    writer = _ImageWriter(output_dir / "images", inputs, FILLS[config.fill])
    pending = _render(inputs, config, writer)
    prompts = {p.hypothesis_id: p.hypothesis.positive_prompt for p in pending if p.variants}
    vectors, prompt_vectors = _embed(writer.pngs, prompts, encoder, cache, batch_size)
    min_identity = _check_identity(vectors, writer.identities, inputs, config)
    results = VerificationResults(
        inputs=dict(inputs.hashes),
        model_fingerprint=encoder.spec.fingerprint,
        config=config,
        min_identity_cosine=min_identity,
        hypotheses=tuple(
            _score(p, vectors, prompt_vectors, writer.identities, inputs, config)
            for p in pending
        ),
    )
    results.save(output_dir / "verification.json")
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

    graph = MapperGraphArtifact.load(graph_path)
    annotations = CocoInstanceSubset.load(annotations_path)
    inputs = _Inputs(
        graph=graph,
        embeddings=EmbeddingBatch.load(embeddings_path),
        annotations=annotations,
        explanations=ExplanationArtifact.load(explanations_path),
        manifest=load_manifest_images(manifest_path, data_root),
        hashes={
            "graph_sha256": sha256_file(graph_path),
            "embeddings_sha256": sha256_file(embeddings_path),
            "annotations_sha256": sha256_file(annotations_path),
            "explanations_sha256": sha256_file(explanations_path),
            "manifest_sha256": sha256_file(manifest_path),
        },
        nodes={node.id: node for node in graph.nodes},
        images=dict(zip(annotations.item_ids, annotations.images)),
    )
    explanations = inputs.explanations
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
    if not graph.item_ids == inputs.embeddings.item_ids == annotations.item_ids:
        raise ValueError("graph, embedding, and annotation item IDs must align in order")
    missing = [item_id for item_id in graph.item_ids if item_id not in inputs.manifest]
    if missing:
        raise ValueError(f"manifest is missing item {missing[0]}")
    return inputs


class _ImageWriter:
    """Writes masked PNGs, plus one unmasked identity PNG per source image it touches."""

    def __init__(self, root: Path, inputs: _Inputs, fill: Tuple[int, int, int]) -> None:
        self._root, self._inputs, self._fill = root, inputs, fill
        self._sources: Dict[str, np.ndarray] = {}
        self.pngs: Dict[str, Path] = {}  # variant ID -> PNG, in write order
        self.identities: Dict[str, str] = {}  # item ID -> variant ID of its unmasked PNG

    def write(self, item_id: str, variants: Sequence[str], masks: Sequence[np.ndarray]) -> None:
        rgb = self._source(item_id)
        for variant, mask in zip(variants, masks):
            perturbed = rgb.copy()
            perturbed[mask] = self._fill
            self._save(item_id, variant, perturbed)

    def _source(self, item_id: str) -> np.ndarray:
        if item_id not in self._sources:
            image = self._inputs.images[item_id]
            rgb = _load_rgb(self._inputs.manifest[item_id].image_path)
            if rgb.shape[:2] != (image.height, image.width):
                raise ValueError(f"image {item_id} does not match its annotations")
            self._sources[item_id] = rgb
            self.identities[item_id] = f"{item_id}__identity"
            self._save(item_id, self.identities[item_id], rgb)
        return self._sources[item_id]

    def _save(self, item_id: str, variant: str, rgb: np.ndarray) -> None:
        self.pngs[variant] = self._root / item_id / f"{variant}.png"
        _write_png(self.pngs[variant], rgb)


def _render(inputs: _Inputs, config: PerturbationConfig, writer: _ImageWriter) -> List[_Pending]:
    """Classify each hypothesis's member images and write PNGs for the testable ones."""

    known = {category.name for category in inputs.annotations.categories}
    style = f"{config.fill}-d{config.dilate_px}"
    pending: List[_Pending] = []
    for explanation in inputs.explanations.explanations:
        node = inputs.nodes[explanation.node_id]
        for index, hypothesis in enumerate(explanation.hypotheses):
            identifier = hypothesis_id(node.id, index)
            categories = tuple(dict.fromkeys(hypothesis.coco_categories))
            unknown = tuple(name for name in categories if name not in known)
            slug = "+".join(sorted(name.replace(" ", "-") for name in categories))
            images: List[ImageResult] = []
            variants: Dict[str, Tuple[str, ...]] = {}
            for item_id in node.item_ids if categories and not unknown else ():
                image, masks = _classify(inputs, item_id, categories, slug, config)
                images.append(image)
                if masks:
                    stem = f"{item_id}__{identifier}__{slug}__{style}__"
                    controls = range(config.controls)
                    variants[item_id] = (
                        f"{stem}object",
                        *(f"{stem}ctrl-s{config.seed}-{k:02d}" for k in controls),
                    )
                    writer.write(item_id, variants[item_id], masks)
            pending.append(
                _Pending(identifier, node, hypothesis, categories, unknown, tuple(images), variants)
            )
    return pending


def _classify(
    inputs: _Inputs,
    item_id: str,
    categories: Sequence[str],
    slug: str,
    config: PerturbationConfig,
) -> Tuple[ImageResult, List[np.ndarray]]:
    """Decide whether one member image can test the categories. Writes nothing.

    Returns the image's status and, when it is testable, ``[object_mask, *control_masks]``.
    """

    image = inputs.images[item_id]
    mask = category_mask(inputs.annotations, item_id, categories, dilate_px=config.dilate_px)
    if mask is None:
        return ImageResult(item_id, "absent"), []
    box = clip_visible_box(image.width, image.height)
    fraction = visible_pixels(mask, box) / ((box[2] - box[0]) * (box[3] - box[1]))
    if fraction < config.min_visible_fraction:
        return ImageResult(item_id, "too_small_in_crop", fraction), []
    controls = control_masks(
        mask,
        box,
        count=config.controls,
        seed_key=f"{config.seed}|{item_id}|{slug}",
        max_overlap=config.max_control_overlap,
    )
    if len(controls) < config.controls:
        return ImageResult(item_id, "no_control_room", fraction), []
    return ImageResult(item_id, "testable", fraction), [mask, *controls]


def _embed(
    pngs: Mapping[str, Path],
    prompts: Mapping[str, str],
    encoder: ClipImageTextEncoder,
    cache: EmbeddingCache,
    batch_size: int,
) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """Embed every PNG and positive prompt; the content-addressed cache makes reruns free."""

    if not pngs:
        return {}, {}
    images = embed_images(
        list(pngs.values()),
        encoder=encoder,
        cache=cache,
        item_ids=list(pngs),
        batch_size=batch_size,
    )
    texts = embed_texts(
        list(prompts.values()), encoder=encoder, cache=cache, item_ids=list(prompts)
    )
    return (
        dict(zip(images.item_ids, images.normalized.astype(np.float64))),
        dict(zip(texts.item_ids, texts.normalized.astype(np.float64))),
    )


def _check_identity(
    vectors: Mapping[str, np.ndarray],
    identities: Mapping[str, str],
    inputs: _Inputs,
    config: PerturbationConfig,
) -> Optional[float]:
    """Check unmasked PNGs re-embed to the stored embeddings; return the lowest cosine."""

    if not identities:
        return None
    rows = {item_id: row for row, item_id in enumerate(inputs.embeddings.item_ids)}
    originals = inputs.embeddings.normalized.astype(np.float64)
    min_identity = min(
        float(vectors[variant] @ originals[rows[item_id]])
        for item_id, variant in identities.items()
    )
    if min_identity < config.min_identity_cosine:
        raise ValueError(
            f"unmasked PNGs do not reproduce the stored embeddings "
            f"(min cosine {min_identity:.6f}); check images and model"
        )
    return min_identity


def _score(
    pending: _Pending,
    vectors: Mapping[str, np.ndarray],
    prompts: Mapping[str, np.ndarray],
    identities: Mapping[str, str],
    inputs: _Inputs,
    config: PerturbationConfig,
) -> HypothesisResult:
    """Score each testable image against its controls, then judge the hypothesis."""

    images = pending.images
    if pending.variants:
        centroid = node_centroid(inputs.embeddings.normalized, pending.node.member_indices)
        centroid = centroid.astype(np.float64)
        positive = prompts[pending.hypothesis_id]
        images = tuple(
            replace(
                image,
                score=_score_image(
                    vectors[identities[image.item_id]],
                    np.stack([vectors[variant] for variant in pending.variants[image.item_id]]),
                    centroid,
                    positive,
                ),
            )
            if image.item_id in pending.variants
            else image
            for image in images
        )
    hypothesis, member_count = pending.hypothesis, len(pending.node.item_ids)
    return HypothesisResult(
        hypothesis_id=pending.hypothesis_id,
        node_id=pending.node.id,
        name=hypothesis.name,
        coco_categories=pending.categories,
        unknown_categories=pending.unknown,
        predicted_effect=hypothesis.predicted_effect,
        positive_prompt=hypothesis.positive_prompt,
        member_count=member_count,
        images=images,
        verdict=decide(
            images,
            categories=pending.categories,
            unknown=pending.unknown,
            predicted_effect=hypothesis.predicted_effect,
            member_count=member_count,
            config=config,
        ),
    )


def _score_image(
    baseline: np.ndarray, points: np.ndarray, centroid: np.ndarray, positive: np.ndarray
) -> ImageScore:
    """``points`` stacks the object variant then its controls; row 0 is the object."""

    values = {
        "centroid": np.linalg.norm(points - centroid, axis=1) - np.linalg.norm(baseline - centroid),
        "prompt": float(baseline @ positive) - points @ positive,
    }
    return ImageScore(
        object={name: float(values[name][0]) for name in STATISTICS},
        controls={name: tuple(values[name][1:].round(8).tolist()) for name in STATISTICS},
        p={
            name: (1 + int((values[name][1:] >= values[name][0]).sum())) / len(values[name])
            for name in STATISTICS
        },
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
