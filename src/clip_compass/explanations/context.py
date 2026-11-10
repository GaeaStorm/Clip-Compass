"""Deterministic target/contrast evidence assembly for Mapper nodes."""

from __future__ import annotations

from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, List, Mapping, Optional, Set, Tuple

import numpy as np

from clip_compass.data.coco_instances import CocoInstanceSubset, sha256_file
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.mapper.diagnostics import ManifestImage, load_manifest_images
from clip_compass.mapper.graph import MapperGraphArtifact

from .types import EvidenceItem, ExplanationContextBundle, NodeExplanationContext


def _nearest_indices(
    candidates: Iterable[int],
    features: np.ndarray,
    centroid: np.ndarray,
    count: int,
) -> Tuple[int, ...]:
    unique = np.asarray(sorted(set(candidates)), dtype=np.int64)
    if not unique.size:
        return ()
    distances = np.linalg.norm(features[unique] - centroid, axis=1)
    order = np.lexsort((unique, distances))
    return tuple(int(unique[index]) for index in order[:count])


def _neighbors(graph: MapperGraphArtifact) -> Dict[str, Set[str]]:
    result: Dict[str, Set[str]] = {node.id: set() for node in graph.nodes}
    for edge in graph.edges:
        result[edge.source].add(edge.target)
        result[edge.target].add(edge.source)
    return result


def _category_counts(
    item_ids: Iterable[str], annotations: CocoInstanceSubset
) -> Dict[str, int]:
    result: Counter[str] = Counter()
    for item_id in item_ids:
        result.update(annotations.category_names_for(item_id))
    return dict(sorted(result.items(), key=lambda value: (-value[1], value[0])))


def build_explanation_contexts(
    graph: MapperGraphArtifact,
    embeddings: EmbeddingBatch,
    annotations: CocoInstanceSubset,
    *,
    graph_path: Path,
    annotations_path: Path,
    manifest_path: Path,
    data_root: Path,
    node_ids: Optional[Tuple[str, ...]] = None,
    representatives: int = 3,
    contrasts: int = 3,
    caption_limit: int = 2,
) -> ExplanationContextBundle:
    """Select centroid-nearest targets and hard contrast examples per node."""

    if representatives < 1 or contrasts < 1 or caption_limit < 1:
        raise ValueError("representatives, contrasts, and caption_limit must be positive")
    if embeddings.modality != "image":
        raise ValueError("explanation contexts require image embeddings")
    if graph.item_ids != embeddings.item_ids:
        raise ValueError("graph and embedding item IDs must align")
    if graph.item_ids != annotations.item_ids:
        raise ValueError("graph and COCO annotation item IDs must align")
    if graph.embedding_model_fingerprint != embeddings.model_spec.fingerprint:
        raise ValueError("graph and embedding model fingerprints must match")

    manifest = load_manifest_images(manifest_path, data_root)
    missing = [item_id for item_id in graph.item_ids if item_id not in manifest]
    if missing:
        raise ValueError(f"manifest is missing graph item ID {missing[0]}")
    nodes_by_id = {node.id: node for node in graph.nodes}
    selected_ids = node_ids or tuple(node.id for node in graph.nodes)
    unknown = [node_id for node_id in selected_ids if node_id not in nodes_by_id]
    if unknown:
        raise ValueError(f"unknown Mapper node ID: {unknown[0]}")

    features = embeddings.normalized
    neighbor_ids = _neighbors(graph)
    contexts: List[NodeExplanationContext] = []
    for node_id in selected_ids:
        node = nodes_by_id[node_id]
        member_set = set(node.member_indices)
        member_array = np.asarray(node.member_indices, dtype=np.int64)
        centroid = features[member_array].mean(axis=0)
        target_indices = _nearest_indices(
            node.member_indices, features, centroid, representatives
        )
        neighbor_candidates = {
            index
            for neighbor_id in neighbor_ids[node_id]
            for index in nodes_by_id[neighbor_id].member_indices
            if index not in member_set
        }
        contrast_indices = list(
            _nearest_indices(neighbor_candidates, features, centroid, contrasts)
        )
        if len(contrast_indices) < contrasts:
            global_candidates = (
                index
                for index in range(len(graph.item_ids))
                if index not in member_set and index not in set(contrast_indices)
            )
            contrast_indices.extend(
                _nearest_indices(
                    global_candidates,
                    features,
                    centroid,
                    contrasts - len(contrast_indices),
                )
            )
        if not contrast_indices:
            raise ValueError(f"node {node_id} has no available contrast items")

        evidence = tuple(
            _evidence_item(
                graph.item_ids[index],
                "target",
                manifest,
                annotations,
                caption_limit,
            )
            for index in target_indices
        ) + tuple(
            _evidence_item(
                graph.item_ids[index],
                "contrast",
                manifest,
                annotations,
                caption_limit,
            )
            for index in contrast_indices
        )
        contrast_item_ids = tuple(graph.item_ids[index] for index in contrast_indices)
        contexts.append(
            NodeExplanationContext(
                node_id=node.id,
                source_id=node.source_id,
                component_id=node.component_id,
                member_count=len(node.member_indices),
                lens_min=node.lens_min,
                lens_mean=node.lens_mean,
                lens_max=node.lens_max,
                neighbor_node_ids=tuple(sorted(neighbor_ids[node_id])),
                target_category_counts=_category_counts(node.item_ids, annotations),
                contrast_category_counts=_category_counts(
                    contrast_item_ids, annotations
                ),
                evidence=evidence,
            )
        )

    config = {
        "representatives": representatives,
        "contrasts": contrasts,
        "caption_limit": caption_limit,
        "feature_space": "normalized",
        "contrast_strategy": "adjacent_then_global_nearest_to_target_centroid",
    }
    return ExplanationContextBundle(
        graph_sha256=sha256_file(graph_path),
        annotations_sha256=sha256_file(annotations_path),
        embedding_model_fingerprint=embeddings.model_spec.fingerprint,
        config=config,
        contexts=tuple(contexts),
    )


def _evidence_item(
    item_id: str,
    role: str,
    manifest: Mapping[str, ManifestImage],
    annotations: CocoInstanceSubset,
    caption_limit: int,
) -> EvidenceItem:
    record = manifest[item_id]
    return EvidenceItem(
        item_id=item_id,
        image_path=str(record.image_path),
        role=role,
        captions=record.captions[:caption_limit],
        category_names=annotations.category_names_for(item_id),
    )
