"""Classical Mapper graph construction adapted from the researcher workflow."""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Mapping, Optional, Set, Tuple

import numpy as np

from clip_compass.embeddings.types import EmbeddingBatch

from .calibration import EpsCalibration, calibrate_dbscan_eps
from .graph import MapperEdge, MapperGraphArtifact, MapperNode
from .lenses import LensResult


@dataclass(frozen=True)
class MapperConfig:
    n_cubes: int = 10
    perc_overlap: float = 0.4
    min_samples: int = 3
    eps: Optional[float] = None
    metric: str = "euclidean"
    feature_space: str = "normalized"
    eps_sample_size: int = 5000
    eps_fallback_quantile: float = 0.9
    random_seed: int = 42
    min_intersection: int = 1
    remove_duplicate_nodes: bool = True
    n_jobs: int = 1
    verbose: int = 0

    def __post_init__(self) -> None:
        if self.n_cubes < 2:
            raise ValueError("n_cubes must be at least 2")
        if not 0.0 < self.perc_overlap < 1.0:
            raise ValueError("perc_overlap must be in (0, 1)")
        if self.min_samples < 2:
            raise ValueError("min_samples must be at least 2")
        if self.eps is not None and self.eps <= 0.0:
            raise ValueError("eps must be positive")
        if self.metric not in {"euclidean", "cosine"}:
            raise ValueError("metric must be 'euclidean' or 'cosine'")
        if self.feature_space not in {"raw", "normalized"}:
            raise ValueError("feature_space must be 'raw' or 'normalized'")
        if self.min_intersection < 1:
            raise ValueError("min_intersection must be at least 1")
        if self.n_jobs == 0 or self.n_jobs < -1:
            raise ValueError("n_jobs must be -1 or a positive integer")

    def as_dict(self) -> Dict[str, object]:
        return {
            "n_cubes": self.n_cubes,
            "perc_overlap": self.perc_overlap,
            "min_samples": self.min_samples,
            "eps": self.eps,
            "metric": self.metric,
            "feature_space": self.feature_space,
            "eps_sample_size": self.eps_sample_size,
            "eps_fallback_quantile": self.eps_fallback_quantile,
            "random_seed": self.random_seed,
            "min_intersection": self.min_intersection,
            "remove_duplicate_nodes": self.remove_duplicate_nodes,
            "n_jobs": self.n_jobs,
        }


def _source_node_sort_key(source_id: str) -> Tuple[int, int, str]:
    match = re.search(r"cube(\d+)_cluster(\d+)", source_id)
    if match:
        return int(match.group(1)), int(match.group(2)), source_id
    return 2**31 - 1, 2**31 - 1, source_id


def _connected_components(
    node_ids: Tuple[str, ...], edges: Tuple[Tuple[str, str], ...]
) -> Tuple[Tuple[str, ...], ...]:
    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in node_ids}
    for source, target in edges:
        adjacency[source].add(target)
        adjacency[target].add(source)
    remaining = set(node_ids)
    components: List[Tuple[str, ...]] = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: Set[str] = set()
        while stack:
            node = stack.pop()
            if node in component:
                continue
            component.add(node)
            stack.extend(sorted(adjacency[node] - component, reverse=True))
        remaining -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


def _fixed_eps_calibration(config: MapperConfig, features: np.ndarray) -> EpsCalibration:
    if config.eps is None:
        raise AssertionError("fixed epsilon requested without a value")
    return EpsCalibration(
        eps=float(config.eps),
        method="configured",
        min_samples=config.min_samples,
        metric=config.metric,
        sample_count=int(features.shape[0]),
        sample_size_limit=config.eps_sample_size,
        quantile=config.eps_fallback_quantile,
        knee_index=-1,
        distance_min=0.0,
        distance_median=0.0,
        distance_max=0.0,
    )


def build_mapper_graph(
    embeddings: EmbeddingBatch,
    lens: LensResult,
    lens_item_ids: Tuple[str, ...],
    *,
    config: MapperConfig,
) -> MapperGraphArtifact:
    """Build a classical Mapper graph and normalize its output schema."""

    try:
        import kmapper as km
        from sklearn.cluster import DBSCAN
    except ImportError as error:
        raise RuntimeError(
            "Mapper graph construction requires the 'mapper' extra: "
            "pip install -e '.[mapper]'"
        ) from error

    if lens_item_ids != embeddings.item_ids:
        raise ValueError("lens and embedding item IDs must match in the same order")
    if lens.values.shape != (len(embeddings.item_ids),):
        raise ValueError("lens values must align with the embedding rows")
    if config.min_samples > len(embeddings.item_ids):
        raise ValueError("min_samples cannot exceed the number of embedding rows")
    if float(np.ptp(lens.values)) <= np.finfo(np.float64).eps:
        raise ValueError("lens is constant and cannot define a Mapper cover")

    features = embeddings.normalized if config.feature_space == "normalized" else embeddings.raw
    features = np.asarray(features, dtype=np.float64)
    calibration = (
        _fixed_eps_calibration(config, features)
        if config.eps is not None
        else calibrate_dbscan_eps(
            features,
            min_samples=config.min_samples,
            metric=config.metric,
            sample_size=config.eps_sample_size,
            fallback_quantile=config.eps_fallback_quantile,
            random_seed=config.random_seed,
            n_jobs=config.n_jobs,
        )
    )

    mapper = km.KeplerMapper(verbose=config.verbose)
    clusterer = DBSCAN(
        eps=calibration.eps,
        min_samples=config.min_samples,
        metric=config.metric,
        algorithm="brute" if config.metric == "cosine" else "auto",
        n_jobs=config.n_jobs,
    )
    graph: Mapping[str, object] = mapper.map(
        lens.values.reshape(-1, 1),
        features,
        clusterer=clusterer,
        cover=km.Cover(n_cubes=config.n_cubes, perc_overlap=config.perc_overlap),
        nerve=km.GraphNerve(min_intersection=config.min_intersection),
        remove_duplicate_nodes=config.remove_duplicate_nodes,
    )
    source_nodes: Mapping[str, object] = graph["nodes"]
    source_links: Mapping[str, object] = graph["links"]

    sorted_source_ids = sorted(source_nodes, key=_source_node_sort_key)
    canonical_by_source = {
        source_id: f"node-{index:04d}" for index, source_id in enumerate(sorted_source_ids)
    }
    member_indices_by_node: Dict[str, Tuple[int, ...]] = {}
    for source_id in sorted_source_ids:
        members = tuple(sorted({int(value) for value in source_nodes[source_id]}))
        if not members:
            raise ValueError(f"Mapper produced an empty node: {source_id}")
        if members[0] < 0 or members[-1] >= len(embeddings.item_ids):
            raise ValueError(f"Mapper produced an out-of-range member index: {source_id}")
        member_indices_by_node[canonical_by_source[source_id]] = members

    edge_pairs: Set[Tuple[str, str]] = set()
    for source_id, targets in source_links.items():
        if source_id not in canonical_by_source:
            continue
        for target_id in targets:
            if target_id not in canonical_by_source:
                continue
            pair = tuple(
                sorted((canonical_by_source[source_id], canonical_by_source[target_id]))
            )
            if pair[0] != pair[1]:
                edge_pairs.add(pair)
    sorted_edges = tuple(sorted(edge_pairs))
    canonical_ids = tuple(canonical_by_source[source_id] for source_id in sorted_source_ids)
    components = _connected_components(canonical_ids, sorted_edges)
    component_by_node = {
        node_id: f"component-{index:04d}"
        for index, component in enumerate(components)
        for node_id in component
    }

    nodes: List[MapperNode] = []
    for source_id in sorted_source_ids:
        node_id = canonical_by_source[source_id]
        members = member_indices_by_node[node_id]
        node_lens = lens.values[np.asarray(members, dtype=np.int64)]
        nodes.append(
            MapperNode(
                id=node_id,
                source_id=source_id,
                component_id=component_by_node[node_id],
                member_indices=members,
                item_ids=tuple(embeddings.item_ids[index] for index in members),
                lens_min=float(node_lens.min()),
                lens_max=float(node_lens.max()),
                lens_mean=float(node_lens.mean()),
            )
        )

    edges: List[MapperEdge] = []
    for source, target in sorted_edges:
        source_members = set(member_indices_by_node[source])
        target_members = set(member_indices_by_node[target])
        intersection = tuple(sorted(source_members & target_members))
        union_size = len(source_members | target_members)
        if len(intersection) < config.min_intersection:
            raise ValueError("Mapper edge does not satisfy the configured minimum intersection")
        edges.append(
            MapperEdge(
                source=source,
                target=target,
                intersection_indices=intersection,
                intersection_item_ids=tuple(
                    embeddings.item_ids[index] for index in intersection
                ),
                jaccard=len(intersection) / union_size,
            )
        )

    assigned: Set[int] = set()
    memberships = np.zeros(len(embeddings.item_ids), dtype=np.int64)
    for members in member_indices_by_node.values():
        assigned.update(members)
        memberships[np.asarray(members, dtype=np.int64)] += 1
    node_sizes = np.asarray([len(node.member_indices) for node in nodes], dtype=np.int64)
    isolated = sum(
        1 for node_id in canonical_ids if not any(node_id in edge for edge in sorted_edges)
    )
    diagnostics: Dict[str, object] = {
        "node_count": len(nodes),
        "edge_count": len(edges),
        "component_count": len(components),
        "isolated_node_count": isolated,
        "assigned_item_count": len(assigned),
        "unassigned_item_count": len(embeddings.item_ids) - len(assigned),
        "coverage_fraction": len(assigned) / len(embeddings.item_ids),
        "multi_membership_item_count": int(np.sum(memberships > 1)),
        "total_memberships": int(memberships.sum()),
        "node_size_min": int(node_sizes.min()) if node_sizes.size else 0,
        "node_size_median": float(np.median(node_sizes)) if node_sizes.size else 0.0,
        "node_size_max": int(node_sizes.max()) if node_sizes.size else 0,
    }
    return MapperGraphArtifact(
        item_ids=embeddings.item_ids,
        embedding_model=embeddings.model_spec.as_dict(),
        embedding_model_fingerprint=embeddings.model_spec.fingerprint,
        feature_space=config.feature_space,
        lens=lens.metadata(),
        config=config.as_dict(),
        eps_calibration=calibration.as_dict(),
        nodes=tuple(nodes),
        edges=tuple(edges),
        components=components,
        diagnostics=diagnostics,
    )
