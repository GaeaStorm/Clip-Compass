"""Stable, validated, and portable Mapper graph representation."""

from __future__ import annotations

import csv
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Mapping, Set, Tuple


MAPPER_GRAPH_SCHEMA_VERSION = 1
MAPPER_EXPORT_SCHEMA_VERSION = 1


def _atomic_text(path: Path, content: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    partial.write_text(content, encoding="utf-8")
    os.replace(partial, path)
    return path


def _atomic_csv(
    path: Path,
    fieldnames: Iterable[str],
    rows: Iterable[Mapping[str, object]],
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    with partial.open("w", encoding="utf-8", newline="") as output:
        writer = csv.DictWriter(output, fieldnames=tuple(fieldnames))
        writer.writeheader()
        writer.writerows(rows)
    os.replace(partial, path)
    return path


def _connected_components(
    node_ids: Tuple[str, ...], edges: Tuple["MapperEdge", ...]
) -> Tuple[Tuple[str, ...], ...]:
    adjacency: Dict[str, Set[str]] = {node_id: set() for node_id in node_ids}
    for edge in edges:
        adjacency[edge.source].add(edge.target)
        adjacency[edge.target].add(edge.source)
    remaining = set(node_ids)
    components = []
    while remaining:
        start = min(remaining)
        stack = [start]
        component: Set[str] = set()
        while stack:
            node_id = stack.pop()
            if node_id in component:
                continue
            component.add(node_id)
            stack.extend(sorted(adjacency[node_id] - component, reverse=True))
        remaining -= component
        components.append(tuple(sorted(component)))
    return tuple(sorted(components, key=lambda component: component[0]))


@dataclass(frozen=True)
class MapperNode:
    id: str
    source_id: str
    component_id: str
    member_indices: Tuple[int, ...]
    item_ids: Tuple[str, ...]
    lens_min: float
    lens_max: float
    lens_mean: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "source_id": self.source_id,
            "component_id": self.component_id,
            "member_indices": list(self.member_indices),
            "item_ids": list(self.item_ids),
            "size": len(self.member_indices),
            "lens": {"min": self.lens_min, "max": self.lens_max, "mean": self.lens_mean},
        }


@dataclass(frozen=True)
class MapperEdge:
    source: str
    target: str
    intersection_indices: Tuple[int, ...]
    intersection_item_ids: Tuple[str, ...]
    jaccard: float

    def as_dict(self) -> Dict[str, object]:
        return {
            "source": self.source,
            "target": self.target,
            "intersection_indices": list(self.intersection_indices),
            "intersection_item_ids": list(self.intersection_item_ids),
            "intersection_size": len(self.intersection_indices),
            "jaccard": self.jaccard,
        }


@dataclass(frozen=True)
class MapperGraphArtifact:
    item_ids: Tuple[str, ...]
    embedding_model: Mapping[str, object]
    embedding_model_fingerprint: str
    feature_space: str
    lens: Mapping[str, object]
    config: Mapping[str, object]
    eps_calibration: Mapping[str, object]
    nodes: Tuple[MapperNode, ...]
    edges: Tuple[MapperEdge, ...]
    components: Tuple[Tuple[str, ...], ...]
    diagnostics: Mapping[str, object]

    def __post_init__(self) -> None:
        self.validate()

    def validate(self) -> None:
        """Reject internally inconsistent or non-canonical graph artifacts."""

        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("Mapper graph item IDs must be unique")
        node_ids = tuple(node.id for node in self.nodes)
        if len(set(node_ids)) != len(node_ids):
            raise ValueError("Mapper graph node IDs must be unique")
        if node_ids != tuple(sorted(node_ids)):
            raise ValueError("Mapper graph nodes must be sorted by canonical ID")
        if len({node.source_id for node in self.nodes}) != len(self.nodes):
            raise ValueError("Mapper graph source node IDs must be unique")

        members_by_node: Dict[str, Set[int]] = {}
        for node in self.nodes:
            if not node.member_indices:
                raise ValueError(f"Mapper node {node.id!r} must contain at least one item")
            if node.member_indices != tuple(sorted(set(node.member_indices))):
                raise ValueError(f"Mapper node {node.id!r} member indices are not canonical")
            if node.member_indices[0] < 0 or node.member_indices[-1] >= len(self.item_ids):
                raise ValueError(f"Mapper node {node.id!r} has an out-of-range member index")
            expected_item_ids = tuple(self.item_ids[index] for index in node.member_indices)
            if node.item_ids != expected_item_ids:
                raise ValueError(f"Mapper node {node.id!r} item IDs do not match member indices")
            lens_statistics = (node.lens_min, node.lens_mean, node.lens_max)
            if not all(math.isfinite(value) for value in lens_statistics):
                raise ValueError(f"Mapper node {node.id!r} has non-finite lens statistics")
            if not node.lens_min <= node.lens_mean <= node.lens_max:
                raise ValueError(f"Mapper node {node.id!r} has invalid lens statistics")
            members_by_node[node.id] = set(node.member_indices)

        edge_pairs = []
        for edge in self.edges:
            if edge.source not in members_by_node or edge.target not in members_by_node:
                raise ValueError("Mapper edge refers to an unknown node")
            if edge.source >= edge.target:
                raise ValueError("Mapper edges must use sorted, distinct endpoints")
            pair = (edge.source, edge.target)
            edge_pairs.append(pair)
            expected_intersection = tuple(
                sorted(members_by_node[edge.source] & members_by_node[edge.target])
            )
            if edge.intersection_indices != expected_intersection or not expected_intersection:
                raise ValueError(f"Mapper edge {pair!r} has an invalid intersection")
            expected_item_ids = tuple(self.item_ids[index] for index in expected_intersection)
            if edge.intersection_item_ids != expected_item_ids:
                raise ValueError(f"Mapper edge {pair!r} item IDs do not match its intersection")
            union_size = len(members_by_node[edge.source] | members_by_node[edge.target])
            expected_jaccard = len(expected_intersection) / union_size
            if not math.isclose(edge.jaccard, expected_jaccard, rel_tol=1e-12, abs_tol=1e-15):
                raise ValueError(f"Mapper edge {pair!r} has an invalid Jaccard score")
        if tuple(edge_pairs) != tuple(sorted(set(edge_pairs))):
            raise ValueError("Mapper edges must be unique and canonically sorted")

        expected_components = _connected_components(node_ids, self.edges)
        if self.components != expected_components:
            raise ValueError("Mapper components do not match graph connectivity")
        expected_component_ids = {
            node_id: f"component-{index:04d}"
            for index, component in enumerate(self.components)
            for node_id in component
        }
        for node in self.nodes:
            if node.component_id != expected_component_ids[node.id]:
                raise ValueError(f"Mapper node {node.id!r} has an invalid component ID")

        memberships = [0] * len(self.item_ids)
        for members in members_by_node.values():
            for index in members:
                memberships[index] += 1
        assigned_count = sum(value > 0 for value in memberships)
        isolated_count = sum(
            all(node_id not in pair for pair in edge_pairs) for node_id in node_ids
        )
        node_sizes = sorted(len(node.member_indices) for node in self.nodes)
        if node_sizes:
            middle = len(node_sizes) // 2
            median = (
                float(node_sizes[middle])
                if len(node_sizes) % 2
                else (node_sizes[middle - 1] + node_sizes[middle]) / 2.0
            )
        else:
            median = 0.0
        expected_diagnostics: Dict[str, object] = {
            "node_count": len(self.nodes),
            "edge_count": len(self.edges),
            "component_count": len(self.components),
            "isolated_node_count": isolated_count,
            "assigned_item_count": assigned_count,
            "unassigned_item_count": len(self.item_ids) - assigned_count,
            "coverage_fraction": assigned_count / len(self.item_ids) if self.item_ids else 0.0,
            "multi_membership_item_count": sum(value > 1 for value in memberships),
            "total_memberships": sum(memberships),
            "node_size_min": node_sizes[0] if node_sizes else 0,
            "node_size_median": median,
            "node_size_max": node_sizes[-1] if node_sizes else 0,
        }
        for key, expected in expected_diagnostics.items():
            if key not in self.diagnostics:
                raise ValueError(f"Mapper diagnostics are missing {key!r}")
            actual = self.diagnostics[key]
            if isinstance(expected, float):
                if not math.isclose(float(actual), expected, rel_tol=1e-12, abs_tol=1e-15):
                    raise ValueError(f"Mapper diagnostic {key!r} is inconsistent")
            elif actual != expected:
                raise ValueError(f"Mapper diagnostic {key!r} is inconsistent")

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": MAPPER_GRAPH_SCHEMA_VERSION,
            "item_ids": list(self.item_ids),
            "embedding_model": dict(self.embedding_model),
            "embedding_model_fingerprint": self.embedding_model_fingerprint,
            "feature_space": self.feature_space,
            "lens": dict(self.lens),
            "config": dict(self.config),
            "eps_calibration": dict(self.eps_calibration),
            "nodes": [node.as_dict() for node in self.nodes],
            "edges": [edge.as_dict() for edge in self.edges],
            "components": [list(component) for component in self.components],
            "diagnostics": dict(self.diagnostics),
        }

    def save(self, path: Path) -> Path:
        self.validate()
        return _atomic_text(path, json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n")

    def export_bundle(self, output_dir: Path) -> Mapping[str, Path]:
        """Write JSON plus normalized CSV tables for notebooks and graph tools."""

        self.validate()
        output_dir.mkdir(parents=True, exist_ok=True)
        node_component = {node.id: node.component_id for node in self.nodes}
        paths = {
            "graph": self.save(output_dir / "graph.json"),
            "nodes": _atomic_csv(
                output_dir / "nodes.csv",
                (
                    "node_id",
                    "source_id",
                    "component_id",
                    "size",
                    "lens_min",
                    "lens_mean",
                    "lens_max",
                ),
                (
                    {
                        "node_id": node.id,
                        "source_id": node.source_id,
                        "component_id": node.component_id,
                        "size": len(node.member_indices),
                        "lens_min": node.lens_min,
                        "lens_mean": node.lens_mean,
                        "lens_max": node.lens_max,
                    }
                    for node in self.nodes
                ),
            ),
            "edges": _atomic_csv(
                output_dir / "edges.csv",
                (
                    "source",
                    "target",
                    "source_component",
                    "target_component",
                    "intersection_size",
                    "jaccard",
                ),
                (
                    {
                        "source": edge.source,
                        "target": edge.target,
                        "source_component": node_component[edge.source],
                        "target_component": node_component[edge.target],
                        "intersection_size": len(edge.intersection_indices),
                        "jaccard": edge.jaccard,
                    }
                    for edge in self.edges
                ),
            ),
            "memberships": _atomic_csv(
                output_dir / "memberships.csv",
                ("node_id", "component_id", "member_index", "item_id"),
                (
                    {
                        "node_id": node.id,
                        "component_id": node.component_id,
                        "member_index": index,
                        "item_id": item_id,
                    }
                    for node in self.nodes
                    for index, item_id in zip(node.member_indices, node.item_ids)
                ),
            ),
            "components": _atomic_csv(
                output_dir / "components.csv",
                ("component_id", "node_count", "node_ids"),
                (
                    {
                        "component_id": f"component-{index:04d}",
                        "node_count": len(component),
                        "node_ids": ";".join(component),
                    }
                    for index, component in enumerate(self.components)
                ),
            ),
        }
        summary = {
            "export_schema_version": MAPPER_EXPORT_SCHEMA_VERSION,
            "graph_schema_version": MAPPER_GRAPH_SCHEMA_VERSION,
            "embedding_model_fingerprint": self.embedding_model_fingerprint,
            "feature_space": self.feature_space,
            "lens": dict(self.lens),
            "config": dict(self.config),
            "eps_calibration": dict(self.eps_calibration),
            "diagnostics": dict(self.diagnostics),
            "files": {key: path.name for key, path in paths.items()},
        }
        paths["summary"] = _atomic_text(
            output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True) + "\n"
        )
        return paths

    @classmethod
    def load(cls, path: Path) -> "MapperGraphArtifact":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != MAPPER_GRAPH_SCHEMA_VERSION:
            raise ValueError("unsupported Mapper graph schema version")
        nodes = tuple(
            MapperNode(
                id=str(row["id"]),
                source_id=str(row["source_id"]),
                component_id=str(row["component_id"]),
                member_indices=tuple(int(value) for value in row["member_indices"]),
                item_ids=tuple(str(value) for value in row["item_ids"]),
                lens_min=float(row["lens"]["min"]),
                lens_max=float(row["lens"]["max"]),
                lens_mean=float(row["lens"]["mean"]),
            )
            for row in data["nodes"]
        )
        edges = tuple(
            MapperEdge(
                source=str(row["source"]),
                target=str(row["target"]),
                intersection_indices=tuple(
                    int(value) for value in row["intersection_indices"]
                ),
                intersection_item_ids=tuple(
                    str(value) for value in row["intersection_item_ids"]
                ),
                jaccard=float(row["jaccard"]),
            )
            for row in data["edges"]
        )
        return cls(
            item_ids=tuple(str(value) for value in data["item_ids"]),
            embedding_model=data["embedding_model"],
            embedding_model_fingerprint=str(data["embedding_model_fingerprint"]),
            feature_space=str(data["feature_space"]),
            lens=data["lens"],
            config=data["config"],
            eps_calibration=data["eps_calibration"],
            nodes=nodes,
            edges=edges,
            components=tuple(tuple(str(value) for value in row) for row in data["components"]),
            diagnostics=data["diagnostics"],
        )
