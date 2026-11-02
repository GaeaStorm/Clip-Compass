import csv
import importlib.util
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path

import numpy as np

from clip_compass.embeddings.types import ClipModelSpec, EmbeddingBatch
from clip_compass.mapper.builder import MapperConfig, build_mapper_graph
from clip_compass.mapper.calibration import calibrate_dbscan_eps
from clip_compass.mapper.graph import MapperGraphArtifact
from clip_compass.mapper.lenses import LensResult


MAPPER_DEPS_AVAILABLE = all(
    importlib.util.find_spec(name) is not None for name in ("kmapper", "kneed", "sklearn")
)


def synthetic_inputs():
    x = np.linspace(0.0, 1.0, 20, dtype=np.float32)
    raw = np.column_stack([np.ones_like(x), x, np.full_like(x, 0.25)])
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    item_ids = tuple(f"item-{index:02d}" for index in range(x.shape[0]))
    embeddings = EmbeddingBatch(
        item_ids=item_ids,
        raw=raw,
        normalized=normalized,
        cache_hits=np.zeros(x.shape[0], dtype=np.bool_),
        model_spec=ClipModelSpec("test", "synthetic", "v1", "identity"),
        input_sha256=tuple(f"{index:064x}" for index in range(x.shape[0])),
    )
    lens = LensResult(
        name="pca1_raw",
        source="raw",
        raw_values=x.astype(np.float64),
        values=x.astype(np.float64),
        scaled=True,
        parameters={"synthetic": True},
    )
    return embeddings, lens, item_ids


@unittest.skipUnless(MAPPER_DEPS_AVAILABLE, "Mapper optional dependencies are not installed")
class MapperBuilderTests(unittest.TestCase):
    def test_builds_stable_path_graph_and_round_trips(self) -> None:
        embeddings, lens, item_ids = synthetic_inputs()
        config = MapperConfig(
            n_cubes=5,
            perc_overlap=0.5,
            min_samples=2,
            eps=0.2,
            feature_space="raw",
        )
        graph = build_mapper_graph(embeddings, lens, item_ids, config=config)

        self.assertGreaterEqual(len(graph.nodes), 4)
        self.assertGreaterEqual(len(graph.edges), 3)
        self.assertEqual(graph.diagnostics["coverage_fraction"], 1.0)
        self.assertEqual(graph.diagnostics["component_count"], 1)
        self.assertEqual([node.id for node in graph.nodes], sorted(node.id for node in graph.nodes))
        for edge in graph.edges:
            self.assertGreaterEqual(len(edge.intersection_indices), 1)
            self.assertGreater(edge.jaccard, 0.0)

        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "graph.json"
            graph.save(path)
            loaded = MapperGraphArtifact.load(path)
        self.assertEqual(loaded.as_dict(), graph.as_dict())

    def test_exports_normalized_graph_bundle(self) -> None:
        embeddings, lens, item_ids = synthetic_inputs()
        graph = build_mapper_graph(
            embeddings,
            lens,
            item_ids,
            config=MapperConfig(
                n_cubes=5,
                perc_overlap=0.5,
                min_samples=2,
                eps=0.2,
                feature_space="raw",
            ),
        )
        with tempfile.TemporaryDirectory() as temporary_directory:
            paths = graph.export_bundle(Path(temporary_directory) / "export")
            self.assertEqual(
                set(paths), {"graph", "nodes", "edges", "memberships", "components", "summary"}
            )
            with paths["nodes"].open(newline="", encoding="utf-8") as input_file:
                nodes = list(csv.DictReader(input_file))
            with paths["memberships"].open(newline="", encoding="utf-8") as input_file:
                memberships = list(csv.DictReader(input_file))
            self.assertEqual(len(nodes), len(graph.nodes))
            self.assertEqual(len(memberships), graph.diagnostics["total_memberships"])
            self.assertEqual(MapperGraphArtifact.load(paths["graph"]).as_dict(), graph.as_dict())

    def test_rejects_inconsistent_serialized_memberships(self) -> None:
        embeddings, lens, item_ids = synthetic_inputs()
        graph = build_mapper_graph(
            embeddings,
            lens,
            item_ids,
            config=MapperConfig(eps=0.2, feature_space="raw"),
        )
        broken_node = replace(
            graph.nodes[0],
            item_ids=("wrong-item",) + graph.nodes[0].item_ids[1:],
        )
        with self.assertRaisesRegex(ValueError, "item IDs do not match"):
            replace(graph, nodes=(broken_node,) + graph.nodes[1:])

    def test_rejects_misaligned_lens_ids(self) -> None:
        embeddings, lens, item_ids = synthetic_inputs()
        with self.assertRaises(ValueError):
            build_mapper_graph(
                embeddings,
                lens,
                tuple(reversed(item_ids)),
                config=MapperConfig(eps=0.2, feature_space="raw"),
            )

    def test_epsilon_calibration_is_deterministic(self) -> None:
        rng = np.random.default_rng(7)
        features = rng.normal(size=(100, 4))
        first = calibrate_dbscan_eps(
            features, min_samples=3, sample_size=40, random_seed=123
        )
        second = calibrate_dbscan_eps(
            features, min_samples=3, sample_size=40, random_seed=123
        )
        self.assertEqual(first, second)
        self.assertGreater(first.eps, 0.0)


if __name__ == "__main__":
    unittest.main()
