import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clip_compass.embeddings.types import ClipModelSpec, EmbeddingBatch
from clip_compass.mapper.diagnostics import write_mapper_diagnostic_report
from clip_compass.mapper.graph import MapperEdge, MapperGraphArtifact, MapperNode


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None


def diagnostic_fixture():
    item_ids = ("1", "2", "3")
    raw = np.asarray([[1.0, 0.0], [0.8, 0.2], [0.0, 1.0]], dtype=np.float32)
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    spec = ClipModelSpec("test", "synthetic", "v1", "identity")
    embeddings = EmbeddingBatch(
        item_ids=item_ids,
        raw=raw,
        normalized=normalized,
        cache_hits=np.zeros(3, dtype=np.bool_),
        model_spec=spec,
        input_sha256=tuple(f"{index:064x}" for index in range(3)),
    )
    nodes = (
        MapperNode(
            id="node-0000",
            source_id="cube0_cluster0",
            component_id="component-0000",
            member_indices=(0, 1),
            item_ids=("1", "2"),
            lens_min=0.0,
            lens_max=0.5,
            lens_mean=0.25,
        ),
        MapperNode(
            id="node-0001",
            source_id="cube1_cluster0",
            component_id="component-0000",
            member_indices=(1, 2),
            item_ids=("2", "3"),
            lens_min=0.5,
            lens_max=1.0,
            lens_mean=0.75,
        ),
    )
    graph = MapperGraphArtifact(
        item_ids=item_ids,
        embedding_model=spec.as_dict(),
        embedding_model_fingerprint=spec.fingerprint,
        feature_space="normalized",
        lens={"name": "synthetic", "source": "normalized"},
        config={"n_cubes": 2},
        eps_calibration={"eps": 0.2, "method": "configured"},
        nodes=nodes,
        edges=(
            MapperEdge(
                source="node-0000",
                target="node-0001",
                intersection_indices=(1,),
                intersection_item_ids=("2",),
                jaccard=1.0 / 3.0,
            ),
        ),
        components=(("node-0000", "node-0001"),),
        diagnostics={
            "node_count": 2,
            "edge_count": 1,
            "component_count": 1,
            "isolated_node_count": 0,
            "assigned_item_count": 3,
            "unassigned_item_count": 0,
            "coverage_fraction": 1.0,
            "multi_membership_item_count": 1,
            "total_memberships": 4,
            "node_size_min": 2,
            "node_size_median": 2.0,
            "node_size_max": 2,
        },
    )
    return graph, embeddings


@unittest.skipUnless(PIL_AVAILABLE, "Pillow optional dependency is not installed")
class MapperDiagnosticTests(unittest.TestCase):
    def test_writes_self_contained_report_with_representatives(self) -> None:
        from PIL import Image

        graph, embeddings = diagnostic_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            rows = []
            for index, item_id in enumerate(graph.item_ids):
                path = root / "images" / f"{item_id}.jpg"
                path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (40, 30), (40 + index * 50, 80, 120)).save(path)
                rows.append(
                    {
                        "image_id": item_id,
                        "image_path": path.relative_to(root).as_posix(),
                        "captions": [f"Caption for image {item_id}."],
                    }
                )
            manifest = root / "manifest.jsonl"
            manifest.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )
            output = root / "report.html"
            write_mapper_diagnostic_report(
                graph,
                embeddings,
                manifest_path=manifest,
                data_root=root,
                output_path=output,
                representatives=2,
                thumbnail_size=96,
            )
            document = output.read_text(encoding="utf-8")

        self.assertIn("data:image/jpeg;base64,", document)
        self.assertIn("node-0000", document)
        self.assertIn("Caption for image 1.", document)
        self.assertNotIn("https://", document)

    def test_rejects_embedding_alignment_mismatch(self) -> None:
        graph, embeddings = diagnostic_fixture()
        mismatched = EmbeddingBatch(
            item_ids=tuple(reversed(embeddings.item_ids)),
            raw=embeddings.raw[::-1].copy(),
            normalized=embeddings.normalized[::-1].copy(),
            cache_hits=embeddings.cache_hits[::-1].copy(),
            model_spec=embeddings.model_spec,
            input_sha256=tuple(reversed(embeddings.input_sha256)),
        )
        with self.assertRaisesRegex(ValueError, "item IDs must match"):
            write_mapper_diagnostic_report(
                graph,
                mismatched,
                manifest_path=Path("unused"),
                data_root=Path("unused"),
                output_path=Path("unused"),
            )


if __name__ == "__main__":
    unittest.main()
