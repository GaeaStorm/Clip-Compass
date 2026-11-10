import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clip_compass.data.coco_instances import (
    CocoCategory,
    CocoImage,
    CocoInstance,
    CocoInstanceSubset,
    sha256_file,
)
from clip_compass.explanations.types import (
    ExplanationArtifact,
    ExplanationHypothesis,
    NodeExplanation,
)
from clip_compass.perturbations.masking import (
    MaskingPerturbationEngine,
    translated_control_mask,
)
from clip_compass.perturbations.service import generate_perturbations
from clip_compass.perturbations.types import PerturbationArtifact
from test_mapper_diagnostics import diagnostic_fixture


PIL_AVAILABLE = importlib.util.find_spec("PIL") is not None
COCO_AVAILABLE = importlib.util.find_spec("pycocotools") is not None


def annotation_fixture(graph) -> CocoInstanceSubset:
    return CocoInstanceSubset(
        split="val",
        source_sha256="a" * 64,
        manifest_sha256="b" * 64,
        item_ids=graph.item_ids,
        categories=(
            CocoCategory(1, "person", "person"),
            CocoCategory(2, "car", "vehicle"),
        ),
        images=tuple(
            CocoImage(item_id, 12, 8, f"{item_id}.jpg")
            for item_id in graph.item_ids
        ),
        annotations=(
            CocoInstance(
                id=1,
                image_id="1",
                category_id=1,
                bbox_xywh=(2.0, 2.0, 4.0, 3.0),
                area=12.0,
                iscrowd=False,
                segmentation=[[2, 2, 6, 2, 6, 5, 2, 5]],
            ),
            CocoInstance(
                id=2,
                image_id="2",
                category_id=2,
                bbox_xywh=(1.0, 1.0, 3.0, 2.0),
                area=6.0,
                iscrowd=False,
                segmentation=[[1, 1, 4, 1, 4, 3, 1, 3]],
            ),
            CocoInstance(
                id=3,
                image_id="3",
                category_id=1,
                bbox_xywh=(7.0, 3.0, 3.0, 3.0),
                area=9.0,
                iscrowd=False,
                segmentation=[[7, 3, 10, 3, 10, 6, 7, 6]],
            ),
        ),
    )


def explanation_fixture(graph_path: Path, annotations_path: Path, graph) -> ExplanationArtifact:
    return ExplanationArtifact(
        graph_sha256=sha256_file(graph_path),
        annotations_sha256=sha256_file(annotations_path),
        embedding_model_fingerprint=graph.embedding_model_fingerprint,
        generator={"provider": "test"},
        context_config={"representatives": 1},
        explanations=(
            NodeExplanation(
                node_id="node-0000",
                summary="People distinguish this node.",
                hypotheses=(
                    ExplanationHypothesis(
                        name="Person presence",
                        description="The node contains people.",
                        positive_prompt="a photo containing a person",
                        negative_prompt="a photo without a person",
                        coco_categories=("person",),
                        predicted_effect="both",
                        rationale="The representative image contains a person.",
                        supporting_item_ids=("1",),
                        counterexample_item_ids=("3",),
                        confidence=0.8,
                    ),
                ),
            ),
        ),
    )


@unittest.skipUnless(
    PIL_AVAILABLE and COCO_AVAILABLE,
    "Pillow and pycocotools optional dependencies are required",
)
class PerturbationTests(unittest.TestCase):
    def test_translated_control_preserves_exact_area(self) -> None:
        target = np.zeros((10, 14), dtype=np.bool_)
        target[2:5, 3:7] = True
        control, overlap = translated_control_mask(target, seed=9, max_attempts=100)
        self.assertEqual(int(control.sum()), int(target.sum()))
        self.assertEqual(overlap, 0)
        np.testing.assert_array_equal(
            control, translated_control_mask(target, seed=9, max_attempts=100)[0]
        )

    def test_generates_paired_images_masks_manifest_and_provenance(self) -> None:
        from PIL import Image

        graph, _ = diagnostic_fixture()
        annotations = annotation_fixture(graph)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_path = root / "graph.json"
            graph.save(graph_path)
            annotations_path = root / "annotations.json"
            annotations.save(annotations_path)
            manifest_rows = []
            for index, item_id in enumerate(graph.item_ids):
                image_path = root / "images" / f"{item_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (12, 8), (20 + index * 30, 60, 100)).save(
                    image_path
                )
                manifest_rows.append(
                    {
                        "image_id": item_id,
                        "image_path": image_path.relative_to(root).as_posix(),
                        "captions": [f"Image {item_id}."],
                    }
                )
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                "".join(json.dumps(row) + "\n" for row in manifest_rows),
                encoding="utf-8",
            )
            explanations = explanation_fixture(
                graph_path, annotations_path, graph
            )
            explanations_path = root / "explanations.json"
            explanations.save(explanations_path)
            engine = MaskingPerturbationEngine(fill_mode="gray")
            artifact, paths = generate_perturbations(
                graph,
                explanations,
                annotations,
                engine,
                graph_path=graph_path,
                explanations_path=explanations_path,
                annotations_path=annotations_path,
                manifest_path=manifest_path,
                data_root=root,
                output_dir=root / "run-one",
                seed=17,
            )
            repeated, _ = generate_perturbations(
                graph,
                explanations,
                annotations,
                engine,
                graph_path=graph_path,
                explanations_path=explanations_path,
                annotations_path=annotations_path,
                manifest_path=manifest_path,
                data_root=root,
                output_dir=root / "run-two",
                seed=17,
            )
            loaded = PerturbationArtifact.load(paths["artifact"])
            rows = [
                json.loads(line)
                for line in paths["embedding_manifest"].read_text(
                    encoding="utf-8"
                ).splitlines()
            ]
            pair = artifact.pairs[0]
            target_mask = np.asarray(
                Image.open(paths["run_root"] / pair.target_mask_path)
            ) > 0
            control_mask = np.asarray(
                Image.open(paths["run_root"] / pair.control_mask_path)
            ) > 0
            target_image = np.asarray(
                Image.open(paths["run_root"] / pair.target_image_path)
            )

        self.assertEqual(len(artifact.pairs), 1)
        self.assertEqual(artifact.pairs[0].item_id, "1")
        self.assertEqual(artifact.skipped[0].item_id, "2")
        self.assertEqual(len(rows), 2)
        self.assertEqual({row["variant"] for row in rows}, {"target", "control"})
        self.assertEqual(int(target_mask.sum()), int(control_mask.sum()))
        self.assertTrue(np.all(target_image[target_mask] == 127))
        self.assertEqual(loaded.as_dict(), artifact.as_dict())
        self.assertEqual(
            repeated.pairs[0].control_image_sha256,
            artifact.pairs[0].control_image_sha256,
        )

    def test_rejects_explanation_graph_provenance_mismatch(self) -> None:
        graph, _ = diagnostic_fixture()
        annotations = annotation_fixture(graph)
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_path = root / "graph.json"
            graph.save(graph_path)
            annotations_path = root / "annotations.json"
            annotations.save(annotations_path)
            explanations = explanation_fixture(graph_path, annotations_path, graph)
            explanations = ExplanationArtifact(
                graph_sha256="0" * 64,
                annotations_sha256=explanations.annotations_sha256,
                embedding_model_fingerprint=explanations.embedding_model_fingerprint,
                generator=explanations.generator,
                context_config=explanations.context_config,
                explanations=explanations.explanations,
            )
            explanations_path = root / "explanations.json"
            explanations.save(explanations_path)
            with self.assertRaisesRegex(ValueError, "different Mapper graph"):
                generate_perturbations(
                    graph,
                    explanations,
                    annotations,
                    MaskingPerturbationEngine(),
                    graph_path=graph_path,
                    explanations_path=explanations_path,
                    annotations_path=annotations_path,
                    manifest_path=root / "unused.jsonl",
                    data_root=root,
                    output_dir=root / "run",
                )


if __name__ == "__main__":
    unittest.main()
