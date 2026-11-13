import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

from test_verification import Fixture

from clip_compass.cli import main
from clip_compass.data.coco_instances import (
    CocoCategory,
    CocoImage,
    CocoInstance,
    CocoInstanceSubset,
)
from clip_compass.explanations.types import ExplanationArtifact
from clip_compass.verification.controls import (
    ControlConfig,
    build_controls,
    load_control_key,
    present_categories,
    save_controls,
)
from clip_compass.verification.evaluation import evaluate_judges
from clip_compass.verification.types import (
    HypothesisVerdict,
    VerdictArtifact,
    VerdictConfig,
)

CATEGORIES = (
    CocoCategory(1, "person", "person"),
    CocoCategory(2, "dog", "animal"),
    CocoCategory(3, "car", "vehicle"),
    CocoCategory(4, "cat", "animal"),
)


def _annotations(item_ids):
    """person everywhere; dog in 0-4 (node-0000); car in 6-8 (node-0001); a tiny cat in 0."""

    rows = []

    def add(item_id, category_id, area):
        rows.append(
            CocoInstance(
                len(rows) + 1, item_id, category_id, (0.0, 0.0, 10.0, 10.0), area, False, []
            )
        )

    for item_id in item_ids:
        add(item_id, 1, 900.0)
        if int(item_id) <= 4:
            add(item_id, 2, 1500.0)
        if 6 <= int(item_id) <= 8:
            add(item_id, 3, 2000.0)
        if item_id == "0":
            add(item_id, 4, 20.0)  # 0.2% of a 100x100 image: below the 1% floor
    return CocoInstanceSubset(
        split="val",
        source_sha256="0" * 64,
        manifest_sha256="0" * 64,
        item_ids=tuple(item_ids),
        categories=CATEGORIES,
        images=tuple(CocoImage(item_id, 100, 100, f"{item_id}.jpg") for item_id in item_ids),
        annotations=tuple(rows),
    )


class ControlTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self._directory.name))
        self.annotations = _annotations(self.fixture.graph.item_ids)
        self.annotations_path = self.fixture.root / "annotations.json"
        self.annotations.save(self.annotations_path)

    def tearDown(self) -> None:
        self._directory.cleanup()

    def build(self, with_explanations=True):
        return build_controls(
            self.fixture.graph,
            self.annotations,
            graph_path=self.fixture.graph_path,
            annotations_path=self.annotations_path,
            explanations=self.fixture.explanations if with_explanations else None,
        )

    def test_tiny_objects_do_not_count_as_present(self) -> None:
        present = present_categories(self.annotations, ("0", "5"), 0.01)
        self.assertEqual(present["0"], {"person", "dog"})
        self.assertEqual(present["5"], {"person"})

    def test_builds_each_kind_with_the_expected_answer(self) -> None:
        artifact, keys = self.build()
        table = {key.hypothesis_id: (key.kind, key.expected, key.categories) for key in keys}
        self.assertEqual(
            table,
            {
                "node-0000-h00": ("obvious", "accept", ("dog",)),
                "node-0000-h01": ("common", "reject", ("person",)),
                "node-0000-h02": ("swapped", "reject", ("car",)),
                "node-0001-h00": ("obvious", "accept", ("car",)),
                "node-0001-h01": ("common", "reject", ("person",)),
                "node-0001-h02": ("swapped", "reject", ("dog",)),
            },
        )
        dog = artifact.explanations[0].hypotheses[0]
        self.assertEqual(dog.positive_prompt, "a photo containing a dog")
        self.assertIn("5 of 6", dog.rationale)
        swapped = artifact.explanations[0].hypotheses[2]
        self.assertIn("swapped from node-0001", swapped.name)
        self.assertEqual(keys[2].source_node_id, "node-0001")

    def test_swaps_are_skipped_without_real_explanations(self) -> None:
        _, keys = self.build(with_explanations=False)
        self.assertNotIn("swapped", {key.kind for key in keys})

    def test_rejects_invalid_config(self) -> None:
        with self.assertRaises(ValueError):
            ControlConfig(min_enrichment=1.0)
        with self.assertRaises(ValueError):
            ControlConfig(max_common_enrichment=3.0)

    def test_save_ties_key_to_the_controls_file(self) -> None:
        artifact, keys = self.build()
        explanations_path, key_path = save_controls(
            artifact, keys, self.fixture.root / "controls.json", self.fixture.root / "key.json"
        )
        self.assertEqual(ExplanationArtifact.load(explanations_path), artifact)
        sha, loaded = load_control_key(key_path)
        self.assertEqual(loaded, keys)
        self.assertEqual(len(sha), 64)

    def _verdicts(self, sha, labels):
        hypotheses = tuple(
            HypothesisVerdict(
                hypothesis_id=hid,
                node_id=hid[:9],
                name=hid,
                coco_categories=("x",),
                predicted_effect="both",
                confidence=0.5,
                member_count=4,
                visible_members=4,
                testable_members=4,
                tests={},
                label=label,
                reason=None,
                prediction_confirmed=None,
            )
            for hid, label in labels.items()
        )
        return VerdictArtifact("r" * 64, sha, "g" * 64, "f", VerdictConfig(), hypotheses)

    def test_evaluation_scores_hits_and_false_alarms_per_judge(self) -> None:
        artifact, keys = self.build()
        _, key_path = save_controls(
            artifact, keys, self.fixture.root / "controls.json", self.fixture.root / "key.json"
        )
        sha, _ = load_control_key(key_path)
        labels = {
            "node-0000-h00": "supported",
            "node-0001-h00": "supported",
            "node-0000-h01": "not_supported",
            "node-0001-h01": "detected_not_explanatory",
            "node-0000-h02": "not_supported",
            "node-0001-h02": "insufficient_evidence",
        }
        verdicts_path = self.fixture.root / "verdicts" / "verdicts.json"
        self._verdicts(sha, labels).save(verdicts_path.parent)
        baseline_path = self.fixture.root / "baseline.json"
        baseline_path.write_text(
            json.dumps(
                {
                    "explanations_sha256": sha,
                    "scores": [
                        {
                            "node_id": key.node_id,
                            "hypothesis_index": int(key.hypothesis_id[-2:]),
                            "accepted": True,
                        }
                        for key in keys
                    ],
                }
            )
        )
        evaluation = evaluate_judges(
            key_path, verdicts_path=verdicts_path, baseline_path=baseline_path
        )
        summary = evaluation["summary"]
        self.assertEqual(
            (summary["agent"]["hit_rate"], summary["agent"]["false_alarm_rate"]), (1.0, 1.0)
        )
        self.assertEqual(summary["text_check"]["false_alarm_rate"], 1.0)
        self.assertEqual(
            (summary["masking"]["hit_rate"], summary["masking"]["false_alarm_rate"]), (1.0, 0.0)
        )
        self.assertEqual(summary["masking"]["undecided"], 1)
        self.assertEqual(summary["masking"]["false_alarm_rate_decided"], 0.0)
        self.assertEqual(summary["masking_lenient"]["false_alarm_rate_decided"], 1 / 3)
        self.assertEqual(summary["masking_lenient"]["false_alarm_rate"], 0.25)
        self.assertEqual(summary["masking"]["accuracy"], 1.0)

        stale = self._verdicts("f" * 64, labels)
        stale.save(self.fixture.root / "stale")
        with self.assertRaisesRegex(ValueError, "different controls file"):
            evaluate_judges(key_path, verdicts_path=self.fixture.root / "stale" / "verdicts.json")

    def test_cli_writes_controls_and_key(self) -> None:
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            code = main(
                [
                    "verification",
                    "controls",
                    "--graph",
                    str(self.fixture.graph_path),
                    "--annotations",
                    str(self.annotations_path),
                    "--explanations",
                    str(self.fixture.explanations_path),
                    "--output",
                    str(self.fixture.root / "controls.json"),
                ]
            )
        self.assertEqual(code, 0)
        summary = json.loads(output.getvalue())
        self.assertEqual(summary["controls"], {"obvious": 2, "common": 2, "swapped": 2})
        self.assertTrue((self.fixture.root / "controls-key.json").is_file())


if __name__ == "__main__":
    unittest.main()
