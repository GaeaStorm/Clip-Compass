import contextlib
import io
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clip_compass.cli import main
from clip_compass.data.coco_instances import sha256_file
from clip_compass.embeddings.types import ClipModelSpec, EmbeddingBatch
from clip_compass.explanations.types import (
    ExplanationArtifact,
    ExplanationHypothesis,
    NodeExplanation,
)
from clip_compass.mapper.graph import MapperGraphArtifact, MapperNode
from clip_compass.verification import (
    VerdictArtifact,
    VerdictConfig,
    aggregate_verdicts,
    auroc,
    benjamini_hochberg,
    object_rank,
    rank_sum_p_value,
    score_prompt_baseline,
)

IMAGE_SPEC = ClipModelSpec("test", "synthetic", "v1", "identity")
TEXT_SPEC = ClipModelSpec("test", "synthetic", "v1", "tokenizer")
AXES = np.eye(4)
K = 39
CONTROLS = [round(0.01 * index, 4) for index in range(K)]


def _unit(vector) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    return vector / np.linalg.norm(vector)


def _batch(item_ids, vectors, spec, modality="image", texts=()) -> EmbeddingBatch:
    raw = np.asarray(vectors, dtype=np.float32)
    normalized = raw / np.linalg.norm(raw, axis=1, keepdims=True)
    return EmbeddingBatch(
        item_ids=tuple(item_ids),
        raw=raw,
        normalized=normalized,
        cache_hits=np.zeros(len(item_ids), dtype=np.bool_),
        model_spec=spec,
        input_sha256=tuple(f"{index:064x}" for index in range(len(item_ids))),
        modality=modality,
        input_texts=tuple(texts),
    )


def _hypothesis(name, effect, categories=("dog",), confidence=0.7) -> ExplanationHypothesis:
    return ExplanationHypothesis(
        name=name,
        description=f"{name} description",
        positive_prompt=f"a photo of {name}",
        negative_prompt=f"a photo without {name}",
        coco_categories=categories,
        predicted_effect=effect,
        rationale="synthetic",
        supporting_item_ids=(),
        counterexample_item_ids=(),
        confidence=confidence,
    )


def _image(item_id, status="testable", centroid=0, prompt=0):
    """A ``perturb run`` image record whose object beats exactly that many controls."""

    record = {"item_id": item_id, "status": status}
    if status == "testable":
        record["object"] = {"centroid": (centroid - 0.5) * 0.01, "prompt": (prompt - 0.5) * 0.01}
        record["controls"] = {"centroid": list(CONTROLS), "prompt": list(CONTROLS)}
        record["p"] = {"centroid": 1.0, "prompt": 1.0}
    return record


class Fixture:
    """Ten images in two Mapper nodes, plus hand-built ``perturb run`` results.

    Hypotheses and the verdict each should reach:
    * node-0000-h00 "dog": objects beat every control on both statistics -> supported.
    * node-0000-h01 "texture": beats controls on prompt only -> detected_not_explanatory.
    * node-0000-h02 "noise": middling ranks -> not_supported (no_effect).
    * node-0001-h00 "car": one testable, one no-room, two absent -> insufficient_evidence.
    * node-0001-h01 "boat": visible in 1 of 4 members -> not_supported (low_coverage).
    * node-0001-h02 "unicorn": not a COCO category -> untestable.
    """

    def __init__(self, root: Path) -> None:
        self.root = root
        item_ids = tuple(str(index) for index in range(10))
        vectors = [
            _unit(AXES[0] + 0.05 * index * AXES[1] + 0.03 * AXES[3]) for index in range(6)
        ] + [_unit(AXES[2] + 0.05 * index * AXES[1]) for index in range(4)]
        self.images = _batch(item_ids, vectors, IMAGE_SPEC)
        nodes = (
            MapperNode(
                "node-0000", "c0", "component-0000", tuple(range(6)), item_ids[:6], 0.0, 0.4, 0.2
            ),
            MapperNode(
                "node-0001",
                "c1",
                "component-0001",
                tuple(range(6, 10)),
                item_ids[6:],
                0.6,
                1.0,
                0.8,
            ),
        )
        self.graph = MapperGraphArtifact(
            item_ids=item_ids,
            embedding_model=IMAGE_SPEC.as_dict(),
            embedding_model_fingerprint=IMAGE_SPEC.fingerprint,
            feature_space="normalized",
            lens={"name": "synthetic"},
            config={},
            eps_calibration={},
            nodes=nodes,
            edges=(),
            components=(("node-0000",), ("node-0001",)),
            diagnostics={
                "node_count": 2,
                "edge_count": 0,
                "component_count": 2,
                "isolated_node_count": 2,
                "assigned_item_count": 10,
                "unassigned_item_count": 0,
                "coverage_fraction": 1.0,
                "multi_membership_item_count": 0,
                "total_memberships": 10,
                "node_size_min": 4,
                "node_size_median": 5.0,
                "node_size_max": 6,
            },
        )
        self.graph_path = root / "graph.json"
        self.graph.save(self.graph_path)

        self.explanations = ExplanationArtifact(
            graph_sha256=sha256_file(self.graph_path),
            annotations_sha256="a" * 64,
            embedding_model_fingerprint=IMAGE_SPEC.fingerprint,
            generator={"provider": "test"},
            context_config={},
            explanations=(
                NodeExplanation(
                    "node-0000",
                    "dogs",
                    (
                        _hypothesis("dog", "both", confidence=0.9),
                        _hypothesis("texture", "decrease_text_similarity"),
                        _hypothesis("noise", "move_away_from_node_centroid"),
                    ),
                ),
                NodeExplanation(
                    "node-0001",
                    "cars",
                    (
                        _hypothesis("car", "move_away_from_node_centroid", ("car",)),
                        _hypothesis("boat", "decrease_text_similarity", ("boat",)),
                        _hypothesis("unicorn", "both", ("unicorn",)),
                    ),
                ),
            ),
        )
        self.explanations_path = root / "explanations.json"
        self.explanations.save(self.explanations_path)

        prompt_ids, prompt_vectors, texts = [], [], []
        for node_id, axis in (("node-0000", 0), ("node-0001", 2)):
            for index in range(3):
                prompt_ids += [
                    f"{node_id}-h{index:02d}-positive",
                    f"{node_id}-h{index:02d}-negative",
                ]
                prompt_vectors += [AXES[axis], AXES[3]]
                texts += ["positive", "negative"]
        self.prompts = _batch(prompt_ids, prompt_vectors, TEXT_SPEC, "text", texts)

        def hypothesis(node_id, index, name, effect, categories, images, unknown=()):
            return {
                "hypothesis_id": f"{node_id}-h{index:02d}",
                "node_id": node_id,
                "name": name,
                "coco_categories": list(categories),
                "unknown_categories": list(unknown),
                "predicted_effect": effect,
                "positive_prompt": f"a photo of {name}",
                "member_count": 6 if node_id == "node-0000" else 4,
                "images": images,
                # the engine's own verdict; required by the results format, unused here
                "verdict": {
                    "label": "untestable",
                    "reason": None,
                    "member_count": 6 if node_id == "node-0000" else 4,
                    "visible_members": 0,
                    "testable_members": 0,
                    "tallies": {
                        name: {"exceeding": 0, "testable": 0, "passes": False}
                        for name in ("centroid", "prompt")
                    },
                    "prediction_confirmed": None,
                },
            }

        self.results = {
            "schema_version": 1,
            "inputs": {
                "graph_sha256": sha256_file(self.graph_path),
                "explanations_sha256": sha256_file(self.explanations_path),
            },
            "model_fingerprint": IMAGE_SPEC.fingerprint,
            "config": {"controls": K},
            "min_identity_cosine": 1.0,
            "hypotheses": [
                hypothesis(
                    "node-0000",
                    0,
                    "dog",
                    "both",
                    ["dog"],
                    [_image(i, centroid=K, prompt=K) for i in item_ids[:5]]
                    + [_image(item_ids[5], "absent")],
                ),
                hypothesis(
                    "node-0000",
                    1,
                    "texture",
                    "decrease_text_similarity",
                    ["dog"],
                    [_image(i, centroid=10, prompt=K) for i in item_ids[:4]]
                    + [_image(i, "absent") for i in item_ids[4:6]],
                ),
                hypothesis(
                    "node-0000",
                    2,
                    "noise",
                    "move_away_from_node_centroid",
                    ["dog"],
                    [_image(i, centroid=19, prompt=19) for i in item_ids[:3]]
                    + [_image(i, "too_small_in_crop") for i in item_ids[3:6]],
                ),
                hypothesis(
                    "node-0001",
                    0,
                    "car",
                    "move_away_from_node_centroid",
                    ["car"],
                    [
                        _image("6", centroid=K, prompt=K),
                        _image("7", "no_control_room"),
                        _image("8", "absent"),
                        _image("9", "absent"),
                    ],
                ),
                hypothesis(
                    "node-0001",
                    1,
                    "boat",
                    "decrease_text_similarity",
                    ["boat"],
                    [_image("6", centroid=K, prompt=K)]
                    + [_image(i, "absent") for i in ("7", "8", "9")],
                ),
                hypothesis("node-0001", 2, "unicorn", "both", ["unicorn"], [], unknown=["unicorn"]),
            ],
        }
        self.results_path = self.write_results(self.results)

    def write_results(self, results, name="run") -> Path:
        path = self.root / name / "verification.json"
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(results), encoding="utf-8")
        return path


class StatisticsTests(unittest.TestCase):
    def test_object_rank_counts_controls_beaten_and_ties_count_against(self) -> None:
        self.assertEqual(object_rank(0.5, [0.1, 0.2, 0.9]), 2)
        self.assertEqual(object_rank(0.2, [0.1, 0.2, 0.2]), 1)
        with self.assertRaises(ValueError):
            object_rank(0.1, [])

    def test_single_image_p_value_matches_the_per_image_empirical_p(self) -> None:
        self.assertAlmostEqual(rank_sum_p_value([K], [K]), 1 / 40)
        self.assertAlmostEqual(rank_sum_p_value([0], [K]), 1.0)
        self.assertAlmostEqual(rank_sum_p_value([K - 1], [K]), 2 / 40)

    def test_two_images_that_beat_every_control_give_one_over_1600(self) -> None:
        self.assertAlmostEqual(rank_sum_p_value([K, K], [K, K]), 1 / 1600)

    def test_consistent_moderate_effects_add_up(self) -> None:
        # Each image beats 32 of 39 controls (p = 8/40 alone, so no image "exceeds"
        # at 0.05 and a majority vote says nothing). Together they are significant.
        self.assertLess(rank_sum_p_value([32] * 5, [K] * 5), 0.01)
        self.assertGreater(rank_sum_p_value([19, 20, 18, 21], [K] * 4), 0.3)

    def test_rejects_invalid_ranks(self) -> None:
        with self.assertRaises(ValueError):
            rank_sum_p_value([40], [K])
        with self.assertRaises(ValueError):
            rank_sum_p_value([], [])

    def test_benjamini_hochberg_matches_worked_example_and_skips_none(self) -> None:
        adjusted = benjamini_hochberg([0.01, None, 0.04, 0.03, 0.005])
        self.assertIsNone(adjusted[1])
        np.testing.assert_allclose(
            [adjusted[0], adjusted[2], adjusted[3], adjusted[4]],
            [0.02, 0.04, 0.04, 0.02],
        )

    def test_auroc_handles_separation_and_ties(self) -> None:
        self.assertEqual(auroc(np.array([0.9, 0.8]), np.array([0.1, 0.2, 0.3])), 1.0)
        self.assertEqual(auroc(np.array([0.5, 0.5]), np.array([0.5])), 0.5)
        self.assertEqual(auroc(np.array([0.1]), np.array([0.9])), 0.0)


class VerdictTests(unittest.TestCase):
    def setUp(self) -> None:
        self._directory = tempfile.TemporaryDirectory()
        self.fixture = Fixture(Path(self._directory.name))

    def tearDown(self) -> None:
        self._directory.cleanup()

    def verdicts(self, **config):
        artifact = aggregate_verdicts(
            self.fixture.results_path,
            self.fixture.explanations_path,
            config=VerdictConfig(**config) if config else None,
        )
        return artifact, {value.hypothesis_id: value for value in artifact.hypotheses}

    def test_assigns_each_label(self) -> None:
        artifact, verdicts = self.verdicts()
        labels = {key: (value.label, value.reason) for key, value in verdicts.items()}
        self.assertEqual(
            labels,
            {
                "node-0000-h00": ("supported", None),
                "node-0000-h01": ("detected_not_explanatory", None),
                "node-0000-h02": ("not_supported", "no_effect"),
                "node-0001-h00": ("insufficient_evidence", "too_few_testable_members"),
                "node-0001-h01": ("not_supported", "low_coverage"),
                "node-0001-h02": ("untestable", "unknown_categories"),
            },
        )
        self.assertEqual(artifact.summary()["label_counts"]["supported"], 1)

    def test_tests_report_percentiles_and_corrected_q_values(self) -> None:
        _, verdicts = self.verdicts()
        dog = verdicts["node-0000-h00"]
        self.assertEqual(dog.tests["centroid"].image_count, 5)
        self.assertAlmostEqual(dog.tests["centroid"].mean_percentile, 1.0)
        self.assertAlmostEqual(dog.tests["centroid"].p_value, (1 / 40) ** 5)
        self.assertGreaterEqual(dog.tests["centroid"].q_value, dog.tests["centroid"].p_value)
        self.assertTrue(dog.prediction_confirmed)
        self.assertEqual(dog.confidence, 0.9)
        self.assertEqual((dog.visible_members, dog.testable_members), (5, 5))
        self.assertEqual(verdicts["node-0001-h00"].tests, {})

    def test_prediction_follows_the_predicted_effect(self) -> None:
        _, verdicts = self.verdicts()
        self.assertTrue(verdicts["node-0000-h01"].prediction_confirmed)
        self.assertFalse(verdicts["node-0000-h02"].prediction_confirmed)
        self.assertIsNone(verdicts["node-0001-h02"].prediction_confirmed)

    def test_lower_min_testable_lets_a_single_strong_image_count(self) -> None:
        _, verdicts = self.verdicts(min_testable=1)
        self.assertEqual(verdicts["node-0001-h00"].testable_members, 1)
        self.assertNotEqual(verdicts["node-0001-h00"].reason, "too_few_testable_members")

    def test_rejects_results_from_other_explanations(self) -> None:
        stale = dict(self.fixture.results)
        stale["inputs"] = {**stale["inputs"], "explanations_sha256": "f" * 64}
        path = self.fixture.write_results(stale, "stale")
        with self.assertRaisesRegex(ValueError, "different explanations"):
            aggregate_verdicts(path, self.fixture.explanations_path)

    def test_save_load_round_trip_and_tables(self) -> None:
        artifact, _ = self.verdicts()
        paths = artifact.save(self.fixture.root / "out")
        self.assertEqual(VerdictArtifact.load(paths["json"]), artifact)
        header = paths["hypotheses_csv"].read_text(encoding="utf-8").splitlines()[0]
        self.assertIn("centroid_q_value", header)
        image_rows = paths["images_csv"].read_text(encoding="utf-8").splitlines()
        self.assertEqual(len(image_rows), len(artifact.images) + 1)
        self.assertIn("prompt_percentile", image_rows[0])

    def test_baseline_accepts_prompts_that_separate_the_node(self) -> None:
        fixture = self.fixture
        baseline = score_prompt_baseline(
            fixture.graph,
            fixture.images,
            fixture.explanations,
            fixture.prompts,
            graph_path=fixture.graph_path,
            explanations_path=fixture.explanations_path,
        )
        first = baseline.scores[0]
        self.assertTrue(first.accepted)
        self.assertEqual(first.auroc, 1.0)
        self.assertEqual(baseline.summary(), {"hypothesis_count": 6, "accepted_count": 6})

    def test_cli_writes_verdict_and_baseline_outputs(self) -> None:
        fixture = self.fixture
        images_path = fixture.images.save(fixture.root / "images.npz")
        prompts_path = fixture.prompts.save(fixture.root / "prompts.npz")
        artifacts = fixture.root / "artifacts"
        output = io.StringIO()
        with contextlib.redirect_stdout(output):
            self.assertEqual(
                main(
                    [
                        "verification",
                        "verdicts",
                        "--results",
                        str(fixture.results_path),
                        "--explanations",
                        str(fixture.explanations_path),
                        "--artifacts-root",
                        str(artifacts),
                    ]
                ),
                0,
            )
            self.assertEqual(
                main(
                    [
                        "verification",
                        "baseline",
                        "--graph",
                        str(fixture.graph_path),
                        "--embeddings",
                        str(images_path),
                        "--explanations",
                        str(fixture.explanations_path),
                        "--prompt-embeddings",
                        str(prompts_path),
                        "--artifacts-root",
                        str(artifacts),
                    ]
                ),
                0,
            )
        self.assertTrue((artifacts / "verification" / "run" / "verdicts.json").is_file())
        self.assertTrue(
            (artifacts / "verification" / "explanations-baseline" / "baseline.json").is_file()
        )
        first_summary = json.loads(output.getvalue().split("\n}\n")[0] + "\n}")
        self.assertEqual(first_summary["label_counts"]["supported"], 1)


if __name__ == "__main__":
    unittest.main()
