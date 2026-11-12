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
from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.service import embed_images
from clip_compass.embeddings.types import ClipModelSpec
from clip_compass.explanations.types import (
    ExplanationArtifact,
    ExplanationHypothesis,
    NodeExplanation,
)
from clip_compass.mapper.graph import MapperGraphArtifact, MapperNode
from clip_compass.perturbation.masks import (
    category_mask,
    clip_visible_box,
    control_masks,
    dilate,
    visible_pixels,
)
from clip_compass.perturbation.service import verify_hypotheses
from clip_compass.perturbation.verify import (
    PerturbationConfig,
    decide,
    load_results,
    verified_explanations,
)

HAS_PIL = importlib.util.find_spec("PIL") is not None
HAS_COCO = importlib.util.find_spec("pycocotools") is not None
HAS_TRANSFORMERS = importlib.util.find_spec("transformers") is not None
PROMPTS = {"a photo of an apple": [1.0, 0.0, 0.0]}


class PixelFakeEncoder:
    """Embeds an image as [10 * share of red pixels in the CLIP crop, 0, 1]."""

    def __init__(self, revision: str = "v1") -> None:
        self.calls = 0
        self._spec = ClipModelSpec("test", "pixel-fake", revision, "pixels-v1")
        self._text_spec = ClipModelSpec("test", "pixel-fake", revision, "prompts-v1")

    @property
    def spec(self) -> ClipModelSpec:
        return self._spec

    @property
    def text_spec(self) -> ClipModelSpec:
        return self._text_spec

    def encode_images(self, image_paths):
        from PIL import Image

        self.calls += 1
        rows = []
        for path in image_paths:
            with Image.open(path) as image:
                rgb = np.asarray(image.convert("RGB"), dtype=np.int32)
            left, top, right, bottom = clip_visible_box(rgb.shape[1], rgb.shape[0])
            crop = rgb[top:bottom, left:right]
            red = (crop[..., 0] > 200) & (crop[..., 1] < 80)
            rows.append([10.0 * red.mean(), 0.0, 1.0])
        return np.asarray(rows, dtype=np.float32)

    def encode_texts(self, texts):
        self.calls += 1
        return np.asarray([PROMPTS[text] for text in texts], dtype=np.float32)


def _square(identifier, image_id, x, y, size):
    return CocoInstance(
        id=identifier,
        image_id=image_id,
        category_id=1,
        bbox_xywh=(float(x), float(y), float(size), float(size)),
        area=float(size * size),
        iscrowd=False,
        segmentation=[[x, y, x + size, y, x + size, y + size, x, y + size]],
    )


def _hypothesis(name, categories, effect="decrease_text_similarity"):
    return ExplanationHypothesis(
        name=name,
        description="Synthetic.",
        positive_prompt="a photo of an apple",
        negative_prompt="a photo without an apple",
        coco_categories=tuple(categories),
        predicted_effect=effect,
        rationale="Synthetic.",
        supporting_item_ids=(),
        counterexample_item_ids=(),
        confidence=0.5,
    )


def _image(status="testable", centroid=False, prompt=False):
    return {
        "status": status,
        "p": {"centroid": 0.025 if centroid else 1.0, "prompt": 0.025 if prompt else 1.0},
    }


def _judge(images, categories=("apple",), unknown=(), effect="decrease_text_similarity"):
    return decide(categories, unknown, effect, len(images) or 2, images, PerturbationConfig())


class MaskTests(unittest.TestCase):
    def test_visible_box_follows_clip_resize_and_centre_crop(self) -> None:
        self.assertEqual(clip_visible_box(640, 480), (81, 0, 559, 480))
        self.assertEqual(clip_visible_box(424, 640), (0, 109, 424, 531))
        self.assertEqual(clip_visible_box(612, 612), (0, 0, 612, 612))

    @unittest.skipUnless(HAS_TRANSFORMERS and HAS_PIL, "transformers and Pillow are required")
    def test_visible_box_matches_clip_image_processor(self) -> None:
        from PIL import Image
        from transformers import CLIPImageProcessor

        processor = CLIPImageProcessor(do_normalize=False, do_rescale=False)
        for width, height in ((640, 480), (424, 640)):
            rgb = np.zeros((height, width, 3), dtype=np.uint8)
            rgb[..., 0] = np.round(np.arange(width) * 255.0 / (width - 1))[None, :]
            rgb[..., 1] = np.round(np.arange(height) * 255.0 / (height - 1))[:, None]
            pixels = processor(images=Image.fromarray(rgb), return_tensors="np")["pixel_values"]
            red, green = pixels[0, 0], pixels[0, 1]
            left, top, right, bottom = clip_visible_box(width, height, inset=0)
            edges = (red[:, 0], red[:, -1], green[0, :], green[-1, :])
            scales = (width, width, height, height)
            for edge, scale, expected in zip(edges, scales, (left, right - 1, top, bottom - 1)):
                self.assertAlmostEqual(edge.mean() * (scale - 1) / 255.0, expected, delta=4.0)

    def test_dilation_and_visible_pixels(self) -> None:
        point = np.zeros((9, 9), dtype=bool)
        point[4, 4] = True
        self.assertEqual(int(dilate(point, 2).sum()), 25)
        mask = np.zeros((480, 640), dtype=bool)
        mask[100:110, 70:90] = True  # straddles the crop's left edge at x=81
        self.assertEqual(visible_pixels(mask, clip_visible_box(640, 480)), 10 * 9)

    @unittest.skipUnless(HAS_COCO, "pycocotools optional dependency is not installed")
    def test_category_mask_unions_instances(self) -> None:
        subset = CocoInstanceSubset(
            split="val",
            source_sha256="1" * 64,
            manifest_sha256="2" * 64,
            item_ids=("1",),
            categories=(CocoCategory(1, "apple", "food"), CocoCategory(2, "cup", "kitchen")),
            images=(CocoImage("1", 20, 10, "1.jpg"),),
            annotations=(_square(1, "1", 1, 1, 3), _square(2, "1", 10, 4, 4)),
        )
        mask = category_mask(subset, "1", ("apple",), dilate_px=0)
        self.assertTrue(mask[2, 2] and mask[5, 11])
        self.assertFalse(mask[0, 19])
        self.assertIsNone(category_mask(subset, "1", ("cup",)))

    def test_controls_are_deterministic_area_matched_and_inside_the_crop(self) -> None:
        box = clip_visible_box(640, 480)
        mask = np.zeros((480, 640), dtype=bool)
        mask[200:240, 60:120] = True
        first = control_masks(mask, box, count=5, seed_key="a")
        extended = control_masks(mask, box, count=10, seed_key="a")
        other = control_masks(mask, box, count=5, seed_key="b")
        self.assertEqual(len(extended), 10)
        for control, again in zip(first, extended):
            np.testing.assert_array_equal(control, again)
        self.assertFalse(np.array_equal(first[0], other[0]))
        for control in extended:
            self.assertEqual(int(control.sum()), visible_pixels(mask, box))
            self.assertEqual(visible_pixels(control, box), int(control.sum()))
            self.assertLessEqual((control & mask).sum(), 0.25 * control.sum())

    def test_mask_filling_most_of_the_crop_has_no_control_room(self) -> None:
        box = clip_visible_box(640, 480)
        mask = np.zeros((480, 640), dtype=bool)
        mask[:, box[0] + 40 : box[2]] = True
        self.assertLess(len(control_masks(mask, box, count=10, seed_key="a")), 10)


class VerdictTests(unittest.TestCase):
    def test_centroid_majority_is_supported(self) -> None:
        verdict = _judge([_image(centroid=True, prompt=True), _image(centroid=True)])
        self.assertEqual(verdict["label"], "supported")
        self.assertTrue(verdict["verified"])
        self.assertFalse(verdict["prediction_confirmed"])  # prompt: 1 of 2 is no majority

    def test_prompt_only_is_detected_not_explanatory(self) -> None:
        images = [_image(prompt=True), _image(prompt=True)]
        verdict = _judge(images)
        self.assertEqual(verdict["label"], "detected_not_explanatory")
        self.assertTrue(verdict["prediction_confirmed"])
        self.assertFalse(_judge(images, effect="both")["prediction_confirmed"])

    def test_strict_majority_and_low_coverage(self) -> None:
        verdict = _judge([_image(centroid=True), _image()])
        self.assertEqual((verdict["label"], verdict["reason"]), ("not_supported", "no_exceedance"))
        low = _judge([_image(centroid=True), _image(centroid=True)] + [_image("absent")] * 3)
        self.assertEqual((low["label"], low["reason"]), ("not_supported", "low_coverage"))

    def test_untestable_reasons(self) -> None:
        cases = {
            "empty_categories": _judge([], categories=()),
            "unknown_categories": _judge([], unknown=("unicorn",)),
            "no_member_contains_categories": _judge([_image("absent"), _image("absent")]),
            "too_few_testable_members": _judge(
                [_image(centroid=True), _image("no_control_room")]
            ),
        }
        for reason, verdict in cases.items():
            self.assertEqual((verdict["label"], verdict["reason"]), ("untestable", reason))
            self.assertIsNone(verdict["prediction_confirmed"])


@unittest.skipUnless(HAS_PIL and HAS_COCO, "Pillow and pycocotools are required")
class VerifyHypothesesTests(unittest.TestCase):
    """Four 64x48 images; node-0000 = {1, 2} with red apples, node-0001 = {3, 4}."""

    APPLES = {"1": (12, 10, 14), "2": (30, 22, 14), "3": (24, 16, 14)}

    def setUp(self) -> None:
        from PIL import Image

        self.directory = tempfile.TemporaryDirectory()
        root = self.root = Path(self.directory.name)
        self.encoder = PixelFakeEncoder()
        self.cache = EmbeddingCache(root / "cache")
        item_ids = ("1", "2", "3", "4")
        rows, instances = [], []
        for item_id in item_ids:
            rgb = np.full((48, 64, 3), (40, 60, 160), dtype=np.uint8)
            if item_id in self.APPLES:
                x, y, size = self.APPLES[item_id]
                rgb[y : y + size, x : x + size] = (230, 20, 20)
                instances.append(_square(len(instances) + 1, item_id, x, y, size))
            (root / "images").mkdir(exist_ok=True)
            Image.fromarray(rgb).save(root / "images" / f"{item_id}.png")
            path = f"images/{item_id}.png"
            rows.append({"image_id": item_id, "image_path": path, "captions": []})
        self.manifest = root / "manifest.jsonl"
        self.manifest.write_text("".join(json.dumps(row) + "\n" for row in rows))
        self.annotations = CocoInstanceSubset(
            split="val",
            source_sha256="1" * 64,
            manifest_sha256="2" * 64,
            item_ids=item_ids,
            categories=(CocoCategory(1, "apple", "food"),),
            images=tuple(CocoImage(item_id, 64, 48, f"{item_id}.png") for item_id in item_ids),
            annotations=tuple(instances),
        ).save(root / "annotations.json")
        self.embeddings = embed_images(
            [root / "images" / f"{item_id}.png" for item_id in item_ids],
            encoder=self.encoder,
            cache=self.cache,
            item_ids=item_ids,
        ).save(root / "embeddings.npz")
        self.graph = self._graph(item_ids).save(root / "graph.json")
        self.explanations = ExplanationArtifact(
            graph_sha256=sha256_file(self.graph),
            annotations_sha256=sha256_file(self.annotations),
            embedding_model_fingerprint=self.encoder.spec.fingerprint,
            generator={"provider": "test"},
            context_config={},
            explanations=(
                NodeExplanation("node-0000", "Apples.", (_hypothesis("Apples", ("apple",)),)),
                NodeExplanation(
                    "node-0001",
                    "Mixed.",
                    (
                        _hypothesis("Apples again", ("apple",), "both"),
                        _hypothesis("Nothing named", ()),
                        _hypothesis("Unicorns", ("unicorn",)),
                    ),
                ),
            ),
        ).save(root / "explanations.json")

    def tearDown(self) -> None:
        self.directory.cleanup()

    def _graph(self, item_ids):
        spec = self.encoder.spec
        nodes = tuple(
            MapperNode(
                id=f"node-{index:04d}",
                source_id=f"cube{index}_cluster0",
                component_id=f"component-{index:04d}",
                member_indices=members,
                item_ids=tuple(item_ids[member] for member in members),
                lens_min=float(index),
                lens_max=index + 0.5,
                lens_mean=index + 0.25,
            )
            for index, members in enumerate(((0, 1), (2, 3)))
        )
        return MapperGraphArtifact(
            item_ids=item_ids,
            embedding_model=spec.as_dict(),
            embedding_model_fingerprint=spec.fingerprint,
            feature_space="normalized",
            lens={"name": "synthetic"},
            config={},
            eps_calibration={"eps": 0.2, "method": "configured"},
            nodes=nodes,
            edges=(),
            components=(("node-0000",), ("node-0001",)),
            diagnostics={
                "node_count": 2,
                "edge_count": 0,
                "component_count": 2,
                "isolated_node_count": 2,
                "assigned_item_count": 4,
                "unassigned_item_count": 0,
                "coverage_fraction": 1.0,
                "multi_membership_item_count": 0,
                "total_memberships": 4,
                "node_size_min": 2,
                "node_size_median": 2.0,
                "node_size_max": 2,
            },
        )

    def _run(self, encoder=None):
        return verify_hypotheses(
            graph_path=self.graph,
            embeddings_path=self.embeddings,
            annotations_path=self.annotations,
            explanations_path=self.explanations,
            manifest_path=self.manifest,
            data_root=self.root,
            encoder=encoder or self.encoder,
            cache=self.cache,
            output_dir=self.root / "run",
            config=PerturbationConfig(controls=19, dilate_px=0),
        )

    def test_verdicts_determinism_and_verified_export(self) -> None:
        results = self._run()
        path = self.root / "run" / "verification.json"
        first = path.read_bytes()
        verdicts = {h["hypothesis_id"]: h["verdict"] for h in results["hypotheses"]}
        self.assertEqual(verdicts["node-0000-h00"]["label"], "supported")
        self.assertTrue(verdicts["node-0000-h00"]["prediction_confirmed"])
        self.assertEqual(verdicts["node-0001-h00"]["reason"], "too_few_testable_members")
        self.assertEqual(verdicts["node-0001-h01"]["reason"], "empty_categories")
        self.assertEqual(verdicts["node-0001-h02"]["reason"], "unknown_categories")
        self.assertAlmostEqual(results["min_identity_cosine"], 1.0, places=6)
        image = results["hypotheses"][0]["images"][0]
        self.assertEqual(image["p"]["prompt"], 1 / 20)
        self.assertEqual(len(image["controls"]["centroid"]), 19)
        self.assertTrue(
            (self.root / "run/images/1/1__node-0000-h00__apple__mean-d0__object.png").is_file()
        )
        self.assertEqual(load_results(path)["hypotheses"], json.loads(first)["hypotheses"])

        calls = self.encoder.calls
        self._run()
        self.assertEqual(self.encoder.calls, calls)  # every image and prompt was cached
        self.assertEqual(path.read_bytes(), first)
        self.assertEqual(list(self.root.rglob("*.part")), [])

        verified = verified_explanations(results, self.explanations)
        self.assertEqual([node.node_id for node in verified.explanations], ["node-0000"])
        self.assertEqual(verified.generator["verification"]["kept"], ["node-0000-h00"])

    def test_refuses_mismatched_inputs(self) -> None:
        with self.assertRaisesRegex(ValueError, "different models"):
            self._run(encoder=PixelFakeEncoder(revision="v2"))
        data = json.loads(self.explanations.read_text())
        data["graph_sha256"] = "f" * 64
        self.explanations.write_text(json.dumps(data))
        with self.assertRaisesRegex(ValueError, "different Mapper graph"):
            self._run()


if __name__ == "__main__":
    unittest.main()
