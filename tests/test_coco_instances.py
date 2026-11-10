import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from clip_compass.data.coco_instances import (
    CocoImage,
    CocoInstance,
    CocoInstanceSubset,
    build_coco_instance_subset,
)
from clip_compass.data.coco_masks import instance_mask


class CocoInstanceTests(unittest.TestCase):
    def test_filters_annotations_in_manifest_order_and_round_trips(self) -> None:
        source = {
            "images": [
                {"id": 2, "width": 20, "height": 10, "file_name": "2.jpg"},
                {"id": 1, "width": 12, "height": 8, "file_name": "1.jpg"},
                {"id": 3, "width": 30, "height": 20, "file_name": "3.jpg"},
            ],
            "categories": [
                {"id": 2, "name": "car", "supercategory": "vehicle"},
                {"id": 1, "name": "person", "supercategory": "person"},
            ],
            "annotations": [
                {
                    "id": 20,
                    "image_id": 2,
                    "category_id": 2,
                    "bbox": [1, 1, 4, 3],
                    "area": 12,
                    "iscrowd": 0,
                    "segmentation": [[1, 1, 5, 1, 5, 4, 1, 4]],
                },
                {
                    "id": 10,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [0, 0, 2, 3],
                    "area": 6,
                    "iscrowd": 0,
                    "segmentation": [[0, 0, 2, 0, 2, 3, 0, 3]],
                },
                {
                    "id": 30,
                    "image_id": 3,
                    "category_id": 1,
                    "bbox": [0, 0, 1, 1],
                    "area": 1,
                    "iscrowd": 0,
                    "segmentation": [[0, 0, 1, 0, 1, 1]],
                },
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "instances_val2017.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                json.dumps({"image_id": "2"})
                + "\n"
                + json.dumps({"image_id": "1"})
                + "\n",
                encoding="utf-8",
            )
            subset = build_coco_instance_subset(
                source_path, manifest_path, split="val"
            )
            artifact_path = root / "subset.json"
            subset.save(artifact_path)
            loaded = CocoInstanceSubset.load(artifact_path)

        self.assertEqual(subset.item_ids, ("2", "1"))
        self.assertEqual(tuple(annotation.id for annotation in subset.annotations), (10, 20))
        self.assertEqual(subset.category_names_for("2"), ("car",))
        self.assertEqual(loaded.as_dict(), subset.as_dict())
        self.assertEqual(subset.summary()["annotation_count"], 2)

    def test_rasterizes_polygon_mask_at_image_dimensions(self) -> None:
        source = {
            "images": [{"id": 1, "width": 12, "height": 8, "file_name": "1.jpg"}],
            "categories": [{"id": 1, "name": "object", "supercategory": "thing"}],
            "annotations": [
                {
                    "id": 1,
                    "image_id": 1,
                    "category_id": 1,
                    "bbox": [2, 2, 5, 4],
                    "area": 20,
                    "iscrowd": 0,
                    "segmentation": [[2, 2, 7, 2, 7, 6, 2, 6]],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            source_path = root / "instances.json"
            source_path.write_text(json.dumps(source), encoding="utf-8")
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(json.dumps({"image_id": 1}) + "\n", encoding="utf-8")
            subset = build_coco_instance_subset(
                source_path, manifest_path, split="val"
            )
        mask = instance_mask(subset.annotations[0], subset.images[0])
        self.assertEqual(mask.shape, (8, 12))
        self.assertEqual(mask.dtype, np.bool_)
        self.assertTrue(mask[3, 3])
        self.assertFalse(mask[0, 0])

    @unittest.skipUnless(
        importlib.util.find_spec("pycocotools"),
        "pycocotools optional dependency is not installed",
    )
    def test_decodes_compressed_rle_mask(self) -> None:
        from pycocotools import mask as mask_utils

        expected = np.zeros((6, 8), dtype=np.uint8)
        expected[1:4, 2:6] = 1
        rle = mask_utils.encode(np.asfortranarray(expected))
        segmentation = {
            "size": [int(value) for value in rle["size"]],
            "counts": rle["counts"].decode("ascii"),
        }
        annotation = CocoInstance(
            id=1,
            image_id="1",
            category_id=1,
            bbox_xywh=(2.0, 1.0, 4.0, 3.0),
            area=12.0,
            iscrowd=True,
            segmentation=segmentation,
        )
        image = CocoImage(id="1", width=8, height=6, file_name="1.jpg")
        np.testing.assert_array_equal(instance_mask(annotation, image), expected.astype(bool))

    @unittest.skipUnless(
        importlib.util.find_spec("pycocotools"),
        "pycocotools optional dependency is not installed",
    )
    def test_decodes_uncompressed_rle_mask(self) -> None:
        annotation = CocoInstance(
            id=1,
            image_id="1",
            category_id=1,
            bbox_xywh=(0.0, 1.0, 1.0, 2.0),
            area=2.0,
            iscrowd=True,
            segmentation={"size": [3, 2], "counts": [1, 2, 3]},
        )
        image = CocoImage(id="1", width=2, height=3, file_name="1.jpg")
        expected = np.asarray(
            [[False, False], [True, False], [True, False]], dtype=np.bool_
        )
        np.testing.assert_array_equal(instance_mask(annotation, image), expected)


if __name__ == "__main__":
    unittest.main()
