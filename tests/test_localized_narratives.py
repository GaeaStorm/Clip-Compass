import json
import tempfile
import unittest
from pathlib import Path

from clip_compass.data.localized_narratives import (
    DatasetFormatError,
    coco_image_filename,
    select_caption_subset,
    validate_jpeg,
)


class LocalizedNarrativesTests(unittest.TestCase):
    def test_selects_unique_images_and_retains_all_captions(self) -> None:
        rows = [
            {
                "dataset_id": "mscoco_val2017",
                "image_id": "42",
                "annotator_id": 1,
                "caption": "A first description.",
            },
            {
                "dataset_id": "mscoco_val2017",
                "image_id": "99",
                "annotator_id": 2,
                "caption": "This image should be outside the limit.",
            },
            {
                "dataset_id": "mscoco_val2017",
                "image_id": "42",
                "annotator_id": 3,
                "caption": "A second description of the selected image.",
            },
        ]
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            annotation_path = root / "captions.jsonl"
            annotation_path.write_text(
                "".join(json.dumps(row) + "\n" for row in rows), encoding="utf-8"
            )

            records = select_caption_subset(
                annotation_path,
                split="val",
                limit=1,
                image_root=root / "images",
            )

        self.assertEqual(len(records), 1)
        self.assertEqual(records[0].image_id, "42")
        self.assertEqual(records[0].annotator_ids, (1, 3))
        self.assertEqual(len(records[0].captions), 2)
        self.assertEqual(records[0].image_path.name, "000000000042.jpg")

    def test_coco_filename_rejects_non_numeric_ids(self) -> None:
        with self.assertRaises(DatasetFormatError):
            coco_image_filename("not-an-id")

    def test_rejects_non_positive_limit(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            with self.assertRaises(ValueError):
                select_caption_subset(
                    Path(temporary_directory) / "unused.jsonl",
                    split="val",
                    limit=0,
                    image_root=Path(temporary_directory),
                )

    def test_validates_jpeg_markers(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            image_path = Path(temporary_directory) / "image.jpg"
            image_path.write_bytes(b"\xff\xd8image payload\xff\xd9")
            validate_jpeg(image_path)

            image_path.write_bytes(b"not a jpeg")
            with self.assertRaises(DatasetFormatError):
                validate_jpeg(image_path)


if __name__ == "__main__":
    unittest.main()
