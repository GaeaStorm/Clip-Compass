"""Load a small COCO subset described by Localized Narratives.

The initial pipeline intentionally uses the caption-only Localized Narratives
files. They contain the text needed by Clip Compass without the much larger
timed mouse traces and voice-recording metadata. COCO images are fetched only
for the selected image IDs.
"""

from __future__ import annotations

import json
import os
import shutil
import urllib.error
import urllib.request
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterator, List, Mapping, Sequence, Tuple

from clip_compass.config import DataPaths


LOCALIZED_NARRATIVES_ANNOTATION_ROOT = (
    "https://storage.googleapis.com/localized-narratives/annotations"
)
# COCO publishes its image links over HTTP. The HTTPS endpoint currently
# presents a certificate whose hostname does not match images.cocodataset.org.
# Keep certificate verification enabled everywhere else instead of bypassing it.
COCO_IMAGE_ROOT = "http://images.cocodataset.org"

_SPLITS: Mapping[str, Mapping[str, str]] = {
    "train": {
        "dataset_id": "mscoco_train2017",
        "annotation_filename": "coco_train_captions.jsonl",
        "image_directory": "train2017",
    },
    "val": {
        "dataset_id": "mscoco_val2017",
        "annotation_filename": "coco_val_captions.jsonl",
        "image_directory": "val2017",
    },
}


class DatasetFormatError(ValueError):
    """Raised when a downloaded JSONL row does not match the expected schema."""


@dataclass(frozen=True)
class LocalizedNarrativeCaption:
    """Caption-only Localized Narratives annotation."""

    dataset_id: str
    image_id: str
    annotator_id: int
    caption: str

    @classmethod
    def from_mapping(cls, value: Mapping[str, object]) -> "LocalizedNarrativeCaption":
        required = ("dataset_id", "image_id", "annotator_id", "caption")
        missing = [key for key in required if key not in value]
        if missing:
            raise DatasetFormatError(f"Localized Narratives row is missing fields: {missing}")

        dataset_id = value["dataset_id"]
        image_id = value["image_id"]
        annotator_id = value["annotator_id"]
        caption = value["caption"]
        if not isinstance(dataset_id, str) or not isinstance(image_id, str):
            raise DatasetFormatError("dataset_id and image_id must be strings")
        if not isinstance(annotator_id, int) or not isinstance(caption, str):
            raise DatasetFormatError("annotator_id must be an integer and caption must be a string")
        if not caption.strip():
            raise DatasetFormatError("caption must not be empty")

        return cls(
            dataset_id=dataset_id,
            image_id=image_id,
            annotator_id=annotator_id,
            caption=caption.strip(),
        )


@dataclass(frozen=True)
class CocoSubsetRecord:
    """One COCO image and all Localized Narrative captions found for it."""

    dataset_id: str
    image_id: str
    image_path: Path
    captions: Tuple[str, ...]
    annotator_ids: Tuple[int, ...]

    def as_manifest_row(self, data_root: Path) -> Dict[str, object]:
        try:
            image_path = self.image_path.relative_to(data_root)
        except ValueError:
            image_path = self.image_path
        return {
            "dataset_id": self.dataset_id,
            "image_id": self.image_id,
            "image_path": image_path.as_posix(),
            "captions": list(self.captions),
            "annotator_ids": list(self.annotator_ids),
        }


@dataclass(frozen=True)
class PreparedSubset:
    """Locations and counts produced by :func:`prepare_coco_subset`."""

    split: str
    annotation_path: Path
    manifest_path: Path
    records: Tuple[CocoSubsetRecord, ...]


def _split_config(split: str) -> Mapping[str, str]:
    try:
        return _SPLITS[split]
    except KeyError as error:
        choices = ", ".join(sorted(_SPLITS))
        raise ValueError(f"Unknown COCO split {split!r}; expected one of: {choices}") from error


def _download_file(url: str, destination: Path, *, force: bool = False) -> Path:
    """Download to a temporary file and atomically publish on success."""

    if destination.exists() and not force:
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(url, headers={"User-Agent": "clip-compass/0.1"})
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open("wb") as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(partial, destination)
    except (OSError, urllib.error.URLError):
        partial.unlink(missing_ok=True)
        raise
    return destination


def annotation_url(split: str) -> str:
    config = _split_config(split)
    return f"{LOCALIZED_NARRATIVES_ANNOTATION_ROOT}/{config['annotation_filename']}"


def coco_image_filename(image_id: str) -> str:
    if not image_id.isdigit():
        raise DatasetFormatError(f"COCO image_id must contain only digits: {image_id!r}")
    return f"{int(image_id):012d}.jpg"


def coco_image_url(image_id: str, split: str) -> str:
    image_directory = _split_config(split)["image_directory"]
    return f"{COCO_IMAGE_ROOT}/{image_directory}/{coco_image_filename(image_id)}"


def validate_jpeg(path: Path) -> None:
    """Perform a dependency-free integrity check for a downloaded JPEG."""

    if path.stat().st_size < 4:
        raise DatasetFormatError(f"Downloaded image is too small to be a JPEG: {path}")
    with path.open("rb") as image:
        start = image.read(2)
        image.seek(-2, os.SEEK_END)
        end = image.read(2)
    if start != b"\xff\xd8" or end != b"\xff\xd9":
        raise DatasetFormatError(f"Downloaded image does not have valid JPEG markers: {path}")


def iter_caption_annotations(path: Path) -> Iterator[LocalizedNarrativeCaption]:
    """Stream and validate caption-only Localized Narratives JSONL."""

    with path.open("r", encoding="utf-8") as annotations:
        for line_number, line in enumerate(annotations, start=1):
            if not line.strip():
                continue
            try:
                raw = json.loads(line)
            except json.JSONDecodeError as error:
                raise DatasetFormatError(f"Invalid JSON at {path}:{line_number}") from error
            if not isinstance(raw, dict):
                raise DatasetFormatError(f"Expected an object at {path}:{line_number}")
            try:
                yield LocalizedNarrativeCaption.from_mapping(raw)
            except DatasetFormatError as error:
                raise DatasetFormatError(f"{path}:{line_number}: {error}") from error


def select_caption_subset(
    path: Path,
    *,
    split: str,
    limit: int,
    image_root: Path,
) -> Tuple[CocoSubsetRecord, ...]:
    """Select the first ``limit`` unique image IDs in deterministic file order.

    The entire annotation file is scanned after choosing the IDs so that every
    available narrative for each selected image is retained.
    """

    if limit < 1:
        raise ValueError("limit must be at least 1")

    config = _split_config(split)
    expected_dataset_id = config["dataset_id"]
    selected_ids: List[str] = []
    captions: Dict[str, List[str]] = {}
    annotators: Dict[str, List[int]] = {}

    for annotation in iter_caption_annotations(path):
        if annotation.dataset_id != expected_dataset_id:
            continue
        if annotation.image_id not in captions:
            if len(selected_ids) >= limit:
                continue
            selected_ids.append(annotation.image_id)
            captions[annotation.image_id] = []
            annotators[annotation.image_id] = []
        captions[annotation.image_id].append(annotation.caption)
        annotators[annotation.image_id].append(annotation.annotator_id)

    if not selected_ids:
        raise DatasetFormatError(
            f"No rows with dataset_id={expected_dataset_id!r} were found in {path}"
        )

    image_directory = image_root / config["image_directory"]
    return tuple(
        CocoSubsetRecord(
            dataset_id=expected_dataset_id,
            image_id=image_id,
            image_path=image_directory / coco_image_filename(image_id),
            captions=tuple(captions[image_id]),
            annotator_ids=tuple(annotators[image_id]),
        )
        for image_id in selected_ids
    )


def _write_manifest(records: Sequence[CocoSubsetRecord], path: Path, data_root: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    partial = path.with_name(f"{path.name}.part")
    with partial.open("w", encoding="utf-8") as manifest:
        for record in records:
            json.dump(record.as_manifest_row(data_root), manifest, sort_keys=True)
            manifest.write("\n")
    os.replace(partial, path)


def prepare_coco_subset(
    *,
    data_root: Path,
    split: str = "val",
    limit: int = 25,
    download_images: bool = True,
    force: bool = False,
) -> PreparedSubset:
    """Download annotations and prepare a deterministic COCO image subset."""

    paths = DataPaths.from_root(data_root)
    config = _split_config(split)
    annotation_path = paths.localized_narratives / config["annotation_filename"]
    _download_file(annotation_url(split), annotation_path, force=force)

    records = select_caption_subset(
        annotation_path,
        split=split,
        limit=limit,
        image_root=paths.coco_images,
    )
    if download_images:
        for record in records:
            _download_file(coco_image_url(record.image_id, split), record.image_path, force=force)
            validate_jpeg(record.image_path)

    manifest_path = paths.manifests / f"coco_{split}_first_{limit}.jsonl"
    _write_manifest(records, manifest_path, paths.root)
    return PreparedSubset(
        split=split,
        annotation_path=annotation_path,
        manifest_path=manifest_path,
        records=records,
    )
