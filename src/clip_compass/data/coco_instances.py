"""COCO instance annotations aligned to a prepared image manifest."""

from __future__ import annotations

import hashlib
import json
import os
import shutil
import urllib.error
import urllib.request
import zipfile
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Optional, Tuple

from clip_compass.config import DataPaths


COCO_INSTANCE_SCHEMA_VERSION = 1
COCO_ANNOTATIONS_URL = (
    "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"
)


class CocoAnnotationFormatError(ValueError):
    """Raised when an official or subset COCO annotation file is malformed."""


@dataclass(frozen=True)
class CocoCategory:
    id: int
    name: str
    supercategory: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "name": self.name,
            "supercategory": self.supercategory,
        }


@dataclass(frozen=True)
class CocoImage:
    id: str
    width: int
    height: int
    file_name: str

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "width": self.width,
            "height": self.height,
            "file_name": self.file_name,
        }


@dataclass(frozen=True)
class CocoInstance:
    id: int
    image_id: str
    category_id: int
    bbox_xywh: Tuple[float, float, float, float]
    area: float
    iscrowd: bool
    segmentation: object

    def as_dict(self) -> Dict[str, object]:
        return {
            "id": self.id,
            "image_id": self.image_id,
            "category_id": self.category_id,
            "bbox": list(self.bbox_xywh),
            "area": self.area,
            "iscrowd": int(self.iscrowd),
            "segmentation": self.segmentation,
        }


@dataclass(frozen=True)
class CocoInstanceSubset:
    split: str
    source_sha256: str
    manifest_sha256: str
    item_ids: Tuple[str, ...]
    categories: Tuple[CocoCategory, ...]
    images: Tuple[CocoImage, ...]
    annotations: Tuple[CocoInstance, ...]

    def __post_init__(self) -> None:
        if self.split not in {"train", "val"}:
            raise ValueError("COCO split must be 'train' or 'val'")
        if len(set(self.item_ids)) != len(self.item_ids):
            raise ValueError("COCO subset item IDs must be unique")
        if tuple(image.id for image in self.images) != self.item_ids:
            raise ValueError("COCO images must align with item IDs in manifest order")
        category_ids = [category.id for category in self.categories]
        if category_ids != sorted(set(category_ids)):
            raise ValueError("COCO categories must be unique and sorted by ID")
        annotation_ids = [annotation.id for annotation in self.annotations]
        if annotation_ids != sorted(set(annotation_ids)):
            raise ValueError("COCO annotations must be unique and sorted by ID")
        item_id_set = set(self.item_ids)
        category_id_set = set(category_ids)
        for annotation in self.annotations:
            if annotation.image_id not in item_id_set:
                raise ValueError("COCO annotation refers to an image outside the subset")
            if annotation.category_id not in category_id_set:
                raise ValueError("COCO annotation refers to an unknown category")
            x, y, width, height = annotation.bbox_xywh
            if min(x, y, width, height, annotation.area) < 0.0:
                raise ValueError("COCO annotation geometry must be non-negative")
            if width == 0.0 or height == 0.0:
                raise ValueError("COCO annotation bounding boxes must have positive size")

    @property
    def categories_by_id(self) -> Mapping[int, CocoCategory]:
        return {category.id: category for category in self.categories}

    def annotations_for(self, item_id: str) -> Tuple[CocoInstance, ...]:
        if item_id not in set(self.item_ids):
            raise KeyError(item_id)
        return tuple(
            annotation for annotation in self.annotations if annotation.image_id == item_id
        )

    def category_names_for(self, item_id: str) -> Tuple[str, ...]:
        categories = self.categories_by_id
        return tuple(
            sorted(
                {
                    categories[annotation.category_id].name
                    for annotation in self.annotations_for(item_id)
                }
            )
        )

    def summary(self) -> Dict[str, object]:
        annotated_ids = {annotation.image_id for annotation in self.annotations}
        used_categories = {annotation.category_id for annotation in self.annotations}
        return {
            "schema_version": COCO_INSTANCE_SCHEMA_VERSION,
            "split": self.split,
            "image_count": len(self.images),
            "annotated_image_count": len(annotated_ids),
            "unannotated_image_count": len(self.images) - len(annotated_ids),
            "annotation_count": len(self.annotations),
            "category_count": len(self.categories),
            "used_category_count": len(used_categories),
            "crowd_annotation_count": sum(
                annotation.iscrowd for annotation in self.annotations
            ),
        }

    def as_dict(self) -> Dict[str, object]:
        return {
            "schema_version": COCO_INSTANCE_SCHEMA_VERSION,
            "split": self.split,
            "source_sha256": self.source_sha256,
            "manifest_sha256": self.manifest_sha256,
            "item_ids": list(self.item_ids),
            "categories": [category.as_dict() for category in self.categories],
            "images": [image.as_dict() for image in self.images],
            "annotations": [annotation.as_dict() for annotation in self.annotations],
            "summary": self.summary(),
        }

    def save(self, path: Path) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        partial = path.with_name(f"{path.name}.part")
        partial.write_text(
            json.dumps(self.as_dict(), indent=2, sort_keys=True) + "\n",
            encoding="utf-8",
        )
        os.replace(partial, path)
        return path

    @classmethod
    def load(cls, path: Path) -> "CocoInstanceSubset":
        data = json.loads(path.read_text(encoding="utf-8"))
        if data.get("schema_version") != COCO_INSTANCE_SCHEMA_VERSION:
            raise ValueError("unsupported COCO instance subset schema version")
        return cls(
            split=str(data["split"]),
            source_sha256=str(data["source_sha256"]),
            manifest_sha256=str(data["manifest_sha256"]),
            item_ids=tuple(str(value) for value in data["item_ids"]),
            categories=tuple(_category_from_mapping(row) for row in data["categories"]),
            images=tuple(_image_from_mapping(row) for row in data["images"]),
            annotations=tuple(
                _instance_from_mapping(row) for row in data["annotations"]
            ),
        )


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as source:
        for chunk in iter(lambda: source.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _category_from_mapping(row: Mapping[str, object]) -> CocoCategory:
    try:
        category_id = int(row["id"])
        name = str(row["name"]).strip()
        supercategory = str(row.get("supercategory", "")).strip()
    except (KeyError, TypeError, ValueError) as error:
        raise CocoAnnotationFormatError("invalid COCO category") from error
    if category_id < 1 or not name:
        raise CocoAnnotationFormatError("invalid COCO category values")
    return CocoCategory(category_id, name, supercategory)


def _image_from_mapping(row: Mapping[str, object]) -> CocoImage:
    try:
        image_id = str(int(row["id"]))
        width = int(row["width"])
        height = int(row["height"])
        file_name = str(row["file_name"])
    except (KeyError, TypeError, ValueError) as error:
        raise CocoAnnotationFormatError("invalid COCO image") from error
    if width < 1 or height < 1 or not file_name:
        raise CocoAnnotationFormatError("invalid COCO image values")
    return CocoImage(image_id, width, height, file_name)


def _instance_from_mapping(row: Mapping[str, object]) -> CocoInstance:
    try:
        raw_bbox = row["bbox"]
        if not isinstance(raw_bbox, list) or len(raw_bbox) != 4:
            raise TypeError("bbox must contain four values")
        bbox = tuple(float(value) for value in raw_bbox)
        segmentation = row["segmentation"]
        if not isinstance(segmentation, (list, dict)):
            raise TypeError("segmentation must be a polygon list or RLE object")
        return CocoInstance(
            id=int(row["id"]),
            image_id=str(int(row["image_id"])),
            category_id=int(row["category_id"]),
            bbox_xywh=(bbox[0], bbox[1], bbox[2], bbox[3]),
            area=float(row["area"]),
            iscrowd=bool(int(row.get("iscrowd", 0))),
            segmentation=segmentation,
        )
    except (KeyError, TypeError, ValueError) as error:
        raise CocoAnnotationFormatError("invalid COCO instance annotation") from error


def manifest_item_ids(manifest_path: Path) -> Tuple[str, ...]:
    item_ids: List[str] = []
    seen = set()
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                item_id = str(int(row["image_id"]))
            except (json.JSONDecodeError, KeyError, TypeError, ValueError) as error:
                raise CocoAnnotationFormatError(
                    f"invalid manifest row at {manifest_path}:{line_number}"
                ) from error
            if item_id in seen:
                raise CocoAnnotationFormatError(f"duplicate manifest image ID: {item_id}")
            item_ids.append(item_id)
            seen.add(item_id)
    if not item_ids:
        raise CocoAnnotationFormatError(f"no image IDs found in {manifest_path}")
    return tuple(item_ids)


def build_coco_instance_subset(
    source_path: Path,
    manifest_path: Path,
    *,
    split: str,
) -> CocoInstanceSubset:
    """Filter an official instances JSON file to manifest images in stable order."""

    if split not in {"train", "val"}:
        raise ValueError("COCO split must be 'train' or 'val'")
    data = json.loads(source_path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise CocoAnnotationFormatError("COCO annotation root must be an object")
    item_ids = manifest_item_ids(manifest_path)
    selected = set(item_ids)
    try:
        categories = tuple(
            sorted(
                (_category_from_mapping(row) for row in data["categories"]),
                key=lambda category: category.id,
            )
        )
        image_by_id = {
            image.id: image
            for image in (_image_from_mapping(row) for row in data["images"])
            if image.id in selected
        }
        annotations = tuple(
            sorted(
                (
                    annotation
                    for annotation in (
                        _instance_from_mapping(row) for row in data["annotations"]
                    )
                    if annotation.image_id in selected
                ),
                key=lambda annotation: annotation.id,
            )
        )
    except KeyError as error:
        raise CocoAnnotationFormatError(f"COCO file is missing {error.args[0]!r}") from error
    missing = [item_id for item_id in item_ids if item_id not in image_by_id]
    if missing:
        raise CocoAnnotationFormatError(
            f"COCO instances file is missing {len(missing)} manifest images; first: {missing[0]}"
        )
    return CocoInstanceSubset(
        split=split,
        source_sha256=sha256_file(source_path),
        manifest_sha256=sha256_file(manifest_path),
        item_ids=item_ids,
        categories=categories,
        images=tuple(image_by_id[item_id] for item_id in item_ids),
        annotations=annotations,
    )


def _download_annotations(destination: Path, *, force: bool) -> Path:
    if destination.exists() and not force:
        return destination
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    request = urllib.request.Request(
        COCO_ANNOTATIONS_URL, headers={"User-Agent": "clip-compass/0.1"}
    )
    try:
        with urllib.request.urlopen(request, timeout=120) as response, partial.open(
            "wb"
        ) as output:
            shutil.copyfileobj(response, output, length=1024 * 1024)
        os.replace(partial, destination)
    except (OSError, urllib.error.URLError):
        partial.unlink(missing_ok=True)
        raise
    return destination


def _extract_instances(
    archive_path: Path, destination: Path, *, split: str, force: bool = False
) -> Path:
    if destination.exists() and not force:
        return destination
    member = f"annotations/instances_{split}2017.json"
    destination.parent.mkdir(parents=True, exist_ok=True)
    partial = destination.with_name(f"{destination.name}.part")
    try:
        with zipfile.ZipFile(archive_path) as archive, archive.open(member) as source:
            with partial.open("wb") as output:
                shutil.copyfileobj(source, output, length=1024 * 1024)
        os.replace(partial, destination)
    except (KeyError, OSError, zipfile.BadZipFile):
        partial.unlink(missing_ok=True)
        raise
    return destination


def prepare_coco_instance_subset(
    *,
    data_root: Path,
    manifest_path: Path,
    split: str = "val",
    output_path: Optional[Path] = None,
    force_download: bool = False,
) -> Tuple[CocoInstanceSubset, Path]:
    """Download official annotations when needed and save a manifest-aligned subset."""

    if split not in {"train", "val"}:
        raise ValueError("COCO split must be 'train' or 'val'")
    paths = DataPaths.from_root(data_root)
    archive_path = paths.coco_annotations / "annotations_trainval2017.zip"
    source_path = paths.coco_annotations / f"instances_{split}2017.json"
    if force_download or not source_path.exists():
        _download_annotations(archive_path, force=force_download)
        _extract_instances(
            archive_path,
            source_path,
            split=split,
            force=force_download,
        )
    subset = build_coco_instance_subset(source_path, manifest_path, split=split)
    output = output_path or (
        paths.instance_manifests / f"{manifest_path.stem}-instances.json"
    )
    subset.save(output)
    return subset, output
