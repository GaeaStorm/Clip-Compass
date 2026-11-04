"""Rasterize COCO polygon and RLE instance segmentations."""

from __future__ import annotations

from typing import Mapping, Sequence

import numpy as np

from .coco_instances import CocoImage, CocoInstance


def _polygon_mask(
    polygons: Sequence[object], *, width: int, height: int
) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError(
            "COCO masks require pycocotools: pip install -e '.[coco]'"
        ) from error
    validated = []
    for raw_polygon in polygons:
        if not isinstance(raw_polygon, list) or len(raw_polygon) < 6:
            raise ValueError("COCO polygons must contain at least three coordinate pairs")
        if len(raw_polygon) % 2:
            raise ValueError("COCO polygon coordinates must contain x/y pairs")
        validated.append([float(value) for value in raw_polygon])
    encoded = mask_utils.frPyObjects(validated, height, width)
    decoded = mask_utils.decode(mask_utils.merge(encoded))
    return np.asarray(decoded, dtype=np.bool_)


def _rle_mask(segmentation: Mapping[str, object]) -> np.ndarray:
    try:
        from pycocotools import mask as mask_utils
    except ImportError as error:
        raise RuntimeError(
            "COCO RLE masks require pycocotools: pip install -e '.[coco]'"
        ) from error
    encoded = dict(segmentation)
    counts = encoded.get("counts")
    size = encoded.get("size")
    if not isinstance(size, list) or len(size) != 2:
        raise ValueError("COCO RLE size must contain height and width")
    if isinstance(counts, list):
        encoded = mask_utils.frPyObjects(encoded, int(size[0]), int(size[1]))
    decoded = mask_utils.decode(encoded)
    if decoded.ndim == 3:
        decoded = np.any(decoded, axis=2)
    return np.asarray(decoded, dtype=np.bool_)


def instance_mask(annotation: CocoInstance, image: CocoImage) -> np.ndarray:
    """Return a boolean ``(height, width)`` mask for one COCO instance."""

    segmentation = annotation.segmentation
    if isinstance(segmentation, list):
        mask = _polygon_mask(segmentation, width=image.width, height=image.height)
    elif isinstance(segmentation, dict):
        mask = _rle_mask(segmentation)
    else:
        raise ValueError("unsupported COCO segmentation type")
    if mask.shape != (image.height, image.width):
        raise ValueError(
            f"decoded mask shape {mask.shape} does not match image "
            f"{(image.height, image.width)}"
        )
    return mask
