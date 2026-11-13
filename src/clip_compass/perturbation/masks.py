"""CLIP-crop-aware object masks and area-matched random control masks."""

from __future__ import annotations

import hashlib
import math
from typing import Iterable, List, Optional, Tuple

import numpy as np

from clip_compass.data.coco_instances import CocoInstanceSubset
from clip_compass.data.coco_masks import instance_mask

Box = Tuple[int, int, int, int]  # left, top, right, bottom in source pixels (half-open)


def clip_visible_box(width: int, height: int, *, size: int = 224, inset: int = 1) -> Box:
    """The part of the source image that survives CLIP's resize and centre crop.

    Mirrors ``CLIPImageProcessor``: the shorter side becomes ``size``, the longer
    side ``int(size * long / short)``, and the crop starts at ``(resized - size) // 2``.
    Cropped axes are inset by ``inset`` source pixels to absorb resampling at the edge.
    """

    if width <= height:
        resized_width, resized_height = size, int(size * height / width)
    else:
        resized_width, resized_height = int(size * width / height), size

    def axis(source: int, resized: int) -> Tuple[int, int]:
        if resized <= size:
            return 0, source
        offset, scale = (resized - size) // 2, source / resized
        start = math.ceil(offset * scale) + inset
        stop = math.floor((offset + size) * scale) - inset
        return max(0, start), min(source, stop)

    left, right = axis(width, resized_width)
    top, bottom = axis(height, resized_height)
    return left, top, right, bottom


def dilate(mask: np.ndarray, radius: int) -> np.ndarray:
    """Binary dilation with a square ``(2r+1)`` kernel, using numpy shifts only."""

    result = mask.copy()
    for axis in (1, 0):
        source = result.copy()
        for step in range(1, radius + 1):
            if axis == 1:
                result[:, step:] |= source[:, :-step]
                result[:, :-step] |= source[:, step:]
            else:
                result[step:, :] |= source[:-step, :]
                result[:-step, :] |= source[step:, :]
    return result


def category_mask(
    annotations: CocoInstanceSubset,
    item_id: str,
    names: Iterable[str],
    *,
    dilate_px: int = 3,
) -> Optional[np.ndarray]:
    """Union of every instance of the named categories, or ``None`` if there are none."""

    wanted = set(names)
    image = annotations.images[annotations.item_ids.index(item_id)]
    categories = annotations.categories_by_id
    instances = [
        annotation
        for annotation in annotations.annotations_for(item_id)
        if categories[annotation.category_id].name in wanted
    ]
    if not instances:
        return None
    mask = np.zeros((image.height, image.width), dtype=np.bool_)
    for annotation in instances:
        mask |= instance_mask(annotation, image)
    return dilate(mask, dilate_px)


def visible_pixels(mask: np.ndarray, box: Box) -> int:
    left, top, right, bottom = box
    return int(mask[top:bottom, left:right].sum())


def _seed(key: str) -> int:
    # Python's hash() is salted per process, so derive seeds from SHA-256 instead.
    return int.from_bytes(hashlib.sha256(key.encode()).digest()[:8], "big")


def control_masks(
    mask: np.ndarray,
    box: Box,
    *,
    count: int,
    seed_key: str,
    max_overlap: float = 0.25,
    max_attempts: int = 500,
) -> List[np.ndarray]:
    """Shift the object's visible mask to ``count`` random places inside the crop.

    Each control keeps the visible mask's exact shape and area, overlaps the object
    by at most ``max_overlap`` of its area, and uses its own seed, so adding controls
    never changes earlier ones. Returns fewer than ``count`` masks when there is no room.
    """

    left, top, right, bottom = box
    visible = np.zeros_like(mask)
    visible[top:bottom, left:right] = mask[top:bottom, left:right]
    rows, cols = np.flatnonzero(visible.any(axis=1)), np.flatnonzero(visible.any(axis=0))
    if not rows.size:
        return []
    template = visible[rows[0] : rows[-1] + 1, cols[0] : cols[-1] + 1]
    height, width = template.shape
    limit = max_overlap * template.sum()
    used = {(int(rows[0]), int(cols[0]))}
    controls: List[np.ndarray] = []
    for index in range(count):
        rng = np.random.default_rng(_seed(f"{seed_key}|{index}"))
        for _ in range(max_attempts):
            y = int(rng.integers(top, bottom - height + 1))
            x = int(rng.integers(left, right - width + 1))
            window = mask[y : y + height, x : x + width]
            if (y, x) not in used and (template & window).sum() <= limit:
                break
        else:
            return controls
        used.add((y, x))
        shifted = np.zeros_like(mask)
        shifted[y : y + height, x : x + width] = template
        controls.append(shifted)
    return controls
