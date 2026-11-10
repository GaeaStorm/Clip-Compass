"""Deterministic mask-based visual perturbations and matched controls."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np

from .base import PerturbedImages


FILL_MODES = ("gray", "mean", "blur")


def translated_control_mask(
    target_mask: np.ndarray,
    *,
    seed: int,
    max_attempts: int = 256,
) -> Tuple[np.ndarray, int]:
    """Translate a cropped target mask to minimize overlap without changing its area."""

    mask = np.asarray(target_mask, dtype=np.bool_)
    if mask.ndim != 2 or not mask.any():
        raise ValueError("target mask must be a non-empty two-dimensional array")
    if max_attempts < 1:
        raise ValueError("max_attempts must be positive")
    rows, columns = np.nonzero(mask)
    top, bottom = int(rows.min()), int(rows.max()) + 1
    left, right = int(columns.min()), int(columns.max()) + 1
    cropped = mask[top:bottom, left:right]
    height, width = mask.shape
    crop_height, crop_width = cropped.shape
    rng = np.random.default_rng(seed)
    best_mask = None
    best_overlap = mask.size + 1
    seen = set()
    position_count = (height - crop_height + 1) * (width - crop_width + 1)
    attempts = min(max_attempts, position_count)
    for _ in range(attempts):
        candidate_top = int(rng.integers(0, height - crop_height + 1))
        candidate_left = int(rng.integers(0, width - crop_width + 1))
        position = (candidate_top, candidate_left)
        if position in seen:
            continue
        seen.add(position)
        candidate = np.zeros_like(mask)
        candidate[
            candidate_top : candidate_top + crop_height,
            candidate_left : candidate_left + crop_width,
        ] = cropped
        overlap = int(np.logical_and(candidate, mask).sum())
        if overlap < best_overlap:
            best_mask = candidate
            best_overlap = overlap
        if overlap == 0:
            break
    if best_mask is None:
        best_mask = mask.copy()
        best_overlap = int(mask.sum())
    if int(best_mask.sum()) != int(mask.sum()):
        raise AssertionError("translated control changed the number of masked pixels")
    return best_mask, best_overlap


@dataclass(frozen=True)
class MaskingPerturbationEngine:
    """Replace target/control pixels using the same deterministic fill operation."""

    fill_mode: str = "gray"
    fill_rgb: Tuple[int, int, int] = (127, 127, 127)
    blur_radius: float = 12.0
    max_control_attempts: int = 256

    def __post_init__(self) -> None:
        if self.fill_mode not in FILL_MODES:
            raise ValueError(f"fill_mode must be one of {FILL_MODES}")
        if len(self.fill_rgb) != 3 or any(not 0 <= value <= 255 for value in self.fill_rgb):
            raise ValueError("fill_rgb must contain three values in [0, 255]")
        if self.blur_radius <= 0.0:
            raise ValueError("blur_radius must be positive")
        if self.max_control_attempts < 1:
            raise ValueError("max_control_attempts must be positive")

    @property
    def metadata(self) -> Dict[str, object]:
        return {
            "name": "coco_mask_pair",
            "fill_mode": self.fill_mode,
            "fill_rgb": list(self.fill_rgb),
            "blur_radius": self.blur_radius,
            "control_strategy": "shape_preserving_random_translation_min_overlap",
            "max_control_attempts": self.max_control_attempts,
            "image_format": "png",
        }

    def perturb(
        self,
        image: object,
        target_mask: np.ndarray,
        *,
        control_seed: int,
    ) -> PerturbedImages:
        control_mask, overlap = translated_control_mask(
            target_mask,
            seed=control_seed,
            max_attempts=self.max_control_attempts,
        )
        return PerturbedImages(
            target_image=self._apply(image, target_mask),
            control_image=self._apply(image, control_mask),
            control_mask=control_mask,
            control_overlap_pixel_count=overlap,
        )

    def _apply(self, image: object, mask: np.ndarray) -> object:
        try:
            from PIL import Image, ImageFilter
        except ImportError as error:
            raise RuntimeError(
                "Visual perturbations require Pillow: pip install -e '.[coco]'"
            ) from error
        if not isinstance(image, Image.Image):
            raise TypeError("image must be a PIL Image")
        pixels = np.asarray(image.convert("RGB"), dtype=np.uint8).copy()
        boolean_mask = np.asarray(mask, dtype=np.bool_)
        if boolean_mask.shape != pixels.shape[:2]:
            raise ValueError("mask dimensions must match the source image")
        if self.fill_mode == "gray":
            pixels[boolean_mask] = np.asarray(self.fill_rgb, dtype=np.uint8)
        elif self.fill_mode == "mean":
            reference = pixels[~boolean_mask]
            if not reference.size:
                reference = pixels.reshape(-1, 3)
            fill = np.rint(reference.mean(axis=0)).astype(np.uint8)
            pixels[boolean_mask] = fill
        else:
            blurred = np.asarray(
                image.convert("RGB").filter(ImageFilter.GaussianBlur(self.blur_radius)),
                dtype=np.uint8,
            )
            pixels[boolean_mask] = blurred[boolean_mask]
        return Image.fromarray(pixels)
