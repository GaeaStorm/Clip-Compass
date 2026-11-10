"""Interfaces shared by visual perturbation backends."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import numpy as np


@dataclass(frozen=True)
class PerturbedImages:
    """Backend result for a targeted mask and a matched control mask."""

    target_image: object
    control_image: object
    control_mask: np.ndarray
    control_overlap_pixel_count: int


class VisualPerturbationEngine(Protocol):
    @property
    def metadata(self) -> Mapping[str, object]:
        ...

    def perturb(
        self,
        image: object,
        target_mask: np.ndarray,
        *,
        control_seed: int,
    ) -> PerturbedImages:
        ...
