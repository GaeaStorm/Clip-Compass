"""Backend protocol for CLIP encoders."""

from pathlib import Path
from typing import Protocol, Sequence

import numpy as np

from .types import ClipModelSpec


class ClipEncoder(Protocol):
    """Minimal interface needed by the cache-aware embedding service."""

    @property
    def spec(self) -> ClipModelSpec:
        ...

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        """Return unnormalized projected CLIP vectors for the given images."""

        ...
