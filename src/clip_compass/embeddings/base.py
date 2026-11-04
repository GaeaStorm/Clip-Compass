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


class ClipTextEncoder(Protocol):
    """Minimal interface needed by cache-aware CLIP text embedding."""

    @property
    def text_spec(self) -> ClipModelSpec:
        ...

    def encode_texts(self, texts: Sequence[str]) -> np.ndarray:
        """Return unnormalized projected CLIP vectors for the given texts."""

        ...
