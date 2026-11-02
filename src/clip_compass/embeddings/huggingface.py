"""Hugging Face Transformers implementation of the CLIP backend."""

from __future__ import annotations

from pathlib import Path
from typing import Optional, Sequence

import numpy as np

from .types import ClipModelSpec


class HuggingFaceClipEncoder:
    """Encode images with unnormalized CLIP projection-layer features."""

    def __init__(
        self,
        *,
        model_id: str = "openai/clip-vit-base-patch32",
        revision: str = "3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268",
        device: str = "auto",
        model_cache_dir: Optional[Path] = None,
        local_files_only: bool = False,
    ):
        try:
            import torch
            import transformers
        except ImportError as error:
            raise RuntimeError(
                "The Hugging Face CLIP backend requires the 'clip' extra: "
                "pip install -e '.[clip]'"
            ) from error

        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                device = "mps"
            else:
                device = "cpu"

        self._torch = torch
        self._transformers = transformers
        self._device = torch.device(device)
        self._model_id = model_id
        self._revision = revision
        self._cache_dir = (
            str(model_cache_dir.expanduser().resolve()) if model_cache_dir else None
        )
        self._local_files_only = local_files_only
        self._processor = None
        self._model = None
        self._spec = ClipModelSpec(
            provider="huggingface-transformers",
            model_id=model_id,
            revision=revision,
            preprocessing_id=f"CLIPImageProcessor:{model_id}@{revision}",
        )

    @property
    def spec(self) -> ClipModelSpec:
        return self._spec

    @property
    def device(self) -> str:
        return str(self._device)

    def _load_model(self) -> None:
        if self._model is not None:
            return
        self._processor = self._transformers.CLIPImageProcessor.from_pretrained(
            self._model_id,
            revision=self._revision,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        )
        self._model = self._transformers.CLIPModel.from_pretrained(
            self._model_id,
            revision=self._revision,
            cache_dir=self._cache_dir,
            local_files_only=self._local_files_only,
        ).to(self._device)
        self._model.eval()

    def encode_images(self, image_paths: Sequence[Path]) -> np.ndarray:
        self._load_model()
        try:
            from PIL import Image
        except ImportError as error:
            raise RuntimeError("Pillow is required to load images") from error

        images = []
        for path in image_paths:
            with Image.open(path) as image:
                images.append(image.convert("RGB").copy())
        if self._processor is None or self._model is None:
            raise AssertionError("CLIP model failed to initialize")
        inputs = self._processor(images=images, return_tensors="pt")
        pixel_values = inputs["pixel_values"].to(self._device)
        with self._torch.inference_mode():
            output = self._model.get_image_features(pixel_values=pixel_values)

        # Transformers <5 returns a Tensor. The current main branch returns a
        # model output whose pooler_output is already projected to CLIP space.
        if self._torch.is_tensor(output):
            features = output
        elif hasattr(output, "pooler_output"):
            features = output.pooler_output
        elif isinstance(output, tuple) and output:
            features = output[0]
        else:
            raise TypeError(f"Unexpected CLIP feature output: {type(output)!r}")
        return features.detach().to("cpu", dtype=self._torch.float32).numpy()
