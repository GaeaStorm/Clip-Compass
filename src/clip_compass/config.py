"""Shared filesystem configuration for Clip Compass artifacts."""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class DataPaths:
    """Canonical locations for downloaded and derived dataset artifacts."""

    root: Path

    @classmethod
    def from_root(cls, root: Path) -> "DataPaths":
        return cls(root=root.expanduser().resolve())

    @property
    def localized_narratives(self) -> Path:
        return self.root / "raw" / "localized_narratives"

    @property
    def coco_images(self) -> Path:
        return self.root / "raw" / "coco" / "images"

    @property
    def manifests(self) -> Path:
        return self.root / "processed" / "coco_localized_narratives"


@dataclass(frozen=True)
class ArtifactPaths:
    """Canonical locations for computed research artifacts."""

    root: Path

    @classmethod
    def from_root(cls, root: Path) -> "ArtifactPaths":
        return cls(root=root.expanduser().resolve())

    @property
    def embedding_cache(self) -> Path:
        return self.root / "cache" / "clip"

    @property
    def embeddings(self) -> Path:
        return self.root / "embeddings"

    @property
    def lenses(self) -> Path:
        return self.root / "lenses"

    @property
    def mapper_graphs(self) -> Path:
        return self.root / "mapper"

    @property
    def mapper_exports(self) -> Path:
        return self.root / "mapper-exports"

    @property
    def diagnostics(self) -> Path:
        return self.root / "diagnostics"
