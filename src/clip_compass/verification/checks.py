"""Input checks shared by the verification stages."""

from __future__ import annotations

from pathlib import Path

from clip_compass.data.coco_instances import sha256_file
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.explanations.types import ExplanationArtifact, hypothesis_id
from clip_compass.mapper.graph import MapperGraphArtifact


def prompt_id(node_id: str, hypothesis_index: int, polarity: str) -> str:
    """Text-embedding ID written by ``explanations export-prompts``."""

    return f"{hypothesis_id(node_id, hypothesis_index)}-{polarity}"


def check_image_text_compatibility(images: EmbeddingBatch, texts: EmbeddingBatch) -> None:
    """Image and text vectors must come from the same pinned CLIP model."""

    if images.modality != "image":
        raise ValueError("image embeddings must have image modality")
    if texts.modality != "text":
        raise ValueError("prompt embeddings must have text modality")
    image_spec, text_spec = images.model_spec, texts.model_spec
    if (image_spec.provider, image_spec.model_id, image_spec.revision) != (
        text_spec.provider,
        text_spec.model_id,
        text_spec.revision,
    ):
        raise ValueError("image and prompt embeddings must use the same CLIP model revision")
    if images.embedding_dim != texts.embedding_dim:
        raise ValueError("image and prompt embedding dimensions must match")


def check_graph_inputs(
    graph: MapperGraphArtifact,
    image_embeddings: EmbeddingBatch,
    explanations: ExplanationArtifact,
    graph_path: Path,
) -> str:
    """Validate the graph/embedding/explanation trio and return the graph hash."""

    if graph.item_ids != image_embeddings.item_ids:
        raise ValueError("graph and image embedding item IDs must align")
    fingerprint = image_embeddings.model_spec.fingerprint
    if graph.embedding_model_fingerprint != fingerprint:
        raise ValueError("graph and image embedding fingerprints do not match")
    if explanations.embedding_model_fingerprint != fingerprint:
        raise ValueError("explanation and image embedding fingerprints do not match")
    graph_sha256 = sha256_file(graph_path)
    if explanations.graph_sha256 != graph_sha256:
        raise ValueError("explanations were prepared from a different Mapper graph")
    return graph_sha256
