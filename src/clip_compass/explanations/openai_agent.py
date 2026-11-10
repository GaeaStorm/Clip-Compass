"""Multimodal OpenAI Responses API implementation of the explanation agent."""

from __future__ import annotations

import base64
import io
import json
from enum import Enum
from pathlib import Path
from typing import Dict, List, Mapping, Optional

try:
    from pydantic import BaseModel, ConfigDict
except ImportError as error:
    raise RuntimeError(
        "The OpenAI explanation agent requires the 'agent' extra: "
        "pip install -e '.[agent]'"
    ) from error

from .types import ExplanationHypothesis, NodeExplanation, NodeExplanationContext


SYSTEM_PROMPT = """You explain why images occupy a node in a CLIP Mapper graph.
Compare TARGET images against CONTRAST images. Propose observable, falsifiable visual
properties that distinguish the target node. Ground claims in the supplied images,
captions, and COCO categories. Do not treat captions or category labels as infallible.
Each positive_prompt and negative_prompt must be a short natural-language CLIP prompt.
Only name COCO categories present in the supplied evidence. Supporting IDs must be TARGET
items and counterexample IDs must be CONTRAST items. Prefer hypotheses that can later be
tested by masking the named object categories. Do not claim that a hypothesis is verified;
this stage only generates candidates for CLIP scoring and perturbation experiments."""


class ExplanationAgentError(RuntimeError):
    """Raised when the model refuses, truncates, or returns invalid evidence links."""


class _PredictedEffect(str, Enum):
    decrease_text_similarity = "decrease_text_similarity"
    move_away_from_node_centroid = "move_away_from_node_centroid"
    both = "both"


class _HypothesisOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    name: str
    description: str
    positive_prompt: str
    negative_prompt: str
    coco_categories: List[str]
    predicted_effect: _PredictedEffect
    rationale: str
    supporting_item_ids: List[str]
    counterexample_item_ids: List[str]
    confidence: float


class _NodeExplanationOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    summary: str
    hypotheses: List[_HypothesisOutput]


class OpenAIExplanationAgent:
    """Generate typed, image-grounded hypotheses for one Mapper node."""

    def __init__(
        self,
        *,
        model: str = "gpt-6-astra",
        api_key: Optional[str] = None,
        image_size: int = 768,
        image_detail: str = "low",
        client: Optional[object] = None,
    ):
        if image_size < 128 or image_size > 2048:
            raise ValueError("image_size must be between 128 and 2048")
        if image_detail not in {"low", "high", "auto", "original"}:
            raise ValueError("unsupported OpenAI image detail")
        if client is None:
            try:
                from openai import OpenAI
            except ImportError as error:
                raise RuntimeError(
                    "The OpenAI explanation agent requires the 'agent' extra: "
                    "pip install -e '.[agent]'"
                ) from error
            client = OpenAI(api_key=api_key)
        self._client = client
        self._model = model
        self._image_size = image_size
        self._image_detail = image_detail
        self._output_type = _NodeExplanationOutput

    @property
    def metadata(self) -> Mapping[str, object]:
        return {
            "provider": "openai-responses",
            "model": self._model,
            "image_size": self._image_size,
            "image_detail": self._image_detail,
            "structured_output": True,
        }

    def explain(
        self, context: NodeExplanationContext, *, max_hypotheses: int
    ) -> NodeExplanation:
        if max_hypotheses < 1 or max_hypotheses > 8:
            raise ValueError("max_hypotheses must be between 1 and 8")
        response = self._client.responses.parse(
            model=self._model,
            input=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": self._content(context, max_hypotheses)},
            ],
            text_format=self._output_type,
            store=False,
        )
        if response.status != "completed" or response.output_parsed is None:
            raise ExplanationAgentError(
                f"OpenAI explanation response did not complete: {response.status}"
            )
        parsed = response.output_parsed
        if not 1 <= len(parsed.hypotheses) <= max_hypotheses:
            raise ExplanationAgentError("model returned an invalid hypothesis count")
        targets = {item.item_id for item in context.evidence if item.role == "target"}
        contrasts = {
            item.item_id for item in context.evidence if item.role == "contrast"
        }
        available_categories = {
            category for item in context.evidence for category in item.category_names
        }
        hypotheses = []
        for value in parsed.hypotheses:
            supporting = tuple(value.supporting_item_ids)
            counterexamples = tuple(value.counterexample_item_ids)
            categories = tuple(sorted(set(value.coco_categories)))
            if not supporting or not set(supporting).issubset(targets):
                raise ExplanationAgentError("hypothesis cites invalid target evidence")
            if not set(counterexamples).issubset(contrasts):
                raise ExplanationAgentError("hypothesis cites invalid contrast evidence")
            if not set(categories).issubset(available_categories):
                raise ExplanationAgentError("hypothesis cites unavailable COCO categories")
            hypotheses.append(
                ExplanationHypothesis(
                    name=value.name,
                    description=value.description,
                    positive_prompt=value.positive_prompt,
                    negative_prompt=value.negative_prompt,
                    coco_categories=categories,
                    predicted_effect=value.predicted_effect.value,
                    rationale=value.rationale,
                    supporting_item_ids=supporting,
                    counterexample_item_ids=counterexamples,
                    confidence=float(value.confidence),
                )
            )
        return NodeExplanation(
            node_id=context.node_id,
            summary=parsed.summary,
            hypotheses=tuple(hypotheses),
            response_id=response.id,
        )

    def _content(
        self, context: NodeExplanationContext, max_hypotheses: int
    ) -> List[Dict[str, object]]:
        context_data = context.as_dict().copy()
        context_data.pop("evidence")
        instructions = {
            "task": (
                f"Generate between 1 and {max_hypotheses} hypotheses that distinguish "
                f"target node {context.node_id} from the contrast examples."
            ),
            "node_context": context_data,
            "evidence_metadata": [
                {
                    "item_id": item.item_id,
                    "role": item.role,
                    "captions": list(item.captions),
                    "category_names": list(item.category_names),
                }
                for item in context.evidence
            ],
        }
        content: List[Dict[str, object]] = [
            {
                "type": "input_text",
                "text": json.dumps(instructions, indent=2, sort_keys=True),
            }
        ]
        for item in context.evidence:
            content.append(
                {
                    "type": "input_text",
                    "text": f"{item.role.upper()} image {item.item_id}",
                }
            )
            content.append(
                {
                    "type": "input_image",
                    "image_url": _image_data_uri(
                        Path(item.image_path), max_size=self._image_size
                    ),
                    "detail": self._image_detail,
                }
            )
        return content


def _image_data_uri(path: Path, *, max_size: int) -> str:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError("Pillow is required to send explanation images") from error
    if not path.is_file():
        raise FileNotFoundError(path)
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((max_size, max_size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=85, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"
