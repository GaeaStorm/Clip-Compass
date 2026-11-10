import json
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from clip_compass.data.coco_instances import (
    CocoCategory,
    CocoImage,
    CocoInstance,
    CocoInstanceSubset,
)
from clip_compass.explanations.context import build_explanation_contexts
from clip_compass.explanations.service import (
    generate_explanations,
    write_hypothesis_prompts,
)
from clip_compass.explanations.types import (
    ExplanationArtifact,
    ExplanationContextBundle,
    ExplanationHypothesis,
    NodeExplanation,
)
from test_mapper_diagnostics import diagnostic_fixture

try:
    from clip_compass.explanations.openai_agent import OpenAIExplanationAgent
except RuntimeError:
    OpenAIExplanationAgent = None


class FakeExplanationAgent:
    @property
    def metadata(self):
        return {"provider": "fake", "model": "deterministic"}

    def explain(self, context, *, max_hypotheses):
        target = next(item for item in context.evidence if item.role == "target")
        contrast = next(item for item in context.evidence if item.role == "contrast")
        category = target.category_names[0] if target.category_names else "object"
        return NodeExplanation(
            node_id=context.node_id,
            summary=f"Synthetic explanation for {context.node_id}",
            hypotheses=(
                ExplanationHypothesis(
                    name="Target object",
                    description="The target images share an annotated object.",
                    positive_prompt=f"a photo containing a {category}",
                    negative_prompt=f"a photo without a {category}",
                    coco_categories=(category,),
                    predicted_effect="both",
                    rationale="The category occurs in the selected target evidence.",
                    supporting_item_ids=(target.item_id,),
                    counterexample_item_ids=(contrast.item_id,),
                    confidence=0.75,
                ),
            ),
            response_id="fake-response",
        )


class FakeResponsesAPI:
    def __init__(self):
        self.request = None

    def parse(self, **kwargs):
        self.request = kwargs
        output_type = kwargs["text_format"]
        parsed = output_type.model_validate(
            {
                "summary": "The target examples contain people.",
                "hypotheses": [
                    {
                        "name": "People",
                        "description": "People distinguish the target examples.",
                        "positive_prompt": "a photo containing a person",
                        "negative_prompt": "a photo without a person",
                        "coco_categories": ["person"],
                        "predicted_effect": "both",
                        "rationale": "The target evidence is annotated with person.",
                        "supporting_item_ids": ["1"],
                        "counterexample_item_ids": ["3"],
                        "confidence": 0.8,
                    }
                ],
            }
        )
        return SimpleNamespace(
            status="completed", output_parsed=parsed, id="response-123"
        )


class FakeOpenAIClient:
    def __init__(self):
        self.responses = FakeResponsesAPI()


def annotation_fixture():
    return CocoInstanceSubset(
        split="val",
        source_sha256="1" * 64,
        manifest_sha256="2" * 64,
        item_ids=("1", "2", "3"),
        categories=(
            CocoCategory(id=1, name="person", supercategory="person"),
            CocoCategory(id=2, name="car", supercategory="vehicle"),
        ),
        images=(
            CocoImage(id="1", width=20, height=10, file_name="1.jpg"),
            CocoImage(id="2", width=20, height=10, file_name="2.jpg"),
            CocoImage(id="3", width=20, height=10, file_name="3.jpg"),
        ),
        annotations=(
            CocoInstance(
                id=1,
                image_id="1",
                category_id=1,
                bbox_xywh=(1.0, 1.0, 4.0, 5.0),
                area=20.0,
                iscrowd=False,
                segmentation=[[1, 1, 5, 1, 5, 6, 1, 6]],
            ),
            CocoInstance(
                id=2,
                image_id="2",
                category_id=1,
                bbox_xywh=(2.0, 1.0, 3.0, 5.0),
                area=15.0,
                iscrowd=False,
                segmentation=[[2, 1, 5, 1, 5, 6, 2, 6]],
            ),
            CocoInstance(
                id=3,
                image_id="3",
                category_id=2,
                bbox_xywh=(1.0, 2.0, 8.0, 4.0),
                area=32.0,
                iscrowd=False,
                segmentation=[[1, 2, 9, 2, 9, 6, 1, 6]],
            ),
        ),
    )


class ExplanationTests(unittest.TestCase):
    def test_context_generation_and_fake_agent_round_trip(self) -> None:
        graph, embeddings = diagnostic_fixture()
        annotations = annotation_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_path = root / "graph.json"
            graph.save(graph_path)
            annotations_path = root / "annotations.json"
            annotations.save(annotations_path)
            manifest_path = root / "manifest.jsonl"
            from PIL import Image

            for item_id in graph.item_ids:
                image_path = root / "images" / f"{item_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (20, 10), (80, 120, 160)).save(image_path)
            manifest_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "image_id": item_id,
                            "image_path": f"images/{item_id}.jpg",
                            "captions": [f"Caption for {item_id}."],
                        }
                    )
                    + "\n"
                    for item_id in graph.item_ids
                ),
                encoding="utf-8",
            )
            bundle = build_explanation_contexts(
                graph,
                embeddings,
                annotations,
                graph_path=graph_path,
                annotations_path=annotations_path,
                manifest_path=manifest_path,
                data_root=root,
                node_ids=("node-0000",),
                representatives=1,
                contrasts=1,
                caption_limit=1,
            )
            context_path = root / "context.json"
            bundle.save(context_path)
            loaded_context = ExplanationContextBundle.load(context_path)
            artifact = generate_explanations(
                loaded_context, FakeExplanationAgent(), max_hypotheses=2
            )
            artifact_path = root / "explanations.json"
            artifact.save(artifact_path)
            loaded_artifact = ExplanationArtifact.load(artifact_path)
            prompts_path = root / "prompts.jsonl"
            write_hypothesis_prompts(loaded_artifact, prompts_path)
            prompt_rows = [
                json.loads(line)
                for line in prompts_path.read_text(encoding="utf-8").splitlines()
            ]

        context = loaded_context.contexts[0]
        self.assertEqual(context.node_id, "node-0000")
        self.assertEqual({item.role for item in context.evidence}, {"target", "contrast"})
        self.assertEqual(len(loaded_artifact.explanations), 1)
        self.assertEqual(len(prompt_rows), 2)
        self.assertEqual({row["polarity"] for row in prompt_rows}, {"positive", "negative"})

    @unittest.skipUnless(
        OpenAIExplanationAgent is not None,
        "OpenAI explanation-agent optional dependencies are not installed",
    )
    def test_openai_agent_uses_images_and_typed_output(self) -> None:
        graph, embeddings = diagnostic_fixture()
        annotations = annotation_fixture()
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            graph_path = root / "graph.json"
            graph.save(graph_path)
            annotations_path = root / "annotations.json"
            annotations.save(annotations_path)
            for item_id in graph.item_ids:
                from PIL import Image

                image_path = root / "images" / f"{item_id}.jpg"
                image_path.parent.mkdir(parents=True, exist_ok=True)
                Image.new("RGB", (20, 10), (80, 120, 160)).save(image_path)
            manifest_path = root / "manifest.jsonl"
            manifest_path.write_text(
                "".join(
                    json.dumps(
                        {
                            "image_id": item_id,
                            "image_path": f"images/{item_id}.jpg",
                            "captions": [f"Caption for {item_id}."],
                        }
                    )
                    + "\n"
                    for item_id in graph.item_ids
                ),
                encoding="utf-8",
            )
            bundle = build_explanation_contexts(
                graph,
                embeddings,
                annotations,
                graph_path=graph_path,
                annotations_path=annotations_path,
                manifest_path=manifest_path,
                data_root=root,
                node_ids=("node-0000",),
                representatives=1,
                contrasts=1,
                caption_limit=1,
            )
            client = FakeOpenAIClient()
            agent = OpenAIExplanationAgent(client=client, model="test-model")
            explanation = agent.explain(bundle.contexts[0], max_hypotheses=2)

        content = client.responses.request["input"][1]["content"]
        image_parts = [part for part in content if part["type"] == "input_image"]
        self.assertEqual(explanation.response_id, "response-123")
        self.assertEqual(explanation.hypotheses[0].coco_categories, ("person",))
        self.assertEqual(len(image_parts), 2)
        self.assertTrue(image_parts[0]["image_url"].startswith("data:image/jpeg;base64,"))
        self.assertFalse(client.responses.request["store"])


if __name__ == "__main__":
    unittest.main()
