"""Build the Clip Compass explainer page for one pipeline run.

The page walks through the pipeline stage by stage (sidebar) using the run's real
photos, groups, hypotheses, masked images and verdicts. It reads:

* the Mapper graph, image embeddings and explanations;
* ``perturb run`` output (per-image object and random-control results, plus the
  masked PNGs it wrote);
* ``verification verdicts`` and ``verification baseline`` output;
* the embedding cache, for the vectors of every masked PNG (no model is loaded).

It also exports TensorFlow Projector files (vectors.tsv + metadata.tsv).

Run from the repository root:

    .venv/bin/python scripts/build_explorer.py

All inputs default to the 25-image fixture run; pass flags to point at another run.
"""

from __future__ import annotations

import argparse
import base64
import csv
import io
import json
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from clip_compass.config import ArtifactPaths
from clip_compass.data.coco_instances import sha256_file
from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.mapper.graph import MapperGraphArtifact
from clip_compass.perturbation.masks import clip_visible_box
from clip_compass.perturbation.types import VerificationResults
from clip_compass.verification import VerdictArtifact

SAMPLE_CONTROLS = 8  # random-patch images embedded in the page per photo (all 39 are plotted)


def encode_image(path: Path, size: int, quality: int = 72) -> str:
    with Image.open(path) as image:
        image = image.convert("RGB")
        image.thumbnail((size, size))
        buffer = io.BytesIO()
        image.save(buffer, format="JPEG", quality=quality)
    return "data:image/jpeg;base64," + base64.b64encode(buffer.getvalue()).decode()


def read_rows(path: Path) -> list:
    if not path.exists():
        return []
    with path.open() as handle:
        return list(csv.DictReader(handle))


def r4(value: float) -> float:
    return round(float(value), 4)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--graph", type=Path, default=Path("artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json"))
    parser.add_argument("--embeddings", type=Path, default=Path("artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz"))
    parser.add_argument("--explanations", type=Path, default=Path("artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json"))
    parser.add_argument("--prompt-embeddings", type=Path, default=Path("artifacts/text-embeddings/fixture-prompts.npz"))
    parser.add_argument("--perturbation-results", type=Path, default=Path("artifacts/perturbations/fixture-dk/verification.json"))
    parser.add_argument("--verdicts", type=Path, default=Path("artifacts/verification/fixture-dk/verdicts.json"))
    parser.add_argument("--baseline", type=Path, default=Path("artifacts/verification/coco_val_first_25-all-nodes-offline-fixture-baseline/baseline.csv"))
    parser.add_argument("--annotations", type=Path, default=Path("data/processed/coco_instances/coco_val_first_25-instances.json"))
    parser.add_argument("--manifest", type=Path, default=Path("data/processed/coco_localized_narratives/coco_val_first_25.jsonl"))
    parser.add_argument("--data-root", type=Path, default=Path("data"))
    parser.add_argument("--cache-root", type=Path, default=ArtifactPaths.from_root(Path("artifacts")).embedding_cache)
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/diagnostics/explorer"))
    args = parser.parse_args()

    graph = MapperGraphArtifact.load(args.graph)
    images = EmbeddingBatch.load(args.embeddings)
    prompts = EmbeddingBatch.load(args.prompt_embeddings)
    explanations = json.loads(args.explanations.read_text())
    annotations = json.loads(args.annotations.read_text())
    results = VerificationResults.load(args.perturbation_results).as_dict()
    verdicts = VerdictArtifact.load(args.verdicts)
    if verdicts.perturbation_results_sha256 != sha256_file(args.perturbation_results):
        raise ValueError("verdicts were computed from different perturbation results")
    cache = EmbeddingCache(args.cache_root)
    spec = images.model_spec
    run_root = args.perturbation_results.parent

    image_row = {item_id: row for row, item_id in enumerate(images.item_ids)}
    originals = images.normalized[[image_row[item_id] for item_id in graph.item_ids]]
    prompt_row = {item_id: row for row, item_id in enumerate(prompts.item_ids)}
    item_index = {item_id: index for index, item_id in enumerate(graph.item_ids)}

    # 2D map: PCA fitted on the original photos only.
    mean = originals.mean(axis=0)
    _, singular, components = np.linalg.svd(originals - mean, full_matrices=False)
    axes = components[:2]
    kept = float((singular[:2] ** 2).sum() / (singular**2).sum())

    def project(vector: np.ndarray) -> list:
        return [r4(value) for value in (vector - mean) @ axes.T]

    similarity = originals @ originals.T
    np.fill_diagonal(similarity, np.nan)

    manifest = {}
    for line in args.manifest.read_text().splitlines():
        record = json.loads(line)
        manifest[record["image_id"]] = record

    category_name = {category["id"]: category["name"] for category in annotations["categories"]}
    image_size = {image["id"]: (image["width"], image["height"]) for image in annotations["images"]}
    objects_of = {item_id: [] for item_id in graph.item_ids}
    for annotation in annotations["annotations"]:
        segmentation = annotation["segmentation"]
        if annotation["image_id"] not in objects_of or not isinstance(segmentation, list):
            continue
        objects_of[annotation["image_id"]].append({
            "name": category_name[annotation["category_id"]],
            "area": round(float(annotation["area"])),
            "poly": [[round(float(value)) for value in polygon] for polygon in segmentation],
        })

    groups_of = {item_id: [] for item_id in graph.item_ids}
    for node in graph.nodes:
        for item_id in node.item_ids:
            groups_of[item_id].append(node.id)

    items = []
    for index, item_id in enumerate(graph.item_ids):
        record = manifest[item_id]
        row = similarity[index]
        width, height = image_size[item_id]
        items.append({
            "id": item_id,
            "xy": project(originals[index]),
            "thumb": encode_image(args.data_root / record["image_path"], 200),
            "img": encode_image(args.data_root / record["image_path"], 640, 70),
            "w": width,
            "h": height,
            "crop": list(clip_visible_box(width, height)),
            "captions": record["captions"],
            "objects": sorted(objects_of[item_id], key=lambda obj: -obj["area"]),
            "groups": groups_of[item_id],
            "nearest": int(np.nanargmax(row)),
            "farthest": int(np.nanargmin(row)),
            "vec": [round(float(value), 3) for value in originals[index]],
        })

    hypotheses = {}
    for explanation in explanations["explanations"]:
        for position, hypothesis in enumerate(explanation["hypotheses"]):
            hypotheses[(explanation["node_id"], position)] = (explanation, hypothesis)

    verdict_by_id = {value.hypothesis_id: value for value in verdicts.hypotheses}
    ranks_by_key = {(value.hypothesis_id, value.item_id): value for value in verdicts.images}
    engine_by_id = {value["hypothesis_id"]: value for value in results["hypotheses"]}
    baseline = {row["node_id"]: row for row in read_rows(args.baseline)}

    nodes = []
    for node in graph.nodes:
        explanation, hypothesis = hypotheses.get((node.id, 0), ({}, {}))
        hid = f"{node.id}-h00"
        prompt_id = f"{hid}-positive"
        prompt = prompts.normalized[prompt_row[prompt_id]] if prompt_id in prompt_row else None
        verdict = verdict_by_id.get(hid)
        base = baseline.get(node.id)
        nodes.append({
            "id": node.id,
            "members": list(node.member_indices),
            "summary": explanation.get("summary", ""),
            "name": hypothesis.get("name", node.id),
            "description": hypothesis.get("description", ""),
            "rationale": hypothesis.get("rationale", ""),
            "confidence": hypothesis.get("confidence"),
            "objects": hypothesis.get("coco_categories", []),
            "prompt": hypothesis.get("positive_prompt", ""),
            "predicted": hypothesis.get("predicted_effect", ""),
            "supporting": [item_index[i] for i in hypothesis.get("supporting_item_ids", []) if i in item_index],
            "contrasts": [item_index[i] for i in hypothesis.get("counterexample_item_ids", []) if i in item_index],
            "matches": [r4(value) for value in originals @ prompt] if prompt is not None else [],
            "verdict": None if verdict is None else {
                "label": verdict.label,
                "reason": verdict.reason,
                "visible": verdict.visible_members,
                "testable": verdict.testable_members,
                "members": verdict.member_count,
                "confirmed": verdict.prediction_confirmed,
                "tests": {
                    name: {"pct": test.mean_percentile, "p": test.p_value, "q": test.q_value, "n": test.image_count}
                    for name, test in verdict.tests.items()
                },
            },
            "engineLabel": engine_by_id.get(hid, {}).get("verdict", {}).get("label"),
            "baseline": {"auroc": r4(base["auroc"]), "accepted": base["accepted"] == "True"} if base else None,
        })

    def cached_vector(path: Path) -> np.ndarray:
        vector = cache.load(spec, "image", sha256_file(path))
        if vector is None:
            raise ValueError(f"no cached embedding for {path}; rerun `perturb run` with this cache")
        vector = np.asarray(vector, dtype=np.float64)
        return vector / np.linalg.norm(vector)

    pairs = []
    vectors, metadata = [], []
    for index, item_id in enumerate(graph.item_ids):
        vectors.append(originals[index])
        metadata.append([item_id, "original", "|".join(groups_of[item_id]), ""])
    for hypothesis in results["hypotheses"]:
        hid = hypothesis["hypothesis_id"]
        prompt = prompts.normalized[prompt_row[f"{hid}-positive"]].astype(np.float64)
        for record in hypothesis["images"]:
            item_id = record["item_id"]
            entry = {
                "id": f"{hid}-{item_id}",
                "node": hypothesis["node_id"],
                "item": item_index[item_id],
                "objects": hypothesis["coco_categories"],
                "status": record["status"],
                "visible": r4(record.get("visible_crop_fraction", 0.0)),
            }
            if record["status"] == "testable":
                folder = run_root / "images" / item_id
                object_png = next(folder.glob(f"{item_id}__{hid}__*__object.png"))
                control_pngs = sorted(folder.glob(f"{item_id}__{hid}__*__ctrl-*.png"))
                original = originals[item_index[item_id]].astype(np.float64)

                def place(path: Path) -> dict:
                    vector = cached_vector(path)
                    shift = vector - original
                    along = float(shift @ prompt)
                    total = float(np.linalg.norm(shift))
                    return {
                        "match": r4(vector @ prompt),
                        "along": r4(along),
                        "sideways": r4(np.sqrt(max(total**2 - along**2, 0.0))),
                        "vector": vector,
                    }

                target = place(object_png)
                controls = [place(path) for path in control_pngs]
                for kind, value in [("object hidden", target)] + [("random patch", c) for c in controls]:
                    vectors.append(value.pop("vector"))
                    metadata.append([item_id, kind, hypothesis["node_id"], hypothesis["name"]])
                ranks = ranks_by_key[(hid, item_id)]
                target["img"] = encode_image(object_png, 640, 70)
                entry.update({
                    "target": target,
                    "controls": [{"a": c["along"], "s": c["sideways"], "m": c["match"]} for c in controls],
                    "controlImgs": [encode_image(path, 640 if k == 0 else 240, 70) for k, path in enumerate(control_pngs[:SAMPLE_CONTROLS])],
                    "K": ranks.control_count,
                    "ranks": dict(ranks.ranks),
                })
            pairs.append(entry)

    data = {
        "items": items,
        "nodes": nodes,
        "edges": [[edge.source, edge.target] for edge in graph.edges],
        "pairs": pairs,
        "engine": {key: results["config"][key] for key in ("controls", "fill", "dilate_px", "min_visible_fraction", "max_control_overlap")},
        "verdictConfig": verdicts.config.as_dict(),
        "varianceKept": r4(kept),
        "stats": {
            "photos": len(items),
            "objects": sum(len(item["objects"]) for item in items),
            "kinds": len({obj["name"] for item in items for obj in item["objects"]}),
            "captions": sum(len(item["captions"]) for item in items),
            "covered": sum(1 for item in items if item["groups"]),
            "topKinds": Counter(obj["name"] for item in items for obj in item["objects"]).most_common(6),
        },
    }

    args.output_dir.mkdir(parents=True, exist_ok=True)
    template = (Path(__file__).parent / "explorer_template.html").read_text()
    page = args.output_dir / "explorer.html"
    page.write_text(template.replace("/*__DATA__*/null", json.dumps(data, separators=(",", ":"))))

    projector = args.output_dir / "projector"
    projector.mkdir(exist_ok=True)
    with (projector / "vectors.tsv").open("w") as handle:
        for vector in vectors:
            handle.write("\t".join(f"{value:.6f}" for value in vector) + "\n")
    with (projector / "metadata.tsv").open("w") as handle:
        handle.write("image_id\tversion\tgroup\thypothesis\n")
        for row in metadata:
            handle.write("\t".join(row) + "\n")

    size_mb = page.stat().st_size / 1e6
    print(json.dumps({"page": str(page), "size_mb": round(size_mb, 2), "projector": str(projector)}, indent=2))


if __name__ == "__main__":
    main()
