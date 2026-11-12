"""Command-line entry point for Clip Compass."""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from clip_compass.config import ArtifactPaths
from clip_compass.data.coco_instances import (
    CocoInstanceSubset,
    prepare_coco_instance_subset,
)
from clip_compass.data.localized_narratives import prepare_coco_subset
from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.huggingface import HuggingFaceClipEncoder
from clip_compass.embeddings.service import embed_images, embed_texts
from clip_compass.embeddings.similarity import image_text_similarity
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.explanations.context import build_explanation_contexts
from clip_compass.explanations.service import (
    generate_explanations,
    write_hypothesis_prompts,
)
from clip_compass.explanations.types import (
    ExplanationArtifact,
    ExplanationContextBundle,
)
from clip_compass.mapper.builder import MapperConfig, build_mapper_graph
from clip_compass.mapper.diagnostics import write_mapper_diagnostic_report
from clip_compass.mapper.graph import MapperGraphArtifact
from clip_compass.mapper.lenses import LENS_NAMES, LensResult, compute_lens
from clip_compass.perturbation.verify import (
    FILLS,
    PerturbationConfig,
    format_summary,
    verified_explanations,
    verify_hypotheses,
)


def _manifest_images(
    manifest_path: Path, data_root: Path, limit: Optional[int]
) -> Tuple[List[str], List[Path]]:
    item_ids: List[str] = []
    image_paths: List[Path] = []
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if limit is not None and len(item_ids) >= limit:
                break
            if not line.strip():
                continue
            row = json.loads(line)
            try:
                image_id = str(row["image_id"])
                relative_path = Path(str(row["image_path"]))
            except (KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid image manifest row at {manifest_path}:{line_number}"
                ) from error
            if relative_path.is_absolute():
                raise ValueError(
                    f"Manifest image paths must be relative to data root: {relative_path}"
                )
            item_ids.append(image_id)
            image_paths.append(data_root.expanduser().resolve() / relative_path)
    if not item_ids:
        raise ValueError(f"No image records found in {manifest_path}")
    return item_ids, image_paths


def _text_inputs(
    inline_texts: Optional[Sequence[str]], texts_file: Optional[Path]
) -> Tuple[List[str], List[str]]:
    if inline_texts:
        values = [value.strip() for value in inline_texts]
        if any(not value for value in values):
            raise ValueError("--text values must not be empty")
        return [f"text-{index:04d}" for index in range(len(values))], values
    if texts_file is None:
        raise ValueError("provide at least one --text or a --texts-file")
    ids: List[str] = []
    values: List[str] = []
    with texts_file.open("r", encoding="utf-8") as source:
        for line_number, line in enumerate(source, start=1):
            if not line.strip():
                continue
            stripped = line.strip()
            if stripped.startswith("{"):
                try:
                    row = json.loads(stripped)
                    item_id = str(row["id"])
                    value = str(row["text"]).strip()
                except (json.JSONDecodeError, KeyError, TypeError) as error:
                    raise ValueError(
                        f"Invalid text JSONL row at {texts_file}:{line_number}"
                    ) from error
            else:
                item_id = f"text-{len(values):04d}"
                value = stripped
            if not item_id or not value:
                raise ValueError(f"Empty text input at {texts_file}:{line_number}")
            ids.append(item_id)
            values.append(value)
    if not values:
        raise ValueError(f"No text inputs found in {texts_file}")
    if len(set(ids)) != len(ids):
        raise ValueError("text input IDs must be unique")
    return ids, values


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="clip-compass")
    commands = parser.add_subparsers(dest="command", required=True)

    data = commands.add_parser("data", help="Download and prepare datasets")
    data_commands = data.add_subparsers(dest="data_command", required=True)
    subset = data_commands.add_parser(
        "prepare-coco-subset",
        help="Prepare COCO images paired with Localized Narrative captions",
    )
    subset.add_argument("--data-root", type=Path, default=Path("data"))
    subset.add_argument("--split", choices=("train", "val"), default="val")
    subset.add_argument("--limit", type=int, default=25, help="Number of unique images")
    subset.add_argument(
        "--annotations-only",
        action="store_true",
        help="Create the manifest without downloading COCO image files",
    )
    subset.add_argument("--force", action="store_true", help="Redownload existing files")

    instances = data_commands.add_parser(
        "prepare-coco-instances",
        help="Align official COCO instance annotations to a prepared manifest",
    )
    instances.add_argument("--manifest", type=Path, required=True)
    instances.add_argument("--data-root", type=Path, default=Path("data"))
    instances.add_argument("--split", choices=("train", "val"), default="val")
    instances.add_argument("--output", type=Path)
    instances.add_argument(
        "--force-download",
        action="store_true",
        help="Redownload the official COCO annotation archive",
    )

    embeddings = commands.add_parser("embeddings", help="Compute and cache CLIP embeddings")
    embedding_commands = embeddings.add_subparsers(dest="embedding_command", required=True)
    encode = embedding_commands.add_parser(
        "encode-images", help="Embed images listed in a prepared dataset manifest"
    )
    encode.add_argument("--manifest", type=Path, required=True)
    encode.add_argument("--data-root", type=Path, default=Path("data"))
    encode.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    encode.add_argument("--cache-root", type=Path)
    encode.add_argument("--output", type=Path)
    encode.add_argument("--model", default="openai/clip-vit-base-patch32")
    encode.add_argument(
        "--revision", default="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    encode.add_argument("--device", default="auto")
    encode.add_argument("--batch-size", type=int, default=16)
    encode.add_argument("--limit", type=int)
    encode.add_argument("--model-cache-dir", type=Path)
    encode.add_argument("--local-files-only", action="store_true")

    encode_text = embedding_commands.add_parser(
        "encode-texts", help="Embed text hypotheses in the shared CLIP space"
    )
    text_source = encode_text.add_mutually_exclusive_group(required=True)
    text_source.add_argument("--text", action="append", dest="texts")
    text_source.add_argument(
        "--texts-file",
        type=Path,
        help="Plain text lines or JSONL objects with id and text fields",
    )
    encode_text.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    encode_text.add_argument("--cache-root", type=Path)
    encode_text.add_argument("--output", type=Path)
    encode_text.add_argument("--model", default="openai/clip-vit-base-patch32")
    encode_text.add_argument(
        "--revision", default="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    encode_text.add_argument("--device", default="auto")
    encode_text.add_argument("--batch-size", type=int, default=32)
    encode_text.add_argument("--model-cache-dir", type=Path)
    encode_text.add_argument("--local-files-only", action="store_true")

    similarity = embedding_commands.add_parser(
        "image-text-similarity",
        help="Compute a validated image-by-text cosine similarity matrix",
    )
    similarity.add_argument("--images", type=Path, required=True)
    similarity.add_argument("--texts", type=Path, required=True)
    similarity.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    similarity.add_argument("--output", type=Path)

    explanations = commands.add_parser(
        "explanations", help="Prepare evidence and generate Mapper-node hypotheses"
    )
    explanation_commands = explanations.add_subparsers(
        dest="explanation_command", required=True
    )
    context = explanation_commands.add_parser(
        "prepare-context",
        help="Select deterministic target and contrast evidence for Mapper nodes",
    )
    context.add_argument("--graph", type=Path, required=True)
    context.add_argument("--embeddings", type=Path, required=True)
    context.add_argument("--annotations", type=Path, required=True)
    context.add_argument("--manifest", type=Path, required=True)
    context.add_argument("--data-root", type=Path, default=Path("data"))
    context.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    context.add_argument("--output", type=Path)
    context.add_argument("--node-id", action="append", dest="node_ids")
    context.add_argument("--representatives", type=int, default=3)
    context.add_argument("--contrasts", type=int, default=3)
    context.add_argument("--caption-limit", type=int, default=2)

    generate = explanation_commands.add_parser(
        "generate", help="Call OpenAI to generate structured node hypotheses"
    )
    generate.add_argument("--context", type=Path, required=True)
    generate.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    generate.add_argument("--output", type=Path)
    generate.add_argument("--model", default="gpt-6-astra")
    generate.add_argument("--max-hypotheses", type=int, default=3)
    generate.add_argument("--image-size", type=int, default=768)
    generate.add_argument(
        "--image-detail", choices=("low", "high", "auto", "original"), default="low"
    )

    prompts = explanation_commands.add_parser(
        "export-prompts", help="Export generated hypothesis prompts for CLIP encoding"
    )
    prompts.add_argument("--explanations", type=Path, required=True)
    prompts.add_argument("--output", type=Path, required=True)

    perturb = commands.add_parser(
        "perturb", help="Verify hypotheses by masking objects and re-embedding images"
    )
    perturb_commands = perturb.add_subparsers(dest="perturb_command", required=True)
    perturb_run = perturb_commands.add_parser(
        "run", help="Mask hypothesis objects and random controls, embed, and judge"
    )
    perturb_run.add_argument("--graph", type=Path, required=True)
    perturb_run.add_argument("--embeddings", type=Path, required=True)
    perturb_run.add_argument("--annotations", type=Path, required=True)
    perturb_run.add_argument("--manifest", type=Path, required=True)
    perturb_run.add_argument("--explanations", type=Path, required=True)
    perturb_run.add_argument("--data-root", type=Path, default=Path("data"))
    perturb_run.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    perturb_run.add_argument("--output-dir", type=Path)
    perturb_run.add_argument("--cache-root", type=Path)
    perturb_run.add_argument("--controls", type=int, default=39)
    perturb_run.add_argument("--seed", type=int, default=0)
    perturb_run.add_argument("--fill", choices=sorted(FILLS), default="mean")
    perturb_run.add_argument(
        "--verified-output",
        type=Path,
        help="Also write only the supported hypotheses as an explanation artifact",
    )
    perturb_run.add_argument("--model", default="openai/clip-vit-base-patch32")
    perturb_run.add_argument(
        "--revision", default="3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268"
    )
    perturb_run.add_argument("--device", default="auto")
    perturb_run.add_argument("--batch-size", type=int, default=16)
    perturb_run.add_argument("--model-cache-dir", type=Path)
    perturb_run.add_argument("--local-files-only", action="store_true")

    lenses = commands.add_parser("lens", help="Compute Mapper lens values")
    lens_commands = lenses.add_subparsers(dest="lens_command", required=True)
    lens = lens_commands.add_parser("compute", help="Compute a lens from an embedding artifact")
    lens.add_argument("--embeddings", type=Path, required=True)
    lens.add_argument("--name", choices=LENS_NAMES, required=True)
    lens.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    lens.add_argument("--output", type=Path)
    lens.add_argument("--neighbors", type=int, default=5)
    lens.add_argument("--block-size", type=int, default=1024)
    lens.add_argument("--no-scale", action="store_true")

    mapper = commands.add_parser("mapper", help="Build, export, and inspect Mapper graphs")
    mapper_commands = mapper.add_subparsers(dest="mapper_command", required=True)
    build = mapper_commands.add_parser(
        "build", help="Build a classical Mapper graph from embeddings and a lens"
    )
    build.add_argument("--embeddings", type=Path, required=True)
    build.add_argument("--lens", type=Path, required=True)
    build.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    build.add_argument("--output", type=Path)
    build.add_argument("--feature-space", choices=("raw", "normalized"), default="normalized")
    build.add_argument("--n-cubes", type=int, default=8)
    build.add_argument("--overlap", type=float, default=0.4)
    build.add_argument("--min-samples", type=int, default=3)
    build.add_argument("--eps", type=float)
    build.add_argument("--metric", choices=("euclidean", "cosine"), default="euclidean")
    build.add_argument("--eps-sample-size", type=int, default=5000)
    build.add_argument("--eps-fallback-quantile", type=float, default=0.9)
    build.add_argument("--seed", type=int, default=42)
    build.add_argument("--min-intersection", type=int, default=1)
    build.add_argument("--n-jobs", type=int, default=1)
    build.add_argument("--keep-duplicate-nodes", action="store_true")
    build.add_argument("--verbose", type=int, choices=(0, 1, 2), default=0)

    export = mapper_commands.add_parser(
        "export", help="Export a validated graph as JSON and normalized CSV tables"
    )
    export.add_argument("--graph", type=Path, required=True)
    export.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    export.add_argument("--output-dir", type=Path)

    visualize = mapper_commands.add_parser(
        "visualize", help="Build a self-contained HTML graph diagnostic report"
    )
    visualize.add_argument("--graph", type=Path, required=True)
    visualize.add_argument("--embeddings", type=Path, required=True)
    visualize.add_argument("--manifest", type=Path, required=True)
    visualize.add_argument("--data-root", type=Path, default=Path("data"))
    visualize.add_argument("--artifacts-root", type=Path, default=Path("artifacts"))
    visualize.add_argument("--output", type=Path)
    visualize.add_argument("--representatives", type=int, default=3)
    visualize.add_argument("--thumbnail-size", type=int, default=180)
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    args = _parser().parse_args(argv)
    if args.command == "data" and args.data_command == "prepare-coco-subset":
        result = prepare_coco_subset(
            data_root=args.data_root,
            split=args.split,
            limit=args.limit,
            download_images=not args.annotations_only,
            force=args.force,
        )
        summary = {
            "split": result.split,
            "images": len(result.records),
            "captions": sum(len(record.captions) for record in result.records),
            "annotation_path": str(result.annotation_path),
            "manifest_path": str(result.manifest_path),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "data" and args.data_command == "prepare-coco-instances":
        subset, output = prepare_coco_instance_subset(
            data_root=args.data_root,
            manifest_path=args.manifest,
            split=args.split,
            output_path=args.output,
            force_download=args.force_download,
        )
        summary = subset.summary()
        summary.update(
            {
                "manifest": str(args.manifest.resolve()),
                "output": str(output.resolve()),
            }
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "embeddings" and args.embedding_command == "encode-images":
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        item_ids, image_paths = _manifest_images(args.manifest, args.data_root, args.limit)
        encoder = HuggingFaceClipEncoder(
            model_id=args.model,
            revision=args.revision,
            device=args.device,
            model_cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
        )
        cache_root = args.cache_root or artifacts.embedding_cache
        batch = embed_images(
            image_paths,
            encoder=encoder,
            cache=EmbeddingCache(cache_root),
            item_ids=item_ids,
            batch_size=args.batch_size,
        )
        output = args.output or (
            artifacts.embeddings
            / f"{args.manifest.stem}-{encoder.spec.fingerprint}.npz"
        )
        batch.save(output)
        summary = batch.metadata()
        summary.update({"device": encoder.device, "output": str(output.resolve())})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "embeddings" and args.embedding_command == "encode-texts":
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        item_ids, texts = _text_inputs(args.texts, args.texts_file)
        encoder = HuggingFaceClipEncoder(
            model_id=args.model,
            revision=args.revision,
            device=args.device,
            model_cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
        )
        cache_root = args.cache_root or artifacts.embedding_cache
        batch = embed_texts(
            texts,
            encoder=encoder,
            cache=EmbeddingCache(cache_root),
            item_ids=item_ids,
            batch_size=args.batch_size,
        )
        source_name = args.texts_file.stem if args.texts_file else "inline-texts"
        output = args.output or (
            artifacts.text_embeddings
            / f"{source_name}-{encoder.text_spec.fingerprint}.npz"
        )
        batch.save(output)
        summary = batch.metadata()
        summary.update({"device": encoder.device, "output": str(output.resolve())})
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if (
        args.command == "embeddings"
        and args.embedding_command == "image-text-similarity"
    ):
        images = EmbeddingBatch.load(args.images)
        texts = EmbeddingBatch.load(args.texts)
        result = image_text_similarity(images, texts)
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output = args.output or (
            artifacts.similarities / f"{args.images.stem}-{args.texts.stem}.npz"
        )
        result.save(output)
        summary = result.metadata()
        summary["output"] = str(output.resolve())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if (
        args.command == "explanations"
        and args.explanation_command == "prepare-context"
    ):
        graph = MapperGraphArtifact.load(args.graph)
        embeddings = EmbeddingBatch.load(args.embeddings)
        annotations = CocoInstanceSubset.load(args.annotations)
        bundle = build_explanation_contexts(
            graph,
            embeddings,
            annotations,
            graph_path=args.graph,
            annotations_path=args.annotations,
            manifest_path=args.manifest,
            data_root=args.data_root,
            node_ids=tuple(args.node_ids) if args.node_ids else None,
            representatives=args.representatives,
            contrasts=args.contrasts,
            caption_limit=args.caption_limit,
        )
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output = args.output or (
            artifacts.explanation_contexts / f"{args.graph.stem}-context.json"
        )
        bundle.save(output)
        summary = {
            "context_count": len(bundle.contexts),
            "node_ids": [context.node_id for context in bundle.contexts],
            "output": str(output.resolve()),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "explanations" and args.explanation_command == "generate":
        from clip_compass.explanations.openai_agent import OpenAIExplanationAgent

        contexts = ExplanationContextBundle.load(args.context)
        agent = OpenAIExplanationAgent(
            model=args.model,
            image_size=args.image_size,
            image_detail=args.image_detail,
        )
        artifact = generate_explanations(
            contexts, agent, max_hypotheses=args.max_hypotheses
        )
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output = args.output or (
            artifacts.explanations / f"{args.context.stem}-{args.model}.json"
        )
        artifact.save(output)
        summary = {
            "node_count": len(artifact.explanations),
            "hypothesis_count": sum(
                len(explanation.hypotheses)
                for explanation in artifact.explanations
            ),
            "model": args.model,
            "output": str(output.resolve()),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if (
        args.command == "explanations"
        and args.explanation_command == "export-prompts"
    ):
        artifact = ExplanationArtifact.load(args.explanations)
        write_hypothesis_prompts(artifact, args.output)
        summary = {
            "explanation_nodes": len(artifact.explanations),
            "output": str(args.output.resolve()),
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "perturb" and args.perturb_command == "run":
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output_dir = args.output_dir or (artifacts.perturbations / args.explanations.stem)
        encoder = HuggingFaceClipEncoder(
            model_id=args.model,
            revision=args.revision,
            device=args.device,
            model_cache_dir=args.model_cache_dir,
            local_files_only=args.local_files_only,
        )
        results = verify_hypotheses(
            graph_path=args.graph,
            embeddings_path=args.embeddings,
            annotations_path=args.annotations,
            explanations_path=args.explanations,
            manifest_path=args.manifest,
            data_root=args.data_root,
            encoder=encoder,
            cache=EmbeddingCache(args.cache_root or artifacts.embedding_cache),
            output_dir=output_dir,
            config=PerturbationConfig(controls=args.controls, seed=args.seed, fill=args.fill),
            batch_size=args.batch_size,
        )
        print(format_summary(results))
        summary = {
            "min_identity_cosine": results["min_identity_cosine"],
            "output": str((output_dir / "verification.json").resolve()),
        }
        if args.verified_output:
            verified = verified_explanations(results, args.explanations)
            if verified is None:
                summary["verified_output"] = "none written: no hypothesis was supported"
            else:
                verified.save(args.verified_output)
                summary["verified_output"] = str(args.verified_output.resolve())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "lens" and args.lens_command == "compute":
        embeddings = EmbeddingBatch.load(args.embeddings)
        result = compute_lens(
            args.name,
            embeddings,
            scale=not args.no_scale,
            neighbors=args.neighbors,
            block_size=args.block_size,
        )
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output = args.output or (
            artifacts.lenses / f"{args.embeddings.stem}-{args.name}.npz"
        )
        result.save(output, embeddings.item_ids)
        summary = result.metadata()
        summary["output"] = str(output.resolve())
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "mapper" and args.mapper_command == "build":
        embeddings = EmbeddingBatch.load(args.embeddings)
        lens, lens_item_ids = LensResult.load(args.lens)
        config = MapperConfig(
            n_cubes=args.n_cubes,
            perc_overlap=args.overlap,
            min_samples=args.min_samples,
            eps=args.eps,
            metric=args.metric,
            feature_space=args.feature_space,
            eps_sample_size=args.eps_sample_size,
            eps_fallback_quantile=args.eps_fallback_quantile,
            random_seed=args.seed,
            min_intersection=args.min_intersection,
            remove_duplicate_nodes=not args.keep_duplicate_nodes,
            n_jobs=args.n_jobs,
            verbose=args.verbose,
        )
        graph = build_mapper_graph(
            embeddings,
            lens,
            lens_item_ids,
            config=config,
        )
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output = args.output or (
            artifacts.mapper_graphs
            / f"{args.embeddings.stem}-{lens.name}-mapper.json"
        )
        graph.save(output)
        summary = dict(graph.diagnostics)
        summary.update(
            {
                "eps": graph.eps_calibration["eps"],
                "eps_method": graph.eps_calibration["method"],
                "lens": lens.name,
                "output": str(output.resolve()),
            }
        )
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "mapper" and args.mapper_command == "export":
        graph = MapperGraphArtifact.load(args.graph)
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output_dir = args.output_dir or (artifacts.mapper_exports / args.graph.stem)
        paths = graph.export_bundle(output_dir)
        summary = {
            "output_dir": str(output_dir.resolve()),
            "files": {key: str(path.resolve()) for key, path in paths.items()},
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    if args.command == "mapper" and args.mapper_command == "visualize":
        graph = MapperGraphArtifact.load(args.graph)
        embeddings = EmbeddingBatch.load(args.embeddings)
        artifacts = ArtifactPaths.from_root(args.artifacts_root)
        output = args.output or (artifacts.diagnostics / f"{args.graph.stem}.html")
        write_mapper_diagnostic_report(
            graph,
            embeddings,
            manifest_path=args.manifest,
            data_root=args.data_root,
            output_path=output,
            representatives=args.representatives,
            thumbnail_size=args.thumbnail_size,
        )
        summary = {
            "graph": str(args.graph.resolve()),
            "output": str(output.resolve()),
            "representatives_per_node": args.representatives,
        }
        print(json.dumps(summary, indent=2, sort_keys=True))
        return 0
    raise AssertionError("unhandled command")


if __name__ == "__main__":
    raise SystemExit(main())
