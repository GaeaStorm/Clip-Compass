"""Command-line entry point for Clip Compass."""

import argparse
import json
from pathlib import Path
from typing import List, Optional, Sequence, Tuple

from clip_compass.config import ArtifactPaths
from clip_compass.data.localized_narratives import prepare_coco_subset
from clip_compass.embeddings.cache import EmbeddingCache
from clip_compass.embeddings.huggingface import HuggingFaceClipEncoder
from clip_compass.embeddings.service import embed_images
from clip_compass.embeddings.types import EmbeddingBatch
from clip_compass.mapper.builder import MapperConfig, build_mapper_graph
from clip_compass.mapper.diagnostics import write_mapper_diagnostic_report
from clip_compass.mapper.graph import MapperGraphArtifact
from clip_compass.mapper.lenses import LENS_NAMES, LensResult, compute_lens


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
