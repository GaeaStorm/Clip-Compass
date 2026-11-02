# Clip Compass

Clip Compass expands on [Explainable Mapper](https://arxiv.org/html/2507.18607v1)
by exploring CLIP image embedding spaces and testing whether visual perturbations
can verify explanations of how images are grouped.

## Current pipeline

The first implemented stage prepares a deterministic subset of COCO images paired
with textual captions from
[Localized Narratives](https://google.github.io/localized-narratives/). Dataset
files are written under `data/`, which is intentionally ignored by Git.

```bash
PYTHONPATH=src python3 -m clip_compass.cli data prepare-coco-subset \
  --split val \
  --limit 25
```

To download and parse the annotations without fetching image files:

```bash
PYTHONPATH=src python3 -m clip_compass.cli data prepare-coco-subset \
  --split val \
  --limit 25 \
  --annotations-only
```

The command creates:

- the source caption JSONL under `data/raw/localized_narratives/`;
- selected COCO images under `data/raw/coco/images/`; and
- a portable JSONL manifest under `data/processed/coco_localized_narratives/`.

Each manifest row contains the COCO image ID, relative image path, all available
Localized Narrative captions for the selected image, and their annotator IDs.

## Development

The initial loader has no third-party runtime dependencies. Run its tests with:

```bash
PYTHONPATH=src python3 -m unittest discover -s tests -v
```

Localized Narratives annotations are distributed under CC BY 4.0. Please retain
attribution to Pont-Tuset et al., *Connecting Vision and Language with Localized
Narratives* (ECCV 2020), when redistributing annotations or derived datasets.

## CLIP embeddings

Install the model runtime into a local virtual environment:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install --upgrade pip setuptools wheel
.venv/bin/python -m pip install -e '.[clip,mapper]'
```

Embed every image in a prepared manifest:

```bash
.venv/bin/clip-compass embeddings encode-images \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --model-cache-dir artifacts/models/huggingface
```

The default model is `openai/clip-vit-base-patch32`, pinned to Hugging Face
revision `3d74acf9a28c67741b2f4f2ea7635f0aaf6f0268`. The resulting artifact stores
both raw projection-layer features and unit-normalized features. Individual raw
vectors are cached by the SHA-256 digest of the image bytes and a model/preprocessor
fingerprint. Changing either the input or model configuration produces a cache miss.

## Mapper lenses

Compute a lens from an embedding artifact:

```bash
.venv/bin/clip-compass lens compute \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --name l2_raw
```

Available lenses are:

- `l2_raw`, using unnormalized CLIP projection features;
- `pca1_raw` and `pca1_normalized`;
- `eccentricity_normalized`, the mean distance to the full point cloud;
- `centroid_distance_normalized`; and
- `density_normalized`, the mean distance to the nearest neighbors.

Lens values are min-max scaled by default for use with a Mapper cover. Use
`--no-scale` to retain the original numerical range. L2 norm intentionally uses
raw features: every normalized CLIP vector has approximately unit norm and would
therefore produce a degenerate lens.

## Mapper graph

Build the classical Mapper graph from an embedding artifact and an aligned lens:

```bash
.venv/bin/clip-compass mapper build \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --lens artifacts/lenses/coco_val_first_25-clip-vit-b32-pca1_normalized.npz \
  --n-cubes 8 \
  --overlap 0.4 \
  --min-samples 3
```

The builder clusters normalized CLIP features with DBSCAN inside each overlapping
lens interval. If `--eps` is omitted, epsilon is selected from a deterministic
k-distance knee with a configured quantile fallback. Saved JSON graphs contain
canonical node IDs, image memberships, overlap edges and Jaccard scores, connected
components, all construction parameters, calibration diagnostics, and coverage
statistics. Pass an explicit `--eps` when reproducing a previously calibrated run.

### Portable graph export

Every graph is validated when it is constructed, loaded, or saved. Validation
checks canonical IDs and ordering, memberships, edge intersections and Jaccard
scores, connected components, lens statistics, and the derived diagnostic counts.

Export a graph for analysis in notebooks or graph tools:

```bash
.venv/bin/clip-compass mapper export \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json
```

The export directory contains the validated `graph.json` plus `nodes.csv`,
`edges.csv`, `memberships.csv`, `components.csv`, and a compact `summary.json`.
All table rows use stable node, component, item, and member-index identifiers.

### Diagnostic visualization

Generate an offline report with graph topology, calibration and coverage summary,
interactive node selection, centroid-nearest representative images and captions,
and a list of images left unassigned by DBSCAN:

```bash
.venv/bin/clip-compass mapper visualize \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --representatives 3
```

The report is self-contained: thumbnails and graph data are embedded in the HTML,
so it can be opened without a server or network connection. Generation refuses
misaligned graph, embedding, model-fingerprint, or manifest inputs rather than
silently pairing the wrong images with nodes.
