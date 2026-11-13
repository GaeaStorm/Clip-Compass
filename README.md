# Clip Compass

Clip Compass expands on [Explainable Mapper](https://arxiv.org/html/2507.18607v1)
by exploring CLIP image embedding spaces and testing whether visual perturbations
can verify explanations of how images are grouped.

**Visual walkthrough:** download
[`docs/clip-compass.html`](docs/clip-compass.html) and open it in a browser. It
walks through every pipeline stage with the 25-photo test run's real photos,
groups, masked images and verdicts, and shows who owns each stage. It is a
snapshot; to refresh it after a new run:

```bash
.venv/bin/python scripts/build_explorer.py
cp artifacts/diagnostics/explorer/explorer.html docs/clip-compass.html
```

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
.venv/bin/python -m pip install -e '.[agent,clip,coco,mapper]'
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

### CLIP text embeddings and similarity

Text hypotheses use the pinned CLIP model's text projection and a separate
tokenizer fingerprint, while compatibility checks ensure image and text features
come from the same provider, model revision, and embedding dimension.

```bash
.venv/bin/clip-compass embeddings encode-texts \
  --text "a photo of a person" \
  --text "a photo of food on a plate" \
  --model-cache-dir artifacts/models/huggingface

.venv/bin/clip-compass embeddings image-text-similarity \
  --images artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --texts artifacts/text-embeddings/inline-texts-34a0934385bb8616310e.npz
```

`encode-texts` also accepts `--texts-file`. Each non-empty line can be plain text
or a JSON object with `id` and `text` fields. Text artifacts retain the original
strings, raw and normalized projections, exact-input hashes, cache status, and
model metadata. Similarity artifacts store the complete image-by-text cosine
matrix together with aligned IDs and text values.

## COCO instance annotations

Prepare the official COCO categories, bounding boxes, polygons, and compressed
RLE segmentations for the images already selected by the Localized Narratives
manifest:

```bash
.venv/bin/clip-compass data prepare-coco-instances \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --split val
```

The command downloads and extracts the official 2017 annotation archive only
when needed, then writes a compact, manifest-aligned artifact under
`data/processed/coco_instances/`. It records source and manifest hashes, preserves
manifest image order, and validates all image/category references. The
`instance_mask` API decodes both polygon and RLE segmentations through the COCO
reference implementation into boolean arrays at the source image dimensions.

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

## Explanation agent

Prepare deterministic evidence before making any model request:

```bash
.venv/bin/clip-compass explanations prepare-context \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --annotations data/processed/coco_instances/coco_val_first_25-instances.json \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --node-id node-0000 \
  --output artifacts/explanation-contexts/node-0000-context.json
```

The context contains centroid-nearest target images, hard contrasts chosen from
adjacent Mapper nodes before global fallbacks, captions, COCO labels, category
frequencies, and source hashes. Inspect this file before using the API.

After setting `OPENAI_API_KEY` in the shell environment, generate structured
hypotheses for the selected node:

```bash
.venv/bin/clip-compass explanations generate \
  --context artifacts/explanation-contexts/node-0000-context.json \
  --model gpt-6-astra \
  --max-hypotheses 3 \
  --output artifacts/explanations/node-0000.json
```

The agent sends the selected target and contrast images at low detail, requests
strict structured output, checks all cited item IDs and COCO categories against
the supplied evidence, and sets API storage to false. It produces candidate
hypotheses, not verified conclusions.

Export the positive and negative CLIP prompts for the next scoring stage:

```bash
.venv/bin/clip-compass explanations export-prompts \
  --explanations artifacts/explanations/node-0000.json \
  --output artifacts/explanations/node-0000-prompts.jsonl
```

## Perturbation verification

Test each hypothesis by masking the COCO objects it names, re-embedding the
images with CLIP, and comparing the shift with area-matched random masks:

```bash
.venv/bin/clip-compass perturb run \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --annotations data/processed/coco_instances/coco_val_first_25-instances.json \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --model-cache-dir artifacts/models/huggingface \
  --verified-output artifacts/explanations/coco_val_first_25-verified.json
```

For every node member containing the named categories, the run fills the
(3 px dilated) segmentation masks with CLIP's mean colour and writes a lossless
PNG. It also writes 39 controls: the same visible mask, shifted to random
positions inside the region CLIP's centre crop actually sees. Seeds derive from
the image and categories, so runs are deterministic, and the perturbed images go
through the normal embedding cache. Objects covering less than 1% of the visible
crop, or too large to leave room for controls, are recorded but not tested.

Each image is scored against its own controls on two statistics: the increase
in distance from the node centroid (computed without the image under test), and
the drop in similarity to the positive prompt. A hypothesis is `supported` when a strict majority of testable members
beat their controls (empirical p <= 0.05) on centroid distance, and
`detected_not_explanatory` when only the prompt test passes. It is
`not_supported` when neither passes or the categories are visible in fewer than
half the members, and `untestable` (with a reason) otherwise.
`prediction_confirmed` records whether the explanation agent's own
`predicted_effect` came true. `--verified-output` writes only the supported
hypotheses as an explanation artifact, for comparison with the unfiltered set.

## Hypothesis verdicts and the text baseline

`perturb run` judges each image on its own and then takes a majority vote.
`verification verdicts` reads the same per-image results and gives each
hypothesis one calibrated verdict instead:

```bash
.venv/bin/clip-compass verification verdicts \
  --results artifacts/perturbations/fixture-dk/verification.json \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json
```

For every testable image it counts how many of the 39 controls the object edit
beat on each statistic. Under the null hypothesis the object is just another
random region, so that rank is uniform on 0..39 in each image. The summed rank
across a hypothesis's images then has an exact null distribution (a convolution
of uniforms), which gives one p-value per hypothesis and statistic.
Benjamini-Hochberg corrects those p-values across hypotheses within each
statistic. Consistent moderate effects therefore add up: five images that each
beat 32 of 39 controls are significant together (p < 0.01), even though none
would pass a per-image threshold. Two images that beat every control give
p = 1/1600.

Labels follow the engine's scheme: `supported` when the centroid test passes
after correction, `detected_not_explanatory` when only the prompt test passes,
`not_supported` when neither does or when the categories are visible in fewer
than half the members, `insufficient_evidence` with fewer than two testable
images, and `untestable` for empty or unknown categories. The output directory
(`artifacts/verification/<run>/`) holds `verdicts.json`, `hypotheses.csv` (one
row per hypothesis, with the mean share of controls beaten, p-values and
q-values) and `images.csv` (one row per image).

The text-only comparison arm asks whether the hypothesis prompt scores the
node's members above every other image, with no perturbation at all:

```bash
.venv/bin/clip-compass verification baseline \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --prompt-embeddings artifacts/text-embeddings/fixture-prompts.npz
```

It reports the inside and outside mean similarity and the AUROC (how often a
member outscores a non-member), and accepts a hypothesis when members score
higher on average. Build the prompt embeddings with `explanations
export-prompts` followed by `embeddings encode-texts`.

## Judging the judges: control hypotheses

Disagreement between the text check and the masking test does not show which
one is right. `verification controls` writes hypotheses whose right answer is
known from COCO labels, in the same format the agent produces, so they run
through the unchanged pipeline:

- `obvious` (should be accepted): a category present in at least half the
  node's photos and at least twice as common inside the node as outside;
- `common` (should be rejected): present in at least half the node's photos but
  no more than 1.25 times as common inside as outside;
- `swapped` (should be rejected): another node's real hypothesis whose objects
  are not enriched in this node, preferring objects the node's photos do contain.

A category counts as present when one of its instances covers at least 1% of
the image. The command also writes an answer key tied to the exact controls
file:

```bash
.venv/bin/clip-compass verification controls \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --annotations data/processed/coco_instances/coco_val_first_25-instances.json \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --output artifacts/explanations/fixture-controls.json
```

Run the controls through `explanations export-prompts`, `embeddings
encode-texts`, `verification baseline`, `perturb run` and `verification
verdicts` exactly as for real hypotheses, then score every judge:

```bash
.venv/bin/clip-compass verification evaluate \
  --key artifacts/explanations/fixture-controls-key.json \
  --verdicts artifacts/verification/fixture-controls/verdicts.json \
  --baseline artifacts/verification/fixture-controls-baseline/baseline.json \
  --output-dir artifacts/verification/fixture-controls-evaluation
```

For each judge (the agent, which accepts everything; the text check; the
masking test; and a lenient masking variant that also accepts
`detected_not_explanatory`) it reports the hit rate on controls that should be
accepted, the false-alarm rate on controls that should be rejected, accuracy,
and how many controls the judge could not decide. Undecided controls count as
not accepted, so `*_decided` rates that leave them out are reported too.
`judge_scores.csv` breaks the results down by control kind and `controls.csv`
lists every control with each judge's decision.

See [the manual pipeline test](docs/manual-pipeline-test.md) for a one-node smoke
test, CLIP discrimination check, perturbation verification, and complete rebuild
sequence.
