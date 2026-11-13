# Manual pipeline test

Run commands from the repository root. The quick path uses the existing 25-image
artifacts and makes only one explanation request.

## 1. Install and verify

```bash
.venv/bin/python -m pip install -e '.[agent,clip,coco,mapper]'
LOKY_MAX_CPU_COUNT=8 .venv/bin/python -m unittest discover -s tests -v
.venv/bin/python -m pip check
```

The suite should report 59 passing tests. `LOKY_MAX_CPU_COUNT` only prevents a
macOS/joblib CPU-detection warning in restricted environments.

## 2. Inspect the Mapper report

Open this file in a browser:

```text
artifacts/diagnostics/coco_val_first_25-pca1_normalized-mapper.html
```

Confirm that selecting nodes changes the representative images and node metadata.
The current smoke graph should show 5 nodes, 3 edges, 2 components, and 88% image
coverage.

## 3. Rebuild and inspect one explanation context

```bash
.venv/bin/clip-compass explanations prepare-context \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --annotations data/processed/coco_instances/coco_val_first_25-instances.json \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --node-id node-0000 \
  --representatives 3 \
  --contrasts 3 \
  --output artifacts/explanation-contexts/node-0000-context.json
```

Open the output JSON. Check that:

- every target image belongs to `node-0000`;
- every contrast image is outside that node;
- image paths exist;
- captions and COCO categories match the images;
- target and contrast category counts look plausible.

This command is deterministic and does not call OpenAI.

## 4. Generate hypotheses for one node

Set `OPENAI_API_KEY` in the current shell without adding it to a repository file.
The following command sends the six selected images, captions, COCO labels, and
node metadata to the OpenAI Responses API and can incur API charges:

```bash
.venv/bin/clip-compass explanations generate \
  --context artifacts/explanation-contexts/node-0000-context.json \
  --model gpt-6-astra \
  --max-hypotheses 3 \
  --image-detail low \
  --output artifacts/explanations/node-0000.json
```

If that model is not available to your API project, pass another vision-capable
model that supports Structured Outputs. Inspect the output and confirm that each
hypothesis has:

- a concise positive and negative CLIP prompt;
- only COCO categories present in the context;
- supporting IDs from the target examples;
- counterexample IDs from the contrast examples;
- a falsifiable predicted perturbation effect;
- confidence between 0 and 1.

The API request uses `store=False`. A hypothesis is still unverified at this
stage, regardless of its confidence.

## 5. Encode and score the generated prompts

```bash
.venv/bin/clip-compass explanations export-prompts \
  --explanations artifacts/explanations/node-0000.json \
  --output artifacts/explanations/node-0000-prompts.jsonl

.venv/bin/clip-compass embeddings encode-texts \
  --texts-file artifacts/explanations/node-0000-prompts.jsonl \
  --model-cache-dir artifacts/models/huggingface \
  --output artifacts/text-embeddings/node-0000-prompts.npz

.venv/bin/clip-compass embeddings image-text-similarity \
  --images artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --texts artifacts/text-embeddings/node-0000-prompts.npz \
  --output artifacts/similarities/node-0000-prompts.npz
```

Compare mean similarity inside the node against images outside it:

```bash
.venv/bin/python - <<'PY'
from pathlib import Path
import numpy as np
from clip_compass.embeddings.similarity import ClipSimilarityMatrix
from clip_compass.mapper.graph import MapperGraphArtifact

graph = MapperGraphArtifact.load(
    Path("artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json")
)
scores = ClipSimilarityMatrix.load(
    Path("artifacts/similarities/node-0000-prompts.npz")
)
node = next(node for node in graph.nodes if node.id == "node-0000")
inside = np.asarray(node.member_indices)
outside = np.asarray(
    [index for index in range(len(graph.item_ids)) if index not in set(inside)]
)
for column, (text_id, text) in enumerate(zip(scores.text_ids, scores.texts)):
    inside_mean = float(scores.scores[inside, column].mean())
    outside_mean = float(scores.scores[outside, column].mean())
    print(text_id)
    print("  prompt:", text)
    print("  inside:", round(inside_mean, 4))
    print("  outside:", round(outside_mean, 4))
    print("  margin:", round(inside_mean - outside_mean, 4))
PY
```

A useful positive hypothesis should generally have a positive inside-minus-outside
margin. This smoke check is descriptive, not the final statistical evaluation.

## 6. Verify hypotheses with image perturbations

This step uses the committed offline fixture, so it makes no API call:

```bash
.venv/bin/clip-compass perturb run \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --annotations data/processed/coco_instances/coco_val_first_25-instances.json \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --model-cache-dir artifacts/models/huggingface \
  --local-files-only \
  --verified-output artifacts/explanations/coco_val_first_25-verified.json
```

Check that `min_identity_cosine` is at least 0.999. The run refuses to continue
below that, because the unmasked PNGs would no longer match the stored
embeddings. The printed table should show this pattern for the current fixture:

| Hypothesis | Verdict | Why |
|---|---|---|
| node-0000 broccoli | `detected_not_explanatory` | prompt drop beats controls on 3/4 images; centroid distance on 0/4 |
| node-0001 bottle+fork | `not_supported: no_exceedance` | 2 testable images, neither beats its controls |
| node-0002 couch+teddy bear | `not_supported: low_coverage` | visible in 2 of 5 members |
| node-0003 / node-0004 person | `not_supported: low_coverage` | visible in 2 of 13 and 3 of 10 members |

No hypothesis is supported, so no verified file is written.

Open `artifacts/perturbations/coco_val_first_25-all-nodes-offline-fixture/images/551660/`.
Check that `*__object.png` greys out the annotated broccoli, and that the
`*__ctrl-s0-*.png` controls are grey patches of the same shape inside the
central 424x424 band that CLIP's centre crop keeps. COCO annotates only five
broccoli pieces in this image, so most of the broccoli in the bowl stays
visible. Run the command again: every image should be a cache hit, and
`shasum verification.json` should not change.

## 7. Turn per-image results into verdicts, and score the text baseline

```bash
.venv/bin/clip-compass verification verdicts \
  --results artifacts/perturbations/coco_val_first_25-all-nodes-offline-fixture/verification.json \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --output-dir artifacts/verification/fixture

.venv/bin/clip-compass explanations export-prompts \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --output artifacts/explanations/fixture-prompts.jsonl

.venv/bin/clip-compass embeddings encode-texts \
  --texts-file artifacts/explanations/fixture-prompts.jsonl \
  --model-cache-dir artifacts/models/huggingface \
  --output artifacts/text-embeddings/fixture-prompts.npz

.venv/bin/clip-compass verification baseline \
  --graph artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json \
  --embeddings artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --prompt-embeddings artifacts/text-embeddings/fixture-prompts.npz \
  --output-dir artifacts/verification/fixture-baseline
```

Open `artifacts/verification/fixture/hypotheses.csv`. With the current fixture,
broccoli should be `supported`: its object edits beat about 82% of controls on
centroid distance across all four testable images (q < 0.05), even though no
single image beats 38 of 39 controls, so the engine's majority vote above says
`detected_not_explanatory`. Bottle+fork should be `not_supported: no_effect`,
and the remaining three `not_supported: low_coverage`. Every hypothesis row in
`perturb run`'s output should appear once, and `images.csv` should have one row
per node member and hypothesis. The baseline should accept all five hypotheses.

## 8. Score the judges on control hypotheses

```bash
C=.venv/bin/clip-compass
G=artifacts/mapper/coco_val_first_25-pca1_normalized-mapper.json
E=artifacts/embeddings/coco_val_first_25-clip-vit-b32.npz
A=data/processed/coco_instances/coco_val_first_25-instances.json
X=artifacts/explanations/fixture-controls.json

$C verification controls --graph $G --annotations $A \
  --explanations artifacts/explanations/coco_val_first_25-all-nodes-offline-fixture.json \
  --output $X
$C explanations export-prompts --explanations $X \
  --output artifacts/explanations/fixture-controls-prompts.jsonl
$C embeddings encode-texts \
  --texts-file artifacts/explanations/fixture-controls-prompts.jsonl \
  --model-cache-dir artifacts/models/huggingface \
  --output artifacts/text-embeddings/fixture-controls-prompts.npz
$C verification baseline --graph $G --embeddings $E --explanations $X \
  --prompt-embeddings artifacts/text-embeddings/fixture-controls-prompts.npz \
  --output-dir artifacts/verification/fixture-controls-baseline
$C perturb run --graph $G --embeddings $E --annotations $A \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --explanations $X --data-root data \
  --output-dir artifacts/perturbations/fixture-controls \
  --model-cache-dir artifacts/models/huggingface --local-files-only
$C verification verdicts \
  --results artifacts/perturbations/fixture-controls/verification.json \
  --explanations $X --output-dir artifacts/verification/fixture-controls
$C verification evaluate \
  --key artifacts/explanations/fixture-controls-key.json \
  --verdicts artifacts/verification/fixture-controls/verdicts.json \
  --baseline artifacts/verification/fixture-controls-baseline/baseline.json \
  --output-dir artifacts/verification/fixture-controls-evaluation
```

With 25 photos expect only seven controls: two `obvious` (broccoli for
node-0000 and node-0001) and five `swapped`; no node has a category common
enough to make a `common` control. The text check should accept both obvious
controls and four of the five swapped ones. The masking test should reject all
five swapped controls and support one obvious control (the other is
`detected_not_explanatory`, which the lenient variant counts). Seven controls
cannot support conclusions; this step only checks that the evaluation runs.

## 9. Rebuild the complete local pipeline

Use explicit output paths so subsequent commands are copy-pasteable:

```bash
.venv/bin/clip-compass data prepare-coco-subset \
  --split val --limit 25 --data-root data

.venv/bin/clip-compass data prepare-coco-instances \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data --split val

.venv/bin/clip-compass embeddings encode-images \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --model-cache-dir artifacts/models/huggingface \
  --output artifacts/embeddings/manual-coco25.npz

.venv/bin/clip-compass lens compute \
  --embeddings artifacts/embeddings/manual-coco25.npz \
  --name pca1_normalized \
  --output artifacts/lenses/manual-coco25-pca1.npz

.venv/bin/clip-compass mapper build \
  --embeddings artifacts/embeddings/manual-coco25.npz \
  --lens artifacts/lenses/manual-coco25-pca1.npz \
  --n-cubes 8 --overlap 0.4 --min-samples 3 \
  --output artifacts/mapper/manual-coco25.json

.venv/bin/clip-compass mapper visualize \
  --graph artifacts/mapper/manual-coco25.json \
  --embeddings artifacts/embeddings/manual-coco25.npz \
  --manifest data/processed/coco_localized_narratives/coco_val_first_25.jsonl \
  --data-root data \
  --output artifacts/diagnostics/manual-coco25.html
```

Then repeat steps 3-7 using `manual-coco25.json` and
`manual-coco25.npz`. Exact topology can change if Mapper parameters change; a
different node count is not automatically a failure.
