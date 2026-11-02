"""Self-contained diagnostic report generation for Mapper graph artifacts."""

from __future__ import annotations

import base64
import html
import io
import json
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Mapping, Sequence, Tuple

import numpy as np

from clip_compass.embeddings.types import EmbeddingBatch

from .graph import MapperGraphArtifact, MapperNode


@dataclass(frozen=True)
class ManifestImage:
    item_id: str
    image_path: Path
    captions: Tuple[str, ...]


def load_manifest_images(manifest_path: Path, data_root: Path) -> Mapping[str, ManifestImage]:
    """Load image paths and captions from the prepared Localized Narratives manifest."""

    resolved_root = data_root.expanduser().resolve()
    records: Dict[str, ManifestImage] = {}
    with manifest_path.open("r", encoding="utf-8") as manifest:
        for line_number, line in enumerate(manifest, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
                item_id = str(row["image_id"])
                relative_path = Path(str(row["image_path"]))
                raw_captions = row["captions"]
            except (json.JSONDecodeError, KeyError, TypeError) as error:
                raise ValueError(
                    f"Invalid image manifest row at {manifest_path}:{line_number}"
                ) from error
            if relative_path.is_absolute():
                raise ValueError(f"Manifest image path must be relative: {relative_path}")
            image_path = (resolved_root / relative_path).resolve()
            try:
                image_path.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError(
                    f"Manifest image path escapes the data root: {relative_path}"
                ) from error
            if not isinstance(raw_captions, list) or not all(
                isinstance(caption, str) and caption.strip() for caption in raw_captions
            ):
                raise ValueError(
                    f"Manifest captions must be non-empty strings at line {line_number}"
                )
            if item_id in records:
                raise ValueError(f"Duplicate image ID in manifest: {item_id}")
            records[item_id] = ManifestImage(
                item_id=item_id,
                image_path=image_path,
                captions=tuple(caption.strip() for caption in raw_captions),
            )
    if not records:
        raise ValueError(f"No image records found in {manifest_path}")
    return records


def _thumbnail_data_uri(path: Path, size: int) -> str:
    try:
        from PIL import Image, ImageOps
    except ImportError as error:
        raise RuntimeError(
            "Diagnostic thumbnails require Pillow: pip install -e '.[clip]'"
        ) from error

    if not path.is_file():
        raise FileNotFoundError(f"Diagnostic image does not exist: {path}")
    with Image.open(path) as image:
        image = ImageOps.exif_transpose(image).convert("RGB")
        image.thumbnail((size, size), Image.Resampling.LANCZOS)
        output = io.BytesIO()
        image.save(output, format="JPEG", quality=78, optimize=True)
    encoded = base64.b64encode(output.getvalue()).decode("ascii")
    return f"data:image/jpeg;base64,{encoded}"


def _representative_indices(
    node: MapperNode, features: np.ndarray, count: int
) -> Tuple[int, ...]:
    members = np.asarray(node.member_indices, dtype=np.int64)
    member_features = features[members]
    centroid = member_features.mean(axis=0)
    distances = np.linalg.norm(member_features - centroid, axis=1)
    order = np.lexsort((members, distances))
    return tuple(int(members[index]) for index in order[:count])


def _graph_layout(graph: MapperGraphArtifact) -> Tuple[Mapping[str, Tuple[float, float]], int]:
    width = 1180
    left = 120.0
    right = width - 50.0
    component_gap = 150.0
    top = 100.0
    lens_values = [node.lens_mean for node in graph.nodes]
    lens_min = min(lens_values, default=0.0)
    lens_max = max(lens_values, default=1.0)
    span = max(lens_max - lens_min, np.finfo(np.float64).eps)
    positions: Dict[str, Tuple[float, float]] = {}
    nodes_by_id = {node.id: node for node in graph.nodes}
    for component_index, component in enumerate(graph.components):
        ordered = sorted(component, key=lambda node_id: (nodes_by_id[node_id].lens_mean, node_id))
        center_y = top + component_index * component_gap
        for index, node_id in enumerate(ordered):
            node = nodes_by_id[node_id]
            x = left + ((node.lens_mean - lens_min) / span) * (right - left)
            offset = 0.0 if len(ordered) == 1 else ((index % 3) - 1) * 24.0
            positions[node_id] = (x, center_y + offset)
    height = max(300, int(top + max(1, len(graph.components)) * component_gap))
    return positions, height


def _svg_graph(graph: MapperGraphArtifact) -> str:
    positions, height = _graph_layout(graph)
    sizes = [len(node.member_indices) for node in graph.nodes]
    size_min = min(sizes, default=1)
    size_max = max(sizes, default=1)
    size_span = max(size_max - size_min, 1)
    component_index = {
        node_id: index
        for index, component in enumerate(graph.components)
        for node_id in component
    }
    lines: List[str] = [
        f'<svg class="mapper-graph" viewBox="0 0 1180 {height}" role="img" '
        'aria-labelledby="graph-title graph-description">',
        '<title id="graph-title">Mapper graph by lens position</title>',
        '<desc id="graph-description">Nodes are positioned horizontally by mean lens value and '
        'grouped vertically by connected component. Node size represents membership count.</desc>',
    ]
    for index, component in enumerate(graph.components):
        y = 100.0 + index * 150.0
        lines.append(
            f'<text class="component-label" x="16" y="{y + 4:.1f}">C{index + 1}</text>'
        )
        lines.append(
            f'<line class="component-guide" x1="80" y1="{y:.1f}" x2="1130" y2="{y:.1f}" />'
        )
    for edge in graph.edges:
        x1, y1 = positions[edge.source]
        x2, y2 = positions[edge.target]
        width = 1.5 + 7.0 * edge.jaccard
        lines.append(
            f'<line class="graph-edge" x1="{x1:.2f}" y1="{y1:.2f}" x2="{x2:.2f}" '
            f'y2="{y2:.2f}" style="stroke-width:{width:.2f}px">'
            f'<title>{html.escape(edge.source)} to {html.escape(edge.target)}; '
            f'{len(edge.intersection_indices)} shared; Jaccard {edge.jaccard:.3f}</title></line>'
        )
    for node in graph.nodes:
        x, y = positions[node.id]
        radius = 15.0 + 18.0 * math.sqrt((len(node.member_indices) - size_min) / size_span)
        component = component_index[node.id]
        label = f"N{int(node.id.rsplit('-', 1)[-1])}"
        lines.extend(
            [
                f'<g class="graph-node component-{component % 6}" '
                f'data-node-id="{html.escape(node.id)}" role="button" '
                f'tabindex="0" aria-label="Inspect {html.escape(node.id)}, '
                f'{len(node.member_indices)} items">',
                f'<circle cx="{x:.2f}" cy="{y:.2f}" r="{radius:.2f}" '
                f'style="--component-index:{component}" />',
                f'<text x="{x:.2f}" y="{y + 4:.2f}">{html.escape(label)}</text>',
                f'<title>{html.escape(node.id)}; {len(node.member_indices)} items; '
                f'lens mean {node.lens_mean:.4f}</title>',
                "</g>",
            ]
        )
    lines.extend(
        [
            f'<text class="axis-label" x="120" y="{height - 18}">lower mean lens</text>',
            f'<text class="axis-label axis-label-end" x="1130" y="{height - 18}">'
            "higher mean lens</text>",
            "</svg>",
        ]
    )
    return "\n".join(lines)


def _report_payload(
    graph: MapperGraphArtifact,
    embeddings: EmbeddingBatch,
    manifest: Mapping[str, ManifestImage],
    representatives: int,
    thumbnail_size: int,
) -> Mapping[str, object]:
    features = embeddings.normalized if graph.feature_space == "normalized" else embeddings.raw
    representative_by_node = {
        node.id: _representative_indices(node, features, representatives) for node in graph.nodes
    }
    assigned = {index for node in graph.nodes for index in node.member_indices}
    unassigned = tuple(index for index in range(len(graph.item_ids)) if index not in assigned)
    required_indices = set(unassigned)
    for indices in representative_by_node.values():
        required_indices.update(indices)

    images: Dict[str, object] = {}
    for index in sorted(required_indices):
        item_id = graph.item_ids[index]
        record = manifest[item_id]
        images[item_id] = {
            "item_id": item_id,
            "captions": list(record.captions),
            "thumbnail": _thumbnail_data_uri(record.image_path, thumbnail_size),
        }
    nodes = {}
    for node in graph.nodes:
        nodes[node.id] = {
            "id": node.id,
            "source_id": node.source_id,
            "component_id": node.component_id,
            "size": len(node.member_indices),
            "lens_min": node.lens_min,
            "lens_mean": node.lens_mean,
            "lens_max": node.lens_max,
            "representative_ids": [
                graph.item_ids[index] for index in representative_by_node[node.id]
            ],
            "item_ids": list(node.item_ids),
        }
    return {
        "nodes": nodes,
        "images": images,
        "unassigned_ids": [graph.item_ids[index] for index in unassigned],
    }


def write_mapper_diagnostic_report(
    graph: MapperGraphArtifact,
    embeddings: EmbeddingBatch,
    *,
    manifest_path: Path,
    data_root: Path,
    output_path: Path,
    representatives: int = 3,
    thumbnail_size: int = 180,
) -> Path:
    """Generate an offline HTML report for topology and representative-image inspection."""

    graph.validate()
    if embeddings.item_ids != graph.item_ids:
        raise ValueError("graph and embedding item IDs must match in the same order")
    if embeddings.model_spec.fingerprint != graph.embedding_model_fingerprint:
        raise ValueError("graph and embedding model fingerprints do not match")
    if representatives < 1:
        raise ValueError("representatives must be at least 1")
    if not 64 <= thumbnail_size <= 512:
        raise ValueError("thumbnail_size must be between 64 and 512 pixels")

    manifest = load_manifest_images(manifest_path, data_root)
    missing = [item_id for item_id in graph.item_ids if item_id not in manifest]
    if missing:
        raise ValueError(f"manifest is missing {len(missing)} graph item IDs; first: {missing[0]}")
    payload = _report_payload(graph, embeddings, manifest, representatives, thumbnail_size)
    payload_json = json.dumps(
        payload, sort_keys=True, separators=(",", ":")
    ).replace("<", "\\u003c")
    diagnostics = graph.diagnostics
    eps = float(graph.eps_calibration["eps"])
    lens_name = str(graph.lens.get("name", "lens"))
    initial_node = graph.nodes[0].id if graph.nodes else ""
    document = f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Mapper diagnostics · {html.escape(lens_name)}</title>
<style>
:root {{
  color-scheme:light dark; --bg:#f7f8fb; --panel:#ffffff; --text:#18212f;
  --muted:#667085; --line:#d7dce5; --edge:#7a8493; --focus:#2563eb;
  --c0:#2563eb; --c1:#d97706; --c2:#059669; --c3:#9333ea; --c4:#dc2626; --c5:#0891b2;
}}
@media (prefers-color-scheme: dark) {{ :root {{
  --bg:#10131a; --panel:#181d27; --text:#edf1f7; --muted:#a8b0bf;
  --line:#343b48; --edge:#8b95a5; --focus:#60a5fa;
  --c0:#60a5fa; --c1:#fbbf24; --c2:#34d399; --c3:#c084fc; --c4:#fb7185; --c5:#22d3ee;
}} }}
* {{ box-sizing:border-box; }}
body {{
  margin:0; background:var(--bg); color:var(--text);
  font:15px/1.5 ui-sans-serif,system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif;
}}
main {{ width:min(1440px,100%); margin:0 auto; padding:32px; }}
h1,h2,h3,p {{ margin:0; }}
h1 {{ font-size:28px; font-weight:650; letter-spacing:-.02em; }}
h2 {{ font-size:18px; font-weight:650; }}
h3 {{ font-size:15px; font-weight:650; }}
.subtitle,.muted {{ color:var(--muted); }}
.subtitle {{ margin-top:6px; }}
.stats {{
  display:grid; grid-template-columns:repeat(5,minmax(130px,1fr)); gap:12px; margin:24px 0;
}}
.stat,.panel {{ background:var(--panel); border:1px solid var(--line); border-radius:12px; }}
.stat {{ padding:14px 16px; }}
.stat-label {{
  color:var(--muted); font-size:12px; text-transform:uppercase; letter-spacing:.06em;
}}
.stat-value {{
  margin-top:2px; font-size:22px; font-weight:650; font-variant-numeric:tabular-nums;
}}
.layout {{
  display:grid; grid-template-columns:minmax(0,1.65fr) minmax(300px,.75fr);
  gap:16px; align-items:start;
}}
.panel {{ padding:18px; }}
.panel-heading {{
  display:flex; justify-content:space-between; gap:16px; align-items:baseline; margin-bottom:12px;
}}
.graph-wrap {{ overflow-x:auto; }}
.mapper-graph {{ display:block; width:100%; min-width:720px; height:auto; }}
.component-guide {{ stroke:var(--line); stroke-width:1; }}
.component-label,.axis-label {{ fill:var(--muted); font-size:12px; }}
.axis-label-end {{ text-anchor:end; }}
.graph-edge {{ stroke:var(--edge); opacity:.58; }}
.graph-node {{ outline:none; cursor:pointer; }}
.graph-node circle {{
  fill:var(--c0); stroke:var(--panel); stroke-width:3;
  transition:stroke-width .15s,filter .15s;
}}
.graph-node.component-1 circle {{ fill:var(--c1); }}
.graph-node.component-2 circle {{ fill:var(--c2); }}
.graph-node.component-3 circle {{ fill:var(--c3); }}
.graph-node.component-4 circle {{ fill:var(--c4); }}
.graph-node.component-5 circle {{ fill:var(--c5); }}
.graph-node:hover circle,.graph-node:focus circle,.graph-node.selected circle {{
  stroke:var(--focus); stroke-width:6; filter:brightness(1.08);
}}
.graph-node text {{
  fill:#fff; font-size:11px; font-weight:650; text-anchor:middle; pointer-events:none;
}}
.facts {{ display:grid; grid-template-columns:1fr 1fr; gap:10px 14px; margin:14px 0 18px; }}
.fact dt {{ color:var(--muted); font-size:12px; }}
.fact dd {{ margin:1px 0 0; font-variant-numeric:tabular-nums; overflow-wrap:anywhere; }}
.representatives {{ display:grid; gap:12px; }}
.representative {{
  display:grid; grid-template-columns:84px minmax(0,1fr); gap:12px; align-items:start;
}}
.representative img {{
  width:84px; height:84px; object-fit:cover; border-radius:8px; background:var(--line);
}}
.representative p {{
  color:var(--muted); font-size:13px; display:-webkit-box; -webkit-line-clamp:4;
  -webkit-box-orient:vertical; overflow:hidden;
}}
.item-id {{ font-size:12px; font-variant-numeric:tabular-nums; margin-bottom:3px; }}
.unassigned {{ margin-top:16px; }}
.unassigned-list {{ margin:10px 0 0; padding-left:20px; color:var(--muted); columns:2; }}
code {{ font-family:ui-monospace,SFMono-Regular,Menlo,monospace; font-size:12px; }}
@media (max-width:900px) {{
  main {{ padding:20px; }} .stats {{ grid-template-columns:repeat(2,1fr); }}
  .layout {{ grid-template-columns:1fr; }}
}}
@media (max-width:520px) {{
  main {{ padding:14px; }} .stats {{ grid-template-columns:1fr 1fr; }}
  .stat {{ padding:11px; }} .panel {{ padding:14px; }} .facts {{ grid-template-columns:1fr; }}
}}
</style>
</head>
<body>
<main>
  <header>
    <h1>Mapper diagnostics</h1>
    <p class="subtitle"><code>{html.escape(lens_name)}</code> over
      <code>{html.escape(graph.feature_space)}</code> CLIP features · select a node to inspect
      representative images.</p>
  </header>
  <section class="stats" aria-label="Graph summary">
    <div class="stat"><div class="stat-label">Coverage</div>
      <div class="stat-value">{float(diagnostics['coverage_fraction']):.0%}</div></div>
    <div class="stat"><div class="stat-label">Nodes</div>
      <div class="stat-value">{int(diagnostics['node_count'])}</div></div>
    <div class="stat"><div class="stat-label">Edges</div>
      <div class="stat-value">{int(diagnostics['edge_count'])}</div></div>
    <div class="stat"><div class="stat-label">Components</div>
      <div class="stat-value">{int(diagnostics['component_count'])}</div></div>
    <div class="stat"><div class="stat-label">DBSCAN ε</div>
      <div class="stat-value">{eps:.4f}</div></div>
  </section>
  <div class="layout">
    <section class="panel" aria-labelledby="topology-title">
      <div class="panel-heading"><h2 id="topology-title">Topology</h2>
        <span class="muted">x: mean lens · size: membership · lane: component</span></div>
      <div class="graph-wrap">{_svg_graph(graph)}</div>
    </section>
    <aside class="panel" aria-labelledby="detail-title" aria-live="polite">
      <h2 id="detail-title">Node detail</h2>
      <dl class="facts" id="node-facts"></dl>
      <h3>Centroid-nearest images</h3>
      <div class="representatives" id="representatives"></div>
    </aside>
  </div>
  <section class="panel unassigned" aria-labelledby="unassigned-title">
    <div class="panel-heading"><h2 id="unassigned-title">Unassigned images</h2>
      <span class="muted">DBSCAN noise across every cover interval</span></div>
    <ul class="unassigned-list" id="unassigned-list"></ul>
  </section>
</main>
<script id="report-data" type="application/json">{payload_json}</script>
<script>
const report = JSON.parse(document.getElementById('report-data').textContent);
const facts = document.getElementById('node-facts');
const representatives = document.getElementById('representatives');
const nodeElements = Array.from(document.querySelectorAll('.graph-node'));
function fact(label, value) {{
  const wrap=document.createElement('div'); wrap.className='fact';
  const dt=document.createElement('dt'); dt.textContent=label;
  const dd=document.createElement('dd'); dd.textContent=value;
  wrap.append(dt,dd); return wrap;
}}
function selectNode(nodeId) {{
  const node=report.nodes[nodeId]; if(!node) return;
  nodeElements.forEach(element =>
    element.classList.toggle('selected', element.dataset.nodeId===nodeId));
  facts.replaceChildren(
    fact('Node',node.id), fact('Source',node.source_id), fact('Component',node.component_id),
    fact('Members',String(node.size)),
    fact('Lens range',`${{node.lens_min.toFixed(4)}}–${{node.lens_max.toFixed(4)}}`),
    fact('Lens mean',node.lens_mean.toFixed(4))
  );
  representatives.replaceChildren();
  node.representative_ids.forEach(itemId => {{
    const image=report.images[itemId];
    const article=document.createElement('article'); article.className='representative';
    const img=document.createElement('img'); img.src=image.thumbnail;
    img.alt=`COCO image ${{itemId}}`;
    const copy=document.createElement('div');
    const id=document.createElement('div'); id.className='item-id'; id.textContent=itemId;
    const caption=document.createElement('p');
    caption.textContent=image.captions[0] || 'No caption';
    copy.append(id,caption); article.append(img,copy); representatives.append(article);
  }});
}}
nodeElements.forEach(element => {{
  element.addEventListener('click',()=>selectNode(element.dataset.nodeId));
  element.addEventListener('keydown',event=>{{
    if(event.key==='Enter'||event.key===' '){{
      event.preventDefault(); selectNode(element.dataset.nodeId);
    }}
  }});
}});
const unassignedList=document.getElementById('unassigned-list');
if(report.unassigned_ids.length===0) {{
  const li=document.createElement('li'); li.textContent='None'; unassignedList.append(li);
}} else {{
  report.unassigned_ids.forEach(itemId=>{{
    const li=document.createElement('li'); li.textContent=itemId; unassignedList.append(li);
  }});
}}
selectNode({json.dumps(initial_node)});
</script>
</body>
</html>
"""
    output_path.parent.mkdir(parents=True, exist_ok=True)
    partial = output_path.with_name(f"{output_path.name}.part")
    partial.write_text(document, encoding="utf-8")
    os.replace(partial, output_path)
    return output_path
