"""
HTML + JSON payload builder for ablation attribution graphs.

Renders a single dashboard page with a grid layout similar to the paper screenshot:
- x-axis: prompt tokens (positions k)
- y-axis: Emb, layers L0..L{n-1}, Lgt
- nodes: tokens, selected neurons (k,L,i), logit target
- edges: token->neuron (same k), neuron->neuron (influence), neuron->logit (importance)

Self-contained: no external JS dependencies.
"""

from __future__ import annotations

import math
import os
from typing import Any


def json_dumps(obj: Any) -> str:
    import json

    return json.dumps(obj, ensure_ascii=False)


def _stable_hash_int(*parts: int) -> int:
    # Deterministic hash for jitter.
    x = 1469598103934665603  # FNV offset basis
    for p in parts:
        x ^= int(p) & 0xFFFFFFFFFFFFFFFF
        x = (x * 1099511628211) & 0xFFFFFFFFFFFFFFFF
    return int(x)


def _jitter(k: int, L: int, i: int, radius: float) -> tuple[float, float]:
    h = _stable_hash_int(k, L, i)
    angle = ((h >> 1) % 10_000) / 10_000.0 * 2.0 * math.pi
    r = (((h >> 17) % 10_000) / 10_000.0) * radius
    return (math.cos(angle) * r, math.sin(angle) * r)


def _rel_link(from_dir: str, neuron_html_root: str, layer: int, neuron: int) -> str | None:
    # Only real MLP neurons have dashboards. Embedding/token sentinel layers are negative.
    if int(layer) < 0:
        return None
    target = os.path.join(
        os.path.abspath(neuron_html_root),
        f"layer{layer}",
        f"layer{layer}_neuron{neuron}.html",
    )
    rel = os.path.relpath(target, os.path.abspath(from_dir))
    return rel.replace(os.sep, "/")


def build_graph_payload(
    meta: dict[str, Any],
    token_strings: list[str],
    ranked: list[Any],  # list[Candidate]
    selected: list[Any],  # list[Candidate]
    influence: dict[str, Any],
    addback_curve: list[dict[str, float]],
    ui_order: list[tuple[int, int, int]],
    beam_sets_by_n: list[list[tuple[int, int, int]]] | None,
    neuron_html_root: str,
    run_dir: str,
    faithfulness_curve: list[dict[str, float]] | None = None,
) -> dict[str, Any]:
    n_layer = int(meta["n_layer"])
    T = len(token_strings)

    # Layout constants (pixels)
    margin_left = 70
    margin_right = 25
    margin_top = 25
    margin_bottom = 90
    width = max(900, margin_left + margin_right + T * 70)
    y_rows = n_layer + 2  # Emb + layers + Lgt
    # Give the graph more vertical room (helps readability of edges / node clusters).
    # Scale with number of rows, with a sensible cap for very deep models.
    row_spacing = 190
    height = max(820, margin_top + margin_bottom + row_spacing * (y_rows - 1))
    height = min(height, 1600)
    plot_h = height - margin_top - margin_bottom
    plot_w = width - margin_left - margin_right
    dx = plot_w / max(1, T - 1) if T > 1 else 1.0
    dy = plot_h / max(1, y_rows - 1) if y_rows > 1 else 1.0

    def x_for_k(k: int) -> float:
        return margin_left + float(k) * dx

    def y_for_row(row: int) -> float:
        # row 0 (Emb) at bottom, row y_rows-1 (Lgt) at top
        return margin_top + (plot_h - float(row) * dy)

    # Map (k,L,i) -> C and activation (from ranked list)
    C_map: dict[tuple[int, int, int], float] = {}
    act_map: dict[tuple[int, int, int], float] = {}
    for c in ranked:
        key = (int(c.k), int(c.layer), int(c.neuron))
        if getattr(c, "C", None) is None:
            continue
        C_map[key] = float(c.C)
        act_map[key] = float(c.activation)

    # Nodes
    nodes: list[dict[str, Any]] = []

    # Token nodes (Emb row = 0)
    for k, tok in enumerate(token_strings):
        nodes.append(
            {
                "id": f"tok_{k}",
                "kind": "token",
                "k": int(k),
                "label": tok,
                "x": x_for_k(k),
                "y": y_for_row(0),
            }
        )

    # Logit node (Lgt row = y_rows-1) at last position
    nodes.append(
        {
            "id": "logit",
            "kind": "logit",
            "k": int(T - 1),
            "label": f"target={meta.get('target_str','?')}  p0={meta.get('p0',0.0):.3f}",
            "x": x_for_k(T - 1),
            "y": y_for_row(y_rows - 1),
        }
    )

    selected_nodes = influence.get("selected_nodes", [])  # list[(k,L,i)]
    base_act = influence.get("base_act", [])
    logp_deltas = influence.get("logp_deltas", [])

    # Compute raw importance scores from C (smaller C => larger importance).
    # Use -log(C) for dynamic range, clamp at 0.
    imp_raw: list[float] = []
    for node in selected_nodes:
        k, L, i = int(node[0]), int(node[1]), int(node[2])
        C = float(C_map.get((k, L, i), 1.0))
        imp_raw.append(float(max(0.0, -math.log(max(C, 1e-12)))))
    imp_max = max(imp_raw) if imp_raw else 1.0
    imp_max = imp_max if imp_max > 0 else 1.0

    for idx, node in enumerate(selected_nodes):
        k, L, i = int(node[0]), int(node[1]), int(node[2])
        # Sentinel negative layers:
        # - L == -1: embedding-position node (placed on Emb row)
        row = 0 if L < 0 else (1 + L)
        jx, jy = _jitter(k, L, i, radius=12.0)
        C = float(C_map.get((k, L, i), 1.0))
        activation = float(act_map.get((k, L, i), base_act[idx] if idx < len(base_act) else 0.0))
        lp_delta = float(logp_deltas[idx]) if idx < len(logp_deltas) else 0.0
        importance_raw = float(max(0.0, -math.log(max(C, 1e-12))))
        importance = float(importance_raw / imp_max)
        subkind = "embedding" if L < 0 else "mlp"
        extra_x = 0.0
        nodes.append(
            {
                "id": f"n_{idx}",
                "kind": "neuron",
                "subkind": subkind,
                "k": k,
                "L": L,
                "i": i,
                "C": C,
                "importance": importance,
                "importance_raw": importance_raw,
                "activation": activation,
                "logp_delta": lp_delta,
                "x": x_for_k(k) + jx + extra_x,
                "y": y_for_row(row) + jy,
                "href": _rel_link(run_dir, neuron_html_root, L, i),
            }
        )

    # Map node key (k,L,i) -> node id for neurons in the rendered superset.
    key_to_node_id: dict[tuple[int, int, int], str] = {}
    for n in nodes:
        if n.get("kind") == "neuron":
            key_to_node_id[(int(n["k"]), int(n["L"]), int(n["i"]))] = str(n["id"])

    # Ranked list for the UI: only include entries that exist in the superset,
    # and carry token text instead of numeric k.
    ranked_ui = []
    for c in ranked:
        key = (int(c.k), int(c.layer), int(c.neuron))
        node_id = key_to_node_id.get((key[0], key[1], key[2]))
        if node_id is None:
            continue
        ranked_ui.append(
            {
                "node_id": node_id,
                "token": token_strings[int(c.k)] if 0 <= int(c.k) < len(token_strings) else f"k={int(c.k)}",
                "k": int(c.k),
                "L": int(c.layer),
                "i": int(c.neuron),
                "activation": float(c.activation),
                "C": float(c.C) if c.C is not None else None,
                "href": _rel_link(run_dir, neuron_html_root, int(c.layer), int(c.neuron)),
            }
        )

    # Order list for threshold filtering (subset selection in UI)
    ui_order_node_ids = []
    for (k, L, i) in ui_order:
        nid = key_to_node_id.get((int(k), int(L), int(i)))
        if nid is not None:
            ui_order_node_ids.append(nid)

    beam_sets_by_n_node_ids: list[list[str]] | None = None
    if beam_sets_by_n is not None:
        beam_sets_by_n_node_ids = []
        for nodes_n in beam_sets_by_n:
            ids_n: list[str] = []
            for (k, L, i) in nodes_n:
                nid = key_to_node_id.get((int(k), int(L), int(i)))
                if nid is not None:
                    ids_n.append(nid)
            beam_sets_by_n_node_ids.append(ids_n)

    # Edges
    edges: list[dict[str, Any]] = []

    # token -> neuron edges (same k), weight = |activation|
    for idx, node in enumerate(selected_nodes):
        k, L, i = int(node[0]), int(node[1]), int(node[2])
        w = abs(float(act_map.get((k, L, i), 0.0)))
        if w <= 0:
            continue
        edges.append({"source": f"tok_{k}", "target": f"n_{idx}", "kind": "tok", "weight": float(w)})

    # neuron -> logit edges (importance), weight = 1 - C
    for idx, node in enumerate(selected_nodes):
        k, L, i = int(node[0]), int(node[1]), int(node[2])
        C = float(C_map.get((k, L, i), 1.0))
        w = float(max(0.0, -math.log(max(C, 1e-12))))
        if w <= 0:
            continue
        edges.append({"source": f"n_{idx}", "target": "logit", "kind": "logit", "weight": float(w)})

    # neuron -> neuron influence edges (from influence runs)
    for e in influence.get("edges_nn", []):
        u = int(e["source"])
        v = int(e["target"])
        edges.append(
            {
                "source": f"n_{u}",
                "target": f"n_{v}",
                "kind": "nn",
                "weight": float(e.get("weight", 0.0)),
                "delta": float(e.get("delta", 0.0)),
                "rel_change": float(e.get("rel_change", 0.0)),
            }
        )

    out = {
        "meta": meta,
        "addback_curve": addback_curve,
        "ui_order_node_ids": ui_order_node_ids,
        "beam_sets_by_n_node_ids": beam_sets_by_n_node_ids,
        "ranked_ui": ranked_ui,
        "faithfulness_curve": faithfulness_curve,
        "token_strings": token_strings,
        "layout": {
            "width": width,
            "height": height,
            "margin_left": margin_left,
            "margin_right": margin_right,
            "margin_top": margin_top,
            "margin_bottom": margin_bottom,
            "n_layer": n_layer,
            "T": T,
        },
        "nodes": nodes,
        "edges": edges,
    }
    return out


def render_index_html(graph: dict[str, Any]) -> str:
    meta = graph.get("meta", {})
    layout = graph.get("layout", {})
    width = int(layout.get("width", 1100))
    height = int(layout.get("height", 560))

    return f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Attribution graph</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      background: #f5f5f5;
      color: #222;
    }}
    .wrap {{
      max-width: none;
      margin: 0;
      padding: 14px;
    }}
    .card {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      padding: 16px 18px;
      margin-bottom: 16px;
    }}
    .title {{
      font-size: 20px;
      font-weight: 700;
      margin: 0 0 8px 0;
      color: #283593;
    }}
    .meta {{
      font-size: 13px;
      color: #444;
      line-height: 1.4;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      margin-top: 10px;
      font-size: 13px;
    }}
    .controls label {{
      display: inline-flex;
      gap: 8px;
      align-items: center;
      user-select: none;
    }}
    .controls input[type=range] {{
      width: 220px;
    }}
    #viz {{
      width: 100%;
      height: 93vh;
      overflow: auto;
    }}
    svg {{
      background: #fafafa;
      border-radius: 12px;
      border: 1px solid #e5e5e5;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(20,20,20,0.92);
      color: #fff;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.35;
      max-width: 340px;
      z-index: 9999;
      display: none;
      box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #eee;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #fff;
      z-index: 1;
    }}
    .small {{
      font-size: 12px;
      color: #555;
    }}
    a {{
      color: #1976d2;
      text-decoration: none;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    details summary {{
      cursor: pointer;
      font-weight: 700;
      color: #333;
      user-select: none;
      padding: 4px 0;
    }}
    .curve-slider-row {{
      display: flex;
      align-items: center;
      gap: 10px;
      margin-top: 10px;
    }}
    /* Match the curve plot left margin (renderCurve uses ml=50) */
    .curve-slider-spacer {{
      width: 50px;
      flex: 0 0 auto;
    }}
    .curve-slider-row input[type=range] {{
      flex: 1 1 auto;
      min-width: 260px;
    }}
    /* Match the curve plot right margin (mr=18) + room for number */
    .curve-slider-value {{
      width: 54px;
      text-align: right;
      color: #444;
      font-size: 13px;
      flex: 0 0 auto;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title">Ablation attribution graph</div>
      <div class="meta">
        <div><b>ckpt</b>: {meta.get("ckpt","")}</div>
        <div><b>target</b>: {meta.get("target_str","?")} (id={meta.get("target_id","?")}), <b>p0</b>={meta.get("p0",0.0):.4f}</div>
        <div><b>retain_ratio</b>: {meta.get("retain_ratio",0.8)}  →  <b>final_ratio</b>: {meta.get("final_ratio",0.0):.4f}</div>
        <div class="small"><b>prompt</b>: {meta.get("prompt","")}</div>
      </div>
      <div class="controls">
        <label><input id="toggleTok" type="checkbox" checked /> token→neuron</label>
        <label><input id="toggleNN" type="checkbox" checked /> neuron→neuron</label>
        <label><input id="toggleLogit" type="checkbox" checked /> neuron→logit</label>
        <label>edge threshold <input id="edgeThr" type="range" min="0" max="1" step="0.01" value="0.00" /> <span id="edgeThrVal">0.00</span></label>
      </div>
    </div>

    <div class="card">
      <div id="curveTitle" style="font-weight:700; margin-bottom:8px;">Add‑back curve</div>
      <svg id="curve" width="100%" height="220" viewBox="0 0 900 220" style="background:#fafafa; border:1px solid #e5e5e5; border-radius:12px;"></svg>
      <div class="curve-slider-row">
        <div class="curve-slider-spacer"></div>
        <input id="sizeSlider" type="range" min="1" max="100" step="1" value="30" />
        <div class="curve-slider-value"><span id="sizeVal">30</span></div>
      </div>
    </div>

    <div class="card" id="heatmapCard">
      <div style="font-weight:700; margin-bottom:8px;">MLP weight interaction heatmap (top-30 neurons by C)</div>
      <svg id="mlpHeatmap" width="100%" height="420" viewBox="0 0 520 520" style="background:#fafafa; border:1px solid #e5e5e5; border-radius:12px;"></svg>
    </div>

    <div class="card" id="heatmapCardRandom">
      <div style="font-weight:700; margin-bottom:8px;">Random baseline heatmap (same neurons, random vectors)</div>
      <svg id="mlpHeatmapRandom" width="100%" height="420" viewBox="0 0 520 520" style="background:#fafafa; border:1px solid #e5e5e5; border-radius:12px;"></svg>
    </div>

    <div class="card" id="viz">
      <svg id="svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"></svg>
    </div>

    <div class="card">
      <details>
        <summary>Ranked neurons (top 500 by C)</summary>
        <div style="max-height: 55vh; overflow: auto; margin-top: 10px;">
        <table id="rankTable">
            <thead>
              <tr>
              <th>token</th><th>L</th><th>i</th><th>act</th><th>C</th><th>link</th>
              </tr>
            </thead>
            <tbody></tbody>
          </table>
        </div>
      </details>
    </div>
  </div>

  <div id="tooltip" class="tooltip"></div>

  <script>
    const graph = {json_dumps(graph)};

    function el(id) {{ return document.getElementById(id); }}
    const svg = el('svg');
    const tooltip = el('tooltip');

    const toggleTok = el('toggleTok');
    const toggleNN = el('toggleNN');
    const toggleLogit = el('toggleLogit');
    const edgeThr = el('edgeThr');
    const edgeThrVal = el('edgeThrVal');

    const nodes = graph.nodes;
    const edges = graph.edges;
    const curve = graph.addback_curve || [];
    const faith = graph.faithfulness_curve || [];
    const uiOrder = graph.ui_order_node_ids || [];
    const beamSetsByN = graph.beam_sets_by_n_node_ids || null;
    const rankedUI = graph.ranked_ui || [];
    const meta = graph.meta || {{}};
    const weightInt = meta.weight_interaction || null;
    const sizeSlider = el('sizeSlider');
    const sizeVal = el('sizeVal');
    const heatSvg = el('mlpHeatmap');
    const heatCard = el('heatmapCard');
    const heatSvgRandom = el('mlpHeatmapRandom');
    const heatCardRandom = el('heatmapCardRandom');

    function clearSvg() {{
      while (svg.firstChild) svg.removeChild(svg.firstChild);
    }}

    function fmt(x) {{
      if (x === null || x === undefined) return '';
      if (typeof x === 'number') return x.toFixed(4);
      return String(x);
    }}

    function showTooltip(html, x, y) {{
      tooltip.innerHTML = html;
      tooltip.style.left = (x + 14) + 'px';
      tooltip.style.top = (y + 14) + 'px';
      tooltip.style.display = 'block';
    }}
    function hideTooltip() {{
      tooltip.style.display = 'none';
    }}

    function nodeById(id) {{
      return nodes.find(n => n.id === id);
    }}

    function edgeVisible(e, thr) {{
      if (e.kind === 'tok' && !toggleTok.checked) return false;
      if (e.kind === 'nn' && !toggleNN.checked) return false;
      if (e.kind === 'logit' && !toggleLogit.checked) return false;
      return ((e.weight || 0) / edgeMax) >= thr;
    }}

    function clamp(x, lo, hi) {{ return Math.max(lo, Math.min(hi, x)); }}

    function activeNeuronSet(n) {{
      const active = new Set();
      if (beamSetsByN && beamSetsByN.length) {{
        const idx = Math.max(0, Math.min(beamSetsByN.length - 1, Math.round(n)));
        for (const nid of (beamSetsByN[idx] || [])) active.add(nid);
        return active;
      }}
      for (let i = 0; i < n && i < uiOrder.length; i++) active.add(uiOrder[i]);
      return active;
    }}

    function render() {{
      clearSvg();
      const thr = parseFloat(edgeThr.value);
      edgeThrVal.textContent = thr.toFixed(2);
      const n = parseInt(sizeSlider.value, 10) || 1;
      sizeVal.textContent = String(n);
      const active = activeNeuronSet(n);

      const layout = graph.layout;
      const ml = layout.margin_left, mr = layout.margin_right, mt = layout.margin_top, mb = layout.margin_bottom;
      const W = layout.width, H = layout.height;
      const plotW = W - ml - mr;
      const plotH = H - mt - mb;
      const T = layout.T;
      const yRows = layout.n_layer + 2;

      function xForK(k) {{
        return ml + (T <= 1 ? 0 : (k * plotW / (T-1)));
      }}
      function yForRow(row) {{
        return mt + (plotH - row * (plotH / (yRows-1)));
      }}

      // grid lines
      for (let k = 0; k < T; k++) {{
        const x = xForK(k);
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1', x); line.setAttribute('x2', x);
        line.setAttribute('y1', mt); line.setAttribute('y2', mt+plotH);
        line.setAttribute('stroke', '#e6e6e6');
        svg.appendChild(line);
      }}
      for (let r = 0; r < yRows; r++) {{
        const y = yForRow(r);
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1', ml); line.setAttribute('x2', ml+plotW);
        line.setAttribute('y1', y); line.setAttribute('y2', y);
        line.setAttribute('stroke', '#ededed');
        svg.appendChild(line);
      }}

      // y labels
      const yLabels = ['Emb'];
      for (let L = 0; L < layout.n_layer; L++) yLabels.push('L'+L);
      yLabels.push('Lgt');
      for (let r = 0; r < yRows; r++) {{
        const y = yForRow(r);
        const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
        txt.textContent = yLabels[r];
        txt.setAttribute('x', 10);
        txt.setAttribute('y', y + 4);
        txt.setAttribute('fill', '#666');
        txt.setAttribute('font-size', '12');
        svg.appendChild(txt);
      }}

      // token labels (rotated)
      for (let k = 0; k < T; k++) {{
        const tok = nodes.find(n => n.kind==='token' && n.k===k);
        if (!tok) continue;
        const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
        txt.textContent = tok.label;
        txt.setAttribute('x', xForK(k));
        txt.setAttribute('y', H - 18);
        txt.setAttribute('fill', '#666');
        txt.setAttribute('font-size', '12');
        txt.setAttribute('transform', `rotate(-0 ${{xForK(k)}} ${{H - 18}})`);
        txt.setAttribute('text-anchor', 'end');
        svg.appendChild(txt);
      }}

      // edges
      const edgeGroup = document.createElementNS('http://www.w3.org/2000/svg','g');
      svg.appendChild(edgeGroup);
      const maxW = edgeMax;
      for (const e of edges) {{
        if (!edgeVisible(e, thr)) continue;
        // Filter edges by active neuron nodes (token/logit nodes always allowed)
        if (e.kind === 'nn' || e.kind === 'logit') {{
          if (!active.has(e.source) || !active.has(e.target) && e.target !== 'logit') continue;
        }}
        if (e.kind === 'tok') {{
          if (!active.has(e.target)) continue;
        }}
        const s = nodeById(e.source);
        const t = nodeById(e.target);
        if (!s || !t) continue;
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1', s.x); line.setAttribute('y1', s.y);
        line.setAttribute('x2', t.x); line.setAttribute('y2', t.y);
        const w = (e.weight || 0) / maxW;
        const base = e.kind === 'nn' ? 0.14 : (e.kind === 'logit' ? 0.18 : 0.10);
        const alpha = Math.max(base, Math.min(0.55, base + 0.45*w));
        line.setAttribute('stroke', `rgba(60,60,60,${{alpha}})`);
        line.setAttribute('stroke-width', (e.kind === 'nn' ? (0.5 + 2.2*w) : (0.4 + 1.4*w)).toFixed(2));
        edgeGroup.appendChild(line);
      }}

      // nodes
      const nodeGroup = document.createElementNS('http://www.w3.org/2000/svg','g');
      svg.appendChild(nodeGroup);

      function drawCircle(n, r, fill, stroke, strokeW) {{
        const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
        c.setAttribute('cx', n.x); c.setAttribute('cy', n.y);
        c.setAttribute('r', r);
        c.setAttribute('fill', fill);
        c.setAttribute('stroke', stroke);
        c.setAttribute('stroke-width', strokeW);
        return c;
      }}
      function drawSquare(n, size, fill, stroke, strokeW) {{
        const s = document.createElementNS('http://www.w3.org/2000/svg','rect');
        s.setAttribute('x', n.x - size/2);
        s.setAttribute('y', n.y - size/2);
        s.setAttribute('width', size);
        s.setAttribute('height', size);
        s.setAttribute('rx', 2);
        s.setAttribute('fill', fill);
        s.setAttribute('stroke', stroke);
        s.setAttribute('stroke-width', strokeW);
        return s;
      }}

      for (const n of nodes) {{
        let shape;
        if (n.kind === 'token') {{
          shape = drawSquare(n, 8, '#ffffff', '#999', 1);
        }} else if (n.kind === 'logit') {{
          shape = drawSquare(n, 12, '#fff', '#111', 2);
        }} else {{
          if (!active.has(n.id)) continue;
          const imp = Math.max(0, Math.min(1, n.importance || 0));
          const r = 4 + 8*imp;
          const act = Math.abs(n.activation || 0);
          const a = Math.max(0.12, Math.min(0.65, 0.10 + Math.log(1 + act)*0.28));
          const sub = (n.subkind || '');
          const fill = (sub === 'token_ablation')
            ? `rgba(230,126,34,${{a}})`    // orange
            : (sub === 'embedding')
              ? `rgba(33,150,243,${{a}})` // blue
              : `rgba(76,175,80,${{a}})`; // green
          shape = drawCircle(n, r, fill, '#111', 2);
          if (imp > 0.45) shape.setAttribute('stroke', '#ff00b3');
        }}

        shape.style.cursor = (n.kind === 'neuron') ? 'pointer' : 'default';
        shape.addEventListener('mousemove', (evt) => {{
          if (n.kind === 'token') {{
            showTooltip(`<b>token</b> k=${{n.k}}<br/>${{n.label}}`, evt.clientX, evt.clientY);
          }} else if (n.kind === 'logit') {{
            showTooltip(`<b>logit</b><br/>${{n.label}}`, evt.clientX, evt.clientY);
          }} else {{
            showTooltip(
              `<b>neuron</b> k=${{n.k}} L=${{n.L}} i=${{n.i}}` +
              `<br/><b>act</b>=${{fmt(n.activation)}}  <b>C</b>=${{fmt(n.C)}}` +
              `<br/><b>logpΔ</b>=${{fmt(n.logp_delta)}}`,
              evt.clientX, evt.clientY
            );
          }}
        }});
        shape.addEventListener('mouseleave', () => hideTooltip());
        if (n.kind === 'neuron' && n.href) {{
          shape.addEventListener('click', () => {{ window.open(n.href, '_blank'); }});
        }}
        nodeGroup.appendChild(shape);
      }}
    }}

    function renderCurve() {{
      const curveSvg = el('curve');
      if (!curveSvg) return;
      while (curveSvg.firstChild) curveSvg.removeChild(curveSvg.firstChild);

      const W = 900, H = 220;
      const ml = 50, mr = 18, mt = 12, mb = 30;
      const pw = W - ml - mr;
      const ph = H - mt - mb;

      // x-axis: always "# neurons added" (n_added)
      const xs = curve.map(p => (p && p.n_added) || 0);
      const ys = curve.map(p => p.retain_ratio || 0);
      const xs2 = faith.map(p => p.n || 0);
      const ys2 = faith.map(p => p.faithfulness || 0);
      const xMax = Math.max(...xs, ...xs2, 1);
      const yMax = Math.max(...ys, ...ys2, 1.0);
      const yMin = 0.0;

      function xFor(x) {{
        const frac = (xMax <= 0) ? 0.0 : (x / xMax);
        return ml + frac * pw;
      }}
      function yFor(y) {{ return mt + (ph - ((y - yMin) / (yMax - yMin + 1e-9)) * ph); }}

      // grid + ticks
      const xTickVals = [];
      let step = 1;
      if (xMax > 50) step = 10;
      if (xMax > 200) step = 25;
      if (xMax > 500) step = 50;
      if (xMax > 1000) step = 100;
      for (let v = 0; v <= xMax; v += step) xTickVals.push(v);
      if (!xTickVals.includes(xMax)) xTickVals.push(Math.round(xMax));

      for (const xv of xTickVals) {{
        const x = xFor(xv);
        const gl = document.createElementNS('http://www.w3.org/2000/svg','line');
        gl.setAttribute('x1', x); gl.setAttribute('x2', x);
        gl.setAttribute('y1', mt); gl.setAttribute('y2', mt+ph);
        gl.setAttribute('stroke', '#e6e6e6');
        curveSvg.appendChild(gl);

        const tick = document.createElementNS('http://www.w3.org/2000/svg','line');
        tick.setAttribute('x1', x); tick.setAttribute('x2', x);
        tick.setAttribute('y1', mt+ph); tick.setAttribute('y2', mt+ph+4);
        tick.setAttribute('stroke', '#888');
        curveSvg.appendChild(tick);

        const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
        txt.textContent = String(Math.round(xv));
        txt.setAttribute('x', x);
        txt.setAttribute('y', mt+ph+16);
        txt.setAttribute('text-anchor', 'middle');
        txt.setAttribute('fill', '#666');
        txt.setAttribute('font-size', '11');
        curveSvg.appendChild(txt);
      }}

      const yTicks = 6;
      for (let ti = 0; ti < yTicks; ti++) {{
        const frac = yTicks <= 1 ? 0 : (ti / (yTicks - 1));
        const yv = yMin + frac * (yMax - yMin);
        const y = yFor(yv);
        const gl = document.createElementNS('http://www.w3.org/2000/svg','line');
        gl.setAttribute('x1', ml); gl.setAttribute('x2', ml+pw);
        gl.setAttribute('y1', y); gl.setAttribute('y2', y);
        gl.setAttribute('stroke', '#ededed');
        curveSvg.appendChild(gl);

        const tick = document.createElementNS('http://www.w3.org/2000/svg','line');
        tick.setAttribute('x1', ml-4); tick.setAttribute('x2', ml);
        tick.setAttribute('y1', y); tick.setAttribute('y2', y);
        tick.setAttribute('stroke', '#888');
        curveSvg.appendChild(tick);

        const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
        txt.textContent = yv.toFixed(2);
        txt.setAttribute('x', ml-8);
        txt.setAttribute('y', y+4);
        txt.setAttribute('text-anchor', 'end');
        txt.setAttribute('fill', '#666');
        txt.setAttribute('font-size', '11');
        curveSvg.appendChild(txt);
      }}

      // axes
      const ax = document.createElementNS('http://www.w3.org/2000/svg','line');
      ax.setAttribute('x1', ml); ax.setAttribute('x2', ml+pw);
      ax.setAttribute('y1', mt+ph); ax.setAttribute('y2', mt+ph);
      ax.setAttribute('stroke', '#999');
      curveSvg.appendChild(ax);
      const ay = document.createElementNS('http://www.w3.org/2000/svg','line');
      ay.setAttribute('x1', ml); ay.setAttribute('x2', ml);
      ay.setAttribute('y1', mt); ay.setAttribute('y2', mt+ph);
      ay.setAttribute('stroke', '#999');
      curveSvg.appendChild(ay);

      // labels
      const xl = document.createElementNS('http://www.w3.org/2000/svg','text');
      xl.textContent = '# neurons added';
      xl.setAttribute('x', ml + pw/2);
      xl.setAttribute('y', H - 8);
      xl.setAttribute('text-anchor', 'middle');
      xl.setAttribute('fill', '#666');
      xl.setAttribute('font-size', '12');
      curveSvg.appendChild(xl);

      const yl = document.createElementNS('http://www.w3.org/2000/svg','text');
      yl.textContent = 'Score';
      yl.setAttribute('x', 14);
      yl.setAttribute('y', mt + ph/2);
      yl.setAttribute('transform', `rotate(-90 14 ${{mt + ph/2}})`);
      yl.setAttribute('text-anchor', 'middle');
      yl.setAttribute('fill', '#666');
      yl.setAttribute('font-size', '12');
      curveSvg.appendChild(yl);

      // Faithfulness curve (solid, color #8a3434)
      if (curve.length > 0) {{
        let d = '';
        for (let i = 0; i < curve.length; i++) {{
          const xv = (curve[i] && curve[i].n_added) || 0;
          const x = xFor(xv);
          const y = yFor(curve[i].retain_ratio || 0);
          d += (i === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2) + ' ';
        }}
        const path = document.createElementNS('http://www.w3.org/2000/svg','path');
        path.setAttribute('d', d.trim());
        path.setAttribute('fill', 'none');
        path.setAttribute('stroke', '#8a3434');
        path.setAttribute('stroke-width', '2');
        curveSvg.appendChild(path);

        // last point marker
        const last = curve[curve.length - 1];
        const lastX = (last && last.n_added) || 0;
        const cx = xFor(lastX);
        const cy = yFor(last.retain_ratio || 0);
        const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
        c.setAttribute('cx', cx); c.setAttribute('cy', cy);
        c.setAttribute('r', 4);
        c.setAttribute('fill', '#8a3434');
        curveSvg.appendChild(c);
      }}

      // Completeness curve (dashed, color #b46464)
      if (faith.length > 0) {{
        let d2 = '';
        for (let i = 0; i < faith.length; i++) {{
          const x = xFor(faith[i].n || 0);
          const y = yFor(faith[i].faithfulness || 0);
          d2 += (i === 0 ? 'M' : 'L') + x.toFixed(2) + ' ' + y.toFixed(2) + ' ';
        }}
        const path2 = document.createElementNS('http://www.w3.org/2000/svg','path');
        path2.setAttribute('d', d2.trim());
        path2.setAttribute('fill', 'none');
        path2.setAttribute('stroke', '#b46464');
        path2.setAttribute('stroke-width', '2');
        path2.setAttribute('stroke-dasharray', '6,4');
        curveSvg.appendChild(path2);

        // markers on completeness curve
        for (let i = 0; i < faith.length; i++) {{
          const cx = xFor(faith[i].n || 0);
          const cy = yFor(faith[i].faithfulness || 0);
          const c2 = document.createElementNS('http://www.w3.org/2000/svg','circle');
          c2.setAttribute('cx', cx); c2.setAttribute('cy', cy);
          c2.setAttribute('r', 3);
          c2.setAttribute('fill', '#b46464');
          curveSvg.appendChild(c2);
        }}
      }}

      // Legend
      const legendY = mt + 14;
      const legendX = ml + 8;
      const leg1 = document.createElementNS('http://www.w3.org/2000/svg','rect');
      leg1.setAttribute('x', legendX);
      leg1.setAttribute('y', legendY - 8);
      leg1.setAttribute('width', 18);
      leg1.setAttribute('height', 3);
      leg1.setAttribute('fill', '#8a3434');
      curveSvg.appendChild(leg1);
      const leg1Text = document.createElementNS('http://www.w3.org/2000/svg','text');
      leg1Text.textContent = 'Faithfulness';
      leg1Text.setAttribute('x', legendX + 24);
      leg1Text.setAttribute('y', legendY - 4);
      leg1Text.setAttribute('fill', '#666');
      leg1Text.setAttribute('font-size', '11');
      curveSvg.appendChild(leg1Text);

      if (faith.length > 0) {{
        const leg2 = document.createElementNS('http://www.w3.org/2000/svg','rect');
        leg2.setAttribute('x', legendX);
        leg2.setAttribute('y', legendY + 6);
        leg2.setAttribute('width', 18);
        leg2.setAttribute('height', 3);
        leg2.setAttribute('fill', '#b46464');
        curveSvg.appendChild(leg2);
        const leg2Text = document.createElementNS('http://www.w3.org/2000/svg','text');
        leg2Text.textContent = 'Completeness';
        leg2Text.setAttribute('x', legendX + 24);
        leg2Text.setAttribute('y', legendY + 10);
        leg2Text.setAttribute('fill', '#666');
        leg2Text.setAttribute('font-size', '11');
        curveSvg.appendChild(leg2Text);
      }}
    }}

    function renderHeatmapCore(svgEl, cardEl, data, neurons, labelPrefix, maxAbsGlobal) {{
      if (!svgEl || !cardEl || !data || !data.length) {{
        if (cardEl) cardEl.style.display = 'none';
        return;
      }}
      cardEl.style.display = 'block';
      while (svgEl.firstChild) svgEl.removeChild(svgEl.firstChild);

      const n = data.length;
      const W = 520, H = 520;
      const ml = 80, mr = 20, mt = 40, mb = 60;
      const pw = W - ml - mr;
      const ph = H - mt - mb;
      const cellW = pw / n;
      const cellH = ph / n;

      let maxAbs = maxAbsGlobal;
      if (!Number.isFinite(maxAbs) || maxAbs <= 0) {{
        maxAbs = 0.0;
        for (let i = 0; i < n; i++) {{
          for (let j = 0; j < n; j++) {{
            const v = data[i][j] || 0.0;
            const av = Math.abs(v);
            if (av > maxAbs) maxAbs = av;
          }}
        }}
        if (maxAbs <= 0) maxAbs = 1.0;
      }}

      for (let i = 0; i < n; i++) {{
        const info = neurons[i] || null;
        const label = info ? `L${{info.L}} i${{info.i}}` : String(i);
        const tx = document.createElementNS('http://www.w3.org/2000/svg','text');
        tx.textContent = label;
        tx.setAttribute('x', ml + (i + 0.5) * cellW);
        tx.setAttribute('y', H - 18);
        tx.setAttribute('text-anchor', 'end');
        tx.setAttribute('transform', `rotate(-45 ${{ml + (i + 0.5) * cellW}} ${{H - 18}})`);
        tx.setAttribute('fill', '#555');
        tx.setAttribute('font-size', '10');
        svgEl.appendChild(tx);

        const ty = document.createElementNS('http://www.w3.org/2000/svg','text');
        ty.textContent = label;
        ty.setAttribute('x', ml - 6);
        ty.setAttribute('y', mt + (i + 0.5) * cellH + 3);
        ty.setAttribute('text-anchor', 'end');
        ty.setAttribute('fill', '#555');
        ty.setAttribute('font-size', '10');
        svgEl.appendChild(ty);
      }}

      for (let i = 0; i < n; i++) {{
        for (let j = 0; j < n; j++) {{
          const v = data[i][j] || 0.0;
          const frac = Math.max(0, Math.min(1, Math.abs(v) / maxAbs));
          const basePos = [33, 150, 243];
          const baseNeg = [244, 67, 54];
          const base = v >= 0 ? basePos : baseNeg;
          const r = base[0];
          const g = base[1];
          const b = base[2];
          const alpha = 0.12 + 0.78 * frac;

          const rect = document.createElementNS('http://www.w3.org/2000/svg','rect');
          rect.setAttribute('x', ml + j * cellW);
          rect.setAttribute('y', mt + i * cellH);
          rect.setAttribute('width', cellW);
          rect.setAttribute('height', cellH);
          rect.setAttribute('fill', `rgba(${{r}},${{g}},${{b}},${{alpha}})`);
          rect.setAttribute('stroke', '#f5f5f5');
          rect.setAttribute('stroke-width', '0.5');
          rect.addEventListener('mousemove', (evt) => {{
            const src = neurons[i] || {{}};
            const dst = neurons[j] || {{}};
            const srcLabel = src ? `L${{src.L}} i${{src.i}}` : `row=${{i}}`;
            const dstLabel = dst ? `L${{dst.L}} i${{dst.i}}` : `col=${{j}}`;
            showTooltip(
              `<b>${{labelPrefix}}</b><br/>row=${{srcLabel}}<br/>col=${{dstLabel}}<br/>value=${{fmt(v)}}`,
              evt.clientX,
              evt.clientY
            );
          }});
          rect.addEventListener('mouseleave', () => hideTooltip());
          svgEl.appendChild(rect);
        }}
      }}

      const xl = document.createElementNS('http://www.w3.org/2000/svg','text');
      xl.textContent = 'W_up (source neuron, row)';
      xl.setAttribute('x', ml + pw / 2);
      xl.setAttribute('y', H - 4);
      xl.setAttribute('text-anchor', 'middle');
      xl.setAttribute('fill', '#666');
      xl.setAttribute('font-size', '11');
      svgEl.appendChild(xl);

      const yl = document.createElementNS('http://www.w3.org/2000/svg','text');
      yl.textContent = 'W_down (target neuron, column)';
      yl.setAttribute('x', 16);
      yl.setAttribute('y', mt + ph / 2);
      yl.setAttribute('transform', `rotate(-90 16 ${{mt + ph / 2}})`);
      yl.setAttribute('text-anchor', 'middle');
      yl.setAttribute('fill', '#666');
      yl.setAttribute('font-size', '11');
      svgEl.appendChild(yl);
    }}

    function renderHeatmaps() {{
      if (!weightInt) {{
        if (heatCard) heatCard.style.display = 'none';
        if (heatCardRandom) heatCardRandom.style.display = 'none';
        return;
      }}
      const neurons = weightInt.neurons || [];
      const metric = weightInt.metric || 'dot';
      const labelMain = metric === 'cos' ? 'interaction (cosine)' : 'interaction (dot)';
      const labelRand = metric === 'cos' ? 'random interaction (cosine)' : 'random interaction (dot)';
      const dataMain = weightInt.matrix || [];
      const dataRand = weightInt.matrix_random || [];

      let maxAbs = 0.0;
      const mats = [dataMain, dataRand];
      for (const m of mats) {{
        if (!m || !m.length) continue;
        for (let i = 0; i < m.length; i++) {{
          for (let j = 0; j < m[i].length; j++) {{
            const v = m[i][j] || 0.0;
            const av = Math.abs(v);
            if (av > maxAbs) maxAbs = av;
          }}
        }}
      }}
      if (maxAbs <= 0) maxAbs = 1.0;

      renderHeatmapCore(heatSvg, heatCard, dataMain, neurons, labelMain, maxAbs);
      renderHeatmapCore(heatSvgRandom, heatCardRandom, dataRand, neurons, labelRand, maxAbs);
    }}

    const edgeMax = Math.max(...edges.map(e => e.weight || 0), 1e-9);
    toggleTok.addEventListener('change', render);
    toggleNN.addEventListener('change', render);
    toggleLogit.addEventListener('change', render);
    edgeThr.addEventListener('input', render);
    sizeSlider.addEventListener('input', () => {{ renderTable(); render(); }});

    sizeSlider.min = '1';
    sizeSlider.max = String(meta.max_nodes ?? 100);
    sizeSlider.value = String(meta.circuit_size ?? 30);
    sizeVal.textContent = String(sizeSlider.value);

    const curveTitle = el('curveTitle');
    if (curveTitle) {{
      curveTitle.textContent = 'Add‑back curve';
    }}

    const tbody = document.querySelector('#rankTable tbody');
    function renderTable() {{
      const n = parseInt(sizeSlider.value, 10) || 1;
      const active = activeNeuronSet(n);
      tbody.innerHTML = '';
      for (const row of rankedUI) {{
        if (!active.has(row.node_id)) continue;
        const tr = document.createElement('tr');
        const cols = [row.token, row.L, row.i, row.activation, row.C];
        for (const c of cols) {{
          const td = document.createElement('td');
          td.textContent = (typeof c === 'number') ? (Number.isFinite(c) ? c.toFixed(4) : '') : String(c);
          tr.appendChild(td);
        }}
        const tdLink = document.createElement('td');
        if (row.href) {{
          const a = document.createElement('a');
          a.href = row.href; a.target = '_blank'; a.textContent = 'open';
          tdLink.appendChild(a);
        }} else {{
          tdLink.textContent = '';
        }}
        tr.appendChild(tdLink);
        tbody.appendChild(tr);
      }}
    }}

    renderCurve();
    renderTable();
    renderHeatmaps();
    render();
  </script>
</body>
</html>
"""


def build_detailed_circuit_payload(
    meta: dict[str, Any],
    token_strings: list[str],
    circuit_nodes: list[Any],  # list[Candidate]
    circuit_C_values: list[Any],  # list[Candidate] with localized C
    circuit_edges_data: dict[str, Any],
    neuron_html_root: str,
    detail_dir: str = "",
) -> dict[str, Any]:
    """Build graph payload for detailed circuit visualization using same style as main graph."""
    n_layer = int(meta["n_layer"])
    T = len(token_strings)

    # Layout constants (pixels) - same as main graph
    margin_left = 70
    margin_right = 25
    margin_top = 25
    margin_bottom = 90
    width = max(900, margin_left + margin_right + T * 70)
    y_rows = n_layer + 2  # Emb + layers + Lgt
    row_spacing = 190
    height = max(820, margin_top + margin_bottom + row_spacing * (y_rows - 1))
    height = min(height, 1600)
    plot_h = height - margin_top - margin_bottom
    plot_w = width - margin_left - margin_right
    dx = plot_w / max(1, T - 1) if T > 1 else 1.0
    dy = plot_h / max(1, y_rows - 1) if y_rows > 1 else 1.0

    def x_for_k(k: int) -> float:
        return margin_left + float(k) * dx

    def y_for_row(row: int) -> float:
        return margin_top + (plot_h - float(row) * dy)

    # Map (k,L,i) -> C_localized
    C_map: dict[tuple[int, int, int], float] = {}
    act_map: dict[tuple[int, int, int], float] = {}
    for c in circuit_C_values:
        key = (int(c.k), int(c.layer), int(c.neuron))
        if getattr(c, "C", None) is None:
            continue
        C_map[key] = float(c.C)
        act_map[key] = float(c.activation)

    # Nodes
    nodes: list[dict[str, Any]] = []

    # Token nodes (Emb row = 0)
    for k, tok in enumerate(token_strings):
        nodes.append(
            {
                "id": f"tok_{k}",
                "kind": "token",
                "k": int(k),
                "label": tok,
                "x": x_for_k(k),
                "y": y_for_row(0),
            }
        )

    # Logit node (Lgt row = y_rows-1)
    nodes.append(
        {
            "id": "logit",
            "kind": "logit",
            "k": int(T - 1),
            "label": f"target={meta.get('target_str','?')}  p0={meta.get('p0',0.0):.3f}",
            "x": x_for_k(T - 1),
            "y": y_for_row(y_rows - 1),
        }
    )

    # Compute importance from localized C
    imp_raw: list[float] = []
    for node in circuit_nodes:
        k, L, i = int(node.k), int(node.layer), int(node.neuron)
        C = float(C_map.get((k, L, i), 1.0))
        imp_raw.append(float(max(0.0, -math.log(max(C, 1e-12)))))
    imp_max = max(imp_raw) if imp_raw else 1.0
    imp_max = imp_max if imp_max > 0 else 1.0

    # Circuit neuron nodes
    for idx, node in enumerate(circuit_nodes):
        k, L, i = int(node.k), int(node.layer), int(node.neuron)
        row = 1 + L
        jx, jy = _jitter(k, L, i, radius=12.0)
        C = float(C_map.get((k, L, i), 1.0))
        activation = float(act_map.get((k, L, i), node.activation))
        imp = float(imp_raw[idx]) if idx < len(imp_raw) else 0.0
        size = 8.0 + 20.0 * (imp / imp_max) if imp_max > 0 else 8.0
        
        nodes.append(
            {
                "id": f"neu_{k}_{L}_{i}",
                "kind": "neuron",
                "k": int(k),
                "layer": int(L),
                "neuron": int(i),
                "label": f"({k},{i})",
                "x": x_for_k(k) + jx,
                "y": y_for_row(row) + jy,
                "size": float(size),
                "C": float(C),
                "activation": float(activation),
                "importance": float(imp),
                "html_link": _rel_link(detail_dir, neuron_html_root, L, i),
            }
        )

    # Edges from circuit_edges_data
    edges: list[dict[str, Any]] = []
    per_node = circuit_edges_data.get("per_node", {})
    
    edge_id = 0
    for source_node_id, node_info in per_node.items():
        parts = source_node_id.split('_')
        if len(parts) != 3:
            continue
        k_s, L_s, i_s = int(parts[1]), int(parts[0]), int(parts[2])
        source_id = f"neu_{k_s}_{L_s}_{i_s}"
        
        for influence in node_info.get("influences_out", []):
            target_id_str = influence["target"]
            parts_t = target_id_str.split('_')
            if len(parts_t) != 3:
                continue
            k_t, L_t, i_t = int(parts_t[1]), int(parts_t[0]), int(parts_t[2])
            target_id = f"neu_{k_t}_{L_t}_{i_t}"
            rel_change = float(influence["rel_change"])
            
            edges.append(
                {
                    "id": f"edge_{edge_id}",
                    "source": source_id,
                    "target": target_id,
                    "weight": float(abs(rel_change)),
                    "rel_change": float(rel_change),
                    "kind": "influence",
                }
            )
            edge_id += 1

    return {
        "layout": {
            "width": int(width),
            "height": int(height),
            "margin_left": margin_left,
            "margin_right": margin_right,
            "margin_top": margin_top,
            "margin_bottom": margin_bottom,
            "T": T,
            "n_layer": n_layer,
        },
        "width": int(width),
        "height": int(height),
        "nodes": nodes,
        "edges": edges,
        "meta": meta,
        "per_node_details": per_node,
    }


def render_detailed_circuit_html(graph: dict[str, Any], circuit_edges: dict[str, Any], neuron_html_root: str = "", detail_dir: str = "") -> str:
    """Render detailed circuit HTML with ablation analysis."""
    meta = graph.get("meta", {})
    layout = graph.get("layout", {})
    width = int(layout.get("width", 1100))
    height = int(layout.get("height", 560))
    
    per_node = circuit_edges.get("per_node", {}) if isinstance(circuit_edges, dict) else {}
    baseline_top_k = circuit_edges.get("baseline_top_k", []) if isinstance(circuit_edges, dict) else []
    per_node_ablation = circuit_edges.get("per_node_ablation_dist", {}) if isinstance(circuit_edges, dict) else {}
    combined_ablation = circuit_edges.get("combined_ablation_dist", {}) if isinstance(circuit_edges, dict) else {}
    token_combined_ablation = circuit_edges.get("token_combined_ablation_dist", {}) if isinstance(circuit_edges, dict) else {}
    
    # Sort nodes by C score (ascending - better importance first)
    sorted_nodes = sorted(per_node.items(), key=lambda x: x[1].get("C", float('inf')))
    
    # Use the exact same HTML structure as main page
    html_content = f"""<!doctype html>
<html>
<head>
  <meta charset="utf-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1" />
  <title>Detailed Circuit Analysis</title>
  <style>
    body {{
      font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
      margin: 0;
      background: #f5f5f5;
      color: #222;
    }}
    .wrap {{
      max-width: none;
      margin: 0;
      padding: 14px;
    }}
    .card {{
      background: #fff;
      border-radius: 10px;
      box-shadow: 0 2px 10px rgba(0,0,0,0.08);
      padding: 16px 18px;
      margin-bottom: 16px;
    }}
    .title {{
      font-size: 20px;
      font-weight: 700;
      margin: 0 0 8px 0;
      color: #283593;
    }}
    .meta {{
      font-size: 13px;
      color: #444;
      line-height: 1.4;
    }}
    .controls {{
      display: flex;
      flex-wrap: wrap;
      gap: 10px 14px;
      align-items: center;
      margin-bottom: 10px;
      font-size: 13px;
    }}
    #viz {{
      width: 100%;
      height: 93vh;
      overflow: auto;
    }}
    svg {{
      background: #fafafa;
      border-radius: 12px;
      border: 1px solid #e5e5e5;
      cursor: pointer;
    }}
    .tooltip {{
      position: fixed;
      pointer-events: none;
      background: rgba(20,20,20,0.92);
      color: #fff;
      padding: 8px 10px;
      border-radius: 8px;
      font-size: 12px;
      line-height: 1.35;
      max-width: 340px;
      z-index: 9999;
      display: none;
      box-shadow: 0 6px 16px rgba(0,0,0,0.25);
    }}
    table {{
      width: 100%;
      border-collapse: collapse;
      font-size: 12px;
    }}
    th, td {{
      padding: 6px 8px;
      border-bottom: 1px solid #eee;
      text-align: left;
      vertical-align: top;
      white-space: nowrap;
    }}
    th {{
      position: sticky;
      top: 0;
      background: #fff;
      z-index: 1;
    }}
    a {{
      color: #1976d2;
      text-decoration: none;
      cursor: pointer;
    }}
    a:hover {{
      text-decoration: underline;
    }}
    .influence-link {{
      color: #ff6b6b;
      font-weight: 500;
    }}
    .influence-link:hover {{
      color: #d63031;
      text-decoration: underline;
    }}
  </style>
</head>
<body>
  <div class="wrap">
    <div class="card">
      <div class="title">Detailed Circuit Analysis</div>
      <div class="meta">
        <div><b>Prompt</b>: {meta.get('prompt', '')}</div>
        <div><b>Circuit Nodes</b>: {len([n for n in graph.get('nodes', []) if n.get('kind') == 'neuron'])}</div>
        <div><b>Target</b>: {meta.get('target_str', '')}</div>
      </div>
    </div>

    <div class="card">
      <div class="controls">
        <label>
          Edge influence threshold (normalized by max |rel_change|):
          <input type="range" id="edgeThr" min="0" max="1" step="0.01" value="0" style="width: 200px;"/>
          <span id="edgeThrVal" style="min-width: 40px;">0.00</span>
        </label>
        <span style="font-size: 12px; color: #666;">(0 = show all, 1 = show only strongest)</span>
      </div>
      <div id="viz">
        <svg id="svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}"></svg>
      </div>
    </div>

    <!-- Sort nodes by C score once, to use in both ablation and influence sections -->
    <!-- (This is just for HTML generation, the actual sorting happens here) -->

    <div class="card">
      <div style="font-weight: 700; margin-bottom: 8px;">Baseline Output Distribution (Circuit Active, No Ablations)</div>
      <div style="overflow: auto;">
        <table style="font-size: 12px;">
          <thead>
            <tr>
              <th>Rank</th>
              <th>Token</th>
              <th>Probability</th>
              <th>Log Probability</th>
            </tr>
          </thead>
          <tbody>
"""
    
    # Add baseline distribution rows
    for rank, token_data in enumerate(baseline_top_k, 1):
        token_str = token_data.get("token_str", "[?]")
        prob = token_data.get("prob", 0.0)
        logp = token_data.get("logp", 0.0)
        html_content += f"""
            <tr>
              <td>{rank}</td>
              <td>{token_str}</td>
              <td>{prob:.4f}</td>
              <td>{logp:.4f}</td>
            </tr>
"""
    
    html_content += """
          </tbody>
        </table>
      </div>
    </div>

    <div class="card">
      <div style="font-weight: 700; margin-bottom: 8px;">Per-Node Output Distribution Changes (Ablation Analysis)</div>
      <div style="max-height: 80vh; overflow: auto;">
"""
    
    # Get per-node ablation data
    for node_id, node_info in sorted_nodes:
        if node_id not in per_node_ablation:
            continue

        ablation_entry = per_node_ablation[node_id]
        if isinstance(ablation_entry, dict) and "set_to_0" in ablation_entry:
            ablation_variants = []
            for key in ("set_to_0", "set_to_-5x", "set_to_-10x", "set_to_-4x"):
                if key in ablation_entry:
                    ablation_variants.append((key, ablation_entry.get(key, {})))
        else:
            # Backward compatibility: older payload with a single ablation profile
            ablation_variants = [("set_to_0", ablation_entry)]
        parts = node_id.split('_')
        L, k, i = int(parts[0]), int(parts[1]), int(parts[2])

        html_content += f'<div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 16px;"><div style="font-weight: 600; margin-bottom: 10px; color: #283593;">L{L}.{k}.{i}</div>'
        for ablation_label, ablation_info in ablation_variants:
            if ablation_label == "set_to_0":
                title = "set to 0"
            elif ablation_label == "set_to_-5x":
                title = "set to -5x (x=original activation)"
            elif ablation_label == "set_to_-10x":
                title = "set to -10x (x=original activation)"
            elif ablation_label == "set_to_-4x":
                title = "set to -4x (x=original activation)"
            else:
                title = str(ablation_label)
            html_content += f"""
          <div style="font-weight: 600; margin: 8px 0 8px 0; font-size: 12px; color: #424242;">Ablation mode: {title}</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
            <div>
              <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens After Ablation</div>
              <table style="font-size: 11px; width: 100%;">
                <tr><th>Token</th><th>Prob</th></tr>
"""
            for token_data in ablation_info.get("top_k_ablated", []):
                token_str = token_data.get("token_str", "[?]")
                prob = token_data.get("prob", 0.0)
                html_content += f'<tr><td>{token_str}</td><td>{prob:.4f}</td></tr>\n'
            html_content += """
              </table>
            </div>
            
            <div>
              <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens with Largest Probability Increase</div>
              <table style="font-size: 11px; width: 100%;">
                <tr><th>Token</th><th>ΔLogp</th></tr>
"""
            for token_data in ablation_info.get("top_increased_logp", []):
                token_str = token_data.get("token_str", "[?]")
                logp_change = token_data.get("logp_change", 0.0)
                html_content += f'<tr><td>{token_str}</td><td style="color: green; font-weight: 500;">+{logp_change:.4f}</td></tr>\n'
            html_content += """
              </table>
            </div>
          </div>
          
          <div>
            <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens with Largest Probability Decrease</div>
            <table style="font-size: 11px; width: 100%; margin-bottom: 10px;">
              <tr><th>Token</th><th>ΔLogp</th></tr>
"""
            for token_data in ablation_info.get("top_decreased_logp", []):
                token_str = token_data.get("token_str", "[?]")
                logp_change = token_data.get("logp_change", 0.0)
                html_content += f'冠军<tr>{token_str}</td><td style="color: red; font-weight: 500;">{logp_change:.4f}</td></tr>\n'
            html_content += """
            </table>
          </div>
"""
        html_content += "</div>\n"
    
    html_content += """
      </div>
    </div>
"""

    if combined_ablation:
        html_content += """
    <div class="card">
      <div style="font-weight: 700; margin-bottom: 8px;">Combined Ablation (same layer + token, multiple neurons)</div>
      <div style="max-height: 80vh; overflow: auto;">
"""
        for group_id in sorted(combined_ablation.keys()):
            group_info = combined_ablation[group_id]
            L = int(group_info.get("L", -1))
            k = int(group_info.get("k", -1))
            neurons = group_info.get("neurons", [])
            html_content += f"""
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
          <div style="font-weight: 600; margin-bottom: 6px; color: #283593;">L{L}.{k} combined</div>
          <div style="font-size: 12px; color: #555; margin-bottom: 8px;">Neurons: {', '.join(str(int(n)) for n in neurons)}</div>
"""
            for ablation_label in ("set_to_0", "set_to_-5x", "set_to_-10x", "set_to_-4x"):
                if ablation_label not in group_info:
                    continue
                ablation_info = group_info.get(ablation_label, {})
                if ablation_label == "set_to_0":
                    title = "set all to 0"
                elif ablation_label == "set_to_-5x":
                    title = "set all to -5x (x=original activation per neuron)"
                elif ablation_label == "set_to_-10x":
                    title = "set all to -10x (x=original activation per neuron)"
                elif ablation_label == "set_to_-4x":
                    title = "set all to -4x (x=original activation per neuron)"
                else:
                    title = str(ablation_label)
                html_content += f"""
          <div style="font-weight: 600; margin: 8px 0 8px 0; font-size: 12px; color: #424242;">Ablation mode: {title}</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
            <div>
              <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens After Ablation</div>
              <table style="font-size: 11px; width: 100%;">
                <tr><th>Token</th><th>Prob</th></tr>
"""
                for token_data in ablation_info.get("top_k_ablated", []):
                    token_str = token_data.get("token_str", "[?]")
                    prob = token_data.get("prob", 0.0)
                    html_content += f'<tr><td>{token_str}</td><td>{prob:.4f}</td></tr>\n'
                html_content += """
              </table>
            </div>
            <div>
              <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens with Largest Probability Increase</div>
              <table style="font-size: 11px; width: 100%;">
                <tr><th>Token</th><th>ΔLogp</th></tr>
"""
                for token_data in ablation_info.get("top_increased_logp", []):
                    token_str = token_data.get("token_str", "[?]")
                    logp_change = token_data.get("logp_change", 0.0)
                    html_content += f'<td>{token_str}</td><td style="color: green; font-weight: 500;">+{logp_change:.4f}</td></tr>\n'
                html_content += """
              </table>
            </div>
          </div>
          <div>
            <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens with Largest Probability Decrease</div>
            <table style="font-size: 11px; width: 100%; margin-bottom: 10px;">
              <tr><th>Token</th><th>ΔLogp</th></tr>
"""
                for token_data in ablation_info.get("top_decreased_logp", []):
                    token_str = token_data.get("token_str", "[?]")
                    logp_change = token_data.get("logp_change", 0.0)
                    html_content += f'<tr><td>{token_str}</td><td style="color: red; font-weight: 500;">{logp_change:.4f}</td></tr>\n'
                html_content += """
            </table>
          </div>
"""
            html_content += "</div>\n"
        html_content += """
      </div>
    </div>
"""

    if token_combined_ablation:
        html_content += """
    <div class="card">
      <div style="font-weight: 700; margin-bottom: 8px;">Combined Ablation by Token (across layers)</div>
      <div style="max-height: 80vh; overflow: auto;">
"""
        for token_group_id in sorted(token_combined_ablation.keys(), key=lambda x: int(str(x).split("_")[-1])):
            group_info = token_combined_ablation[token_group_id]
            k = int(group_info.get("k", -1))
            nodes = group_info.get("nodes", [])
            node_label = ", ".join(f"L{int(n.get('L', -1))}.i{int(n.get('i', -1))}" for n in nodes)
            html_content += f"""
        <div style="border: 1px solid #e0e0e0; border-radius: 8px; padding: 12px; margin-bottom: 16px;">
          <div style="font-weight: 600; margin-bottom: 6px; color: #283593;">Token k={k} combined</div>
          <div style="font-size: 12px; color: #555; margin-bottom: 8px;">Nodes: {node_label}</div>
"""
            for ablation_label in ("set_to_0", "set_to_-5x", "set_to_-10x", "set_to_-4x"):
                if ablation_label not in group_info:
                    continue
                ablation_info = group_info.get(ablation_label, {})
                if ablation_label == "set_to_0":
                    title = "set all to 0"
                elif ablation_label == "set_to_-5x":
                    title = "set all to -5x (x=original activation per neuron)"
                elif ablation_label == "set_to_-10x":
                    title = "set all to -10x (x=original activation per neuron)"
                elif ablation_label == "set_to_-4x":
                    title = "set all to -4x (x=original activation per neuron)"
                else:
                    title = str(ablation_label)
                html_content += f"""
          <div style="font-weight: 600; margin: 8px 0 8px 0; font-size: 12px; color: #424242;">Ablation mode: {title}</div>
          <div style="display: grid; grid-template-columns: 1fr 1fr; gap: 16px; margin-bottom: 12px;">
            <div>
              <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens After Ablation</div>
              <table style="font-size: 11px; width: 100%;">
                <tr><th>Token</th><th>Prob</th></tr>
"""
                for token_data in ablation_info.get("top_k_ablated", []):
                    token_str = token_data.get("token_str", "[?]")
                    prob = token_data.get("prob", 0.0)
                    html_content += f'<tr><td>{token_str}</td><td>{prob:.4f}</td></tr>\n'
                html_content += """
              <table>
            </div>
            <div>
              <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens with Largest Probability Increase</div>
              <table style="font-size: 11px; width: 100%;">
                <tr><th>Token</th><th>ΔLogp</th></tr>
"""
                for token_data in ablation_info.get("top_increased_logp", []):
                    token_str = token_data.get("token_str", "[?]")
                    logp_change = token_data.get("logp_change", 0.0)
                    html_content += f'<tr><td>{token_str}</td><td style="color: green; font-weight: 500;">+{logp_change:.4f}</td></tr>\n'
                html_content += """
              </table>
            </div>
          </div>
          <div>
            <div style="font-weight: 500; margin-bottom: 8px; font-size: 12px;">Top-10 Tokens with Largest Probability Decrease</div>
            <table style="font-size: 11px; width: 100%; margin-bottom: 10px;">
              <tr><th>Token</th><th>ΔLogp</th></table>
"""
                for token_data in ablation_info.get("top_decreased_logp", []):
                    token_str = token_data.get("token_str", "[?]")
                    logp_change = token_data.get("logp_change", 0.0)
                    html_content += f'<tr><td>{token_str}</td><td style="color: red; font-weight: 500;">{logp_change:.4f}</td></tr>\n'
                html_content += """
            </table>
          </div>
"""
            html_content += "</div>\n"
        html_content += """
      </div>
    </div>
"""

    html_content += """
    <div class="card">
      <div style="font-weight: 700; margin-bottom: 8px;">Per-Node Influence Analysis</div>
      <div style="max-height: 55vh; overflow: auto;">
      <table>
        <thead>
          <tr>
            <th>Node</th>
            <th>L</th>
            <th>Token</th>
            <th>C Score</th>
            <th>Base Act</th>
            <th>Influences Out (Top 5)</th>
            <th>Influenced By (Top 5)</th>
          </tr>
        </thead>
        <tbody>
"""
    
    # Add rows for each circuit node (already sorted by C score)
    for node_id, node_info in sorted_nodes:
        parts = node_id.split('_')
        L, k, i = int(parts[0]), int(parts[1]), int(parts[2])
        token_str = meta.get('token_strings', [])[k] if k < len(meta.get('token_strings', [])) else f"[{k}]"
        base_act = node_info.get("base_activation", 0)
        
        # Filter out self-edges from influences_out and influences_in
        influences_out_all = node_info.get("influences_out", [])
        influences_in_all = node_info.get("influences_in", [])
        
        # Remove self-edges: skip if target has same (k, L, i)
        influences_out = [inf for inf in influences_out_all if inf['target'] != node_id][:5]
        # Remove self-edges: skip if source has same (k, L, i)
        influences_in = [inf for inf in influences_in_all if inf['source'] != node_id][:5]
        
        # Build clickable links for influences_out
        out_parts = []
        for inf in influences_out:
            target_id = inf['target']
            target_parts = target_id.split('_')
            if len(target_parts) == 3:
                L_t, k_t, i_t = int(target_parts[0]), int(target_parts[1]), int(target_parts[2])
                rel_change = inf['rel_change']
                # Generate relative path to neuron dashboard
                neuron_path = _rel_link(detail_dir, neuron_html_root, L_t, i_t)
                if neuron_path:
                    out_parts.append(f'<a href="{neuron_path}" target="_blank" class="influence-link" title="L{L_t} i{i_t}">L{L_t}.{i_t}</a>({rel_change:+.2f})')
                else:
                    out_parts.append(f'<span class="influence-link" title="L{L_t} i{i_t}">L{L_t}.{i_t}</span>({rel_change:+.2f})')
        out_str = " ".join(out_parts) or "-"
        
        # Build clickable links for influences_in
        in_parts = []
        for inf in influences_in:
            source_id = inf['source']
            source_parts = source_id.split('_')
            if len(source_parts) == 3:
                L_s, k_s, i_s = int(source_parts[0]), int(source_parts[1]), int(source_parts[2])
                rel_change = inf['rel_change']
                # Generate relative path to neuron dashboard
                neuron_path = _rel_link(detail_dir, neuron_html_root, L_s, i_s)
                if neuron_path:
                    in_parts.append(f'<a href="{neuron_path}" target="_blank" class="influence-link" title="L{L_s} i{i_s}">L{L_s}.{i_s}</a>({rel_change:+.2f})')
                else:
                    in_parts.append(f'<span class="influence-link" title="L{L_s} i{i_s}">L{L_s}.{i_s}</span>({rel_change:+.2f})')
        in_str = " ".join(in_parts) or "-"
        
        html_link = f"javascript:alert('Open neuron L{L} i{i} dashboard')"
        C_score = node_info.get("C", float('inf'))
        C_score_str = f"{C_score:.4f}" if C_score != float('inf') else "—"
        html_content += f"""
          <tr>
            <td><a href="{html_link}">L{L}.{k}.{i}</a></td>
            <td>{L}</td>
            <td>k={k} {token_str}</td>
            <td>{C_score_str}</td>
            <td>{base_act:.4f}</td>
            <td><span class="small">{out_str}</span></td>
            <td><span class="small">{in_str}</span></td>
          </tr>
"""
    
    html_content += f"""
        </tbody>
      </table>
      </div>
    </div>
  </div>

  <div id="tooltip" class="tooltip"></div>

  <script>
    const graph = {json_dumps(graph)};

    function el(id) {{ return document.getElementById(id); }}
    const svg = el('svg');
    const tooltip = el('tooltip');
    const edgeThr = el('edgeThr');
    const edgeThrVal = el('edgeThrVal');

    const nodes = graph.nodes;
    const edges = graph.edges;
    const meta = graph.meta || {{}};

    function nodeById(id) {{
      return nodes.find(n => n.id === id);
    }}

    function fmt(x) {{
      if (x === null || x === undefined) return '';
      if (typeof x === 'number') return x.toFixed(4);
      return String(x);
    }}

    function showTooltip(html, x, y) {{
      tooltip.innerHTML = html;
      tooltip.style.left = (x + 14) + 'px';
      tooltip.style.top = (y + 14) + 'px';
      tooltip.style.display = 'block';
    }}
    function hideTooltip() {{
      tooltip.style.display = 'none';
    }}

    function render() {{
      while (svg.firstChild) svg.removeChild(svg.firstChild);
      
      const thr = parseFloat(edgeThr.value);
      edgeThrVal.textContent = thr.toFixed(2);

      const layout = graph.layout;
      const ml = layout.margin_left, mr = layout.margin_right, mt = layout.margin_top, mb = layout.margin_bottom;
      const W = layout.width, H = layout.height;
      const plotW = W - ml - mr;
      const plotH = H - mt - mb;
      const T = layout.T;
      const yRows = layout.n_layer + 2;

      function xForK(k) {{
        return ml + (T <= 1 ? 0 : (k * plotW / (T-1)));
      }}
      function yForRow(row) {{
        return mt + (plotH - row * (plotH / (yRows-1)));
      }}

      // grid lines (vertical)
      for (let k = 0; k < T; k++) {{
        const x = xForK(k);
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1', x); line.setAttribute('x2', x);
        line.setAttribute('y1', mt); line.setAttribute('y2', mt+plotH);
        line.setAttribute('stroke', '#e6e6e6');
        svg.appendChild(line);
      }}
      // grid lines (horizontal)
      for (let r = 0; r < yRows; r++) {{
        const y = yForRow(r);
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1', ml); line.setAttribute('x2', ml+plotW);
        line.setAttribute('y1', y); line.setAttribute('y2', y);
        line.setAttribute('stroke', '#ededed');
        svg.appendChild(line);
      }}

      // y labels
      const yLabels = ['Emb'];
      for (let L = 0; L < layout.n_layer; L++) yLabels.push('L'+L);
      yLabels.push('Lgt');
      for (let r = 0; r < yRows; r++) {{
        const y = yForRow(r);
        const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
        txt.textContent = yLabels[r];
        txt.setAttribute('x', 10);
        txt.setAttribute('y', y + 4);
        txt.setAttribute('fill', '#666');
        txt.setAttribute('font-size', '12');
        svg.appendChild(txt);
      }}

      // token labels
      for (let k = 0; k < T; k++) {{
        const tok = nodes.find(n => n.kind==='token' && n.k===k);
        if (!tok) continue;
        const txt = document.createElementNS('http://www.w3.org/2000/svg','text');
        txt.textContent = tok.label;
        txt.setAttribute('x', xForK(k));
        txt.setAttribute('y', H - 18);
        txt.setAttribute('fill', '#666');
        txt.setAttribute('font-size', '12');
        txt.setAttribute('text-anchor', 'end');
        svg.appendChild(txt);
      }}

      // edges
      const maxW = Math.max(...edges.map(e => e.weight || 0)) || 1;
      for (const e of edges) {{
        const s = nodeById(e.source);
        const t = nodeById(e.target);
        if (!s || !t) continue;
        const w = (e.weight || 0) / maxW;
        if (w < thr) continue;
        const line = document.createElementNS('http://www.w3.org/2000/svg','line');
        line.setAttribute('x1', s.x); line.setAttribute('y1', s.y);
        line.setAttribute('x2', t.x); line.setAttribute('y2', t.y);
        const base = e.kind === 'nn' ? 0.14 : (e.kind === 'logit' ? 0.18 : 0.10);
        const alpha = Math.max(base, Math.min(0.55, base + 0.45*w));
        line.setAttribute('stroke', `rgba(60,60,60,${{alpha}})`);
        line.setAttribute('stroke-width', (e.kind === 'nn' ? (0.5 + 2.2*w) : (0.4 + 1.4*w)).toFixed(2));
        svg.appendChild(line);
      }}

      // nodes
      function drawSquare(n, size, fill, stroke, strokeW) {{
        const s = document.createElementNS('http://www.w3.org/2000/svg','rect');
        s.setAttribute('x', n.x - size/2);
        s.setAttribute('y', n.y - size/2);
        s.setAttribute('width', size);
        s.setAttribute('height', size);
        s.setAttribute('rx', 2);
        s.setAttribute('fill', fill);
        s.setAttribute('stroke', stroke);
        s.setAttribute('stroke-width', strokeW);
        return s;
      }}
      function drawCircle(n, r, fill, stroke, strokeW) {{
        const c = document.createElementNS('http://www.w3.org/2000/svg','circle');
        c.setAttribute('cx', n.x); c.setAttribute('cy', n.y);
        c.setAttribute('r', r);
        c.setAttribute('fill', fill);
        c.setAttribute('stroke', stroke);
        c.setAttribute('stroke-width', strokeW);
        return c;
      }}
      
      for (const n of nodes) {{
        let shape;
        if (n.kind === 'token') {{
          shape = drawSquare(n, 8, '#ffffff', '#999', 1);
        }} else if (n.kind === 'logit') {{
          shape = drawSquare(n, 12, '#fff', '#111', 2);
        }} else {{
          // Neuron node with transparent styling
          const imp = Math.max(0, Math.min(1, n.importance || 0));
          const r = 4 + 8*imp;
          const act = Math.abs(n.activation || 0);
          const a = Math.max(0.12, Math.min(0.65, 0.10 + Math.log(1 + act)*0.28));
          const fill = `rgba(76,175,80,${{a}})`;  // green with transparency
          shape = drawCircle(n, r, fill, '#111', 2);
          if (imp > 0.45) shape.setAttribute('stroke', '#ff00b3');
        }}

        shape.style.cursor = (n.kind === 'neuron') ? 'pointer' : 'default';
        shape.addEventListener('mousemove', (evt) => {{
          if (n.kind === 'token') {{
            showTooltip(`<b>token</b> k=${{n.k}}<br/>${{n.label}}`, evt.clientX, evt.clientY);
          }} else if (n.kind === 'logit') {{
            showTooltip(`<b>logit</b><br/>${{n.label}}`, evt.clientX, evt.clientY);
          }} else {{
            showTooltip(
              `<b>neuron</b> k=${{n.k}} L=${{n.layer}} i=${{n.neuron}}` +
              `<br/><b>act</b>=${{fmt(n.activation)}}  <b>C</b>=${{fmt(n.C)}}`,
              evt.clientX, evt.clientY
            );
          }}
        }});
        shape.addEventListener('mouseleave', () => hideTooltip());
        if (n.kind === 'neuron' && n.html_link) {{
          shape.addEventListener('click', () => {{ window.open(n.html_link, '_blank'); }});
        }}
        svg.appendChild(shape);
      }}
    }}

    render();
    edgeThr.addEventListener('input', render);
  </script>
</body>
</html>
"""
    return html_content