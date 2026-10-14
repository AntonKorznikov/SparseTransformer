"""
Create an HTML dashboard for neuron visualizations.

This creates an HTML dashboard showing neuron activations with sequences
where neurons fire, similar to SAE visualization dashboards.
"""

from __future__ import annotations

import argparse
import json
import os
import pickle

import numpy as np
import torch

from model_adapter import create_hooked_model

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

_MODEL_CACHE: dict[tuple[str, str], object] = {}


def _get_cached_model(ckpt_path: str, device: str) -> object:
    """Process-local cache so multiple layer renders don't reload the model."""
    key = (os.path.abspath(ckpt_path), str(device))
    cached = _MODEL_CACHE.get(key)
    if cached is not None:
        return cached
    model = create_hooked_model(ckpt_path, device)
    model.eval()
    _MODEL_CACHE[key] = model
    return model


def escape_html(text: str) -> str:
    """Escape HTML special characters in the given string."""
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def normalize_embedding_matrix(matrix: np.ndarray) -> np.ndarray:
    """Return row-wise normalized embeddings (safe for zero vectors)."""
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1.0, norms)
    return matrix / (norms + 1e-8)


def top_k_positive_negative_sims(similarities: np.ndarray, k: int) -> tuple[list[tuple[int, float]], list[tuple[int, float]]]:
    """Return top-k most positive and top-k most negative cosine similarities."""
    if similarities is None or similarities.size == 0:
        return [], []
    k = min(k, len(similarities))
    positive_indices = np.argsort(-similarities)[:k]
    negative_indices = np.argsort(similarities)[:k]
    positives = [(int(idx), float(similarities[idx])) for idx in positive_indices]
    negatives = [(int(idx), float(similarities[idx])) for idx in negative_indices]
    return positives, negatives


def dedup_positions(positions: list[int], tokens: np.ndarray, radius: int, max_examples: int) -> list[int]:
    """Return up to max_examples positions while deduping similar contexts."""
    seen = set()
    deduped = []
    length = len(tokens)
    for pos in positions:
        start = max(0, pos - radius)
        end = min(length, pos + radius + 1)
        snippet = tuple(int(tokens[i]) for i in range(start, end))
        if snippet in seen:
            continue
        seen.add(snippet)
        deduped.append(pos)
        if len(deduped) >= max_examples:
            break
    return deduped


def select_top_positions(
    activated_positions: np.ndarray,
    activations: np.ndarray,
    tokens: np.ndarray,
    max_examples: int,
    dedup_radius: int,
) -> list[int]:
    """Select top activations (largest values) after deduping similar contexts."""
    if activated_positions.size == 0:
        return []
    valid_positions = [p for p in activated_positions.tolist() if 0 <= p < len(activations)]
    sorted_positions = sorted(valid_positions, key=lambda p: activations[p], reverse=True)
    return dedup_positions(sorted_positions, tokens, dedup_radius, max_examples)


def select_closest_positions(
    activated_positions: np.ndarray,
    activations: np.ndarray,
    target_value: float,
    tokens: np.ndarray,
    max_examples: int,
    dedup_radius: int,
) -> list[int]:
    """Select activations closest to the target value with deduplication."""
    if activated_positions.size == 0:
        return []
    valid_positions = [p for p in activated_positions.tolist() if 0 <= p < len(activations)]
    sorted_positions = sorted(
        valid_positions,
        key=lambda p: abs(activations[p] - target_value),
    )
    return dedup_positions(sorted_positions, tokens, dedup_radius, max_examples)


def get_next_token_predictions(
    model,
    tokens: np.ndarray,
    pos: int,
    block_size: int,
    top_k: int,
    device: str,
) -> list[tuple[int, float]]:
    """Return top-k next-token probabilities after position `pos`."""
    seq_start = (pos // block_size) * block_size
    context = tokens[seq_start : pos + 1]
    if context.size == 0:
        return []
    device_tensor = torch.device(device)
    context_tensor = torch.tensor(context, dtype=torch.long, device=device_tensor).unsqueeze(0)
    with torch.no_grad():
        logits = model(context_tensor)
    last_logits = logits[0, -1]
    probs = torch.softmax(last_logits, dim=-1)
    topk = torch.topk(probs, top_k)
    return [
        (int(idx), float(prob))
        for idx, prob in zip(topk.indices.tolist(), topk.values.tolist())
    ]


def format_prediction_html(predictions: list[tuple[int, float]], enc) -> str:
    """Format the next-token predictions as HTML chips."""
    if not predictions:
        return ""
    chips = []
    for token_idx, prob in predictions:
        if enc is None:
            token_str = f"[{int(token_idx)}]"
        else:
            token_str = escape_html(enc.decode([token_idx]))
        chips.append(
            f'<span class="next-token-chip">{token_str}<span class="next-token-probability">{prob:.1%}</span></span>'
        )
    return '<span class="next-token-label">Next tokens:</span> ' + " ".join(chips)


def render_cosine_subsection(
    title: str,
    token_sim_pairs: list[tuple[int, float]],
    enc,
    positive_default: bool = True,
) -> str:
    """Render a subsection for cosine similarity tokens."""
    if not token_sim_pairs:
        return ""
    subsection = f"""
            <div class="cosine-subheader">{title}</div>
            <div>
"""
    for token_idx, sim_val in token_sim_pairs:
        if enc is None:
            token_str = f"[{int(token_idx)}]"
        else:
            token_str = escape_html(enc.decode([token_idx]))
        css_class = "cosine-positive" if positive_default or sim_val >= 0 else "cosine-negative"
        subsection += f"""                <span class="cosine-item {css_class}">{token_str}<span class="cosine-value">{sim_val:+.4f}</span></span>
"""
    subsection += """            </div>
"""
    return subsection


def render_cosine_group(
    title: str,
    positive_pairs: list[tuple[int, float]],
    negative_pairs: list[tuple[int, float]],
    enc,
    positive_default: bool = True,
) -> str:
    """Render a grouped cosine block with a section header."""
    if not positive_pairs and not negative_pairs:
        return ""
    section = f'            <div class="cosine-header">{title}</div>\n'
    section += render_cosine_subsection("Top 20 positive", positive_pairs, enc, positive_default=positive_default)
    section += render_cosine_subsection("Top 20 negative", negative_pairs, enc, positive_default=False)
    return section


NEURON_PAGE_STYLE = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            background: #f5f5f5;
        }
        .page-wrapper {
            max-width: 1100px;
            margin: 0 auto;
        }
        h1 {
            color: #333;
            border-bottom: 2px solid #4CAF50;
            padding-bottom: 10px;
        }
        .neuron-card {
            background: white;
            border-radius: 8px;
            padding: 25px;
            margin: 25px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }
        .neuron-header {
            font-size: 28px;
            font-weight: bold;
            color: #2196F3;
            margin-bottom: 20px;
            padding-bottom: 10px;
            border-bottom: 2px solid #e0e0e0;
        }
        .stats {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
            gap: 15px;
            margin: 20px 0;
        }
        .stat {
            background: #f9f9f9;
            padding: 12px;
            border-radius: 6px;
            border-left: 3px solid #4CAF50;
        }
        .stat-label {
            font-size: 11px;
            color: #666;
            text-transform: uppercase;
            letter-spacing: 0.5px;
        }
        .stat-value {
            font-size: 22px;
            font-weight: bold;
            color: #333;
            margin-top: 5px;
        }
        .examples-section {
            margin-top: 30px;
        }
        .examples-header {
            font-size: 18px;
            font-weight: 600;
            color: #555;
            margin-bottom: 15px;
        }
        .example-sequence {
            background: #fafafa;
            border: 1px solid #e0e0e0;
            border-radius: 6px;
            padding: 15px;
            margin: 15px 0;
            font-family: 'Courier New', monospace;
            font-size: 14px;
            line-height: 1.8;
            word-wrap: break-word;
        }
        .example-header {
            font-size: 12px;
            color: #888;
            margin-bottom: 10px;
            font-weight: 600;
        }
        .token {
            padding: 2px 4px;
            margin: 0 1px;
            border-radius: 3px;
            display: inline-block;
        }
        .token-highlight {
            background: #fff3cd;
            border: 1px solid #ffc107;
            font-weight: 600;
            color: #856404;
        }
        .token-highlight-strong {
            background: #ffeb3b;
            border: 1px solid #ff9800;
            font-weight: 700;
            color: #e65100;
        }
        .activation-value {
            font-size: 11px;
            color: #666;
            margin-left: 5px;
        }
        .cosine-section {
            margin-top: 25px;
            padding-top: 20px;
            border-top: 1px solid #e0e0e0;
        }
        .cosine-header {
            font-size: 16px;
            font-weight: 600;
            color: #555;
            margin-bottom: 12px;
        }
        .cosine-item {
            display: inline-block;
            padding: 6px 10px;
            margin: 4px;
            border-radius: 5px;
            font-family: monospace;
            font-size: 13px;
            font-weight: 500;
        }
        .cosine-positive {
            background: #ffcdd2;
            color: #b71c1c;
            border: 2px solid #d32f2f;
        }
        .cosine-negative {
            background: #bbdefb;
            color: #0d47a1;
            border: 2px solid #1976d2;
        }
        .cosine-value {
            font-weight: 700;
            margin-left: 6px;
        }
        .next-token-predictions {
            font-size: 12px;
            color: #444;
            margin-top: 10px;
        }
        .next-token-label {
            font-weight: 600;
            margin-right: 6px;
        }
        .next-token-chip {
            background: #f0f0f0;
            border-radius: 4px;
            padding: 2px 6px;
            margin-right: 6px;
            display: inline-flex;
            align-items: center;
        }
        .next-token-probability {
            font-size: 11px;
            margin-left: 4px;
            color: #333;
        }
    </style>
"""

INDEX_STYLE = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        .search-shell {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        h1 {
            margin-bottom: 15px;
            color: #283593;
        }
        .search-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
        }
        .search-bar input {
            flex: 1;
            padding: 10px 12px;
            font-size: 16px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        .search-bar button {
            padding: 10px 18px;
            font-size: 16px;
            border-radius: 6px;
            border: none;
            background: #1976d2;
            color: white;
            cursor: pointer;
        }
        .search-bar button:hover {
            background: #115293;
        }
        .neuron-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }
        .neuron-list button {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 6px 10px;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        .neuron-list button:hover {
            background: #eceff1;
        }
        .neuron-list button.active {
            background: #d0e1f9;
            border-color: #1976d2;
        }
        .preview-frame iframe {
            width: 100%;
            height: 70vh;
            border: 1px solid #ccc;
            border-radius: 10px;
        }
    </style>
"""

INDEX_SCRIPT_TEMPLATE = """
    <script>
        const neurons = {neurons_json};
        const layerIdx = {layer_idx};
        const iframe = document.getElementById('neuronFrame');
        const list = document.getElementById('neuronList');
        const searchInput = document.getElementById('neuronSearch');

        function loadNeuron(identifier) {{
            if (!identifier) {{
                return;
            }}
            const normalized = identifier.trim();
            const match = neurons.find(n => n.neuron.toString() === normalized || n.file_name.includes(normalized));
            const fileName = match ? match.file_name : `layer${{layerIdx}}_neuron${{normalized}}.html`;
            iframe.src = fileName;
            highlightButton(match ? match.neuron : normalized);
        }}

        function highlightButton(neuronId) {{
            document.querySelectorAll('#neuronList button').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.neuron === neuronId.toString());
            }});
        }}

        function filterList(term) {{
            const normalized = term.trim();
            document.querySelectorAll('#neuronList button').forEach(btn => {{
                const matches = btn.dataset.neuron.includes(normalized);
                btn.style.display = normalized === '' || matches ? '' : 'none';
            }});
        }}

        document.getElementById('searchForm').addEventListener('submit', event => {{
            event.preventDefault();
            filterList(searchInput.value);
            loadNeuron(searchInput.value);
        }});

        document.getElementById('searchButton').addEventListener('click', () => {{
            filterList(searchInput.value);
            loadNeuron(searchInput.value);
        }});

        searchInput.addEventListener('input', () => {{
            filterList(searchInput.value);
        }});

        if (neurons.length > 0) {{
            iframe.src = neurons[0].file_name;
            highlightButton(neurons[0].neuron);
        }}
    </script>
"""


GLOBAL_INDEX_STYLE = """
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 0;
            background: #f5f5f5;
        }
        .search-shell {
            max-width: 1200px;
            margin: 0 auto;
            padding: 30px 20px;
        }
        h1 {
            margin-bottom: 15px;
            color: #283593;
        }
        .search-bar {
            display: flex;
            gap: 10px;
            margin-bottom: 10px;
            align-items: center;
            flex-wrap: wrap;
        }
        .search-bar select,
        .search-bar input {
            padding: 10px 12px;
            font-size: 16px;
            border-radius: 6px;
            border: 1px solid #ccc;
        }
        .search-bar input {
            flex: 1;
            min-width: 240px;
        }
        .search-bar button {
            padding: 10px 18px;
            font-size: 16px;
            border-radius: 6px;
            border: none;
            background: #1976d2;
            color: white;
            cursor: pointer;
        }
        .search-bar button:hover {
            background: #115293;
        }
        .neuron-list {
            display: flex;
            flex-wrap: wrap;
            gap: 8px;
            margin-bottom: 20px;
        }
        .neuron-list button {
            background: #fff;
            border: 1px solid #ddd;
            border-radius: 4px;
            padding: 6px 10px;
            cursor: pointer;
            transition: background 0.2s ease;
        }
        .neuron-list button:hover {
            background: #eceff1;
        }
        .neuron-list button.active {
            background: #d0e1f9;
            border-color: #1976d2;
        }
        .preview-frame iframe {
            width: 100%;
            height: 70vh;
            border: 1px solid #ccc;
            border-radius: 10px;
        }
        .hint {
            font-size: 13px;
            color: #444;
            margin-bottom: 12px;
        }
    </style>
"""


GLOBAL_INDEX_SCRIPT_TEMPLATE = """
    <script>
        const layers = {layers_json};
        const exampleNeuronsByLayer = {example_neurons_json};
        const iframe = document.getElementById('neuronFrame');
        const list = document.getElementById('neuronList');
        const layerSelect = document.getElementById('layerSelect');
        const searchInput = document.getElementById('neuronSearch');

        function currentLayer() {{
            return parseInt(layerSelect.value, 10);
        }}

        function neuronFile(layerIdx, neuronId) {{
            const normalized = neuronId.toString().trim();
            return `layer${{layerIdx}}/layer${{layerIdx}}_neuron${{normalized}}.html`;
        }}

        function highlightButton(neuronId) {{
            document.querySelectorAll('#neuronList button').forEach(btn => {{
                btn.classList.toggle('active', btn.dataset.neuron === neuronId.toString());
            }});
        }}

        function loadNeuron(identifier) {{
            if (!identifier) {{
                return;
            }}
            const layerIdx = currentLayer();
            const normalized = identifier.toString().trim();
            iframe.src = neuronFile(layerIdx, normalized);
            highlightButton(normalized);
        }}

        function rebuildButtons() {{
            const layerIdx = currentLayer();
            const example = exampleNeuronsByLayer[layerIdx] || [];
            list.innerHTML = example.map(n => `<button data-neuron=\"${{n}}\" onclick=\"loadNeuron('${{n}}')\">Neuron ${{n}}</button>`).join(' ');
        }}

        function filterList(term) {{
            const normalized = term.toString().trim();
            document.querySelectorAll('#neuronList button').forEach(btn => {{
                const matches = btn.dataset.neuron.includes(normalized);
                btn.style.display = normalized === '' || matches ? '' : 'none';
            }});
        }}

        document.getElementById('searchForm').addEventListener('submit', event => {{
            event.preventDefault();
            filterList(searchInput.value);
            loadNeuron(searchInput.value);
        }});

        document.getElementById('searchButton').addEventListener('click', () => {{
            filterList(searchInput.value);
            loadNeuron(searchInput.value);
        }});

        searchInput.addEventListener('input', () => {{
            filterList(searchInput.value);
        }});

        layerSelect.addEventListener('change', () => {{
            rebuildButtons();
            filterList(searchInput.value);
        }});

        if (layers.length > 0) {{
            layerSelect.value = layers[0].toString();
            rebuildButtons();
            const firstLayer = currentLayer();
            const firstNeuron = (exampleNeuronsByLayer[firstLayer] || [])[0];
            if (firstNeuron !== undefined) {{
                iframe.src = neuronFile(firstLayer, firstNeuron);
                highlightButton(firstNeuron);
            }}
        }}
    </script>
"""


def neuron_page_template(title: str, body: str) -> str:
    """Wrap the neuron card HTML in a standalone page."""
    return f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>{escape_html(title)}</title>
    {NEURON_PAGE_STYLE}
</head>
<body>
    <div class="page-wrapper">
        <h1>{escape_html(title)}</h1>
        {body}
    </div>
</body>
</html>
"""


def create_search_index(output_dir: str, layer_idx: int, neurons: list[dict], max_buttons: int = 20) -> None:
    """Create a landing page with a search box that loads individual neuron pages."""
    if not neurons:
        return
    neuron_buttons = []
    for entry in neurons[:max_buttons]:
        neuron_buttons.append(
            f'<button data-neuron="{entry["neuron"]}" onclick="loadNeuron(\'{entry["neuron"]}\')">Neuron {entry["neuron"]}</button>'
        )
    neurons_json = json.dumps(neurons[:max_buttons], ensure_ascii=False)
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Layer {layer_idx} Neurons</title>
    {INDEX_STYLE}
</head>
<body>
    <div class="search-shell">
        <h1>Layer {layer_idx} neuron search</h1>
        <form id="searchForm" class="search-bar">
            <input id="neuronSearch" type="text" placeholder="Enter neuron number..." />
            <button id="searchButton" type="button">Load neuron</button>
        </form>
        <div class="hint">Buttons show only the first {max_buttons} neurons. You can still type any neuron id and press Load.</div>
        <div id="neuronList" class="neuron-list">
            {' '.join(neuron_buttons)}
        </div>
        <div class="preview-frame">
            <iframe id="neuronFrame" src=""></iframe>
        </div>
    </div>
    {INDEX_SCRIPT_TEMPLATE.format(neurons_json=neurons_json, layer_idx=layer_idx)}
</body>
</html>
"""
    index_path = os.path.join(output_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)


def create_global_search_index(
    root_output_dir: str,
    layers: list[int],
    example_neurons_by_layer: dict[int, list[int]],
    max_buttons: int = 20,
) -> None:
    """Create a top-level index.html that can navigate layer + neuron pages."""
    if not layers:
        return
    layers_sorted = sorted(int(x) for x in layers)
    # Ensure the example lists are capped.
    capped = {int(k): [int(x) for x in v[:max_buttons]] for k, v in example_neurons_by_layer.items()}
    layers_json = json.dumps(layers_sorted, ensure_ascii=False)
    example_json = json.dumps(capped, ensure_ascii=False)
    layer_options = "\n".join([f'<option value="{li}">Layer {li}</option>' for li in layers_sorted])
    index_html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="utf-8">
    <title>Neuron dashboards</title>
    {GLOBAL_INDEX_STYLE}
</head>
<body>
    <div class="search-shell">
        <h1>Neuron dashboard</h1>
        <form id="searchForm" class="search-bar">
            <select id="layerSelect">
                {layer_options}
            </select>
            <input id="neuronSearch" type="text" placeholder="Enter neuron number..." />
            <button id="searchButton" type="button">Load neuron</button>
        </form>
        <div class="hint">Buttons show only the first {max_buttons} neurons for the selected layer. You can still type any neuron id and press Load.</div>
        <div id="neuronList" class="neuron-list"></div>
        <div class="preview-frame">
            <iframe id="neuronFrame" src=""></iframe>
        </div>
    </div>
    {GLOBAL_INDEX_SCRIPT_TEMPLATE.format(layers_json=layers_json, example_neurons_json=example_json)}
</body>
</html>
"""
    index_path = os.path.join(root_output_dir, "index.html")
    with open(index_path, "w", encoding="utf-8") as f:
        f.write(index_html)
def render_example_section(
    title: str,
    positions: list[int],
    activations: np.ndarray,
    token_strings: list[str],
    tokens: np.ndarray,
    context_window: int,
    block_size: int,
    prediction_model,
    prediction_device: str,
    next_token_top_k: int,
    enc,
    max_activation: float,
) -> str:
    """Render a block of sequence examples for the given activation positions."""
    if not positions:
        return ""
    section = f'            <div class="examples-header">{title}</div>\n'
    examples_added = 0
    highlight_threshold = max_activation * 0.7 if max_activation > 0 else 0.0
    for pos in positions:
        if pos >= len(token_strings):
            continue
        act_val = activations[pos]
        if act_val <= 0:
            continue
        examples_added += 1
        start_pos = max(0, pos - context_window)
        end_pos = min(len(token_strings), pos + context_window + 1)
        sequence_html = []
        for i in range(start_pos, end_pos):
            token_str = token_strings[i]
            if i == pos and act_val > 0:
                token_class = "token token-highlight-strong" if act_val > highlight_threshold else "token token-highlight"
                sequence_html.append(
                    f'<span class="{token_class}">{token_str}<span class="activation-value">({act_val:.3f})</span></span>'
                )
            else:
                sequence_html.append(f'<span class="token">{token_str}</span>')
        prediction_block = ""
        if prediction_model is not None:
            predictions = get_next_token_predictions(
                prediction_model,
                tokens,
                pos,
                block_size,
                next_token_top_k,
                prediction_device,
            )
            formatted = format_prediction_html(predictions, enc)
            if formatted:
                prediction_block = f'                <div class="next-token-predictions">{formatted}</div>\n'
        section += f"""
            <div class="example-sequence">
                <div class="example-header">Example {examples_added} - Position {pos}, Post-BatchTopK Activation: {act_val:.4f}</div>
                <div>{"".join(sequence_html)}</div>
"""
        if prediction_block:
            section += prediction_block
        section += "            </div>\n"
    if examples_added == 0:
        return ""
    return section


def render_example_section_from_examples(
    title: str,
    examples: list[dict[str, float]],
    token_strings: list[str],
    tokens: np.ndarray,
    context_window: int,
    block_size: int,
    prediction_model,
    prediction_device: str,
    next_token_top_k: int,
    enc,
    max_activation: float,
) -> str:
    if not examples:
        return ""
    section = f'            <div class="examples-header">{title}</div>\n'
    examples_added = 0
    highlight_threshold = max_activation * 0.7 if max_activation > 0 else 0.0
    for example in examples:
        pos = example.get("pos")
        act_val = example.get("activation", 0.0)
        if pos is None or pos < 0 or pos >= len(token_strings) or act_val <= 0:
            continue
        examples_added += 1
        start_pos = max(0, pos - context_window)
        end_pos = min(len(token_strings), pos + context_window + 1)
        sequence_html = []
        for i in range(start_pos, end_pos):
            token_str = token_strings[i]
            if i == pos:
                token_class = "token token-highlight-strong" if act_val > highlight_threshold else "token token-highlight"
                sequence_html.append(
                    f'<span class="{token_class}">{token_str}<span class="activation-value">({act_val:.3f})</span></span>'
                )
            else:
                sequence_html.append(f'<span class="token">{token_str}</span>')
        prediction_block = ""
        if prediction_model is not None:
            predictions = get_next_token_predictions(
                prediction_model,
                tokens,
                pos,
                block_size,
                next_token_top_k,
                prediction_device,
            )
            formatted = format_prediction_html(predictions, enc)
            if formatted:
                prediction_block = f'                <div class="next-token-predictions">{formatted}</div>\n'
        section += f"""
            <div class="example-sequence">
                <div class="example-header">Example {examples_added} - Position {pos}, Post-BatchTopK Activation: {act_val:.4f}</div>
                <div>{"".join(sequence_html)}</div>
"""
        if prediction_block:
            section += prediction_block
        section += "            </div>\n"
    if examples_added == 0:
        return ""
    return section


def render_neuron_card(
    neuron_idx: int,
    neuron_activations: np.ndarray | None,
    tokens: np.ndarray,
    token_strings: list[str],
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    normalized_input_embeddings: np.ndarray | None,
    normalized_unembedding: np.ndarray | None,
    context_window: int,
    block_size: int,
    prediction_model,
    prediction_device: str,
    next_token_top_k: int,
    max_examples_per_quantile: int,
    dedup_radius: int,
    enc,
    neuron_example: dict | None = None,
) -> str:
    """Build a single neuron card HTML snippet."""
    down_cos_pos, down_cos_neg = [], []
    up_cos_pos, up_cos_neg = [], []
    if normalized_unembedding is not None:
        down_proj_vec = decoder_weights[:, neuron_idx]
        down_proj_norm = down_proj_vec / (np.linalg.norm(down_proj_vec) + 1e-8)
        down_proj_sims = normalized_unembedding @ down_proj_norm
        down_cos_pos, down_cos_neg = top_k_positive_negative_sims(down_proj_sims, 20)
    if normalized_input_embeddings is not None:
        up_proj_vec = encoder_weights[neuron_idx]
        up_proj_norm = up_proj_vec / (np.linalg.norm(up_proj_vec) + 1e-8)
        up_proj_sims = normalized_input_embeddings @ up_proj_norm
        up_cos_pos, up_cos_neg = top_k_positive_negative_sims(up_proj_sims, 20)

    if neuron_example is not None:
        stats = neuron_example.get('stats', {})
        mean_act = float(stats.get('mean', 0.0))
        max_act = float(stats.get('max', 0.0))
        std_act = float(stats.get('std', 0.0))
        activation_count = int(stats.get('activation_count', 0))
        activation_pct = float(stats.get('activation_pct', 0.0)) * 100.0
        E = float(stats.get('E', 0.0))
        examples = neuron_example.get('examples', {})
        top_examples = examples.get('top', [])
        mid_examples = examples.get('mid', [])
        low_examples = examples.get('low', [])
        top_section = render_example_section_from_examples(
            f'Top Activations (Mean E = {E:.4f})',
            top_examples,
            token_strings,
            tokens,
            context_window,
            block_size,
            prediction_model,
            prediction_device,
            next_token_top_k,
            enc,
            max_act,
        )
        mid_section = render_example_section_from_examples(
            f'Medium Activations (around 0.5*E = {0.5*E:.4f})',
            mid_examples,
            token_strings,
            tokens,
            context_window,
            block_size,
            prediction_model,
            prediction_device,
            next_token_top_k,
            enc,
            max_act,
        )
        low_section = render_example_section_from_examples(
            f'Low Activations (around 0.1*E = {0.1*E:.4f})',
            low_examples,
            token_strings,
            tokens,
            context_window,
            block_size,
            prediction_model,
            prediction_device,
            next_token_top_k,
            enc,
            max_act,
        )
    else:
        if neuron_activations is None:
            raise ValueError('neuron_activations are required when no neuron example data is provided')
        mean_act = float(np.mean(neuron_activations))
        max_act = float(np.max(neuron_activations))
        std_act = float(np.std(neuron_activations))
        activated_positions = np.where(neuron_activations > 0)[0]
        activation_count = len(activated_positions)
        activation_pct = (
            float(activation_count / len(neuron_activations) * 100) if len(neuron_activations) > 0 else 0.0
        )
        if activation_count > 0:
            top_values = sorted(
                [(p, neuron_activations[p]) for p in activated_positions], key=lambda x: x[1], reverse=True
            )
            top_5_values = [val for _, val in top_values[:5]]
            E = float(np.mean(top_5_values)) if top_5_values else 0.0
            top_positions = select_top_positions(
                activated_positions,
                neuron_activations,
                tokens,
                max_examples_per_quantile,
                dedup_radius,
            )
            positions_05E = select_closest_positions(
                activated_positions,
                neuron_activations,
                0.5 * E,
                tokens,
                max_examples_per_quantile,
                dedup_radius,
            )
            positions_01E = select_closest_positions(
                activated_positions,
                neuron_activations,
                0.1 * E,
                tokens,
                max_examples_per_quantile,
                dedup_radius,
            )
        else:
            E = 0.0
            top_positions = []
            positions_05E = []
            positions_01E = []
        top_section = render_example_section(
            f'Top Activations (Mean E = {E:.4f})',
            top_positions,
            neuron_activations,
            token_strings,
            tokens,
            context_window,
            block_size,
            prediction_model,
            prediction_device,
            next_token_top_k,
            enc,
            max_act,
        )
        mid_section = render_example_section(
            f'Medium Activations (around 0.5*E = {0.5*E:.4f})',
            positions_05E,
            neuron_activations,
            token_strings,
            tokens,
            context_window,
            block_size,
            prediction_model,
            prediction_device,
            next_token_top_k,
            enc,
            max_act,
        )
        low_section = render_example_section(
            f'Low Activations (around 0.1*E = {0.1*E:.4f})',
            positions_01E,
            neuron_activations,
            token_strings,
            tokens,
            context_window,
            block_size,
            prediction_model,
            prediction_device,
            next_token_top_k,
            enc,
            max_act,
        )
    cos_section_groups = []
    down_group = render_cosine_group(
        "Down Projection vs Unembedding",
        down_cos_pos,
        down_cos_neg,
        enc,
        positive_default=True,
    )
    if down_group:
        cos_section_groups.append(down_group)
    up_group = render_cosine_group(
        "Up Projection vs Input Embeddings",
        up_cos_pos,
        up_cos_neg,
        enc,
        positive_default=True,
    )
    if up_group:
        cos_section_groups.append(up_group)
    cos_section_html = ""
    if cos_section_groups:
        cos_section_html = "        <div class=\"cosine-section\">\n" + "".join(cos_section_groups) + "        </div>\n"
    card = f"""
    <div class="neuron-card">
        <div class="neuron-header">Neuron {neuron_idx}</div>
        <div class="stats">
            <div class="stat">
                <div class="stat-label">Mean Activation</div>
                <div class="stat-value">{mean_act:.4f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Max Activation</div>
                <div class="stat-value">{max_act:.4f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Std Activation</div>
                <div class="stat-value">{std_act:.4f}</div>
            </div>
            <div class="stat">
                <div class="stat-label">Activation Rate</div>
                <div class="stat-value">{activation_pct:.2f}%</div>
            </div>
            <div class="stat">
                <div class="stat-label">Activation Count</div>
                <div class="stat-value">{activation_count}</div>
            </div>
        </div>
        {cos_section_html}
        <div class="examples-section">
            {top_section}
            {mid_section}
            {low_section}
        </div>
    """
    card += "    </div>\n"
    return card
def create_dashboard_html(
    activations: np.ndarray | None,
    tokens: np.ndarray,
    neuron_indices: list[int],
    encoder_weights: np.ndarray | None,
    decoder_weights: np.ndarray | None,
    layer_idx: int,
    hidden_size: int,
    output_dir: str,
    block_size: int,
    context_window: int = 30,
    max_examples_per_quantile: int = 10,
    dedup_radius: int = 3,
    next_token_top_k: int = 5,
    ckpt_path: str | None = None,
    prediction_device: str = "cpu",
    input_embeddings: np.ndarray | None = None,
    unembedding_matrix: np.ndarray | None = None,
    neuron_examples: dict[int, dict] | None = None,
    create_index: bool = True,
    index_max_buttons: int = 20,
    # Performance/caching knobs (optional)
    token_strings: list[str] | None = None,
    normalized_input_embeddings: np.ndarray | None = None,
    normalized_unembedding: np.ndarray | None = None,
    prediction_model=None,
    enable_next_token_predictions: bool = True,
):
    """Create per-neuron HTML files plus an index/search page."""
    enc = tiktoken.get_encoding("gpt2") if tiktoken is not None else None

    # Decode tokens once if caller didn't precompute them.
    if token_strings is None:
        print("Decoding tokens...")
        token_strings = []
        for t in tokens:
            if enc is None:
                token_strings.append(f"[{int(t)}]")
            else:
                try:
                    decoded = enc.decode([int(t)])
                    token_strings.append(escape_html(decoded))
                except Exception:
                    token_strings.append(f"[{int(t)}]")

    # Decide whether we will compute next-token predictions.
    enable_predictions = bool(enable_next_token_predictions) and int(next_token_top_k) > 0

    # If we need assets (weights/embeddings) but they weren't provided (new smaller pickles),
    # load them from the checkpoint.
    assets_model = prediction_model
    if (
        (input_embeddings is None and normalized_input_embeddings is None)
        or (unembedding_matrix is None and normalized_unembedding is None)
        or encoder_weights is None
        or decoder_weights is None
    ):
        if ckpt_path is None:
            raise ValueError(
                "Missing weights/embeddings in input data. Provide ckpt_path/--ckpt (or generate data with --store_large_matrices)."
            )
        # If predictions are on, load on prediction_device; otherwise CPU is fine for assets.
        assets_device = prediction_device if enable_predictions else "cpu"
        assets_model = assets_model or _get_cached_model(ckpt_path, assets_device)
        if input_embeddings is None and normalized_input_embeddings is None:
            input_embeddings = assets_model.model.wte.weight.data.cpu().numpy()
        if unembedding_matrix is None and normalized_unembedding is None:
            unembedding_matrix = assets_model.model.lm_head.weight.data.cpu().numpy()
        if encoder_weights is None or decoder_weights is None:
            layer_mlp = assets_model.model.blocks[layer_idx].mlp
            if encoder_weights is None:
                encoder_weights = layer_mlp.fc1.weight.data.cpu().numpy()
            if decoder_weights is None:
                decoder_weights = layer_mlp.fc2.weight.data.cpu().numpy()

    # Normalize embedding matrices if caller didn't provide pre-normalized versions.
    if normalized_unembedding is None:
        normalized_unembedding = (
            normalize_embedding_matrix(unembedding_matrix) if unembedding_matrix is not None else None
        )
    if normalized_input_embeddings is None:
        normalized_input_embeddings = (
            normalize_embedding_matrix(input_embeddings) if input_embeddings is not None else None
        )

    # Load prediction model only if needed and not already provided.
    if enable_predictions:
        if prediction_model is None:
            if ckpt_path is None:
                raise ValueError("ckpt_path is required to compute next-token predictions")
            print(f"Loading model from {ckpt_path} for next-token predictions...")
            prediction_model = _get_cached_model(ckpt_path, prediction_device)
        else:
            prediction_model.eval()
    else:
        prediction_model = None

    if activations is None and not neuron_examples:
        raise ValueError("Provide either raw activations or neuron_examples to build the dashboard")

    os.makedirs(output_dir, exist_ok=True)
    neuron_pages = []
    neuron_examples_map: dict[int, dict] = neuron_examples or {}

    for idx, neuron_idx in enumerate(neuron_indices):
        neuron_activations = activations[:, idx] if activations is not None else None
        neuron_example = neuron_examples_map.get(neuron_idx)
        card_body = render_neuron_card(
            neuron_idx=neuron_idx,
            neuron_activations=neuron_activations,
            tokens=tokens,
            token_strings=token_strings,
            encoder_weights=encoder_weights,
            decoder_weights=decoder_weights,
            normalized_input_embeddings=normalized_input_embeddings,
            normalized_unembedding=normalized_unembedding,
            context_window=context_window,
            block_size=block_size,
            prediction_model=prediction_model,
            prediction_device=prediction_device,
            next_token_top_k=next_token_top_k,
            max_examples_per_quantile=max_examples_per_quantile,
            dedup_radius=dedup_radius,
            enc=enc,
            neuron_example=neuron_example,
        )
        file_name = f"layer{layer_idx}_neuron{neuron_idx}.html"
        page_html = neuron_page_template(f"Layer {layer_idx} Neuron {neuron_idx}", card_body)
        with open(os.path.join(output_dir, file_name), "w", encoding="utf-8") as f:
            f.write(page_html)
        neuron_pages.append({"neuron": neuron_idx, "file_name": file_name})

    if create_index:
        create_search_index(output_dir, layer_idx, neuron_pages, max_buttons=index_max_buttons)
    print(f"Created {len(neuron_pages)} neuron pages plus index in {output_dir}")

def main():
    parser = argparse.ArgumentParser(description="Create HTML dashboard from saved data")
    parser.add_argument(
        "--data_file",
        type=str,
        default="out/A/neuron_dashboards/dashboard_data.pkl",
        help="Path to saved dashboard data pickle file",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="out/A/neuron_dashboards",
        help="Directory to save per-neuron HTML files (each layer gets its own subfolder)",
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default=None,
        help="Checkpoint path for next-token predictions (overrides saved path)",
    )
    parser.add_argument(
        "--prediction_device",
        type=str,
        default="cpu",
        help="Device used when computing next-token logits",
    )
    parser.add_argument(
        "--context_window",
        type=int,
        default=30,
        help="Context window size for displaying tokens",
    )
    parser.add_argument(
        "--max_examples_per_quantile",
        type=int,
        default=10,
        help="Maximum examples per quantile group (top, 0.5E, 0.1E)",
    )
    parser.add_argument(
        "--dedup_radius",
        type=int,
        default=3,
        help="Tokens on each side used to dedupe repeated contexts",
    )
    parser.add_argument(
        "--next_token_top_k",
        type=int,
        default=5,
        help="Number of next-token predictions to show per example",
    )
    parser.add_argument(
        "--no_next_token_predictions",
        action="store_true",
        help="Disable next-token predictions (faster; avoids loading a model for logits).",
    )
    args = parser.parse_args()
    
    print(f"Loading data from {args.data_file}")
    with open(args.data_file, "rb") as f:
        data = pickle.load(f)
    
    print("Creating HTML dashboard...")
    ckpt_path = args.ckpt or data.get('ckpt_path')
    layer_output_dir = os.path.join(args.output_dir, f"layer{data['layer_idx']}")
    create_dashboard_html(
        activations=data.get('activations'),
        tokens=data['tokens'],
        neuron_indices=data['neuron_indices'],
        encoder_weights=data.get('encoder_weights'),
        decoder_weights=data.get('decoder_weights'),
        layer_idx=data['layer_idx'],
        hidden_size=data['hidden_size'],
        output_dir=layer_output_dir,
        input_embeddings=data.get('input_embeddings'),
        unembedding_matrix=data.get('unembedding_matrix'),
        block_size=data.get('block_size', 256),
        context_window=args.context_window,
        max_examples_per_quantile=args.max_examples_per_quantile,
        dedup_radius=args.dedup_radius,
        next_token_top_k=args.next_token_top_k,
        ckpt_path=ckpt_path,
        prediction_device=args.prediction_device,
        neuron_examples=data.get('neuron_examples'),
        enable_next_token_predictions=(not args.no_next_token_predictions),
    )
    
    print(f"Dashboard saved to {layer_output_dir}")


if __name__ == "__main__":
    main()

