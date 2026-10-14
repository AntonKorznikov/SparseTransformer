#!/usr/bin/env python3
"""
Create PNG snapshots of existing neuron dashboard HTML pages with a tighter
example budget (default 4 per activation threshold). This script samples a
fixed number of neurons per layer, trims each HTML page down to the first
`--examples-per-threshold` sequences per section, and uses a headless browser
to render the result into a PNG that preserves the original dashboard styling.
"""

from __future__ import annotations

import argparse
import asyncio
import math
import random
import textwrap
import zlib
from pathlib import Path

try:
    from pyppeteer import launch
except ImportError as exc:  # pragma: no cover - installation-time guidance
    raise ImportError(
        "pyppeteer is required to render neuron dashboards to PNG. "
        "Install it via `pip install pyppeteer` before running this script."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Randomly sample neuron dashboards per layer and export PNGs."
    )
    parser.add_argument(
        "dashboard_root",
        type=Path,
        help="Path to the existing neuron dashboard output directory "
        "(e.g. out/.../neuron_dashboards).",
    )
    parser.add_argument(
        "--samples-per-layer",
        type=int,
        default=20,
        help="Number of random neurons to render per layer.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Directory where PNG snapshots will be saved. Defaults to "
        "`<dashboard_root>/png_samples`.",
    )
    parser.add_argument(
        "--examples-per-threshold",
        type=int,
        default=4,
        help="Maximum number of example sequences to keep per activation threshold.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=0,
        help="Random seed for reproducible sampling across layers.",
    )
    parser.add_argument(
        "--context-tokens",
        type=int,
        default=20,
        help="Number of tokens before/after the highlighted token to keep when trimming examples.",
    )
    return parser.parse_args()


def _iter_layer_dirs(root: Path) -> list[Path]:
    layers = [
        path
        for path in sorted(root.iterdir())
        if path.is_dir() and path.name.startswith("layer")
    ]
    return layers


def _select_html_files(
    layer_dir: Path, max_samples: int, seed: int
) -> list[Path]:
    html_files = sorted(
        filename
        for filename in layer_dir.iterdir()
        if filename.is_file() and filename.suffix == ".html"
    )
    if not html_files:
        return []
    rng = random.Random(seed + zlib.crc32(layer_dir.name.encode()))
    sample_count = min(max_samples, len(html_files))
    return rng.sample(html_files, sample_count)


async def _capture_png(
    browser,
    html_path: Path,
    png_path: Path,
    examples_per_threshold: int,
    context_tokens: int,
) -> None:
    page = await browser.newPage()
    await page.setViewport({"width": 1400, "height": 900})
    await page.goto(html_path.resolve().as_uri(), waitUntil="networkidle0")
    script = textwrap.dedent(
        """\
        (limit) => {
            document.querySelectorAll('.examples-section').forEach((section) => {
                let groupCount = 0;
                Array.from(section.children).forEach((child) => {
                    if (child.classList.contains('examples-header')) {
                        groupCount = 0;
                        return;
                    }
                    if (child.classList.contains('example-sequence')) {
                        groupCount += 1;
                        if (groupCount > limit) {
                            child.remove();
                        }
                    }
                });
            });
        }
        """
    )
    await page.evaluate(script, examples_per_threshold)
    await asyncio.sleep(0.3)
    png_path.parent.mkdir(parents=True, exist_ok=True)
    layout_script = textwrap.dedent(
        """
        (contextWindow) => {
            const card = document.querySelector('.neuron-card');
            const page = document.querySelector('.page-wrapper');
            if (!card || !page) {
                return;
            }
            let titleText = '';
            const headerElement = card.querySelector('.neuron-header');
            if (headerElement && headerElement.textContent) {
                titleText = headerElement.textContent.trim();
            }
            const stats = card.querySelector('.stats');
            const cosine = card.querySelector('.cosine-section');
            const examples = card.querySelector('.examples-section');
            const trimSequenceContext = (sequenceEl, windowSize) => {
                const tokens = Array.from(sequenceEl.querySelectorAll('.token'));
                const highlightIndex = tokens.findIndex(
                    (token) =>
                        token.classList.contains('token-highlight') ||
                        token.classList.contains('token-highlight-strong')
                );
                if (highlightIndex < 0) {
                    return;
                }
                const start = Math.max(0, highlightIndex - windowSize);
                const end = Math.min(tokens.length, highlightIndex + windowSize + 1);
                tokens.forEach((token, idx) => {
                    if (idx < start || idx >= end) {
                        token.remove();
                    }
                });
            };
            const removePredictions = (sequenceEl) => {
                const predictions = sequenceEl.querySelector('.next-token-predictions');
                if (predictions) {
                    predictions.remove();
                }
            };
            const blockGather = [];
            if (examples) {
                let current = null;
                Array.from(examples.children).forEach((child) => {
                    if (child.classList.contains('examples-header')) {
                        current = { title: child.textContent.trim(), nodes: [] };
                        blockGather.push(current);
                        return;
                    }
                    if (current && child.classList.contains('example-sequence')) {
                        const clone = child.cloneNode(true);
                        trimSequenceContext(clone, contextWindow);
                        removePredictions(clone);
                        current.nodes.push(clone);
                    }
                });
            }

            const createBlock = (title, nodes) => {
                const block = document.createElement('div');
                block.className = 'render-block';
                const header = document.createElement('div');
                header.className = 'render-block-title';
                header.textContent = title;
                block.appendChild(header);
                if (nodes.length === 0) {
                    const notice = document.createElement('div');
                    notice.textContent = 'No examples available';
                    notice.style.fontSize = '13px';
                    notice.style.color = '#666';
                    block.appendChild(notice);
                    return block;
                }
                nodes.forEach((node, idx) => {
                    const clone = node.cloneNode(true);
                    const exampleHeader = clone.querySelector('.example-header');
                    if (exampleHeader) {
                        exampleHeader.textContent = `Example ${idx + 1}`;
                    }
                    block.appendChild(clone);
                });
                return block;
            };

            const newWrapper = document.createElement('div');
            newWrapper.className = 'render-grid-wrapper';
            const info = document.createElement('div');
            info.className = 'render-info';
            const layerInfo = document.createElement('div');
            layerInfo.className = 'render-layer-info';
            const pageHeader = document.querySelector('h1');
            layerInfo.textContent = pageHeader ? pageHeader.textContent.trim() : titleText;
            info.appendChild(layerInfo);
            if (stats) {
                const statsClone = stats.cloneNode(true);
                statsClone.classList.add('render-stats-clone');
                statsClone.querySelectorAll('.stat').forEach((stat) => {
                    const label = stat.querySelector('.stat-label');
                    if (label) {
                        const labelText = label.textContent.toLowerCase();
                        if (labelText.includes('mean activation') || labelText.includes('std')) {
                            stat.remove();
                        }
                    }
                });
                info.appendChild(statsClone);
            }
            const grid = document.createElement('div');
            grid.className = 'render-grid';

            const logitBlock = createBlock(
                'Logits (cosine similarities)',
                cosine ? [cosine.cloneNode(true)] : []
            );
            logitBlock.classList.add('render-block-logits');
            logitBlock.style.rowGap = '4px';
            const logitCosine = logitBlock.querySelector('.cosine-section');
            if (logitCosine) {
                logitCosine.style.marginTop = '8px';
                logitCosine.style.paddingTop = '7px';
            }
            grid.appendChild(logitBlock);

            const topSection =
                blockGather.find((section) => /^top\\b/i.test(section.title)) || blockGather[0];
            const midSection =
                blockGather.find((section) => /medium\\b/i.test(section.title)) || blockGather[1];
            const lowSection =
                blockGather.find((section) => /^low\\b/i.test(section.title)) || blockGather[2];

            grid.appendChild(
                createBlock('Top Activations (highest peaks)', topSection ? topSection.nodes : [])
            );
            grid.appendChild(
                createBlock(
                    'Mid-range Activations (≈0.5×peak)',
                    midSection ? midSection.nodes : []
                )
            );
            grid.appendChild(
                createBlock('Low Activations (≈0.1×peak)', lowSection ? lowSection.nodes : [])
            );

            page.innerHTML = '';
            page.appendChild(newWrapper);
            newWrapper.appendChild(info);
            newWrapper.appendChild(grid);
            grid.style.display = 'grid';
            grid.style.gridTemplateColumns = '1fr 1fr';
            grid.style.columnGap = '12px';
            grid.style.rowGap = '12px';
            grid.style.width = '100%';
            const limitCosineItems = () => {
                newWrapper.querySelectorAll('.cosine-section').forEach((section) => {
                    section.querySelectorAll('div').forEach((div) => {
                        const items = Array.from(div.querySelectorAll('.cosine-item'));
                        items.forEach((item, idx) => {
                            if (idx >= 9) {
                                item.remove();
                            }
                        });
                    });
                });
            };
            limitCosineItems();

            const style = document.createElement('style');
            const cssRules = [
                '.render-grid-wrapper {',
                '    max-width: none;',
                '    width: 100%;',
                '    margin: 0;',
                '    padding: 20px 0;',
                '}',
                '.render-info {',
                '    text-align: center;',
                '    margin-bottom: 14px;',
                '}',
                '.render-layer-info {',
                '    font-size: 24px;',
                '    font-weight: 700;',
                '    margin-bottom: 4px;',
                '}',
                '.render-stats-clone {',
                '    margin: 0 auto;',
                '    max-width: 1000px;',
                '}',
                '.render-grid {',
                '    display: grid;',
                '    grid-template-columns: repeat(2, minmax(0, 1fr));',
                '    gap: 12px;',
                '}',
                '.render-block {',
                '    background: #ffffff;',
                '    border: 1px solid #e0e0e0;',
                '    border-radius: 10px;',
                '    padding: 12px;',
                '    box-shadow: 0 4px 12px rgba(0, 0, 0, 0.04);',
                '    display: flex;',
                '    flex-direction: column;',
                '    row-gap: 12px;',
                '    min-height: 160px;',
                '}',
                '.render-block-title {',
                '    font-size: 18px;',
                '    font-weight: 600;',
                '    color: #1f2933;',
                '}',
                '.render-block-logits {',
                '    row-gap: 4px;',
                '}',
                '.render-block-logits .cosine-section {',
                '    margin-top: 8px;',
                '    padding-top: 7px;',
                '}',
                '.render-block .example-sequence {',
                '    margin: 6px 0;',
                '    padding: 8px;',
                '    font-size: 12px;',
                '    line-height: 1.4;',
                '}',
            ];
            style.appendChild(document.createTextNode(cssRules.join('\\n')));
            document.head.appendChild(style);
            document.documentElement.style.margin = '0';
            document.documentElement.style.padding = '0';
            document.body.style.margin = '0';
            document.body.style.padding = '0';
            document.body.style.background = '#ffffff';
            page.style.margin = '0';
            page.style.padding = '0';
            page.style.background = 'transparent';
            page.style.width = '100%';
            newWrapper.style.margin = '0';
            newWrapper.style.width = '100%';
        }
        """
    ).strip()
    await page.evaluate(layout_script, context_tokens)
    await asyncio.sleep(0.2)
    clip_rect = await page.evaluate(
        """
        () => {
            const el = document.querySelector('.render-grid-wrapper');
            if (!el) {
                return null;
            }
            return {
                width: el.scrollWidth,
                height: el.scrollHeight,
            };
        }
        """
    )
    screenshot_options = {"path": str(png_path)}
    if clip_rect:
        clip_width = max(1, math.ceil(float(clip_rect["width"])))
        clip_height = max(1, math.ceil(float(clip_rect["height"])))
        await page.setViewport({"width": clip_width, "height": clip_height})
        await asyncio.sleep(0.1)
        clip_height = max(
            clip_height,
            math.ceil(
                float(
                    await page.evaluate(
                        """
                        () => {
                            const el = document.querySelector('.render-grid-wrapper');
                            return el ? el.scrollHeight : 0;
                        }
                        """
                    )
                )
            ),
        )
        await page.setViewport({"width": clip_width, "height": clip_height})
        screenshot_options["clip"] = {
            "x": 0,
            "y": 0,
            "width": clip_width,
            "height": clip_height,
            "scale": 1,
        }
    await page.screenshot(screenshot_options)
    await page.close()


async def _render_layers(
    root: Path,
    output_root: Path,
    samples_per_layer: int,
    examples_per_threshold: int,
    context_tokens: int,
    seed: int,
) -> None:
    browser = await launch(args=["--no-sandbox"])
    try:
        for layer_dir in _iter_layer_dirs(root):
            html_files = _select_html_files(layer_dir, samples_per_layer, seed)
            if not html_files:
                continue
            for html_path in html_files:
                png_path = (
                    output_root / layer_dir.name / f"{html_path.stem}.png"
                )
                print(f"Rendering {html_path.relative_to(root)} -> {png_path}")
                await _capture_png(
                    browser,
                    html_path,
                    png_path,
                    examples_per_threshold,
                    context_tokens,
                )
    finally:
        await browser.close()


def main() -> None:
    args = parse_args()
    root = args.dashboard_root
    if not root.is_dir():
        raise SystemExit(f"{root} is not a directory")
    output_root = args.output_dir or (root / "png_samples")
    asyncio.run(
        _render_layers(
            root,
            output_root,
            args.samples_per_layer,
            args.examples_per_threshold,
            args.context_tokens,
            args.seed,
        )
    )
    print(f"Saved PNG snapshots under {output_root}")


if __name__ == "__main__":
    main()
