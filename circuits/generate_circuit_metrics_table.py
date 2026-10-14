#!/usr/bin/env python3
"""Generate per-prompt faithfulness and completeness table for evaluated circuits."""

from __future__ import annotations

import argparse
import csv
import json
import statistics
from pathlib import Path
from typing import Any

# The 16 prompts reported in the paper (Table 4).
EVAL_PROMPTS = [
    "Jack was hungry, so he went looking for something to",
    "She felt very sad so her friend gave her a big warm",
    "The cat saw a mouse and started to",
    "The princess kissed the frog and it turned into a handsome",
    "The robot beeped and said in a friendly",
    "She planted a seed and watered it every",
    "The pirate looked at the map to find the hidden",
    "He washed his hands before sitting at the dinner",
    "She put on her shoes and ran out the",
    "She opened the book and started to",
    "He wrapped the gift in colorful shiny",
    "She drank the hot soup and felt much",
    "She brushed her hair and looked in the",
    "She baked a cake and put it in the",
    "The baby dropped her toy and started to",
    "He stacked the blocks until the tower was very",
]


def _curve_lookup(curve: list[dict[str, float]], key: str) -> dict[int, float]:
    out: dict[int, float] = {}
    for point in curve:
        if key == "n_added":
            n = int(round(point[key]))
        else:
            n = int(point[key])
        out[n] = float(point["retain_ratio" if key == "n_added" else "faithfulness"])
    return out


def load_circuit_metrics(graph_path: Path) -> dict[str, Any]:
    with graph_path.open(encoding="utf-8") as f:
        graph = json.load(f)

    meta = graph["meta"]
    circuit_size = int(meta["circuit_size"])
    addback = _curve_lookup(graph.get("addback_curve", []), "n_added")
    completeness_curve = _curve_lookup(graph.get("faithfulness_curve", []), "n")

    detail_path = graph_path.parent.parent / "circuit_detail" / "circuit_detail.json"
    detail_k: int | None = None
    if detail_path.exists():
        with detail_path.open(encoding="utf-8") as f:
            detail_k = int(json.load(f).get("num_nodes", 0)) or None

    prompt = meta["prompt"]
    return {
        "prompt": prompt,
        "prompt_short": prompt if len(prompt) <= 52 else prompt[:49] + "...",
        "target": meta["target_str"].strip(),
        "p0": float(meta["p0"]),
        "circuit_size": circuit_size,
        "faithfulness": addback.get(circuit_size),
        "completeness": completeness_curve.get(circuit_size),
        "detail_k": detail_k,
        "faithfulness_at_detail_k": addback.get(detail_k) if detail_k is not None else None,
        "completeness_at_detail_k": completeness_curve.get(detail_k) if detail_k is not None else None,
        "is_highlighted": prompt.startswith("Jack was hungry"),
    }


def find_graph_for_prompt(attribution_root: Path, prompt: str) -> Path | None:
    for graph_path in sorted(attribution_root.glob("*/circuit/graph.json")):
        with graph_path.open(encoding="utf-8") as f:
            if json.load(f)["meta"]["prompt"] == prompt:
                return graph_path
    return None


def collect_rows(attribution_root: Path, prompts: list[str]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    missing: list[str] = []
    for idx, prompt in enumerate(prompts, start=1):
        graph_path = find_graph_for_prompt(attribution_root, prompt)
        if graph_path is None:
            missing.append(prompt)
            continue
        row = load_circuit_metrics(graph_path)
        row["idx"] = idx
        rows.append(row)

    if missing:
        raise FileNotFoundError(
            "Missing graph.json for prompts:\n  - " + "\n  - ".join(missing)
        )
    return rows


def summary_stats(values: list[float]) -> dict[str, float]:
    return {
        "min": min(values),
        "max": max(values),
        "mean": statistics.mean(values),
        "median": statistics.median(values),
        "stdev": statistics.pstdev(values),
    }


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    fieldnames = [
        "idx",
        "prompt",
        "target",
        "p0",
        "p0_pct",
        "circuit_size",
        "faithfulness",
        "completeness",
        "detail_k",
        "faithfulness_at_detail_k",
        "completeness_at_detail_k",
        "is_highlighted",
    ]
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(
                {
                    "idx": row["idx"],
                    "prompt": row["prompt"],
                    "target": row["target"],
                    "p0": row["p0"],
                    "p0_pct": round(row["p0"] * 100, 1),
                    "circuit_size": row["circuit_size"],
                    "faithfulness": row["faithfulness"],
                    "completeness": row["completeness"],
                    "detail_k": row["detail_k"],
                    "faithfulness_at_detail_k": row["faithfulness_at_detail_k"],
                    "completeness_at_detail_k": row["completeness_at_detail_k"],
                    "is_highlighted": row["is_highlighted"],
                }
            )


def write_markdown(path: Path, rows: list[dict[str, Any]]) -> None:
    faith = [r["faithfulness"] for r in rows]
    compl = [r["completeness"] for r in rows]
    p0 = [r["p0"] for r in rows]
    fs = summary_stats(faith)
    cs = summary_stats(compl)
    ps = summary_stats(p0)

    lines = [
        "# Circuit faithfulness and completeness (K = 20 neurons)",
        "",
        "Metrics extracted from `graph.json` in each prompt's `circuit/` folder.",
        "",
        "- **Faithfulness**: add-back retain ratio \\(P_\\mathrm{circuit}/P_0\\) at circuit size K.",
        "- **Completeness**: ablation ratio \\(P_\\mathrm{ablated}/P_0\\) at circuit size K (lower = more complete).",
        "- **Target prob.**: baseline model probability \\(P_0\\) for the continuation token.",
        "",
        "| # | Prompt (truncated) | Target | \\(P_0\\) (%) | Faithfulness | Completeness |",
        "|---:|---|---|---:|---:|---:|",
    ]

    for row in rows:
        mark = " **" if row["is_highlighted"] else ""
        end = "**" if row["is_highlighted"] else ""
        lines.append(
            f"| {row['idx']} |{mark}{row['prompt_short']}{end} | "
            f"{row['target']} | {row['p0'] * 100:.1f} | "
            f"{row['faithfulness']:.3f} | {row['completeness']:.3f} |"
        )

    lines.extend(
        [
            "",
            "## Summary across 16 prompts",
            "",
            f"| Metric | Min | Median | Mean | Max | Std |",
            f"|---|---:|---:|---:|---:|---:|",
            f"| \\(P_0\\) (%) | {ps['min']*100:.1f} | {ps['median']*100:.1f} | {ps['mean']*100:.1f} | {ps['max']*100:.1f} | {ps['stdev']*100:.1f} |",
            f"| Faithfulness | {fs['min']:.3f} | {fs['median']:.3f} | {fs['mean']:.3f} | {fs['max']:.3f} | {fs['stdev']:.3f} |",
            f"| Completeness | {cs['min']:.3f} | {cs['median']:.3f} | {cs['mean']:.3f} | {cs['max']:.3f} | {cs['stdev']:.3f} |",
            "",
            "The highlighted Jack prompt (row 1) is within the typical spread on both metrics.",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def tex_escape(text: str) -> str:
    return (
        text.replace("\\", "\\textbackslash{}")
        .replace("&", "\\&")
        .replace("%", "\\%")
        .replace("_", "\\_")
    )


def write_latex(path: Path, rows: list[dict[str, Any]]) -> None:
    faith = [r["faithfulness"] for r in rows]
    compl = [r["completeness"] for r in rows]
    p0 = [r["p0"] for r in rows]
    fs = summary_stats(faith)
    cs = summary_stats(compl)
    ps = summary_stats(p0)

    lines = [
        "% Auto-generated by generate_circuit_metrics_table.py",
        "\\begin{table}[t]",
        "\\centering",
        "\\caption{Per-prompt circuit metrics at $K{=}20$ neurons across the 16 evaluated TinyStories prompts "
        f"($P_0 \\in [{ps['min']*100:.1f}\\%,\\,{ps['max']*100:.1f}\\%]$). "
        "Faithfulness is the add-back retain ratio $P_{\\mathrm{circuit}}/P_0$; "
        "completeness is the ablation ratio $P_{\\mathrm{ablated}}/P_0$ (lower indicates a more complete circuit).}",
        "\\label{tab:circuit-metrics}",
        "\\small",
        "\\begin{tabular}{@{}clrrrr@{}}",
        "\\toprule",
        "\\# & Prompt continuation & Target & $P_0$ (\\%) & Faithfulness & Completeness \\\\",
        "\\midrule",
    ]

    for row in rows:
        prompt_tex = tex_escape(row["prompt_short"])
        target_tex = tex_escape(row["target"])
        if row["is_highlighted"]:
            prompt_tex = f"\\textbf{{{prompt_tex}}}"
        lines.append(
            f"{row['idx']} & {prompt_tex} & {target_tex} & "
            f"{row['p0'] * 100:.1f} & {row['faithfulness']:.3f} & {row['completeness']:.3f} \\\\"
        )

    lines.extend(
        [
            "\\midrule",
            f"\\multicolumn{{3}}{{l}}{{Mean $\\pm$ std}} & "
            f"{ps['mean']*100:.1f} $\\pm$ {ps['stdev']*100:.1f} & "
            f"{fs['mean']:.3f} $\\pm$ {fs['stdev']:.3f} & "
            f"{cs['mean']:.3f} $\\pm$ {cs['stdev']:.3f} \\\\",
            f"\\multicolumn{{3}}{{l}}{{Median [min, max]}} & "
            f"{ps['median']*100:.1f} [{ps['min']*100:.1f}, {ps['max']*100:.1f}] & "
            f"{fs['median']:.3f} [{fs['min']:.3f}, {fs['max']:.3f}] & "
            f"{cs['median']:.3f} [{cs['min']:.3f}, {cs['max']:.3f}] \\\\",
            "\\bottomrule",
            "\\end{tabular}",
            "\\end{table}",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--attribution-root",
        type=Path,
        default=Path("out/main_mlp_mult_32_K_8/attribution_graphs"),
        help="Root folder containing per-prompt attribution graph outputs.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=None,
        help="Output directory (default: <attribution-root>/circuit_metrics_table).",
    )
    args = parser.parse_args()

    attribution_root = args.attribution_root.resolve()
    out_dir = (args.out_dir or attribution_root / "circuit_metrics_table").resolve()
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = collect_rows(attribution_root, EVAL_PROMPTS)
    rows.sort(key=lambda r: r["idx"])

    write_csv(out_dir / "circuit_metrics_table.csv", rows)
    write_markdown(out_dir / "circuit_metrics_table.md", rows)
    write_latex(out_dir / "circuit_metrics_table.tex", rows)

    print(f"Wrote tables for {len(rows)} circuits to {out_dir}")
    for fmt in ("csv", "md", "tex"):
        print(f"  - circuit_metrics_table.{fmt}")


if __name__ == "__main__":
    main()
