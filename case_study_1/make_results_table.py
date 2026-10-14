"""Summarise Case Study 1 runs (Transcoder vs SAE) from their TensorBoard logs.

For every run found under --root, prints the validation NMSE averaged over the last
--tail evaluations (with its std over those evaluations) and the final MMCS values.

    python -m case_study_1.make_results_table --root out/distill_suite
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from tensorboard.backend.event_processing import event_accumulator as ea

SETTINGS = [
    ("TopK (K=8)", "target_models/results_match_real"),
]

MMCS_TAGS = {
    "Transcoder": ("params/key_max_cos_mean", "params/value_max_cos_mean"),
    "SAE": ("params/encoder_vs_gt_value_max_cos_mean", "params/decoder_vs_gt_value_max_cos_mean"),
}


def load(run_dir: Path) -> dict[str, np.ndarray]:
    acc = ea.EventAccumulator(str(run_dir), size_guidance={ea.SCALARS: 0})
    acc.Reload()
    return {t: np.array([e.value for e in acc.Scalars(t)]) for t in acc.Tags().get("scalars", [])}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", type=Path, default=Path("out/distill_suite"))
    ap.add_argument("--tail", type=int, default=33, help="number of final validation evaluations to average NMSE over")
    args = ap.parse_args()

    header = f"{'setting':<18} {'method':<14} {'NMSE':>18} {'MMCS enc/key':>13} {'MMCS dec/value':>15}"
    print(header)
    print("-" * len(header))
    for setting, sub in SETTINGS:
        base = args.root / sub
        if not base.is_dir():
            print(f"{setting:<18} (missing {base})")
            continue
        runs = sorted(p for p in base.iterdir() if p.is_dir() and p.name.startswith(("h", "sae_h")))
        for run in runs:
            if run.name.startswith("sae_"):
                method = "SAE (pb" + run.name.split("_pb")[1][0] + ")"
                key_tag, val_tag = MMCS_TAGS["SAE"]
            else:
                method = "Transcoder"
                key_tag, val_tag = MMCS_TAGS["Transcoder"]
            s = load(run)
            if "nmse/val" not in s:
                print(f"{setting:<18} {method:<14} (no validation logs in {run})")
                continue
            tail = s["nmse/val"][-args.tail:]
            nmse = f"{tail.mean():.4f} ± {tail.std():.4f}"
            key = f"{s[key_tag][-1]:.3f}" if key_tag in s else "--"
            val = f"{s[val_tag][-1]:.3f}" if val_tag in s else "--"
            print(f"{setting:<18} {method:<14} {nmse:>18} {key:>13} {val:>15}")
    print("\npb0 = SAE without biases (reported in the paper), pb1 = SAE with a shared pre-bias.")


if __name__ == "__main__":
    main()
