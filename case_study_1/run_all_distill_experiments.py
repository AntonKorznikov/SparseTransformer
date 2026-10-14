from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path


# Hardcoded experiment target from AntonKorznikov/SparseTransformer.
CKPT_REF = "main_mlp_mult_32_K_8/ckpt_iter300000.pt"
DATA_DIR = "data/tinystories"
LAYER_IDX = 2  # If you mean "2nd layer" in 1-based indexing, set to 1.


def run(cmd: list[str]) -> None:
    print(">>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--save_target_final_ckpts",
        action="store_true",
        help="run and save final checkpoints for 3 target models",
    )
    parser.add_argument(
        "--only_target_models",
        action="store_true",
        help="if set with --save_target_final_ckpts, run only those 3 target models",
    )
    args = parser.parse_args()

    py = sys.executable
    base_common = [
        py,
        "-m",
        "case_study_1.distill_mlp",
        "--ckpt",
        CKPT_REF,
        "--data_dir",
        DATA_DIR,
        "--layer_idx",
        str(LAYER_IDX),
        "--train_topk_sae",
    ]

    if args.only_target_models and not args.save_target_final_ckpts:
        raise SystemExit("--only_target_models requires --save_target_final_ckpts")

    if args.save_target_final_ckpts:
        # 1) + 2): matched real student MLP + matched real student SAE
        run(
            [
                *base_common,
                "--teacher_source",
                "real",
                "--match_teacher_hparams",
                "--save_final_checkpoints",
                "--teacher_dir",
                "out/distill_suite/target_models/teacher_real",
                "--out_dir",
                "out/distill_suite/target_models/results_match_real",
            ]
        )

        # 3): real student MLP with K=None and hidden_size=1024
        run(
            [
                py,
                "-m",
                "case_study_1.distill_mlp",
                "--ckpt",
                CKPT_REF,
                "--data_dir",
                DATA_DIR,
                "--layer_idx",
                str(LAYER_IDX),
                "--teacher_source",
                "real",
                "--hidden_sizes_override",
                "1024",
                "--mlp_topk_override",
                "none",
                "--save_final_checkpoints",
                "--teacher_dir",
                "out/distill_suite/target_models/teacher_real",
                "--out_dir",
                "out/distill_suite/target_models/results_real_h10240_knone",
            ]
        )

        if args.only_target_models:
            print("\nTarget-model-only run completed.")
            return

    # Regime A: teacher-matched hyperparameters (logs parameter-recovery metrics).
    base_match = base_common + ["--match_teacher_hparams"]

    # Real teacher experiment (matched).
    run(
        base_match
        + [
            "--teacher_source",
            "real",
            "--teacher_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/teacher_real",
            "--out_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/results_real_match",
        ]
    )

    # Random teacher experiment (matched).
    run(
        base_match
        + [
            "--teacher_source",
            "random",
            "--random_teacher_seed",
            "1234",
            "--teacher_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/teacher_random",
            "--out_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/results_random_match",
        ]
    )

    # Regime B: non-matched grid hyperparameters (K/dict size can differ; no param-recovery metrics).
    base_grid = list(base_common)

    # Real teacher experiment (grid).
    run(
        base_grid
        + [
            "--teacher_source",
            "real",
            "--teacher_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/teacher_real",
            "--out_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/results_real_grid",
        ]
    )

    # Random teacher experiment (grid).
    run(
        base_grid
        + [
            "--teacher_source",
            "random",
            "--random_teacher_seed",
            "1234",
            "--teacher_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/teacher_random",
            "--out_dir",
            "out/distill_suite/main_mlp_mult_32_K_8/layer2/results_random_grid",
        ]
    )

    print("\nAll experiments completed.")
    print("TensorBoard logs/checkpoints are under out/distill_suite/main_mlp_mult_32_K_8/layer2/")


if __name__ == "__main__":
    # Ensure script runs from repository root if called elsewhere.
    repo_root = Path(__file__).resolve().parent.parent
    if Path.cwd() != repo_root:
        print(f"Switching cwd to {repo_root}")
        os.chdir(repo_root)
    main()
