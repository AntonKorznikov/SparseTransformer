from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model_transformer import SparseTransformerLM, TransformerConfig
from utils.common import get_batch, load_bin


class StudentMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, mlp_topk: int | None, use_relu_before_topk: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.use_relu = use_relu_before_topk
        self.mlp_topk = None if mlp_topk is None else int(mlp_topk)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=True)

    def batchtopk(self, x: torch.Tensor) -> torch.Tensor:
        if self.use_relu:
            x = torch.relu(x)
        if self.mlp_topk is None:
            return x
        vals, idx = torch.topk(x, k=self.mlp_topk, dim=-1, sorted=False)
        out = torch.zeros_like(x)
        batch_idx = torch.arange(x.shape[0], device=x.device, dtype=torch.long)[:, None]
        out[batch_idx, idx] = vals
        return out

    def hidden_activations(self, u: torch.Tensor) -> torch.Tensor:
        return self.batchtopk(self.fc1(u))


@torch.no_grad()
def compute_conditional_matrix(
    student: StudentMLP,
    teacher: SparseTransformerLM,
    layer_idx: int,
    student_indices: np.ndarray,
    data,
    num_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      cond:  [teacher_hidden, num_student] = E[s_i | teacher_j active]
      denom: [teacher_hidden]             = count(teacher_j active)
    """
    blk = teacher.blocks[layer_idx]
    teacher_hidden = int(teacher.cfg.n_embd * teacher.cfg.mlp_mult)
    n_students = int(len(student_indices))

    numer = torch.zeros((teacher_hidden, n_students), dtype=torch.float64, device="cpu")
    denom = torch.zeros((teacher_hidden,), dtype=torch.float64, device="cpu")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    for _ in range(num_batches):
        caches: dict[str, torch.Tensor] = {}

        def ln2_hook(_mod, _inp, out):
            caches["u"] = out.detach()

        def t_act_hook(_mod, _inp, out):
            caches["t_act"] = out.detach()

        h1 = blk.ln2.register_forward_hook(ln2_hook)
        h2 = blk.mlp.act.register_forward_hook(t_act_hook)
        xb, _ = get_batch(data, batch_size, block_size, device, rng)
        _logits, _loss = teacher(xb, None)
        h1.remove()
        h2.remove()

        u = caches["u"].reshape(-1, caches["u"].size(-1))
        t_act = caches["t_act"].reshape(-1, caches["t_act"].size(-1))

        s_act = student.hidden_activations(u)[:, student_indices]
        t_bin = (t_act != 0).to(torch.float64).cpu()          # [N, Ht]
        s_cpu = s_act.to(torch.float64).cpu()                 # [N, n_students]

        numer.add_(t_bin.t().mm(s_cpu))
        denom.add_(t_bin.sum(dim=0))

    cond = numer / denom.clamp_min(1.0).unsqueeze(1)
    return cond, denom


def build_output(
    cond: torch.Tensor,
    denom: torch.Tensor,
    student_indices: np.ndarray,
    topk_teacher: int,
) -> dict:
    teacher_hidden, n_students = cond.shape
    thresholds = [i / 100.0 for i in range(1, 11)] + [i / 100.0 for i in range(20, 101, 10)]

    per_student = []
    for col in range(n_students):
        s_idx = int(student_indices[col])
        v = cond[:, col].clone()  # [teacher_hidden]
        total = float(v.sum().item())

        if total > 0.0:
            probs = v / total
        else:
            probs = torch.zeros_like(v)

        probs_sorted, idx_sorted = torch.sort(probs, descending=True)
        cdf = torch.cumsum(probs_sorted, dim=0)

        coverage = {}
        for thr in thresholds:
            key = f"{int(thr * 100)}%"
            if total <= 0.0:
                coverage[key] = None
                continue
            k = int(torch.searchsorted(cdf, torch.tensor(thr, dtype=cdf.dtype), right=False).item()) + 1
            k = min(k, teacher_hidden)
            coverage[key] = k

        topn = min(int(topk_teacher), teacher_hidden)
        top_items = []
        for r in range(topn):
            j = int(idx_sorted[r].item())
            top_items.append(
                {
                    "teacher_neuron_idx": j,
                    "probability": float(probs_sorted[r].item()),
                    "E_s_i_given_teacher_active": float(v[j].item()),
                    "teacher_active_count": float(denom[j].item()),
                }
            )

        per_student.append(
            {
                "student_neuron_idx": s_idx,
                "sum_E_over_teachers": total,
                "coverage_k_for_probability_mass": coverage,
                "top_teacher_neurons": top_items,
            }
        )

    return {
        "teacher_hidden": int(teacher_hidden),
        "num_students_analyzed": int(n_students),
        "student_indices": [int(x) for x in student_indices.tolist()],
        "per_student": per_student,
    }


def tonal_palette_20() -> list[str]:
    # Earthy palette centered around #5f5a4a, #8a3434, #dca35b
    return [
        "#5f5a4a", "#6f6754", "#7f745e", "#8f8068", "#9f8d72",
        "#8a3434", "#984444", "#a65454", "#b46464", "#c27474",
        "#dca35b", "#d09b57", "#c49353", "#b88b4f", "#ac834b",
        "#73513f", "#835949", "#936153", "#a3695d", "#b37167",
    ]


def plot_fk_curves(cond: torch.Tensor, student_indices: np.ndarray, out_png: str, max_neurons_to_plot: int = 10) -> None:
    _teacher_hidden, n_students = cond.shape
    thresholds = torch.tensor(
        [i / 100.0 for i in range(1, 11)] + [i / 100.0 for i in range(20, 101, 10)],
        dtype=torch.float64,
    )
    y = thresholds.cpu().numpy().astype(np.float32)
    colors = tonal_palette_20()
    n_plot = min(int(max_neurons_to_plot), n_students)
    k_eval = 10

    # Plot 10 median neurons by f(k=10) = mass in top-10 teacher neurons.
    scores = []
    for col in range(n_students):
        v = cond[:, col]
        total = float(v.sum().item())
        if total <= 0.0:
            scores.append(0.0)
            continue
        probs_sorted = torch.sort(v / total, descending=True).values
        kk = min(k_eval, int(probs_sorted.numel()))
        scores.append(float(probs_sorted[:kk].sum().item()))
    order_desc = np.argsort(np.asarray(scores, dtype=np.float64))[::-1]
    if n_plot >= len(order_desc):
        selected_cols = order_desc.tolist()
    else:
        start = max(0, (len(order_desc) - n_plot) // 2)
        selected_cols = order_desc[start:start + n_plot].tolist()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for rank, col in enumerate(selected_cols):
        s_idx = int(student_indices[col])
        v = cond[:, col]
        total = float(v.sum().item())
        if total <= 0.0:
            x = np.ones((10,), dtype=np.float32)
        else:
            probs_sorted = torch.sort(v / total, descending=True).values
            cdf = torch.cumsum(probs_sorted, dim=0)
            x = []
            for thr in thresholds:
                k = int(torch.searchsorted(cdf, thr, right=False).item()) + 1
                x.append(k)
            x = np.asarray(x, dtype=np.float32)
        ax.plot(
            x,
            y,
            lw=2.5,
            alpha=0.95,
            color=colors[rank % len(colors)],
            marker="o",
            markersize=8.0,
            markeredgewidth=0.0,
            label=f"student_neuron = {s_idx}",
        )

    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("k (top teacher neurons kept)")
    ax.set_ylabel("f(k) = cumulative mass fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=12, ncol=1, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", type=str, default="out/main_mlp_mult_32_K_8/ckpt_iter300000.pt")
    p.add_argument("--student_ckpt", type=str, default="out/distill_suite/target_models/results_real_h10240_knone/h1024_knone_relu1_final.pt")
    p.add_argument("--data_dir", type=str, default="data/tinystories")
    p.add_argument("--split", type=str, choices=["train", "val"], default="val")
    p.add_argument("--layer_idx", type=int, default=2)
    p.add_argument("--num_batches", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--num_student_neurons", type=int, default=20)
    p.add_argument("--topk_teacher", type=int, default=20)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default="figures/teacher_to_student_first20.json")
    p.add_argument("--out_png", type=str, default="figures/teacher_to_student_first20_fk.png")
    args = p.parse_args()

    device = args.device

    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    t_cfg = TransformerConfig(**t_ckpt["model_cfg"])
    teacher = SparseTransformerLM(t_cfg).to(device)
    teacher.load_state_dict(t_ckpt["model"], strict=False)
    teacher.eval()

    s_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    s_state = s_ckpt["state_dict"]
    s_hidden = int(s_ckpt["hidden_size"])
    s_topk = s_ckpt.get("mlp_topk", None)
    s_relu = bool(s_ckpt.get("use_relu_before_topk", True))
    student = StudentMLP(
        d_in=t_cfg.n_embd,
        d_hidden=s_hidden,
        d_out=t_cfg.n_embd,
        mlp_topk=s_topk,
        use_relu_before_topk=s_relu,
    ).to(device)
    student.load_state_dict(s_state, strict=True)
    student.eval()

    n_students = min(int(args.num_student_neurons), s_hidden)
    rs = np.random.default_rng(int(args.seed))
    student_indices = np.sort(rs.choice(s_hidden, size=n_students, replace=False).astype(np.int64))

    split_path = "train.bin" if args.split == "train" else "val.bin"
    data = load_bin(os.path.join(args.data_dir, split_path))

    cond, denom = compute_conditional_matrix(
        student=student,
        teacher=teacher,
        layer_idx=args.layer_idx,
        student_indices=student_indices,
        data=data,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
        seed=args.seed,
    )

    out = build_output(
        cond=cond,
        denom=denom,
        student_indices=student_indices,
        topk_teacher=args.topk_teacher,
    )
    out["meta"] = {
        "teacher_ckpt": args.teacher_ckpt,
        "student_ckpt": args.student_ckpt,
        "layer_idx": int(args.layer_idx),
        "split": args.split,
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "block_size": int(args.block_size),
        "seed": int(args.seed),
    }

    out_json_dir = os.path.dirname(args.out_json) or "."
    out_png_dir = os.path.dirname(args.out_png) or "."
    os.makedirs(out_json_dir, exist_ok=True)
    os.makedirs(out_png_dir, exist_ok=True)

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    plot_fk_curves(cond=cond, student_indices=student_indices, out_png=args.out_png)

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved figure: {args.out_png}")
    print(f"Analyzed student neurons: {n_students}")
    print(f"Teacher hidden size: {out['teacher_hidden']}")


if __name__ == "__main__":
    main()
