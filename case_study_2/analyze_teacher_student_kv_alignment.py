from __future__ import annotations

import argparse
import json
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_transformer import SparseTransformerLM, TransformerConfig
from utils.common import get_batch, load_bin


class StudentMLP(nn.Module):
    def __init__(self, d_in: int, d_hidden: int, d_out: int, mlp_topk: int | None, use_relu_before_topk: bool = True):
        super().__init__()
        self.fc1 = nn.Linear(d_in, d_hidden, bias=True)
        self.use_relu = bool(use_relu_before_topk)
        self.mlp_topk = None if mlp_topk is None else int(mlp_topk)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=True)

    def hidden_activations(self, u: torch.Tensor) -> torch.Tensor:
        x = self.fc1(u)
        if self.use_relu:
            x = torch.relu(x)
        if self.mlp_topk is None:
            return x
        vals, idx = torch.topk(x, k=self.mlp_topk, dim=-1, sorted=False)
        out = torch.zeros_like(x)
        batch_idx = torch.arange(x.shape[0], device=x.device, dtype=torch.long)[:, None]
        out[batch_idx, idx] = vals
        return out


@torch.no_grad()
def collect_conditional_s_given_t(
    student: StudentMLP,
    teacher: SparseTransformerLM,
    layer_idx: int,
    data,
    num_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    seed: int,
) -> tuple[torch.Tensor, torch.Tensor]:
    """
    Returns:
      cond_s_given_t: [teacher_hidden, student_hidden] = E[s_i | teacher_j active]
      teacher_active_counts: [teacher_hidden]
    """
    blk = teacher.blocks[layer_idx]
    teacher_hidden = int(teacher.cfg.n_embd * teacher.cfg.mlp_mult)
    student_hidden = int(student.fc1.out_features)

    numer = torch.zeros((teacher_hidden, student_hidden), dtype=torch.float64, device="cpu")
    denom = torch.zeros((teacher_hidden,), dtype=torch.float64, device="cpu")

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))

    for _ in range(num_batches):
        cache: dict[str, torch.Tensor] = {}

        def ln2_hook(_mod, _inp, out):
            cache["u"] = out.detach()

        def t_act_hook(_mod, _inp, out):
            cache["t_act"] = out.detach()

        h1 = blk.ln2.register_forward_hook(ln2_hook)
        h2 = blk.mlp.act.register_forward_hook(t_act_hook)
        xb, _ = get_batch(data, batch_size, block_size, device, rng)
        _logits, _loss = teacher(xb, None)
        h1.remove()
        h2.remove()

        u = cache["u"].reshape(-1, cache["u"].size(-1))
        t_act = cache["t_act"].reshape(-1, cache["t_act"].size(-1))
        s_act = student.hidden_activations(u)

        t_bin = (t_act != 0).to(torch.float64).cpu()   # [N, T]
        s_cpu = s_act.to(torch.float64).cpu()          # [N, S]
        numer.add_(t_bin.t().mm(s_cpu))
        denom.add_(t_bin.sum(dim=0))

    cond_s_given_t = numer / denom.clamp_min(1.0).unsqueeze(1)
    return cond_s_given_t, denom


def cosine_sim_matrix(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    a_n = F.normalize(a, p=2, dim=-1, eps=1e-12)
    b_n = F.normalize(b, p=2, dim=-1, eps=1e-12)
    return a_n @ b_n.t()


def student_side_alignment_json(
    cond_s_given_t: torch.Tensor,
    teacher_keys: torch.Tensor,
    teacher_values: torch.Tensor,
    student_keys: torch.Tensor,
    student_values: torch.Tensor,
) -> dict:
    # p(teacher_j | student_i) from E[s_i | teacher_j active], normalized over all teacher_j.
    probs_t_given_s = cond_s_given_t / cond_s_given_t.sum(dim=0, keepdim=True).clamp_min(1e-12)

    approx_keys = probs_t_given_s.t().to(torch.float32) @ teacher_keys.to(torch.float32)       # [S, D]
    approx_values = probs_t_given_s.t().to(torch.float32) @ teacher_values.to(torch.float32)    # [S, D]

    key_cos = F.cosine_similarity(approx_keys, student_keys.to(torch.float32), dim=1, eps=1e-12)
    value_cos = F.cosine_similarity(approx_values, student_values.to(torch.float32), dim=1, eps=1e-12)

    per_student = []
    student_hidden = int(student_keys.size(0))
    for i in range(student_hidden):
        p = probs_t_given_s[:, i]
        topv, topidx = torch.topk(p, k=min(20, int(p.numel())), dim=0)
        per_student.append(
            {
                "student_neuron_idx": int(i),
                "key_cosine_with_teacher_mixture": float(key_cos[i].item()),
                "value_cosine_with_teacher_mixture": float(value_cos[i].item()),
                "teacher_prob_sum": float(p.sum().item()),
                "top_teacher_neurons": [
                    {"teacher_neuron_idx": int(topidx[k].item()), "probability": float(topv[k].item())}
                    for k in range(int(topv.numel()))
                ],
            }
        )

    return {
        "student_hidden": student_hidden,
        "teacher_hidden": int(teacher_keys.size(0)),
        "mean_key_cosine": float(key_cos.mean().item()),
        "mean_value_cosine": float(value_cos.mean().item()),
        "median_key_cosine": float(torch.median(key_cos).item()),
        "median_value_cosine": float(torch.median(value_cos).item()),
        "per_student": per_student,
    }


def plot_random_teacher_scatter_grid_single_metric(
    cond_s_given_t: torch.Tensor,
    cos_student_teacher: torch.Tensor,
    teacher_indices: np.ndarray,
    out_png: str,
    metric_name: str,
) -> None:
    probs_s_given_t = cond_s_given_t[teacher_indices] / cond_s_given_t[teacher_indices].sum(dim=1, keepdim=True).clamp_min(1e-12)

    fig, axes = plt.subplots(4, 5, figsize=(18, 14), sharex=True, sharey=True)
    axes = axes.flatten()

    for m, j in enumerate(teacher_indices.tolist()):
        ax = axes[m]
        x = probs_s_given_t[m].cpu().numpy()
        y = cos_student_teacher[:, m].cpu().numpy()
        x_safe = np.clip(x, 1e-12, None)

        ax.scatter(x_safe, y, s=14, alpha=0.6, color="#8a3434")
        ax.set_xscale("log")
        ax.grid(True, alpha=0.25)
        ax.set_title(f"teacher_neuron = {int(j)}", fontsize=9)

    for m in range(len(teacher_indices), len(axes)):
        axes[m].axis("off")

    fig.text(0.5, 0.04, "P(student_neuron | teacher_neuron active)", ha="center")
    fig.text(0.04, 0.5, f"cosine similarity ({metric_name})", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1, 1])
    plt.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_hist(values: torch.Tensor, out_png: str, title: str, color: str) -> None:
    arr = values.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(arr, bins=50, density=True, alpha=0.65, color=color)
    ax.hist(arr, bins=50, histtype="step", color="black", linewidth=1.2, density=True)
    ax.axvline(float(arr.mean()), color=color, linestyle="--", linewidth=1.8, alpha=0.95)
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def plot_cosine_hist_with_baseline(
    values_real: torch.Tensor,
    out_png: str,
    title: str,
    color: str,
    mean_real_20: float,
    mean_shuffled_20: float,
) -> None:
    arr = values_real.detach().cpu().numpy()
    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    ax.hist(arr, bins=50, density=True, alpha=0.65, color=color)
    ax.hist(arr, bins=50, histtype="step", color="black", linewidth=1.2, density=True)
    ax.axvline(float(mean_real_20), color=color, linestyle="--", linewidth=2.0, alpha=0.95, label="mean real (20)")
    ax.axvline(float(mean_shuffled_20), color="gray", linestyle="--", linewidth=2.0, alpha=0.95, label="mean shuffled baseline (20)")
    ax.set_xlabel("cosine similarity")
    ax.set_ylabel("density")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best", fontsize=11, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def topk_renorm_probs(probs_t_given_s: torch.Tensor, k: int) -> torch.Tensor:
    teacher_hidden, student_hidden = probs_t_given_s.shape
    k_use = min(int(k), int(teacher_hidden))
    vals, idx = torch.topk(probs_t_given_s, k=k_use, dim=0, largest=True, sorted=False)
    out = torch.zeros_like(probs_t_given_s)
    out.scatter_(0, idx, vals)
    out = out / out.sum(dim=0, keepdim=True).clamp_min(1e-12)
    return out


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
    p.add_argument("--num_random_teacher", type=int, default=20)
    p.add_argument("--num_student_for_baseline", type=int, default=20)
    p.add_argument("--topk_teacher_for_approx", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_json", type=str, default="figures/student_teacher_kv_alignment.json")
    p.add_argument("--out_png_scatter_key", type=str, default="figures/random20_teacher_to_student_prob_vs_key_cos.png")
    p.add_argument("--out_png_scatter_value", type=str, default="figures/random20_teacher_to_student_prob_vs_value_cos.png")
    p.add_argument("--out_png_hist_key", type=str, default="figures/student_teacher_key_cos_hist.png")
    p.add_argument("--out_png_hist_value", type=str, default="figures/student_teacher_value_cos_hist.png")
    p.add_argument("--out_png_hist_key_topk", type=str, default="figures/student_teacher_key_cos_hist_top100.png")
    p.add_argument("--out_png_hist_value_topk", type=str, default="figures/student_teacher_value_cos_hist_top100.png")
    args = p.parse_args()

    os.makedirs(os.path.dirname(args.out_json) or ".", exist_ok=True)
    for pth in [
        args.out_png_scatter_key,
        args.out_png_scatter_value,
        args.out_png_hist_key,
        args.out_png_hist_value,
        args.out_png_hist_key_topk,
        args.out_png_hist_value_topk,
    ]:
        os.makedirs(os.path.dirname(pth) or ".", exist_ok=True)

    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    t_cfg = TransformerConfig(**t_ckpt["model_cfg"])
    teacher = SparseTransformerLM(t_cfg).to(args.device)
    teacher.load_state_dict(t_ckpt["model"], strict=False)
    teacher.eval()

    s_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    s_hidden = int(s_ckpt["hidden_size"])
    student = StudentMLP(
        d_in=t_cfg.n_embd,
        d_hidden=s_hidden,
        d_out=t_cfg.n_embd,
        mlp_topk=s_ckpt.get("mlp_topk", None),
        use_relu_before_topk=bool(s_ckpt.get("use_relu_before_topk", True)),
    ).to(args.device)
    student.load_state_dict(s_ckpt["state_dict"], strict=True)
    student.eval()

    split_path = "train.bin" if args.split == "train" else "val.bin"
    data = load_bin(os.path.join(args.data_dir, split_path))

    cond_s_given_t, teacher_active_counts = collect_conditional_s_given_t(
        student=student,
        teacher=teacher,
        layer_idx=args.layer_idx,
        data=data,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=args.device,
        seed=args.seed,
    )

    blk = teacher.blocks[args.layer_idx]
    teacher_keys = blk.mlp.fc1.weight.detach().cpu()                      # [T, D]
    teacher_values = blk.mlp.fc2.weight.detach().cpu().t().contiguous()   # [T, D]
    student_keys = student.fc1.weight.detach().cpu()                      # [S, D]
    student_values = student.fc2.weight.detach().cpu().t().contiguous()   # [S, D]

    out = student_side_alignment_json(
        cond_s_given_t=cond_s_given_t,
        teacher_keys=teacher_keys,
        teacher_values=teacher_values,
        student_keys=student_keys,
        student_values=student_values,
    )

    rs = np.random.default_rng(int(args.seed))
    teacher_hidden = int(teacher_keys.size(0))
    n_rand = min(int(args.num_random_teacher), teacher_hidden)
    random_teacher_idx = np.sort(rs.choice(teacher_hidden, size=n_rand, replace=False).astype(np.int64))
    out["random_teacher_indices_for_scatter"] = [int(x) for x in random_teacher_idx.tolist()]
    out["teacher_active_counts"] = {
        str(i): float(teacher_active_counts[i].item()) for i in random_teacher_idx.tolist()
    }
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

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    key_cos_all = cosine_sim_matrix(student_keys.to(torch.float32), teacher_keys[random_teacher_idx].to(torch.float32))      # [S, 20]
    value_cos_all = cosine_sim_matrix(student_values.to(torch.float32), teacher_values[random_teacher_idx].to(torch.float32)) # [S, 20]

    plot_random_teacher_scatter_grid_single_metric(
        cond_s_given_t=cond_s_given_t,
        cos_student_teacher=key_cos_all,
        teacher_indices=random_teacher_idx,
        out_png=args.out_png_scatter_key,
        metric_name="key",
    )
    plot_random_teacher_scatter_grid_single_metric(
        cond_s_given_t=cond_s_given_t,
        cos_student_teacher=value_cos_all,
        teacher_indices=random_teacher_idx,
        out_png=args.out_png_scatter_value,
        metric_name="value",
    )

    # Shuffled baseline: for each student neuron, shuffle teacher probabilities
    # P(teacher_j | student_i), then build teacher mixture and recompute cosine.
    probs_t_given_s = cond_s_given_t / cond_s_given_t.sum(dim=0, keepdim=True).clamp_min(1e-12)
    probs_t_given_s_shuf = torch.empty_like(probs_t_given_s)
    for i in range(probs_t_given_s.size(1)):
        perm = rs.permutation(probs_t_given_s.size(0))
        probs_t_given_s_shuf[:, i] = probs_t_given_s[perm, i]

    approx_key_shuf = probs_t_given_s_shuf.t().to(torch.float32) @ teacher_keys.to(torch.float32)
    approx_value_shuf = probs_t_given_s_shuf.t().to(torch.float32) @ teacher_values.to(torch.float32)
    key_cos_real = torch.tensor([x["key_cosine_with_teacher_mixture"] for x in out["per_student"]], dtype=torch.float32)
    value_cos_real = torch.tensor([x["value_cosine_with_teacher_mixture"] for x in out["per_student"]], dtype=torch.float32)
    key_cos_shuf = F.cosine_similarity(approx_key_shuf, student_keys.to(torch.float32), dim=1, eps=1e-12)
    value_cos_shuf = F.cosine_similarity(approx_value_shuf, student_values.to(torch.float32), dim=1, eps=1e-12)

    n_s20 = min(int(args.num_student_for_baseline), int(student_keys.size(0)))
    student_idx_20 = np.sort(rs.choice(int(student_keys.size(0)), size=n_s20, replace=False).astype(np.int64))
    idx20 = torch.from_numpy(student_idx_20)
    key_mean_real_20 = float(key_cos_real[idx20].mean().item())
    key_mean_shuf_20 = float(key_cos_shuf[idx20].mean().item())
    value_mean_real_20 = float(value_cos_real[idx20].mean().item())
    value_mean_shuf_20 = float(value_cos_shuf[idx20].mean().item())

    out["baseline_shuffled"] = {
        "num_student_for_baseline": int(n_s20),
        "student_indices": [int(x) for x in student_idx_20.tolist()],
        "mean_key_cos_real_20": key_mean_real_20,
        "mean_key_cos_shuffled_20": key_mean_shuf_20,
        "mean_value_cos_real_20": value_mean_real_20,
        "mean_value_cos_shuffled_20": value_mean_shuf_20,
        "mean_key_cos_shuffled_all": float(key_cos_shuf.mean().item()),
        "mean_value_cos_shuffled_all": float(value_cos_shuf.mean().item()),
    }

    # Top-K approximation (e.g. top-100 teacher neurons per student), with renormalization.
    probs_t_given_s_topk = topk_renorm_probs(probs_t_given_s, k=int(args.topk_teacher_for_approx))
    approx_key_topk = probs_t_given_s_topk.t().to(torch.float32) @ teacher_keys.to(torch.float32)
    approx_value_topk = probs_t_given_s_topk.t().to(torch.float32) @ teacher_values.to(torch.float32)
    key_cos_topk = F.cosine_similarity(approx_key_topk, student_keys.to(torch.float32), dim=1, eps=1e-12)
    value_cos_topk = F.cosine_similarity(approx_value_topk, student_values.to(torch.float32), dim=1, eps=1e-12)

    probs_t_given_s_topk_shuf = torch.empty_like(probs_t_given_s_topk)
    for i in range(probs_t_given_s_topk.size(1)):
        perm = rs.permutation(probs_t_given_s_topk.size(0))
        probs_t_given_s_topk_shuf[:, i] = probs_t_given_s_topk[perm, i]
    approx_key_topk_shuf = probs_t_given_s_topk_shuf.t().to(torch.float32) @ teacher_keys.to(torch.float32)
    approx_value_topk_shuf = probs_t_given_s_topk_shuf.t().to(torch.float32) @ teacher_values.to(torch.float32)
    key_cos_topk_shuf = F.cosine_similarity(approx_key_topk_shuf, student_keys.to(torch.float32), dim=1, eps=1e-12)
    value_cos_topk_shuf = F.cosine_similarity(approx_value_topk_shuf, student_values.to(torch.float32), dim=1, eps=1e-12)

    key_mean_topk_real_20 = float(key_cos_topk[idx20].mean().item())
    key_mean_topk_shuf_20 = float(key_cos_topk_shuf[idx20].mean().item())
    value_mean_topk_real_20 = float(value_cos_topk[idx20].mean().item())
    value_mean_topk_shuf_20 = float(value_cos_topk_shuf[idx20].mean().item())
    out["topk_teacher_approx"] = {
        "k": int(min(int(args.topk_teacher_for_approx), int(teacher_keys.size(0)))),
        "mean_key_cos_all": float(key_cos_topk.mean().item()),
        "mean_value_cos_all": float(value_cos_topk.mean().item()),
        "mean_key_cos_real_20": key_mean_topk_real_20,
        "mean_key_cos_shuffled_20": key_mean_topk_shuf_20,
        "mean_value_cos_real_20": value_mean_topk_real_20,
        "mean_value_cos_shuffled_20": value_mean_topk_shuf_20,
    }

    with open(args.out_json, "w", encoding="utf-8") as f:
        json.dump(out, f, ensure_ascii=False, indent=2)

    plot_cosine_hist_with_baseline(
        values_real=key_cos_real,
        out_png=args.out_png_hist_key,
        title="Student-vs-teacher-mixture cosine (key)",
        color="#8a3434",
        mean_real_20=key_mean_real_20,
        mean_shuffled_20=key_mean_shuf_20,
    )
    plot_cosine_hist_with_baseline(
        values_real=value_cos_real,
        out_png=args.out_png_hist_value,
        title="Student-vs-teacher-mixture cosine (value)",
        color="#5f5a4a",
        mean_real_20=value_mean_real_20,
        mean_shuffled_20=value_mean_shuf_20,
    )
    plot_cosine_hist_with_baseline(
        values_real=key_cos_topk,
        out_png=args.out_png_hist_key_topk,
        title=f"Student-vs-teacher-mixture cosine (key, top-{out['topk_teacher_approx']['k']})",
        color="#8a3434",
        mean_real_20=key_mean_topk_real_20,
        mean_shuffled_20=key_mean_topk_shuf_20,
    )
    plot_cosine_hist_with_baseline(
        values_real=value_cos_topk,
        out_png=args.out_png_hist_value_topk,
        title=f"Student-vs-teacher-mixture cosine (value, top-{out['topk_teacher_approx']['k']})",
        color="#5f5a4a",
        mean_real_20=value_mean_topk_real_20,
        mean_shuffled_20=value_mean_topk_shuf_20,
    )

    print(f"Saved JSON: {args.out_json}")
    print(f"Saved key scatter grid: {args.out_png_scatter_key}")
    print(f"Saved value scatter grid: {args.out_png_scatter_value}")
    print(f"Saved key histogram: {args.out_png_hist_key}")
    print(f"Saved value histogram: {args.out_png_hist_value}")
    print(f"Saved key histogram (top-{out['topk_teacher_approx']['k']}): {args.out_png_hist_key_topk}")
    print(f"Saved value histogram (top-{out['topk_teacher_approx']['k']}): {args.out_png_hist_value_topk}")
    print(f"Mean key cosine: {out['mean_key_cosine']:.4f}")
    print(f"Mean value cosine: {out['mean_value_cosine']:.4f}")
    print(f"Mean key cosine (real/shuffled, {n_s20} students): {key_mean_real_20:.4f} / {key_mean_shuf_20:.4f}")
    print(f"Mean value cosine (real/shuffled, {n_s20} students): {value_mean_real_20:.4f} / {value_mean_shuf_20:.4f}")


if __name__ == "__main__":
    main()
