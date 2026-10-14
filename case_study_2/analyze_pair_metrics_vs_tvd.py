from __future__ import annotations

import argparse
import os
from itertools import combinations

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
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


def sample_pairs(n_items: int, num_pairs: int, seed: int) -> np.ndarray:
    total = n_items * (n_items - 1) // 2
    if num_pairs <= 0 or num_pairs >= total:
        return np.array(list(combinations(np.arange(n_items), 2)), dtype=np.int64)
    rng = np.random.default_rng(seed)
    pairs = set()
    while len(pairs) < num_pairs:
        i = int(rng.integers(0, n_items))
        j = int(rng.integers(0, n_items - 1))
        if j >= i:
            j += 1
        a, b = (i, j) if i < j else (j, i)
        pairs.add((a, b))
    return np.array(sorted(pairs), dtype=np.int64)


def cosine_for_pairs(mat: np.ndarray, pairs: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(mat, axis=1, keepdims=True) + 1e-12
    z = mat / norms
    return np.sum(z[pairs[:, 0]] * z[pairs[:, 1]], axis=1)


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", type=str, default="out/main_mlp_mult_32_K_8/ckpt_iter300000.pt")
    p.add_argument("--student_ckpt", type=str, default="out/distill_suite/target_models/results_real_h10240_knone/h1024_knone_relu1_final.pt")
    p.add_argument("--data_dir", type=str, default="data/tinystories")
    p.add_argument("--layer_idx", type=int, default=2)
    p.add_argument("--split", choices=["train", "val"], default="val")
    p.add_argument("--num_batches", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=8)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--num_teacher_neurons", type=int, default=200)
    p.add_argument(
        "--num_pairs",
        type=int,
        default=0,
        help="number of teacher pairs to sample; <=0 means all possible pairs",
    )
    p.add_argument(
        "--min_teacher_active_count",
        type=int,
        default=50,
        help="filter out teacher neurons that activated fewer than this many times; pairs containing them are removed",
    )
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--out_csv", type=str, default="pair_metrics_vs_tvd.csv")
    p.add_argument("--out_png", type=str, default="pair_metrics_vs_tvd_scatter.png")
    args = p.parse_args()

    device = args.device

    # Load teacher
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    t_cfg = TransformerConfig(**t_ckpt["model_cfg"])
    teacher = SparseTransformerLM(t_cfg).to(device)
    teacher.load_state_dict(t_ckpt["model"], strict=False)
    teacher.eval()
    blk = teacher.blocks[args.layer_idx]
    teacher_hidden = int(t_cfg.n_embd * t_cfg.mlp_mult)

    # Load student
    s_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    student = StudentMLP(
        d_in=t_cfg.n_embd,
        d_hidden=int(s_ckpt["hidden_size"]),
        d_out=t_cfg.n_embd,
        mlp_topk=s_ckpt.get("mlp_topk", None),
        use_relu_before_topk=bool(s_ckpt.get("use_relu_before_topk", True)),
    ).to(device)
    student.load_state_dict(s_ckpt["state_dict"], strict=True)
    student.eval()
    student_hidden = int(s_ckpt["hidden_size"])

    if args.num_teacher_neurons > teacher_hidden:
        raise SystemExit("num_teacher_neurons is larger than teacher hidden size")

    rng_np = np.random.default_rng(args.seed)
    teacher_idx = np.sort(rng_np.choice(teacher_hidden, size=args.num_teacher_neurons, replace=False))
    pairs = sample_pairs(args.num_teacher_neurons, args.num_pairs, args.seed + 1)
    pi = torch.from_numpy(pairs[:, 0]).long()
    pj = torch.from_numpy(pairs[:, 1]).long()

    # Data
    split_path = "train.bin" if args.split == "train" else "val.bin"
    data = load_bin(os.path.join(args.data_dir, split_path))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(args.seed))

    # Accumulators
    # We avoid building [N_tokens, num_pairs] tensors (which cause OOM when num_pairs is large).
    # Instead we accumulate:
    # - act_counts: [M]
    # - co_counts: [M, M] with co_counts[a,b] = count(tokens where teacher a & b active)
    M = args.num_teacher_neurons
    act_counts = torch.zeros((M,), dtype=torch.float64)
    co_counts = torch.zeros((M, M), dtype=torch.float64)
    numer = torch.zeros((M, student_hidden), dtype=torch.float64)
    total_tokens = 0

    # Weight student neuron contributions by their value-vector norm.
    # Student fc2 weight has shape [d_out, hidden]; each column corresponds to value vector v_i.
    student_value_norms = student.fc2.weight.detach().norm(dim=0).to(torch.float64).cpu()  # [Hs]

    cache = {}

    def ln2_hook(_m, _inp, out):
        cache["u"] = out.detach()

    def t_act_hook(_m, _inp, out):
        cache["t_act"] = out.detach()

    h1 = blk.ln2.register_forward_hook(ln2_hook)
    h2 = blk.mlp.act.register_forward_hook(t_act_hook)

    for _ in range(args.num_batches):
        xb, _ = get_batch(data, args.batch_size, args.block_size, device, rng)
        _logits, _loss = teacher(xb, None)
        u = cache["u"].reshape(-1, cache["u"].size(-1))                    # [N, d]
        t_act = cache["t_act"].reshape(-1, cache["t_act"].size(-1))        # [N, Ht]
        s_act = student.hidden_activations(u).to(torch.float64)            # [N, Hs]

        # t_bin: [N, M]
        t_bin = (t_act[:, teacher_idx] != 0).to(torch.float64).cpu()

        act_counts += t_bin.sum(dim=0)
        # co_counts += t_bin^T t_bin => [M, M]
        # This counts joint activations over tokens.
        co_counts += t_bin.t().mm(t_bin)

        # Weighted conditional expectation numerator.
        s_weighted_cpu = s_act.cpu() * student_value_norms.unsqueeze(0)    # [N, Hs]
        numer += t_bin.t().mm(s_weighted_cpu)

        total_tokens += int(t_bin.shape[0])

    h1.remove()
    h2.remove()

    # Filter out low-activation teacher neurons (and remove any pairs involving them).
    act_counts_np = act_counts.numpy()
    valid_teacher = act_counts_np >= float(args.min_teacher_active_count)
    pair_mask = valid_teacher[pairs[:, 0]] & valid_teacher[pairs[:, 1]]
    pairs = pairs[pair_mask]
    if pairs.shape[0] == 0:
        raise SystemExit(
            f"No pairs left after filtering by min_teacher_active_count={args.min_teacher_active_count}."
        )

    # Student distributions p_j over student neurons (for each teacher neuron j)
    numer_np = numer.numpy()
    dist = numer_np / (numer_np.sum(axis=1, keepdims=True) + 1e-12)  # [M, Hs]

    # TVD for pairs
    tvd = 0.5 * np.abs(dist[pairs[:, 0]] - dist[pairs[:, 1]]).sum(axis=1)

    # Coactivation metrics
    p1 = (act_counts.numpy() / max(total_tokens, 1))  # [M]
    # p11 for each pair from co_counts matrix
    p11 = (co_counts.numpy()[pairs[:, 0], pairs[:, 1]] / max(total_tokens, 1))
    p_i = p1[pairs[:, 0]]
    p_j = p1[pairs[:, 1]]
    union = np.clip(p_i + p_j - p11, 1e-12, None)
    jaccard = p11 / union
    phi_den = np.sqrt(np.clip(p_i * (1 - p_i) * p_j * (1 - p_j), 1e-12, None))
    phi = (p11 - p_i * p_j) / phi_den
    pmi = np.log((p11 + 1e-12) / np.clip(p_i * p_j, 1e-12, None))

    # Vector-similarity metrics
    teacher_fc1 = blk.mlp.fc1.weight.detach().cpu().numpy()[teacher_idx]   # keys [M, d]
    teacher_vals = blk.mlp.fc2.weight.detach().cpu().numpy().T[teacher_idx]  # values [M, d]
    value_cos = cosine_for_pairs(teacher_vals, pairs)
    key_cos = cosine_for_pairs(teacher_fc1, pairs)
    value_l2 = np.linalg.norm(teacher_vals[pairs[:, 0]] - teacher_vals[pairs[:, 1]], axis=1)

    # 6 metrics: p11, jaccard, phi, pmi, value_cos, key_cos
    df = pd.DataFrame(
        {
            "teacher_i": teacher_idx[pairs[:, 0]],
            "teacher_j": teacher_idx[pairs[:, 1]],
            "tvd": tvd,
            "p11_coactivation": p11,
            "jaccard_active": jaccard,
            "phi_corr": phi,
            "pmi_active": pmi,
            "value_cos": value_cos,
            "key_cos": key_cos,
            "value_l2": value_l2,
            "p_i": p_i,
            "p_j": p_j,
        }
    )
    df.to_csv(args.out_csv, index=False)

    # Scatter plots for 6 requested metrics
    eps = 1e-12
    phi_log_signed = np.sign(df["phi_corr"].values) * np.log10(np.abs(df["phi_corr"].values) + eps)
    p11_log10 = np.log10(df["p11_coactivation"].values + eps)
    jaccard_log10 = np.log10(df["jaccard_active"].values + eps)

    metrics = [
        ("p11_log10", "log10 P(i,j active)"),
        ("jaccard_log10", "log10 Jaccard(active sets)"),
        ("phi_log_signed", "signed log10 Phi correlation"),
        ("pmi_active", "PMI(active)"),
        ("value_cos", "Cosine(value vectors)"),
        ("key_cos", "Cosine(key vectors)"),
    ]
    fig, axes = plt.subplots(2, 3, figsize=(16, 9))
    axes = axes.flatten()
    for ax, (col, title) in zip(axes, metrics):
        if col == "p11_log10":
            x = p11_log10
        elif col == "jaccard_log10":
            x = jaccard_log10
        elif col == "phi_log_signed":
            x = phi_log_signed
        else:
            x = df[col].values
        ax.scatter(x, df["tvd"].values, s=8, alpha=0.35, color="steelblue", edgecolors="none")
        ax.set_xlabel(title)
        ax.set_ylabel("TVD(student distributions)")
        ax.set_title(f"{title} vs TVD")
        ax.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(args.out_png, dpi=170)

    print(f"Saved pair table to: {args.out_csv}")
    print(f"Saved scatter figure to: {args.out_png}")
    print(f"Teacher neurons used: {args.num_teacher_neurons}, pairs plotted: {len(df)}")


if __name__ == "__main__":
    main()
