#!/usr/bin/env python3
"""
Unified script that produces exactly five figures:
  1) combined_concentration.png
  2) combined_concentration_filled.png  (no CI; area under curves)
  3) pmi_groups_tvd_hist.png
  4) student_teacher_key_cos_hist.png
  5) student_teacher_value_cos_hist.png
All data is collected in a single forward pass over the specified dataset split.
"""

from __future__ import annotations

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

from model_transformer import SparseTransformerLM, TransformerConfig
from utils.common import get_batch, load_bin

# ---------------------------------------------------------------------------
# Student model (shared by all sub‑analyses)
# ---------------------------------------------------------------------------
class StudentMLP(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        d_out: int,
        mlp_topk: int | None,
        use_relu_before_topk: bool = True,
    ):
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


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------
def cosine_normed_rows(mat: torch.Tensor) -> torch.Tensor:
    return mat / (mat.norm(dim=1, keepdim=True) + 1e-12)


def _interp_curve_y(x_src: np.ndarray, y_src: np.ndarray, x_dst: np.ndarray) -> np.ndarray:
    """Interpolate (x_src, y_src) onto x_dst (x_src must be strictly increasing)."""
    order = np.argsort(x_src)
    xs = np.asarray(x_src, dtype=np.float64)[order]
    ys = np.asarray(y_src, dtype=np.float64)[order]
    keep = np.concatenate([[True], np.diff(xs) > 0])
    xs, ys = xs[keep], ys[keep]
    return np.interp(x_dst, xs, ys, left=float(ys[0]), right=float(ys[-1]))


def make_random_teacher_params(d_in: int, hidden: int, k: int, seed: int, device: str):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    w = torch.randn((hidden, d_in), generator=g, dtype=torch.float32) / np.sqrt(float(d_in))
    b = torch.zeros((hidden,), dtype=torch.float32)
    return {"w": w.to(device), "b": b.to(device), "k": int(k), "use_relu": True}


def _topk_act(x: torch.Tensor, k: int | None, use_relu: bool = True) -> torch.Tensor:
    if use_relu:
        x = torch.relu(x)
    if k is None:
        return x
    vals, idx = torch.topk(x, k=int(k), dim=-1, sorted=False)
    out = torch.zeros_like(x)
    batch_idx = torch.arange(x.shape[0], device=x.device, dtype=torch.long)[:, None]
    out[batch_idx, idx] = vals
    return out


@torch.no_grad()
def calibrate_random_student_bias(
    student_rand: StudentMLP,
    teacher: SparseTransformerLM | None,
    layer_idx: int,
    data,
    batch_size: int,
    block_size: int,
    device: str,
    seed: int,
    target_active_count: float,
    calib_batches: int,
    use_gaussian_input: bool = False,
    gaussian_sigma: float = 1.0,
    d_in: int | None = None,
) -> float:
    """Set a single shared bias so that random student matches target mean #active neurons/token."""
    if student_rand.mlp_topk is not None:
        # With TopK the active count is largely fixed, keep default bias.
        return float(student_rand.fc1.bias.mean().item())

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) + 12345)
    blk = teacher.blocks[layer_idx] if teacher is not None else None
    z_all = []
    for _ in range(calib_batches):
        if use_gaussian_input:
            if d_in is None:
                raise ValueError("d_in must be given for gaussian calibration")
            n = batch_size * block_size
            u = torch.randn((n, d_in), generator=rng, device="cpu", dtype=torch.float32) * float(gaussian_sigma)
            u = u.to(device)
        else:
            if teacher is None or blk is None:
                raise ValueError("teacher required when not using gaussian")
            u_cache = {}
            def ln2_hook(_mod, _inp, out):
                u_cache["u"] = out.detach()
            h = blk.ln2.register_forward_hook(ln2_hook)
            xb, _ = get_batch(data, batch_size, block_size, device, rng)
            _logits, _loss = teacher(xb, None)
            h.remove()
            u = u_cache["u"].reshape(-1, u_cache["u"].size(-1))
        z = torch.addmm(student_rand.fc1.bias.detach(), u, student_rand.fc1.weight.detach().t())
        z_all.append(z.reshape(-1).cpu())

    z_flat = torch.cat(z_all, dim=0).numpy()
    target_prob = np.clip(float(target_active_count / max(student_rand.fc1.out_features, 1)), 1e-6, 1 - 1e-6)
    q = np.quantile(z_flat, 1.0 - target_prob)
    b = float(-q)
    student_rand.fc1.bias.data.fill_(b)
    return b


@torch.no_grad()
def estimate_ln2_sigma(
    teacher: SparseTransformerLM,
    layer_idx: int,
    data,
    num_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    seed: int,
) -> float:
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) + 2222)
    blk = teacher.blocks[layer_idx]
    s = 0.0
    ss = 0.0
    c = 0
    cache = {}
    def ln2_hook(_mod, _inp, out):
        cache["u"] = out.detach()
    h = blk.ln2.register_forward_hook(ln2_hook)
    for _ in range(num_batches):
        xb, _ = get_batch(data, batch_size, block_size, device, rng)
        _logits, _loss = teacher(xb, None)
        u = cache["u"]
        s += float(u.sum().item())
        ss += float((u * u).sum().item())
        c += int(u.numel())
    h.remove()
    mean = s / max(c, 1)
    var = ss / max(c, 1) - mean * mean
    return float(max(var, 0.0) ** 0.5)


# ---------------------------------------------------------------------------
# Main script
# ---------------------------------------------------------------------------
@torch.no_grad()
def main():
    parser = argparse.ArgumentParser(description="Unified analysis script for four figures.")
    # ---- Common ----
    parser.add_argument("--teacher_ckpt", type=str, default="out/main_mlp_mult_32_K_8/ckpt_iter300000.pt")
    parser.add_argument("--student_ckpt", type=str, default="out/distill_suite/target_models/results_real_h10240_knone/h1024_knone_relu1_final.pt")
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument("--layer_idx", type=int, default=2)
    parser.add_argument("--split", type=str, choices=["train", "val"], default="val")
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--block_size", type=int, default=256)
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--seed", type=int, default=0)

    # ---- Figure 1: concentration curves ----
    parser.add_argument("--num_teacher_neurons_for_conc", type=int, default=200)
    parser.add_argument("--min_denom_count", type=int, default=50)
    parser.add_argument("--num_student_for_coverage", type=int, default=20)
    parser.add_argument("--calib_batches", type=int, default=100)
    parser.add_argument("--out_combined_concentration", type=str,
                        default="figures/combined_concentration.png")
    parser.add_argument("--out_combined_concentration_filled", type=str,
                        default="figures/combined_concentration_filled.png")

    # ---- Figure 2: PMI/TVD histogram ----
    parser.add_argument("--min_teacher_active_tvd", type=int, default=50)
    parser.add_argument("--tvd_bins", type=int, default=50)
    parser.add_argument("--tvd_min", type=float, default=0.0)
    parser.add_argument("--tvd_max", type=float, default=65)
    parser.add_argument("--pmi_group_sample_size", type=int, default=5000)
    parser.add_argument("--out_pmi_tvd_hist", type=str,
                        default="figures/pmi_groups_tvd_hist.png")

    # ---- Figures 3 & 4: key/value cosine histograms ----
    parser.add_argument("--num_rand_teacher_for_cos", type=int, default=20)
    parser.add_argument("--num_student_for_baseline_cos", type=int, default=20)
    parser.add_argument("--out_key_cos_hist", type=str,
                        default="figures/student_teacher_key_cos_hist.png")
    parser.add_argument("--out_value_cos_hist", type=str,
                        default="figures/student_teacher_value_cos_hist.png")

    args = parser.parse_args()
    for out_path in [args.out_combined_concentration, args.out_combined_concentration_filled,
                     args.out_pmi_tvd_hist, args.out_key_cos_hist, args.out_value_cos_hist]:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = args.device
    # -----------------------------------------------------------------------
    # 1. Load models
    # -----------------------------------------------------------------------
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    t_cfg = TransformerConfig(**t_ckpt["model_cfg"])
    teacher_hidden = int(t_cfg.n_embd * t_cfg.mlp_mult)
    teacher = SparseTransformerLM(t_cfg).to(device)
    teacher.load_state_dict(t_ckpt["model"], strict=False)
    teacher.eval()
    blk = teacher.blocks[args.layer_idx]

    if t_cfg.mlp_topk is None or int(t_cfg.mlp_topk) <= 0:
        raise SystemExit("Teacher must use TopK with k > 0.")
    teacher_k = int(t_cfg.mlp_topk)
    teacher_use_relu = bool(getattr(t_cfg, "use_relu_before_topk", True))

    s_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    s_hidden = int(s_ckpt["hidden_size"])
    s_topk = s_ckpt.get("mlp_topk", None)
    s_relu = bool(s_ckpt.get("use_relu_before_topk", True))
    student = StudentMLP(
        d_in=t_cfg.n_embd, d_hidden=s_hidden, d_out=t_cfg.n_embd,
        mlp_topk=s_topk, use_relu_before_topk=s_relu,
    ).to(device)
    student.load_state_dict(s_ckpt["state_dict"], strict=True)
    student.eval()

    # Teacher weight matrices (for cosine analysis)
    teacher_keys = blk.mlp.fc1.weight.detach().cpu()                     # [T, d_in]
    teacher_values = blk.mlp.fc2.weight.detach().cpu().t().contiguous()  # [T, d_out]
    student_keys = student.fc1.weight.detach().cpu()                     # [S, d_in]
    student_values = student.fc2.weight.detach().cpu().t().contiguous()  # [S, d_out]
    student_value_norms = student.fc2.weight.detach().norm(dim=0).to(torch.float32).cpu()

    # -----------------------------------------------------------------------
    # 2. Single forward pass: collect u, t_act, s_act
    # -----------------------------------------------------------------------
    split_path = "train.bin" if args.split == "train" else "val.bin"
    data = load_bin(os.path.join(args.data_dir, split_path))
    rng_cpu = torch.Generator(device="cpu").manual_seed(int(args.seed))

    M = teacher_hidden
    Hs = s_hidden
    K = teacher_k

    # Accumulators (all on CPU to save GPU memory)
    act_counts = torch.zeros((M,), dtype=torch.float64, device="cpu")
    co_counts = torch.zeros((M, M), dtype=torch.float64, device="cpu")
    numer_raw = torch.zeros((M, Hs), dtype=torch.float64, device="cpu")
    numer_weighted = torch.zeros((M, Hs), dtype=torch.float64, device="cpu")
    total_tokens = 0

    for step in range(int(args.num_batches)):
        # Hooks to capture activations
        cache = {}
        def ln2_hook(_mod, _inp, out):
            cache["u"] = out.detach()
        def t_act_hook(_mod, _inp, out):
            cache["t_act"] = out.detach()

        h1 = blk.ln2.register_forward_hook(ln2_hook)
        h2 = blk.mlp.act.register_forward_hook(t_act_hook)
        xb, _ = get_batch(data, args.batch_size, args.block_size, device, rng_cpu)
        _logits, _loss = teacher(xb, None)
        h1.remove()
        h2.remove()

        u = cache["u"].reshape(-1, cache["u"].size(-1))          # [N, d_in] on device
        t_act = cache["t_act"].reshape(-1, cache["t_act"].size(-1))  # [N, T] on device
        s_act = student.hidden_activations(u).cpu().to(torch.float64)   # [N, S] CPU
        s_weighted = s_act * student_value_norms.unsqueeze(0)           # CPU

        t_bin = (t_act != 0).to(torch.float64).cpu()  # [N, T] CPU
        N = t_bin.size(0)
        total_tokens += N

        # Accumulate act_counts and conditional expectations
        act_counts += t_bin.sum(dim=0)
        numer_raw += t_bin.t().mm(s_act)
        numer_weighted += t_bin.t().mm(s_weighted)

        # ---- Co‑occurrence counts (ordered pairs) from TopK indices ----
        # Правильно: получаем и topk_idx, и topk_vals (pre‑activation до обнуления),
        # как в оригинальном скрипте.
        W1 = blk.mlp.fc1.weight.detach()
        b1 = blk.mlp.fc1.bias.detach()
        h1_pre = torch.addmm(b1, u, W1.t())   # [N, T] на device
        if teacher_use_relu:
            h1_pre = torch.relu(h1_pre)

        # Берём TopK со значениями – именно они показывают, активен ли нейрон
        topk_vals, topk_idx = torch.topk(h1_pre, k=K, dim=1, sorted=False)  # [N, K] на device

        # Маска для TopK позиций: True, если значение > 0 (нейрон действительно активен)
        active_k_mask = (topk_vals != 0).cpu()   # [N, K] bool CPU
        topk_idx = topk_idx.cpu()                # [N, K] int64 CPU

        a = topk_idx.unsqueeze(2).expand(N, K, K)
        b = topk_idx.unsqueeze(1).expand(N, K, K)
        m = active_k_mask.unsqueeze(2) & active_k_mask.unsqueeze(1)  # [N, K, K] bool CPU
        a_flat = a[m]
        b_flat = b[m]
        if a_flat.numel() > 0:
            co_counts[a_flat, b_flat] += 1.0

        if (step + 1) % 25 == 0:
            print(f"[collect] {step+1}/{args.num_batches} total_tokens={total_tokens}")

    # Derived probabilities
    p_act = act_counts / total_tokens
    cond_raw = numer_raw / act_counts.clamp_min(1.0).unsqueeze(1)
    cond_weighted = numer_weighted / act_counts.clamp_min(1.0).unsqueeze(1)

    # -----------------------------------------------------------------------
    # 3. Figure 2: PMI/TVD histogram
    # -----------------------------------------------------------------------
    print("Generating PMI/TVD histogram...")
    valid_tvd = act_counts >= args.min_teacher_active_tvd
    tvd_idx = torch.nonzero(valid_tvd, as_tuple=False).squeeze(1)
    Mv = tvd_idx.numel()
    if Mv < 2:
        raise SystemExit("Too few active teacher neurons for TVD computation.")

    act_v = p_act[tvd_idx]
    co_v = co_counts[tvd_idx][:, tvd_idx] / total_tokens  # joint probability p11(i,j)
    # distribution over student neurons conditioned on teacher (weighted)
    sum_w = cond_weighted[tvd_idx].sum(dim=1, keepdim=True)
    dist_v = cond_weighted[tvd_idx] / (sum_w + 1e-12)   # [Mv, Hs]

    nb = int(args.tvd_bins)
    tvd_edges = np.linspace(args.tvd_min, args.tvd_max, nb + 1)
    group_labels = ["pmi>5", "5>=pmi>0", "0>pmi"]
    group_tvd = {k: [] for k in group_labels}
    eps = 1e-12

    pair_chunk = 32
    for i0 in range(0, Mv, pair_chunk):
        i1 = min(Mv, i0 + pair_chunk)
        dist_i = dist_v[i0:i1]
        for j0 in range(i0, Mv, pair_chunk):
            j1 = min(Mv, j0 + pair_chunk)
            dist_j = dist_v[j0:j1]
            co_block = co_v[i0:i1, j0:j1]

            p_i = act_v[i0:i1].unsqueeze(1)
            p_j = act_v[j0:j1].unsqueeze(0)
            p11 = co_block

            pmi = torch.log((p11 + eps) / (p_i * p_j).clamp_min(eps))
            diff = dist_i.unsqueeze(1) - dist_j.unsqueeze(0)
            tvd_block = 0.5 * diff.abs().sum(dim=-1)

            # Upper triangular only for i==j block
            if j0 == i0:
                mask = torch.triu(torch.ones((i1 - i0, j1 - j0), dtype=torch.bool, device=tvd_block.device), diagonal=1)
                tvd_flat = tvd_block[mask].cpu().numpy()
                pmi_flat = pmi[mask].cpu().numpy()
            else:
                tvd_flat = tvd_block.cpu().numpy().ravel()
                pmi_flat = pmi.cpu().numpy().ravel()

            m0 = pmi_flat > 5.0
            m1 = (pmi_flat <= 5.0) & (pmi_flat > 0.0)
            m2 = pmi_flat < 0.0
            for msk, grp in [(m0, "pmi>5"), (m1, "5>=pmi>0"), (m2, "0>pmi")]:
                if msk.any():
                    group_tvd[grp].extend(tvd_flat[msk].tolist())

    # Plot histogram
    fig_h, ax_h = plt.subplots(figsize=(8, 8))
    group_colors = {"pmi>5": "#7d1111", "5>=pmi>0": "#85472a", "0>pmi": "#dca35b"}
    legend_labels = {
        "pmi>5": r"$\mathrm{PMI} > 5$",
        "5>=pmi>0": r"$5 \geq \mathrm{PMI} > 0$",
        "0>pmi": r"$0 > \mathrm{PMI}$",
    }
    rng = np.random.default_rng(int(args.seed))
    for label in group_labels:
        vals = np.asarray(group_tvd[label], dtype=np.float32)
        if vals.size == 0:
            continue
        if vals.size > int(args.pmi_group_sample_size):
            idx = rng.choice(vals.size, size=int(args.pmi_group_sample_size), replace=False)
            vals = vals[idx]
        ax_h.hist(vals, bins=50, alpha=0.65, color=group_colors[label],
                  label=legend_labels[label], density=True)
        ax_h.hist(vals, bins=50, histtype="step", color="black",
                  linewidth=1.2, density=True)
        ax_h.axvline(vals.mean(), color=group_colors[label],
                     linestyle="--", linewidth=1.5, alpha=0.95)
    ax_h.set_xlabel("Total Variation Distance (TVD)", fontsize=17)
    ax_h.set_ylabel("Density", fontsize=17)
    ax_h.tick_params(axis='both', which='major', labelsize=15)
    ax_h.grid(True, alpha=0.25)
    ax_h.set_xlim(0, 0.72)
    ax_h.legend(loc="best", fontsize=17)
    fig_h.tight_layout()
    fig_h.savefig(args.out_pmi_tvd_hist, dpi=180, bbox_inches="tight")
    plt.close(fig_h)
    print(f"[saved] {args.out_pmi_tvd_hist}")

    # -----------------------------------------------------------------------
    # 4. Figure 1: Combined concentration curves
    # -----------------------------------------------------------------------
    print("Generating combined concentration plot...")
    # 4a. Teacher → Student (weighted)
    valid_conc = act_counts >= args.min_denom_count
    f_real = cond_weighted[valid_conc].sort(dim=1, descending=True).values
    f_real = f_real.cumsum(dim=1) / f_real.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mean_t2s = f_real.mean(dim=0).numpy()
    lower_t2s = torch.quantile(f_real, 0.025, dim=0).numpy()
    upper_t2s = torch.quantile(f_real, 0.975, dim=0).numpy()
    ks_t2s = np.arange(1, Hs + 1)

    # 4b. Student → Teacher (using threshold method)
    probs_t_given_s = cond_raw / cond_raw.sum(dim=0, keepdim=True).clamp_min(1e-12)  # [T, S]
    rs = np.random.default_rng(int(args.seed))
    n_stud = min(int(args.num_student_for_coverage), Hs)
    stud_idx = np.sort(rs.choice(Hs, size=n_stud, replace=False))
    thresholds = np.array(
        [i / 100.0 for i in range(1, 11)] + [i / 100.0 for i in range(20, 101, 10)],
        dtype=np.float32,
    )
    k_matrix = []
    for i in stud_idx:
        p = probs_t_given_s[:, i]
        if p.sum() == 0:
            continue
        sorted_p, _ = torch.sort(p, descending=True)
        cdf = torch.cumsum(sorted_p, dim=0)
        ks = []
        for thr in thresholds:
            k = torch.searchsorted(cdf, thr, right=False).item() + 1
            ks.append(min(k, M))
        k_matrix.append(ks)
    if not k_matrix:
        raise RuntimeError("No valid students for coverage analysis")
    k_mat = np.array(k_matrix, dtype=np.float32)
    mean_s2t = k_mat.mean(axis=0)
    lower_s2t = np.quantile(k_mat, 0.025, axis=0)
    upper_s2t = np.quantile(k_mat, 0.975, axis=0)

    # 4c. Random baseline (Teacher → Student, weighted)
    ln2_sigma = estimate_ln2_sigma(
        teacher, args.layer_idx, data,
        max(20, min(args.num_batches, args.calib_batches)),
        args.batch_size, args.block_size, device, args.seed,
    )

    # Create random student and calibrate its bias
    student_rand = StudentMLP(
        d_in=t_cfg.n_embd, d_hidden=Hs, d_out=t_cfg.n_embd,
        mlp_topk=s_topk, use_relu_before_topk=s_relu,
    ).to(device)
    torch.manual_seed(int(args.seed) + 777)
    nn.init.xavier_uniform_(student_rand.fc1.weight)
    nn.init.xavier_uniform_(student_rand.fc2.weight)
    nn.init.zeros_(student_rand.fc1.bias)
    nn.init.zeros_(student_rand.fc2.bias)

    # Mean active count from real student (approximate)
    mean_active_real = (act_counts / total_tokens * M).mean().item()
    bias_shift = calibrate_random_student_bias(
        student_rand, teacher, args.layer_idx, data,
        args.batch_size, args.block_size, device, args.seed,
        mean_active_real, args.calib_batches,
        use_gaussian_input=True, gaussian_sigma=ln2_sigma,
        d_in=t_cfg.n_embd,
    )
    # Move random student to CPU for the baseline sweep
    student_rand.to("cpu")
    student_rand.eval()

    # Create random teacher parameters (k=8 as in original script)
    rtc = make_random_teacher_params(
        d_in=t_cfg.n_embd, hidden=M, k=8,
        seed=args.seed + 2026, device="cpu",
    )
    W_rand = rtc["w"]       # [M, d_in]
    b_rand = rtc["b"]       # [M]

    act_rand = torch.zeros((M,), dtype=torch.float64, device="cpu")
    numer_rand = torch.zeros((M, Hs), dtype=torch.float64, device="cpu")

    rng_cpu = torch.Generator(device="cpu").manual_seed(int(args.seed) + 9999)
    for _ in range(int(args.num_batches)):
        n = args.batch_size * args.block_size
        u = torch.randn((n, t_cfg.n_embd), generator=rng_cpu, dtype=torch.float32) * ln2_sigma
        t_pre = torch.addmm(b_rand, u, W_rand.t())
        t_act_rand = _topk_act(t_pre, k=8, use_relu=True).cpu()   # [N, M] CPU
        t_bin_rand = (t_act_rand != 0).to(torch.float64)
        s_act_rand = student_rand.hidden_activations(u).cpu().to(torch.float64)
        s_weighted_rand = s_act_rand * student_rand.fc2.weight.detach().norm(dim=0).cpu().unsqueeze(0)

        act_rand += t_bin_rand.sum(dim=0)
        numer_rand += t_bin_rand.t().mm(s_weighted_rand)

    cond_rand = numer_rand / act_rand.clamp_min(1.0).unsqueeze(1)
    valid_rand = act_rand >= args.min_denom_count
    if valid_rand.any():
        f_rand = cond_rand[valid_rand].sort(dim=1, descending=True).values
        f_rand = f_rand.cumsum(dim=1) / f_rand.sum(dim=1, keepdim=True).clamp_min(1e-12)
        mean_rand = f_rand.mean(dim=0).numpy()
    else:
        mean_rand = np.zeros(Hs) * np.nan

    # 4d. Plot
    fig, ax = plt.subplots(figsize=(11, 8))
    color_t2s = "#8a3434"
    color_s2t = "#dca35b"
    color_baseline = "gray"

    ax.plot(ks_t2s, mean_t2s, color=color_t2s, lw=4, label="Teacher → Student")
    ax.fill_between(ks_t2s, lower_t2s, upper_t2s, color=color_t2s, alpha=0.45)

    ax.plot(mean_s2t, thresholds, color=color_s2t, lw=4, label="Student → Teacher")
    ax.fill_betweenx(thresholds, lower_s2t, upper_s2t, color=color_s2t, alpha=0.25)

    if not np.isnan(mean_rand).all():
        ax.plot(ks_t2s, mean_rand, color=color_baseline, lw=4, ls="--",
                label="Random baseline (T→S)")

    ax.axhline(y=0.8, color="red", linestyle="--", linewidth=2.5, alpha=0.8)

    idx_t2s = np.searchsorted(mean_t2s, 0.8, side="left")
    k_t2s = ks_t2s[min(idx_t2s, len(ks_t2s) - 1)]
    idx_bl = np.searchsorted(mean_rand, 0.8, side="left") if not np.isnan(mean_rand).all() else len(ks_t2s) - 1
    k_bl = ks_t2s[min(idx_bl, len(ks_t2s) - 1)]
    thr_idx = np.argmin(np.abs(thresholds - 0.8))
    k_s2t = int(np.round(mean_s2t[thr_idx]))

    ax.scatter([k_t2s, k_bl, k_s2t], [0.8, 0.8, 0.8], color="red", s=60, zorder=5)
    for k_val, lbl in [(k_t2s, str(k_t2s)), (k_bl, str(k_bl)), (k_s2t, str(k_s2t))]:
        ax.annotate(lbl, (k_val, 0.8), textcoords="offset points", xytext=(-5, 5),
                    color="red", fontsize=14, fontweight="bold",
                    ha="right", va="bottom")

    ax.set_xscale("log")
    ax.set_xlim(1, 1000)
    ax.tick_params(axis='both', which='major', labelsize=15)
    ax.set_xlabel("Top Neurons Kept (log scale)", fontsize=17)
    ax.set_ylabel("Cumulative Mass Fraction", fontsize=17)
    ax.set_ylim(0.0, 1.01)
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=17, framealpha=0.9)
    fig.tight_layout()
    fig.savefig(args.out_combined_concentration, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[saved] {args.out_combined_concentration}")

    # 4e. Same plot without CI; grey random fill+line on top (covers curves below rand)
    print("Generating combined concentration plot (filled, no CI)...")
    fig_f, ax_f = plt.subplots(figsize=(11, 8))
    x_fill = ks_t2s.astype(np.float64)
    y_t2s_fill = mean_t2s.astype(np.float64)
    y_s2t_fill = _interp_curve_y(mean_s2t, thresholds, x_fill)
    fill_alpha = 0.42
    has_rand = not np.isnan(mean_rand).all()
    y_rand_fill = mean_rand.astype(np.float64) if has_rand else None

    ax_f.fill_between(
        x_fill, 0.0, y_s2t_fill, color=color_s2t, alpha=fill_alpha,
        edgecolor="none", zorder=1,
    )
    ax_f.fill_between(
        x_fill, y_s2t_fill, y_t2s_fill, color=color_t2s, alpha=fill_alpha,
        edgecolor="none", zorder=2,
    )

    n_markers = 20
    x_mark = np.geomspace(1.0, 1000.0, n_markers)
    y_t2s_mark = _interp_curve_y(x_fill, y_t2s_fill, x_mark)
    y_s2t_mark = _interp_curve_y(mean_s2t, thresholds, x_mark)
    marker_kw = dict(s=55, zorder=4, linewidths=0.8, edgecolors="white")

    ax_f.plot(
        x_fill, y_t2s_fill, color=color_t2s, lw=4, label="Teacher → Student", zorder=3,
    )
    ax_f.scatter(x_mark, y_t2s_mark, color=color_t2s, **marker_kw)
    ax_f.plot(mean_s2t, thresholds, color=color_s2t, lw=4, label="Student → Teacher", zorder=3)
    ax_f.scatter(x_mark, y_s2t_mark, color=color_s2t, **marker_kw)

    color_baseline_fill = "#d6d6d6"
    color_baseline_line = "black"
    if has_rand:
        ax_f.fill_between(
            x_fill, 0.0, y_rand_fill, color=color_baseline_fill, alpha=1.0,
            edgecolor="none", zorder=5,
        )
        ax_f.plot(
            x_fill, y_rand_fill, color=color_baseline_line, lw=4, ls="--",
            label="Random baseline (T→S)", zorder=6,
        )

    ax_f.axhline(y=0.8, color="red", linestyle="--", linewidth=2.5, alpha=0.8, zorder=7)
    ax_f.scatter(
        [k_t2s, k_bl, k_s2t], [0.8, 0.8, 0.8], color="red", s=60, zorder=8,
    )
    for k_val, lbl in [(k_t2s, str(k_t2s)), (k_bl, str(k_bl)), (k_s2t, str(k_s2t))]:
        ax_f.annotate(
            lbl, (k_val, 0.8), textcoords="offset points", xytext=(-5, 5),
            color="red", fontsize=14, fontweight="bold", ha="right", va="bottom", zorder=8,
        )

    ax_f.set_xscale("log")
    ax_f.set_xlim(1, 1000)
    ax_f.tick_params(axis="both", which="major", labelsize=15)
    ax_f.set_xlabel("Top Neurons Kept (log scale)", fontsize=17)
    ax_f.set_ylabel("Cumulative Mass Fraction", fontsize=17)
    ax_f.set_ylim(0.0, 1.01)
    ax_f.grid(True, alpha=0.25)
    ax_f.legend(loc="lower right", fontsize=17, framealpha=0.9)
    fig_f.tight_layout()
    fig_f.savefig(args.out_combined_concentration_filled, dpi=180, bbox_inches="tight")
    plt.close(fig_f)
    print(f"[saved] {args.out_combined_concentration_filled}")

    # -----------------------------------------------------------------------
    # 5. Figures 3 & 4: Student–teacher key/value cosine histograms
    # -----------------------------------------------------------------------
    print("Generating key/value cosine histograms...")
    # Already have probs_t_given_s from earlier
    approx_key = probs_t_given_s.t().to(torch.float32) @ teacher_keys.to(torch.float32)   # [S, D]
    approx_value = probs_t_given_s.t().to(torch.float32) @ teacher_values.to(torch.float32)
    key_cos = F.cosine_similarity(approx_key, student_keys.to(torch.float32), dim=1, eps=1e-12)
    value_cos = F.cosine_similarity(approx_value, student_values.to(torch.float32), dim=1, eps=1e-12)

    # Shuffled baseline
    probs_shuf = torch.empty_like(probs_t_given_s)
    for i in range(probs_t_given_s.size(1)):
        perm = rs.permutation(probs_t_given_s.size(0))
        probs_shuf[:, i] = probs_t_given_s[perm, i]
    approx_key_shuf = probs_shuf.t().to(torch.float32) @ teacher_keys.to(torch.float32)
    approx_value_shuf = probs_shuf.t().to(torch.float32) @ teacher_values.to(torch.float32)
    key_cos_shuf = F.cosine_similarity(approx_key_shuf, student_keys.to(torch.float32), dim=1, eps=1e-12)
    value_cos_shuf = F.cosine_similarity(approx_value_shuf, student_values.to(torch.float32), dim=1, eps=1e-12)

    n_s20 = min(int(args.num_student_for_baseline_cos), Hs)
    stud_idx_20 = np.sort(rs.choice(Hs, size=n_s20, replace=False))
    idx20 = torch.from_numpy(stud_idx_20)
    key_mean_real_20 = float(key_cos[idx20].mean().item())
    key_mean_shuf_20 = float(key_cos_shuf[idx20].mean().item())
    value_mean_real_20 = float(value_cos[idx20].mean().item())
    value_mean_shuf_20 = float(value_cos_shuf[idx20].mean().item())

    def plot_cos_hist(values, out_path, title, color, mean_real, mean_shuf):
        arr = values.numpy()
        fig, ax = plt.subplots(figsize=(8, 8))
        ax.hist(arr, bins=50, density=True, alpha=0.65, color=color)
        ax.hist(arr, bins=50, histtype="step", color="black", linewidth=1.2, density=True)
        # ax.axvline(mean_real, color=color, linestyle="--", linewidth=2.0,
        #            alpha=0.95, label="mean real")
        ax.axvline(mean_shuf, color="gray", linestyle="--", linewidth=3.0,
                   alpha=0.95, label="mean shuffled baseline")
        ax.set_xlabel("Cosine Similarity", fontsize=17)
        ax.set_ylabel("Density", fontsize=17)
        # ax.set_title(title, fontsize=15)
        ax.tick_params(axis='both', which='major', labelsize=15)
        ax.grid(True, alpha=0.25)
        ax.legend(loc="best", fontsize=17, framealpha=0.9)
        fig.tight_layout()
        fig.savefig(out_path, dpi=180, bbox_inches="tight")
        plt.close(fig)

    plot_cos_hist(key_cos, args.out_key_cos_hist,
                  "Student‑teacher mixture cosine (key)", "#8a3434",
                  key_mean_real_20, key_mean_shuf_20)
    plot_cos_hist(value_cos, args.out_value_cos_hist,
                  "Student‑teacher mixture cosine (value)", "#5f5a4a",
                  value_mean_real_20, value_mean_shuf_20)

    print(f"[saved] {args.out_key_cos_hist}")
    print(f"[saved] {args.out_value_cos_hist}")
    print("All figures generated successfully.")


if __name__ == "__main__":
    main()