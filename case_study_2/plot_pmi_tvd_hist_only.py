from __future__ import annotations

# pyright: reportMissingImports=false

import argparse
import os

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn

from model_transformer import SparseTransformerLM, TransformerConfig
from utils.common import get_batch, load_bin


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


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--teacher_ckpt", type=str, default="out/main_mlp_mult_32_K_8/ckpt_iter300000.pt")
    ap.add_argument("--student_ckpt", type=str, default="out/distill_suite/target_models/results_real_h10240_knone/h1024_knone_relu1_final.pt")
    ap.add_argument("--data_dir", type=str, default="data/tinystories")
    ap.add_argument("--layer_idx", type=int, default=2)
    ap.add_argument("--split", choices=["train", "val"], default="val")
    ap.add_argument("--num_batches", type=int, default=200)
    ap.add_argument("--batch_size", type=int, default=16)
    ap.add_argument("--block_size", type=int, default=256)
    ap.add_argument("--min_teacher_active_count", type=int, default=50)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--device_act", type=str, default="cpu")
    ap.add_argument("--device_pairs", type=str, default="cpu")
    ap.add_argument("--pair_chunk", type=int, default=32)
    ap.add_argument("--pmi_group_sample_size", type=int, default=5000)
    ap.add_argument("--hist_bins", type=int, default=75)
    ap.add_argument("--out_png", type=str, default="figures/pmi_groups_tvd_hist.png")
    args = ap.parse_args()

    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    t_cfg = TransformerConfig(**t_ckpt["model_cfg"])
    teacher_hidden = int(t_cfg.n_embd * t_cfg.mlp_mult)
    if t_cfg.mlp_topk is None:
        raise SystemExit("Teacher must use TopK (mlp_topk is None).")
    teacher_k = int(t_cfg.mlp_topk)
    teacher_use_relu = bool(getattr(t_cfg, "use_relu_before_topk", True))

    teacher = SparseTransformerLM(t_cfg).to(args.device_act)
    teacher.load_state_dict(t_ckpt["model"], strict=False)
    teacher.eval()
    blk = teacher.blocks[args.layer_idx]

    s_ckpt = torch.load(args.student_ckpt, map_location="cpu")
    student_hidden = int(s_ckpt["hidden_size"])
    s_topk = s_ckpt.get("mlp_topk", None)
    s_relu = bool(s_ckpt.get("use_relu_before_topk", True))
    student = StudentMLP(
        d_in=t_cfg.n_embd,
        d_hidden=student_hidden,
        d_out=t_cfg.n_embd,
        mlp_topk=s_topk,
        use_relu_before_topk=s_relu,
    ).to(args.device_act)
    student.load_state_dict(s_ckpt["state_dict"], strict=True)
    student.eval()

    student_value_norms = student.fc2.weight.detach().norm(dim=0).to(torch.float32)

    split_path = "train.bin" if args.split == "train" else "val.bin"
    data = load_bin(os.path.join(args.data_dir, split_path))
    rng_cpu = torch.Generator(device="cpu").manual_seed(int(args.seed))

    M = teacher_hidden
    Hs = student_hidden
    K = int(teacher_k)
    act_counts = torch.zeros((M,), dtype=torch.float32, device="cpu")
    co_counts = torch.zeros((M, M), dtype=torch.float32, device="cpu")
    numer = torch.zeros((M, Hs), dtype=torch.float32, device="cpu")
    total_tokens = 0

    cache: dict[str, torch.Tensor] = {}

    def ln2_hook(_m, _i, out):
        cache["u"] = out.detach()

    def fc1_hook(_m, _i, out):
        cache["h1"] = out.detach()

    h_ln2 = blk.ln2.register_forward_hook(ln2_hook)
    h_fc1 = blk.mlp.fc1.register_forward_hook(fc1_hook)

    for step in range(int(args.num_batches)):
        xb, _ = get_batch(data, args.batch_size, args.block_size, args.device_act, rng_cpu)
        _logits, _loss = teacher(xb, None)

        u = cache["u"]
        h1_pre = cache["h1"]
        u_flat = u.reshape(-1, u.size(-1))
        h1_flat = h1_pre.reshape(-1, h1_pre.size(-1))
        if teacher_use_relu:
            h1_flat = torch.relu(h1_flat)
        topk_vals, topk_idx = torch.topk(h1_flat, k=K, dim=1, sorted=False)
        active_mask = topk_vals != 0

        s_act = student.hidden_activations(u_flat).to(torch.float32)
        s_weighted = s_act * student_value_norms.unsqueeze(0)
        N = topk_idx.size(0)
        total_tokens += int(N)

        for p in range(K):
            m_p = active_mask[:, p]
            if not bool(m_p.any()):
                continue
            neuron_p = topk_idx[:, p][m_p]
            act_counts += torch.bincount(neuron_p.to("cpu"), minlength=M).to(torch.float32)
            numer.index_add_(0, neuron_p.to("cpu"), s_weighted[m_p].to("cpu"))

        a = topk_idx.unsqueeze(2).expand(N, K, K)
        b = topk_idx.unsqueeze(1).expand(N, K, K)
        m = active_mask.unsqueeze(2) & active_mask.unsqueeze(1)
        a_flat = a[m].to("cpu")
        b_flat = b[m].to("cpu")
        if a_flat.numel() > 0:
            co_counts[a_flat, b_flat] += 1.0

        if (step + 1) % 50 == 0:
            print(f"[collect] {step+1}/{args.num_batches} total_tokens={total_tokens}")

    h_ln2.remove()
    h_fc1.remove()

    valid = act_counts >= float(args.min_teacher_active_count)
    valid_idx = torch.nonzero(valid, as_tuple=False).squeeze(1)
    Mv = int(valid_idx.numel())
    if Mv < 2:
        raise SystemExit("Too few valid teacher neurons after filtering.")

    act_counts_v = act_counts[valid_idx]
    co_counts_v = co_counts.index_select(0, valid_idx).index_select(1, valid_idx)
    numer_v = numer.index_select(0, valid_idx)
    dist_v = numer_v / (numer_v.sum(dim=1, keepdim=True) + 1e-12)

    dist_pairs = dist_v.to(args.device_pairs, dtype=torch.float32)
    act_pairs = (act_counts_v / float(total_tokens)).to(args.device_pairs)
    co_pairs = (co_counts_v / float(total_tokens)).to(args.device_pairs)

    eps = 1e-12
    pair_chunk = int(args.pair_chunk)

    group_labels = ["pmi>5", "5>=pmi>0", "0>pmi"]
    group_tvd: dict[str, list[float]] = {k: [] for k in group_labels}
    legend_labels = {
        "pmi>5": r"$\mathrm{PMI} > 5$",
        "5>=pmi>0": r"$5 \geq \mathrm{PMI} > 0$",
        "0>pmi": r"$0 > \mathrm{PMI}$",
    }

    for i0 in range(0, Mv, pair_chunk):
        i1 = min(Mv, i0 + pair_chunk)
        dist_i = dist_pairs[i0:i1]
        act_i = act_pairs[i0:i1]
        for j0 in range(i0, Mv, pair_chunk):
            j1 = min(Mv, j0 + pair_chunk)
            dist_j = dist_pairs[j0:j1]
            act_j = act_pairs[j0:j1]
            co_block = co_pairs[i0:i1, j0:j1].to(torch.float32)

            tvd_block = 0.5 * (dist_i[:, None, :] - dist_j[None, :, :]).abs().sum(dim=-1)
            p_i = act_i[:, None].to(torch.float32)
            p_j = act_j[None, :].to(torch.float32)
            pmi = torch.log((co_block + eps) / (p_i * p_j).clamp_min(eps))

            if j0 == i0:
                mask = torch.triu(
                    torch.ones((dist_i.size(0), dist_j.size(0)), dtype=torch.bool, device=tvd_block.device),
                    diagonal=1,
                )
                tvd_np = tvd_block[mask].detach().cpu().numpy().ravel()
                pmi_np = pmi[mask].detach().cpu().numpy().ravel()
            else:
                tvd_np = tvd_block.detach().cpu().numpy().ravel()
                pmi_np = pmi.detach().cpu().numpy().ravel()

            m0 = pmi_np > 5.0
            if np.any(m0):
                group_tvd["pmi>5"].extend(tvd_np[m0].tolist())
            m1 = (pmi_np <= 5.0) & (pmi_np > 0.0)
            if np.any(m1):
                group_tvd["5>=pmi>0"].extend(tvd_np[m1].tolist())
            m2 = pmi_np < 0.0
            if np.any(m2):
                group_tvd["0>pmi"].extend(tvd_np[m2].tolist())

    os.makedirs(os.path.dirname(args.out_png) or ".", exist_ok=True)
    rng = np.random.default_rng(int(args.seed))
    colors = {
        "pmi>5": "#5f5a4a",
        "5>=pmi>0": "#8a3434",
        "0>pmi": "#dca35b",
    }

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    for label in group_labels:
        vals = np.asarray(group_tvd[label], dtype=np.float32)
        if vals.size == 0:
            continue
        if vals.size > int(args.pmi_group_sample_size):
            idx = rng.choice(vals.size, size=int(args.pmi_group_sample_size), replace=False)
            vals = vals[idx]
        ax.hist(
            vals,
            bins=int(args.hist_bins),
            alpha=0.65,
            color=colors[label],
            label=legend_labels[label],
            density=True,
        )
        # Emphasize only the top contour (not individual bar borders).
        ax.hist(
            vals,
            bins=int(args.hist_bins),
            histtype="step",
            color="black",
            linewidth=1.2,
            density=True,
        )
        ax.axvline(vals.mean(), color=colors[label], linestyle="--", linewidth=1.5, alpha=0.95)

    ax.set_xlabel("TVD")
    ax.set_ylabel("Density")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(args.out_png, dpi=180, bbox_inches="tight")
    plt.close(fig)
    print(f"[save] {args.out_png}")


if __name__ == "__main__":
    main()

