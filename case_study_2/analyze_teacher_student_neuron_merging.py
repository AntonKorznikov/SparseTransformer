from __future__ import annotations

import argparse
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


def make_random_teacher_params(d_in: int, hidden: int, k: int, seed: int, device: str):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(seed))
    # Xavier-like scale
    w = torch.randn((hidden, d_in), generator=g, dtype=torch.float32) / np.sqrt(float(d_in))
    b = torch.zeros((hidden,), dtype=torch.float32)
    return {
        "w": w.to(device),
        "b": b.to(device),
        "k": int(k),
        "use_relu": True,
    }


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
    """Set a single shared bias value so random student matches target mean #active neurons/token."""
    if student_rand.mlp_topk is not None:
        # With TopK, active count is largely constrained by K. Keep default bias.
        return float(student_rand.fc1.bias.mean().item())

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed) + 12345)
    blk = teacher.blocks[layer_idx] if teacher is not None else None
    z_all = []
    for _ in range(calib_batches):
        if use_gaussian_input:
            if d_in is None:
                raise SystemExit("d_in must be provided when use_gaussian_input=True")
            n = batch_size * block_size
            u = torch.randn((n, d_in), generator=rng, device="cpu", dtype=torch.float32) * float(gaussian_sigma)
            u = u.to(device)
        else:
            if teacher is None or blk is None:
                raise SystemExit("teacher must be provided when use_gaussian_input=False")
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
    target_prob = float(target_active_count / max(student_rand.fc1.out_features, 1))
    target_prob = float(np.clip(target_prob, 1e-6, 1 - 1e-6))
    q = np.quantile(z_flat, 1.0 - target_prob)  # want P(z + b > 0) = target_prob
    b = float(-q)
    student_rand.fc1.bias.data.fill_(b)
    return b


@torch.no_grad()
def analyze_student_conditionals(
    student: StudentMLP,
    teacher: SparseTransformerLM | None,
    layer_idx: int,
    teacher_idx: np.ndarray,
    data,
    num_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    seed: int,
    use_gaussian_input: bool = False,
    gaussian_sigma: float = 1.0,
    random_teacher_params: dict | None = None,
    d_in: int | None = None,
):
    s_hidden = int(student.fc1.out_features)
    teacher_hidden = int(teacher.cfg.n_embd * teacher.cfg.mlp_mult) if teacher is not None else int(random_teacher_params["w"].shape[0])
    numer = torch.zeros((len(teacher_idx), s_hidden), dtype=torch.float64, device="cpu")
    denom = torch.zeros((len(teacher_idx),), dtype=torch.float64, device="cpu")
    teacher_counts_all = torch.zeros((teacher_hidden,), dtype=torch.float64, device="cpu")
    s_weighted_sum = torch.zeros((s_hidden,), dtype=torch.float64, device="cpu")
    total_tokens = 0
    active_sum = 0.0

    student_value_norms = student.fc2.weight.detach().norm(dim=0).to(torch.float64).cpu()

    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(seed))
    blk = teacher.blocks[layer_idx] if teacher is not None else None
    for _ in range(num_batches):
        if use_gaussian_input:
            if d_in is None:
                raise SystemExit("d_in must be provided when use_gaussian_input=True")
            n = batch_size * block_size
            u = torch.randn((n, d_in), generator=rng, device="cpu", dtype=torch.float32) * float(gaussian_sigma)
            u = u.to(device)
            if random_teacher_params is None:
                if blk is None:
                    raise SystemExit("teacher or random_teacher_params must be provided")
                t_act = blk.mlp.act(blk.mlp.fc1(u))
            else:
                t_pre = torch.addmm(random_teacher_params["b"], u, random_teacher_params["w"].t())
                t_act = _topk_act(
                    t_pre,
                    k=random_teacher_params["k"],
                    use_relu=bool(random_teacher_params.get("use_relu", True)),
                )
        else:
            if teacher is None or blk is None:
                raise SystemExit("teacher must be provided when use_gaussian_input=False")
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
        s_act = student.hidden_activations(u)

        t_sel_bin = (t_act[:, teacher_idx] != 0).to(torch.float64).cpu()
        s_cpu = s_act.to(torch.float64).cpu()
        s_weighted = s_cpu * student_value_norms.unsqueeze(0)

        numer.add_(t_sel_bin.t().mm(s_weighted))
        denom.add_(t_sel_bin.sum(dim=0))
        teacher_counts_all.add_((t_act != 0).to(torch.float64).cpu().sum(dim=0))
        s_weighted_sum.add_(s_weighted.sum(dim=0))
        total_tokens += int(s_weighted.shape[0])
        active_sum += float((s_cpu != 0).sum(dim=1).sum().item())
    cond = numer / denom.clamp_min(1.0).unsqueeze(1)
    s_weighted_mean = s_weighted_sum / max(total_tokens, 1)
    cond_bc = (cond - s_weighted_mean.unsqueeze(0)).clamp_min(0.0)

    cond_sorted, _ = torch.sort(cond, dim=1, descending=True)
    cond_bc_sorted, _ = torch.sort(cond_bc, dim=1, descending=True)
    f = torch.cumsum(cond_sorted, dim=1) / cond_sorted.sum(dim=1, keepdim=True).clamp_min(1e-12)
    f_bc = torch.cumsum(cond_bc_sorted, dim=1) / cond_bc_sorted.sum(dim=1, keepdim=True).clamp_min(1e-12)
    mean_active_count = active_sum / max(total_tokens, 1)

    return {
        "denom": denom,
        "student_value_norms": student_value_norms,
        "weighted_unconditional_mean": s_weighted_mean,
        "cond": cond,
        "cond_sorted": cond_sorted,
        "cond_bc": cond_bc,
        "cond_bc_sorted": cond_bc_sorted,
        "f": f,
        "f_bc": f_bc,
        "teacher_counts_all": teacher_counts_all,
        "teacher_total_tokens": int(total_tokens),
        "mean_active_count": float(mean_active_count),
    }


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


@torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--teacher_ckpt", type=str, default="out/main_mlp_mult_32_K_8/ckpt_iter300000.pt")
    p.add_argument("--student_ckpt", type=str, default="out/distill_suite/target_models/results_real_h10240_knone/h1024_knone_relu1_final.pt")
    p.add_argument("--data_dir", type=str, default="data/tinystories")
    p.add_argument("--layer_idx", type=int, default=2)
    p.add_argument("--split", type=str, choices=["train", "val"], default="val")
    p.add_argument("--num_batches", type=int, default=200)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--block_size", type=int, default=256)
    p.add_argument("--num_teacher_neurons", type=int, default=200)
    p.add_argument("--min_denom_count", type=int, default=50)
    p.add_argument("--calib_batches", type=int, default=100)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--auto_cpu_fallback",
        action="store_true",
        default=True,
        help="automatically switch to CPU when CUDA memory is insufficient or OOM occurs",
    )
    p.add_argument("--out_pt", type=str, default="figures/teacher_student_conditional_analysis.pt")
    p.add_argument("--out_png", type=str, default="figures/teacher_student_conditional_concentration.png")
    p.add_argument(
        "--out_png_linear",
        type=str,
        default="figures/teacher_student_conditional_concentration_linear.png",
        help="second plot with linear k-axis",
    )
    p.add_argument(
        "--out_png_top20",
        type=str,
        default="figures/teacher_top20_individual_f_curves.png",
        help="plot with individual f(k) curves for 20 most active teacher neurons (real teacher)",
    )
    p.add_argument(
        "--out_png_top10",
        type=str,
        default="figures/teacher_top10_lineplots.png",
        help="single plot with 10 individual neuron lineplots (percentile style)",
    )
    args = p.parse_args()
    for out_path in [args.out_pt, args.out_png, args.out_png_linear, args.out_png_top20, args.out_png_top10]:
        os.makedirs(os.path.dirname(out_path) or ".", exist_ok=True)

    device = args.device
    if device.startswith("cuda") and args.auto_cpu_fallback:
        try:
            free_mem, total_mem = torch.cuda.mem_get_info()
            # If less than 2 GiB free, CPU is safer in shared GPU environments.
            if free_mem < 2 * (1024 ** 3):
                print(
                    f"[warn] Low free CUDA memory ({free_mem / (1024 ** 3):.2f} GiB / "
                    f"{total_mem / (1024 ** 3):.2f} GiB). Falling back to CPU."
                )
                device = "cpu"
        except Exception:
            pass

    # Load teacher
    t_ckpt = torch.load(args.teacher_ckpt, map_location="cpu")
    t_cfg = TransformerConfig(**t_ckpt["model_cfg"])
    teacher = SparseTransformerLM(t_cfg).to(device)
    teacher.load_state_dict(t_ckpt["model"], strict=False)
    teacher.eval()
    blk = teacher.blocks[args.layer_idx]

    # Load student
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

    # Data
    split_path = "train.bin" if args.split == "train" else "val.bin"
    data = load_bin(os.path.join(args.data_dir, split_path))
    rng = torch.Generator(device="cpu")
    rng.manual_seed(int(args.seed))

    teacher_hidden = int(t_cfg.n_embd * t_cfg.mlp_mult)
    if args.num_teacher_neurons > teacher_hidden:
        raise SystemExit(f"num_teacher_neurons={args.num_teacher_neurons} > teacher_hidden={teacher_hidden}")

    # Sample teacher neuron IDs
    rs = np.random.default_rng(int(args.seed))
    teacher_idx = np.sort(rs.choice(teacher_hidden, size=args.num_teacher_neurons, replace=False))

    try:
        real_res = analyze_student_conditionals(
            student=student,
            teacher=teacher,
            layer_idx=args.layer_idx,
            teacher_idx=teacher_idx,
            data=data,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
            seed=args.seed,
            d_in=t_cfg.n_embd,
        )
    except torch.OutOfMemoryError:
        if (not args.auto_cpu_fallback) or device == "cpu":
            raise
        print("[warn] CUDA OOM during analysis. Retrying on CPU...")
        device = "cpu"
        teacher = teacher.to(device)
        student = student.to(device)
        real_res = analyze_student_conditionals(
            student=student,
            teacher=teacher,
            layer_idx=args.layer_idx,
            teacher_idx=teacher_idx,
            data=data,
            num_batches=args.num_batches,
            batch_size=args.batch_size,
            block_size=args.block_size,
            device=device,
            seed=args.seed,
            d_in=t_cfg.n_embd,
        )
    valid_mask_real = real_res["denom"] >= float(args.min_denom_count)

    ln2_sigma = estimate_ln2_sigma(
        teacher=teacher,
        layer_idx=args.layer_idx,
        data=data,
        num_batches=max(20, min(args.num_batches, args.calib_batches)),
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
        seed=args.seed,
    )

    # Sanity-check baseline: random student with bias calibrated to match mean #active.
    student_rand = StudentMLP(
        d_in=t_cfg.n_embd,
        d_hidden=s_hidden,
        d_out=t_cfg.n_embd,
        mlp_topk=s_topk,
        use_relu_before_topk=s_relu,
    ).to(device)
    torch.manual_seed(int(args.seed) + 777)
    nn.init.xavier_uniform_(student_rand.fc1.weight)
    nn.init.xavier_uniform_(student_rand.fc2.weight)
    nn.init.zeros_(student_rand.fc1.bias)
    nn.init.zeros_(student_rand.fc2.bias)
    bias_shift = calibrate_random_student_bias(
        student_rand=student_rand,
        teacher=None,
        layer_idx=args.layer_idx,
        data=data,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
        seed=args.seed,
        target_active_count=real_res["mean_active_count"],
        calib_batches=args.calib_batches,
        use_gaussian_input=True,
        gaussian_sigma=ln2_sigma,
        d_in=t_cfg.n_embd,
    )

    # Fully random world: random teacher MLP (K=8), random student MLP, random isotropic inputs.
    random_teacher = make_random_teacher_params(
        d_in=t_cfg.n_embd,
        hidden=teacher_hidden,
        k=8,
        seed=args.seed + 2026,
        device=device,
    )
    teacher_idx_random = np.sort(rs.choice(teacher_hidden, size=args.num_teacher_neurons, replace=False))
    rand_res = analyze_student_conditionals(
        student=student_rand,
        teacher=None,
        layer_idx=args.layer_idx,
        teacher_idx=teacher_idx_random,
        data=data,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
        seed=args.seed + 999,
        use_gaussian_input=True,
        gaussian_sigma=ln2_sigma,
        random_teacher_params=random_teacher,
        d_in=t_cfg.n_embd,
    )
    valid_mask_random = rand_res["denom"] >= float(args.min_denom_count)

    # Aggregate across selected teacher neurons
    f_valid = real_res["f"][valid_mask_real]
    f_bc_valid = real_res["f_bc"][valid_mask_real]
    f_rand_valid = rand_res["f"][valid_mask_random]
    f_rand_bc_valid = rand_res["f_bc"][valid_mask_random]
    mean_curve = f_valid.mean(dim=0).numpy()
    p10_curve = torch.quantile(f_valid, 0.10, dim=0).numpy()
    p90_curve = torch.quantile(f_valid, 0.90, dim=0).numpy()
    mean_curve_bc = f_bc_valid.mean(dim=0).numpy()
    p10_curve_bc = torch.quantile(f_bc_valid, 0.10, dim=0).numpy()
    p90_curve_bc = torch.quantile(f_bc_valid, 0.90, dim=0).numpy()
    mean_curve_rand = f_rand_valid.mean(dim=0).numpy()
    mean_curve_rand_bc = f_rand_bc_valid.mean(dim=0).numpy()
    ks = np.arange(1, s_hidden + 1)
    y_thresholds = np.asarray([i / 100.0 for i in range(1, 11)] + [i / 100.0 for i in range(20, 101, 10)], dtype=np.float32)

    def k_at_thresholds(curve: np.ndarray, thresholds: np.ndarray) -> np.ndarray:
        out = np.ones_like(thresholds, dtype=np.float32)
        for i, thr in enumerate(thresholds):
            idx = int(np.searchsorted(curve, thr, side="left"))
            out[i] = float(min(max(idx + 1, 1), len(curve)))
        return out

    def plot_raw_curves(xscale: str, out_path: str):
        plt.figure(figsize=(9, 6))
        x_real = k_at_thresholds(mean_curve, y_thresholds)
        x_p10 = k_at_thresholds(p10_curve, y_thresholds)
        x_p90 = k_at_thresholds(p90_curve, y_thresholds)
        x_rand = k_at_thresholds(mean_curve_rand, y_thresholds)
        real_color = "#8a3434"

        plt.plot(x_real, y_thresholds, color=real_color, lw=2.3, marker="o", markersize=5.0, label="real student raw mean")
        plt.fill_betweenx(y_thresholds, x_p10, x_p90, color=real_color, alpha=0.20, label="real student raw 10-90 pct")
        plt.plot(x_rand, y_thresholds, color="gray", lw=2, ls="--", marker="o", markersize=4.0, label="random baseline raw mean")
        if xscale == "log":
            plt.xscale("log")
        plt.ylim(0.0, 1.01)
        plt.xlabel("k (top student neurons kept)")
        plt.ylabel("f(k) = cumulative mass fraction")
        plt.title("Concentration of weighted E[s_i | teacher neuron active] (raw only)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_path, dpi=160)
        plt.close()

    # Plot 1: log-k
    plot_raw_curves("log", args.out_png)
    # Plot 2: linear-k
    plot_raw_curves("linear", args.out_png_linear)

    # Additional analysis: top-20 most frequently active teacher neurons (real teacher),
    # and individual raw f(k) curves for each of them.
    teacher_freq_all = real_res["teacher_counts_all"] / max(real_res["teacher_total_tokens"], 1)
    top20_idx = torch.topk(teacher_freq_all, k=min(20, teacher_hidden)).indices.cpu().numpy()
    top20_res = analyze_student_conditionals(
        student=student,
        teacher=teacher,
        layer_idx=args.layer_idx,
        teacher_idx=top20_idx,
        data=data,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
        seed=args.seed + 4242,
        d_in=t_cfg.n_embd,
    )
    f_top20 = top20_res["f"].cpu().numpy()  # [20, Hs]
    top20_freq = teacher_freq_all[top20_idx].cpu().numpy()

    fig, axes = plt.subplots(4, 5, figsize=(20, 14), sharex=True, sharey=True)
    axes = axes.flatten()
    for m in range(len(top20_idx)):
        ax = axes[m]
        ax.plot(ks, f_top20[m], color="steelblue", lw=1.8)
        ax.set_xscale("log")
        ax.set_ylim(0.0, 1.01)
        ax.set_title(f"t={int(top20_idx[m])}, p={top20_freq[m]:.3e}", fontsize=9)
        ax.grid(True, alpha=0.25)
    for m in range(len(top20_idx), len(axes)):
        axes[m].axis("off")
    fig.suptitle("Individual f(k) for top-20 most active teacher neurons (real teacher)", fontsize=13)
    fig.text(0.5, 0.04, "k (top student neurons kept, log scale)", ha="center")
    fig.text(0.04, 0.5, "f(k)", va="center", rotation="vertical")
    plt.tight_layout(rect=[0.05, 0.05, 1, 0.95])
    plt.savefig(args.out_png_top20, dpi=170)
    plt.close(fig)

    # Additional single-panel plot with 10 most active teacher neurons, same percentile-point style.
    top10_idx = torch.topk(teacher_freq_all, k=min(10, teacher_hidden)).indices.cpu().numpy()
    top10_res = analyze_student_conditionals(
        student=student,
        teacher=teacher,
        layer_idx=args.layer_idx,
        teacher_idx=top10_idx,
        data=data,
        num_batches=args.num_batches,
        batch_size=args.batch_size,
        block_size=args.block_size,
        device=device,
        seed=args.seed + 9090,
        d_in=t_cfg.n_embd,
    )
    f_top10 = top10_res["f"].cpu().numpy()
    top10_freq = teacher_freq_all[top10_idx].cpu().numpy()

    fig, ax = plt.subplots(1, 1, figsize=(8, 8))
    palette = [
        "#5f5a4a", "#6f6754", "#7f745e", "#8f8068", "#9f8d72",
        "#8a3434", "#984444", "#a65454", "#b46464", "#c27474",
    ]
    for m in range(len(top10_idx)):
        x_vals = k_at_thresholds(f_top10[m], y_thresholds)
        ax.plot(
            x_vals,
            y_thresholds,
            lw=2.5,
            color=palette[m % len(palette)],
            marker="o",
            markersize=8.0,
            markeredgewidth=0.0,
            alpha=0.95,
            label=f"teacher_neuron = {int(top10_idx[m])}",
        )
    x_rand_top10 = k_at_thresholds(mean_curve_rand, y_thresholds)
    ax.plot(
        x_rand_top10,
        y_thresholds,
        color="gray",
        lw=2.0,
        ls="--",
        alpha=0.95,
        label="random baseline",
    )
    ax.set_xscale("log")
    ax.set_ylim(0.0, 1.01)
    ax.set_xlabel("k (top student neurons kept)")
    ax.set_ylabel("f(k) = cumulative mass fraction")
    ax.grid(True, alpha=0.25)
    ax.legend(loc="lower right", fontsize=12, ncol=1, framealpha=0.9)
    fig.tight_layout()
    plt.savefig(args.out_png_top10, dpi=170, bbox_inches="tight")
    plt.close(fig)

    # Save all tensors
    out = {
        "teacher_indices_real": torch.from_numpy(teacher_idx.astype(np.int64)),
        "teacher_indices_random": torch.from_numpy(teacher_idx_random.astype(np.int64)),
        "denom_counts_real": real_res["denom"].to(torch.float32),
        "denom_counts_random": rand_res["denom"].to(torch.float32),
        "min_denom_count": int(args.min_denom_count),
        "real_student_value_norms": real_res["student_value_norms"].to(torch.float32),
        "random_student_value_norms": rand_res["student_value_norms"].to(torch.float32),
        "weighted_unconditional_mean_real": real_res["weighted_unconditional_mean"].to(torch.float32),
        "weighted_unconditional_mean_random": rand_res["weighted_unconditional_mean"].to(torch.float32),
        "conditional_expectation_weighted_real": real_res["cond"].to(torch.float32),
        "conditional_expectation_weighted_sorted_real": real_res["cond_sorted"].to(torch.float32),
        "concentration_curves_weighted_real": real_res["f"].to(torch.float32),
        "conditional_expectation_weighted_baseline_corrected_real": real_res["cond_bc"].to(torch.float32),
        "conditional_expectation_weighted_baseline_corrected_sorted_real": real_res["cond_bc_sorted"].to(torch.float32),
        "concentration_curves_weighted_baseline_corrected_real": real_res["f_bc"].to(torch.float32),
        "conditional_expectation_weighted_random": rand_res["cond"].to(torch.float32),
        "conditional_expectation_weighted_sorted_random": rand_res["cond_sorted"].to(torch.float32),
        "concentration_curves_weighted_random": rand_res["f"].to(torch.float32),
        "conditional_expectation_weighted_baseline_corrected_random": rand_res["cond_bc"].to(torch.float32),
        "conditional_expectation_weighted_baseline_corrected_sorted_random": rand_res["cond_bc_sorted"].to(torch.float32),
        "concentration_curves_weighted_baseline_corrected_random": rand_res["f_bc"].to(torch.float32),
        "valid_teacher_mask_real": valid_mask_real,
        "valid_teacher_mask_random": valid_mask_random,
        "mean_curve_weighted": torch.from_numpy(mean_curve.astype(np.float32)),
        "p10_curve_weighted": torch.from_numpy(p10_curve.astype(np.float32)),
        "p90_curve_weighted": torch.from_numpy(p90_curve.astype(np.float32)),
        "mean_curve_weighted_baseline_corrected": torch.from_numpy(mean_curve_bc.astype(np.float32)),
        "p10_curve_weighted_baseline_corrected": torch.from_numpy(p10_curve_bc.astype(np.float32)),
        "p90_curve_weighted_baseline_corrected": torch.from_numpy(p90_curve_bc.astype(np.float32)),
        "mean_curve_weighted_random": torch.from_numpy(mean_curve_rand.astype(np.float32)),
        "mean_curve_weighted_baseline_corrected_random": torch.from_numpy(mean_curve_rand_bc.astype(np.float32)),
        "real_student_mean_active_count": float(real_res["mean_active_count"]),
        "random_student_mean_active_count": float(rand_res["mean_active_count"]),
        "random_student_bias_shift": float(bias_shift),
        "sanity_gaussian_sigma": float(ln2_sigma),
        "teacher_freq_all_real": teacher_freq_all.to(torch.float32),
        "top20_teacher_indices_real": torch.from_numpy(top20_idx.astype(np.int64)),
        "top20_teacher_freq_real": torch.from_numpy(top20_freq.astype(np.float32)),
        "top20_individual_f_curves_real": torch.from_numpy(f_top20.astype(np.float32)),
        "top10_teacher_indices_real": torch.from_numpy(top10_idx.astype(np.int64)),
        "top10_teacher_freq_real": torch.from_numpy(top10_freq.astype(np.float32)),
        "top10_individual_f_curves_real": torch.from_numpy(f_top10.astype(np.float32)),
        "student_hidden": int(s_hidden),
        "teacher_hidden": int(teacher_hidden),
        "num_batches": int(args.num_batches),
        "batch_size": int(args.batch_size),
        "block_size": int(args.block_size),
        "split": args.split,
        "layer_idx": int(args.layer_idx),
    }
    torch.save(out, args.out_pt)

    print(f"Saved analysis tensors to: {args.out_pt}")
    print(f"Saved concentration plot (log-k) to: {args.out_png}")
    print(f"Saved concentration plot (linear-k) to: {args.out_png_linear}")
    print(f"Saved top-20 individual curves plot to: {args.out_png_top20}")
    print(f"Saved top-10 lineplot figure to: {args.out_png_top10}")
    print(
        f"Valid sampled teacher neurons (real teacher, denom >= {args.min_denom_count}): "
        f"{int(valid_mask_real.sum().item())}/{args.num_teacher_neurons}"
    )
    print(
        f"Valid sampled teacher neurons (random teacher, denom >= {args.min_denom_count}): "
        f"{int(valid_mask_random.sum().item())}/{args.num_teacher_neurons}"
    )
    print(
        f"Mean active neurons/token: real={real_res['mean_active_count']:.3f}, "
        f"random={rand_res['mean_active_count']:.3f}, bias_shift={bias_shift:.5f}"
    )


if __name__ == "__main__":
    main()
