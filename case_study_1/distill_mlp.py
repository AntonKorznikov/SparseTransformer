from __future__ import annotations

import os
import sys

import argparse
import pickle
from dataclasses import dataclass
from types import SimpleNamespace
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter

from model_transformer import SparseTransformerLM, TransformerConfig
from utils.common import (
    device_from_cfg,
    get_batch,
    load_bin,
    load_config,
    load_meta,
)

HF_REPO_ID = "AntonKorznikov/SparseTransformer"
HF_CKPT_REL_CANDIDATES = [
    "main_mlp_mult_32_K_8/ckpt_iter300000.pt",
    "out/main_mlp_mult_32_K_8/ckpt_iter300000.pt",
]


@dataclass
class TeacherDump:
    u_train: str
    y_train: str
    u_val: str
    y_val: str
    meta: dict


def _topk_sparsify(x: torch.Tensor, k: int | None, use_relu: bool) -> torch.Tensor:
    """Per-sample TopK: keep top-k activations per row."""
    if use_relu:
        x = torch.relu(x)
    if k is None:
        return x
    vals, idx = torch.topk(x, k=int(k), dim=-1, sorted=False)
    out = torch.zeros_like(x)
    batch_idx = torch.arange(x.shape[0], device=x.device, dtype=torch.long)[:, None]
    out[batch_idx, idx] = vals
    return out


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
        self.use_relu = use_relu_before_topk
        self.mlp_topk = None if mlp_topk is None else int(mlp_topk)
        self.fc2 = nn.Linear(d_hidden, d_out, bias=True)

    def sparsify(self, x: torch.Tensor) -> torch.Tensor:
        return _topk_sparsify(x, self.mlp_topk, self.use_relu)

    def forward_hidden(self, u: torch.Tensor) -> torch.Tensor:
        return self.sparsify(self.fc1(u))

    def forward(self, u: torch.Tensor) -> torch.Tensor:
        return self.fc2(self.forward_hidden(u))


class TopKSAE(nn.Module):
    def __init__(
        self,
        d_in: int,
        d_hidden: int,
        k: int | None,
        use_relu_before_topk: bool = True,
        use_pre_bias: bool = True,
    ):
        super().__init__()
        self.encoder = nn.Linear(d_in, d_hidden, bias=False)
        self.decoder = nn.Linear(d_hidden, d_in, bias=False)
        self.k = None if k is None else int(k)
        self.use_relu = use_relu_before_topk
        self.use_pre_bias = bool(use_pre_bias)
        if self.use_pre_bias:
            self.pre_bias = nn.Parameter(torch.zeros(d_in))
        else:
            self.register_buffer("pre_bias", torch.zeros(d_in), persistent=False)

        # Always use tied initialization, then enforce unit-norm decoder columns.
        with torch.no_grad():
            self.decoder.weight.copy_(self.encoder.weight.t())
            self.unit_norm_decoder_()

    def sparsify(self, z: torch.Tensor) -> torch.Tensor:
        return _topk_sparsify(z, self.k, self.use_relu)

    def forward_codes(self, x: torch.Tensor) -> torch.Tensor:
        return self.sparsify(self.encoder(x - self.pre_bias))

    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        z_sparse = self.forward_codes(x)
        x_hat = self.decoder(z_sparse) + self.pre_bias
        return x_hat, z_sparse

    def unit_norm_decoder_(self) -> None:
        with torch.no_grad():
            w = self.decoder.weight
            w.div_(w.norm(dim=0, keepdim=True).clamp_min(1e-12))

    def unit_norm_decoder_grad_adjustment_(self) -> None:
        if self.decoder.weight.grad is None:
            return
        with torch.no_grad():
            w = self.decoder.weight
            g = self.decoder.weight.grad
            # Project out the radial component so gradient updates preserve column directions.
            g.sub_(w * (w * g).sum(dim=0, keepdim=True))


def save_teacher_dump(
    out_dir: str,
    u_train: np.ndarray,
    y_train: np.ndarray,
    u_val: np.ndarray,
    y_val: np.ndarray,
    meta: dict,
):
    os.makedirs(out_dir, exist_ok=True)
    u_train_path = os.path.join(out_dir, "u_train.npy")
    y_train_path = os.path.join(out_dir, "y_train.npy")
    u_val_path = os.path.join(out_dir, "u_val.npy")
    y_val_path = os.path.join(out_dir, "y_val.npy")
    np.save(u_train_path, u_train)
    np.save(y_train_path, y_train)
    np.save(u_val_path, u_val)
    np.save(y_val_path, y_val)
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)
    return TeacherDump(u_train_path, y_train_path, u_val_path, y_val_path, meta)


def load_teacher_dump(teacher_dir: str):
    with open(os.path.join(teacher_dir, "meta.pkl"), "rb") as f:
        meta = pickle.load(f)
    u_train = np.load(os.path.join(teacher_dir, "u_train.npy"), mmap_mode="r")
    y_train = np.load(os.path.join(teacher_dir, "y_train.npy"), mmap_mode="r")
    u_val = np.load(os.path.join(teacher_dir, "u_val.npy"), mmap_mode="r")
    y_val = np.load(os.path.join(teacher_dir, "y_val.npy"), mmap_mode="r")
    return u_train, y_train, u_val, y_val, meta


def _sample_random_mlp_params_like_teacher(
    fc1_w: torch.Tensor,
    fc1_b: torch.Tensor,
    fc2_w: torch.Tensor,
    fc2_b: torch.Tensor,
    random_seed: int,
):
    g = torch.Generator(device="cpu")
    g.manual_seed(int(random_seed))

    # Match the average per-neuron projection norms from the real teacher.
    up_target = float(fc1_w.norm(dim=1).mean().item())       # rows of fc1
    down_target = float(fc2_w.norm(dim=0).mean().item())     # columns of fc2

    w1 = torch.randn(fc1_w.shape, generator=g, dtype=fc1_w.dtype)
    w2 = torch.randn(fc2_w.shape, generator=g, dtype=fc2_w.dtype)

    up_rand = float(w1.norm(dim=1).mean().item())
    down_rand = float(w2.norm(dim=0).mean().item())
    if up_rand > 0:
        w1 = w1 * (up_target / up_rand)
    if down_rand > 0:
        w2 = w2 * (down_target / down_rand)

    # Sample biases with teacher-matched std (fallback to zeros if degenerate).
    b1_std = float(fc1_b.std(unbiased=False).item())
    b2_std = float(fc2_b.std(unbiased=False).item())
    b1 = torch.randn(fc1_b.shape, generator=g, dtype=fc1_b.dtype) * b1_std if b1_std > 0 else torch.zeros_like(fc1_b)
    b2 = torch.randn(fc2_b.shape, generator=g, dtype=fc2_b.dtype) * b2_std if b2_std > 0 else torch.zeros_like(fc2_b)

    stats = {
        "target_up_proj_avg_norm": up_target,
        "target_down_proj_avg_norm": down_target,
        "sampled_up_proj_avg_norm": float(w1.norm(dim=1).mean().item()),
        "sampled_down_proj_avg_norm": float(w2.norm(dim=0).mean().item()),
        "teacher_fc1_bias_std": b1_std,
        "teacher_fc2_bias_std": b2_std,
    }
    return w1, b1, w2, b2, stats


def resolve_ckpt_path(ckpt_path: str) -> str:
    """Resolve checkpoint path locally, otherwise download known checkpoint from HF."""
    if ckpt_path and os.path.exists(ckpt_path):
        return ckpt_path

    # Try local relative candidates in this workspace.
    for rel in HF_CKPT_REL_CANDIDATES:
        p = Path(rel)
        if p.exists():
            return str(p)

    try:
        from huggingface_hub import hf_hub_download
    except ImportError as exc:
        raise SystemExit(
            "Checkpoint not found locally and huggingface_hub is not installed. "
            "Install with `pip install huggingface_hub` or place checkpoint locally."
        ) from exc

    cache_dir = Path("out/hf_downloads")
    cache_dir.mkdir(parents=True, exist_ok=True)

    # If user supplied a non-empty path, try it first as HF relative filename.
    hf_candidates = []
    if ckpt_path:
        hf_candidates.append(ckpt_path)
    hf_candidates.extend(HF_CKPT_REL_CANDIDATES)

    last_err = None
    for filename in hf_candidates:
        try:
            local_file = hf_hub_download(
                repo_id=HF_REPO_ID,
                filename=filename,
                local_dir=str(cache_dir),
            )
            return str(local_file)
        except Exception as err:  # noqa: BLE001
            last_err = err

    raise SystemExit(
        f"Could not resolve checkpoint from local paths or HF repo {HF_REPO_ID}. "
        f"Tried: {hf_candidates}. Last error: {last_err}"
    )


@torch.no_grad()
def collect_teacher_pairs(
    ckpt_path: str,
    data_dir: str,
    layer_idx: int,
    num_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    data_seed: int = 0,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = TransformerConfig(**ckpt["model_cfg"])
    model = SparseTransformerLM(mcfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    train_data = load_bin(os.path.join(data_dir, "train.bin"))
    val_data = load_bin(os.path.join(data_dir, "val.bin"))

    g = torch.Generator(device="cpu")
    g.manual_seed(int(data_seed))

    def run_split(data):
        us = []
        ys = []

        # hook: capture ln2(x) and mlp(ln2(x)) at a single layer
        ln2_in = {}

        def ln2_hook(_mod, _inp, out):
            # Capture post-ln2 activations: this is the true MLP input.
            ln2_in["u"] = out.detach()

        def mlp_hook(_mod, _inp, out):
            ln2_in["y"] = out.detach()

        blk = model.blocks[layer_idx]
        h1 = blk.ln2.register_forward_hook(ln2_hook)
        h2 = blk.mlp.register_forward_hook(mlp_hook)

        for _ in range(num_batches):
            xb, _ = get_batch(data, batch_size, block_size, device, g)
            _logits, _loss = model(xb, None)
            u = ln2_in["u"]  # [B,T,C]
            y = ln2_in["y"]  # [B,T,C]
            us.append(u.reshape(-1, u.size(-1)).cpu().numpy())
            ys.append(y.reshape(-1, y.size(-1)).cpu().numpy())

        h1.remove()
        h2.remove()
        return np.concatenate(us, axis=0), np.concatenate(ys, axis=0)

    u_train, y_train = run_split(train_data)
    u_val, y_val = run_split(val_data)
    meta = {
        "ckpt": ckpt_path,
        "layer_idx": layer_idx,
        "n_embd": mcfg.n_embd,
        "mlp_mult": mcfg.mlp_mult,
        "mlp_topk": mcfg.mlp_topk,
        "use_relu_before_topk": mcfg.use_relu_before_topk,
        "block_size": block_size,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "data_seed": data_seed,
    }
    return u_train, y_train, u_val, y_val, meta


@torch.no_grad()
def collect_random_teacher_pairs(
    ckpt_path: str,
    data_dir: str,
    layer_idx: int,
    num_batches: int,
    batch_size: int,
    block_size: int,
    device: str,
    data_seed: int = 0,
    random_teacher_seed: int = 1234,
):
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = TransformerConfig(**ckpt["model_cfg"])
    model = SparseTransformerLM(mcfg).to(device)
    model.load_state_dict(ckpt["model"], strict=False)
    model.eval()

    train_data = load_bin(os.path.join(data_dir, "train.bin"))
    val_data = load_bin(os.path.join(data_dir, "val.bin"))

    data_g = torch.Generator(device="cpu")
    data_g.manual_seed(int(data_seed))
    synth_g = torch.Generator(device="cpu")
    synth_g.manual_seed(int(data_seed) + 10_000)

    blk = model.blocks[layer_idx]
    teacher_fc1 = blk.mlp.fc1
    teacher_fc2 = blk.mlp.fc2
    rw1, rb1, rw2, rb2, rand_stats = _sample_random_mlp_params_like_teacher(
        teacher_fc1.weight.detach().cpu(),
        teacher_fc1.bias.detach().cpu(),
        teacher_fc2.weight.detach().cpu(),
        teacher_fc2.bias.detach().cpu(),
        random_seed=random_teacher_seed,
    )
    rw1 = rw1.to(device)
    rb1 = rb1.to(device)
    rw2 = rw2.to(device)
    rb2 = rb2.to(device)

    def estimate_ln2_std(data) -> float:
        # Estimate a single isotropic sigma from real post-ln2 activations.
        stats = {"sum": 0.0, "sumsq": 0.0, "count": 0}
        ln2_out = {}

        def ln2_hook(_mod, _inp, out):
            ln2_out["u"] = out.detach()

        h = blk.ln2.register_forward_hook(ln2_hook)
        for _ in range(num_batches):
            xb, _ = get_batch(data, batch_size, block_size, device, data_g)
            _logits, _loss = model(xb, None)
            u = ln2_out["u"]
            stats["sum"] += float(u.sum().item())
            stats["sumsq"] += float((u * u).sum().item())
            stats["count"] += int(u.numel())
        h.remove()

        mean = stats["sum"] / max(stats["count"], 1)
        var = stats["sumsq"] / max(stats["count"], 1) - mean * mean
        return float(max(var, 0.0) ** 0.5)

    sigma = estimate_ln2_std(train_data)

    def run_split_synth():
        us = []
        ys = []
        n_rows = batch_size * block_size
        d_model = int(mcfg.n_embd)
        for _ in range(num_batches):
            u_flat = torch.randn((n_rows, d_model), generator=synth_g, dtype=rw1.dtype)
            u_flat = (u_flat * sigma).to(device)
            h1 = torch.addmm(rb1, u_flat, rw1.t())
            h1_sparse = _topk_sparsify(h1, k=mcfg.mlp_topk, use_relu=bool(mcfg.use_relu_before_topk))
            y_flat = torch.addmm(rb2, h1_sparse, rw2.t())
            us.append(u_flat.cpu().numpy())
            ys.append(y_flat.cpu().numpy())
        return np.concatenate(us, axis=0), np.concatenate(ys, axis=0)

    u_train, y_train = run_split_synth()
    u_val, y_val = run_split_synth()
    meta = {
        "teacher_source": "random",
        "teacher_input_source": "isotropic_gaussian",
        "teacher_input_sigma": sigma,
        "ckpt": ckpt_path,
        "layer_idx": layer_idx,
        "n_embd": mcfg.n_embd,
        "mlp_mult": mcfg.mlp_mult,
        "mlp_topk": mcfg.mlp_topk,
        "use_relu_before_topk": mcfg.use_relu_before_topk,
        "block_size": block_size,
        "batch_size": batch_size,
        "num_batches": num_batches,
        "data_seed": data_seed,
        "random_teacher_seed": int(random_teacher_seed),
        **rand_stats,
    }
    return u_train, y_train, u_val, y_val, meta


def nmse(y_hat: torch.Tensor, y: torch.Tensor, eps: float) -> torch.Tensor:
    mse = torch.mean((y_hat - y) ** 2)
    var = torch.var(y, unbiased=False)
    return mse / (var + eps)


def _mean_max_cosine_sim(gt_rows: torch.Tensor, cand_rows: torch.Tensor, chunk_size: int = 1024) -> float:
    """For each GT row vector, take max cosine to candidate rows, then average."""
    gt = F.normalize(gt_rows, dim=1, eps=1e-12)
    cand = F.normalize(cand_rows, dim=1, eps=1e-12)
    vals = []
    for i in range(0, gt.shape[0], chunk_size):
        g = gt[i : i + chunk_size]
        sims = g @ cand.t()
        vals.append(sims.max(dim=1).values)
    return float(torch.cat(vals, dim=0).mean().item())


def build_teacher_reference_vectors(meta: dict, device: str) -> dict[str, torch.Tensor]:
    ckpt = torch.load(meta["ckpt"], map_location="cpu")
    layer_idx = int(meta["layer_idx"])
    w1 = ckpt["model"][f"blocks.{layer_idx}.mlp.fc1.weight"].detach().cpu()
    b1 = ckpt["model"][f"blocks.{layer_idx}.mlp.fc1.bias"].detach().cpu()
    w2 = ckpt["model"][f"blocks.{layer_idx}.mlp.fc2.weight"].detach().cpu()
    b2 = ckpt["model"][f"blocks.{layer_idx}.mlp.fc2.bias"].detach().cpu()

    if meta.get("teacher_source", "real") == "random":
        random_seed = int(meta.get("random_teacher_seed", 1234))
        w1, b1, w2, b2, _ = _sample_random_mlp_params_like_teacher(
            w1, b1, w2, b2, random_seed=random_seed
        )

    # Keys are fc1 rows [H, D]. Values are fc2 columns [H, D].
    gt_keys = w1.float().to(device)
    gt_values = w2.t().float().to(device)
    return {"gt_keys": gt_keys, "gt_values": gt_values}


def _parse_optional_int_list(raw: str) -> list[int | None]:
    out: list[int | None] = []
    for part in raw.split(","):
        p = part.strip().lower()
        if p in {"none", "null"}:
            out.append(None)
        elif p:
            out.append(int(p))
    if not out:
        raise ValueError("parsed list is empty")
    return out


def _mean_active_per_sample(z_sparse: torch.Tensor) -> float:
    return float((z_sparse != 0).sum(dim=-1).float().mean().item())


def train_one_student(
    u_train: np.ndarray,
    y_train: np.ndarray,
    u_val: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int,
    mlp_topk: int | None,
    use_relu_before_topk: bool,
    out_dir: str,
    cfg: SimpleNamespace,
    device: str,
    teacher_meta: dict | None = None,
    teacher_refs: dict[str, torch.Tensor] | None = None,
    save_final_checkpoint: bool = False,
):
    d_in = int(u_train.shape[1])
    d_out = int(y_train.shape[1])
    model = StudentMLP(
        d_in=d_in,
        d_hidden=hidden_size,
        d_out=d_out,
        mlp_topk=mlp_topk,
        use_relu_before_topk=use_relu_before_topk,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(int(cfg.max_steps), 1),
        eta_min=float(cfg.learning_rate) * min_lr_ratio,
    )

    k_name = "none" if mlp_topk is None else str(int(mlp_topk))
    run_name = f"h{hidden_size}_k{k_name}_relu{int(bool(use_relu_before_topk))}"
    writer = SummaryWriter(log_dir=os.path.join(out_dir, run_name))
    writer.add_text(
        "hparams",
        (
            f"hidden_size={hidden_size}, mlp_topk={mlp_topk}, "
            f"use_relu_before_topk={bool(use_relu_before_topk)}"
        ),
        0,
    )
    if teacher_meta is not None:
        writer.add_text("teacher_meta", repr(teacher_meta), 0)
        for k_meta, v_meta in teacher_meta.items():
            if isinstance(v_meta, (int, float, bool)):
                writer.add_scalar(f"teacher/{k_meta}", float(v_meta), 0)

    # small helper to sample random minibatches from numpy arrays
    rng = np.random.default_rng(0)

    def batch_from(arr_u, arr_y):
        idx = rng.integers(0, arr_u.shape[0], size=(cfg.batch_size,))
        u = torch.from_numpy(np.asarray(arr_u[idx])).to(device)
        y = torch.from_numpy(np.asarray(arr_y[idx])).to(device)
        return u, y

    best_nmse = float("inf")
    for step in range(cfg.max_steps + 1):
        model.train()
        u, y = batch_from(u_train, y_train)
        yhat = model(u)
        loss = torch.mean((yhat - y) ** 2)
        nmse_train = float(nmse(yhat, y, cfg.eps).item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        scheduler.step()

        if step % cfg.log_interval == 0:
            writer.add_scalar("mse/train", loss.item(), step)
            writer.add_scalar("nmse/train", nmse_train, step)
            writer.add_scalar("optim/lr", float(opt.param_groups[0]["lr"]), step)

        if step % cfg.eval_interval == 0:
            model.eval()
            u2, y2 = batch_from(u_val, y_val)
            with torch.no_grad():
                y2hat = model(u2)
                mse_val = torch.mean((y2hat - y2) ** 2).item()
                nmse_val = float(nmse(y2hat, y2, cfg.eps).item())
                h2 = model.forward_hidden(u2)
                active_per_sample = _mean_active_per_sample(h2)
            writer.add_scalar("mse/val", mse_val, step)
            writer.add_scalar("nmse/val", nmse_val, step)
            writer.add_scalar("codes/mean_active_per_sample_val", active_per_sample, step)
            if teacher_refs is not None:
                key_cos = _mean_max_cosine_sim(
                    teacher_refs["gt_keys"],
                    model.fc1.weight.detach(),
                )
                value_cos = _mean_max_cosine_sim(
                    teacher_refs["gt_values"],
                    model.fc2.weight.detach().t(),
                )
                writer.add_scalar("params/key_max_cos_mean", key_cos, step)
                writer.add_scalar("params/value_max_cos_mean", value_cos, step)
            if nmse_val < best_nmse:
                best_nmse = nmse_val
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "hidden_size": hidden_size,
                        "mlp_topk": mlp_topk,
                        "use_relu_before_topk": bool(use_relu_before_topk),
                        "best_nmse": best_nmse,
                    },
                    os.path.join(out_dir, f"{run_name}.pt"),
                )
            writer.add_scalar("nmse/best_val", best_nmse, step)

    writer.flush()
    writer.close()
    if save_final_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "hidden_size": hidden_size,
                "mlp_topk": mlp_topk,
                "use_relu_before_topk": bool(use_relu_before_topk),
                "final_nmse": best_nmse,
                "final_step": int(cfg.max_steps),
            },
            os.path.join(out_dir, f"{run_name}_final.pt"),
        )
    return best_nmse


def train_one_sae(
    y_train: np.ndarray,
    y_val: np.ndarray,
    hidden_size: int,
    sae_topk: int | None,
    use_relu_before_topk: bool,
    use_pre_bias: bool,
    out_dir: str,
    cfg: SimpleNamespace,
    device: str,
    teacher_meta: dict | None = None,
    teacher_refs: dict[str, torch.Tensor] | None = None,
    save_final_checkpoint: bool = False,
):
    d_in = int(y_train.shape[1])
    model = TopKSAE(
        d_in=d_in,
        d_hidden=hidden_size,
        k=sae_topk,
        use_relu_before_topk=use_relu_before_topk,
        use_pre_bias=use_pre_bias,
    ).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=cfg.learning_rate, betas=(cfg.beta1, cfg.beta2), weight_decay=cfg.weight_decay)
    min_lr_ratio = float(getattr(cfg, "min_lr_ratio", 0.1))
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt,
        T_max=max(int(cfg.max_steps), 1),
        eta_min=float(cfg.learning_rate) * min_lr_ratio,
    )

    k_name = "none" if sae_topk is None else str(int(sae_topk))
    run_name = f"sae_h{hidden_size}_k{k_name}_relu{int(bool(use_relu_before_topk))}_pb{int(bool(use_pre_bias))}"
    writer = SummaryWriter(log_dir=os.path.join(out_dir, run_name))
    writer.add_text(
        "hparams",
        (
            f"sae_hidden_size={hidden_size}, sae_topk={sae_topk}, "
            f"use_relu_before_topk={bool(use_relu_before_topk)}, use_pre_bias={bool(use_pre_bias)}"
        ),
        0,
    )
    if teacher_meta is not None:
        writer.add_text("teacher_meta", repr(teacher_meta), 0)
        for k_meta, v_meta in teacher_meta.items():
            if isinstance(v_meta, (int, float, bool)):
                writer.add_scalar(f"teacher/{k_meta}", float(v_meta), 0)

    rng = np.random.default_rng(0)

    def batch_from(arr_y):
        idx = rng.integers(0, arr_y.shape[0], size=(cfg.batch_size,))
        y = torch.from_numpy(np.asarray(arr_y[idx])).to(device)
        return y

    best_nmse = float("inf")
    for step in range(cfg.max_steps + 1):
        model.train()
        y = batch_from(y_train)
        y_hat, z_sparse = model(y)
        loss = torch.mean((y_hat - y) ** 2)
        nmse_train = float(nmse(y_hat, y, cfg.eps).item())
        opt.zero_grad(set_to_none=True)
        loss.backward()
        model.unit_norm_decoder_()
        model.unit_norm_decoder_grad_adjustment_()
        opt.step()
        scheduler.step()
        model.unit_norm_decoder_()

        if step % cfg.log_interval == 0:
            writer.add_scalar("mse/train", loss.item(), step)
            writer.add_scalar("nmse/train", nmse_train, step)
            sparsity = float((z_sparse != 0).float().mean().item())
            writer.add_scalar("codes/nonzero_fraction_train", sparsity, step)
            writer.add_scalar("codes/mean_active_per_sample_train", _mean_active_per_sample(z_sparse), step)
            writer.add_scalar("optim/lr", float(opt.param_groups[0]["lr"]), step)

        if step % cfg.eval_interval == 0:
            model.eval()
            y2 = batch_from(y_val)
            with torch.no_grad():
                y2_hat, z2 = model(y2)
                mse_val = torch.mean((y2_hat - y2) ** 2).item()
                nmse_val = float(nmse(y2_hat, y2, cfg.eps).item())
                sparsity_val = float((z2 != 0).float().mean().item())
            writer.add_scalar("mse/val", mse_val, step)
            writer.add_scalar("nmse/val", nmse_val, step)
            writer.add_scalar("codes/nonzero_fraction_val", sparsity_val, step)
            writer.add_scalar("codes/mean_active_per_sample_val", _mean_active_per_sample(z2), step)
            if teacher_refs is not None:
                # GT features are teacher value vectors.
                enc_cos = _mean_max_cosine_sim(
                    teacher_refs["gt_values"],
                    model.encoder.weight.detach(),
                )
                dec_cos = _mean_max_cosine_sim(
                    teacher_refs["gt_values"],
                    model.decoder.weight.detach().t(),
                )
                writer.add_scalar("params/encoder_vs_gt_value_max_cos_mean", enc_cos, step)
                writer.add_scalar("params/decoder_vs_gt_value_max_cos_mean", dec_cos, step)
            if nmse_val < best_nmse:
                best_nmse = nmse_val
                torch.save(
                    {
                        "state_dict": model.state_dict(),
                        "sae_hidden_size": hidden_size,
                        "sae_topk": sae_topk,
                        "use_relu_before_topk": bool(use_relu_before_topk),
                        "use_pre_bias": bool(use_pre_bias),
                        "best_nmse": best_nmse,
                    },
                    os.path.join(out_dir, f"{run_name}.pt"),
                )
            writer.add_scalar("nmse/best_val", best_nmse, step)

    writer.flush()
    writer.close()
    if save_final_checkpoint:
        torch.save(
            {
                "state_dict": model.state_dict(),
                "sae_hidden_size": hidden_size,
                "sae_topk": sae_topk,
                "use_relu_before_topk": bool(use_relu_before_topk),
                "use_pre_bias": bool(use_pre_bias),
                "final_nmse": best_nmse,
                "final_step": int(cfg.max_steps),
            },
            os.path.join(out_dir, f"{run_name}_final.pt"),
        )
    return best_nmse


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=str, default="main_mlp_mult_32_K_8/ckpt_iter300000.pt")
    parser.add_argument("--data_dir", type=str, default="data/tinystories")
    parser.add_argument("--teacher_dir", type=str, default="out/distill_suite/target_models/teacher_real", help="directory to save/load teacher data")
    parser.add_argument("--out_dir", type=str, default="out/distill_suite/main_mlp_mult_32_K_8/layer2", help="directory to save student models")
    parser.add_argument("--layer_idx", type=int, default=2)
    parser.add_argument("--num_batches", type=int, default=200)
    parser.add_argument("--teacher_batch_size", type=int, default=16)
    parser.add_argument("--teacher_block_size", type=int, default=256)
    parser.add_argument("--teacher_data_seed", type=int, default=0)
    parser.add_argument("--teacher_source", type=str, choices=["real", "random"], default="real")
    parser.add_argument(
        "--random_teacher_seed",
        type=int,
        default=1234,
        help="seed for random teacher MLP when --teacher_source=random",
    )

    parser.add_argument("--skip_collection", action="store_true", help="skip teacher data collection (assume it exists)")
    parser.add_argument("--skip_training", action="store_true", help="skip student training (only collect teacher data)")
    parser.add_argument("--grid_config", type=str, default="config/distill_mlp_grid.py")
    parser.add_argument(
        "--hidden_sizes_override",
        type=str,
        default=None,
        help="comma-separated hidden sizes override, e.g. '10240,16384'",
    )
    parser.add_argument(
        "--mlp_topk_override",
        type=str,
        default=None,
        help="comma-separated topk override, supports 'none', e.g. 'none,32,64'",
    )
    parser.add_argument(
        "--match_teacher_hparams",
        action="store_true",
        help="train one student using teacher MLP hyperparameters (h=mlp_mult*n_embd, k=mlp_topk, relu=use_relu_before_topk)",
    )
    parser.add_argument(
        "--train_topk_sae",
        action="store_true",
        help="train TopK SAE(s) on teacher MLP outputs y to test sparse-neuron decomposition quality",
    )
    parser.add_argument("--max_steps", type=int, default=None, help="override grid max_steps")
    parser.add_argument(
        "--save_final_checkpoints",
        action="store_true",
        help="save additional final-step checkpoints for each trained model",
    )
    parser.add_argument(
        "--skip_student_training",
        action="store_true",
        help="skip transcoder training (train SAE only)",
    )
    parser.add_argument(
        "--sae_pre_bias",
        type=str,
        choices=["both", "0", "1"],
        default="both",
        help="which SAE pre-bias variants to train: both, pb0 (0), or pb1 (1)",
    )
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    args = parser.parse_args()

    device = args.device
    args.ckpt = resolve_ckpt_path(args.ckpt)

    # Check if teacher data already exists
    teacher_meta_path = os.path.join(args.teacher_dir, "meta.pkl")
    teacher_data_exists = os.path.exists(teacher_meta_path)

    # Step 1: Collect teacher data (unless skipped or already exists)
    if not args.skip_collection and (not teacher_data_exists or args.skip_training == False):
        if args.ckpt is None:
            raise SystemExit("provide --ckpt to collect teacher pairs")
        
        print(f"Collecting teacher data from {args.ckpt} (source={args.teacher_source})...")
        _ = load_meta(args.data_dir)  # sanity check meta exists
        if args.teacher_source == "real":
            u_train, y_train, u_val, y_val, meta = collect_teacher_pairs(
                ckpt_path=args.ckpt,
                data_dir=args.data_dir,
                layer_idx=args.layer_idx,
                num_batches=args.num_batches,
                batch_size=args.teacher_batch_size,
                block_size=args.teacher_block_size,
                device=device,
                data_seed=args.teacher_data_seed,
            )
            meta["teacher_source"] = "real"
        else:
            u_train, y_train, u_val, y_val, meta = collect_random_teacher_pairs(
                ckpt_path=args.ckpt,
                data_dir=args.data_dir,
                layer_idx=args.layer_idx,
                num_batches=args.num_batches,
                batch_size=args.teacher_batch_size,
                block_size=args.teacher_block_size,
                device=device,
                data_seed=args.teacher_data_seed,
                random_teacher_seed=args.random_teacher_seed,
            )
        save_teacher_dump(args.teacher_dir, u_train, y_train, u_val, y_val, meta)
        print(f"Saved teacher dump to {args.teacher_dir}")
        if meta.get("teacher_source") == "random":
            print(
                "Random teacher norms "
                f"(target up/down={meta['target_up_proj_avg_norm']:.4f}/{meta['target_down_proj_avg_norm']:.4f}, "
                f"sampled up/down={meta['sampled_up_proj_avg_norm']:.4f}/{meta['sampled_down_proj_avg_norm']:.4f})"
            )
    elif args.skip_collection or teacher_data_exists:
        print(f"Loading teacher data from {args.teacher_dir}...")
        u_train, y_train, u_val, y_val, meta = load_teacher_dump(args.teacher_dir)

    # Step 2: Train student models (unless skipped)
    if not args.skip_training:
        print("Starting student model training...")
        grid = load_config(args.grid_config)
        if args.hidden_sizes_override:
            grid.hidden_sizes = [int(x.strip()) for x in args.hidden_sizes_override.split(",") if x.strip()]
        if args.mlp_topk_override:
            grid.mlp_topk_list = _parse_optional_int_list(args.mlp_topk_override)
        if args.max_steps is not None:
            grid.max_steps = int(args.max_steps)
        teacher_refs = build_teacher_reference_vectors(meta, device=device) if args.match_teacher_hparams else None
        os.makedirs(args.out_dir, exist_ok=True)
        with open(os.path.join(args.out_dir, "teacher_meta.pkl"), "wb") as f:
            pickle.dump(meta, f)

        def _resolve_teacher_matched_hparams() -> tuple[int, int | None, bool]:
            if "mlp_mult" not in meta or "n_embd" not in meta or "mlp_topk" not in meta:
                raise SystemExit(
                    "Teacher metadata missing MLP hyperparameters. Re-collect teacher dump without --skip_collection."
                )
            h = int(meta["mlp_mult"]) * int(meta["n_embd"])
            k = None if meta.get("mlp_topk") is None else int(meta["mlp_topk"])
            if args.mlp_topk_override:
                parsed = _parse_optional_int_list(args.mlp_topk_override)
                if len(parsed) != 1:
                    raise SystemExit(
                        "With --match_teacher_hparams, --mlp_topk_override must be a single value, e.g. '4' or '16'."
                    )
                k = parsed[0]
            use_relu = bool(meta.get("use_relu_before_topk", True))
            if k is not None and k >= h:
                raise SystemExit(f"Invalid teacher hparams: mlp_topk={k} must be < hidden_size={h}.")
            return h, k, use_relu

        def _sae_pre_bias_variants() -> tuple[bool, ...]:
            if args.sae_pre_bias == "0":
                return (False,)
            if args.sae_pre_bias == "1":
                return (True,)
            return (False, True)

        if args.match_teacher_hparams and not args.skip_student_training:
            h, k, use_relu = _resolve_teacher_matched_hparams()
            print(f"Training teacher-matched student: h={h} k={k} use_relu_before_topk={use_relu}")
            train_one_student(
                u_train,
                y_train,
                u_val,
                y_val,
                hidden_size=h,
                mlp_topk=k,
                use_relu_before_topk=use_relu,
                out_dir=args.out_dir,
                cfg=grid,
                device=device,
                teacher_meta=meta,
                teacher_refs=teacher_refs,
                save_final_checkpoint=args.save_final_checkpoints,
            )
        elif not args.skip_student_training:
            # train all (hidden_size, topk) where topk < hidden_size
            for h in grid.hidden_sizes:
                for k in grid.mlp_topk_list:
                    if k is not None and k >= h:
                        continue
                    use_relu = bool(getattr(grid, "use_relu_before_topk", meta.get("use_relu_before_topk", True)))
                    print(f"Training student: h={h} k={k} use_relu_before_topk={use_relu}")
                    train_one_student(
                        u_train,
                        y_train,
                        u_val,
                        y_val,
                        hidden_size=h,
                        mlp_topk=k,
                        use_relu_before_topk=use_relu,
                        out_dir=args.out_dir,
                        cfg=grid,
                        device=device,
                        teacher_meta=meta,
                        teacher_refs=teacher_refs,
                        save_final_checkpoint=args.save_final_checkpoints,
                    )
        if args.train_topk_sae:
            print("Starting TopK SAE training on teacher MLP outputs...")
            if args.match_teacher_hparams:
                h, k, use_relu = _resolve_teacher_matched_hparams()
                for use_pre_bias in _sae_pre_bias_variants():
                    print(
                        f"Training teacher-matched TopK SAE: h={h} k={k} use_relu_before_topk={use_relu} "
                        f"use_pre_bias={use_pre_bias}"
                    )
                    train_one_sae(
                        y_train=y_train,
                        y_val=y_val,
                        hidden_size=h,
                        sae_topk=k,
                        use_relu_before_topk=use_relu,
                        use_pre_bias=use_pre_bias,
                        out_dir=args.out_dir,
                        cfg=grid,
                        device=device,
                        teacher_meta=meta,
                        teacher_refs=teacher_refs,
                        save_final_checkpoint=args.save_final_checkpoints,
                    )
            else:
                for h in grid.hidden_sizes:
                    for k in grid.mlp_topk_list:
                        if k is not None and k >= h:
                            continue
                        use_relu = bool(getattr(grid, "use_relu_before_topk", meta.get("use_relu_before_topk", True)))
                        for use_pre_bias in _sae_pre_bias_variants():
                            print(
                                f"Training TopK SAE: h={h} k={k} use_relu_before_topk={use_relu} "
                                f"use_pre_bias={use_pre_bias}"
                            )
                            train_one_sae(
                                y_train=y_train,
                                y_val=y_val,
                                hidden_size=h,
                                sae_topk=k,
                                use_relu_before_topk=use_relu,
                                use_pre_bias=use_pre_bias,
                                out_dir=args.out_dir,
                                cfg=grid,
                                device=device,
                                teacher_meta=meta,
                                teacher_refs=teacher_refs,
                                save_final_checkpoint=args.save_final_checkpoints,
                            )
        print(f"Student/SAE training complete. Models saved to {args.out_dir}")


if __name__ == "__main__":
    main()


