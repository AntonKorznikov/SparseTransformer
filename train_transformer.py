from __future__ import annotations

import os
import sys
import copy
import tempfile

import os
import pickle
import time
from dataclasses import asdict

import numpy as np
import torch
from torch.utils.tensorboard import SummaryWriter
from huggingface_hub import upload_file

from model_transformer import SparseTransformerLM, TransformerConfig
from utils.common import (
    device_from_cfg,
    get_batch,
    get_lr,
    load_bin,
    load_config,
    load_meta,
    set_seeds,
    StreamingDataset,
)


@torch.no_grad()
def estimate_loss(model: SparseTransformerLM, train_data, val_data, cfg, device, data_gen, use_streaming=False):
    out = {}
    model.eval()
    for split, data in [("train", train_data), ("val", val_data)]:
        losses = torch.zeros(cfg.eval_iters)
        for k in range(cfg.eval_iters):  # noqa: B007
            if use_streaming:
                xb, yb = data.get_batch(cfg.batch_size, cfg.block_size, device)
            else:
                xb, yb = get_batch(data, cfg.batch_size, cfg.block_size, device, data_gen)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def rotate_checkpoints(out_dir: str, keep_last: int) -> None:
    if keep_last <= 0:
        return
    import re

    pat = re.compile(r"^ckpt_iter(\d+)\.pt$")
    iters_and_files = []
    for fn in os.listdir(out_dir):
        m = pat.match(fn)
        if m:
            iters_and_files.append((int(m.group(1)), fn))
    iters_and_files.sort(reverse=True)
    for _, fn in iters_and_files[keep_last:]:
        try:
            os.remove(os.path.join(out_dir, fn))
        except FileNotFoundError:
            pass


def _zero_adamw_state_for_index(param: torch.nn.Parameter, optimizer: torch.optim.Optimizer, idx):
    st = optimizer.state.get(param)
    if not st:
        return
    for key in ("exp_avg", "exp_avg_sq"):
        if key in st and st[key] is not None:
            try:
                st[key][idx] = 0
            except Exception:
                # best-effort: if indexing doesn't match, skip
                pass


@torch.no_grad()
def _resample_rows_like(w: torch.Tensor, row_idx: torch.Tensor) -> None:
    """
    Reinitialize selected rows of w with random directions (uniform on unit sphere)
    and row norms matched to the mean row norm of the existing weight matrix.
    """
    if row_idx.numel() == 0:
        return
    # mean row norm (avoid zeros)
    row_norms = w.norm(dim=1)
    mean_norm = row_norms.mean().clamp_min(1e-12)
    dirs = torch.randn((row_idx.numel(), w.size(1)), device=w.device, dtype=w.dtype)
    dirs = dirs / dirs.norm(dim=1, keepdim=True).clamp_min(1e-12)
    w[row_idx] = dirs * (mean_norm * 1.0)


@torch.no_grad()
def _resample_cols_like(w: torch.Tensor, col_idx: torch.Tensor) -> None:
    """
    Reinitialize selected columns of w with random directions (uniform on unit sphere)
    and column norms matched to the mean column norm of the existing weight matrix.
    """
    if col_idx.numel() == 0:
        return
    col_norms = w.norm(dim=0)
    mean_norm = col_norms.mean().clamp_min(1e-12)
    dirs = torch.randn((w.size(0), col_idx.numel()), device=w.device, dtype=w.dtype)
    dirs = dirs / dirs.norm(dim=0, keepdim=True).clamp_min(1e-12)
    w[:, col_idx] = dirs * (mean_norm * 1.0)


@torch.no_grad()
def resample_dead_neurons(model: SparseTransformerLM, optimizer: torch.optim.Optimizer, dead_masks):
    """
    Resample dead hidden neurons in each layer MLP.
    For a hidden neuron j, we reinit:
    - fc1.weight[j, :], fc1.bias[j]
    - fc2.weight[:, j]
    and zero AdamW moments for those slices (best-effort).
    """
    for li, blk in enumerate(model.blocks):
        mask = dead_masks[li]
        if mask is None or mask.numel() == 0:
            continue
        dead_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
        if dead_idx.numel() == 0:
            continue

        fc1 = blk.mlp.fc1
        fc2 = blk.mlp.fc2

        # weights live on GPU; reinit directions uniformly on unit sphere and match mean norms
        _resample_rows_like(fc1.weight, dead_idx)
        _resample_cols_like(fc2.weight, dead_idx)
        # biases -> 0
        fc1.bias[dead_idx].zero_()

        # clear optimizer moments for those slices
        _zero_adamw_state_for_index(fc1.weight, optimizer, (dead_idx, slice(None)))
        _zero_adamw_state_for_index(fc1.bias, optimizer, dead_idx)
        _zero_adamw_state_for_index(fc2.weight, optimizer, (slice(None), dead_idx))


def _public_cfg(cfg):
    return {k: v for k, v in vars(cfg).items() if k != "hf_token"}


def save_and_upload_checkpoint(ckpt, cfg, filename, hf_repo_id=None, hf_token=None):
    """
    Save checkpoint to a temporary file and upload to HuggingFace Hub.
    Does not save to disk to save storage.
    """
    # Use environment variable if token not provided
    if hf_token is None:
        hf_token = os.environ.get("HF_TOKEN")
    
    if hf_repo_id is None or hf_token is None:
        # Fallback: save to disk if HF settings not provided
        os.makedirs(cfg.out_dir, exist_ok=True)
        filepath = os.path.join(cfg.out_dir, filename)
        torch.save(ckpt, filepath)
        print(f"Saved checkpoint to {filepath}")
        return
    
    # Save to temporary file and upload
    with tempfile.NamedTemporaryFile(suffix=".pt", delete=False) as tmp:
        tmp_path = tmp.name
    
    try:
        torch.save(ckpt, tmp_path)
        
        # Upload to HF Hub
        path_in_repo = f"{cfg.run_name}/{filename}"
        upload_file(
            path_or_fileobj=tmp_path,
            path_in_repo=path_in_repo,
            repo_id=hf_repo_id,
            token=hf_token,
        )
        print(f"Uploaded {filename} to {hf_repo_id}/{path_in_repo}")
    finally:
        # Clean up temp file
        if os.path.exists(tmp_path):
            os.remove(tmp_path)


def train_single_config(cfg, mlp_mult_val, mlp_topk_val):
    """
    Train a single configuration with given mlp_mult and mlp_topk values.
    """
    # Update config with specific hyperparameter values
    cfg.mlp_mult = mlp_mult_val
    cfg.mlp_topk = mlp_topk_val
    
    # Update run name to reflect the hyperparameters
    mlp_topk_str = str(mlp_topk_val) if mlp_topk_val is not None else "None"
    cfg.run_name = f"{cfg.run_name}_mlp_mult_{mlp_mult_val}_K_{mlp_topk_str}"
    
    # Get HuggingFace settings from config
    hf_repo_id = getattr(cfg, "hf_repo_id", None)
    hf_token = getattr(cfg, "hf_token", None)
    
    # Update output directory to include hyperparameters (only used for TensorBoard)
    base_out_dir = cfg.out_dir
    cfg.out_dir = os.path.join(base_out_dir, f"mlp_mult_{mlp_mult_val}_K_{mlp_topk_str}")
    cfg.tb_log_dir = os.path.join(base_out_dir, "tensorboard", f"mlp_mult_{mlp_mult_val}_K_{mlp_topk_str}")
    
    device = device_from_cfg(cfg)

    # init seed controls weight init
    set_seeds(cfg.init_seed)

    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.tb_log_dir, exist_ok=True)
    writer = SummaryWriter(log_dir=cfg.tb_log_dir)

    # Check if using streaming dataset or binary files
    use_streaming = hasattr(cfg, "dataset_name") and cfg.dataset_name is not None
    
    if use_streaming:
        # Streaming mode: load dataset from HuggingFace
        dataset_name = cfg.dataset_name
        train_split = getattr(cfg, "train_split", "train")
        val_split = getattr(cfg, "val_split", "validation")
        tokenizer_name = getattr(cfg, "tokenizer", "gpt2")
        
        print(f"Loading streaming dataset: {dataset_name}")
        train_data = StreamingDataset(dataset_name, train_split, tokenizer_name, seed=cfg.data_seed)
        val_data = StreamingDataset(dataset_name, val_split, tokenizer_name, seed=cfg.data_seed + 1)
        vocab_size = train_data.vocab_size
        print(f"Vocab size: {vocab_size}")
    else:
        # Binary file mode (backward compatibility)
        meta = load_meta(cfg.data_dir)
        vocab_size = int(meta["vocab_size"])
        train_data = load_bin(os.path.join(cfg.data_dir, "train.bin"))
        val_data = load_bin(os.path.join(cfg.data_dir, "val.bin"))

    mcfg = TransformerConfig(
        vocab_size=vocab_size,
        block_size=cfg.block_size,
        n_layer=cfg.n_layer,
        n_head=cfg.n_head,
        n_embd=cfg.n_embd,
        mlp_mult=cfg.mlp_mult,
        mlp_topk=cfg.mlp_topk,
        use_relu_before_topk=cfg.use_relu_before_topk,
        tie_weights=cfg.tie_weights,
    )
    model = SparseTransformerLM(mcfg).to(device)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=cfg.learning_rate,
        betas=(cfg.beta1, cfg.beta2),
        weight_decay=cfg.weight_decay,
    )

    scaler = torch.cuda.amp.GradScaler(enabled=cfg.amp and device.startswith("cuda"))

    # dead neuron tracking (cumulative, per layer)
    dead_thresholds = getattr(cfg, "dead_freq_thresholds", [1e-8, 1e-7, 1e-6, 1e-5])
    hidden = int(cfg.mlp_mult * cfg.n_embd)
    layer_counts = [torch.zeros(hidden, dtype=torch.int64) for _ in range(cfg.n_layer)]
    total_tokens_seen = 0
    prev_layer_counts = [c.clone() for c in layer_counts]
    prev_total_tokens_seen = 0

    # data-order seed controls batch sampling order
    data_gen = torch.Generator(device="cpu")
    data_gen.manual_seed(int(cfg.data_seed))

    best_val = 1e9
    t0 = time.time()

    # optional dead-neuron resampling
    resample_enabled = bool(getattr(cfg, "resample_dead_neurons", False))
    resample_thr = float(getattr(cfg, "resample_dead_threshold", 1e-6))
    resample_max = int(getattr(cfg, "resample_max_per_layer", 0))  # 0 => unlimited
    resample_interval = int(getattr(cfg, "resample_interval", 1))  # resample every K eval intervals
    resample_counter = 0  # counter for eval intervals

    # gradient accumulation
    grad_accum_steps = int(getattr(cfg, "gradient_accumulation_steps", 1))
    if grad_accum_steps < 1:
        grad_accum_steps = 1

    # checkpoint intervals setup
    checkpoint_fractions = getattr(cfg, "checkpoint_fractions", None)
    save_initial_checkpoint = getattr(cfg, "save_initial_checkpoint", True)
    save_final_checkpoint = getattr(cfg, "save_final_checkpoint", True)
    
    checkpoint_steps = set()
    if checkpoint_fractions is not None:
        for frac in checkpoint_fractions:
            step = int(frac * cfg.max_iters)
            checkpoint_steps.add(step)
    if save_final_checkpoint:
        checkpoint_steps.add(cfg.max_iters)
    
    # Track which checkpoints have been saved to avoid duplicates
    saved_checkpoints = set()

    # Save initial checkpoint
    if save_initial_checkpoint:
        ckpt = {
            "model": model.state_dict(),
            "model_cfg": asdict(mcfg),
            "train_cfg": _public_cfg(cfg),
            "iter": 0,
            "best_val": best_val,
        }
        save_and_upload_checkpoint(ckpt, cfg, "ckpt_iter0.pt", hf_repo_id, hf_token)
        saved_checkpoints.add(0)

    for it in range(cfg.max_iters + 1):
        lr = get_lr(it, cfg)
        for param_group in optimizer.param_groups:
            param_group["lr"] = lr

        if it % cfg.eval_interval == 0:
            losses = estimate_loss(model, train_data, val_data, cfg, device, data_gen, use_streaming=use_streaming)
            dt = time.time() - t0
            print(
                f"{cfg.run_name} iter {it}: train {losses['train']:.4f}, val {losses['val']:.4f}, lr {lr:.3e}, dt {dt:.1f}s"
            )
            t0 = time.time()

            writer.add_scalar("loss/train_eval", losses["train"], it)
            writer.add_scalar("loss/val", losses["val"], it)
            writer.add_scalar("lr", lr, it)

            improved = losses["val"] < best_val
            if improved:
                best_val = losses["val"]

            # Create checkpoint dict for interval-based saves
            ckpt = {
                "model": model.state_dict(),
                "model_cfg": asdict(mcfg),
                "train_cfg": _public_cfg(cfg),
                "iter": it,
                "best_val": best_val,
            }

            if getattr(cfg, "save_all_checkpoints", False):
                save_and_upload_checkpoint(ckpt, cfg, f"ckpt_iter{it}.pt", hf_repo_id, hf_token)

        # Save checkpoint at specified intervals (outside eval block to save at any iteration)
        if it in checkpoint_steps and it not in saved_checkpoints:
            # Create checkpoint dict if not already created (for non-eval iterations)
            if it % cfg.eval_interval != 0:
                ckpt = {
                    "model": model.state_dict(),
                    "model_cfg": asdict(mcfg),
                    "train_cfg": _public_cfg(cfg),
                    "iter": it,
                    "best_val": best_val,
                }
            save_and_upload_checkpoint(ckpt, cfg, f"ckpt_iter{it}.pt", hf_repo_id, hf_token)
            saved_checkpoints.add(it)
            print(f"Checkpoint at iter {it} ({it/cfg.max_iters*100:.1f}% of max_iters)")

        # Gradient accumulation: only zero gradients at the start of accumulation cycle
        is_accum_step = (it % grad_accum_steps == 0)
        if is_accum_step:
            optimizer.zero_grad(set_to_none=True)

        if use_streaming:
            xb, yb = train_data.get_batch(cfg.batch_size, cfg.block_size, device)
        else:
            xb, yb = get_batch(train_data, cfg.batch_size, cfg.block_size, device, data_gen)

        with torch.cuda.amp.autocast(enabled=scaler.is_enabled()):
            _, loss_raw = model(xb, yb)
            # Scale loss by accumulation steps to get average gradient
            loss = loss_raw / grad_accum_steps

        # update cumulative activation counts from each layer's BatchTopK indices
        # (uses bincount over selected indices, avoids scanning dense hidden activations)
        with torch.no_grad():
            n_tokens = int(xb.shape[0] * xb.shape[1])
            total_tokens_seen += n_tokens
            for li, blk in enumerate(model.blocks):
                fc = blk.mlp.act.last_feature_counts
                if fc is None:
                    continue
                layer_counts[li] += fc.detach().to("cpu")

        scaler.scale(loss).backward()
        
        # Only step optimizer and clip gradients at the end of accumulation cycle
        if (it + 1) % grad_accum_steps == 0:
            if cfg.grad_clip > 0:
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip)
            scaler.step(optimizer)
            scaler.update()

        if it % cfg.log_interval == 0:
            # Log unscaled loss (raw loss per batch)
            loss_to_log = loss_raw.item()
            stats = model.batchtopk_stats()
            if stats:
                print(
                    f"{cfg.run_name} iter {it}: loss {loss_to_log:.4f} | "
                    f"active/tok(mean,min,max)=({stats['batchtopk_mean_active_per_token_mean']:.1f},"
                    f"{stats['batchtopk_mean_active_per_token_min']:.1f},"
                    f"{stats['batchtopk_mean_active_per_token_max']:.1f})"
                )
            else:
                print(f"{cfg.run_name} iter {it}: loss {loss_to_log:.4f}")

            writer.add_scalar("loss/train", loss_to_log, it)
            if stats:
                writer.add_scalar("batchtopk/active_per_token_mean", stats["batchtopk_mean_active_per_token_mean"], it)
                writer.add_scalar("batchtopk/active_per_token_min", stats["batchtopk_mean_active_per_token_min"], it)
                writer.add_scalar("batchtopk/active_per_token_max", stats["batchtopk_mean_active_per_token_max"], it)

        # log dead neuron proportions per layer at eval cadence (recent-window only)
        if it % cfg.eval_interval == 0 and total_tokens_seen > 0:
            # log recent-window dead neurons (matches what analyze_activation_freq.py sees)
            # and prepare for resampling if enabled
            if it > 0 and prev_total_tokens_seen > 0:
                delta_tokens = total_tokens_seen - prev_total_tokens_seen
                if delta_tokens > 0:
                    for li in range(cfg.n_layer):
                        delta = (layer_counts[li] - prev_layer_counts[li]).double()
                        freq_recent = delta / float(delta_tokens)
                        for thr in dead_thresholds:
                            dead_prop_recent = float((freq_recent < thr).double().mean().item())
                            writer.add_scalar(f"dead_recent/layer{li}/freq_lt_{thr:g}", dead_prop_recent, it)

                    # dead-neuron resampling (uses *recent window* frequency since last eval)
                    # only resample every resample_interval eval intervals
                    resample_counter += 1
                    if resample_enabled and resample_counter >= resample_interval:
                        dead_masks = []
                        for li in range(cfg.n_layer):
                            delta = (layer_counts[li] - prev_layer_counts[li]).double()
                            freq_recent = delta / float(delta_tokens)
                            mask = (freq_recent < resample_thr)
                            if resample_max > 0 and int(mask.sum().item()) > resample_max:
                                # keep only the most-dead (lowest freq_recent) up to resample_max
                                vals = freq_recent
                                dead_idx = torch.nonzero(mask, as_tuple=False).squeeze(1)
                                picked = dead_idx[torch.argsort(vals[dead_idx])[:resample_max]]
                                mask2 = torch.zeros_like(mask)
                                mask2[picked] = True
                                mask = mask2
                            dead_masks.append(mask.to(device))

                        resample_dead_neurons(model, optimizer, dead_masks)
                        writer.add_scalar("resample/dead_threshold", resample_thr, it)
                        resample_counter = 0  # reset counter after resampling

            # update prev_* for next eval window (needed for both logging and resampling)
            prev_total_tokens_seen = total_tokens_seen
            prev_layer_counts = [c.clone() for c in layer_counts]

    # Save final checkpoint if not already saved
    if save_final_checkpoint and cfg.max_iters not in saved_checkpoints:
        losses = estimate_loss(model, train_data, val_data, cfg, device, data_gen, use_streaming=use_streaming)
        ckpt = {
            "model": model.state_dict(),
            "model_cfg": asdict(mcfg),
            "train_cfg": _public_cfg(cfg),
            "iter": cfg.max_iters,
            "best_val": best_val,
        }
        save_and_upload_checkpoint(ckpt, cfg, f"ckpt_iter{cfg.max_iters}.pt", hf_repo_id, hf_token)
        print(f"Final checkpoint at iter {cfg.max_iters}")

    # save per-neuron frequencies for analysis (based on final window only)
    delta_tokens = total_tokens_seen - prev_total_tokens_seen
    if delta_tokens > 0:
        neuron_freqs = {}
        for li in range(cfg.n_layer):
            # Use only counts from the final eval window, not cumulative from entire training
            delta_counts = (layer_counts[li] - prev_layer_counts[li]).double()
            freq = delta_counts.numpy() / float(delta_tokens)
            neuron_freqs[f"layer{li}"] = freq.astype(np.float64)

        neuron_freqs_ckpt = {
            "neuron_freqs": neuron_freqs,  # dict[layer_name] -> np.ndarray[hidden]
            "total_tokens": delta_tokens,  # tokens in the final window
            "mlp_topk": cfg.mlp_topk,
            "hidden_size": hidden,
        }
        save_and_upload_checkpoint(neuron_freqs_ckpt, cfg, "neuron_freqs.pt", hf_repo_id, hf_token)
        print(f"Neuron frequencies (final window only)")


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("config", type=str, help="path to a config file, e.g. config/main_config.py")
    args = parser.parse_args()

    base_cfg = load_config(args.config)
    
    # Read grid search parameters from config
    mlp_mult_values = list(getattr(base_cfg, "mlp_mult_values", [getattr(base_cfg, "mlp_mult", 64)]))
    topk_values = list(getattr(base_cfg, "topk_values", [getattr(base_cfg, "mlp_topk", 8)]))
    with_no_topk = bool(getattr(base_cfg, "with_no_topk", False))
    
    # Build list of (mlp_mult, mlp_topk) tuples to train
    if with_no_topk:
        topk_values.append(None)
    
    total_configs = len(mlp_mult_values) * len(topk_values)
    print(f"\n{'='*80}")
    print(f"Training {total_configs} configurations:")
    print(f"  mlp_mult values: {mlp_mult_values}")
    print(f"  K (mlp_topk) values: {topk_values}")
    print(f"{'='*80}\n")
    
    config_idx = 0
    for mlp_mult_val in mlp_mult_values:
        for mlp_topk_val in topk_values:
            config_idx += 1
            mlp_topk_str = str(mlp_topk_val) if mlp_topk_val is not None else "None"
            print(f"\n{'='*80}")
            print(f"Config {config_idx}/{total_configs}: mlp_mult={mlp_mult_val}, K={mlp_topk_str}")
            print(f"{'='*80}\n")
            
            # Create a fresh copy of the base config for this run
            cfg = copy.deepcopy(base_cfg)
            train_single_config(cfg, mlp_mult_val, mlp_topk_val)
            
            print(f"\nCompleted config {config_idx}/{total_configs}\n")


if __name__ == "__main__":
    main()


