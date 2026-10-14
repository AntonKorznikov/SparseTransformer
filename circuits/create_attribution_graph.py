"""
Compute an ablation-based attribution graph for SparseTransformerLM.

We compute C_{k,L,i} for every token position k, every layer L, and every neuron i
that is active (non-zero) at (k,L) in the post-BatchTopK activations.

Only MLP neurons are extracted as nodes and plotted. Embedding vectors are NOT
included as nodes in the graph.

Now includes FAITHFULNESS CURVE: for each circuit size n, ablate exactly those n nodes
(the circuit discovered for that size) and compute p_ablated/p0. This measures necessity.
"""

from __future__ import annotations

import argparse
import html as html_module
import json
import logging
import math
import os
import re
import time
from dataclasses import dataclass
from typing import Any

import numpy as np
import torch
from tqdm.auto import tqdm  # type: ignore[reportMissingModuleSource]

from model_adapter import create_hooked_model

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None

from circuits import attribution_graph_html

logger = logging.getLogger(__name__)

# ------------------------------------------------------------------------------
# Constants
# ------------------------------------------------------------------------------
# NOTE: Negative layers are reserved for sentinel nodes.
# - layer == -1: embedding-position node (whole embedding vector at token position k)


def save_curve_png(
    addback_curve: list[dict[str, float]],
    faithfulness_curve: list[dict[str, float]],
    meta: dict[str, Any],
    out_dir: str,
    palette: list[str] = None,
) -> None:
    """Plot add‑back curve (Faithfulness) and Completeness curve and save as PNG."""
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not installed – cannot save curve PNG.")
        return

    if palette is None:
        palette = [
            "#5f5a4a", "#6f6754", "#7f745e", "#8f8068", "#9f8d72",
            "#8a3434", "#984444", "#a65454", "#b46464", "#c27474",
        ]

    # Prepare data
    x1 = [p.get("n_added", 0) for p in addback_curve]
    y1 = [p.get("retain_ratio", 0.0) for p in addback_curve]
    x2 = [p.get("n", 0) for p in faithfulness_curve]
    y2 = [p.get("faithfulness", 0.0) for p in faithfulness_curve]

    fig, ax = plt.subplots(figsize=(8, 5))
    ax.plot(x1, y1, color=palette[5], linewidth=2.5, marker='o', label="Faithfulness")
    ax.plot(x2, y2, color=palette[8], linewidth=3, linestyle="--", label="Completeness")

    # ax.legend(
    #     loc='center left',           # Якорь легенды: левая середина самой легенды
    #     bbox_to_anchor=(1, 0.5),     # Позиция: 1 (справа от графика), 0.5 (вертикальный центр)
    #     fontsize='x-large',          # Увеличенный размер шрифта ('xx-large', 14, 16 и т.д.)
    #     frameon=False                # Убрать рамку (по желанию)
    # )

    ax.set_xlabel("Number of neurons in circuit", fontsize=14)
    ax.set_ylabel("Score", fontsize=14)
    # ax.set_title("Add‑back Curve")
    ax.grid(True, linestyle=":", alpha=0.6)
    ax.legend(loc='center right', fontsize=14)

    # Add a small horizontal line at y=1 for reference
    ax.axhline(y=1.0, color="gray", linestyle=":", linewidth=1, alpha=0.5)

    # Save
    png_path = os.path.join(out_dir, "addback_curve.png")
    plt.tight_layout()
    plt.savefig(png_path, dpi=150)
    plt.close(fig)
    logger.info("Saved curve PNG to %s", png_path)


# ------------------------------------------------------------------------------
# Data structures
# ------------------------------------------------------------------------------
@dataclass(frozen=True)
class Candidate:
    k: int
    layer: int
    neuron: int
    activation: float
    C: float | None = None


# ------------------------------------------------------------------------------
# Helper functions
# ------------------------------------------------------------------------------
def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _encode_prompt(prompt: str | None, tokens_csv: str | None) -> tuple[list[int], str]:
    if tokens_csv:
        toks = [int(x.strip()) for x in tokens_csv.split(",") if x.strip() != ""]
        return toks, "<tokens_csv>"
    if prompt is None:
        raise ValueError("Provide either --prompt or --tokens_csv")
    if tiktoken is None:
        raise ValueError("tiktoken not available; provide --tokens_csv instead of --prompt")
    enc = tiktoken.get_encoding("gpt2")
    toks = enc.encode_ordinary(prompt)
    return toks, prompt


def _prompt_dir_name(prompt_repr: str, max_len: int = 64) -> str:
    slug = re.sub(r"[^0-9a-zA-Z_-]+", "_", prompt_repr).strip("_")
    if not slug:
        slug = "prompt"
    if len(slug) > max_len:
        slug = slug[:max_len].rstrip("_")
    return slug


def _resolve_run_name(
    base_run_name: str | None,
    prompt_repr: str,
    *,
    n_prompts: int,
) -> str:
    if n_prompts == 1:
        if base_run_name:
            return base_run_name
        return time.strftime("run_%Y%m%d_%H%M%S")
    slug = _prompt_dir_name(prompt_repr)
    if base_run_name:
        return os.path.join(base_run_name, slug)
    return slug


def _decode_tokens(tokens: list[int]) -> list[str]:
    """Decode token IDs to their string representations using tiktoken."""
    out: list[str] = []
    
    # Try to get tiktoken encoder
    enc = None
    if tiktoken is not None:
        try:
            enc = tiktoken.get_encoding("gpt2")
        except Exception as e:
            print(f"Warning: tiktoken encoding failed: {e}")
    
    for t in tokens:
        s = None
        # Try tiktoken first
        if enc is not None:
            try:
                s = enc.decode([int(t)])
            except Exception:
                pass
        
        # Fallback if tiktoken failed
        if s is None:
            s = f"[{t}]"
        
        # HTML escape
        s = (
            s.replace("&", "&amp;")
            .replace("<", "&lt;")
            .replace(">", "&gt;")
            .replace('"', "&quot;")
            .replace("'", "&#39;")
        )
        out.append(s)
    return out


def _logprob_target_at_final(logits: torch.Tensor, target_id: int) -> torch.Tensor:
    last_logits = logits[:, -1, :]
    logprobs = torch.log_softmax(last_logits, dim=-1)
    return logprobs[:, int(target_id)]


def _get_top_k_tokens(logits: torch.Tensor, k: int = 10) -> list[dict[str, Any]]:
    """
    Get top-k tokens by probability from final position logits.
    Returns list of {token_id, token_str, logp, prob}
    """
    last_logits = logits[0, -1, :]  # batch_size=1, final position
    logprobs = torch.log_softmax(last_logits, dim=-1)
    probs = torch.softmax(last_logits, dim=-1)
    
    top_k_ids = torch.topk(probs, k=min(k, len(probs)), dim=-1).indices.cpu().numpy()
    
    result = []
    for token_id in top_k_ids:
        token_id = int(token_id)
        
        # Decode using the existing _decode_tokens function
        token_str = _decode_tokens([token_id])[0]
        
        logp_val = float(logprobs[token_id].item())
        prob_val = float(probs[token_id].item())
        result.append({
            "token_id": token_id,
            "token_str": token_str,
            "logp": logp_val,
            "prob": prob_val,
        })
    return result


def _register_forward_hook_prepend(module: torch.nn.Module, hook_fn):
    try:
        return module.register_forward_hook(hook_fn, prepend=True)
    except TypeError:
        return module.register_forward_hook(hook_fn)


def _get_wte_module(hooked_model) -> torch.nn.Module:
    """
    Return the token embedding module (wte) for different wrapper/model layouts.

    We support:
    - TransformerLensCompatibleModel: hooked_model.model.wte
    - HookedTransformer-like: hooked_model.model.transformer.wte (legacy)
    - Direct model: hooked_model.wte
    """
    # Most common in this repo: TransformerLensCompatibleModel(model=SparseTransformerLM)
    inner = getattr(hooked_model, "model", hooked_model)

    if hasattr(inner, "wte"):
        return inner.wte
    if hasattr(inner, "transformer") and hasattr(inner.transformer, "wte"):
        return inner.transformer.wte
    if hasattr(hooked_model, "wte"):
        return hooked_model.wte
    if hasattr(hooked_model, "transformer") and hasattr(hooked_model.transformer, "wte"):
        return hooked_model.transformer.wte

    raise AttributeError("Could not locate token embedding module (wte) on hooked_model")


# ------------------------------------------------------------------------------
# Ablation hooks – existing (embedding coordinates, MLP neurons)
# ------------------------------------------------------------------------------
def _ablation_hook_embedding_single_positions(k_tensor: torch.Tensor, i_tensor: torch.Tensor):
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3:
            return out
        out2 = out.clone()
        b = out2.shape[0]
        idx = torch.arange(b, device=out2.device)
        out2[idx, k_tensor, i_tensor] = 0
        return out2
    return hook_fn


def _ablation_hook_embedding_single_positions_norm_match(
    k_tensor: torch.Tensor,
    i_tensor: torch.Tensor,
    base_norm_by_k: torch.Tensor,
):
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3:
            return out
        out2 = out.clone()
        bsz = out2.shape[0]
        idx = torch.arange(bsz, device=out2.device)
        out2[idx, k_tensor, i_tensor] = 0
        v = out2[idx, k_tensor, :]
        cur = torch.linalg.norm(v, dim=-1)
        tgt = base_norm_by_k[k_tensor].to(device=out2.device, dtype=out2.dtype)
        extra2 = torch.clamp(tgt * tgt - cur * cur, min=0.0)
        extra = torch.sqrt(extra2)
        out2[idx, k_tensor, i_tensor] = extra
        return out2
    return hook_fn


def _ablation_hook_embedding_many_positions(k_list: list[int], i_list: list[int]):
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3 or out.shape[0] != 1:
            return out
        if not k_list:
            return out
        out2 = out.clone()
        k = torch.tensor(k_list, device=out2.device, dtype=torch.long)
        i = torch.tensor(i_list, device=out2.device, dtype=torch.long)
        out2[0, k, i] = 0
        return out2
    return hook_fn


def _ablation_hook_embedding_many_positions_norm_match(
    k_list: list[int],
    i_list: list[int],
    base_norm_by_k: torch.Tensor,
):
    grouped: dict[int, list[int]] = {}
    for k, i in zip(k_list, i_list):
        grouped.setdefault(int(k), []).append(int(i))

    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3 or out.shape[0] != 1:
            return out
        if not k_list:
            return out
        out2 = out.clone()
        k_t = torch.tensor(k_list, device=out2.device, dtype=torch.long)
        i_t = torch.tensor(i_list, device=out2.device, dtype=torch.long)
        out2[0, k_t, i_t] = 0
        for k, idxs in grouped.items():
            idx_t = torch.tensor(idxs, device=out2.device, dtype=torch.long)
            v = out2[0, k, :]
            cur = torch.linalg.norm(v)
            tgt = base_norm_by_k[int(k)].to(device=out2.device, dtype=out2.dtype)
            extra2 = torch.clamp(tgt * tgt - cur * cur, min=0.0)
            extra = torch.sqrt(extra2)
            if float(extra.item()) <= 0.0:
                continue
            if idx_t.numel() == 1:
                out2[0, k, idx_t[0]] = extra
            else:
                noise = torch.rand((idx_t.numel(),), device=out2.device, dtype=out2.dtype)
                denom = torch.linalg.norm(noise).clamp_min(1e-12)
                noise = noise / denom
                out2[0, k, idx_t] = noise * extra
        return out2
    return hook_fn


def _ablation_hook_single_positions(k_tensor: torch.Tensor, i_tensor: torch.Tensor):
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3:
            return out
        out2 = out.clone()
        b = out2.shape[0]
        idx = torch.arange(b, device=out2.device)
        out2[idx, k_tensor, i_tensor] = 0
        return out2
    return hook_fn


def _ablation_hook_single_positions_norm_match(
    k_tensor: torch.Tensor,
    i_tensor: torch.Tensor,
    base_norm_by_k: torch.Tensor,
):
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3:
            return out
        out2 = out.clone()
        bsz = out2.shape[0]
        idx = torch.arange(bsz, device=out2.device)
        out2[idx, k_tensor, i_tensor] = 0
        v = out2[idx, k_tensor, :]
        cur = torch.linalg.norm(v, dim=-1)
        tgt = base_norm_by_k[k_tensor].to(device=out2.device, dtype=out2.dtype)
        extra2 = torch.clamp(tgt * tgt - cur * cur, min=0.0)
        extra = torch.sqrt(extra2)
        out2[idx, k_tensor, i_tensor] = extra
        return out2
    return hook_fn


def _ablation_hook_many_positions(k_list: list[int], i_list: list[int]):
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3 or out.shape[0] != 1:
            return out
        if not k_list:
            return out
        out2 = out.clone()
        k = torch.tensor(k_list, device=out2.device, dtype=torch.long)
        i = torch.tensor(i_list, device=out2.device, dtype=torch.long)
        out2[0, k, i] = 0
        return out2
    return hook_fn


def _set_hook_many_positions(k_list: list[int], i_list: list[int], values: list[float]):
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3 or out.shape[0] != 1:
            return out
        if not k_list:
            return out
        out2 = out.clone()
        k_t = torch.tensor(k_list, device=out2.device, dtype=torch.long)
        i_t = torch.tensor(i_list, device=out2.device, dtype=torch.long)
        v_t = torch.tensor(values, device=out2.device, dtype=out2.dtype)
        out2[0, k_t, i_t] = v_t
        return out2
    return hook_fn


def _ablation_hook_many_positions_norm_match(
    k_list: list[int],
    i_list: list[int],
    base_norm_by_k: torch.Tensor,
):
    grouped: dict[int, list[int]] = {}
    for k, i in zip(k_list, i_list):
        grouped.setdefault(int(k), []).append(int(i))

    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3 or out.shape[0] != 1:
            return out
        if not k_list:
            return out
        out2 = out.clone()
        k_t = torch.tensor(k_list, device=out2.device, dtype=torch.long)
        i_t = torch.tensor(i_list, device=out2.device, dtype=torch.long)
        out2[0, k_t, i_t] = 0
        for k, idxs in grouped.items():
            idx_t = torch.tensor(idxs, device=out2.device, dtype=torch.long)
            v = out2[0, k, :]
            cur = torch.linalg.norm(v)
            tgt = base_norm_by_k[int(k)].to(device=out2.device, dtype=out2.dtype)
            extra2 = torch.clamp(tgt * tgt - cur * cur, min=0.0)
            extra = torch.sqrt(extra2)
            if float(extra.item()) <= 0.0:
                continue
            if idx_t.numel() == 1:
                out2[0, k, idx_t[0]] = extra
            else:
                noise = torch.rand((idx_t.numel(),), device=out2.device, dtype=out2.dtype)
                denom = torch.linalg.norm(noise).clamp_min(1e-12)
                noise = noise / denom
                out2[0, k, idx_t] = noise * extra
        return out2
    return hook_fn


# ------------------------------------------------------------------------------
# Ablation hooks for whole‑token ablation (zero entire embedding vector)
# ------------------------------------------------------------------------------
def _ablation_hook_embedding_whole_token_single_positions(k_tensor: torch.Tensor):
    """
    Batched ablation: zero the entire embedding vector at token positions k_tensor.
    """
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3:
            return out
        out2 = out.clone()
        b = out2.shape[0]
        idx = torch.arange(b, device=out2.device)
        out2[idx, k_tensor, :] = 0
        return out2
    return hook_fn


def _ablation_hook_embedding_whole_token_many_positions(k_list: list[int]):
    """
    Single‑batch ablation: zero the entire embedding vector at multiple token positions.
    """
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3 or out.shape[0] != 1:
            return out
        if not k_list:
            return out
        out2 = out.clone()
        k = torch.tensor(k_list, device=out2.device, dtype=torch.long)
        out2[0, k, :] = 0
        return out2
    return hook_fn


# ------------------------------------------------------------------------------
# Ablation hooks for whole‑token replacement with random vector (same norm)
# ------------------------------------------------------------------------------
def _ablation_hook_embedding_whole_token_single_positions_random_norm_match(
    k_tensor: torch.Tensor,
    base_norm_by_k: torch.Tensor,
):
    """
    Batched replacement: for each batch element b, replace out[b, k_tensor[b], :]
    with a random vector whose L2 norm matches base_norm_by_k[k_tensor[b]].
    """
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3:
            return out
        out2 = out.clone()
        bsz, _t, d = out2.shape
        idx = torch.arange(bsz, device=out2.device)
        ks = k_tensor.to(device=out2.device, dtype=torch.long)
        tgt = base_norm_by_k[ks].to(device=out2.device, dtype=out2.dtype)  # [bsz]
        noise = torch.randn((bsz, d), device=out2.device, dtype=out2.dtype)
        noise_norm = torch.linalg.norm(noise, dim=-1).clamp_min(1e-12)
        noise_unit = noise / noise_norm[:, None]
        out2[idx, ks, :] = noise_unit * tgt[:, None]
        return out2
    return hook_fn


def _ablation_hook_embedding_whole_token_many_positions_random_norm_match(
    k_list: list[int],
    base_norm_by_k: torch.Tensor,
):
    """
    Single-batch replacement (batch=1): for each k in k_list, replace out[0, k, :]
    with a random vector whose L2 norm matches base_norm_by_k[k].
    """
    def hook_fn(_module, _inp, out: torch.Tensor):
        if out.ndim != 3 or out.shape[0] != 1:
            return out
        if not k_list:
            return out
        out2 = out.clone()
        d = out2.shape[-1]
        for k in k_list:
            kk = int(k)
            tgt = base_norm_by_k[kk].to(device=out2.device, dtype=out2.dtype)
            noise = torch.randn((d,), device=out2.device, dtype=out2.dtype)
            noise_norm = torch.linalg.norm(noise).clamp_min(1e-12)
            out2[0, kk, :] = noise * (tgt / noise_norm)
        return out2
    return hook_fn


# ------------------------------------------------------------------------------
# Baseline collection
# ------------------------------------------------------------------------------
@torch.no_grad()
def collect_baseline(
    hooked_model,
    tokens_t: torch.Tensor,
) -> dict[str, Any]:
    logits = hooked_model(tokens_t)
    b, t, v = logits.shape
    if b != 1:
        raise ValueError("baseline expects batch=1")
    target_id = int(torch.argmax(logits[0, -1]).item())
    logp0 = float(_logprob_target_at_final(logits, target_id)[0].item())

    n_layer = int(hooked_model.model.cfg.n_layer)
    acts_by_layer: dict[int, torch.Tensor] = {}
    
    # Embeddings (layer -1)
    emb_act = hooked_model.get_activations("transformer.wte.hook_embed")
    if emb_act is not None:
        acts_by_layer[-1] = emb_act.detach()
    else:
        # Some HookedModel variants may not expose the embedding hook activation name.
        # We still need embedding norms for token-level nodes; compute directly from wte.
        try:
            emb_direct = _get_wte_module(hooked_model)(tokens_t)
            if emb_direct is not None:
                acts_by_layer[-1] = emb_direct.detach()
        except Exception:
            pass
    
    for L in range(n_layer):
        hook_name = f"blocks.{L}.mlp.hook_post"
        act = hooked_model.get_activations(hook_name)
        if act is None:
            raise RuntimeError(f"missing activation for {hook_name}")
        acts_by_layer[L] = act.detach()

    base_norms_by_layer: dict[int, torch.Tensor] = {}
    for L, act in acts_by_layer.items():
        base_norms_by_layer[int(L)] = torch.linalg.norm(act[0], dim=-1).detach()

    return {
        "logits": logits.detach(),
        "target_id": target_id,
        "logp0": logp0,
        "acts_by_layer": acts_by_layer,
        "base_norms_by_layer": base_norms_by_layer,
    }


# ------------------------------------------------------------------------------
# Candidate extraction
# ------------------------------------------------------------------------------
def _extract_candidates_all_positions(
    acts_by_layer: dict[int, torch.Tensor],
    base_norms_by_layer: dict[int, torch.Tensor] | None = None,
) -> list[Candidate]:
    """
    Extract non‑zero MLP neurons only.
    
    Embedding vectors are NOT extracted as candidates or plotted as nodes.
    """
    cands: list[Candidate] = []
    # MLP neurons (layers 0..n-1): extract sparse non-zeros
    for L, act in acts_by_layer.items():
        if int(L) < 0:
            continue
        a = act[0]
        nz = (a != 0).nonzero(as_tuple=False)
        for row in nz.tolist():
            k, i = int(row[0]), int(row[1])
            val = float(a[k, i].item())
            cands.append(Candidate(k=k, layer=int(L), neuron=i, activation=val, C=None))

    return cands


# ------------------------------------------------------------------------------
# Compute C values (batched by layer)
# ------------------------------------------------------------------------------
@torch.no_grad()
def compute_C_batched_by_layer(
    hooked_model,
    tokens_t: torch.Tensor,
    candidates: list[Candidate],
    target_id: int,
    logp0: float,
    device: str,
    base_norms_by_layer: dict[int, torch.Tensor] | None = None,
    norm_match: bool = False,
    num_repeats: int = 1,
    show_progress: bool = True,
) -> list[Candidate]:
    """
    Compute C values for MLP neuron candidates only.
    """
    if not candidates:
        return []
    n_layer = int(hooked_model.model.cfg.n_layer)
    # Group by layer: 0..n_layer-1 (MLP only)
    by_layer: dict[int, list[Candidate]] = {L: [] for L in range(n_layer)}
    for c in candidates:
        by_layer[int(c.layer)].append(c)

    out: list[Candidate] = []
    logp_accumulator: dict[tuple[int, int, int], list[float]] = {}

    repeat_iter = tqdm(
        range(num_repeats),
        desc="C-values repeats",
        disable=(not show_progress) or num_repeats <= 1,
        leave=False,
    )
    for repeat_idx in repeat_iter:
        # Ensure deterministic averaging when randomness is involved (norm-match).
        if num_repeats > 1 and norm_match:
            torch.manual_seed(12345 + repeat_idx)
            np.random.seed(12345 + repeat_idx)
        
        # ---- MLP layers (0..n_layer-1) ----
        layer_iter = tqdm(
            range(n_layer),
            desc=f"C-values layers (repeat {repeat_idx + 1}/{num_repeats})",
            disable=not show_progress,
            leave=False,
        )
        for L in layer_iter:
            layer_cands = by_layer.get(L, [])
            if not layer_cands:
                continue
            ks = torch.tensor([c.k for c in layer_cands], device=device, dtype=torch.long)
            is_ = torch.tensor([c.neuron for c in layer_cands], device=device, dtype=torch.long)
            tok_batch = tokens_t.repeat(len(layer_cands), 1)
            module = hooked_model.model.blocks[L].mlp.act
            if norm_match and base_norms_by_layer is not None:
                base_norm_by_k = base_norms_by_layer[int(L)].to(device=device)
                hook = _ablation_hook_single_positions_norm_match(ks, is_, base_norm_by_k)
            else:
                hook = _ablation_hook_single_positions(ks, is_)
            handle = _register_forward_hook_prepend(module, hook)
            try:
                logits = hooked_model(tok_batch)
                logp = _logprob_target_at_final(logits, target_id)
            finally:
                handle.remove()
            logp_vals = logp.detach().cpu().numpy().astype(np.float64)
            for c, logp_val in zip(layer_cands, logp_vals.tolist()):
                key = (int(c.k), int(c.layer), int(c.neuron))
                logp_accumulator.setdefault(key, []).append(float(logp_val))

    # Average logp and compute C
    for key, logp_list in logp_accumulator.items():
        k, L, i = key
        cand = next((c for c in candidates if int(c.k) == k and int(c.layer) == L and int(c.neuron) == i), None)
        if cand is not None:
            p_values = np.exp(np.array(logp_list))
            mean_p = float(np.mean(p_values))
            avg_logp = float(np.log(mean_p)) if mean_p > 0 else -1e30
            C_avg = float(np.exp(avg_logp - float(logp0)))
            out.append(Candidate(k=cand.k, layer=cand.layer, neuron=cand.neuron, activation=cand.activation, C=C_avg))

    out.sort(key=lambda x: float("inf") if x.C is None else x.C)
    return out


# ------------------------------------------------------------------------------
# logp with arbitrary ablation sets (supports TOKEN_LAYER)
# ------------------------------------------------------------------------------
@torch.no_grad()
def logp_with_ablate_sets(
    hooked_model,
    tokens_t: torch.Tensor,
    target_id: int,
    ablate_by_layer: dict[int, list[tuple[int, int]]],
    base_norms_by_layer: dict[int, torch.Tensor] | None = None,
    norm_match: bool = False,
    num_repeats: int = 1,
) -> float:
    logp_values: list[float] = []

    for repeat_idx in range(num_repeats):
        # Ensure deterministic averaging when randomness is involved (norm-match).
        if num_repeats > 1 and norm_match:
            torch.manual_seed(12345 + repeat_idx)
            np.random.seed(12345 + repeat_idx)
        
        handles = []
        try:
            for L, pairs in ablate_by_layer.items():
                if not pairs:
                    continue
                ks = [int(k) for k, _ in pairs]
                is_ = [int(i) for _, i in pairs]
                
                # MLP layers only (no embedding layer -1)
                module = hooked_model.model.blocks[L].mlp.act
                if norm_match and base_norms_by_layer is not None:
                    base_norm_by_k = base_norms_by_layer[int(L)].to(device=tokens_t.device)
                    hook = _ablation_hook_many_positions_norm_match(ks, is_, base_norm_by_k)
                else:
                    hook = _ablation_hook_many_positions(ks, is_)
                handles.append(_register_forward_hook_prepend(module, hook))
            logits = hooked_model(tokens_t)
            logp = float(_logprob_target_at_final(logits, target_id)[0].item())
            logp_values.append(logp)
        finally:
            for h in handles:
                try:
                    h.remove()
                except Exception:
                    pass
    
    if logp_values:
        p_values = np.exp(np.array(logp_values))
        mean_p = float(np.mean(p_values))
        return float(np.log(mean_p)) if mean_p > 0 else 0.0
    else:
        return 0.0


# ------------------------------------------------------------------------------
# Circuit selection (add_back / delete_down)
# ------------------------------------------------------------------------------
def select_circuit_add_back(
    hooked_model,
    tokens_t: torch.Tensor,
    ranked: list[Candidate],
    target_id: int,
    logp0: float,
    retain_ratio: float,
    base_norms_by_layer: dict[int, torch.Tensor] | None = None,
    norm_match: bool = False,
    max_steps: int | None = None,
    num_repeats: int = 1,
    show_progress: bool = True,
) -> tuple[list[Candidate], float, list[dict[str, float]]]:
    ablate_by_layer: dict[int, dict[tuple[int, int], Candidate]] = {}
    add_iter = tqdm(ranked, desc="Greedy add-back", disable=not show_progress, leave=False)
    for c in add_iter:
        ablate_by_layer.setdefault(int(c.layer), {})[(int(c.k), int(c.neuron))] = c

    def current_ablate_lists() -> dict[int, list[tuple[int, int]]]:
        out: dict[int, list[tuple[int, int]]] = {}
        for L, m in ablate_by_layer.items():
            out[L] = list(m.keys())
        return out

    curve: list[dict[str, float]] = []

    logp_all_ablated = logp_with_ablate_sets(
        hooked_model,
        tokens_t,
        target_id,
        current_ablate_lists(),
        base_norms_by_layer=base_norms_by_layer,
        norm_match=norm_match,
        num_repeats=num_repeats,
    )
    ratio = math.exp(logp_all_ablated - logp0) if logp0 > -1e30 else 0.0
    curve.append({"n_added": 0.0, "retain_ratio": float(ratio)})

    selected: list[Candidate] = []
    if ratio >= retain_ratio:
        return selected, float(ratio), curve

    steps = 0
    for c in ranked:
        if max_steps is not None and steps >= int(max_steps):
            break
        key = (int(c.k), int(c.neuron))
        layer_map = ablate_by_layer.get(int(c.layer), {})
        if key in layer_map:
            selected.append(layer_map.pop(key))
        steps += 1
        logp_cur = logp_with_ablate_sets(
            hooked_model,
            tokens_t,
            target_id,
            current_ablate_lists(),
            base_norms_by_layer=base_norms_by_layer,
            norm_match=norm_match,
            num_repeats=num_repeats,
        )
        ratio = math.exp(logp_cur - logp0)
        curve.append({"n_added": float(len(selected)), "retain_ratio": float(ratio)})
        if ratio >= retain_ratio:
            break

    return selected, float(ratio), curve


def select_circuit_delete_down(
    hooked_model,
    tokens_t: torch.Tensor,
    ranked: list[Candidate],
    target_id: int,
    logp0: float,
    retain_ratio: float,
    base_norms_by_layer: dict[int, torch.Tensor] | None = None,
    norm_match: bool = False,
    max_steps: int | None = None,
    num_repeats: int = 1,
    show_progress: bool = True,
) -> tuple[list[Candidate], float, list[dict[str, float]]]:
    delete_order = list(reversed(ranked))

    curve: list[dict[str, float]] = []
    curve.append({"n_deleted": 0.0, "n_kept": float(len(ranked)), "retain_ratio": 1.0})

    ablate_by_layer: dict[int, list[tuple[int, int]]] = {}
    kept: list[Candidate] = list(ranked)
    last_ok_kept = list(kept)
    last_ok_ratio = 1.0

    steps = 0
    del_iter = tqdm(delete_order, desc="Greedy delete-down", disable=not show_progress, leave=False)
    for c in del_iter:
        if max_steps is not None and steps >= int(max_steps):
            break
        steps += 1
        ablate_by_layer.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
        logp_cur = logp_with_ablate_sets(
            hooked_model,
            tokens_t,
            target_id,
            ablate_by_layer,
            base_norms_by_layer=base_norms_by_layer,
            norm_match=norm_match,
            num_repeats=num_repeats,
        )
        ratio = math.exp(logp_cur - logp0)
        kept = [x for x in kept if not (x.k == c.k and x.layer == c.layer and x.neuron == c.neuron)]
        n_deleted_total = sum(len(v) for v in ablate_by_layer.values())
        curve.append({"n_deleted": float(n_deleted_total), "n_kept": float(len(kept)), "retain_ratio": float(ratio)})

        if ratio >= retain_ratio:
            last_ok_kept = list(kept)
            last_ok_ratio = float(ratio)
        else:
            break

    return last_ok_kept, float(last_ok_ratio), curve


# ------------------------------------------------------------------------------
# Faithfulness curve – uses the actual circuits discovered for each n
# ------------------------------------------------------------------------------
def get_circuit_ablation_for_n(
    n: int,
    ranked: list[Candidate],
    selection_mode: str,
    selected_superset: list[Candidate] | None = None,
    beam_sets_by_n: list[list[tuple[int, int, int]]] | None = None,
) -> dict[int, list[tuple[int, int]]]:
    """Return ablation dict (layer → list of (k,i)) for the circuit of size n."""
    if n == 0:
        return {}
    ablate: dict[int, list[tuple[int, int]]] = {}

    # Beam search: use the best set found for size n
    if beam_sets_by_n is not None and n < len(beam_sets_by_n):
        nodes = beam_sets_by_n[n]
        for k, L, i in nodes:
            ablate.setdefault(int(L), []).append((int(k), int(i)))
        return ablate

    # Greedy add_back: circuit = first n nodes of the ordered superset
    if selection_mode == "add_back" and selected_superset is not None:
        for c in selected_superset[:n]:
            ablate.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
        return ablate

    # Greedy delete_down: circuit = top n nodes by C (last n of ranked)
    if selection_mode == "delete_down":
        for c in ranked[-n:]:
            ablate.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
        return ablate

    # Fallback: empty set
    return {}


@torch.no_grad()
def compute_faithfulness_curve_from_circuits(
    hooked_model,
    tokens_t: torch.Tensor,
    target_id: int,
    logp0: float,
    base_norms_by_layer: dict[int, torch.Tensor] | None,
    norm_match: bool,
    num_repeats: int,
    circuit_dict: dict[int, dict[int, list[tuple[int, int]]]],
    show_progress: bool = True,
) -> list[dict[str, float]]:
    """
    For each n where circuit_dict[n] is non‑empty, compute p/p0 when that circuit is ablated.
    Returns list of {n: int, faithfulness: float} sorted by n.
    """
    curve = []
    circuit_sizes = sorted(circuit_dict.keys())
    for n in tqdm(circuit_sizes, desc="Faithfulness curve", disable=not show_progress, leave=False):
        ablate_set = circuit_dict[n]
        logp = logp_with_ablate_sets(
            hooked_model,
            tokens_t,
            target_id,
            ablate_set,
            base_norms_by_layer=base_norms_by_layer,
            norm_match=norm_match,
            num_repeats=num_repeats,
        )
        faith = math.exp(logp - logp0) if logp0 > -1e30 else 0.0
        curve.append({"n": n, "faithfulness": faith})
    return curve


# ------------------------------------------------------------------------------
# Influence edges – adapted to handle token nodes
# ------------------------------------------------------------------------------
@torch.no_grad()
def compute_influence_edges(
    hooked_model,
    tokens_t: torch.Tensor,
    selected: list[Candidate],
    baseline_acts_by_layer: dict[int, torch.Tensor],
    base_norms_by_layer: dict[int, torch.Tensor] | None,
    target_id: int,
    logp0: float,
    max_neighbors: int,
    norm_match: bool = False,
    num_repeats: int = 1,
) -> dict[str, Any]:
    """
    Compute influence edges for MLP neurons only.
    """
    selected_nodes = [(int(c.k), int(c.layer), int(c.neuron)) for c in selected]
    idx = {node: j for j, node in enumerate(selected_nodes)}

    # Baseline scalar per node activation
    base_act = np.zeros(len(selected_nodes), dtype=np.float64)
    for node, j in idx.items():
        k, L, i = node
        base_act[j] = float(baseline_acts_by_layer[L][0, k, i].item())

    n_layer = int(hooked_model.model.cfg.n_layer)
    deltas_acc = np.zeros((len(selected_nodes), len(selected_nodes)), dtype=np.float64)
    rel_changes_acc = np.zeros((len(selected_nodes), len(selected_nodes)), dtype=np.float64)
    logp_all = np.zeros((num_repeats, len(selected_nodes)), dtype=np.float64)

    for repeat_idx in range(num_repeats):
        if norm_match and num_repeats > 1:
            torch.manual_seed(12345 + repeat_idx)
            np.random.seed(12345 + repeat_idx)

        deltas = np.zeros((len(selected_nodes), len(selected_nodes)), dtype=np.float64)
        rel_changes = np.zeros((len(selected_nodes), len(selected_nodes)), dtype=np.float64)

        for u_idx, (k_u, L_u, i_u) in enumerate(selected_nodes):
            # ---- Ablation hook for MLP neuron source node u ----
            module = hooked_model.model.blocks[L_u].mlp.act
            if norm_match and base_norms_by_layer is not None:
                base_norm_by_k = base_norms_by_layer[int(L_u)].to(device=tokens_t.device)
                hook = _ablation_hook_many_positions_norm_match([int(k_u)], [int(i_u)], base_norm_by_k)
            else:
                hook = _ablation_hook_many_positions([int(k_u)], [int(i_u)])
            handle = _register_forward_hook_prepend(module, hook)
            try:
                logits = hooked_model(tokens_t)
                logp_u = float(_logprob_target_at_final(logits, target_id)[0].item())
                logp_all[repeat_idx, u_idx] = logp_u

                # Collect post‑ablation activations for all layers
                acts_new: dict[int, torch.Tensor] = {}
                emb_act = hooked_model.get_activations("transformer.wte.hook_embed")
                if emb_act is not None:
                    acts_new[-1] = emb_act.detach()
                for L in range(n_layer):
                    hook_name = f"blocks.{L}.mlp.hook_post"
                    act = hooked_model.get_activations(hook_name)
                    if act is None:
                        raise RuntimeError(f"missing activation for {hook_name} during influence run")
                    acts_new[L] = act.detach()

                # Compute Δ for every target MLP node v
                for v_node, v_idx in idx.items():
                    k_v, L_v, i_v = v_node
                    if L_v in acts_new:
                        new_val = float(acts_new[L_v][0, k_v, i_v].item())
                        delta = new_val - base_act[v_idx]
                        deltas[u_idx, v_idx] = delta
                        denom = base_act[v_idx]
                        if denom == 0.0:
                            rel_changes[u_idx, v_idx] = 0.0
                        else:
                            rel_changes[u_idx, v_idx] = delta / denom
            finally:
                handle.remove()

        deltas_acc += deltas
        rel_changes_acc += rel_changes

    deltas = deltas_acc / num_repeats
    rel_changes = rel_changes_acc / num_repeats

    # Average logp deltas
    logp_deltas = np.zeros(len(selected_nodes), dtype=np.float64)
    for u_idx in range(len(selected_nodes)):
        p_values = np.exp(logp_all[:, u_idx])
        mean_p = float(np.mean(p_values))
        avg_logp = float(np.log(mean_p)) if mean_p > 0 else -1e30
        logp_deltas[u_idx] = avg_logp - logp0

    # Build outgoing edges (only from nodes that have non‑zero influence)
    edges_nn: list[dict[str, Any]] = []
    for u in range(len(selected_nodes)):
        weights = np.abs(rel_changes[u, :])
        weights[u] = -1.0
        top_idx = np.argsort(-weights)[: max_neighbors]
        for v in top_idx.tolist():
            w = float(weights[v])
            if w <= 0:
                continue
            edges_nn.append({
                "source": u,
                "target": int(v),
                "weight": w,
                "delta": float(deltas[u, v]),
                "rel_change": float(rel_changes[u, v]),
                "kind": "nn",
            })

    return {
        "selected_nodes": selected_nodes,
        "base_act": base_act.tolist(),
        "logp_deltas": logp_deltas.tolist(),
        "edges_nn": edges_nn,
    }

# "Jack was thirsty, so he went looking for something to"
# ------------------------------------------------------------------------------
# Detailed circuit generation (removed - now uses attribution_graph_html.py functions)
# ------------------------------------------------------------------------------


# ------------------------------------------------------------------------------
# Detailed circuit computation
# ------------------------------------------------------------------------------
def select_circuit_nodes(
    ranked: list[Candidate],
    mode: str,
    k: int | None = None,
    threshold_t: float | None = None,
    hooked_model = None,
    tokens_t: torch.Tensor = None,
    target_id: int = None,
    logp0: float = None,
    base_norms_by_layer: dict[int, torch.Tensor] | None = None,
    device: str = "cuda",
    norm_match: bool = False,
    num_repeats: int = 1,
) -> list[Candidate]:
    """
    Select circuit nodes: either fixed K or minimum K for retain_ratio >= threshold_t.
    """
    if mode == "fixed_k":
        if k is None:
            raise ValueError("k must be specified for mode='fixed_k'")
        return ranked[:min(k, len(ranked))]
    
    elif mode == "threshold_t":
        if threshold_t is None or hooked_model is None or tokens_t is None:
            raise ValueError("threshold_t, hooked_model, tokens_t required for mode='threshold_t'")
        
        # Binary search / incremental to find minimum K where retain_ratio >= threshold_t
        # for size in range(1, len(ranked) + 1):
        #     circuit_nodes = ranked[:size]
        #     ablate_by_layer: dict[int, list[tuple[int, int]]] = {}
        #     for c in circuit_nodes:
        #         ablate_by_layer.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
        #     logp = logp_with_ablate_sets(
        #         hooked_model,
        #         tokens_t,
        #         target_id,
        #         ablate_by_layer,
        #         base_norms_by_layer=base_norms_by_layer,
        #         norm_match=norm_match,
        #         num_repeats=num_repeats,
        #     )
        #     ratio = math.exp(logp - logp0) if logp0 > -1e30 else 0.0
        #     if ratio >= threshold_t:
        #         return circuit_nodes

        for size in range(1, len(ranked) + 1):
            keep_nodes = ranked[:size]
            # Ablate all other nodes (the complement)
            ablate_by_layer = {}
            for c in ranked[size:]:                            # nodes NOT kept
                ablate_by_layer.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
            logp = logp_with_ablate_sets(
                hooked_model,
                tokens_t,
                target_id,
                ablate_by_layer,
                base_norms_by_layer=base_norms_by_layer,
                norm_match=norm_match,
                num_repeats=num_repeats,
            )
            ratio = math.exp(logp - logp0) if logp0 > -1e30 else 0.0
            if ratio >= threshold_t:
                return keep_nodes
        
        # If threshold never reached, return all
        return ranked
    
    else:
        raise ValueError(f"Unknown circuit selection mode: {mode}")


@torch.no_grad()
def compute_localized_C_values(
    hooked_model,
    tokens_t: torch.Tensor,
    circuit_nodes: list[Candidate],
    target_id: int,
    logp0: float,
    device: str,
    base_norms_by_layer: dict[int, torch.Tensor] | None = None,
    norm_match: bool = False,
    num_repeats: int = 1,
) -> list[Candidate]:
    """
    Recalculate C values for circuit nodes with all OTHER circuit nodes ablated.
    This measures the isolated importance of each node within the circuit.
    """
    if not circuit_nodes:
        return []
    
    circuit_set = set((int(c.k), int(c.layer), int(c.neuron)) for c in circuit_nodes)
    n_layer = int(hooked_model.model.cfg.n_layer)
    
    out: list[Candidate] = []
    logp_accumulator: dict[tuple[int, int, int], list[float]] = {}
    
    for repeat_idx in range(num_repeats):
        if num_repeats > 1 and norm_match:
            torch.manual_seed(12345 + repeat_idx)
            np.random.seed(12345 + repeat_idx)
        
        for L in range(n_layer):
            layer_nodes = [c for c in circuit_nodes if int(c.layer) == L]
            if not layer_nodes:
                continue
            
            # For each node in this layer, ablate all OTHER circuit nodes
            for target_node in layer_nodes:
                # Build ablation set: all circuit nodes EXCEPT this one
                ablate_by_layer: dict[int, list[tuple[int, int]]] = {}
                for c in circuit_nodes:
                    if c.k == target_node.k and c.layer == target_node.layer and c.neuron == target_node.neuron:
                        continue  # Skip the target node
                    ablate_by_layer.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
                
                logp = logp_with_ablate_sets(
                    hooked_model,
                    tokens_t,
                    target_id,
                    ablate_by_layer,
                    base_norms_by_layer=base_norms_by_layer,
                    norm_match=norm_match,
                    num_repeats=1,
                )
                
                key = (int(target_node.k), int(target_node.layer), int(target_node.neuron))
                logp_accumulator.setdefault(key, []).append(logp)
    
    # Compute localized C from averaged logp
    for key, logp_list in logp_accumulator.items():
        k, L, i = key
        cand = next((c for c in circuit_nodes if int(c.k) == k and int(c.layer) == L and int(c.neuron) == i), None)
        if cand is not None:
            p_values = np.exp(np.array(logp_list))
            mean_p = float(np.mean(p_values))
            avg_logp = float(np.log(mean_p)) if mean_p > 0 else -1e30
            C_local = float(np.exp(avg_logp - float(logp0)))
            out.append(Candidate(k=cand.k, layer=cand.layer, neuron=cand.neuron, activation=cand.activation, C=C_local))
    
    out.sort(key=lambda x: float("inf") if x.C is None else x.C)
    return out


@torch.no_grad()
def compute_detailed_circuit_edges(
    hooked_model,
    tokens_t: torch.Tensor,
    circuit_nodes: list[Candidate],
    baseline_acts_by_layer: dict[int, torch.Tensor],
    base_norms_by_layer: dict[int, torch.Tensor] | None,
    target_id: int,
    logp0: float,
    device: str,
    token_strings: list[str] | None = None,
    norm_match: bool = False,
    num_repeats: int = 1,
    circuit_C_values: list[Candidate] | None = None,
) -> dict[str, Any]:
    """
    Compute detailed influence edges within the circuit.
    For each node, find top 10 nodes it most influences and top 10 nodes that most influence it.
    All measurements taken with the rest of the circuit active (baseline).
    """
    if not circuit_nodes:
        return {"nodes": [], "edges": [], "per_node": {}}
    
    circuit_nodes_set = [(int(c.k), int(c.layer), int(c.neuron)) for c in circuit_nodes]
    idx = {node: j for j, node in enumerate(circuit_nodes_set)}
    n_layer = int(hooked_model.model.cfg.n_layer)
    
    # Baseline activations for circuit nodes
    base_act = np.zeros(len(circuit_nodes), dtype=np.float64)
    for node, j in idx.items():
        k, L, i = node
        base_act[j] = float(baseline_acts_by_layer[L][0, k, i].item())
    
    # Compute how each node affects others
    deltas = np.zeros((len(circuit_nodes), len(circuit_nodes)), dtype=np.float64)
    rel_changes = np.zeros((len(circuit_nodes), len(circuit_nodes)), dtype=np.float64)
    
    for repeat_idx in range(num_repeats):
        if norm_match and num_repeats > 1:
            torch.manual_seed(12345 + repeat_idx)
            np.random.seed(12345 + repeat_idx)
        
        for u_idx, (k_u, L_u, i_u) in enumerate(circuit_nodes_set):
            # Ablate source node u
            module = hooked_model.model.blocks[L_u].mlp.act
            if norm_match and base_norms_by_layer is not None:
                base_norm_by_k = base_norms_by_layer[int(L_u)].to(device=device)
                hook = _ablation_hook_many_positions_norm_match([int(k_u)], [int(i_u)], base_norm_by_k)
            else:
                hook = _ablation_hook_many_positions([int(k_u)], [int(i_u)])
            handle = _register_forward_hook_prepend(module, hook)
            
            try:
                logits = hooked_model(tokens_t)
                
                # Collect activations for all circuit nodes
                acts_new: dict[int, torch.Tensor] = {}
                for L in range(n_layer):
                    hook_name = f"blocks.{L}.mlp.hook_post"
                    act = hooked_model.get_activations(hook_name)
                    if act is not None:
                        acts_new[L] = act.detach()
                
                # Measure deltas for every target node v in circuit
                for v_node, v_idx in idx.items():
                    k_v, L_v, i_v = v_node
                    if L_v in acts_new:
                        new_val = float(acts_new[L_v][0, k_v, i_v].item())
                        delta = new_val - base_act[v_idx]
                        deltas[u_idx, v_idx] += delta / num_repeats
                        denom = base_act[v_idx]
                        if denom != 0.0:
                            rel_changes[u_idx, v_idx] += (delta / denom) / num_repeats
            finally:
                handle.remove()
    
    # Build detailed per-node data
    # Create a map of (k,L,i) -> C value for quick lookup
    C_map: dict[tuple[int, int, int], float] = {}
    if circuit_C_values is not None:
        for c in circuit_C_values:
            key = (int(c.k), int(c.layer), int(c.neuron))
            C_map[key] = float(c.C) if c.C is not None else float('inf')
    
    per_node_data = {}
    for u_idx, (k_u, L_u, i_u) in enumerate(circuit_nodes_set):
        node_id = f"{L_u}_{k_u}_{i_u}"
        
        # Get C value for this node
        node_C = C_map.get((k_u, L_u, i_u), float('inf'))
        
        # Top 10 nodes this one influences (highest rel_change)
        influences_out = []
        for v_idx in np.argsort(-np.abs(rel_changes[u_idx, :]))[:10]:
            if rel_changes[u_idx, v_idx] != 0:
                k_v, L_v, i_v = circuit_nodes_set[v_idx]
                influences_out.append({
                    "target": f"{L_v}_{k_v}_{i_v}",
                    "rel_change": float(rel_changes[u_idx, v_idx]),
                    "delta": float(deltas[u_idx, v_idx]),
                })
        
        # Top 10 nodes that influence this one (highest rel_change TO this node)
        influences_in = []
        for u_idx_in in np.argsort(-np.abs(rel_changes[:, u_idx]))[:10]:
            if rel_changes[u_idx_in, u_idx] != 0:
                k_s, L_s, i_s = circuit_nodes_set[u_idx_in]
                influences_in.append({
                    "source": f"{L_s}_{k_s}_{i_s}",
                    "rel_change": float(rel_changes[u_idx_in, u_idx]),
                    "delta": float(deltas[u_idx_in, u_idx]),
                })
        
        per_node_data[node_id] = {
            "k": int(k_u),
            "L": int(L_u),
            "i": int(i_u),
            "C": float(node_C),
            "base_activation": float(base_act[u_idx]),
            "influences_out": influences_out,
            "influences_in": influences_in,
        }
    
    # Compute baseline output distribution (with all circuit nodes active)
    baseline_logits = hooked_model(tokens_t)
    baseline_top_k = _get_top_k_tokens(baseline_logits, k=10)
    
    baseline_logp = torch.log_softmax(baseline_logits[0, -1, :], dim=-1).cpu().numpy()

    def _collect_ablation_stats_for_nodes(
        nodes_src: list[tuple[int, int, int]],
        ablation_kind: str,
    ) -> dict[str, Any]:
        grouped: dict[int, tuple[list[int], list[int]]] = {}
        for k_pos, L_pos, i_pos in nodes_src:
            ks, is_ = grouped.setdefault(int(L_pos), ([], []))
            ks.append(int(k_pos))
            is_.append(int(i_pos))

        handles = []
        try:
            for L_src, (ks_src, is_src) in grouped.items():
                module = hooked_model.model.blocks[int(L_src)].mlp.act
                if ablation_kind == "set_to_0":
                    if norm_match and base_norms_by_layer is not None:
                        base_norm_by_k = base_norms_by_layer[int(L_src)].to(device=device)
                        hook = _ablation_hook_many_positions_norm_match(ks_src, is_src, base_norm_by_k)
                    else:
                        hook = _ablation_hook_many_positions(ks_src, is_src)
                elif ablation_kind in ("set_to_-5x", "set_to_-10x"):
                    scale = -5.0 if ablation_kind == "set_to_-5x" else -10.0
                    values = []
                    for k_pos, i_pos in zip(ks_src, is_src):
                        x0 = float(baseline_acts_by_layer[int(L_src)][0, int(k_pos), int(i_pos)].item())
                        values.append(scale * x0)
                    hook = _set_hook_many_positions(ks_src, is_src, values)
                else:
                    raise ValueError(f"Unknown ablation kind: {ablation_kind}")
                handles.append(_register_forward_hook_prepend(module, hook))

            ablated_logits = hooked_model(tokens_t)
        finally:
            for handle in handles:
                handle.remove()

        ablated_top_k = _get_top_k_tokens(ablated_logits, k=10)
        ablated_logp = torch.log_softmax(ablated_logits[0, -1, :], dim=-1).cpu().numpy()
        logp_diffs = ablated_logp - baseline_logp  # negative = decreased prob

        top_decreased = np.argsort(logp_diffs)[:10]
        decreased_tokens = []
        for token_id in top_decreased:
            token_id = int(token_id)
            token_str = _decode_tokens([token_id])[0]
            decreased_tokens.append({
                "token_id": token_id,
                "token_str": token_str,
                "logp_change": float(logp_diffs[token_id]),
                "baseline_logp": float(baseline_logp[token_id]),
                "ablated_logp": float(ablated_logp[token_id]),
            })

        top_increased = np.argsort(-logp_diffs)[:10]
        increased_tokens = []
        for token_id in top_increased:
            token_id = int(token_id)
            token_str = _decode_tokens([token_id])[0]
            increased_tokens.append({
                "token_id": token_id,
                "token_str": token_str,
                "logp_change": float(logp_diffs[token_id]),
                "baseline_logp": float(baseline_logp[token_id]),
                "ablated_logp": float(ablated_logp[token_id]),
            })

        return {
            "top_k_ablated": ablated_top_k,
            "top_decreased_logp": decreased_tokens,
            "top_increased_logp": increased_tokens,
        }

    # Compute per-node ablation effects on output distribution
    per_node_ablation_dist = {}
    for u_idx, (k_u, L_u, i_u) in enumerate(circuit_nodes_set):
        node_id = f"{L_u}_{k_u}_{i_u}"
        node_group = [(int(k_u), int(L_u), int(i_u))]
        per_node_ablation_dist[node_id] = {
            "set_to_0": _collect_ablation_stats_for_nodes(node_group, "set_to_0"),
            "set_to_-5x": _collect_ablation_stats_for_nodes(node_group, "set_to_-5x"),
            "set_to_-10x": _collect_ablation_stats_for_nodes(node_group, "set_to_-10x"),
        }

    # Combined ablation for groups that share the same (layer, token) with multiple neurons.
    grouped_nodes: dict[tuple[int, int], list[int]] = {}
    for k_u, L_u, i_u in circuit_nodes_set:
        grouped_nodes.setdefault((int(L_u), int(k_u)), []).append(int(i_u))

    combined_ablation_dist: dict[str, Any] = {}
    for (L_u, k_u), neurons in grouped_nodes.items():
        uniq_neurons = sorted(set(neurons))
        if len(uniq_neurons) <= 1:
            continue
        group_id = f"{L_u}_{k_u}"
        nodes_group = [(int(k_u), int(L_u), int(i_val)) for i_val in uniq_neurons]
        combined_ablation_dist[group_id] = {
            "L": int(L_u),
            "k": int(k_u),
            "neurons": uniq_neurons,
            "num_neurons": int(len(uniq_neurons)),
            "set_to_0": _collect_ablation_stats_for_nodes(nodes_group, "set_to_0"),
            "set_to_-5x": _collect_ablation_stats_for_nodes(nodes_group, "set_to_-5x"),
            "set_to_-10x": _collect_ablation_stats_for_nodes(nodes_group, "set_to_-10x"),
        }

    # Combined ablation for groups that share the same token across any layers.
    token_grouped_nodes: dict[int, list[tuple[int, int, int]]] = {}
    for k_u, L_u, i_u in circuit_nodes_set:
        token_grouped_nodes.setdefault(int(k_u), []).append((int(k_u), int(L_u), int(i_u)))

    token_combined_ablation_dist: dict[str, Any] = {}
    for k_u, nodes_group in token_grouped_nodes.items():
        if len(nodes_group) <= 1:
            continue
        token_group_id = f"k_{int(k_u)}"
        sorted_nodes_group = sorted(nodes_group, key=lambda x: (int(x[1]), int(x[2])))
        token_combined_ablation_dist[token_group_id] = {
            "k": int(k_u),
            "nodes": [{"k": int(k), "L": int(L), "i": int(i)} for (k, L, i) in sorted_nodes_group],
            "num_nodes": int(len(sorted_nodes_group)),
            "set_to_0": _collect_ablation_stats_for_nodes(sorted_nodes_group, "set_to_0"),
            "set_to_-5x": _collect_ablation_stats_for_nodes(sorted_nodes_group, "set_to_-5x"),
            "set_to_-10x": _collect_ablation_stats_for_nodes(sorted_nodes_group, "set_to_-10x"),
        }

    return {
        "nodes": circuit_nodes_set,
        "per_node": per_node_data,
        "baseline_top_k": baseline_top_k,
        "per_node_ablation_dist": per_node_ablation_dist,
        "combined_ablation_dist": combined_ablation_dist,
        "token_combined_ablation_dist": token_combined_ablation_dist,
        "rel_changes_matrix": rel_changes.tolist(),
        "deltas_matrix": deltas.tolist(),
    }


# ------------------------------------------------------------------------------
# Main
# ------------------------------------------------------------------------------
def run_attribution_graph(
    args: argparse.Namespace,
    *,
    hooked_model: Any,
    prompt: str | None,
    tokens_csv: str | None,
    n_prompts: int,
) -> None:
    show_progress = not bool(args.no_progress)

    logger.info("Stage 1/9: encoding prompt")
    tokens, prompt_repr = _encode_prompt(prompt, tokens_csv)
    cfg = hooked_model.model.cfg
    if len(tokens) == 0:
        raise ValueError("Prompt tokenization produced empty token list")
    if len(tokens) > int(cfg.block_size):
        raise ValueError(f"Prompt length {len(tokens)} exceeds model block_size={cfg.block_size}")

    tokens_t = torch.tensor(tokens, dtype=torch.long, device=args.device).unsqueeze(0)
    token_strings = _decode_tokens(tokens)

    logger.info("Stage 2/9: collecting baseline activations and target")
    baseline = collect_baseline(hooked_model, tokens_t)
    target_id = int(baseline["target_id"])
    logp0 = float(baseline["logp0"])
    acts_by_layer: dict[int, torch.Tensor] = baseline["acts_by_layer"]
    base_norms_by_layer: dict[int, torch.Tensor] = baseline["base_norms_by_layer"]

    norm_match_C = bool(args.norm_match_for_C)
    norm_match_downstream = bool(args.norm_match)

    metric_t = float(args.metric_threshold) if args.metric_threshold is not None else float(args.retain_ratio)
    metric_t_max = float(args.metric_threshold_max) if args.metric_threshold_max is not None else float(metric_t)
    metric_t_min = float(args.metric_threshold_min)
    max_nodes = max(1, int(args.max_nodes))
    # In beam+delete_down, the size slider is naturally interpreted as "# deleted".
    # Allow circuit_size=0 for that case (0 deletions).
    circuit_size_raw = int(args.circuit_size)
    circuit_size = max(0, circuit_size_raw)

    # ---- Candidates ----
    logger.info("Stage 3/9: extracting candidates")
    candidates = _extract_candidates_all_positions(
        acts_by_layer,
        base_norms_by_layer=base_norms_by_layer,
    )
    logger.info("Candidates extracted: %d", len(candidates))

    # ---- Compute C values ----
    logger.info("Stage 4/9: computing C values")
    num_repeats = max(1, int(args.num_repeats))
    ranked = compute_C_batched_by_layer(
        hooked_model=hooked_model,
        tokens_t=tokens_t,
        candidates=candidates,
        target_id=target_id,
        logp0=logp0,
        device=args.device,
        base_norms_by_layer=base_norms_by_layer,
        norm_match=norm_match_C,
        num_repeats=num_repeats,
        show_progress=show_progress,
    )
    logger.info("C values computed for %d ranked candidates", len(ranked))

    # ---- MLP weight interaction heatmap (top-30 neurons by C) ----
    # We take the top-30 MLP neurons (layer >= 0) with the smallest C values,
    # extract their per-neuron "down" (fc1) and "up" (fc2) weight vectors in R^{d_model},
    # and compute an interaction matrix M between these vectors.
    #
    # Metric (configurable via --interaction_metric):
    #   - "dot": M[u, v] = W_up[u] · W_down[v]
    #   - "cos": M[u, v] = cos_sim(W_up[u], W_down[v])
    #
    # This yields an n×n matrix (n <= 30) summarizing how strongly each neuron's
    # output weights align with each neuron's input weights.
    TOPK_INTERACTION = 30
    top_mlp: list[Candidate] = [c for c in ranked if int(c.layer) >= 0][:TOPK_INTERACTION]
    weight_interaction: dict[str, Any] | None = None
    if top_mlp:
        interaction_metric = str(getattr(args, "interaction_metric", "dot"))
        n_int = len(top_mlp)
        mat = torch.zeros((n_int, n_int), dtype=torch.float32)
        mat_random = torch.zeros((n_int, n_int), dtype=torch.float32)
        # Pre-cache per-neuron weight vectors to avoid repeated indexing work.
        w_up_list: list[torch.Tensor] = []
        w_down_list: list[torch.Tensor] = []
        for c in top_mlp:
            L_c = int(c.layer)
            i_c = int(c.neuron)
            fc1 = hooked_model.model.blocks[L_c].mlp.fc1
            fc2 = hooked_model.model.blocks[L_c].mlp.fc2
            # fc1.weight: [hidden, d_model] → per-neuron "down" weights are rows
            w_down = fc1.weight[i_c, :].detach().float().cpu()
            # fc2.weight: [d_model, hidden] → per-neuron "up" weights are columns
            w_up = fc2.weight[:, i_c].detach().float().cpu()
            w_down_list.append(w_down)
            w_up_list.append(w_up)

        # Optional cosine-normalized copies (to avoid repeated norm work inside loops).
        if interaction_metric == "cos":
            w_up_normed: list[torch.Tensor] = []
            w_down_normed: list[torch.Tensor] = []
            for w_u in w_up_list:
                nu = torch.linalg.norm(w_u).clamp_min(1e-8)
                w_up_normed.append(w_u / nu)
            for w_d in w_down_list:
                nd = torch.linalg.norm(w_d).clamp_min(1e-8)
                w_down_normed.append(w_d / nd)
        else:
            w_up_normed = w_up_list
            w_down_normed = w_down_list

        # Build main interaction matrix.
        for u in range(n_int):
            w_up_u = w_up_list[u]
            w_up_u_n = w_up_normed[u]
            for v in range(n_int):
                w_down_v = w_down_list[v]
                w_down_v_n = w_down_normed[v]
                if interaction_metric == "cos":
                    mat[u, v] = torch.dot(w_up_u_n, w_down_v_n)
                else:
                    mat[u, v] = torch.dot(w_up_u, w_down_v)

        # Build random-baseline interaction matrix: random vectors with the same norms.
        # This provides a comparison heatmap for what "chance-level" interactions look like.
        gen = torch.Generator(device="cpu")
        gen.manual_seed(0)
        w_up_rand: list[torch.Tensor] = []
        w_down_rand: list[torch.Tensor] = []
        for w_u in w_up_list:
            r = torch.randn_like(w_u, generator=gen)
            nu = torch.linalg.norm(w_u).clamp_min(1e-8)
            nr = torch.linalg.norm(r).clamp_min(1e-8)
            w_up_rand.append(r * (nu / nr))
        for w_d in w_down_list:
            r = torch.randn_like(w_d, generator=gen)
            nd = torch.linalg.norm(w_d).clamp_min(1e-8)
            nr = torch.linalg.norm(r).clamp_min(1e-8)
            w_down_rand.append(r * (nd / nr))

        if interaction_metric == "cos":
            w_up_rand_normed: list[torch.Tensor] = []
            w_down_rand_normed: list[torch.Tensor] = []
            for w_u in w_up_rand:
                nu = torch.linalg.norm(w_u).clamp_min(1e-8)
                w_up_rand_normed.append(w_u / nu)
            for w_d in w_down_rand:
                nd = torch.linalg.norm(w_d).clamp_min(1e-8)
                w_down_rand_normed.append(w_d / nd)
        else:
            w_up_rand_normed = w_up_rand
            w_down_rand_normed = w_down_rand

        for u in range(n_int):
            w_up_u = w_up_rand[u]
            w_up_u_n = w_up_rand_normed[u]
            for v in range(n_int):
                w_down_v = w_down_rand[v]
                w_down_v_n = w_down_rand_normed[v]
                if interaction_metric == "cos":
                    mat_random[u, v] = torch.dot(w_up_u_n, w_down_v_n)
                else:
                    mat_random[u, v] = torch.dot(w_up_u, w_down_v)

        weight_interaction = {
            "size": int(n_int),
            "metric": interaction_metric,
            "neurons": [
                {
                    "k": int(c.k),
                    "L": int(c.layer),
                    "i": int(c.neuron),
                    "C": float(c.C) if c.C is not None else None,
                }
                for c in top_mlp
            ],
            "matrix": mat.tolist(),
            "matrix_random": mat_random.tolist(),
        }

    # ---- Circuit selection ----
    selection_mode = str(args.selection_mode)
    search_mode = str(args.search_mode)
    logger.info("Stage 5/9: selecting circuit (search_mode=%s, selection_mode=%s)", search_mode, selection_mode)

    # These will be populated by the selection process
    selected_superset: list[Candidate] = []
    selected: list[Candidate] = []
    addback_curve: list[dict[str, float]] = []
    final_ratio: float = 0.0
    final_ratio_max: float = 0.0
    ui_order: list[tuple[int, int, int]] = []
    beam_sets_by_n: list[list[tuple[int, int, int]]] | None = None

    if search_mode == "beam":
        beam_w = max(1, int(args.beam_width))
        win = max(1, int(args.beam_window))
        if selection_mode == "delete_down":
            # Beam search over DELETION sets:
            # choose a set of nodes to ablate (delete) of size n that maximizes retain_ratio.
            def logp_for_deleted(idx_set: set[int]) -> float:
                ablate: dict[int, list[tuple[int, int]]] = {}
                for j in idx_set:
                    c = ranked[int(j)]
                    ablate.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
                return logp_with_ablate_sets(
                    hooked_model,
                    tokens_t,
                    target_id,
                    ablate,
                    base_norms_by_layer=base_norms_by_layer,
                    norm_match=norm_match_downstream,
                    num_repeats=num_repeats,
                )

            # n=0: nothing deleted => baseline ratio = 1.
            beams: list[tuple[set[int], float]] = [(set(), 1.0)]
            best_by_n: dict[int, tuple[float, set[int]]] = {0: (1.0, set())}
            frontier = list(beams)

            max_n = min(int(max_nodes), len(ranked))
            for n in tqdm(range(1, max_n + 1), desc="Beam delete-down", disable=not show_progress, leave=False):
                expanded: list[tuple[set[int], float]] = []
                seen: set[frozenset[int]] = set()
                for s, _r in frontier:
                    # Candidate pool: among nodes not yet deleted, try the bottom `win`
                    # by importance (ranked is sorted by increasing importance / C).
                    cand_js: list[int] = []
                    if win > 0:
                        for j in range(len(ranked) - 1, -1, -1):
                            if j in s:
                                continue
                            cand_js.append(j)
                            if len(cand_js) >= win:
                                break
                    for j in cand_js:
                        s2 = set(s)
                        s2.add(int(j))
                        key = frozenset(s2)
                        if key in seen:
                            continue
                        seen.add(key)
                        lp = logp_for_deleted(s2)
                        ratio = float(math.exp(lp - logp0))
                        expanded.append((s2, ratio))
                if not expanded:
                    break
                expanded.sort(key=lambda x: (-x[1], len(x[0])))
                frontier = expanded[:beam_w]
                best_by_n[n] = (float(frontier[0][1]), set(frontier[0][0]))

            discovered_max = max(best_by_n.keys()) if best_by_n else 0

            # For delete_down, curve x-axis is "# deleted". Keep legacy key `n_added`
            # too, because the dashboard historically reads `n_added`.
            addback_curve = []
            last_ratio = float(best_by_n.get(0, (1.0, set()))[0])
            for n in range(0, int(max_nodes) + 1):
                if n in best_by_n:
                    last_ratio = float(best_by_n[n][0])
                addback_curve.append(
                    {
                        "n_added": float(n),  # legacy
                        "n_deleted": float(n),
                        "n_kept": float(max(0, len(ranked) - int(n))),
                        "retain_ratio": float(last_ratio),
                    }
                )

            chosen_n = max(0, min(int(circuit_size), int(max_nodes), int(discovered_max)))
            chosen_set = best_by_n.get(chosen_n, best_by_n.get(discovered_max, (last_ratio, set())))[1]
            # In delete_down, "selected" is the deletion set (nodes being ablated).
            selected = [ranked[j] for j in sorted(chosen_set)]
            final_ratio = float(best_by_n.get(chosen_n, (last_ratio, set()))[0])

            union_set: set[int] = set()
            for n in range(0, int(discovered_max) + 1):
                union_set |= best_by_n.get(n, (0.0, set()))[1]
            selected_superset = [ranked[j] for j in sorted(union_set)]
            # Order deletions as in greedy delete_down: least-important first (largest C).
            selected_superset.sort(key=lambda c: float("inf") if c.C is None else float(c.C), reverse=True)
            ui_order = [(c.k, c.layer, c.neuron) for c in selected_superset]

            beam_sets_by_n = []
            for n in range(0, int(max_nodes) + 1):
                if n in best_by_n:
                    s = best_by_n[n][1]
                    nodes_n = [(ranked[j].k, ranked[j].layer, ranked[j].neuron) for j in sorted(s)]
                else:
                    nodes_n = []
                beam_sets_by_n.append(nodes_n)
            final_ratio_max = float(best_by_n.get(discovered_max, (last_ratio, set()))[0])
        else:
            # Beam search over ADD-BACK sets:
            # choose a set of kept nodes of size n that maximizes retain_ratio when all others are ablated.

            def logp_for_added(idx_set: set[int]) -> float:
                ablate: dict[int, list[tuple[int, int]]] = {}
                keep = idx_set
                for j, c in enumerate(ranked):
                    if j in keep:
                        continue
                    ablate.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
                return logp_with_ablate_sets(
                    hooked_model,
                    tokens_t,
                    target_id,
                    ablate,
                    base_norms_by_layer=base_norms_by_layer,
                    norm_match=norm_match_downstream,
                    num_repeats=num_repeats,
                )

            ablate_all: dict[int, list[tuple[int, int]]] = {}
            for c in ranked:
                ablate_all.setdefault(int(c.layer), []).append((int(c.k), int(c.neuron)))
            logp_start = logp_with_ablate_sets(
                hooked_model,
                tokens_t,
                target_id,
                ablate_all,
                base_norms_by_layer=base_norms_by_layer,
                norm_match=norm_match_downstream,
                num_repeats=num_repeats,
            )
            beams: list[tuple[set[int], float]] = [(set(), float(math.exp(logp_start - logp0)))]

            max_n = min(int(max_nodes), len(ranked))
            best_by_n: dict[int, tuple[float, set[int]]] = {0: (float(beams[0][1]), set())}
            frontier = list(beams)
            for n in tqdm(range(1, max_n + 1), desc="Beam add-back", disable=not show_progress, leave=False):
                expanded: list[tuple[set[int], float]] = []
                seen: set[frozenset[int]] = set()
                for s, _r in frontier:
                    # Candidate pool: among all nodes not yet selected, try the top `win`
                    # by ranking (ranked is already sorted by importance / C).
                    cand_js: list[int] = []
                    if win > 0:
                        for j in range(len(ranked)):
                            if j in s:
                                continue
                            cand_js.append(j)
                            if len(cand_js) >= win:
                                break
                    for j in cand_js:
                        s2 = set(s)
                        s2.add(j)
                        key = frozenset(s2)
                        if key in seen:
                            continue
                        seen.add(key)
                        lp = logp_for_added(s2)
                        ratio = float(math.exp(lp - logp0))
                        expanded.append((s2, ratio))
                if not expanded:
                    break
                expanded.sort(key=lambda x: (-x[1], len(x[0])))
                frontier = expanded[:beam_w]
                best_by_n[n] = (float(frontier[0][1]), set(frontier[0][0]))

            discovered_max = max(best_by_n.keys()) if best_by_n else 0

            addback_curve = []
            last_ratio = float(best_by_n.get(0, (0.0, set()))[0])
            for n in range(0, int(max_nodes) + 1):
                if n in best_by_n:
                    last_ratio = float(best_by_n[n][0])
                addback_curve.append({"n_added": float(n), "retain_ratio": float(last_ratio)})

            chosen_n = max(1, min(int(circuit_size), int(max_nodes), int(discovered_max)))
            chosen_set = best_by_n.get(chosen_n, best_by_n.get(discovered_max, (last_ratio, set())))[1]
            selected = [ranked[j] for j in sorted(chosen_set)]
            final_ratio = float(best_by_n.get(chosen_n, (last_ratio, set()))[0])

            union_set: set[int] = set()
            for n in range(0, int(discovered_max) + 1):
                union_set |= best_by_n.get(n, (0.0, set()))[1]
            selected_superset = [ranked[j] for j in sorted(union_set)]
            ui_order = [(c.k, c.layer, c.neuron) for c in selected_superset]

            beam_sets_by_n = []
            for n in range(0, int(max_nodes) + 1):
                if n in best_by_n:
                    s = best_by_n[n][1]
                    nodes_n = [(ranked[j].k, ranked[j].layer, ranked[j].neuron) for j in sorted(s)]
                else:
                    nodes_n = []
                beam_sets_by_n.append(nodes_n)
            final_ratio_max = float(best_by_n.get(discovered_max, (last_ratio, set()))[0])
    else:
        # Greedy mode
        if selection_mode == "delete_down":
            selected_superset, final_ratio_max, curve = select_circuit_delete_down(
                hooked_model=hooked_model,
                tokens_t=tokens_t,
                ranked=ranked,
                target_id=target_id,
                logp0=logp0,
                retain_ratio=float(metric_t_max),
                base_norms_by_layer=base_norms_by_layer,
                norm_match=norm_match_downstream,
                max_steps=None,
                num_repeats=num_repeats,
                show_progress=show_progress,
            )
            selected, final_ratio, _curve_t = select_circuit_delete_down(
                hooked_model=hooked_model,
                tokens_t=tokens_t,
                ranked=ranked,
                target_id=target_id,
                logp0=logp0,
                retain_ratio=float(metric_t),
                base_norms_by_layer=base_norms_by_layer,
                norm_match=norm_match_downstream,
                max_steps=None,
                num_repeats=num_repeats,
                show_progress=show_progress,
            )
            addback_curve = curve
            ui_order = [(c.k, c.layer, c.neuron) for c in selected_superset]
            beam_sets_by_n = None
        else:  # add_back
            selected_superset, final_ratio_max, curve = select_circuit_add_back(
                hooked_model=hooked_model,
                tokens_t=tokens_t,
                ranked=ranked,
                target_id=target_id,
                logp0=logp0,
                retain_ratio=float(metric_t_max),
                base_norms_by_layer=base_norms_by_layer,
                norm_match=norm_match_downstream,
                max_steps=None,
                num_repeats=num_repeats,
                show_progress=show_progress,
            )
            n_default = 0
            for p in curve:
                if float(p.get("retain_ratio", 0.0)) >= float(metric_t):
                    n_default = int(p.get("n_added", 0.0))
                    break
            selected = list(selected_superset[:n_default]) if n_default > 0 else []
            final_ratio = float(curve[n_default].get("retain_ratio", final_ratio_max)) if n_default < len(curve) else float(final_ratio_max)
            addback_curve = curve
            ui_order = [(c.k, c.layer, c.neuron) for c in selected_superset]
            beam_sets_by_n = None

    # ---- Faithfulness curve using the actual circuits for each n ----
    logger.info("Stage 6/9: computing faithfulness curve")
    circuit_dict: dict[int, dict[int, list[tuple[int, int]]]] = {}
    max_n = min(max_nodes, len(ranked))

    if search_mode == "beam":
        # Use the best circuits discovered by beam search
        for n in range(0, len(beam_sets_by_n)):
            if beam_sets_by_n[n]:   # non‑empty circuit
                circuit_dict[n] = {}
                for k, L, i in beam_sets_by_n[n]:
                    circuit_dict[n].setdefault(int(L), []).append((int(k), int(i)))
    else:
        # Greedy modes: compute circuit for each possible n
        for n in tqdm(range(0, max_n + 1), desc="Build circuits by n", disable=not show_progress, leave=False):
            ablate = get_circuit_ablation_for_n(
                n,
                ranked,
                selection_mode,
                selected_superset if selection_mode == "add_back" else None,
                None,   # no beam
            )
            if ablate or n == 0:
                circuit_dict[n] = ablate

    faithfulness_curve = compute_faithfulness_curve_from_circuits(
        hooked_model=hooked_model,
        tokens_t=tokens_t,
        target_id=target_id,
        logp0=logp0,
        base_norms_by_layer=base_norms_by_layer,
        # IMPORTANT: faithfulness should reflect pure zero ablations of the discovered circuit,
        # independent of any norm-matching noise used elsewhere.
        norm_match=False,
        num_repeats=1,
        circuit_dict=circuit_dict,
        show_progress=show_progress,
    )

    # ---- Influence edges ----
    logger.info("Stage 7/9: computing influence edges")
    influence = compute_influence_edges(
        hooked_model=hooked_model,
        tokens_t=tokens_t,
        selected=selected_superset,
        baseline_acts_by_layer=acts_by_layer,
        base_norms_by_layer=base_norms_by_layer,
        target_id=target_id,
        logp0=logp0,
        max_neighbors=int(args.max_neighbors),
        norm_match=norm_match_downstream,
        num_repeats=num_repeats,
    )

    # ---- Detailed circuit visualization (optional) ----
    logger.info("Stage 8/9: detailed circuit artifacts")
    detailed_circuit_data = None
    if args.circuit_detail_mode != "none":
        logger.info("Computing detailed circuit (mode=%s)", args.circuit_detail_mode)
        circuit_detail_nodes: list[Candidate] | None = None
        # Keep detailed-circuit selection consistent with the displayed beam curve.
        if search_mode == "beam" and beam_sets_by_n is not None:
            node_to_candidate = {(int(c.k), int(c.layer), int(c.neuron)): c for c in ranked}
            selected_n: int | None = None
            if args.circuit_detail_mode == "fixed_k":
                selected_n = max(0, min(int(args.circuit_detail_k), len(beam_sets_by_n) - 1))
            elif args.circuit_detail_mode == "threshold_t":
                threshold_t = float(args.circuit_detail_t)
                for p in addback_curve:
                    n_val = int(p.get("n_added", 0.0))
                    ratio_val = float(p.get("retain_ratio", 0.0))
                    if ratio_val >= threshold_t:
                        selected_n = n_val
                        break
            if selected_n is not None:
                if 0 <= selected_n < len(beam_sets_by_n) and beam_sets_by_n[selected_n]:
                    circuit_detail_nodes = []
                    for node in beam_sets_by_n[selected_n]:
                        cand = node_to_candidate.get((int(node[0]), int(node[1]), int(node[2])))
                        if cand is not None:
                            circuit_detail_nodes.append(cand)
                else:
                    logger.warning(
                        "Beam circuit for n=%s is unavailable/empty; falling back to ranked-based detailed selection",
                        selected_n,
                    )
        if circuit_detail_nodes is None:
            circuit_detail_nodes = select_circuit_nodes(
                ranked,
                mode=args.circuit_detail_mode,
                k=int(args.circuit_detail_k) if args.circuit_detail_mode == "fixed_k" else None,
                threshold_t=float(args.circuit_detail_t) if args.circuit_detail_mode == "threshold_t" else None,
                hooked_model=hooked_model,
                tokens_t=tokens_t,
                target_id=target_id,
                logp0=logp0,
                base_norms_by_layer=base_norms_by_layer,
                device=args.device,
                norm_match=norm_match_downstream,
                num_repeats=num_repeats,
            )
        logger.info("Selected %d circuit nodes for detailed view", len(circuit_detail_nodes))
    
        # Recalculate C values with circuit isolation
        circuit_C_values = compute_localized_C_values(
            hooked_model,
            tokens_t,
            circuit_detail_nodes,
            target_id,
            logp0,
            args.device,
            base_norms_by_layer=base_norms_by_layer,
            norm_match=norm_match_downstream,
            num_repeats=num_repeats,
        )
        logger.info("Computed localized C values")
    
        # Compute detailed influence edges within circuit
        circuit_edges = compute_detailed_circuit_edges(
            hooked_model,
            tokens_t,
            circuit_detail_nodes,
            acts_by_layer,
            base_norms_by_layer=base_norms_by_layer,
            target_id=target_id,
            logp0=logp0,
            device=args.device,
            token_strings=token_strings,
            norm_match=norm_match_downstream,
            num_repeats=num_repeats,
            circuit_C_values=circuit_C_values,
        )
        logger.info("Computed detailed circuit edges")
    
        detailed_circuit_data = {
            "mode": args.circuit_detail_mode,
            "num_nodes": len(circuit_detail_nodes),
            "nodes": circuit_detail_nodes,
            "C_values": circuit_C_values,
            "edges": circuit_edges,
        }

    # ---- Output ----
    logger.info("Stage 9/9: writing outputs")
    run_name = _resolve_run_name(
        args.run_name,
        prompt_repr,
        n_prompts=n_prompts,
    )
    base_run_dir = os.path.join(args.out_dir, run_name)
    _ensure_dir(base_run_dir)
    
    # Use separate directories for detailed vs non-detailed circuits
    if args.circuit_detail_mode != "none":
        run_dir = os.path.join(base_run_dir, "circuit")
        detail_dir = os.path.join(base_run_dir, "circuit_detail")
    else:
        run_dir = os.path.join(base_run_dir, "circuit")
        detail_dir = None
    
    _ensure_dir(run_dir)

    meta = {
        "ckpt": os.path.abspath(args.ckpt),
        "device": str(args.device),
        "prompt": prompt_repr,
        "tokens": tokens,
        "target_id": int(target_id),
        "target_str": (_decode_tokens([target_id])[0] if target_id is not None else None),
        "logp0": float(logp0),
        "p0": float(math.exp(logp0)),
        "max_nodes": int(max_nodes),
        "circuit_size": int(circuit_size),
        "final_ratio": float(final_ratio),
        "final_ratio_max": float(final_ratio_max) if search_mode != "beam" else float(final_ratio_max),
        "norm_match_for_C": bool(norm_match_C),
        "norm_match_downstream": bool(norm_match_downstream),
        "num_repeats": int(num_repeats),
        "include_token_nodes": False,
        "selection_mode": selection_mode,
        "search_mode": search_mode,
        "n_layer": int(cfg.n_layer),
        "n_embd": int(cfg.n_embd),
        "hidden": int(cfg.mlp_mult * cfg.n_embd),
        "mlp_topk": int(cfg.mlp_topk),
        "created_at": time.time(),
    }
    meta["base_norms_by_layer"] = {int(L): v.detach().cpu().tolist() for L, v in base_norms_by_layer.items()}
    if weight_interaction is not None:
        meta["weight_interaction"] = weight_interaction

    graph = attribution_graph_html.build_graph_payload(
        meta=meta,
        token_strings=token_strings,
        ranked=ranked,
        selected=selected_superset,
        influence=influence,
        addback_curve=addback_curve,
        ui_order=ui_order,
        beam_sets_by_n=beam_sets_by_n if search_mode == "beam" else None,
        neuron_html_root=args.neuron_html_root,
        run_dir=run_dir,
        faithfulness_curve=faithfulness_curve,   # new parameter
    )

    # After Stage 6 (faithfulness_curve computed) and before Stage 9
    save_curve_png(addback_curve, faithfulness_curve, meta, run_dir)

    json_path = os.path.join(run_dir, "graph.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(graph, f, ensure_ascii=False)

    html_path = os.path.join(run_dir, "index.html")
    with open(html_path, "w", encoding="utf-8") as f:
        f.write(attribution_graph_html.render_index_html(graph))

    logger.info("Wrote %s", html_path)
    logger.info("Wrote %s", json_path)
    
    # ---- Write detailed circuit HTML (if requested) ----
    if detailed_circuit_data is not None and detail_dir is not None:
        _ensure_dir(detail_dir)
        circuit_json_path = os.path.join(detail_dir, "circuit_detail.json")
        with open(circuit_json_path, "w", encoding="utf-8") as f:
            # Prepare circuit data for JSON serialization
            circuit_output = {
                "mode": detailed_circuit_data["mode"],
                "num_nodes": detailed_circuit_data["num_nodes"],
                "nodes": [
                    {
                        "k": int(c.k),
                        "L": int(c.layer),
                        "i": int(c.neuron),
                        "activation": float(c.activation),
                        "C": float(c.C) if c.C is not None else None,
                    }
                    for c in detailed_circuit_data["nodes"]
                ],
                "C_values": [
                    {
                        "k": int(c.k),
                        "L": int(c.layer),
                        "i": int(c.neuron),
                        "activation": float(c.activation),
                        "C_localized": float(c.C) if c.C is not None else None,
                    }
                    for c in detailed_circuit_data["C_values"]
                ],
                "edges": detailed_circuit_data["edges"],
                "token_strings": token_strings,
                "prompt": prompt_repr,
            }
            json.dump(circuit_output, f, ensure_ascii=False)
    
        logger.info("Wrote %s", circuit_json_path)
    
        # Generate graph visualization for detailed circuit using same style as main graph
        circuit_graph_payload = attribution_graph_html.build_detailed_circuit_payload(
            meta=meta,
            token_strings=token_strings,
            circuit_nodes=circuit_detail_nodes,
            circuit_C_values=circuit_C_values,
            circuit_edges_data=circuit_edges,
            neuron_html_root=args.neuron_html_root,
            detail_dir=detail_dir,
        )
    
        circuit_html_path = os.path.join(detail_dir, "index.html")
        circuit_html = attribution_graph_html.render_detailed_circuit_html(
            circuit_graph_payload,
            circuit_edges,
            neuron_html_root=args.neuron_html_root,
            detail_dir=detail_dir,
        )
        with open(circuit_html_path, "w", encoding="utf-8") as f:
            f.write(circuit_html)
    
        logger.info("Wrote %s", circuit_html_path)

default_prompt = [
    "Jack was hungry, so he went looking for something to",
    'She felt very sad so her friend gave her a big warm',
    'The cat saw a mouse and started to',
    'The princess kissed the frog and it turned into a handsome',
    'The robot beeped and said in a friendly', 
    'She planted a seed and watered it every',
    'He lost his way in the forest and shouted for',
    'She drew a picture and showed it to her',
    'The pirate looked at the map to find the hidden',
    'He washed his hands before sitting at the dinner',
    'She put on her shoes and ran out the',
    'She opened the book and started to', 
    'The rocket blasted off and flew into outer',
    'He wrapped the gift in colorful shiny',
    'She drank the hot soup and felt much',
    'She brushed her hair and looked in the',
    'She baked a cake and put it in the', 
    'They sailed across the sea in a little wooden',
    'She fell off her bike and scraped her',
    'The baby dropped her toy and started to',
    'He found a coin on the ground and put it in his',
    'She picked a red apple from the tallest', 
    'He stacked the blocks until the tower was very',
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description="Build ablation attribution graph for a prompt")
    parser.add_argument("--ckpt", type=str, default="out/main_mlp_mult_32_K_8/ckpt_iter300000.pt")
    parser.add_argument(
        "--prompt",
        type=str,
        nargs="+",
        default=default_prompt,
        help="One or more prompts; each gets its own output folder under --out_dir.",
    )
    parser.add_argument("--tokens_csv", type=str, default=None, help="Comma-separated token ids (bypass tiktoken)")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--retain_ratio", type=float, default=0.8, help="Alias for --metric_threshold.")
    parser.add_argument("--metric_threshold", type=float, default=None, help="Threshold t for P_new/P_orig (default: --retain_ratio).")
    parser.add_argument(
        "--metric_threshold_max",
        type=float,
        default=None,
        help="Max t supported by the dashboard slider (default: metric_threshold). Larger means more nodes/compute.",
    )
    parser.add_argument(
        "--metric_threshold_min",
        type=float,
        default=0.5,
        help="Min t shown in the dashboard slider (used only for UI).",
    )
    parser.add_argument(
        "--selection_mode",
        type=str,
        choices=["add_back", "delete_down"],
        default="add_back",
        # default="delete_down",
        help="How to build a circuit from the ranked list.",
    )
    parser.add_argument(
        "--search_mode",
        type=str,
        choices=["greedy", "beam"],
        default="beam",
        help="Subset search strategy. Beam is a fixed-output alternative (UI threshold slider disabled).",
    )
    parser.add_argument(
        "--interaction_metric",
        type=str,
        choices=["dot", "cos"],
        default="cos",
        help="Interaction metric for MLP weight heatmap: 'dot' (dot product) or 'cos' (cosine similarity).",
    )
    parser.add_argument("--max_nodes", type=int, default=50, help="Compute beam curve for n=1..max_nodes (capped by available candidates).")
    parser.add_argument("--circuit_size", type=int, default=20, help="Initial circuit size shown in the dashboard (size slider).")
    parser.add_argument("--beam_width", type=int, default=40)
    parser.add_argument(
        "--beam_window",
        type=int,
        default=40,
        help="Beam expansion window: for each partial circuit, try adding each of the top-N ranked nodes (smallest C) not already selected.",
    )
    parser.add_argument("--max_neighbors", type=int, default=10)
    parser.add_argument(
        "--norm_match",
        default=False,
        action="store_true",
        help="Enable norm-matching noise injection after ranking (selection + influence stages). Off in the paper.",
    )
    parser.add_argument(
        "--norm_match_for_C",
        default=False,
        action="store_true",
        help="Apply norm-matching noise injection during C_{k,L,i} computation (ranking stage). Off by default for stable, pure-ablation C.",
    )
    parser.add_argument(
        "--num_repeats",
        type=int,
        default=1,
        help="Number of repeats for norm-match robustness. When >1 and norm_match=True, averages results across different random seeds.",
    )
    parser.add_argument(
        "--include_token_nodes",
        default=False,
        action="store_true",
        help="(Deprecated; no effect) Token embedding nodes are always represented by layer=-1.",
    )
    parser.add_argument("--out_dir", type=str, default="out/main_mlp_mult_32_K_8/attribution_graphs")
    parser.add_argument("--neuron_html_root", type=str, default="out/main_mlp_mult_32_K_8/neuron_dashboards")
    parser.add_argument("--run_name", type=str, default=None)
    parser.add_argument("--no_progress", default=False, action="store_true", help="Disable tqdm progress bars.")
    
    # Detailed circuit visualization options
    parser.add_argument(
        "--circuit_detail_mode",
        type=str,
        choices=["none", "fixed_k", "threshold_t"],
        default="threshold_t",
        help="Generate detailed circuit HTML. 'fixed_k': use --circuit_detail_k nodes. 'threshold_t': minimum nodes for retain_ratio >= --circuit_detail_t.",
    )
    parser.add_argument(
        "--circuit_detail_k",
        type=int,
        default=20,
        help="Number of nodes in the detailed circuit (used when circuit_detail_mode='fixed_k').",
    )
    parser.add_argument(
        "--circuit_detail_t",
        type=float,
        default=0.95,
        help="Retain ratio threshold (used when circuit_detail_mode='threshold_t').",
    )
    
    args = parser.parse_args()
    prompts: list[str] = list(args.prompt)
    if args.tokens_csv and len(prompts) > 1:
        raise ValueError("--tokens_csv cannot be used with multiple --prompt values")

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%H:%M:%S",
    )
    logger.info("Loading model (shared across prompts)")
    hooked_model = create_hooked_model(args.ckpt, args.device)

    n_prompts = len(prompts)
    for prompt_index, prompt in enumerate(prompts):
        logger.info("=== Prompt %d/%d ===", prompt_index + 1, n_prompts)
        run_attribution_graph(
            args,
            hooked_model=hooked_model,
            prompt=prompt,
            tokens_csv=args.tokens_csv,
            n_prompts=n_prompts,
        )


if __name__ == "__main__":
    main()
