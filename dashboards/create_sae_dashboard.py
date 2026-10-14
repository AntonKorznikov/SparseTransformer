"""
Main script to extract neuron activations and generate HTML dashboards.

This script:
1. Loads model A
2. Extracts post-BatchTopK activations from specified layer
3. Saves activation data
4. Generates HTML dashboards
"""

from __future__ import annotations

import argparse
import os
import pickle
import random
from typing import Any

import numpy as np
import torch
from tqdm import tqdm

from model_adapter import create_hooked_model
from utils.common import get_batch, load_bin

try:
    import tiktoken  # type: ignore
except Exception:  # pragma: no cover
    tiktoken = None


# Global cache for dedup operations (shared across function calls)
_dedup_cache = {}


class LazyTokenDecoder:
    """Lazy decoder for tokens to avoid decoding all tokens upfront.
    Significantly speeds up initialization for large token sets."""
    
    def __init__(self, tokens_flat: np.ndarray, enc=None):
        self.tokens_flat = tokens_flat
        self.enc = enc
        self.cache = {}
        self.uncacheable = set()
    
    def decode(self, token_id: int) -> str:
        """Decode a single token ID, using cache if available."""
        token_id = int(token_id)
        
        if token_id in self.cache:
            return self.cache[token_id]
        
        if token_id in self.uncacheable:
            return f"[{token_id}]"
        
        if self.enc is None:
            result = f"[{token_id}]"
            self.uncacheable.add(token_id)
            return result
        
        try:
            decoded = self.enc.decode([token_id])
            # Simple HTML escaping
            result = (
                decoded.replace("&", "&amp;")
                .replace("<", "&lt;")
                .replace(">", "&gt;")
                .replace('"', "&quot;")
                .replace("'", "&#39;")
            )
        except Exception:
            result = f"[{token_id}]"
            self.uncacheable.add(token_id)
            return result
        
        # Only cache a reasonable number of tokens
        if len(self.cache) < 50000:
            self.cache[token_id] = result
        
        return result

def dedup_positions(positions: list[int], tokens: np.ndarray, radius: int, max_examples: int) -> list[int]:
    """Return up to max_examples positions that have distinct token windows.
    Uses caching to avoid redundant dedup operations."""
    
    # Create a cache key from inputs
    positions_tuple = tuple(sorted(positions[:20]))  # Use first 20 as signature
    cache_key = (positions_tuple, radius, max_examples)
    
    # Check cache
    if cache_key in _dedup_cache:
        cached_result = _dedup_cache[cache_key]
        # If we need more, fall through to compute
        if len(cached_result) >= min(max_examples, len(positions)):
            return cached_result[:max_examples]
    
    seen = set()
    deduped = []
    tokens_arr = tokens if isinstance(tokens, np.ndarray) else np.array(tokens, dtype=np.int32)
    length = len(tokens_arr)
    for pos in positions:
        if pos < 0 or pos >= length:
            continue
        start = max(0, pos - radius)
        end = min(length, pos + radius + 1)
        snippet = tuple(int(tokens_arr[i]) for i in range(start, end))
        if snippet in seen:
            continue
        seen.add(snippet)
        deduped.append(int(pos))
        if len(deduped) >= max_examples:
            break
    
    # Cache result
    if len(_dedup_cache) < 10000:  # Limit cache size
        _dedup_cache[cache_key] = deduped
    
    return deduped


class ActivationSummaryCollector:
    """Track per-neuron statistics and top-k activations without storing the full activation map.
    Optimized for memory efficiency using bfloat16."""

    def __init__(self, num_neurons: int, top_k: int):
        self.num_neurons = num_neurons
        self.top_k = top_k
        self.topk_values = np.full((num_neurons, top_k), -np.inf, dtype=np.float32)  # Keep float32 for topk
        self.mean = np.zeros(num_neurons, dtype=np.float32)  # Keep float32 for running stats
        self.count = np.zeros(num_neurons, dtype=np.int32)
        self.M2 = np.zeros(num_neurons, dtype=np.float32)  # Keep float32 for variance
        self.max = np.full(num_neurons, -np.inf, dtype=np.float32)
        self.activation_positive = np.zeros(num_neurons, dtype=np.int32)

    def update(self, chunk_values: np.ndarray) -> None:
        if chunk_values.size == 0:
            return
        seq_len = chunk_values.shape[0]
        chunk_values = chunk_values.astype(np.float32)  # Convert from bf16 to float32 for stats
        chunk_mean = chunk_values.mean(axis=0)
        chunk_M2 = ((chunk_values - chunk_mean) ** 2).sum(axis=0)
        delta = chunk_mean - self.mean
        total_count = self.count + seq_len
        combined_mean = np.where(total_count > 0, (self.mean * self.count + chunk_mean * seq_len) / total_count, chunk_mean)
        self.M2 += chunk_M2 + np.where(total_count > 0, delta**2 * (self.count * seq_len) / total_count, 0.0)
        self.mean = combined_mean
        self.count = total_count
        self.max = np.maximum(self.max, chunk_values.max(axis=0))
        self.activation_positive += (chunk_values > 0).sum(axis=0).astype(np.int32)
        self._update_topk(chunk_values)

    def _update_topk(self, chunk_values: np.ndarray) -> None:
        if chunk_values.size == 0:
            return
        values_T = chunk_values.T  # [num_neurons, seq_len]
        combined = np.concatenate([self.topk_values, values_T], axis=1)
        if combined.shape[1] <= self.top_k:
            selected = combined
        else:
            idx = np.argpartition(combined, -self.top_k, axis=1)[:, -self.top_k:]
            selected = np.take_along_axis(combined, idx, axis=1)
        self.topk_values = np.sort(selected, axis=1)

    def summary(self, total_tokens: int) -> dict[str, np.ndarray]:
        valid_counts = (self.topk_values > -np.inf / 2).sum(axis=1)
        topk_sums = np.where(self.topk_values > -np.inf / 2, self.topk_values, 0.0).sum(axis=1)
        topk_means = np.where(valid_counts > 0, topk_sums / valid_counts, 0.0).astype(np.float32)
        safe_count = np.where(self.count > 0, self.count, 1.0)
        std = np.sqrt(np.where(self.count > 1, self.M2 / self.count, 0.0)).astype(np.float32)
        activation_pct = (self.activation_positive / max(total_tokens, 1)).astype(np.float32)
        return {
            "mean": self.mean,
            "std": std,
            "max": self.max,
            "activation_count": self.activation_positive,
            "activation_pct": activation_pct,
            "topk_means": topk_means,
        }


def update_top_candidates(
    values: np.ndarray,
    positions: np.ndarray,
    new_values: np.ndarray,
    new_positions: np.ndarray,
    pool_size: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray]:
    """GPU-accelerated top-k candidate update."""
    try:
        # Convert to torch tensors on GPU (bf16 for fast computation)
        values_t = torch.from_numpy(values).to(device, dtype=torch.bfloat16)
        positions_t = torch.from_numpy(positions).to(device, dtype=torch.int32)
        new_values_t = torch.from_numpy(new_values).to(device, dtype=torch.bfloat16)
        new_positions_t = torch.from_numpy(new_positions).to(device, dtype=torch.int32)
        
        # Concatenate on GPU
        combined_values_t = torch.cat([values_t, new_values_t], dim=1)
        combined_positions_t = torch.cat([positions_t, new_positions_t], dim=1)
        
        # Get top-k on GPU
        if combined_values_t.shape[1] <= pool_size:
            topk_vals, topk_idx = torch.topk(combined_values_t, k=combined_values_t.shape[1], dim=1, largest=True, sorted=True)
        else:
            topk_vals, topk_idx = torch.topk(combined_values_t, k=pool_size, dim=1, largest=True, sorted=True)
        
        # Gather positions
        selected_positions_t = torch.gather(combined_positions_t, 1, topk_idx)
        
        # Convert bf16 to float32 on GPU before numpy conversion
        return topk_vals.float().cpu().numpy().astype(np.float32), selected_positions_t.cpu().numpy().astype(np.int32)
    except (RuntimeError, AttributeError):
        # Fallback to CPU if GPU not available
        combined_values = np.concatenate([values, new_values], axis=1)
        combined_positions = np.concatenate([positions, new_positions], axis=1)
        if combined_values.shape[1] <= pool_size:
            order = np.argsort(-combined_values, axis=1)
            return np.take_along_axis(combined_values, order, axis=1), np.take_along_axis(combined_positions, order, axis=1)
        idx = np.argpartition(combined_values, -pool_size, axis=1)[:, -pool_size:]
        selected_values = np.take_along_axis(combined_values, idx, axis=1)
        selected_positions = np.take_along_axis(combined_positions, idx, axis=1)
        order = np.argsort(-selected_values, axis=1)
        return np.take_along_axis(selected_values, order, axis=1), np.take_along_axis(selected_positions, order, axis=1)


def update_closest_candidates(
    scores: np.ndarray,
    values: np.ndarray,
    positions: np.ndarray,
    new_scores: np.ndarray,
    new_values: np.ndarray,
    new_positions: np.ndarray,
    pool_size: int,
    device: str = "cuda",
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """GPU-accelerated closest-k candidate update (for mid/low quantiles)."""
    try:
        # Convert to torch tensors on GPU (bf16 for computation)
        scores_t = torch.from_numpy(scores).to(device, dtype=torch.bfloat16)
        values_t = torch.from_numpy(values).to(device, dtype=torch.bfloat16)
        positions_t = torch.from_numpy(positions).to(device, dtype=torch.int32)
        new_scores_t = torch.from_numpy(new_scores).to(device, dtype=torch.bfloat16)
        new_values_t = torch.from_numpy(new_values).to(device, dtype=torch.bfloat16)
        new_positions_t = torch.from_numpy(new_positions).to(device, dtype=torch.int32)
        
        # Concatenate on GPU
        combined_scores_t = torch.cat([scores_t, new_scores_t], dim=1)
        combined_values_t = torch.cat([values_t, new_values_t], dim=1)
        combined_positions_t = torch.cat([positions_t, new_positions_t], dim=1)
        
        # Get smallest scores (closest to target) on GPU
        if combined_scores_t.shape[1] <= pool_size:
            topk_scores, topk_idx = torch.topk(combined_scores_t, k=combined_scores_t.shape[1], dim=1, largest=False, sorted=True)
        else:
            topk_scores, topk_idx = torch.topk(combined_scores_t, k=pool_size, dim=1, largest=False, sorted=True)
        
        # Gather values and positions
        selected_values_t = torch.gather(combined_values_t, 1, topk_idx)
        selected_positions_t = torch.gather(combined_positions_t, 1, topk_idx)
        
        # Convert bf16 to float32 on GPU before numpy conversion
        return (
            topk_scores.float().cpu().numpy().astype(np.float32),
            selected_values_t.float().cpu().numpy().astype(np.float32),
            selected_positions_t.cpu().numpy().astype(np.int32),
        )
    except (RuntimeError, AttributeError):
        # Fallback to CPU if GPU not available
        combined_scores = np.concatenate([scores, new_scores], axis=1)
        combined_values = np.concatenate([values, new_values], axis=1)
        combined_positions = np.concatenate([positions, new_positions], axis=1)
        if combined_scores.shape[1] <= pool_size:
            order = np.argsort(combined_scores, axis=1)
            return (
                np.take_along_axis(combined_scores, order, axis=1),
                np.take_along_axis(combined_values, order, axis=1),
                np.take_along_axis(combined_positions, order, axis=1),
            )
        idx = np.argpartition(combined_scores, pool_size, axis=1)[:, :pool_size]
        selected_scores = np.take_along_axis(combined_scores, idx, axis=1)
        selected_values = np.take_along_axis(combined_values, idx, axis=1)
        selected_positions = np.take_along_axis(combined_positions, idx, axis=1)
        order = np.argsort(selected_scores, axis=1)
        return (
            np.take_along_axis(selected_scores, order, axis=1),
            np.take_along_axis(selected_values, order, axis=1),
            np.take_along_axis(selected_positions, order, axis=1),
        )


def gather_examples(
    positions: np.ndarray,
    values: np.ndarray,
    tokens: np.ndarray,
    max_examples: int,
    dedup_radius: int,
) -> list[dict[str, float]]:
    if positions.size == 0:
        return []
    valid_mask = positions >= 0
    if not np.any(valid_mask):
        return []
    filtered_positions = positions[valid_mask]
    filtered_values = values[valid_mask]
    deduped = dedup_positions(filtered_positions.tolist(), tokens, dedup_radius, max_examples)
    value_map: dict[int, float] = {}
    for pos, value in zip(filtered_positions.tolist(), filtered_values.tolist()):
        value_map[int(pos)] = float(value)
    return [{"pos": int(pos), "activation": value_map.get(int(pos), 0.0)} for pos in deduped]


def load_summary(summary_path: str) -> dict[str, np.ndarray]:
    with open(summary_path, "rb") as f:
        return pickle.load(f)


def run_summary_phase(
    layer_idx: int,
    hook_name: str,
    hooked_model,
    tokens: torch.Tensor,
    neuron_indices: list[int],
    neuron_indices_tensor: torch.Tensor,
    summary_top_k: int,
    total_tokens: int,
    inference_batch_size: int = 1,
) -> dict[str, np.ndarray]:
    collector = ActivationSummaryCollector(len(neuron_indices), top_k=summary_top_k)
    hooked_model.model.eval()
    with torch.no_grad():
        bs = max(int(inference_batch_size), 1)
        for start in tqdm(range(0, tokens.shape[0], bs), desc=f"   Summary (layer {layer_idx})"):
            batch_tokens = tokens[start : start + bs]
            _ = hooked_model(batch_tokens)
            act = hooked_model.get_activations(hook_name)
            if act is None:
                continue
            # act: [batch, seq, hidden] (expected); flatten to align with tokens.flatten() order.
            act_flat = act.reshape(-1, act.shape[-1])
            # Use bf16 for storage, convert to float32 for computation
            act_selected = act_flat[:, neuron_indices_tensor].cpu().numpy().astype(np.float32)
            collector.update(act_selected)
    summary_stats = collector.summary(total_tokens)
    summary_data = {
        "layer_idx": layer_idx,
        "neuron_indices": neuron_indices,
        "stats": summary_stats,
        "total_tokens": total_tokens,
        "hook_name": hook_name,
    }
    return summary_data


def run_example_phase(
    layer_idx: int,
    hook_name: str,
    hooked_model,
    tokens: torch.Tensor,
    tokens_flat: np.ndarray,
    neuron_indices: list[int],
    neuron_indices_tensor: torch.Tensor,
    summary_data: dict[str, np.ndarray],
    args: argparse.Namespace,
    hidden_size: int,
    encoder_weights: np.ndarray,
    decoder_weights: np.ndarray,
    input_embeddings: np.ndarray,
    unembedding_matrix: np.ndarray,
    store_large_matrices: bool,
) -> dict[str, Any]:
    """Process examples in neuron batches to manage memory on large neuron counts."""
    stats = summary_data["stats"]
    num_neurons = len(neuron_indices)
    pool_size = max(args.candidate_pool_size, args.max_examples_per_quantile * 3)
    total_tokens = tokens_flat.shape[0]
    neuron_batch_size = getattr(args, "neuron_batch_size", 2000)
    num_batches = (num_neurons + neuron_batch_size - 1) // neuron_batch_size
    
    neuron_examples: dict[int, dict[str, Any]] = {}
    
    # Process neurons in batches
    for batch_idx in tqdm(range(num_batches), desc=f"   Neuron Batches (layer {layer_idx})"):
        start_neuron = batch_idx * neuron_batch_size
        end_neuron = min(start_neuron + neuron_batch_size, num_neurons)
        batch_neuron_indices = neuron_indices[start_neuron:end_neuron]
        batch_neuron_indices_tensor = torch.tensor(
            batch_neuron_indices,
            device=args.device,
            dtype=torch.long
        )
        
        batch_size_actual = len(batch_neuron_indices)
        # Initialize candidate pools ONLY for this batch
        top_values = np.full((batch_size_actual, pool_size), -np.inf, dtype=np.float32)
        top_positions = np.full((batch_size_actual, pool_size), -1, dtype=np.int32)
        mid_scores = np.full((batch_size_actual, pool_size), np.inf, dtype=np.float32)
        mid_values = np.zeros((batch_size_actual, pool_size), dtype=np.float32)
        mid_positions = np.full((batch_size_actual, pool_size), -1, dtype=np.int32)
        low_scores = np.full((batch_size_actual, pool_size), np.inf, dtype=np.float32)
        low_values = np.zeros((batch_size_actual, pool_size), dtype=np.float32)
        low_positions = np.full((batch_size_actual, pool_size), -1, dtype=np.int32)
        
        target_mid = (0.5 * stats["topk_means"][start_neuron:end_neuron]).astype(np.float32)
        target_low = (0.1 * stats["topk_means"][start_neuron:end_neuron]).astype(np.float32)
        
        hooked_model.model.eval()
        with torch.no_grad():
            bs = max(int(getattr(args, "inference_batch_size", 1)), 1)
            seq_len = int(tokens.shape[1])
            for start_seq in tqdm(
                range(0, tokens.shape[0], bs),
                desc=f"     Tokens (batch {batch_idx+1}/{num_batches})",
                leave=False
            ):
                batch_tokens = tokens[start_seq : start_seq + bs]
                _ = hooked_model(batch_tokens)
                act = hooked_model.get_activations(hook_name)
                if act is None:
                    continue
                
                act_flat = act.reshape(-1, act.shape[-1])
                # Only select this batch's neurons
                act_selected = act_flat[:, batch_neuron_indices_tensor]
                flat_len = act_selected.shape[0]
                
                if flat_len == 0:
                    continue
                
                # Store in bf16 for memory efficiency
                values = act_selected.cpu().numpy().astype(np.float16)  # [flat_len, batch_size_actual]
                global_pos = int(start_seq) * seq_len
                end_pos = min(global_pos + int(flat_len), int(total_tokens))
                effective_len = max(0, end_pos - global_pos)
                
                if effective_len == 0:
                    continue
                
                if effective_len != flat_len:
                    values = values[:effective_len]
                
                # Convert to float32 for computation (bf16 operations in GPU functions)
                values_fp32 = values.astype(np.float32)
                
                positions = np.arange(global_pos, global_pos + effective_len, dtype=np.int32)
                values_T = values_fp32.T  # [batch_size_actual, effective_len]
                positions_matrix = np.broadcast_to(positions, (batch_size_actual, effective_len)).copy()
                
                # Update candidates with GPU acceleration
                top_values, top_positions = update_top_candidates(
                    top_values, top_positions, values_T, positions_matrix, pool_size,
                    device=args.device
                )
                mid_scores, mid_values, mid_positions = update_closest_candidates(
                    mid_scores, mid_values, mid_positions,
                    np.abs(values_T - target_mid[:, None]),
                    values_T, positions_matrix, pool_size,
                    device=args.device
                )
                low_scores, low_values, low_positions = update_closest_candidates(
                    low_scores, low_values, low_positions,
                    np.abs(values_T - target_low[:, None]),
                    values_T, positions_matrix, pool_size,
                    device=args.device
                )
        
        # Extract examples for this batch and add to dictionary
        for local_idx, neuron_idx in enumerate(batch_neuron_indices):
            idx_int = int(neuron_idx)
            global_idx = start_neuron + local_idx
            
            neuron_examples[idx_int] = {
                "stats": {
                    "mean": float(stats["mean"][global_idx]),
                    "std": float(stats["std"][global_idx]),
                    "max": float(stats["max"][global_idx]),
                    "activation_count": int(stats["activation_count"][global_idx]),
                    "activation_pct": float(stats["activation_pct"][global_idx]),
                    "E": float(stats["topk_means"][global_idx]),
                },
                "examples": {
                    "top": gather_examples(
                        top_positions[local_idx],
                        top_values[local_idx],
                        tokens_flat,
                        args.max_examples_per_quantile,
                        args.dedup_radius,
                    ),
                    "mid": gather_examples(
                        mid_positions[local_idx],
                        mid_values[local_idx],
                        tokens_flat,
                        args.max_examples_per_quantile,
                        args.dedup_radius,
                    ),
                    "low": gather_examples(
                        low_positions[local_idx],
                        low_values[local_idx],
                        tokens_flat,
                        args.max_examples_per_quantile,
                        args.dedup_radius,
                    ),
                },
            }

    out: dict[str, Any] = {
        "layer_idx": layer_idx,
        "hidden_size": hidden_size,
        "encoder_weights": encoder_weights,
        "decoder_weights": decoder_weights,
        "tokens": tokens_flat,
        "neuron_indices": neuron_indices,
        "block_size": args.block_size,
        "ckpt_path": args.ckpt,
        "neuron_examples": neuron_examples,
    }
    # Huge matrices (vocab_size x n_embd) are shared across layers; don't store them per-layer by default.
    if store_large_matrices:
        out["input_embeddings"] = input_embeddings
        out["unembedding_matrix"] = unembedding_matrix
    return out


def main():
    parser = argparse.ArgumentParser(
        description="Extract neuron activations and generate HTML dashboards"
    )
    parser.add_argument(
        "--ckpt",
        type=str,
        default="out/main_mlp_mult_32_K_8/ckpt_iter300000.pt",
        help="Path to model checkpoint",
    )
    parser.add_argument(
        "--data_dir",
        type=str,
        default="data/tinystories",
        help="Data directory",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="val",
        choices=["train", "val"],
        help="Data split to use",
    )
    parser.add_argument(
        "--layer_idx",
        type=int,
        default=1,
        help="Layer index (0-indexed) for MLP (single layer)",
    )
    parser.add_argument(
        "--layers",
        type=str,
        default='all',
        help="Comma-separated list of layer indices (e.g., '0,1,2') or 'all' for all layers. Overrides --layer_idx if provided.",
    )
    parser.add_argument(
        "--num_neurons",
        type=int,
        default=1000,
        help="Number of random neurons to visualize (ignored if --all_neurons is set)",
    )
    parser.add_argument(
        "--all_neurons",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Process ALL neurons in the layer (default); --no-all_neurons samples --num_neurons random ones",
    )
    parser.add_argument(
        "--num_sequences",
        type=int,
        default=10_000,
        help="Number of sequences to use for visualization",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,
        help="Batch size for processing",
    )
    parser.add_argument(
        "--inference_batch_size",
        type=int,
        default=64,
        help="Batch size for model forward passes during summary/examples (larger is faster, uses more memory; 64+ recommended for A100)",
    )
    parser.add_argument(
        "--neuron_batch_size",
        type=int,
        default=1000,
        help="Number of neurons to process in one batch (reduces memory usage for large neuron counts; 1000-4000 recommended)",
    )
    parser.add_argument(
        "--block_size",
        type=int,
        default=256,
        help="Sequence length",
    )
    parser.add_argument(
        "--data_seed",
        type=int,
        default=0,
        help="Random seed for data sampling",
    )
    parser.add_argument(
        "--neuron_seed",
        type=int,
        default=0,
        help="Random seed for neuron selection",
    )
    parser.add_argument(
        "--out_dir",
        type=str,
        default="out/main_mlp_mult_32_K_8/neuron_dashboards",
        help="Output directory for dashboards",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu",
        help="Device to use",
    )
    parser.add_argument(
        "--phase",
        type=str,
        choices=["both", "summary", "examples"],
        default="both",
        help="Run phase: summary (stats), examples (example selection), or both sequentially",
    )
    parser.add_argument(
        "--summary_dir",
        type=str,
        default=None,
        help="Directory to cache phase-1 summary pickles (defaults to <out_dir>/summaries)",
    )
    parser.add_argument(
        "--force_summary",
        action="store_true",
        help="Force recomputing the summary statistics even if a cache exists",
    )
    parser.add_argument(
        "--summary_top_k",
        type=int,
        default=10,
        help="Number of top activations per neuron to average during the summary pass",
    )
    parser.add_argument(
        "--candidate_pool_size",
        type=int,
        default=256,
        help="How many candidate contexts to keep per bucket before deduping",
    )
    parser.add_argument(
        "--max_examples_per_quantile",
        type=int,
        default=15,
        help="How many examples to keep per activation quantile (top, 0.5E, 0.1E)",
    )
    parser.add_argument(
        "--dedup_radius",
        type=int,
        default=2,
        help="Token radius for deduping similar examples",
    )
    parser.add_argument(
        "--html_only",
        default=True,
        action=argparse.BooleanOptionalAction,
        help="Generate HTML dashboards without writing .pkl files (default); --no-html_only also saves layer*_dashboard_data.pkl.",
    )
    parser.add_argument(
        "--skip_individual_html",
        default=False,
        action="store_true",
        help="Skip generating individual neuron HTML files (generates only index.html). Much faster for large neuron counts.",
    )
    parser.add_argument(
        "--store_large_matrices",
        action="store_true",
        help="Store input embeddings + unembedding matrix inside each layer*_dashboard_data.pkl (large; off by default)",
    )
    parser.add_argument(
        "--no_next_token_predictions",
        action="store_true",
        help="Disable next-token predictions in generated HTML (faster; avoids extra model work)",
    )
    parser.add_argument(
        "--next_token_top_k",
        type=int,
        default=5,
        help="How many next-token predictions to show per example (ignored if --no_next_token_predictions)",
    )
    parser.add_argument(
        "--html_prediction_device",
        type=str,
        default="cuda",
        help="Device used when computing next-token logits for HTML (e.g. 'cpu' or 'cuda')",
    )
    args = parser.parse_args()
    
    os.makedirs(args.out_dir, exist_ok=True)
    
    print("=" * 80)
    print("SAE Dashboard Neuron Visualization")
    print("=" * 80)
    
    # Load model
    print(f"\n[1/6] Loading model from {args.ckpt}")
    hooked_model = create_hooked_model(args.ckpt, args.device)
    model_cfg = hooked_model.model.cfg
    
    hidden_size = int(model_cfg.mlp_mult * model_cfg.n_embd)
    print(f"   Model config: n_layer={model_cfg.n_layer}, n_embd={model_cfg.n_embd}")
    print(f"   MLP hidden size: {hidden_size}")
    
    # Determine which layers to process
    if args.layers is not None:
        if args.layers.lower() == "all":
            layer_indices = list(range(model_cfg.n_layer))
        else:
            try:
                layer_indices = [int(x.strip()) for x in args.layers.split(",")]
                # Validate layer indices
                for li in layer_indices:
                    if li < 0 or li >= model_cfg.n_layer:
                        raise ValueError(f"Layer index {li} is out of range [0, {model_cfg.n_layer})")
            except ValueError as e:
                raise ValueError(f"Invalid --layers format: {args.layers}. Use comma-separated integers or 'all'. Error: {e}")
    else:
        layer_indices = [args.layer_idx]
    
    print(f"   Processing layers: {layer_indices}")
    
    # Get input embeddings and unembedding matrix (shared across all layers)
    input_embeddings = hooked_model.model.wte.weight.data.cpu().numpy()  # [vocab_size, n_embd]
    unembedding_matrix = hooked_model.model.lm_head.weight.data.cpu().numpy()  # [vocab_size, n_embd]
    
    # Load data (shared across all layers)
    print(f"\n[2/6] Loading {args.split} data")
    data_path = os.path.join(args.data_dir, f"{args.split}.bin")
    data = load_bin(data_path)
    
    # Collect token sequences (shared across all layers)
    print(f"   Collecting {args.num_sequences} sequences")
    g = torch.Generator(device="cpu")
    g.manual_seed(int(args.data_seed))
    
    tokens_list = []
    num_batches = (args.num_sequences + args.batch_size - 1) // args.batch_size
    
    for _ in tqdm(range(num_batches), desc="   Collecting sequences"):
        xb, _ = get_batch(data, args.batch_size, args.block_size, args.device, g)
        tokens_list.append(xb.cpu())
        if len(tokens_list) * args.batch_size >= args.num_sequences:
            break
    
    tokens = torch.cat(tokens_list, dim=0)[:args.num_sequences].to(args.device)
    print(f"   Collected {tokens.shape[0]} sequences of length {tokens.shape[1]}")

    # Use int32 for tokens (already optimized)
    tokens_flat = tokens.flatten().cpu().numpy().astype(np.int32)

    summary_dir = args.summary_dir or os.path.join(args.out_dir, "summaries")
    if not args.html_only:
        os.makedirs(summary_dir, exist_ok=True)

    # Import HTML module once
    import sys
    import importlib.util
    html_module_path = os.path.join(os.path.dirname(__file__), "create_html_dashboard.py")
    html_module = None
    if os.path.exists(html_module_path):
        spec = importlib.util.spec_from_file_location("create_html_dashboard", html_module_path)
        html_module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(html_module)

    total_tokens = tokens_flat.shape[0]

    layer_to_neurons_for_index: dict[int, list[int]] = {}

    # Prepare shared rendering caches once (tokens decoding + normalized embedding matrices + optional prediction model)
    lazy_token_decoder = None
    normalized_input_embeddings = None
    normalized_unembedding = None
    prediction_model = None
    if html_module is not None and args.phase in ("examples", "both"):
        try:
            print("\n[HTML] Preparing shared caches (lazy token decoder, embedding normalization, prediction model)")
            # Use lazy token decoder instead of pre-decoding all tokens
            if tiktoken is not None:
                try:
                    enc = tiktoken.get_encoding("gpt2")
                    lazy_token_decoder = LazyTokenDecoder(tokens_flat, enc)
                    print(f"   ✓ Lazy token decoder ready (will decode on-demand)")
                except Exception:
                    lazy_token_decoder = LazyTokenDecoder(tokens_flat, None)
            else:
                lazy_token_decoder = LazyTokenDecoder(tokens_flat, None)
            
            normalized_input_embeddings = html_module.normalize_embedding_matrix(input_embeddings)
            normalized_unembedding = html_module.normalize_embedding_matrix(unembedding_matrix)
            if not args.no_next_token_predictions:
                prediction_model = create_hooked_model(args.ckpt, args.html_prediction_device)
                prediction_model.eval()
        except Exception as e:
            print(f"   ⚠ Could not build shared HTML caches: {e}")
            lazy_token_decoder = None
            normalized_input_embeddings = None
            normalized_unembedding = None
            prediction_model = None

    # Process each layer
    for layer_idx in layer_indices:
        print("\n" + "=" * 80)
        print(f"Processing Layer {layer_idx}")
        print("=" * 80)

        # Select neurons for this layer
        if args.all_neurons:
            print(f"\n[3/6] Processing ALL {hidden_size} neurons for layer {layer_idx}")
            neuron_indices = sorted(list(range(hidden_size)))
            print(f"   Total neurons: {len(neuron_indices)}")
        else:
            print(f"\n[3/6] Selecting {args.num_neurons} random neurons for layer {layer_idx}")
            random.seed(args.neuron_seed + layer_idx)  # Different seed per layer
            np.random.seed(args.neuron_seed + layer_idx)
            neuron_indices = sorted(random.sample(range(hidden_size), args.num_neurons))
            print(f"   Selected neurons: {neuron_indices[:10]}... (showing first 10)")
        layer_to_neurons_for_index[int(layer_idx)] = [int(x) for x in neuron_indices]

        # Get MLP weights for this layer
        print(f"\n[4/6] Extracting MLP weights for layer {layer_idx}")
        layer_mlp = hooked_model.model.blocks[layer_idx].mlp
        encoder_weights = layer_mlp.fc1.weight.data.cpu().numpy()  # [hidden_size, n_embd] - up projection
        decoder_weights = layer_mlp.fc2.weight.data.cpu().numpy()  # [n_embd, hidden_size] - down projection

        hook_name = f"blocks.{layer_idx}.mlp.hook_post"
        neuron_indices_tensor = torch.tensor(neuron_indices, device=args.device, dtype=torch.long)

        summary_path = os.path.join(summary_dir, f"layer{layer_idx}_summary.pkl")
        summary_data = None
        needs_summary = (
            args.phase in ("summary", "both")
            or (args.phase == "examples" and (args.force_summary or not os.path.exists(summary_path)))
        )
        if needs_summary:
            print(f"\n[5/6] Computing summary statistics for layer {layer_idx}")
            summary_data = run_summary_phase(
                layer_idx,
                hook_name,
                hooked_model,
                tokens,
                neuron_indices,
                neuron_indices_tensor,
                args.summary_top_k,
                total_tokens,
                args.inference_batch_size,
            )
            if not args.html_only:
                with open(summary_path, "wb") as f:
                    pickle.dump(summary_data, f)
                print(f"   ✓ Saved summary to {summary_path}")
        else:
            if os.path.exists(summary_path):
                summary_data = load_summary(summary_path)
                print(f"\n[5/6] Reusing summary data from {summary_path}")
            else:
                # html_only + phase=examples in a clean dir: compute summary in-memory.
                print(f"\n[5/6] No cached summary found; computing in-memory for layer {layer_idx}")
                summary_data = run_summary_phase(
                    layer_idx,
                    hook_name,
                    hooked_model,
                    tokens,
                    neuron_indices,
                    neuron_indices_tensor,
                    args.summary_top_k,
                    total_tokens,
                    args.inference_batch_size,
                )

        if args.phase in ("examples", "both"):
            if summary_data is None:
                summary_data = load_summary(summary_path)
            print(f"\n[6/6] Selecting curated examples for layer {layer_idx}")
            dashboard_data = run_example_phase(
                layer_idx,
                hook_name,
                hooked_model,
                tokens,
                tokens_flat,
                neuron_indices,
                neuron_indices_tensor,
                summary_data,
                args,
                hidden_size,
                encoder_weights,
                decoder_weights,
                input_embeddings,
                unembedding_matrix,
                args.store_large_matrices,
            )
            data_path = os.path.join(args.out_dir, f"layer{layer_idx}_dashboard_data.pkl")
            if not args.html_only:
                with open(data_path, "wb") as f:
                    pickle.dump(dashboard_data, f)
                print(f"   ✓ Saved dashboard data to {data_path}")

            # Determine if we should skip HTML generation (only if explicitly requested)
            skip_html = args.skip_individual_html
            
            if html_module is not None and not skip_html:
                try:
                    layer_output_dir = os.path.join(args.out_dir, f"layer{layer_idx}")
                    os.makedirs(layer_output_dir, exist_ok=True)
                    
                    html_module.create_dashboard_html(
                        activations=None,
                        tokens=dashboard_data["tokens"],
                        neuron_indices=dashboard_data["neuron_indices"],
                        encoder_weights=dashboard_data["encoder_weights"],
                        decoder_weights=dashboard_data["decoder_weights"],
                        layer_idx=dashboard_data["layer_idx"],
                        hidden_size=dashboard_data["hidden_size"],
                        output_dir=layer_output_dir,
                        block_size=dashboard_data["block_size"],
                        context_window=30,
                        max_examples_per_quantile=args.max_examples_per_quantile,
                        dedup_radius=args.dedup_radius,
                        next_token_top_k=args.next_token_top_k,
                        input_embeddings=dashboard_data.get("input_embeddings"),
                        unembedding_matrix=dashboard_data.get("unembedding_matrix"),
                        ckpt_path=dashboard_data.get("ckpt_path"),
                        prediction_device=args.html_prediction_device,
                        neuron_examples=dashboard_data["neuron_examples"],
                        create_index=False,
                        token_strings=None,
                        normalized_input_embeddings=normalized_input_embeddings,
                        normalized_unembedding=normalized_unembedding,
                        prediction_model=prediction_model,
                        enable_next_token_predictions=(not args.no_next_token_predictions),
                    )
                    print(f"   ✓ Created HTML dashboard at {layer_output_dir}")
                except Exception as e:
                    print(f"   ⚠ Could not create HTML dashboard: {e}")
                    print(f"   You can create it manually: python -m dashboards.create_html_dashboard --data_file {data_path}")
            elif skip_html:
                print(f"   ⏭️  Skipped HTML generation for {len(dashboard_data['neuron_indices'])} neurons")
                print(f"   📊 Data saved to {data_path} - use to generate HTML later if needed")
                if html_module is not None:
                    try:
                        layer_output_dir = os.path.join(args.out_dir, f"layer{layer_idx}")
                        os.makedirs(layer_output_dir, exist_ok=True)
                        # Create a simple index pointing to the data file
                        index_path = os.path.join(layer_output_dir, "index.html")
                        with open(index_path, "w") as f:
                            f.write(f"""<!DOCTYPE html>
<html>
<head>
    <title>Layer {layer_idx} Dashboard Data</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 40px; background: #f5f5f5; }}
        .container {{ background: white; padding: 30px; border-radius: 8px; box-shadow: 0 2px 4px rgba(0,0,0,0.1); }}
        h1 {{ color: #333; }}
        p {{ color: #666; line-height: 1.6; }}
        code {{ background: #f0f0f0; padding: 2px 6px; border-radius: 3px; }}
        .info {{ background: #e3f2fd; padding: 15px; border-left: 4px solid #2196F3; margin: 20px 0; }}
    </style>
</head>
<body>
    <div class="container">
        <h1>Layer {layer_idx} - {len(dashboard_data['neuron_indices'])} Neurons</h1>
        <div class="info">
            <strong>HTML generation was skipped for performance.</strong><br>
            Data has been saved and can be used to generate individual neuron pages later.
        </div>
        <h2>Data Summary</h2>
        <p><strong>Neurons analyzed:</strong> {len(dashboard_data['neuron_indices'])}</p>
        <p><strong>Total tokens:</strong> {len(dashboard_data['tokens'])}</p>
        <p><strong>Data file:</strong> <code>{data_path}</code></p>
        
        <h2>To Generate Individual HTML Files Later</h2>
        <p>Run the HTML generation manually:</p>
        <code style="display: block; background: #f0f0f0; padding: 10px; border-radius: 4px; margin: 10px 0;">
        python -m dashboards.create_html_dashboard --data_file {data_path}
        </code>
        
        <h2>Performance</h2>
        <p>By skipping individual neuron files for large neuron counts, the generation completes much faster.</p>
        <p>All neuron data is preserved in the pickle file for future use.</p>
    </div>
</body>
</html>""")
                        print(f"   ✓ Created placeholder index at {index_path}")
                    except Exception as e:
                        print(f"   ⚠ Could not create placeholder index: {e}")
            else:
                print("   ⚠ HTML dashboard module not found")
                print(f"   You can create it manually: python -m dashboards.create_html_dashboard --data_file {data_path}")

    # Create a single top-level index.html that can navigate layer+neuron
    if html_module is not None and args.phase in ("examples", "both"):
        try:
            html_module.create_global_search_index(
                root_output_dir=args.out_dir,
                layers=[int(x) for x in layer_indices],
                example_neurons_by_layer=layer_to_neurons_for_index,
                max_buttons=20,
            )
            print(f"   ✓ Created global index at {os.path.join(args.out_dir, 'index.html')}")
        except Exception as e:
            print(f"   ⚠ Could not create global index.html: {e}")
    
    print("\n" + "=" * 80)
    print("Done! Processed layers:", layer_indices)
    print("=" * 80)


if __name__ == "__main__":
    main()

