"""
Adapter to make SparseTransformerLM compatible with hook-based activation extraction.

This creates a wrapper that allows extracting activations from the custom model using hooks.
"""

from __future__ import annotations

import torch
import torch.nn as nn

from model_transformer import SparseTransformerLM, TransformerConfig


class TransformerLensCompatibleModel(nn.Module):
    """
    Wrapper that makes SparseTransformerLM compatible with hook-based activation extraction.
    
    This allows hooking into the model to extract activations at specific points.
    """
    
    def __init__(self, model: SparseTransformerLM):
        super().__init__()
        self.model = model
        self.cfg = model.cfg
        self.tokenizer = None  # Will be set externally
        
        # Store activations for hooks
        self._activations = {}
        
        # Register hooks on the model
        self._register_hooks()
    
    def _register_hooks(self):
        """Register forward hooks to capture activations."""
        def embed_hook_fn(module, input, output):
            # Store token embedding output (pre-position-add). This matches the hook name
            # expected by create_attribution_graph.py.
            self._activations["transformer.wte.hook_embed"] = output.detach()

        def make_pre_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # Store fc1 output (pre-activations)
                hook_name = f"blocks.{layer_idx}.mlp.hook_pre"
                self._activations[hook_name] = output.detach()
            return hook_fn
        
        def make_post_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # Store BatchTopK output (post-activations: after ReLU + top-k sparsification)
                # This is the final activation after BatchTopK, where only top-k neurons are non-zero
                hook_name = f"blocks.{layer_idx}.mlp.hook_post"
                self._activations[hook_name] = output.detach()
            return hook_fn

        def make_mlp_out_hook(layer_idx: int):
            def hook_fn(module, input, output):
                # Store MLP output (after fc2) before it is added back to the residual stream.
                # This is the vector y whose norm we use for MLP norm-matching in
                # create_attribution_graph.py.
                hook_name = f"blocks.{layer_idx}.mlp.hook_out"
                self._activations[hook_name] = output.detach()
            return hook_fn
        
        # Register hooks on fc1 (pre-activations) and BatchTopK (post-activations)
        # Also register a hook on token embeddings.
        self.model.wte.register_forward_hook(embed_hook_fn)
        for i, blk in enumerate(self.model.blocks):
            blk.mlp.fc1.register_forward_hook(make_pre_hook(i))
            blk.mlp.act.register_forward_hook(make_post_hook(i))
            blk.mlp.fc2.register_forward_hook(make_mlp_out_hook(i))
    
    def forward(self, tokens: torch.Tensor, return_type: str = "logits"):
        """
        Forward pass compatible with transformer_lens API.
        
        Args:
            tokens: Input token IDs [batch, seq_len]
            return_type: What to return ("logits" or "loss")
            
        Returns:
            Logits tensor [batch, seq_len, vocab_size]
        """
        # Clear previous activations
        self._activations.clear()
        
        # Standard forward pass
        logits, loss = self.model(tokens, None)
        
        if return_type == "loss" and loss is not None:
            return loss
        return logits
    
    def get_activations(self, hook_name: str) -> torch.Tensor | None:
        """Get stored activations for a hook point."""
        return self._activations.get(hook_name)
    
    def run_with_cache(self, tokens: torch.Tensor):
        """
        Run model and return logits with cached activations.
        
        Compatible with transformer_lens API.
        """
        logits = self.forward(tokens)
        return logits, self._activations.copy()


def create_hooked_model(ckpt_path: str, device: str = "cuda") -> TransformerLensCompatibleModel:
    """
    Load a SparseTransformerLM and wrap it for transformer_lens compatibility.
    
    Args:
        ckpt_path: Path to model checkpoint
        device: Device to load model on
        
    Returns:
        TransformerLensCompatibleModel instance
    """
    ckpt = torch.load(ckpt_path, map_location="cpu")
    mcfg = TransformerConfig(**ckpt["model_cfg"])
    model = SparseTransformerLM(mcfg).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    
    hooked_model = TransformerLensCompatibleModel(model)
    return hooked_model

