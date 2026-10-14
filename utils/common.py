from __future__ import annotations

import math
import os
import pickle
from types import SimpleNamespace
from typing import Iterator

import numpy as np
import torch
import tiktoken
from datasets import load_dataset


def load_config(py_path: str) -> SimpleNamespace:
    cfg = {}
    with open(py_path, "r", encoding="utf-8") as f:
        code = f.read()
    exec(compile(code, py_path, "exec"), cfg, cfg)
    cfg = {k: v for k, v in cfg.items() if not k.startswith("__")}
    return SimpleNamespace(**cfg)


def set_seeds(init_seed: int, deterministic: bool = False) -> None:
    torch.manual_seed(init_seed)
    np.random.seed(init_seed)
    if deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def get_lr(it: int, cfg: SimpleNamespace) -> float:
    if it < cfg.warmup_iters:
        return cfg.learning_rate * it / max(1, cfg.warmup_iters)
    if it > cfg.lr_decay_iters:
        return cfg.min_lr
    decay_ratio = (it - cfg.warmup_iters) / max(1, (cfg.lr_decay_iters - cfg.warmup_iters))
    coeff = 0.5 * (1.0 + math.cos(math.pi * decay_ratio))
    return cfg.min_lr + coeff * (cfg.learning_rate - cfg.min_lr)


def load_meta(data_dir: str) -> dict:
    with open(os.path.join(data_dir, "meta.pkl"), "rb") as f:
        return pickle.load(f)


def load_bin(path: str) -> np.memmap:
    return np.memmap(path, dtype=np.uint16, mode="r")


def get_batch(
    data: np.memmap,
    batch_size: int,
    block_size: int,
    device: str,
    g: torch.Generator,
):
    max_ix = len(data) - block_size - 1
    ix = torch.randint(low=0, high=max_ix, size=(batch_size,), generator=g)
    x = torch.stack([torch.from_numpy((data[int(i) : int(i) + block_size]).astype(np.int64)) for i in ix])
    y = torch.stack(
        [torch.from_numpy((data[int(i) + 1 : int(i) + 1 + block_size]).astype(np.int64)) for i in ix]
    )
    return x.to(device), y.to(device)


def device_from_cfg(cfg: SimpleNamespace) -> str:
    device = cfg.device
    if device.startswith("cuda") and not torch.cuda.is_available():
        print("warning: config requested CUDA but torch.cuda.is_available() is False; falling back to CPU")
        return "cpu"
    return device


class StreamingDataset:
    """Streaming dataset that tokenizes text on-the-fly."""
    def __init__(self, dataset_name: str, split: str, tokenizer_name: str = "gpt2", seed: int = 42):
        self.dataset_name = dataset_name
        self.split = split
        self.dataset_iter = iter(load_dataset(dataset_name, split=split, streaming=True))
        self.enc = tiktoken.get_encoding(tokenizer_name)
        self.vocab_size = self.enc.n_vocab
        self.token_buffer = []
        self.generator = torch.Generator()
        self.generator.manual_seed(seed)
        
    def _get_text_field(self, ex: dict) -> str:
        """Extract text from example, handling various field names."""
        if "text" in ex:
            return ex["text"]
        # fallback: first string-valued column
        for _, v in ex.items():
            if isinstance(v, str):
                return v
        raise KeyError(f"could not find a text field in example keys={list(ex.keys())}")
    
    def _fill_buffer(self, min_tokens: int):
        """Fill token buffer with at least min_tokens."""
        while len(self.token_buffer) < min_tokens:
            try:
                ex = next(self.dataset_iter)
                text = self._get_text_field(ex)
                tokens = self.enc.encode_ordinary(text)
                self.token_buffer.extend(tokens)
            except StopIteration:
                # Dataset exhausted, restart from beginning
                self.dataset_iter = iter(load_dataset(self.dataset_name, split=self.split, streaming=True))
    
    def get_batch(self, batch_size: int, block_size: int, device: str) -> tuple[torch.Tensor, torch.Tensor]:
        """Get a batch of sequences from the streaming dataset."""
        # Ensure we have enough tokens
        needed_tokens = batch_size * block_size + batch_size
        self._fill_buffer(needed_tokens)
        
        # Sample random starting positions
        max_start = len(self.token_buffer) - block_size - 1
        if max_start <= 0:
            # Not enough tokens, fill more
            self._fill_buffer(needed_tokens * 2)
            max_start = len(self.token_buffer) - block_size - 1
        
        starts = torch.randint(low=0, high=max_start, size=(batch_size,), generator=self.generator)
        
        x_list = []
        y_list = []
        for start in starts:
            start_idx = int(start)
            x_tokens = self.token_buffer[start_idx:start_idx + block_size]
            y_tokens = self.token_buffer[start_idx + 1:start_idx + 1 + block_size]
            x_list.append(torch.tensor(x_tokens, dtype=torch.int64))
            y_list.append(torch.tensor(y_tokens, dtype=torch.int64))
        
        x = torch.stack(x_list).to(device)
        y = torch.stack(y_list).to(device)
        
        # Remove used tokens from buffer (keep some overlap for efficiency)
        # Keep last block_size tokens for potential overlap
        keep_tokens = min(block_size, len(self.token_buffer))
        self.token_buffer = self.token_buffer[-keep_tokens:]
        
        return x, y


