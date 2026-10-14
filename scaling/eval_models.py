#!/usr/bin/env python3
"""
eval_models.py

Полностью автоматический пересчёт метрик для всех конфигураций,
используя только финальные чекпоинты на Hugging Face Hub.

Запуск:
    python -m scaling.eval_models
"""

import os
import pickle
from types import SimpleNamespace

import numpy as np
import torch
from tqdm import tqdm
from huggingface_hub import hf_hub_download

# Импорты ваших модулей (убедитесь, что пути корректны)
from model_transformer import SparseTransformerLM, TransformerConfig
from utils.common import (
    device_from_cfg,
    get_batch,
    load_bin,
    load_config,
    StreamingDataset,
)

# ============================================================
# ПУТЬ К КОНФИГУ (измените на свой)
# ============================================================
CONFIG_PATH = "config/main_config.py"

# Параметры вычислений (можно менять)
EVAL_BATCHES = 300            # батчей для усреднения val loss
DEATH_TOKENS = 10_000_000          # токенов для оценки мёртвых нейронов
ACTIVE_BATCHES = 20           # батчей для active_per_token_mean
TEMP_CACHE_DIR = None         # папка для кэша (None = стандартная)
OUT_PATH = "scaling/recomputed_metrics_rerun.pkl"

# ============================================================

def download_checkpoint(repo_id, token, subfolder, filename):
    """Скачать файл из подпапки репозитория."""
    return hf_hub_download(
        repo_id=repo_id,
        filename=f"{subfolder}/{filename}",
        token=token,
        cache_dir=TEMP_CACHE_DIR,
        force_download=False,
    )

def compute_validation_loss(model, val_data, cfg, device, use_streaming, data_gen):
    """Усреднённый validation loss."""
    model.eval()
    losses = []
    with torch.no_grad():
        for _ in range(EVAL_BATCHES):
            if use_streaming:
                xb, yb = val_data.get_batch(cfg.batch_size, cfg.block_size, device)
            else:
                xb, yb = get_batch(val_data, cfg.batch_size, cfg.block_size, device, data_gen)
            _, loss = model(xb, yb)
            losses.append(loss.item())
    model.train()
    return float(np.mean(losses))

def compute_neuron_death_proportion(model, val_data, cfg, device, use_streaming, data_gen):
    model.eval()
    hidden_sizes = [blk.mlp.fc1.weight.shape[0] for blk in model.blocks]
    # накопители для каждого слоя
    acc_counts = [torch.zeros(hsz, dtype=torch.int64, device='cpu') for hsz in hidden_sizes]

    total_processed = 0
    pbar = tqdm(total=DEATH_TOKENS, desc="Collecting activations")
    while total_processed < DEATH_TOKENS:
        if use_streaming:
            xb, yb = val_data.get_batch(cfg.batch_size, cfg.block_size, device)
        else:
            xb, yb = get_batch(val_data, cfg.batch_size, cfg.block_size, device, data_gen)
        with torch.no_grad():
            _ = model(xb, yb)
        n_tokens = xb.shape[0] * xb.shape[1]
        total_processed += n_tokens

        # Накопление счетчиков (приведение к cpu)
        for li, blk in enumerate(model.blocks):
            counts = blk.mlp.act._last_feature_counts  # уже на cpu? Нет, на device
            if counts is not None:
                acc_counts[li] += counts.cpu()

        pbar.update(n_tokens)
    pbar.close()

    dead_threshold = 1e-6
    dead_props = []
    for li, counts in enumerate(acc_counts):
        freq = counts.float() / total_processed
        dead_prop = (freq < dead_threshold).float().mean().item()
        dead_props.append(dead_prop)
    return float(np.mean(dead_props))

def compute_active_per_token_mean(model, val_data, cfg, device, use_streaming, data_gen):
    """Усреднённое значение batchtopk_mean_active_per_token_mean."""
    model.eval()
    active_means = []
    with torch.no_grad():
        for _ in range(ACTIVE_BATCHES):
            if use_streaming:
                xb, yb = val_data.get_batch(cfg.batch_size, cfg.block_size, device)
            else:
                xb, yb = get_batch(val_data, cfg.batch_size, cfg.block_size, device, data_gen)
            _ = model(xb, yb)
            stats = model.batchtopk_stats()
            if stats:
                active_means.append(stats["batchtopk_mean_active_per_token_mean"])
    model.train()
    return float(np.mean(active_means)) if active_means else 0.0

def main():
    print("Загрузка конфигурации...")
    cfg = load_config(CONFIG_PATH)

    # Параметры сетки
    if not hasattr(cfg, "mlp_mult_values"):
        raise AttributeError("Config must define mlp_mult_values")
    mlp_mult_values = list(cfg.mlp_mult_values)

    if not hasattr(cfg, "topk_values"):
        raise AttributeError("Config must define topk_values")
    topk_values = list(cfg.topk_values)

    if getattr(cfg, "with_no_topk", False):
        topk_values.append(None)

    # Hugging Face параметры

    hf_repo_id = getattr(cfg, "hf_repo_id", None) or "AntonKorznikov/SparseTransformer"
    hf_token = getattr(cfg, "hf_token", None) or os.environ.get("HF_TOKEN")
    run_name = cfg.run_name
    max_iters = cfg.max_iters

    print(f"Конфигураций: {len(mlp_mult_values)} × {len(topk_values)} = {len(mlp_mult_values)*len(topk_values)}")
    print(f"Репозиторий: {hf_repo_id}")
    print(f"Итерация чекпоинта: {max_iters}")

    # Валидационные данные
    use_streaming = hasattr(cfg, "dataset_name") and cfg.dataset_name is not None
    if use_streaming:
        tokenizer_name = getattr(cfg, "tokenizer", "gpt2")
        val_split = getattr(cfg, "val_split", "validation")
        val_data = StreamingDataset(
            cfg.dataset_name, val_split, tokenizer_name,
            seed=cfg.data_seed + 1
        )
        vocab_size = val_data.vocab_size
        print(f"Стриминговый датасет: {cfg.dataset_name}")
    else:
        if not hasattr(cfg, "data_dir") or not cfg.data_dir:
            raise ValueError("data_dir не задан в конфиге")
        val_data = load_bin(os.path.join(cfg.data_dir, "val.bin"))
        with open(os.path.join(cfg.data_dir, "meta.pkl"), "rb") as f:
            meta = pickle.load(f)
        vocab_size = meta["vocab_size"]
        print(f"Бинарные данные из {cfg.data_dir}")

    device = device_from_cfg(cfg)
    data_gen = torch.Generator(device="cpu")
    data_gen.manual_seed(cfg.data_seed)

    results = {
        "val_loss": {},
        "dead_proportion": {},
        "active_per_token": {},
    }

    total = len(mlp_mult_values) * len(topk_values)
    idx = 0

    for mlp_mult in mlp_mult_values:
        for topk in topk_values:
            idx += 1
            topk_str = str(topk) if topk is not None else "None"
            print(f"\n[{idx}/{total}] mlp_mult={mlp_mult}, K={topk_str}")

            subfolder = f"{run_name}_mlp_mult_{mlp_mult}_K_{topk_str}"
            ckpt_filename = f"ckpt_iter{max_iters}.pt"

            try:
                ckpt_path = download_checkpoint(hf_repo_id, hf_token, subfolder, ckpt_filename)
                print(f"  Чекпоинт загружен: {ckpt_path}")
            except Exception as e:
                print(f"  Ошибка скачивания: {e}")
                continue

            ckpt = torch.load(ckpt_path, map_location="cpu", weights_only=False)
            data_gen.manual_seed(cfg.data_seed)  # same val batches for every model

            model = SparseTransformerLM(TransformerConfig(**ckpt["model_cfg"])).to(device)
            model.load_state_dict(ckpt["model"])

            # Метрики
            val_loss = compute_validation_loss(model, val_data, cfg, device, use_streaming, data_gen)
            print(f"  val_loss = {val_loss:.4f}")

            active_mean = compute_active_per_token_mean(model, val_data, cfg, device, use_streaming, data_gen)
            print(f"  active_per_token_mean = {active_mean:.2f}")

            dead_prop = compute_neuron_death_proportion(model, val_data, cfg, device, use_streaming, data_gen)
            print(f"  dead proportion (freq<1e-6) = {dead_prop:.4f}")

            results["val_loss"][(mlp_mult, topk)] = val_loss
            results["active_per_token"][(mlp_mult, topk)] = active_mean
            results["dead_proportion"][(mlp_mult, topk)] = dead_prop

            del model, ckpt
            torch.cuda.empty_cache()

    # Сохраняем результаты
    # scaling/recomputed_metrics.pkl holds the numbers reported in the paper; do not overwrite it
    with open(OUT_PATH, "wb") as f:
        pickle.dump(results, f)
    print(f"\nSaved results to {OUT_PATH}")

    # Сводка
    print("\n" + "="*80)
    print("СВОДКА ПЕРЕСЧИТАННЫХ МЕТРИК")
    print("="*80)
    for (mp, tk), loss in results["val_loss"].items():
        dead = results["dead_proportion"][(mp, tk)]
        active = results["active_per_token"][(mp, tk)]
        tk_str = str(tk) if tk is not None else "None"
        print(f"mlp_mult={mp:3d}, K={tk_str:4s} | loss={loss:.4f} | dead={dead:.4f} | active={active:.2f}")

if __name__ == "__main__":
    main()