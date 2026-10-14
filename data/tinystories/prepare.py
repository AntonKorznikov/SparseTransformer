from __future__ import annotations

import os
import pickle

import numpy as np
import tiktoken
from datasets import load_dataset
from tqdm import tqdm


def write_split(ds, out_path: str, enc, eot: int):
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, "wb") as f:
        for ex in tqdm(ds, desc=f"writing {os.path.basename(out_path)}"):
            # dataset column name can vary; prefer "text"
            if "text" in ex:
                text = ex["text"]
            else:
                # fallback: first string-valued column
                text = None
                for _, v in ex.items():
                    if isinstance(v, str):
                        text = v
                        break
                if text is None:
                    raise KeyError(f"could not find a text field in example keys={list(ex.keys())}")
            ids = enc.encode_ordinary(text)
            ids.append(eot)
            arr = np.array(ids, dtype=np.uint16)
            arr.tofile(f)


def main():
    out_dir = os.path.join("data", "tinystories")
    os.makedirs(out_dir, exist_ok=True)

    # The canonical TinyStories dataset on HF is typically roneneldan/TinyStories.
    ds = load_dataset("roneneldan/TinyStories")
    if "validation" in ds:
        train_ds = ds["train"]
        val_ds = ds["validation"]
    elif "valid" in ds:
        train_ds = ds["train"]
        val_ds = ds["valid"]
    else:
        # fallback: create a small validation slice
        train_ds = ds["train"]
        val_ds = train_ds.select(range(min(10_000, len(train_ds))))
        train_ds = train_ds.select(range(min(len(train_ds), 2_000_000)))

    enc = tiktoken.get_encoding("gpt2")
    eot = enc.eot_token

    write_split(train_ds, os.path.join(out_dir, "train.bin"), enc, eot)
    write_split(val_ds, os.path.join(out_dir, "val.bin"), enc, eot)

    meta = {
        "vocab_size": enc.n_vocab,
        "eot_token": eot,
        "tokenizer": "tiktoken:gpt2",
        "dataset": "roneneldan/TinyStories",
    }
    with open(os.path.join(out_dir, "meta.pkl"), "wb") as f:
        pickle.dump(meta, f)

    print("done")


if __name__ == "__main__":
    main()


