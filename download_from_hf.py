#!/usr/bin/env python3
"""
Download Sparse Transformer checkpoints from the Hugging Face Hub.

Example:
    python download_from_hf.py --folder main_mlp_mult_32_K_8
    python download_from_hf.py --folder main_mlp_mult_32_K_8 --pattern ckpt_iter300000.pt
"""

import argparse
import os
from huggingface_hub import snapshot_download

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download a folder from HF Hub")
    parser.add_argument("--repo_name", default="AntonKorznikov/SparseTransformer",  help="HF repo name (e.g., username/repo-name)")
    parser.add_argument("--local_path", default="out",  help="Local folder to download to")
    parser.add_argument("--folder", default="main_mlp_mult_32_K_8", help="Model folder inside the repo (e.g. main_mlp_mult_32_K_8). Pass an empty string to download the whole repo (~all 35 models x 11 checkpoints).")
    parser.add_argument("--token", default=None, help="HF API token (optional; the model repo is public). Can also be set via HF_TOKEN.")
    
    parser.add_argument("--pattern", default="*", help="Glob for files inside --folder (e.g. ckpt_iter300000.pt)")
    args = parser.parse_args()
    
    # Use token from argument or environment variable
    token = args.token or os.environ.get("HF_TOKEN")
    
    # Build allow_patterns if a specific folder was requested
    allow_patterns = None
    if args.folder:
        # Ensure the pattern matches files inside that folder
        allow_patterns = [f"{args.folder}/{args.pattern}"]
        print(f"Downloading only folder '{args.folder}' from {args.repo_name}...")
    else:
        print(f"Downloading entire repository {args.repo_name}...")
    
    snapshot_download(
        repo_id=args.repo_name,
        local_dir=args.local_path,
        token=token,
        allow_patterns=allow_patterns,
        # Optional: add ignore_patterns if needed
    )
    print("✓ Done!")