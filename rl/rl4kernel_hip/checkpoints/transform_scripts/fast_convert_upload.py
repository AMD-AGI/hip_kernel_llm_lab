#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Fast FSDP→HF converter: merges shards and saves directly via safetensors,
skipping model instantiation entirely."""

import os
import re
import gc
import json
import shutil
import time
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed

import torch
from safetensors.torch import save_file
from huggingface_hub import HfApi, upload_folder


def _load_one_shard(checkpoint_dir, world_size, rank, t0):
    path = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
    print(f"[{time.time()-t0:.1f}s] Loading shard {rank}...")
    sd = torch.load(path, map_location="cpu", weights_only=False)
    local = {}
    for k, v in sd.items():
        if isinstance(v, torch.Tensor):
            local[k] = (v._local_tensor.clone() if hasattr(v, "_local_tensor") else v.clone())
        else:
            local[k] = v
    del sd
    print(f"[{time.time()-t0:.1f}s] Loaded shard {rank}")
    return rank, local


def fast_convert(checkpoint_dir, hf_config_dir, output_dir, load_workers=4):
    t0 = time.time()

    # Detect world_size
    world_size = 0
    for f in os.listdir(checkpoint_dir):
        m = re.match(r"model_world_size_(\d+)_rank_0\.pt", f)
        if m:
            world_size = int(m.group(1))
            break
    assert world_size > 0, f"No model shards found in {checkpoint_dir}"
    print(f"[{time.time()-t0:.1f}s] Found {world_size} shards")

    # Load shards in parallel to reduce single-stream I/O bottleneck.
    load_workers = max(1, min(int(load_workers), world_size))
    print(f"[{time.time()-t0:.1f}s] Using {load_workers} parallel loading workers")
    shards = [None] * world_size
    with ThreadPoolExecutor(max_workers=load_workers) as executor:
        futures = [
            executor.submit(_load_one_shard, checkpoint_dir, world_size, rank, t0)
            for rank in range(world_size)
        ]
        for fut in as_completed(futures):
            rank, local = fut.result()
            shards[rank] = local
    print(f"[{time.time()-t0:.1f}s] All shards loaded")

    # Merge and immediately free shard data per-key
    keys = list(shards[0].keys())
    merged = {}
    for key in keys:
        parts = [shards[r][key] for r in range(world_size)]
        if isinstance(parts[0], torch.Tensor):
            merged[key] = torch.cat(parts, dim=0).contiguous().half()
        else:
            merged[key] = parts[0]
        for r in range(world_size):
            del shards[r][key]
    del shards
    gc.collect()
    print(f"[{time.time()-t0:.1f}s] Merged {len(merged)} parameters")

    # Save as sharded safetensors (~5GB per shard)
    os.makedirs(output_dir, exist_ok=True)
    for fname in os.listdir(output_dir):
        if fname.endswith(".safetensors") or fname == "model.safetensors.index.json":
            os.remove(os.path.join(output_dir, fname))
    MAX_SHARD = 5 * 1024**3

    sorted_keys = sorted(k for k in merged if isinstance(merged[k], torch.Tensor))
    file_shards = []
    cur, cur_sz = {}, 0
    for k in sorted_keys:
        t = merged[k]
        sz = t.numel() * t.element_size()
        if cur_sz + sz > MAX_SHARD and cur:
            file_shards.append(cur)
            cur, cur_sz = {}, 0
        cur[k] = t
        cur_sz += sz
    if cur:
        file_shards.append(cur)

    weight_map = {}
    total_size = 0
    n = len(file_shards)
    for i, shard_dict in enumerate(file_shards):
        fname = f"model-{i+1:05d}-of-{n:05d}.safetensors"
        fpath = os.path.join(output_dir, fname)
        print(f"[{time.time()-t0:.1f}s] Saving {fname} ({len(shard_dict)} tensors)...")
        save_file(shard_dict, fpath)
        for k in shard_dict:
            weight_map[k] = fname
            total_size += shard_dict[k].numel() * shard_dict[k].element_size()

    index = {
        "metadata": {"total_size": total_size},
        "weight_map": weight_map,
    }
    with open(os.path.join(output_dir, "model.safetensors.index.json"), "w") as f:
        json.dump(index, f, indent=2)
    print(f"[{time.time()-t0:.1f}s] Saved index ({n} shards, {total_size/1e9:.2f} GB)")

    # Copy config + tokenizer files
    for fname in os.listdir(hf_config_dir):
        src = os.path.join(hf_config_dir, fname)
        if os.path.isfile(src):
            shutil.copy2(src, os.path.join(output_dir, fname))
    print(f"[{time.time()-t0:.1f}s] Copied config & tokenizer")

    del merged
    gc.collect()
    print(f"[{time.time()-t0:.1f}s] Conversion complete!")
    return output_dir


def upload(folder, repo_id, path_in_repo, token):
    t0 = time.time()
    api = HfApi(token=token)
    api.create_repo(repo_id=repo_id, repo_type="model", exist_ok=True, private=True)
    print(f"[{time.time()-t0:.1f}s] Uploading to {repo_id}/{path_in_repo}...")
    upload_folder(
        folder_path=folder,
        repo_id=repo_id,
        repo_type="model",
        path_in_repo=path_in_repo,
        commit_message=f"Upload {path_in_repo}",
        ignore_patterns=["*.ipynb", "*.tmp", ".uploaded"],
        token=token,
    )
    print(f"[{time.time()-t0:.1f}s] Upload complete: https://huggingface.co/{repo_id}/tree/main/{path_in_repo}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", required=True)
    parser.add_argument("--config_dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--repo_id", default=None)
    parser.add_argument("--path_in_repo", default=None)
    parser.add_argument("--token", default=None)
    parser.add_argument("--upload_only", action="store_true")
    parser.add_argument("--load_workers", type=int, default=4)
    args = parser.parse_args()

    if not args.upload_only:
        fast_convert(args.ckpt, args.config_dir, args.output, load_workers=args.load_workers)

    if args.repo_id:
        upload(args.output, args.repo_id, args.path_in_repo, args.token)
