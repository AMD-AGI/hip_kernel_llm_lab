# Copyright 2024 Bytedance Ltd. and/or its affiliates
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Unified generation & evaluation script.
This version **removes** the inner ``for n_sample in …`` loop and generates
all ``n_samples`` per‑prompt in **one single forward pass**, following the
batched strategy sketched in the reference implementation provided by the
user.  The rest of the pipeline – padding logic, JSON/Parquet IO, metric
computation, and optional response‑length analysis – remains unchanged.

Key changes
===========
1. **Batch repetition**  ‑ Each prompt in the current micro‑batch is
   duplicated ``n_samples`` times before tokenisation so that the model
   naturally emits *n* completions.
2. **Data‑parallel padding** ‑ Instead of ``pad_dataproto_to_divisor`` we
   follow the lighter "dummy sample" approach when the effective batch is
   not divisible by ``dp_size``.
3. **Post‑processing**    ‑ Text outputs are reshaped from a flat list of
   ``(n_data × n_samples)`` back to ``(n_data, n_samples)``.

NOTE: If your deployment still relies on tensor‑level padding via
``pad_dataproto_to_divisor`` you can uncomment the marked block – the two
approaches are functionally equivalent.
"""
import csv
import json
import os
from pprint import pprint
from typing import List

import hydra
import numpy as np
import pandas as pd
import ray
from omegaconf import OmegaConf
from tabulate import tabulate
from tqdm import tqdm

# Environment tweaks ---------------------------------------------------------
os.environ["NCCL_DEBUG"] = "WARN"
os.environ["TOKENIZERS_PARALLELISM"] = "true"
# os.environ["TORCH_COMPILE_DISABLE"] = "1"

# veRL / local helpers -------------------------------------------------------
from verl import DataProto
from verl.protocol import unpad_dataproto  # only needed if you keep the original pad logic
from verl.single_controller.ray import (
    RayClassWithInitArgs,
    RayResourcePool,
    RayWorkerGroup,
)
from verl.utils.fs import copy_local_path_from_hdfs
from verl.utils.hdfs_io import makedirs
from verl.utils.model import compute_position_id_with_mask
from verl.workers.fsdp_workers import ActorRolloutRefWorker

# ----------------------------------------------------------------------------
# Optional analysis helper (unchanged from the user‑provided reference)
# ----------------------------------------------------------------------------
# from response_analysis import save_response_analysis  # type: ignore  # assumes helper is saved as separate module


def save_response_analysis(output, tokenizer, config, n_data):
    """
    Processes the model output to:
      1. Remove all token IDs equal to 151643 from:
         - the generated tokens slice of input_ids
         - the prompts
         - the responses
      2. Convert each filtered sequence to string tokens
      3. Compute per-sequence lengths, reshape into [n_data, n_samples], and compute the mean
      4. Save all results into a single JSON file under
         <os.path.dirname(config.data.output_path)>/response_analysis.json
    """
    def filter_and_stats(tensor_batch):
        """
        Helper: given a batch of ID tensors, filter out 151643,
        convert to tokens, compute lengths, reshape lengths, and mean.
        Returns:
          filtered_ids, tokens, flat_lengths, lengths_2d, mean_length
        """
        # filter out the unwanted ID
        filtered = [row[row != 151643] for row in tensor_batch]
        # convert to token strings
        tokens = [tokenizer.convert_ids_to_tokens(seq) for seq in filtered]
        # compute flat lengths
        flat_lengths = [len(t) for t in filtered]
        # reshape into [n_data, n_samples]
        lengths_2d = (
            np.array(flat_lengths)
              .reshape(n_data, config.data.n_samples)
              .tolist()
        )
        # mean tokens per sequence
        mean_length = float(np.mean(lengths_2d))
        return filtered, tokens, flat_lengths, lengths_2d, mean_length

    # 1. Process generated slice from input_ids
    gen_slice = output.batch['input_ids']
    gen_filt, gen_tokens, gen_lens, gen_lens_2d, gen_mean = filter_and_stats(gen_slice)

    # 2. Process prompts
    prompts_tensor = output.batch['prompts']
    prm_filt, prm_tokens, prm_lens, prm_lens_2d, prm_mean = filter_and_stats(prompts_tensor)

    # 3. Process responses
    responses_tensor = output.batch['responses']
    rsp_filt, rsp_tokens, rsp_lens, rsp_lens_2d, rsp_mean = filter_and_stats(responses_tensor)

    # 4. Build JSON-serializable dictionary
    output_data = {
        # generated
        "generated_filtered":       [seq.tolist() for seq in gen_filt],
        # "generated_tokens":         gen_tokens,
        "generated_lengths":        gen_lens,
        "generated_lengths_2d":     gen_lens_2d,
        "generated_mean_tokens":    gen_mean,
        # prompts
        "prompts_filtered":         [seq.tolist() for seq in prm_filt],
        # "prompts_tokens":           prm_tokens,
        "prompts_lengths":          prm_lens,
        "prompts_lengths_2d":       prm_lens_2d,
        "prompts_mean_tokens":      prm_mean,
        # responses
        "responses_filtered":       [seq.tolist() for seq in rsp_filt],
        # "responses_tokens":         rsp_tokens,
        "responses_lengths":        rsp_lens,
        "responses_lengths_2d":     rsp_lens_2d,
        "responses_mean_tokens":    rsp_mean,
    }

    # 5. Save to JSON file
    output_dir = os.path.dirname(config.data.output_path)
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, "response_analysis.json")
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output_data, f, indent=2, ensure_ascii=False)
    
    print(f"✅ Saved all stats to {output_path}")
    print(f"📊 Mean Tokens:")
    print(f"  - Generated : {gen_mean:.2f}")
    print(f"  - Prompts   : {prm_mean:.2f}")
    print(f"  - Responses : {rsp_mean:.2f}")
    # return output_path



# ---------------------------------------------------------------------------
# Main driver
# ---------------------------------------------------------------------------


def _select_reward_fn(data_source: str):
    """Tiny wrapper to lazily import reward fns."""
    if data_source == "lighteval/MATH":
        from verl.utils.reward_score import math as math_reward  # noqa: WPS433 (dynamic import)

        return math_reward.compute_score
    # default (RL‑LM repo)
    from rllm.rewards.rl_reward import rllm_reward_fn  # noqa: WPS433

    return rllm_reward_fn


@hydra.main(config_path="config", config_name="generation", version_base=None)
def main(cfg):  # noqa: WPS231 (high complexity) – inevitable in orchestration script
    # ---------------------------------------------------------------------
    # 1.  Resolve & pretty‑print config
    # ---------------------------------------------------------------------
    pprint(OmegaConf.to_container(cfg, resolve=True))
    OmegaConf.resolve(cfg)

    # ---------------------------------------------------------------------
    # 2.  Shortcut: reuse existing outputs if they are already on disk
    # ---------------------------------------------------------------------
    if os.path.exists(cfg.data.output_path):
        dataset = _lazy_load_dataframe(cfg.data.output_path)
        print(
            f"⚠️  Found existing file at {cfg.data.output_path} – "
            "skipping generation and jumping to evaluation."
        )
        _run_evaluation(dataset, cfg)
        return

    # ---------------------------------------------------------------------
    # 3.  Prepare model & Ray worker group
    # ---------------------------------------------------------------------
    local_model_path = copy_local_path_from_hdfs(cfg.model.path)
    from verl.utils import hf_tokenizer  # local import to defer HF module loading

    tokenizer = hf_tokenizer(local_model_path)

    if cfg.rollout.temperature == 0.0:
        assert (
            cfg.data.n_samples == 1
        ), "When temperature = 0, n_samples must be 1."

    # ---------- read prompt dataset ------------------------------------
    dataset, chat_list = _read_prompt_dataset(cfg)
    tokenizer.padding_side = "left"
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # ---------- spin up Ray -------------------------------------------
    ray_cls = RayClassWithInitArgs(cls=ray.remote(ActorRolloutRefWorker), config=cfg, role="rollout")
    pool = RayResourcePool(process_on_nodes=[cfg.trainer.n_gpus_per_node] * cfg.trainer.nnodes)
    workers = RayWorkerGroup(resource_pool=pool, ray_cls_with_init=ray_cls)
    workers.init_model()

    dp_size = workers.world_size
    total_prompts = len(dataset)
    batch_size_cfg = cfg.data.batch_size
    n_batches = (total_prompts + batch_size_cfg - 1) // batch_size_cfg  # ceil‑div

    # ------------------------------------------------------------------
    # 4.  Generation loop (no inner n_sample loop!)
    # ------------------------------------------------------------------
    outputs_flat: List[str] = []

    for batch_idx in range(n_batches):
        print(f"[{batch_idx + 1}/{n_batches}] Tokenising batch …")
        start, end = batch_idx * batch_size_cfg, (batch_idx + 1) * batch_size_cfg
        batch_prompts = chat_list[start:end]

        # Duplicate each prompt n_samples times so the model generates all
        # completions in one call.
        repeated_batch = [chat for chat in batch_prompts for _ in range(cfg.data.n_samples)]

        model_inputs = tokenizer.apply_chat_template(
            repeated_batch,
            add_generation_prompt=True,
            padding=True,
            truncation=True,
            max_length=cfg.rollout.prompt_length,
            return_tensors="pt",
            return_dict=True,
            tokenize=True,
        )

        pos_ids = compute_position_id_with_mask(model_inputs["attention_mask"])
        batch_dict = {
            "input_ids": model_inputs["input_ids"],
            "attention_mask": model_inputs["attention_mask"],
            "position_ids": pos_ids,
        }
        data = DataProto.from_dict(batch_dict)
        real_bs = data.batch["input_ids"].shape[0]

        # ---- ensure divisibility wrt data‑parallel size --------------
        if real_bs % dp_size != 0:
            dummy = data[: dp_size - real_bs % dp_size]
            data = DataProto.concat([data, dummy])
            print(
                f"Padding batch with {len(dummy)} dummy samples "
                f"to satisfy dp_size = {dp_size}."
            )

        # --------------------------------------------------------------
        print(f"[{batch_idx + 1}/{n_batches}] Generating …")
        output = workers.generate_sequences(data)
        output = output[:real_bs]  # strip dummies

        # ------------------------------------------------------------------
        # Convert responses – we slice the last ``response_length`` tokens of
        # each *input_ids* sequence because ActorRolloutRefWorker concatenates
        # prompt and generated tokens.
        # ------------------------------------------------------------------
        gen_slice = output.batch["input_ids"][:, -cfg.rollout.response_length :]
        text_batch = tokenizer.batch_decode(gen_slice, skip_special_tokens=True)
        outputs_flat.extend(text_batch)

    # ------------------------------------------------------------------
    # 5.  Reshape & attach to dataframe
    # ------------------------------------------------------------------
    n_data = len(outputs_flat) // cfg.data.n_samples
    outputs_2d = np.array(outputs_flat).reshape(n_data, cfg.data.n_samples).tolist()
    dataset["responses"] = outputs_2d

    # Optional detailed token stats ------------------------------------
    # if cfg.get("save_response_analysis", True):
    save_response_analysis(output, tokenizer, cfg, n_data)

    # ------------------------------------------------------------------
    # 6.  Persist to Parquet & evaluate
    # ------------------------------------------------------------------
    out_dir = os.path.dirname(cfg.data.output_path)
    makedirs(out_dir, exist_ok=True)
    dataset.to_parquet(cfg.data.output_path)

    _run_evaluation(dataset, cfg)


# ---------------------------------------------------------------------------
# Helper functions
# ---------------------------------------------------------------------------

def _read_prompt_dataset(cfg):
    """Load dataset and return (DataFrame, chat_list)."""
    try:
        df = pd.read_parquet(cfg.data.path)
        chat = [c.tolist() for c in df[cfg.data.prompt_key]]
        return df, chat
    except Exception:
        with open(cfg.data.path.replace(".parquet", ".json"), "r", encoding="utf-8") as fh:
            df = pd.read_json(fh)
        chat = df[cfg.data.prompt_key].tolist()
        return df, chat


def _lazy_load_dataframe(path: str):
    """Try Parquet → JSON → Polars fallback."""
    try:
        return pd.read_parquet(path)
    except Exception:
        try:
            with open(path.replace(".parquet", ".json"), "r", encoding="utf-8") as fh:
                return pd.read_json(fh)
        except Exception:
            import polars as pl  # type: ignore

            return pl.read_parquet(path)


def _run_evaluation(dataset: pd.DataFrame, cfg):  # noqa: WPS231 (high complexity)
    """Compute pass@k metrics and append / create CSV."""
    prompts = dataset[cfg.data.prompt_key]
    responses = dataset["responses"]
    data_sources = dataset[cfg.data.data_source_key]
    reward_blob = dataset[cfg.data.reward_model_key]

    passes = 0
    total_scores = []
    for resp_list, source, prompt, rwd in zip(responses, data_sources, prompts, reward_blob):
        reward_fn = _select_reward_fn(source)
        gt = rwd["ground_truth"]
        try:
            score_list = [reward_fn(r, gt) for r in resp_list]
        except Exception:  # fallback signature
            score_list = [reward_fn(source, r, gt) for r in resp_list]
        total_scores.append(score_list)
        passes += int(max(score_list) == 1)

    pass_at_n = passes / len(dataset)
    pass_at_1 = float(np.mean(total_scores))

    # ---- CSV bookkeeping --------------------------------------------
    row = {
        "model_path": cfg.model.path,
        "dataset": os.path.basename(cfg.data.path),
        "pass@1": pass_at_1,
        f"pass@{cfg.data.n_samples}": pass_at_n,
    }
    csv_path = os.path.join(os.path.dirname(cfg.data.output_path), "pass.csv")
    file_exists = os.path.isfile(csv_path)
    with open(csv_path, "a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=row.keys())
        if not file_exists:
            writer.writeheader()
        writer.writerow(row)

    # pretty print
    print(tabulate([[k, v] for k, v in row.items()], headers=["Metric", "Value"], tablefmt="grid"))

    # ---- Save per‑sample boolean scores ------------------------------
    bool_scores = [[1.0 if s else 0.0 for s in lst] for lst in total_scores]
    with open(os.path.join(os.path.dirname(cfg.data.output_path), "results.json"), "w") as fh:
        json.dump(bool_scores, fh)


if __name__ == "__main__":
    main()
