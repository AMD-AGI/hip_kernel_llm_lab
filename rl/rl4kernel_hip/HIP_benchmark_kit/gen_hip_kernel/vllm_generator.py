#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""vLLM-based kernel-agent generator."""

import os
from typing import Dict, List, Optional, Sequence, Tuple

from omegaconf import OmegaConf
from vllm import LLM, SamplingParams
from transformers import AutoTokenizer

os.environ["TOKENIZERS_PARALLELISM"] = "true"


def _resolve_path(config_dir: str, path_value: str) -> str:
    """Resolve config-relative paths while leaving absolute paths untouched."""
    if not path_value or os.path.isabs(path_value):
        return path_value
    return os.path.normpath(os.path.join(config_dir, path_value))


def load_config(config_path: str) -> OmegaConf:
    """Load configuration from YAML file.
    
    Args:
        config_path: Path to the YAML configuration file
        
    Returns:
        OmegaConf configuration object
        
    Raises:
        FileNotFoundError: If config file doesn't exist
    """
    config_path = os.path.abspath(config_path)
    if not os.path.exists(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    config_dir = os.path.dirname(config_path)
    config = OmegaConf.load(config_path)

    if "model" in config and "path" in config.model:
        config.model.path = _resolve_path(config_dir, config.model.path)
    if "data" in config:
        if "input_dir" in config.data:
            config.data.input_dir = _resolve_path(config_dir, config.data.input_dir)
        if "output_dir" in config.data:
            config.data.output_dir = _resolve_path(config_dir, config.data.output_dir)

    # Set environment variables if specified in config
    if "environment" in config:
        for key, value in config.environment.items():
            os.environ[key] = str(value)

    return config


class VLLMGenerator:
    """HIP code generator using vLLM for inference."""
    
    def __init__(self, config: OmegaConf):
        """Initialize VLLMGenerator with configuration.
        
        Args:
            config: OmegaConf configuration object loaded from YAML
        """
        print(f"Initializing VLLMGenerator...")
        print(f"Model: {config.model.path}")
        
        self.target_gpu = config.generation.get("target_gpu", "mi300x")
        self.output_contract = config.generation.get("output_contract", "sample_json_v1")
        print(f"Target GPU: {self.target_gpu}")
        print(f"Output contract: {self.output_contract}")
        
        # Initialize tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(config.model.path)
        self.tokenizer.padding_side = "left"
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Initialize vLLM
        self.llm = LLM(
            model=config.model.path,
            tensor_parallel_size=config.rollout.tensor_model_parallel_size,
            gpu_memory_utilization=config.rollout.gpu_memory_utilization,
            dtype=config.rollout.dtype,
            enforce_eager=config.rollout.enforce_eager,
            max_model_len=config.rollout.max_model_len,
            trust_remote_code=True,
        )
        
        self.n_rollouts = int(config.rollout.get("n", 1))
        self.seed_base = config.generation.get("seed_base", None)
        print(f"Seed base: {self.seed_base}")
        self.sampling_params_kwargs = {
            "temperature": config.generation.temperature,
            "top_p": config.generation.top_p,
            "top_k": config.generation.top_k,
            "max_tokens": config.generation.response_length,
            "n": 1,
            "ignore_eos": config.rollout.ignore_eos,
        }
        
        print(f"Initialized vLLM successfully")
        
        # Store config for later use
        self.config = config
    
    def _build_sampling_params(self, rollout_idx: int) -> SamplingParams:
        """Build sampling params for one serial rollout."""
        kwargs = dict(self.sampling_params_kwargs)
        if self.seed_base is not None:
            kwargs["seed"] = int(self.seed_base) + int(rollout_idx)
        return SamplingParams(**kwargs)

    def generate(
        self,
        prompts: List[Dict],
        rollout_indices: Optional[Sequence[int]] = None,
    ) -> List[List[Tuple[int, str]]]:
        """Generate optimized HIP code from prompts.
        
        Args:
            prompts: List of chat message dictionaries
            
        Returns:
            List of lists of ``(rollout_idx, generated_text)`` tuples.
            Outer list: batch dimension (one per input file)
            Inner list: requested serial generations per input file
        """
        if rollout_indices is None:
            rollout_indices = list(range(self.n_rollouts))
        else:
            rollout_indices = [int(idx) for idx in rollout_indices]
        
        # Apply chat template to prompts
        formatted_prompts = []
        for prompt in prompts:
            formatted = self.tokenizer.apply_chat_template(
                prompt,
                add_generation_prompt=True,
                tokenize=False,
            )
            formatted_prompts.append(formatted)
        
        print(
            f"[DEBUG] Batch size: {len(formatted_prompts)}, "
            f"serial_rollouts: {len(rollout_indices)}, "
            f"rollout_indices: {','.join(map(str, rollout_indices)) or 'none'}, "
            f"per_call_n: 1"
        )

        # Strict rollout semantics: run N independent n=1 generations.
        results = [[] for _ in formatted_prompts]
        for rollout_idx in rollout_indices:
            sampling_params = self._build_sampling_params(rollout_idx)
            outputs = self.llm.generate(formatted_prompts, sampling_params)
            for batch_idx, output in enumerate(outputs):
                text = output.outputs[0].text if output.outputs else ""
                results[batch_idx].append((rollout_idx, text))

        return results

def read_hip_file(filepath: str) -> str:
    """Read HIP code from a file.
    
    Args:
        filepath: Path to the HIP file
        
    Returns:
        File contents as string
    """
    with open(filepath, 'r', encoding='utf-8') as f:
        return f.read()

