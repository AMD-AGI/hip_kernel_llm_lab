# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Parallel Batch Reward Manager for HIP Kernel Evaluation
支持批量并行调用server API实现端到端加速
基于 DAPORewardManager 的实现，增加批量并行处理能力
"""
import torch
from typing import Callable, Optional
from collections import defaultdict

from verl import DataProto
from verl.utils.reward_score import default_compute_score
from verl.workers.reward_manager import register
from verl.workers.reward_manager.reward_debug_printer import emit_reward_distribution_debug


@register("batch_parallel")
class BatchParallelRewardManager:
    """
    批量并行奖励管理器
    
    相比原始 DAPORewardManager 的优化：
    1. 收集整个batch的所有样本
    2. 批量调用server API（利用server侧多核CPU并行编译）
    3. 分配结果到各个样本
    4. 保留详细的调试信息输出
    
    使用方法：
    在配置中设置：
        reward_model.compute_score_batch.path: /path/to/reward_batch.py
        reward_model.compute_score_batch.name: compute_score_batch
        reward_model.reward_manager: batch_parallel
    """
    
    def __init__(
        self,
        tokenizer,
        num_examine,
        compute_score: Optional[Callable] = None,
        compute_score_batch: Optional[Callable] = None,  # 批量处理函数
        reward_fn_key: str = "data_source",
        max_resp_len: Optional[int] = None,
        overlong_buffer_cfg: Optional[dict] = None,
    ) -> None:
        self.tokenizer = tokenizer
        self.num_examine = num_examine  # the number of batches of decoded responses to print to the console
        self.compute_score = compute_score or default_compute_score
        self.compute_score_batch = compute_score_batch  # 批量函数
        self.reward_fn_key = reward_fn_key
        self.overlong_buffer_cfg = overlong_buffer_cfg
        self.max_resp_len = max_resp_len

        if self.overlong_buffer_cfg is not None:
            assert self.max_resp_len is not None, f"max_resp_len must be provided if {overlong_buffer_cfg=}, but got None"
        
        # 优先使用批量函数
        if self.compute_score_batch is not None:
            print(f"[BatchParallelRewardManager] Using batch compute function for parallel evaluation")
        else:
            print(f"[BatchParallelRewardManager] No batch function provided, fallback to sequential processing")

    @staticmethod
    def _enrich_extra_info(data_item, fallback_index: int, train_step):
        raw_extra_info = data_item.non_tensor_batch.get("extra_info", None)
        extra_info = dict(raw_extra_info) if isinstance(raw_extra_info, dict) else {}

        if train_step is not None and extra_info.get("train_step") is None:
            extra_info["train_step"] = train_step

        prompt_uid = data_item.non_tensor_batch.get("uid", None)
        if prompt_uid is not None and not extra_info.get("prompt_uid"):
            extra_info["prompt_uid"] = str(prompt_uid)

        sample_index = data_item.non_tensor_batch.get("index", fallback_index)
        if extra_info.get("sample_index") is None:
            extra_info["sample_index"] = sample_index
        if extra_info.get("index") is None:
            extra_info["index"] = sample_index

        return extra_info
    
    def _decode_and_prepare_data(self, data: DataProto):
        """
        解码所有样本并准备批量处理所需的数据
        
        Returns:
            Tuple of (decoded_data_list, data_sources, solution_strs, ground_truths, extra_infos)
        """
        batch_size = len(data)
        decoded_data = []
        data_sources = []
        solution_strs = []
        ground_truths = []
        extra_infos = []
        train_step = data.meta_info.get("train_step") if data.meta_info else None
        
        eos_token = self.tokenizer.eos_token
        
        for i in range(batch_size):
            data_item = data[i]  # DataProtoItem
            
            # Decode prompt
            prompt_ids = data_item.batch["prompts"]
            prompt_length = prompt_ids.shape[-1]
            valid_prompt_length = data_item.batch["attention_mask"][:prompt_length].sum()
            valid_prompt_ids = prompt_ids[-valid_prompt_length:]
            prompt_str = self.tokenizer.decode(valid_prompt_ids, skip_special_tokens=True)
            
            # Decode response
            response_ids = data_item.batch["responses"]
            valid_response_length = data_item.batch["attention_mask"][prompt_length:].sum()
            valid_response_ids = response_ids[:valid_response_length]
            response_str = self.tokenizer.decode(valid_response_ids, skip_special_tokens=True)
            
            # Remove EOS token if present
            if eos_token and response_str.endswith(eos_token):
                response_str = response_str[: -len(eos_token)]
            
            # Get metadata
            ground_truth = data_item.non_tensor_batch["reward_model"]["ground_truth"]
            data_source = data_item.non_tensor_batch[self.reward_fn_key]
            extra_info = self._enrich_extra_info(data_item, i, train_step)
            
            # Store decoded data
            decoded_data.append({
                "prompt_str": prompt_str,
                "response_str": response_str,
                "valid_response_length": valid_response_length,
                "ground_truth": ground_truth,
                "data_source": data_source,
                "extra_info": extra_info,
            })
            
            # Collect for batch processing
            data_sources.append(data_source)
            solution_strs.append(response_str)
            ground_truths.append(ground_truth)
            extra_infos.append(extra_info)
        
        return decoded_data, data_sources, solution_strs, ground_truths, extra_infos
    
    def verify(self, data: DataProto):
        """串行处理版本（当没有批量函数时使用）"""
        decoded_data, data_sources, solution_strs, ground_truths, extra_infos = self._decode_and_prepare_data(data)
        
        scores = []
        for i in range(len(data)):
            score = self.compute_score(
                data_source=data_sources[i],
                solution_str=solution_strs[i],
                ground_truth=ground_truths[i],
                extra_info=extra_infos[i],
            )
            scores.append(score)
        
        return scores, decoded_data
    
    def verify_batch(self, data: DataProto):
        """批量并行处理版本（当有批量函数时使用）"""
        decoded_data, data_sources, solution_strs, ground_truths, extra_infos = self._decode_and_prepare_data(data)
        
        # 批量调用 - 关键性能优化点
        scores = self.compute_score_batch(
            data_sources=data_sources,
            solution_strs=solution_strs,
            ground_truths=ground_truths,
            extra_infos=extra_infos,
        )
        
        return scores, decoded_data
    
    def __call__(self, data: DataProto, return_dict: bool = False):
        """主调用函数，计算奖励值并生成详细的调试信息"""
        
        # If there is rm score, we directly return rm score. Otherwise, we compute via rm_score_fn
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]

        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        already_print_data_sources = {}
        
        # 使用批量函数或串行函数
        if self.compute_score_batch is not None:
            scores, decoded_data = self.verify_batch(data)
        else:
            scores, decoded_data = self.verify(data)
        
        # 处理每个样本的奖励值
        for i in range(len(data)):
            decoded_item = decoded_data[i]
            valid_response_length = decoded_item["valid_response_length"]
            prompt_str = decoded_item["prompt_str"]
            response_str = decoded_item["response_str"]
            ground_truth = decoded_item["ground_truth"]
            data_source = decoded_item["data_source"]
            
            result = scores[i]
            
            # 提取 reward 值
            score: float
            if isinstance(result, dict):
                score = result["score"]
                # Store the information including original reward
                for key, value in result.items():
                    reward_extra_info[key].append(value)
            else:
                score = result

            reward = score

            # 可选：过长惩罚（保留接口，但默认不启用）
            ## [Note] Canceled - 2025.11.11
            # if self.overlong_buffer_cfg and self.overlong_buffer_cfg.get("enable", False):
            #     overlong_buffer_len = self.overlong_buffer_cfg["len"]
            #     expected_len = self.max_resp_len - overlong_buffer_len
            #     exceed_len = valid_response_length - expected_len
            #     overlong_penalty_factor = self.overlong_buffer_cfg["penalty_factor"]
            #     overlong_reward = min(-exceed_len / overlong_buffer_len * overlong_penalty_factor, 0)
            #     reward += overlong_reward
            #     if self.overlong_buffer_cfg.get("log", False):
            #         reward_extra_info["overlong_reward"].append(overlong_reward)
            #         reward_extra_info["overlong"].append(overlong_reward < 0)

            reward_tensor[i, valid_response_length - 1] = reward

            # 打印调试信息
            if data_source not in already_print_data_sources:
                already_print_data_sources[data_source] = 0

            if already_print_data_sources[data_source] < self.num_examine:
                already_print_data_sources[data_source] += 1
                print("[prompt]", prompt_str)
                print("[response]", response_str)
                print("[ground_truth]", ground_truth)
                if isinstance(result, dict):
                    for key, value in result.items():
                        print(f"[{key}]", value)
                else:
                    print("[score]", score)

        # ========== 调试：打印批内 reward 分布 ==========
        debug_reward = True
        
        if debug_reward:
            # 收集所有最后一个有效 token 的 reward 和相关信息
            batch_rewards = []
            kernel_names = []
            uids = []
            
            for i in range(len(data)):
                data_item = data[i]
                prompt_ids = data_item.batch["prompts"]
                prompt_len = prompt_ids.shape[-1]
                response_length = data_item.batch["responses"].shape[-1]
                valid_length = data_item.batch["attention_mask"][prompt_len:].sum()
                if valid_length > 0:
                    reward_val = reward_tensor[i, valid_length - 1].item()
                    batch_rewards.append(reward_val)
                    
                    # 收集额外信息
                    gt = data_item.non_tensor_batch.get("reward_model", {}).get("ground_truth", {})
                    extra = data_item.non_tensor_batch.get("extra_info", {})
                    kernel_name = (gt if isinstance(gt, dict) else {}).get("kernel_name") or \
                                  extra.get("kernel_name", "unknown")
                    kernel_names.append(kernel_name)
                    
                    # 获取 uid (prompt group id)
                    uid = data_item.non_tensor_batch.get("uid", i)
                    uids.append(uid)
            
            if batch_rewards:
                import numpy as np
                emit_reward_distribution_debug(
                    batch_rewards=np.array(batch_rewards),
                    uids=np.array(uids, dtype=object),
                    kernel_names=kernel_names,
                )
        # ===============================================

        if return_dict:
            return {
                "reward_tensor": reward_tensor,
                "reward_extra_info": reward_extra_info,
            }
        else:
            return reward_tensor

