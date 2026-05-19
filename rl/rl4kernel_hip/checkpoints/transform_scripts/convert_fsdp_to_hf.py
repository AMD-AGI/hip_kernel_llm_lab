import os
import re
from concurrent.futures import ThreadPoolExecutor
import torch
from safetensors.torch import save_file
from transformers import AutoConfig, AutoModelForCausalLM

def convert_fsdp_checkpoints_to_hfmodels(checkpoint_dir, hf_model_path, target_dir):
    # 确定分片数量
    world_size = 0
    for filename in os.listdir(checkpoint_dir):
        match = re.match(r"model_world_size_(\d+)_rank_0\.pt", filename)
        if match:
            world_size = int(match.group(1))
            break
    assert world_size > 0, "未找到有效的分片文件"

    # 加载各分片参数
    shard_params = [{} for _ in range(world_size)]

    def load_shard(rank):
        file_path = os.path.join(checkpoint_dir, f"model_world_size_{world_size}_rank_{rank}.pt")
        state_dict = torch.load(file_path, map_location="cpu")
        
        # 提取DTensor的本地张量
        local_state_dict = {}
        for k, v in state_dict.items():
            if isinstance(v, torch.Tensor) and hasattr(v, "_local_tensor"):
                local_state_dict[k] = v._local_tensor.clone()
            else:
                local_state_dict[k] = v.clone()
        return local_state_dict

    with ThreadPoolExecutor() as executor:
        futures = [executor.submit(load_shard, rank) for rank in range(world_size)]
        for rank, future in enumerate(futures):
            shard_params[rank] = future.result()

    # 合并参数（按FSDP分片维度）
    merged_state_dict = {}
    for key in shard_params[0].keys():
        # 跳过非张量参数
        if not isinstance(shard_params[0][key], torch.Tensor):
            merged_state_dict[key] = shard_params[0][key]
            continue

        # 确定合并维度（关键修改）
        if shard_params[0][key].dim() == 1:  # 偏置等参数
            dim = 0
        else:  # 权重参数（假设FSDP在dim=0分片）
            dim = 0

        # 拼接所有分片
        parts = [shard_params[r][key] for r in range(world_size)]
        merged_state_dict[key] = torch.cat(parts, dim=dim).contiguous()

    # 创建并保存模型
    config = AutoConfig.from_pretrained(hf_model_path)
    model = AutoModelForCausalLM.from_config(config).half()
    
    # 加载合并参数
    model.load_state_dict(merged_state_dict, strict=True)
    
    # 保存模型
    os.makedirs(target_dir, exist_ok=True)

    model.save_pretrained(target_dir, safe_serialization=True)
    # save_file(
    #     model.state_dict(),
    #     os.path.join(target_dir, "model.safetensors"),
    #     metadata={"format": "pt"},
    # )
    config.save_pretrained(target_dir)
    print(f"转换成功！模型保存至：{target_dir}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="Convert FSDP-sharded checkpoints into a single HF safetensors model"
    )
    parser.add_argument(
        "--checkpoint_dir", "-i", required=True,
        help="Directory containing FSDP shards"
    )
    parser.add_argument(
        "--hf_model_path", "-m", required=True,
        help="Hugging Face model path or local folder (for config/tokenizer)"
    )
    parser.add_argument(
        "--target_dir", "-o", required=True,
        help="Where to save the merged HF model"
    )
    args = parser.parse_args()
    convert_fsdp_checkpoints_to_hfmodels(
        checkpoint_dir=args.checkpoint_dir,
        hf_model_path=args.hf_model_path,
        target_dir=args.target_dir,
    )
