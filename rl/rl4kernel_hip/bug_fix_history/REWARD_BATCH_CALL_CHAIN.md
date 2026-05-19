# reward_batch.py 在 veRL 中的完整调用链

## 📍 调用路径总览

```
配置文件 (verl_config_batch_parallel.yaml)
    ↓
main_ppo.py (训练入口)
    ↓
load_reward_manager() [verl/trainer/ppo/reward.py]
    ↓
get_custom_reward_batch_fn() [加载 reward_batch.py]
    ↓
BatchParallelRewardManager [初始化时接收函数]
    ↓
RayPPOTrainer.fit() [训练循环]
    ↓
compute_reward() [调用 reward_fn]
    ↓
BatchParallelRewardManager.__call__()
    ↓
BatchParallelRewardManager.verify_batch()
    ↓
compute_score_batch() [来自 reward_batch.py] ⭐
    ↓
Server API: /run_code_batch
```

## 🔍 详细调用流程

### 1. 训练启动 - 配置加载

**文件**: `verl/verl/trainer/main_ppo.py` (第160-161行)

```python
@ray.remote(num_cpus=1)
class TaskRunner:
    def run(self, config):
        # ...
        
        # Load the reward manager for training and validation
        reward_fn = load_reward_manager(
            config, 
            tokenizer, 
            num_examine=0, 
            **config.reward_model.get("reward_kwargs", {})
        )
        val_reward_fn = load_reward_manager(
            config, 
            tokenizer, 
            num_examine=1, 
            **config.reward_model.get("reward_kwargs", {})
        )
```

**作用**: 在训练开始时加载 reward manager

---

### 2. 加载 Reward Manager

**文件**: `verl/verl/trainer/ppo/reward.py` (第116-176行)

```python
def load_reward_manager(config, tokenizer, num_examine, **reward_kwargs):
    """加载和初始化 reward manager"""
    from verl.workers.reward_manager import get_reward_manager_cls
    
    # 1️⃣ 获取 reward manager 类
    reward_manager_name = config.reward_model.get("reward_manager", "naive")
    # 这里会得到 "batch_parallel"
    reward_manager_cls = get_reward_manager_cls(reward_manager_name)
    # 返回 BatchParallelRewardManager 类
    
    # 2️⃣ 加载单样本函数 (compute_score)
    compute_score = get_custom_reward_fn(config)
    # 从 config.custom_reward_function.path 加载
    # 返回: reward_batch.compute_score 函数
    
    # 3️⃣ 加载批量函数 (compute_score_batch) ⭐ 关键步骤
    compute_score_batch = get_custom_reward_batch_fn(config)
    # 从 config.reward_model.compute_score_batch.path 加载
    # 返回: reward_batch.compute_score_batch 函数
    
    # 4️⃣ 准备参数
    manager_kwargs = {
        "tokenizer": tokenizer,
        "num_examine": num_examine,
        "compute_score": compute_score,
        "reward_fn_key": config.data.reward_fn_key,
        **reward_kwargs,
    }
    
    # 5️⃣ 如果批量函数存在，添加到参数中
    if compute_score_batch is not None:
        manager_kwargs["compute_score_batch"] = compute_score_batch
    
    # 6️⃣ 初始化并返回 reward manager
    return reward_manager_cls(**manager_kwargs)
    # 返回: BatchParallelRewardManager 实例，
    #       其中 self.compute_score_batch = reward_batch.compute_score_batch
```

---

### 3. 动态加载 reward_batch.py

**文件**: `verl/verl/trainer/ppo/reward.py` (第67-113行)

```python
def get_custom_reward_batch_fn(config):
    """从配置中动态加载批量评估函数"""
    import importlib.util
    import sys
    
    # 1️⃣ 从配置读取文件路径和函数名
    batch_fn_config = config.reward_model.get("compute_score_batch") or {}
    file_path = batch_fn_config.get("path")
    # 例如: /path/to/reward/reward_batch.py
    
    function_name = batch_fn_config.get("name")
    # 例如: "compute_score_batch"
    
    # 2️⃣ 动态加载模块
    spec = importlib.util.spec_from_file_location("custom_batch_module", file_path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["custom_batch_module"] = module
    spec.loader.exec_module(module)
    
    # 3️⃣ 获取函数
    raw_batch_fn = getattr(module, function_name)
    # 返回: reward_batch.compute_score_batch 函数对象
    
    print(f"using customized batch reward function '{function_name}' from '{file_path}'")
    
    return raw_batch_fn
```

**输出日志**:
```
using customized batch reward function 'compute_score_batch' from '/path/to/reward_batch.py'
```

---

### 4. 训练循环中调用

**文件**: `verl/verl/trainer/ppo/ray_trainer.py` (第1024-1034行)

```python
class RayPPOTrainer:
    def fit(self):
        """PPO 训练循环"""
        # ... 生成响应 ...
        
        with marked_timer("reward", timing_raw, color="yellow"):
            # 计算 reward model score (如果启用)
            if self.use_rm:
                reward_tensor = self.rm_wg.compute_rm_score(batch)
                batch = batch.union(reward_tensor)
            
            # 调用 reward function
            if self.config.reward_model.launch_reward_fn_async:
                future_reward = compute_reward_async.remote(batch, self.config, self.tokenizer)
            else:
                # ⭐ 关键调用
                reward_tensor, reward_extra_infos_dict = compute_reward(batch, self.reward_fn)
                # self.reward_fn 就是 BatchParallelRewardManager 实例
```

---

### 5. compute_reward 函数

**文件**: `verl/verl/trainer/ppo/reward.py` (第179-197行)

```python
def compute_reward(data: DataProto, reward_fn):
    """计算一个 batch 的 reward"""
    try:
        # ⭐ 调用 reward_fn (即 BatchParallelRewardManager.__call__)
        reward_result = reward_fn(data, return_dict=True)
        reward_tensor = reward_result["reward_tensor"]
        reward_extra_infos_dict = reward_result.get("reward_extra_info", {})
    except Exception as e:
        print(f"Error in reward_fn: {e}")
        reward_tensor = reward_fn(data)
        reward_extra_infos_dict = {}
    
    return reward_tensor, reward_extra_infos_dict
```

---

### 6. BatchParallelRewardManager.__call__

**文件**: `verl/verl/workers/reward_manager/batch_parallel.py` (第100-158行)

```python
class BatchParallelRewardManager:
    def __call__(self, data: DataProto, return_dict=False):
        """处理一个 batch 的数据"""
        
        # 1️⃣ 检查是否有预计算的 rm_scores
        if "rm_scores" in data.batch.keys():
            if return_dict:
                return {"reward_tensor": data.batch["rm_scores"]}
            else:
                return data.batch["rm_scores"]
        
        # 2️⃣ 初始化 reward tensor
        reward_tensor = torch.zeros_like(data.batch["responses"], dtype=torch.float32)
        reward_extra_info = defaultdict(list)
        
        # 3️⃣ 获取数据信息
        prompt_ids = data.batch["prompts"]
        prompt_len = prompt_ids.shape[-1]
        attention_mask = data.batch["attention_mask"]
        valid_response_lengths = attention_mask[:, prompt_len:].sum(dim=-1)
        data_sources = data.non_tensor_batch[self.reward_fn_key]
        
        # 4️⃣ ⭐ 关键分支：使用批量函数或串行函数
        if self.compute_score_batch is not None:
            # 批量并行模式
            scores = self.verify_batch(data)
        else:
            # 串行模式（向后兼容）
            scores = self.verify(data)
        
        # 5️⃣ 处理结果
        rewards = []
        already_printed = {}
        
        for i in range(len(data)):
            length = valid_response_lengths[i].item()
            score = scores[i]
            
            # 处理 score (可能是 dict 或 float)
            if isinstance(score, dict):
                reward = score["score"]
                for key, value in score.items():
                    reward_extra_info[key].append(value)
            else:
                reward = score
            
            rewards.append(reward)
            reward_tensor[i, length - 1] = reward
            
            # 打印样本信息（用于调试）
            data_source = data_sources[i]
            if already_printed.get(data_source, 0) < self.num_examine:
                # ... 打印 prompt, response, ground_truth, score ...
                already_printed[data_source] = already_printed.get(data_source, 0) + 1
        
        # 6️⃣ 保存准确率信息
        data.batch["acc"] = torch.tensor(rewards, dtype=torch.float32, device=prompt_ids.device)
        
        # 7️⃣ 返回结果
        if return_dict:
            return {"reward_tensor": reward_tensor, "reward_extra_info": reward_extra_info}
        else:
            return reward_tensor
```

---

### 7. verify_batch - 批量验证

**文件**: `verl/verl/workers/reward_manager/batch_parallel.py` (第68-98行)

```python
def verify_batch(self, data: DataProto):
    """批量验证（并行优化版本）"""
    batch_size = len(data)
    
    # 1️⃣ 收集所有样本数据
    data_sources = []
    solution_strs = []
    ground_truths = []
    extra_infos = []
    
    for i in range(batch_size):
        # 从 DataProto 中提取数据
        data_sources.append(data.non_tensor_batch[self.reward_fn_key][i])
        # 例如: ["hip2hip-train", "hip2hip-train", ...]
        
        ground_truths.append(
            data[i].non_tensor_batch["reward_model"].get("ground_truth", None)
        )
        # 例如: [{"kernel_name": "kernel1", "hip_code": "...", ...}, ...]
        
        extra_infos.append(
            data[i].non_tensor_batch["reward_model"].get("extra_info", None)
        )
        # 例如: [{"sandbox_url": "...", "atol": 1e-4, ...}, ...]
        
        solution_strs.append(
            data[i].non_tensor_batch["reward_model"]["solution"]
        )
        # 例如: ["#include <hip/hip_runtime.h>\n...", ...]
    
    # 2️⃣ ⭐⭐⭐ 调用批量函数！！！
    scores = self.compute_score_batch(
        data_sources=data_sources,      # 32个数据源
        solution_strs=solution_strs,    # 32个HIP代码
        ground_truths=ground_truths,    # 32个参考数据
        extra_infos=extra_infos,        # 32个额外信息
    )
    # 这里 self.compute_score_batch 就是 reward_batch.compute_score_batch ⭐
    
    # 3️⃣ 返回批量分数
    return scores  # [score1, score2, ..., score32]
```

---

### 8. reward_batch.compute_score_batch - 最终执行

**文件**: `reward/reward_batch.py` (第84-200行)

```python
def compute_score_batch(
    data_sources: List[str],
    solution_strs: List[str],
    ground_truths: List[T.Any],
    extra_infos: List[Optional[dict]] = None
) -> List[float]:
    """
    批量计算score，充分利用server侧并行编译
    
    Args:
        data_sources: 数据源列表 (例如: ["hip2hip-train"] * 32)
        solution_strs: HIP代码列表 (32个生成的HIP代码)
        ground_truths: ground truth列表 (32个参考数据)
        extra_infos: 额外信息列表 (32个配置信息)
        
    Returns:
        scores: 分数列表 [score1, score2, ..., score32]
    """
    batch_size = len(data_sources)
    extra_infos = extra_infos or [{}] * batch_size
    
    # 1️⃣ 过滤出需要评估的样本
    valid_indices = []
    batch_requests = []
    
    for i in range(batch_size):
        data_source = data_sources[i]
        
        # 只处理HIP相关数据源
        if data_source not in ("torch2hip-train", "torch2hip-val", "hip2hip-train", "hip2hip-val"):
            continue
        
        extra = extra_infos[i] or {}
        
        # 解析参数并构造请求
        # ...
        
        batch_requests.append({
            "kernel_name": kernel_name,
            "hip_code": hip_src,
            "hip_ref_code": hip_ref,
            "pytorch_module_code": module_src or "",
            "pytorch_functional_code": functional_src or "",
            "atol": atol,
            "rtol": rtol,
        })
        valid_indices.append(i)
    
    # 2️⃣ 初始化结果（默认0分）
    scores = [0.0] * batch_size
    
    if not batch_requests:
        return scores
    
    # 3️⃣ ⭐⭐⭐ 批量调用server（1次HTTP请求）
    try:
        sf_timeout_s = int(extra_infos[0].get("sf_timeout_s", 600))
        resp = call_batch_run_code(sf_url, batch_requests, timeout_s=sf_timeout_s)
        
        if resp.status_code != 200:
            # 批量失败，所有样本给负分
            for idx in valid_indices:
                scores[idx] = 0.0
            return scores
        
        # 4️⃣ 解析批量响应
        resp_data = resp.json()
        responses = resp_data.get("responses", [])
        
        # 5️⃣ 分配结果
        for i, resp_item in enumerate(responses):
            if i >= len(valid_indices):
                break
            
            idx = valid_indices[i]
            score = _compute_single_score(resp_item)
            scores[idx] = score
            
    except Exception as e:
        # 批量请求失败，给所有有效样本负分
        print(f"Batch evaluation exception: {e}")
        for idx in valid_indices:
            scores[idx] = 0.0
    
    # 6️⃣ 返回所有分数
    return scores
```

---

### 9. 调用 Server API

**文件**: `reward/reward_batch.py` (第62-78行)

```python
def call_batch_run_code(url: str, requests_data: List[dict], timeout_s: int = 600) -> requests.Response:
    """
    调用批量评估API
    url: 应为 http://server:port/run_code_batch
    """
    batch_url = url.replace("/run_code", "/run_code_batch")
    
    batch_payload = {
        "requests": requests_data  # 32个请求
    }
    
    # ⭐ 发送单个 HTTP POST 请求
    resp = requests.post(
        batch_url,
        json=batch_payload,
        timeout=timeout_s
    )
    return resp
```

**Server端**: `hip_kernel_evaluation_server/hip_kernel_check_utils_hip2hip_parallel.py`
- 接收批量请求
- 使用 `ProcessPoolExecutor` 多进程并行编译
- 返回批量结果

---

## 📊 数据流示例

### 输入 (batch_size=32)

```python
data_sources = ["hip2hip-train"] * 32
solution_strs = [
    "#include <hip/hip_runtime.h>\n__global__ void kernel1() {...}",
    "#include <hip/hip_runtime.h>\n__global__ void kernel2() {...}",
    # ... 30 more ...
]
ground_truths = [
    {"kernel_name": "kernel1", "hip_code": "...", ...},
    # ... 31 more ...
]
extra_infos = [
    {"sandbox_url": "http://localhost:8080/run_code", "atol": 1e-4, ...},
    # ... 31 more ...
]
```

### 中间处理

1. **BatchParallelRewardManager.verify_batch()**: 收集数据
2. **compute_score_batch()**: 构造批量请求
3. **call_batch_run_code()**: 1次HTTP POST
4. **Server处理**: 8进程并行编译

### 输出 (batch_size=32)

```python
scores = [
    1.2,   # compile✅ run✅ match✅ speedup=1.5x
    -0.5,  # compile✅ run❌
    -0.9,  # compile❌
    # ... 29 more ...
]
```

## ⚡ 性能对比

| 模式 | HTTP请求 | 并行度 | 时间 (batch=32) |
|------|---------|--------|----------------|
| **串行** | 32次 | 1进程 | ~320秒 |
| **批量并行** | **1次** | **8进程** | **~50秒** |

**加速比**: 6.4x ⚡

## 🎯 关键要点总结

### reward_batch.py 被调用的路径

1. **配置阶段**: `load_reward_manager()` → `get_custom_reward_batch_fn()` → 动态加载 `reward_batch.py`
2. **初始化**: `BatchParallelRewardManager.__init__()` → 保存 `compute_score_batch` 函数引用
3. **训练循环**: `RayPPOTrainer.fit()` → `compute_reward()` → `BatchParallelRewardManager.__call__()`
4. **批量评估**: `verify_batch()` → **`self.compute_score_batch()`** → 来自 `reward_batch.py` ⭐
5. **Server调用**: `call_batch_run_code()` → HTTP POST → Server端并行处理

### 调用频率

- **每个训练 batch 调用一次** `BatchParallelRewardManager.__call__()`
- **每次调用处理 N 个样本**（N = batch_size，例如32）
- **只发送 1 次 HTTP 请求**（而不是 N 次）

### 关键优势

✅ **批量处理**: 1次HTTP请求处理32个样本  
✅ **并行编译**: Server端8进程并行  
✅ **端到端加速**: 从veRL训练到HIP评估的完整优化  
✅ **向后兼容**: 未配置时自动回退串行模式

---

**文档版本**: v1.0  
**最后更新**: 2025-11-18

