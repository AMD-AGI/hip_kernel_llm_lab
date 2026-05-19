"""
Kernel Novelty Tracker
用于训练过程中观测 kernel 重复性，记录实际代码和统计指标
"""
from __future__ import annotations
import os
import json
import time
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Optional, List, Tuple, Dict


@dataclass
class KernelRecord:
    """单个 kernel 的观测记录"""
    kernel_name: str
    dtw_distance: float
    token_len: int
    is_copy: bool
    novelty: float
    thresholds: Tuple[float, float, float]  # (d_copy, d_low, d_high)
    ref_code: str
    gen_code: str
    reward: float
    timestamp: float = field(default_factory=time.time)
    step: int = 0
    extra_info: Dict = field(default_factory=dict)


class KernelNoveltyTracker:
    """
    Kernel 重复性追踪器
    
    用于训练过程中观测生成代码的重复性/新颖性，记录实际代码便于分析
    
    使用示例:
    ---------
    ```python
    from reward.kernel_novelty_tracker import KernelNoveltyTracker
    
    # 初始化（可选指定日志文件）
    tracker = KernelNoveltyTracker(
        log_dir="./kernel_novelty_logs",
        log_copies=True,      # 是否记录完全复制的case
        log_all=False,        # 是否记录所有case（可能很大）
        max_records=10000     # 最大记录数
    )
    
    # 在 reward 计算后调用
    tracker.record(
        kernel_name="my_kernel",
        dtw_distance=0.15,
        token_len=150,
        is_copy=False,
        novelty=0.6,
        thresholds=(0.08, 0.10, 0.20),
        ref_code=hip_ref_code,
        gen_code=hip_gen_code,
        reward=1.2,
        step=global_step
    )
    
    # 获取统计信息
    stats = tracker.get_stats()
    print(stats)
    
    # 导出分析报告
    tracker.export_report("novelty_report.json")
    ```
    """
    
    def __init__(
        self,
        log_dir: Optional[str] = None,
        log_copies: bool = True,
        log_all: bool = False,
        max_records: int = 10000,
        verbose: bool = True
    ):
        """
        Args:
            log_dir: 日志文件保存目录（None则不保存文件）
            log_copies: 是否记录完全复制的case
            log_all: 是否记录所有case（False则只记录copies和异常情况）
            max_records: 最大记录数（防止内存溢出）
            verbose: 是否打印详细信息
        """
        self.log_dir = log_dir
        self.log_copies = log_copies
        self.log_all = log_all
        self.max_records = max_records
        self.verbose = verbose
        
        # 统计计数器
        self.total_count = 0
        self.copy_count = 0
        self.dtw_sum = 0.0
        self.novelty_sum = 0.0
        self.reward_sum = 0.0
        
        # DTW 分布直方图（bins: 0-0.1, 0.1-0.2, ..., 0.9-1.0）
        self.dtw_histogram = [0] * 10
        
        # 按 kernel_name 分组的统计
        self.kernel_stats: Dict[str, Dict] = defaultdict(lambda: {
            "count": 0, "copy_count": 0, "dtw_sum": 0.0, "novelty_sum": 0.0
        })
        
        # 详细记录（可选）
        self.records: List[KernelRecord] = []
        
        # 初始化日志目录
        if self.log_dir:
            os.makedirs(self.log_dir, exist_ok=True)
            self._init_log_files()
    
    def _init_log_files(self):
        """初始化日志文件"""
        if not self.log_dir:
            return
        
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        
        # 复制记录文件
        self.copy_log_path = os.path.join(self.log_dir, f"copies_{timestamp}.jsonl")
        
        # 统计文件
        self.stats_log_path = os.path.join(self.log_dir, f"stats_{timestamp}.json")
        
        # 全量记录文件（如果启用）
        if self.log_all:
            self.all_log_path = os.path.join(self.log_dir, f"all_records_{timestamp}.jsonl")
    
    def record(
        self,
        kernel_name: str,
        dtw_distance: float,
        token_len: int,
        is_copy: bool,
        novelty: float,
        thresholds: Tuple[float, float, float],
        ref_code: str,
        gen_code: str,
        reward: float,
        step: int = 0,
        extra_info: Optional[Dict] = None
    ) -> None:
        """
        记录一次 kernel 生成的观测数据
        
        Args:
            kernel_name: kernel 函数名
            dtw_distance: DTW 距离 ∈ [0, 1]
            token_len: token 数量
            is_copy: 是否被判定为完全复制
            novelty: novelty 值 ∈ [0, 1]（复制时为 -1）
            thresholds: (d_copy, d_low, d_high) 阈值
            ref_code: 参考代码
            gen_code: 生成代码
            reward: 最终 reward
            step: 训练 step
            extra_info: 额外信息
        """
        self.total_count += 1
        self.dtw_sum += dtw_distance
        self.reward_sum += reward
        
        # 更新直方图
        bin_idx = min(int(dtw_distance * 10), 9)
        self.dtw_histogram[bin_idx] += 1
        
        # 更新 kernel 级别统计
        kstat = self.kernel_stats[kernel_name]
        kstat["count"] += 1
        kstat["dtw_sum"] += dtw_distance
        
        if is_copy:
            self.copy_count += 1
            kstat["copy_count"] += 1
        else:
            # novelty 只在非复制情况下有效
            self.novelty_sum += novelty
            kstat["novelty_sum"] += novelty
        
        # 创建记录
        record = KernelRecord(
            kernel_name=kernel_name,
            dtw_distance=dtw_distance,
            token_len=token_len,
            is_copy=is_copy,
            novelty=novelty,
            thresholds=thresholds,
            ref_code=ref_code,
            gen_code=gen_code,
            reward=reward,
            step=step,
            extra_info=extra_info or {}
        )
        
        # 是否记录到内存
        should_record = (
            self.log_all or 
            is_copy or 
            dtw_distance < 0.05 or  # 极低差异
            dtw_distance > 0.5      # 极高差异
        )
        
        if should_record and len(self.records) < self.max_records:
            self.records.append(record)
        
        # 写入日志文件
        if self.log_dir:
            self._write_to_log(record, is_copy)
        
        # 打印详细信息
        if self.verbose and is_copy:
            print(f"\n{'=' * 72}")
            print(f"[COPY DETECTED] step={step}, kernel={kernel_name}")
            print(f"  dtw_distance : {dtw_distance:.4f}")
            print(f"  copy_thresh  : {thresholds[0]:.4f}")
            print(f"  token_len    : {token_len}")
            print(f"  reward       : {reward:.4f}")
            print(f"{'-' * 72}")
            print("[Reference Code Preview]")
            print((ref_code or "")[:500])
            print(f"{'-' * 72}")
            print("[Generated Code Preview]")
            print((gen_code or "")[:500])
            print(f"{'=' * 72}\n")
    
    def _write_to_log(self, record: KernelRecord, is_copy: bool) -> None:
        """将记录写入日志文件"""
        record_dict = {
            "kernel_name": record.kernel_name,
            "dtw_distance": record.dtw_distance,
            "token_len": record.token_len,
            "is_copy": record.is_copy,
            "novelty": record.novelty,
            "thresholds": record.thresholds,
            "reward": record.reward,
            "timestamp": record.timestamp,
            "step": record.step,
            "extra_info": record.extra_info,
            # 代码可能很长，截断存储
            "ref_code_preview": record.ref_code[:2000] if record.ref_code else "",
            "gen_code_preview": record.gen_code[:2000] if record.gen_code else "",
        }
        
        # 复制记录
        if is_copy and self.log_copies:
            with open(self.copy_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
        
        # 全量记录
        if self.log_all:
            with open(self.all_log_path, "a", encoding="utf-8") as f:
                f.write(json.dumps(record_dict, ensure_ascii=False) + "\n")
    
    def get_stats(self) -> Dict:
        """
        获取统计信息
        
        Returns:
            包含各项统计指标的字典
        """
        if self.total_count == 0:
            return {"total_count": 0, "error": "No records yet"}
        
        non_copy_count = self.total_count - self.copy_count
        
        stats = {
            "total_count": self.total_count,
            "copy_count": self.copy_count,
            "copy_rate": self.copy_count / self.total_count,
            "avg_dtw_distance": self.dtw_sum / self.total_count,
            "avg_novelty": self.novelty_sum / non_copy_count if non_copy_count > 0 else 0.0,
            "avg_reward": self.reward_sum / self.total_count,
            "dtw_histogram": {
                f"{i/10:.1f}-{(i+1)/10:.1f}": self.dtw_histogram[i]
                for i in range(10)
            },
            "dtw_histogram_pct": {
                f"{i/10:.1f}-{(i+1)/10:.1f}": self.dtw_histogram[i] / self.total_count * 100
                for i in range(10)
            },
        }
        
        # 按 kernel 分组的统计
        kernel_summary = {}
        for kname, kstat in self.kernel_stats.items():
            if kstat["count"] > 0:
                kernel_summary[kname] = {
                    "count": kstat["count"],
                    "copy_count": kstat["copy_count"],
                    "copy_rate": kstat["copy_count"] / kstat["count"],
                    "avg_dtw": kstat["dtw_sum"] / kstat["count"],
                }
        stats["per_kernel"] = kernel_summary
        
        return stats
    
    def get_copy_records(self) -> List[KernelRecord]:
        """获取所有完全复制的记录"""
        return [r for r in self.records if r.is_copy]
    
    def export_report(self, output_path: str) -> None:
        """
        导出分析报告到 JSON 文件
        
        Args:
            output_path: 输出文件路径
        """
        report = {
            "summary": self.get_stats(),
            "copy_records": [
                {
                    "kernel_name": r.kernel_name,
                    "dtw_distance": r.dtw_distance,
                    "token_len": r.token_len,
                    "step": r.step,
                    "ref_code": r.ref_code[:1000],
                    "gen_code": r.gen_code[:1000],
                }
                for r in self.get_copy_records()[:100]  # 最多导出100条
            ],
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
        }
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2, ensure_ascii=False)
        
        print(f"[KernelNoveltyTracker] Report exported to: {output_path}")
    
    def print_summary(self) -> None:
        """打印统计摘要"""
        stats = self.get_stats()
        
        print("\n" + "=" * 60)
        print("Kernel Novelty Tracking Summary")
        print("=" * 60)
        print(f"Total samples:     {stats['total_count']}")
        print(f"Copy detected:     {stats['copy_count']} ({stats['copy_rate']*100:.1f}%)")
        print(f"Avg DTW distance:  {stats['avg_dtw_distance']:.4f}")
        print(f"Avg novelty:       {stats['avg_novelty']:.4f}")
        print(f"Avg reward:        {stats['avg_reward']:.4f}")
        print("\nDTW Distance Distribution:")
        for bin_name, pct in stats["dtw_histogram_pct"].items():
            bar = "█" * int(pct / 2) + "░" * (50 - int(pct / 2))
            print(f"  {bin_name}: {bar} {pct:.1f}%")
        print("=" * 60 + "\n")
    
    def reset(self) -> None:
        """重置所有统计数据"""
        self.total_count = 0
        self.copy_count = 0
        self.dtw_sum = 0.0
        self.novelty_sum = 0.0
        self.reward_sum = 0.0
        self.dtw_histogram = [0] * 10
        self.kernel_stats.clear()
        self.records.clear()


# ============================================================================
# 全局 Tracker 管理
# ============================================================================

# 全局 tracker 实例（可选使用）
_global_tracker: Optional[KernelNoveltyTracker] = None


def get_global_tracker() -> KernelNoveltyTracker:
    """获取全局 tracker 实例（懒初始化）"""
    global _global_tracker
    if _global_tracker is None:
        log_dir = os.environ.get("KERNEL_NOVELTY_LOG_DIR")
        _global_tracker = KernelNoveltyTracker(
            log_dir=log_dir,
            log_copies=True,
            log_all=os.environ.get("KERNEL_NOVELTY_LOG_ALL", "0") == "1",
            verbose=True
        )
    return _global_tracker


def init_global_tracker(
    log_dir: Optional[str] = None,
    log_copies: bool = True,
    log_all: bool = False,
    verbose: bool = True
) -> KernelNoveltyTracker:
    """
    初始化全局 tracker（建议在训练开始时调用）
    
    Args:
        log_dir: 日志目录
        log_copies: 是否记录复制case
        log_all: 是否记录所有case
        verbose: 是否打印详细信息
        
    Returns:
        初始化后的全局 tracker
    """
    global _global_tracker
    _global_tracker = KernelNoveltyTracker(
        log_dir=log_dir,
        log_copies=log_copies,
        log_all=log_all,
        verbose=verbose
    )
    return _global_tracker

