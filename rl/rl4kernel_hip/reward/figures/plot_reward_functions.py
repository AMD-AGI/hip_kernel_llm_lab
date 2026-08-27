#!/usr/bin/env python3
# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""
Plot reward functions from reward/reward_batch.py with seaborn.

The figure compares the two reward modes that currently exist in code:
1) default mode: _compute_single_score_with_novelty
2) soft_clip_novelty mode: _compute_single_score_soft_clip_novelty
"""

from __future__ import annotations

import argparse
import math
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns

try:
    from reward.utils import get_adaptive_thresholds
except ModuleNotFoundError:
    # Allow running the script from inside reward/ as well.
    from utils import get_adaptive_thresholds


def _clip(x: float, lo: float, hi: float) -> float:
    if x < lo:
        return lo
    if x > hi:
        return hi
    return x


def _novelty_from_dtw(d_ref: float, d_low: float, d_high: float) -> float:
    if d_ref <= d_low:
        return 0.0
    if d_ref >= d_high:
        return 1.0
    return (d_ref - d_low) / (d_high - d_low)


def _base_reward_default(speedup: float, s_ref: float = 100.0) -> float:
    s = max(0.0, float(speedup))
    gain = math.log1p(s) / math.log1p(s_ref) if s_ref > 0 else 0.0
    gain = _clip(gain, 0.0, 1.0)
    return 0.5 + gain


def _base_reward_soft_clip(
    speedup: float,
    eps: float = 1e-6,
    a: float = 1.0,
    b: float = 9.0,
    r_ok: float = 0.3,
    beta: float = 0.5,
) -> float:
    s = max(float(eps), float(speedup))
    r_perf = _clip(s - 1.0, -float(a), float(b))
    return float(r_ok) + float(beta) * float(r_perf)


def _final_reward_default(speedup: float, d_ref: float, token_len: int) -> float:
    d_copy, d_low, d_high = get_adaptive_thresholds(token_len)
    if d_ref < d_copy:
        return -0.2

    base = _base_reward_default(speedup)
    novelty = _novelty_from_dtw(d_ref, d_low, d_high)
    diversity_bonus = 0.25 * (novelty - 0.5)
    return base + diversity_bonus


def _final_reward_soft(speedup: float, d_ref: float, token_len: int) -> float:
    d_copy, d_low, d_high = get_adaptive_thresholds(token_len)
    base = _base_reward_soft_clip(speedup)
    novelty = _novelty_from_dtw(d_ref, d_low, d_high)
    shaped = base + 0.5 * (novelty - 0.5)

    if d_ref < d_copy:
        return min(shaped, -0.2)
    return shaped


def _parse_int_list(raw: str) -> List[int]:
    vals = []
    for p in raw.split(","):
        p = p.strip()
        if not p:
            continue
        vals.append(int(p))
    if not vals:
        raise ValueError("token_lens cannot be empty")
    return vals


def _build_gate_df() -> pd.DataFrame:
    rows = [
        ("compile_fail", "default_with_novelty", -0.9),
        ("run_fail", "default_with_novelty", -0.5),
        ("match_fail", "default_with_novelty", 0.0),
        ("compile_fail", "soft_clip_novelty", -0.9),
        ("run_fail", "soft_clip_novelty", -0.7),
        ("match_fail", "soft_clip_novelty", -0.3),
    ]
    return pd.DataFrame(rows, columns=["stage", "mode", "reward"])


def _build_base_df(speedups: Sequence[float]) -> pd.DataFrame:
    rows = []
    for s in speedups:
        rows.append((s, "default_with_novelty", _base_reward_default(s)))
        rows.append((s, "soft_clip_novelty", _base_reward_soft_clip(s)))
    return pd.DataFrame(rows, columns=["speedup", "mode", "reward"])


def _build_novelty_df(d_refs: Sequence[float], token_lens: Iterable[int]) -> pd.DataFrame:
    rows = []
    for token_len in token_lens:
        d_copy, d_low, d_high = get_adaptive_thresholds(token_len)
        for d_ref in d_refs:
            novelty = _novelty_from_dtw(d_ref, d_low, d_high)
            rows.append((token_len, d_ref, novelty, d_copy, d_low, d_high))
    return pd.DataFrame(
        rows,
        columns=["token_len", "d_ref", "novelty", "d_copy", "d_low", "d_high"],
    )


def _build_final_df(
    speedups: Sequence[float],
    token_len: int,
    d_ref_cases: Sequence[Tuple[str, float]],
) -> pd.DataFrame:
    rows = []
    for case_name, d_ref in d_ref_cases:
        for s in speedups:
            rows.append(
                (
                    s,
                    "default_with_novelty",
                    case_name,
                    d_ref,
                    _final_reward_default(s, d_ref=d_ref, token_len=token_len),
                )
            )
            rows.append(
                (
                    s,
                    "soft_clip_novelty",
                    case_name,
                    d_ref,
                    _final_reward_soft(s, d_ref=d_ref, token_len=token_len),
                )
            )
    return pd.DataFrame(rows, columns=["speedup", "mode", "case", "d_ref", "reward"])


def plot_reward_functions(
    output_path: Path,
    token_len: int,
    novelty_token_lens: Sequence[int],
    speedup_max: float,
    points: int,
) -> None:
    sns.set_theme(style="whitegrid", context="talk")

    speedups = [speedup_max * i / (points - 1) for i in range(points)]
    d_refs = [0.5 * i / (points - 1) for i in range(points)]

    d_copy, d_low, d_high = get_adaptive_thresholds(token_len)
    eps = 1e-4
    d_ref_cases = [
        ("copy_zone", max(0.0, d_copy * 0.5)),
        ("near_copy_edge", min(0.5, d_copy + eps)),
        ("mid_novelty", 0.5 * (d_low + d_high)),
        ("high_novelty", min(0.5, d_high + 0.1)),
    ]

    gate_df = _build_gate_df()
    base_df = _build_base_df(speedups)
    novelty_df = _build_novelty_df(d_refs=d_refs, token_lens=novelty_token_lens)
    final_df = _build_final_df(speedups=speedups, token_len=token_len, d_ref_cases=d_ref_cases)

    fig, axes = plt.subplots(2, 2, figsize=(18, 12))

    sns.barplot(data=gate_df, x="stage", y="reward", hue="mode", ax=axes[0, 0])
    axes[0, 0].axhline(0.0, color="black", linewidth=1.0)
    axes[0, 0].set_title("Gate penalties")
    axes[0, 0].set_xlabel("")
    axes[0, 0].set_ylabel("reward")

    sns.lineplot(data=base_df, x="speedup", y="reward", hue="mode", linewidth=2.4, ax=axes[0, 1])
    axes[0, 1].set_title("Base reward vs speedup (match_ok path)")
    axes[0, 1].set_xlabel("speedup")
    axes[0, 1].set_ylabel("base reward")

    sns.lineplot(
        data=novelty_df,
        x="d_ref",
        y="novelty",
        hue="token_len",
        linewidth=2.4,
        palette="viridis",
        ax=axes[1, 0],
    )
    for token_len_marker in novelty_token_lens:
        t_copy, _, _ = get_adaptive_thresholds(token_len_marker)
        axes[1, 0].axvline(t_copy, color="gray", linestyle="--", linewidth=1.2, alpha=0.35)
    axes[1, 0].set_title("Novelty mapping with adaptive thresholds")
    axes[1, 0].set_xlabel("d_ref (DTW distance)")
    axes[1, 0].set_ylabel("novelty")
    axes[1, 0].set_xlim(0.0, 0.5)
    axes[1, 0].set_ylim(-0.02, 1.02)

    sns.lineplot(
        data=final_df,
        x="speedup",
        y="reward",
        hue="case",
        style="mode",
        linewidth=2.2,
        ax=axes[1, 1],
    )
    axes[1, 1].axhline(-0.2, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
    axes[1, 1].set_title(f"Final reward vs speedup (token_len={token_len})")
    axes[1, 1].set_xlabel("speedup")
    axes[1, 1].set_ylabel("final reward")

    fig.suptitle(
        "Actual reward functions in reward_batch.py\n"
        "README pipeline corresponds to soft_clip_novelty mode",
        y=1.02,
    )
    fig.tight_layout()

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def plot_soft_clip_novelty_only(
    output_path: Path,
    token_len: int,
    speedup_max: float,
    points: int,
) -> None:
    """
    Plot one figure with two windows for soft_clip_novelty mode:
    - match_ok reward curves / envelope
    - copy-cap behavior
    - gate values and overall range annotation
    - a dedicated zoom window for speedup in [0, 2]
    """
    sns.set_theme(style="whitegrid", context="talk")

    # Default hyperparams from _compute_single_score_soft_clip_novelty.
    eps = 1e-6
    a = 1.0
    b = 9.0
    r_ok = 0.3
    beta = 0.5
    alpha = 0.5

    speedups = [speedup_max * i / (points - 1) for i in range(points)]

    lower_capped_curve = []
    center_curve = []
    upper_curve = []

    for s in speedups:
        r_base = _base_reward_soft_clip(s, eps=eps, a=a, b=b, r_ok=r_ok, beta=beta)
        lower_raw = r_base + alpha * (0.0 - 0.5)
        lower_capped = min(lower_raw, -0.2)
        center = r_base
        upper = r_base + alpha * (1.0 - 0.5)

        lower_capped_curve.append(lower_capped)
        center_curve.append(center)
        upper_curve.append(upper)

    curve_rows = []
    for i, s in enumerate(speedups):
        curve_rows.extend(
            [
                (s, "novelty=0 (capped)", lower_capped_curve[i]),
                (s, "novelty=0.5", center_curve[i]),
                (s, "novelty=1.0", upper_curve[i]),
            ]
        )
    curve_df = pd.DataFrame(curve_rows, columns=["speedup", "curve", "reward"])

    match_min = min(lower_capped_curve)
    match_max = max(upper_curve)
    overall_min = min(-0.9, -0.7, -0.3, match_min)
    overall_max = max(-0.9, -0.7, -0.3, match_max)

    zoom_x_min, zoom_x_max = 0.0, 2.0
    zoom_points = [(s, lo, hi) for s, lo, hi in zip(speedups, lower_capped_curve, upper_curve) if zoom_x_min <= s <= zoom_x_max]
    zoom_speedups = [p[0] for p in zoom_points]
    zoom_lower = [p[1] for p in zoom_points]
    zoom_upper = [p[2] for p in zoom_points]
    zoom_curve_df = curve_df[(curve_df["speedup"] >= zoom_x_min) & (curve_df["speedup"] <= zoom_x_max)].copy()

    curve_palette = {
        "novelty=0 (capped)": "#4C72B0",
        "novelty=0.5": "#DD8452",
        "novelty=1.0": "#55A868",
    }
    gate_lines = [
        ("compile_fail", -0.9, "#8e1b1b"),
        ("run_fail", -0.7, "#c77700"),
        ("match_fail", -0.3, "#1f77b4"),
    ]

    fig, (ax_full, ax_zoom) = plt.subplots(
        1,
        2,
        figsize=(20, 8),
        gridspec_kw={"width_ratios": [1.9, 1.1]},
    )

    sns.lineplot(
        data=curve_df,
        x="speedup",
        y="reward",
        hue="curve",
        palette=curve_palette,
        linewidth=2.6,
        ax=ax_full,
    )
    ax_full.fill_between(speedups, lower_capped_curve, upper_curve, alpha=0.14)

    for name, value, color in gate_lines:
        ax_full.axhline(value, color=color, linestyle="--", linewidth=1.2, alpha=0.85)
        ax_full.text(
            speedup_max * 0.98,
            value + 0.015,
            f"{name} = {value:.1f}",
            ha="right",
            va="bottom",
            fontsize=10,
            color=color,
        )

    ax_full.axhline(-0.2, color="red", linestyle="--", linewidth=1.2, alpha=0.6)
    ax_full.axhline(0.3, color="red", linestyle="--", linewidth=1.1, alpha=0.6)
    ax_full.text(speedup_max * 0.98, -0.2 + 0.015, "copy cap = -0.2", ha="right", va="bottom", fontsize=10, color="red")
    ax_full.set_xlim(0.0, speedup_max)
    ax_full.set_ylim(min(-1.0, overall_min - 0.1), overall_max + 0.2)
    ax_full.set_xlabel("speedup")
    ax_full.set_ylabel("reward")
    ax_full.set_title("soft_clip_novelty: full range")

    info_text = (
        f"match_ok range: [{match_min:.2f}, {match_max:.2f}]\n"
        f"overall range: [{overall_min:.2f}, {overall_max:.2f}]"
    )
    ax_full.text(
        0.02,
        0.98,
        info_text,
        transform=ax_full.transAxes,
        va="top",
        ha="left",
        fontsize=10,
        bbox={"boxstyle": "round", "facecolor": "white", "alpha": 0.88, "edgecolor": "gray"},
    )

    sns.lineplot(
        data=zoom_curve_df,
        x="speedup",
        y="reward",
        hue="curve",
        palette=curve_palette,
        linewidth=2.4,
        legend=False,
        ax=ax_zoom,
    )
    ax_zoom.fill_between(zoom_speedups, zoom_lower, zoom_upper, alpha=0.14)
    ax_zoom.axhline(-0.2, color="red", linestyle="--", linewidth=1.1, alpha=0.6)
    ax_zoom.axhline(0.3, color="red", linestyle="--", linewidth=1.1, alpha=0.6)
    ax_zoom.axhline(-0.3, color="#1f77b4", linestyle="--", linewidth=1.0, alpha=0.7)
    ax_zoom.axvline(1.0, color="red", linestyle="--", linewidth=1.1, alpha=0.8)
    ax_zoom.set_xlim(zoom_x_min, zoom_x_max)
    ax_zoom.set_ylim(min(zoom_lower) - 0.08, max(zoom_upper) + 0.08)
    ax_zoom.set_xlabel("speedup")
    ax_zoom.set_ylabel("reward")
    ax_zoom.set_title("zoom window: speedup 0-2")

    fig.tight_layout()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Plot reward functions with seaborn.")
    parser.add_argument(
        "--output",
        type=str,
        default="reward/figures/reward_functions_actual.png",
        help="Output image path.",
    )
    parser.add_argument(
        "--token-len",
        type=int,
        default=100,
        help="token_len used for final-reward curves.",
    )
    parser.add_argument(
        "--novelty-token-lens",
        type=str,
        default="32,100,320",
        help="Comma-separated token lengths for novelty mapping curves.",
    )
    parser.add_argument(
        "--speedup-max",
        type=float,
        default=12.0,
        help="Max speedup shown on x-axis.",
    )
    parser.add_argument(
        "--points",
        type=int,
        default=240,
        help="Sampling points for each curve.",
    )
    parser.add_argument(
        "--plot-soft-only",
        action="store_true",
        help="Generate one dedicated figure for soft_clip_novelty only.",
    )
    parser.add_argument(
        "--soft-only-output",
        type=str,
        default="reward/figures/reward_soft_clip_novelty_only.png",
        help="Output image path for --plot-soft-only mode.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    novelty_token_lens = _parse_int_list(args.novelty_token_lens)
    output_path = Path(args.output)
    points = max(10, args.points)

    if args.plot_soft_only:
        soft_output = Path(args.soft_only_output)
        plot_soft_clip_novelty_only(
            output_path=soft_output,
            token_len=args.token_len,
            speedup_max=args.speedup_max,
            points=points,
        )
        d_copy, d_low, d_high = get_adaptive_thresholds(args.token_len)
        print(f"[OK] saved soft-only plot to: {soft_output}")
        print(
            "[INFO] token_len=%d thresholds: d_copy=%.4f d_low=%.4f d_high=%.4f"
            % (args.token_len, d_copy, d_low, d_high)
        )
        print("[INFO] mode: soft_clip_novelty")
        return

    plot_reward_functions(
        output_path=output_path,
        token_len=args.token_len,
        novelty_token_lens=novelty_token_lens,
        speedup_max=args.speedup_max,
        points=points,
    )

    d_copy, d_low, d_high = get_adaptive_thresholds(args.token_len)
    print(f"[OK] saved plot to: {output_path}")
    print(
        "[INFO] token_len=%d thresholds: d_copy=%.4f d_low=%.4f d_high=%.4f"
        % (args.token_len, d_copy, d_low, d_high)
    )
    print("[INFO] README pipeline <-> mode: soft_clip_novelty")


if __name__ == "__main__":
    main()
