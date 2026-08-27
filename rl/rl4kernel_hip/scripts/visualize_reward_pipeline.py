# SPDX-License-Identifier: Apache-2.0
# Copyright (C) 2026, Advanced Micro Devices, Inc. All rights reserved.

"""Generate a pipeline diagram for correct_speedup_copy_penalty reward mode."""
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

fig, ax = plt.subplots(1, 1, figsize=(18, 28))
ax.set_xlim(0, 18)
ax.set_ylim(0, 28)
ax.axis('off')
fig.patch.set_facecolor('#0d1117')

# ── Colors ──
C_BG      = '#0d1117'
C_STAGE   = '#161b22'
C_BLUE    = '#58a6ff'
C_TEAL    = '#3fb950'
C_YELLOW  = '#d29922'
C_RED     = '#f85149'
C_PURPLE  = '#bc8cff'
C_GRAY    = '#8b949e'
C_WHITE   = '#e6edf3'
C_ORANGE  = '#db6d28'
C_GREEN   = '#238636'
C_PINK    = '#f778ba'

def draw_box(x, y, w, h, text, color, fontsize=9, textcolor='white', alpha=0.9, style='round,pad=0.3'):
    box = FancyBboxPatch((x - w/2, y - h/2), w, h,
                         boxstyle=style,
                         facecolor=color, edgecolor='white',
                         linewidth=1.0, alpha=alpha, zorder=3)
    ax.add_patch(box)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color=textcolor, fontweight='bold', zorder=4, family='monospace')

def draw_diamond(x, y, w, h, text, color, fontsize=8):
    diamond = plt.Polygon([(x, y+h/2), (x+w/2, y), (x, y-h/2), (x-w/2, y)],
                          facecolor=color, edgecolor='white', linewidth=1.0,
                          alpha=0.9, zorder=3)
    ax.add_patch(diamond)
    ax.text(x, y, text, ha='center', va='center', fontsize=fontsize,
            color='white', fontweight='bold', zorder=4, family='monospace')

def arrow(x1, y1, x2, y2, color='white', lw=1.5, style='->', label='', label_color=None):
    ax.annotate('', xy=(x2, y2), xytext=(x1, y1),
                arrowprops=dict(arrowstyle=style, color=color, lw=lw),
                zorder=2)
    if label:
        mx, my = (x1+x2)/2, (y1+y2)/2
        ax.text(mx+0.15, my+0.05, label, fontsize=7, color=label_color or color,
                fontweight='bold', family='monospace', zorder=5)

def stage_label(y, text, color):
    ax.text(0.3, y, text, fontsize=10, color=color, fontweight='bold',
            family='monospace', rotation=90, va='center', ha='center', zorder=5)

def section_bg(y_top, y_bot, color, alpha=0.08):
    rect = plt.Rectangle((0.5, y_bot), 17, y_top - y_bot,
                          facecolor=color, alpha=alpha, zorder=0,
                          linewidth=0)
    ax.add_patch(rect)
    ax.plot([0.5, 17.5], [y_top, y_top], color=color, alpha=0.3, lw=0.8, ls='--', zorder=1)

# ══════════════════════════════════════════════════════════════
# Title
# ══════════════════════════════════════════════════════════════
ax.text(9, 27.5, 'correct_speedup_copy_penalty  Reward Pipeline',
        ha='center', va='center', fontsize=16, color=C_WHITE,
        fontweight='bold', family='monospace')
ax.text(9, 27.1, 'reward/reward_batch.py  ·  compute_score_batch → _compute_single_score_correct_speedup_copy_penalty',
        ha='center', va='center', fontsize=8, color=C_GRAY, family='monospace')

# ══════════════════════════════════════════════════════════════
# Stage 0: Input
# ══════════════════════════════════════════════════════════════
draw_box(9, 26.5, 4.5, 0.55, 'LLM Response  (raw_response)', C_BLUE, fontsize=10)

# ══════════════════════════════════════════════════════════════
# Stage 1: Code Parsing (y: 24.5 ~ 26.0)
# ══════════════════════════════════════════════════════════════
section_bg(26.1, 23.0, C_TEAL)
stage_label(24.5, 'STAGE 1: PARSE', C_TEAL)

arrow(9, 26.22, 9, 25.85, C_GRAY)

# three branches
draw_box(4, 25.5, 4.2, 0.5, 'hip2hip\nstrip_code_fences()', C_TEAL, fontsize=7.5)
draw_box(9, 25.5, 4.2, 0.5, 'agent-single-sft\nextract_code_from_json()', C_TEAL, fontsize=7.5)
draw_box(14, 25.5, 4.2, 0.5, 'agent-react\nextract_hip_kernel_code()', C_TEAL, fontsize=7.5)

arrow(9, 25.85, 4, 25.78, C_GRAY, lw=1)
arrow(9, 25.85, 14, 25.78, C_GRAY, lw=1)

# react validation diamond
draw_diamond(14, 24.8, 2.4, 0.55, 'Valid kernel?\n(__global__)', C_RED, fontsize=7)
arrow(14, 25.22, 14, 25.1, C_GRAY)
draw_box(17, 24.8, 1.8, 0.4, 'reward = -0.9\nSTOP', C_RED, fontsize=7)
arrow(15.2, 24.8, 16.1, 24.8, C_RED, label='No', label_color=C_RED)

# merge
arrow(4, 25.22, 9, 24.15, C_GRAY, lw=1)
arrow(9, 25.22, 9, 24.15, C_GRAY, lw=1)
arrow(14, 24.5, 9, 24.15, C_GRAY, lw=1)

draw_box(9, 23.9, 5.5, 0.4, 'replace_kernel_in_hip_code(hip_ref, kernel_src)', '#2ea043', fontsize=8)
arrow(9, 23.68, 9, 23.45, C_GRAY)
draw_box(9, 23.25, 4.5, 0.35, 'kernel_name += "_" + md5(code)[:8]', '#2ea043', fontsize=8)

# ══════════════════════════════════════════════════════════════
# Stage 2: Batch Server Evaluation (y: 21.5 ~ 23.0)
# ══════════════════════════════════════════════════════════════
section_bg(23.0, 21.0, C_YELLOW)
stage_label(22.0, 'STAGE 2: EVAL', C_YELLOW)

arrow(9, 23.05, 9, 22.65, C_GRAY)
draw_box(9, 22.35, 6.0, 0.5, 'HTTP POST  /run_code_batch\n→ HIP Evaluation Server (compile+run+benchmark)', C_YELLOW, fontsize=8, textcolor='black')

arrow(9, 22.08, 9, 21.72, C_GRAY)
draw_box(9, 21.5, 5.5, 0.35, 'Returns:  compile_ok  |  run_ok  |  match_ok  |  speedup', '#b08800', fontsize=8)

# HTTP error side branch
draw_box(15.5, 22.35, 2.2, 0.4, 'HTTP Error\nreward = 0.0\n∀ samples', C_ORANGE, fontsize=7)
arrow(12, 22.35, 14.4, 22.35, C_ORANGE, label='Error', label_color=C_ORANGE)

# ══════════════════════════════════════════════════════════════
# Stage 3: DTW Similarity (y: 19.5 ~ 21.0)
# ══════════════════════════════════════════════════════════════
section_bg(21.0, 19.2, C_GREEN)
stage_label(20.1, 'STAGE 3: DTW', C_GREEN)

arrow(9, 21.3, 9, 20.72, C_GRAY)
draw_box(9, 20.45, 6.5, 0.45, 'compute_dtw_to_ref(hip_ref, hip_gen, kernel_name)\n→ dtw_to_ref ∈ [0,1],  token_len', C_GREEN, fontsize=8)

arrow(9, 20.2, 9, 19.85, C_GRAY)
draw_box(9, 19.6, 6.0, 0.4, 'get_adaptive_thresholds(token_len)\n→ (d_copy, d_low, d_high)', '#196c2e', fontsize=8)

# ══════════════════════════════════════════════════════════════
# Stage 4: Reward Computation (y: 14.5 ~ 19.0)
# ══════════════════════════════════════════════════════════════
section_bg(19.2, 14.2, C_PURPLE)
stage_label(16.7, 'STAGE 4: REWARD', C_PURPLE)

arrow(9, 19.38, 9, 19.0, C_GRAY)

# Header
ax.text(9, 18.85, '_compute_single_score_correct_speedup_copy_penalty()', ha='center',
        fontsize=9, color=C_PURPLE, fontweight='bold', family='monospace')

# Gate 1: compile
draw_diamond(9, 18.2, 2.5, 0.55, 'compile_ok?', C_RED, fontsize=8)
arrow(9, 18.6, 9, 18.5, C_GRAY)
draw_box(14.5, 18.2, 2.2, 0.35, 'reward = 0.0', '#6e4040', fontsize=8)
arrow(10.25, 18.2, 13.4, 18.2, C_RED, label='No', label_color=C_RED)

# Gate 2: run
arrow(9, 17.92, 9, 17.65, C_GRAY, label='Yes', label_color=C_GREEN)
draw_diamond(9, 17.3, 2.5, 0.55, 'run_ok?', C_RED, fontsize=8)
draw_box(14.5, 17.3, 2.2, 0.35, 'reward = 0.0', '#6e4040', fontsize=8)
arrow(10.25, 17.3, 13.4, 17.3, C_RED, label='No', label_color=C_RED)

# Gate 3: match
arrow(9, 17.02, 9, 16.75, C_GRAY, label='Yes', label_color=C_GREEN)
draw_diamond(9, 16.4, 2.5, 0.55, 'match_ok?', C_RED, fontsize=8)
draw_box(14.5, 16.4, 2.2, 0.35, 'reward = 0.0', '#6e4040', fontsize=8)
arrow(10.25, 16.4, 13.4, 16.4, C_RED, label='No', label_color=C_RED)

# speedup_eff
arrow(9, 16.12, 9, 15.85, C_GRAY, label='Yes', label_color=C_GREEN)
draw_box(9, 15.6, 5.5, 0.4, 'speedup_eff = clip(speedup, 0, cap=10.0)', C_PURPLE, fontsize=9)

# Copy detection diamond
arrow(9, 15.38, 9, 15.05, C_GRAY)
draw_diamond(9, 14.65, 3.0, 0.6, 'd_ref < d_copy ?\n(Copy Detected?)', C_PINK, fontsize=8)

# Copy branch
draw_box(14.5, 14.65, 2.8, 0.5, 'reward =\ncopy_reward\n= 0.0', C_ORANGE, fontsize=8)
arrow(10.5, 14.65, 13.1, 14.65, C_ORANGE, label='Yes (Copy)', label_color=C_ORANGE)

# Novel branch
draw_box(4.0, 14.65, 3.2, 0.5, 'reward =\nr_ok + speedup_eff\n= 0.3 + speedup', C_GREEN, fontsize=8)
arrow(7.5, 14.65, 5.6, 14.65, C_GREEN, label='No (Novel)', label_color=C_GREEN)

# ══════════════════════════════════════════════════════════════
# Stage 5: Output
# ══════════════════════════════════════════════════════════════
section_bg(14.2, 13.0, C_BLUE)

arrow(9, 14.32, 9, 13.9, C_GRAY)
draw_box(9, 13.6, 7.0, 0.5, 'Final Reward', C_BLUE, fontsize=11)

# Reward table
table_y = 12.7
ax.text(9, table_y, '┌──────────────────────────────────────────────────────────────┐', ha='center', fontsize=7.5, color=C_GRAY, family='monospace')
table_y -= 0.3
ax.text(9, table_y, '│  react parse fail          │  -0.9                          │', ha='center', fontsize=7.5, color=C_RED, family='monospace')
table_y -= 0.25
ax.text(9, table_y, '│  compile/run/match fail     │   0.0                          │', ha='center', fontsize=7.5, color=C_YELLOW, family='monospace')
table_y -= 0.25
ax.text(9, table_y, '│  correct but copy           │   0.0  (= copy_reward)         │', ha='center', fontsize=7.5, color=C_ORANGE, family='monospace')
table_y -= 0.25
ax.text(9, table_y, '│  correct & novel, spd=0     │   0.3  (= r_ok only)           │', ha='center', fontsize=7.5, color=C_GREEN, family='monospace')
table_y -= 0.25
ax.text(9, table_y, '│  correct & novel, spd=5     │   5.3  (= 0.3 + 5.0)          │', ha='center', fontsize=7.5, color=C_GREEN, family='monospace')
table_y -= 0.25
ax.text(9, table_y, '│  correct & novel, spd=15    │  10.3  (= 0.3 + cap 10.0)     │', ha='center', fontsize=7.5, color=C_TEAL, family='monospace')
table_y -= 0.25
ax.text(9, table_y, '└──────────────────────────────────────────────────────────────┘', ha='center', fontsize=7.5, color=C_GRAY, family='monospace')

# Tracker
arrow(9, 13.33, 9, 11.1, C_GRAY, lw=1)
draw_box(9, 10.85, 4.5, 0.4, 'Record to Kernel Novelty Tracker', '#30363d', fontsize=8, textcolor=C_GRAY)

plt.tight_layout(pad=0.5)
plt.savefig('/wekafs/zeping/rl4kernel_hip/correct_speedup_copy_penalty_pipeline.png',
            dpi=180, bbox_inches='tight', facecolor=C_BG)
print("Saved to correct_speedup_copy_penalty_pipeline.png")
