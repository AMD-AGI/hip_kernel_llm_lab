"""Canonical output layout helpers for HIP_benchmark_kit runs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from .eval_schema import (
    BASELINE_RESULTS_CSV,
    BASELINE_RESULTS_JSON,
    COMPARISON_PERF_TRACE_CSV,
    COMPARISON_RESULTS_CSV,
    COMPARISON_RESULTS_JSON,
)
from .manifests import GENERATION_MANIFEST, SUBSET_MANIFEST

DEFAULT_LEVELS = ("level-1", "level-2", "level-3")
KERNELBENCH_SUBSET_NAME = "kernelbench_hip_100"


def repo_root() -> Path:
    return Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class LevelRunLayout:
    """Paths for one level inside a KernelBench HIP run."""

    run_root: Path
    level: str

    @property
    def root(self) -> Path:
        return self.run_root / self.level

    @property
    def generated_dir(self) -> Path:
        return self.root / "generated"

    @property
    def generation_manifest(self) -> Path:
        return self.generated_dir / GENERATION_MANIFEST

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def origin_eval_dir(self) -> Path:
        return self.eval_dir / "origin_eval"

    @property
    def optimized_eval_dir(self) -> Path:
        return self.eval_dir / "optimized_eval"

    @property
    def comparison_dir(self) -> Path:
        return self.eval_dir / "comparison"

    @property
    def comparison_json(self) -> Path:
        return self.comparison_dir / COMPARISON_RESULTS_JSON

    @property
    def comparison_csv(self) -> Path:
        return self.comparison_dir / COMPARISON_RESULTS_CSV

    @property
    def comparison_perf_trace_csv(self) -> Path:
        return self.comparison_dir / COMPARISON_PERF_TRACE_CSV

    @property
    def staging_dir(self) -> Path:
        return self.eval_dir / "staging"


@dataclass(frozen=True)
class TurnRunLayout:
    """Paths for one feedback turn inside a level run."""

    level_root: Path
    turn: int

    @property
    def name(self) -> str:
        return f"turn_{self.turn:02d}"

    @property
    def root(self) -> Path:
        return self.level_root / self.name

    @property
    def generated_dir(self) -> Path:
        return self.root / "generated"

    @property
    def raw_response_dir(self) -> Path:
        return self.root / "raw_responses"

    @property
    def eval_dir(self) -> Path:
        return self.root / "eval"

    @property
    def comparison_json(self) -> Path:
        return self.eval_dir / "comparison" / COMPARISON_RESULTS_JSON

    @property
    def profiling_dir(self) -> Path:
        return self.root / "profiling"

    @property
    def profiling_generated_dir(self) -> Path:
        return self.profiling_dir / "generated"

    @property
    def profiling_staging_dir(self) -> Path:
        return self.profiling_dir / "staging" / "generated_hip_code"

    @property
    def feedback_dir(self) -> Path:
        return self.root / "feedback"

    def feedback_context_json_for_next_turn(self) -> Path:
        return self.feedback_dir / f"turn_{self.turn + 1:02d}_context.json"


@dataclass(frozen=True)
class KernelBenchRunLayout:
    """Canonical paths for a KernelBench HIP rollout run."""

    run_root: Path
    subset_name: str = KERNELBENCH_SUBSET_NAME

    @property
    def subset_root(self) -> Path:
        return self.run_root / "subset" / self.subset_name

    @property
    def subset_manifest(self) -> Path:
        return self.run_root / "subset" / SUBSET_MANIFEST

    @property
    def summary_dir(self) -> Path:
        return self.run_root / "summary"

    def level(self, level: str) -> LevelRunLayout:
        return LevelRunLayout(self.run_root, level)

    def turn(self, level: str, turn: int) -> TurnRunLayout:
        return TurnRunLayout(self.run_root / level, turn)


@dataclass(frozen=True)
class BaselineEvalLayout:
    """Paths for a single baseline eval output directory."""

    output_dir: Path

    @property
    def json_path(self) -> Path:
        return self.output_dir / BASELINE_RESULTS_JSON

    @property
    def csv_path(self) -> Path:
        return self.output_dir / BASELINE_RESULTS_CSV
