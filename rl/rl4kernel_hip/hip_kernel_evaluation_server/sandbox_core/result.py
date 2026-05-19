from __future__ import annotations

from typing import Any, Dict, NamedTuple


class EvalRunResult(NamedTuple):
    compile_ok: bool
    run_ok: bool
    match_ok: bool
    speedup: float
    timing: Dict[str, Any]

    @classmethod
    def failure(
        cls,
        *,
        compile_ok: bool,
        run_ok: bool,
        match_ok: bool,
        speedup: float = 0.0,
        timing: Dict[str, Any] | None = None,
    ) -> "EvalRunResult":
        return cls(
            compile_ok=compile_ok,
            run_ok=run_ok,
            match_ok=match_ok,
            speedup=speedup,
            timing=timing or {},
        )
