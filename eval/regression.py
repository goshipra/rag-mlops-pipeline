"""
eval/regression.py — compare a fresh eval run's metrics against a stored
baseline and decide pass/fail. Pure function, no I/O, so it is directly
unit-testable (see tests/test_regression.py). check_regression.py is the
thin CLI wrapper that loads JSON files and calls this.
"""
from __future__ import annotations

from dataclasses import dataclass

# Metrics we gate on, and the tolerance (in absolute score points, e.g. 0.02
# == 2 percentage points) each is allowed to drop below baseline before it
# counts as a regression.
GATED_METRICS: dict[str, float] = {
    "retrieval_recall_at_k": 0.02,
    "answer_quality_avg": 0.02,
}


@dataclass
class RegressionResult:
    passed: bool
    messages: list[str]


def check_regression(
    current: dict,
    baseline: dict,
    tolerance_overrides: dict[str, float] | None = None,
) -> RegressionResult:
    """Compare ``current`` metrics against ``baseline`` metrics.

    A metric regresses if ``current[metric] < baseline[metric] - tolerance``.
    Missing a gated metric in either dict is treated as a hard failure (fail
    closed, not open — a broken eval run should never silently pass CI).
    """
    tolerances = {**GATED_METRICS, **(tolerance_overrides or {})}
    messages: list[str] = []
    passed = True

    for metric, tolerance in tolerances.items():
        if metric not in baseline:
            passed = False
            messages.append(f"FAIL {metric}: missing from baseline.json")
            continue
        if metric not in current:
            passed = False
            messages.append(f"FAIL {metric}: missing from current run")
            continue

        base_val = baseline[metric]
        cur_val = current[metric]
        floor = base_val - tolerance
        delta = cur_val - base_val

        if cur_val < floor:
            passed = False
            messages.append(
                f"FAIL {metric}: {cur_val:.4f} < baseline {base_val:.4f} "
                f"- tolerance {tolerance:.4f} (floor {floor:.4f}, delta {delta:+.4f})"
            )
        else:
            verdict = "OK" if delta >= 0 else "OK (within tolerance)"
            messages.append(
                f"{verdict} {metric}: {cur_val:.4f} vs baseline {base_val:.4f} "
                f"(delta {delta:+.4f}, floor {floor:.4f})"
            )

    return RegressionResult(passed=passed, messages=messages)
