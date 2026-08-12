#!/usr/bin/env python3
"""
eval/check_regression.py — the CI regression gate.

Compares a fresh eval run's metrics (eval/results/latest.json, produced by
eval/run_eval.py) against the stored baseline (eval/baseline.json) and exits
non-zero if retrieval recall@k or answer-quality dropped by more than the
allowed tolerance. This is the script GitHub Actions calls to fail the
build on a quality regression.

Usage:
    python eval/check_regression.py \\
        [--current eval/results/latest.json] [--baseline eval/baseline.json]

To intentionally move the baseline forward after a verified improvement:
    python eval/run_eval.py
    cp eval/results/latest.json eval/baseline.json   # then review + commit
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from eval.regression import check_regression  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--current", default=str(REPO_ROOT / "eval" / "results" / "latest.json"))
    parser.add_argument("--baseline", default=str(REPO_ROOT / "eval" / "baseline.json"))
    args = parser.parse_args()

    current_path = Path(args.current)
    baseline_path = Path(args.baseline)

    if not current_path.exists():
        print(f"FAIL: no eval results found at {current_path} — did run_eval.py run first?")
        return 1
    if not baseline_path.exists():
        print(f"FAIL: no baseline found at {baseline_path}")
        return 1

    current = json.loads(current_path.read_text(encoding="utf-8"))
    baseline = json.loads(baseline_path.read_text(encoding="utf-8"))

    result = check_regression(current, baseline)

    print("=== CI regression gate: eval/check_regression.py ===")
    print(f"baseline: {baseline_path}")
    print(f"current:  {current_path}")
    for message in result.messages:
        print(f"  {message}")

    if result.passed:
        print("PASS: no metric regressed beyond tolerance.")
        return 0

    print("FAIL: quality regression detected — failing the build.")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
