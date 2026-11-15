"""
test_calibration.py

Pytest suite for the Von Stosch Hybrid Digital Twin sub-agent.
Run with:
    python -u -m pytest agents/von_stosch_twin/test_calibration.py -s
"""

from __future__ import annotations

from agents.von_stosch_twin.auditor import (
    run_benchmark_pinto_2019,
    run_benchmark_willis_2017,
)


def test_benchmark_1_willis_2017() -> None:
    """Gate 3: Willis & von Stosch (2017) sparse structure recovery & identifiability."""
    res = run_benchmark_willis_2017(seed=2017)
    assert res["passed"], f"Willis 2017 benchmark failed: {res}"
    assert res["selected_names"] == ["(S/(S+0.5))/(1+P/15)"]
    assert res["n_decoys"] == 0
    assert res["gamma_col"] < 20.0


def test_benchmark_2_pinto_2019() -> None:
    """Gates 1, 2, 4, 5: Pinto & von Stosch (2019) fed-batch DoE bagging benchmark."""
    res = run_benchmark_pinto_2019(seed=2019)
    assert res["elapsed_s"] < 30.0, f"Training took too long: {res['elapsed_s']:.2f}s"
    assert res["min_s_pred"] >= 0.0, f"Substrate went negative: {res['min_s_pred']}"
    wmse_true = res["wmse_true_by_n"]
    assert wmse_true[8] < 2.0, f"Bagged (n=8) True WMSE too high: {wmse_true[8]:.4f}"
    assert wmse_true[8] < res["single_median_true"], (
        f"Ensemble (n=8) True WMSE {wmse_true[8]:.4f} did not beat single-model median {res['single_median_true']:.4f}"
    )
    assert res["gamma_col_top1"] < 20.0, (
        f"Collinearity index gamma_col={res['gamma_col_top1']:.2f} >= 20"
    )
    assert res["coverage_pct"] >= 85.0, (
        f"95% Epistemic CI coverage {res['coverage_pct']:.2f}% < 85%"
    )
