"""Command-line reproduction runner for 6-param Stage-1 recovery, GAM-86 hybrid training, and control."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, Optional

from .controller import benchmark_controllers
from .estimator import BioreactorDigitalTwin
from .mechanistic import BioreactorParams
from .neural_ode import recover_mechanistic_parameters, train_default_hybrid_model


def _print_recovery_table(recovery: Dict[str, Any]) -> None:
    print("\n=== Stage 1: Multi-Batch iDoE Mechanistic Parameter Recovery ===")
    print(f"{'Parameter':<10} {'True':>10} {'Recovered':>12} {'SE':>10} {'RelErr %':>10}")
    print("-" * 56)
    true_dict = recovery["true_values"]
    rec_dict = recovery["recovered_values"]
    se_dict = recovery["standard_errors"]
    rel_dict = recovery["relative_error_pct"]
    for name in recovery["param_names"]:
        print(
            f"{name:<10} {true_dict[name]:>10.4f} {rec_dict[name]:>12.4f} "
            f"{se_dict[name]:>10.4f} {rel_dict[name]:>9.2f}%"
        )
    print(
        f"Max Rel Error: {recovery['max_relative_error_pct']:.2f}% | "
        f"Relative FIM Cond Num: {recovery['condition_number_rel']:.2f} | "
        f"SHA256: {str(recovery['calibration_sha256'])[:12]}"
    )


def _print_neural_ode_table(model_metrics: Dict[str, Any]) -> None:
    print(f"\n=== Stage 2: Parsimonious GAM-{int(model_metrics['n_params'])} Hybrid Neural ODE + Bootstrap Ensemble ===")
    print(
        f"Params: {int(model_metrics['n_params'])} | "
        f"Bootstrap B: {int(model_metrics['n_bootstrap'])} | "
        f"AIC: {model_metrics['aic']:.1f} | BIC: {model_metrics['bic']:.1f} | "
        f"DoV p99: {model_metrics['dov_p99']:.2f}"
    )
    print(
        f"Specific Rates (held-out) R2 -> "
        f"k_d: {model_metrics['kd_r2']:.4f} | "
        f"k_La: {model_metrics['kla_r2']:.4f} | "
        f"q_A_net: {model_metrics['qAnet_r2']:.4f} | "
        f"mu: {model_metrics['mu_r2']:.4f}"
    )
    if "train_kd_r2" in model_metrics:
        print(
            f"Specific Rates (train)    R2 -> "
            f"k_d: {model_metrics['train_kd_r2']:.4f} | "
            f"k_La: {model_metrics['train_kla_r2']:.4f} | "
            f"q_A_net: {model_metrics['train_qAnet_r2']:.4f} | "
            f"mu: {model_metrics['train_mu_r2']:.4f}"
        )
    imp_pct = 100.0 * (
        1.0
        - model_metrics["hybrid_norm_rmse"]
        / max(1e-12, model_metrics["nominal_norm_rmse"])
    )
    print(
        f"Open-Loop Rollout nRMSE  -> "
        f"Nominal: {model_metrics['nominal_norm_rmse']:.4f} | "
        f"Hybrid: {model_metrics['hybrid_norm_rmse']:.4f} | "
        f"Improvement: {imp_pct:.1f}%"
    )


def _print_control_table(bench: Dict[str, Dict[str, Any]]) -> None:
    print("\n=== Stage 3: Closed-Loop Control & Sensor-Parity EKF Benchmark ===")
    print(
        f"{'Strategy':<20} {'Xv_end':>8} {'P_end':>8} {'A_peak':>8} "
        f"{'mu_RMSE':>8} {'Hyb_nRMSE':>10} {'Nom_nRMSE':>10} {'max_Dm':>8}"
    )
    print("-" * 86)
    for label, entry in bench.items():
        s = entry["summary"]
        print(
            f"{label:<20} "
            f"{s['final_Xv_g_L']:>8.2f} "
            f"{s['final_P_g_L']:>8.2f} "
            f"{s['peak_acetate_g_L']:>8.2f} "
            f"{s['mu_rmse']:>8.4f} "
            f"{s['hybrid_norm_rmse']:>10.4f} "
            f"{s['nominal_norm_rmse']:>10.4f} "
            f"{s['max_d_m']:>8.2f}"
        )


def run_cli(
    mode: str = "all",
    export_json: Optional[str] = None,
    n_bootstrap: int = 15,
) -> Dict[str, Any]:
    """Execute end-to-end calibration, hybrid training, and control benchmark pipeline."""
    true_params = BioreactorParams()
    results: Dict[str, Any] = {}

    recovery = recover_mechanistic_parameters(true_params=true_params)
    params_rec = recovery["recovered_params"]
    if mode in ("recovery", "all"):
        _print_recovery_table(recovery)
    results["recovery"] = {
        "param_names": recovery["param_names"],
        "true_values": recovery["true_values"],
        "recovered_values": recovery["recovered_values"],
        "standard_errors": recovery["standard_errors"],
        "relative_error_pct": recovery["relative_error_pct"],
        "max_relative_error_pct": recovery["max_relative_error_pct"],
        "condition_number_rel": recovery["condition_number_rel"],
        "calibration_sha256": recovery["calibration_sha256"],
    }

    model = None
    if mode in ("neural", "control", "all"):
        model, neural_metrics = train_default_hybrid_model(
            params=params_rec,
            plant_params=true_params,
            n_bootstrap=n_bootstrap,
        )
        if mode in ("neural", "all"):
            _print_neural_ode_table(neural_metrics)
        results["neural_ode"] = {
            k: (float(v) if isinstance(v, (int, float)) else str(v))
            for k, v in neural_metrics.items()
        }
        results["neural_ode"]["weights_sha256"] = model.weights_sha256

    if mode in ("control", "all") and model is not None:
        bench = benchmark_controllers(
            params=params_rec,
            residual_model=model,
            duration_h=16.0,
            mu_target=0.28,
            rpm=450.0,
            vvm=1.2,
            plant_params=true_params,
        )
        _print_control_table(bench)
        results["control"] = {k: v["summary"] for k, v in bench.items()}

        twin = BioreactorDigitalTwin.from_simulation_run(
            bench["hybrid_closed_loop"]["simulation"],
            params=params_rec,
            residual_model=model,
            calibration_metadata=results["recovery"],
        )
        results["alcoa_provenance"] = twin.to_audit_record()

    if export_json:
        out_path = Path(export_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(results, indent=2, default=str), encoding="utf-8")
        print(f"\nSaved JSON report -> {out_path}")

    return results


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Hybrid Mechanistic-ML Bioreactor State Estimator CLI"
    )
    parser.add_argument(
        "--mode",
        choices=("recovery", "neural", "control", "all"),
        default="all",
        help="Pipeline stage(s) to run.",
    )
    parser.add_argument(
        "--export-json",
        type=str,
        default=None,
        help="Optional path to export benchmark and ALCOA+ provenance summary JSON.",
    )
    parser.add_argument(
        "--n-bootstrap",
        type=int,
        default=15,
        help="Number of bootstrap ensemble members for epistemic UQ.",
    )
    args = parser.parse_args()
    run_cli(mode=args.mode, export_json=args.export_json, n_bootstrap=args.n_bootstrap)


if __name__ == "__main__":
    main()
