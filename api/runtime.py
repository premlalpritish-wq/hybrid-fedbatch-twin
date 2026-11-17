"""Asset loading, scale-up parameterization, and BioreactorDigitalTwin serialization helpers."""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
from typing import Any, Dict, Literal, Optional, Tuple
import numpy as np

from hybrid_fedbatch_twin.estimator import BioreactorDigitalTwin
from hybrid_fedbatch_twin.mechanistic import (
    BioreactorParams,
    OperatingInputs,
    STATE_NAMES,
    default_initial_state,
)
from hybrid_fedbatch_twin.neural_ode import PhysicsInformedNeuralODE

ROOT_DIR = Path(__file__).resolve().parent.parent
DEFAULT_BUNDLE_PATH = ROOT_DIR / "docs" / "assets" / "twin_bundle.json"

VesselScale = Literal[5, 50, 1000]


def _load_bundle_model(bundle_path: Path) -> Tuple[BioreactorParams, PhysicsInformedNeuralODE]:
    """Hydrate Stage-1 BioreactorParams and Stage-2 GAM-86 (B=15) from exported JSON."""
    with bundle_path.open("r", encoding="utf-8") as f:
        bundle = json.load(f)

    p_rec = BioreactorParams(**bundle["p_rec"])
    g = bundle["gam86"]

    model = PhysicsInformedNeuralODE(n_bootstrap=int(g.get("n_bootstrap", 15)))
    model.weights = np.asarray(g["weights"], dtype=float)
    model.ensemble_weights = [np.asarray(w, dtype=float) for w in g["ensemble_weights"]]

    dov_raw = g.get("dov_stats", {})
    dov_stats: Dict[str, Tuple[np.ndarray, ...]] = {}
    for key in ("kd", "kla", "met", "global"):
        if key in dov_raw:
            entry = dov_raw[key]
            dov_stats[key] = (
                np.asarray(entry["mean"], dtype=float),
                np.asarray(entry["cov_inv"], dtype=float),
                float(entry["p99"]),
                np.asarray(entry["std"], dtype=float),
            )
    model.dov_stats = dov_stats
    model.fit_metadata = dict(g.get("fit_metadata", {}))
    model.is_trained = True
    model._update_checksum()
    return p_rec, model


@lru_cache(maxsize=1)
def load_calibrated_assets(
    bundle_path: Optional[str] = None,
) -> Tuple[BioreactorParams, PhysicsInformedNeuralODE]:
    """Return cached calibrated Stage-1 params and Stage-2 GAM-86 model."""
    path = Path(bundle_path) if bundle_path else DEFAULT_BUNDLE_PATH
    if path.exists():
        return _load_bundle_model(path)
    return BioreactorParams(), PhysicsInformedNeuralODE(n_bootstrap=1)


def scale_params(vessel_scale_L: VesselScale, base_params: BioreactorParams) -> BioreactorParams:
    """Scale geometric vessel dimensions for 5 L, 50 L, or 1000 L operation."""
    v_scale = float(vessel_scale_L)
    if v_scale == float(base_params.V_vessel):
        return base_params
    return BioreactorParams(
        **{
            **base_params.to_dict(),
            "V_vessel": v_scale,
            "V_0": 0.4 * v_scale,
            "V_max": v_scale,
        }
    )


def scale_feed_cap(vessel_scale_L: VesselScale, base_f_max_5L: float = 0.30) -> float:
    """Scale maximum feed flow rate proportionally to vessel volume."""
    return float(base_f_max_5L) * (float(vessel_scale_L) / 5.0)


def create_twin_instance(
    vessel_scale_L: VesselScale,
    batch_id: str,
    operator_id: str,
    calibration_metadata: Optional[Dict[str, Any]] = None,
) -> BioreactorDigitalTwin:
    """Instantiate a fresh BioreactorDigitalTwin at the requested vessel scale."""
    base_params, model = load_calibrated_assets()
    p_scaled = scale_params(vessel_scale_L, base_params)
    x0 = default_initial_state(p_scaled)
    meta = dict(calibration_metadata or {})
    meta.setdefault("vessel_scale_L", int(vessel_scale_L))
    return BioreactorDigitalTwin(
        x0=x0,
        params=p_scaled,
        residual_model=model,
        calibration_metadata=meta,
        operator_id=operator_id,
        batch_id=batch_id,
    )


def serialize_operating_inputs(u: OperatingInputs) -> Dict[str, Any]:
    """Convert OperatingInputs to a JSON-serializable dict."""
    return {
        "F": float(u.F),
        "rpm": float(u.rpm),
        "vvm": float(u.vvm),
        "S_feed": None if u.S_feed is None else float(u.S_feed),
    }


def deserialize_operating_inputs(payload: Dict[str, Any]) -> OperatingInputs:
    """Reconstruct OperatingInputs from a JSON dict."""
    return OperatingInputs(
        F=float(payload["F"]),
        rpm=float(payload["rpm"]),
        vvm=float(payload["vvm"]),
        S_feed=None if payload.get("S_feed") is None else float(payload["S_feed"]),
    )


def state_dict_from_vector(vec: np.ndarray) -> Dict[str, float]:
    """Map a 7-element state or std vector to named state dict."""
    return {name: float(vec[i]) for i, name in enumerate(STATE_NAMES)}
