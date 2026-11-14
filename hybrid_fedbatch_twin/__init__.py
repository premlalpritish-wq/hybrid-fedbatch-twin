"""Hybrid Mechanistic-ML Bioprocess Digital Twin for E. coli BL21 fed-batch fermentation."""

import os

for _env_key in ("OMP_NUM_THREADS", "MKL_NUM_THREADS", "OPENBLAS_NUM_THREADS"):
    os.environ.setdefault(_env_key, "1")

from .controller import (
    ClosedLoopFeedController,
    ManualFeedController,
    NominalMechanisticController,
    OpenLoopExponentialController,
    STATE_NORM_SCALES,
    benchmark_controllers,
    summarize_control_run,
)
from .estimator import (
    BioreactorDigitalTwin,
    HybridExtendedKalmanFilter,
    Q_DEFAULT,
    run_twin_estimation_loop,
)
from .mechanistic import (
    BioreactorParams,
    OperatingInputs,
    STATE_NAMES,
    bioreactor_rhs,
    compute_hydrodynamics,
    compute_rates,
    default_initial_state,
    integrate_step,
    nominal_kla,
    simulate_bioreactor,
    true_acetate_rates,
    true_dynamic_death_rate,
    true_kla_viscosity_factor,
    vessel_geometry,
)
from .neural_ode import (
    PhysicsInformedNeuralODE,
    generate_training_and_test_batches,
    recover_mechanistic_parameters,
    train_default_hybrid_model,
)

__all__ = [
    "BioreactorParams",
    "OperatingInputs",
    "STATE_NAMES",
    "STATE_NORM_SCALES",
    "Q_DEFAULT",
    "bioreactor_rhs",
    "compute_hydrodynamics",
    "compute_rates",
    "default_initial_state",
    "integrate_step",
    "nominal_kla",
    "simulate_bioreactor",
    "true_acetate_rates",
    "true_dynamic_death_rate",
    "true_kla_viscosity_factor",
    "vessel_geometry",
    "PhysicsInformedNeuralODE",
    "generate_training_and_test_batches",
    "recover_mechanistic_parameters",
    "train_default_hybrid_model",
    "HybridExtendedKalmanFilter",
    "BioreactorDigitalTwin",
    "run_twin_estimation_loop",
    "ClosedLoopFeedController",
    "NominalMechanisticController",
    "OpenLoopExponentialController",
    "ManualFeedController",
    "benchmark_controllers",
    "summarize_control_run",
]
