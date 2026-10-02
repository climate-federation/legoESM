"""4D-Var data assimilation module for legoESM.

Provides grid-agnostic variational data assimilation exploiting JAX's
automatic differentiation for adjoint-free 4D-Var.

Modules
-------
control_vector : Control variable <-> state transforms
background_error : B matrix (diagonal, diffusion, spectral, hybrid)
observation : Observation containers and operators
cost_function : 4D-Var cost function J = J_b + J_o
minimizer : On-device L-BFGS and CG solvers
preconditioning : Change-of-variable B^{1/2} preconditioning
incremental : Incremental 4D-Var outer/inner loop driver
cycling : Sequential analysis-forecast cycling
"""

from legoesm.da.control_vector import (
    ControlEntry,
    ControlVectorSpec,
    build_control_spec,
    control_to_increment,
    control_to_state,
    state_to_control,
)
from legoesm.da.background_error import (
    DiagonalB,
    DiffusionB,
    HybridB,
)
from legoesm.da.observation import (
    Observation,
    DiagonalR,
    DirectObsOperator,
    InterpolatingObsOperator,
    ColumnIntegralObsOperator,
    CompositeObsOperator,
    generate_synthetic_obs,
)
from legoesm.da.cost_function import (
    build_cost_fn,
    build_cost_and_grad_fn,
)
from legoesm.da.minimizer import (
    MinimizationResult,
    minimize_lbfgs,
    minimize_cg,
)
from legoesm.da.preconditioning import preconditioned_cost_fn
from legoesm.da.incremental import (
    IncrementalConfig,
    IncrementalDiagnostics,
    incremental_4dvar,
)
from legoesm.da.cycling import (
    CyclingConfig,
    run_cycling,
)
__all__ = [
    # control_vector
    "ControlEntry",
    "ControlVectorSpec",
    "build_control_spec",
    "control_to_increment",
    "control_to_state",
    "state_to_control",
    # background_error
    "DiagonalB",
    "DiffusionB",
    "HybridB",
    # observation
    "Observation",
    "DiagonalR",
    "DirectObsOperator",
    "InterpolatingObsOperator",
    "ColumnIntegralObsOperator",
    "CompositeObsOperator",
    "generate_synthetic_obs",
    # cost_function
    "build_cost_fn",
    "build_cost_and_grad_fn",
    # minimizer
    "MinimizationResult",
    "minimize_lbfgs",
    "minimize_cg",
    # preconditioning
    "preconditioned_cost_fn",
    # incremental
    "IncrementalConfig",
    "IncrementalDiagnostics",
    "incremental_4dvar",
    # cycling
    "CyclingConfig",
    "run_cycling",
]
