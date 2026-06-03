"""The legoESM ``Experiment`` contract + in-repo characterization rungs.

An :class:`~legoesm.experiments.abstract.Experiment` is a *reproducible,
self-describing* test case: it runs, produces named metrics, and checks each
against a reference within a tolerance.  This is the executable form of the §6
verification matrices — each matrix cell is one ``Experiment`` — and the seam
that lets "run Williamson" or "gradient-check this column" be a single call with
a pass/fail verdict.

Only *characterization* rungs live here (the in-repo half of the matrices, see
master plan §7.1); scientific *campaigns* live in external runner repos.
"""

from legoesm.experiments.abstract import (  # noqa: F401
    Experiment,
    ExperimentResult,
    MetricCheck,
)
from legoesm.experiments.gradient_check import (  # noqa: F401
    GradientCheckExperiment,
    finite_difference_grad,
    relative_grad_error,
    single_column_thermo_gradient_check,
)

__all__ = [
    "Experiment",
    "ExperimentResult",
    "MetricCheck",
    "GradientCheckExperiment",
    "finite_difference_grad",
    "relative_grad_error",
    "single_column_thermo_gradient_check",
]
