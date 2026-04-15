"""Ocean test matrix package — organized test runner for legoESM ocean cores.

Provides modular experiment runners, diagnostics, regridding, and CLI for
the ocean dynamical core validation suite.
"""

# Explicit __all__ so that `from ocean_test_matrix import *` re-exports
# underscore-prefixed names needed by external scripts
# (e.g., run_geoadj_fluxform.py monkey-patches _create_ocean_setup).
__all__ = [
    "GRID_RESOLUTIONS", "GRID_TYPES", "REGIONAL_GRID_TYPES",
    "DEFAULT_NLEV", "DEFAULT_DT", "DEFAULT_H_MAX",
    "FIELD_RANGES", "ALL_RESULTS", "record",
    "_A_EARTH", "_OMEGA_E", "_G_EARTH",
    "TestCase", "TEST_MATRIX",
    "check_finite", "_snapshot_steps", "_compute_drift", "_run_timeloop",
    "_create_ocean_setup", "_parse_resolution",
    "_extract_fv_ocean", "_extract_mpas_ocean", "_extract_spectral_ocean",
    "_make_check_fn", "_make_scalar_fn", "_make_extract_fn", "_key_array_fn",
    "_make_baroclinic_scalar_fn",
    "_write_results_txt", "_save_case_diagnostics",
    "_save_velocity_profiles", "_ensure_required_artifacts",
    "RUNNERS",
    "main", "build_parser", "filter_tests",
]

from ocean_test_matrix.config import (  # noqa: F401
    GRID_RESOLUTIONS, GRID_TYPES, REGIONAL_GRID_TYPES,
    DEFAULT_NLEV, DEFAULT_DT, DEFAULT_H_MAX,
    FIELD_RANGES, ALL_RESULTS, record,
    _A_EARTH, _OMEGA_E, _G_EARTH,
)
from ocean_test_matrix.testcase import TestCase, TEST_MATRIX  # noqa: F401
from ocean_test_matrix.timeloop import (  # noqa: F401
    check_finite, _snapshot_steps, _compute_drift, _run_timeloop,
)
from ocean_test_matrix.setup import _create_ocean_setup, _parse_resolution  # noqa: F401
from ocean_test_matrix.extraction import (  # noqa: F401
    _extract_fv_ocean, _extract_mpas_ocean, _extract_spectral_ocean,
    _make_check_fn, _make_scalar_fn, _make_extract_fn, _key_array_fn,
    _make_baroclinic_scalar_fn,
)
from ocean_test_matrix.diagnostic_io import (  # noqa: F401
    _write_results_txt, _save_case_diagnostics,
    _save_velocity_profiles, _ensure_required_artifacts,
)
from ocean_test_matrix.experiments import RUNNERS  # noqa: F401
from ocean_test_matrix.cli import main, build_parser, filter_tests  # noqa: F401
