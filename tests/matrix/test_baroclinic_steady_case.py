"""The `baroclinic_steady` matrix case — the #1028 unperturbed control.

`baroclinic` (Jablonowski-Williamson) carries a localized 1 m/s perturbation
that grows into a baroclinic wave.  `baroclinic_steady` is the SAME balanced
initial state with that perturbation switched off, so the pair differs in
exactly one thing and any jet decay in the steady arm is the dycore failing to
hold the base state rather than the wave consuming it.

The tests below exist because the control is worthless if `perturbed=False`
does not actually reach the initial condition — a control that perturbs
nothing proves nothing.  They therefore assert (a) the catalog registers the
case for every grid at sigma, (b) the runner dispatch resolves it, and (c) the
two initial conditions REALLY differ, and only in the perturbation.
"""

from __future__ import annotations

import importlib.util
import pathlib
import sys

import numpy as np
import pytest

_SCRIPT = (pathlib.Path(__file__).resolve().parents[2]
           / "scripts" / "matrix" / "run_atmosphere_test_matrix.py")


@pytest.fixture(scope="module")
def matrix_mod():
    spec = importlib.util.spec_from_file_location("_atmmatrix", _SCRIPT)
    mod = importlib.util.module_from_spec(spec)
    # Register BEFORE exec: the script defines @dataclass types, and
    # dataclasses resolves annotations through ``sys.modules[cls.__module__]``,
    # which is None for a module that was never registered.
    sys.modules[spec.name] = mod
    try:
        spec.loader.exec_module(mod)
    except Exception:                                    # pragma: no cover
        sys.modules.pop(spec.name, None)
        raise
    yield mod
    sys.modules.pop(spec.name, None)


def _steady_cases(matrix_mod):
    return [t for t in matrix_mod._build_test_matrix()
            if t.case == "baroclinic_steady"]


def test_catalog_registers_the_control_on_every_grid(matrix_mod):
    cases = _steady_cases(matrix_mod)
    grids = {t.grid_type for t in cases}
    assert grids == set(matrix_mod.GRID_TYPES), grids
    for t in cases:
        # Sigma only: the hybrid coordinate carries negative layer mass below
        # p_s ~ 664 hPa, and `rotated_steady` (the pre-existing unperturbed
        # case) is pinned to it and to alpha=45 deg, which is why it could not
        # serve as this control.
        assert t.vertical_coord == "sigma"
        assert t.run_kwargs.get("perturbed") is False
        assert t.duration_days == 10.0


def test_the_control_pairs_one_to_one_with_the_perturbed_case(matrix_mod):
    """Same grid AND resolution as the sigma `baroclinic` arm it controls."""
    pert = {(t.grid_type, t.resolution) for t in matrix_mod._build_test_matrix()
            if t.case == "baroclinic" and t.vertical_coord == "sigma"}
    steady = {(t.grid_type, t.resolution) for t in _steady_cases(matrix_mod)}
    assert steady == pert


def test_runner_dispatch_resolves_the_case(matrix_mod):
    runners = matrix_mod.RUNNERS
    assert "baroclinic_steady" in runners
    assert runners["baroclinic_steady"] is runners["baroclinic"]


def test_exact_case_selector_matches_it(matrix_mod):
    """`--test =baroclinic_steady` must select something, not silently nothing."""
    exact = [t for t in matrix_mod._build_test_matrix()
             if t.case == "baroclinic_steady"]
    assert len(exact) == len(matrix_mod.GRID_TYPES)


def test_perturbed_false_really_changes_the_initial_condition():
    """The control must not be a no-op: assert the bump is present in one IC,
    absent in the other, and that the steady state is zonally symmetric."""
    from legoesm.grids.latlon import create_latlon_grid
    from legoesm.grids.vertical import create_sigma_coordinate
    from tests.test_cases.baroclinic_wave import baroclinic_wave_init_latlon

    grid = create_latlon_grid(36, 72)
    sigma = create_sigma_coordinate(10)
    pert = np.asarray(
        baroclinic_wave_init_latlon(grid, sigma, perturbed=True).u.data)
    steady = np.asarray(
        baroclinic_wave_init_latlon(grid, sigma, perturbed=False).u.data)

    du = np.abs(pert - steady)
    # The J-W perturbation is a LOCALIZED ~1 m/s bump, not a global offset.
    assert du.max() > 0.5, du.max()
    assert 0 < (du > 1e-6).sum() < 0.05 * du.size

    # ... and the unperturbed state is the zonal base state (float32 noise).
    zonal_dev = np.abs(steady - steady.mean(axis=1, keepdims=True)).max()
    assert zonal_dev < 1.0e-3, zonal_dev
    assert np.abs(pert - pert.mean(axis=1, keepdims=True)).max() > 0.1
