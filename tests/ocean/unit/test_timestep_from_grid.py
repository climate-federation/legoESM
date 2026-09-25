"""Tests for the grid-derived timestep selector.

Assertions specified by GLM alongside the implementation; written here by
Claude as reviewer. The golden case is the measurement that motivated the
whole change: on the tripole mesh the minimum spacing is 1 km but those cells
are dry, the smallest wet cell is 23.3 km, and the rule then returns a
timestep essentially equal to NEMO ORCA1's own rn_Dt=3600.

These tests check the SELECTOR'S ARITHMETIC, not production stability. Codex's
review established that this Courant bound is necessary but not sufficient
here -- tracers are still explicit, and the dry-cell exclusion is unsafe where
sea ice advects onto nominally dry cells -- so nothing below licenses running
at the returned timestep. That takes an integration.
"""
from __future__ import annotations

import numpy as np
import pytest

from legoesm.ocean.timestep import derive_timestep


def test_golden_case_reproduces_the_measured_tripole_timestep():
    """The measured wet minimum of 23.26 km at C=0.6 and V=4 m/s gives ~3490 s,
    which is the oracle's timestep. This is the number behind the finding that
    the 150 s literal is what the same rule returns when dry cells are left
    in -- an arithmetic fixture, not a stability claim."""
    dx = np.full((4, 4), 80.0e3)
    dx[2, 2] = 23253.0
    wet = np.ones_like(dx, dtype=bool)
    dt, prov = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                               u_max=0.0)
    assert dt == pytest.approx(3487.95, rel=1e-4)
    assert prov["binding_constraint"] == "courant"
    assert prov["courant_resulting"] == pytest.approx(0.6, rel=1e-12)
    assert prov["dx_min_index"] == (2, 2)


def test_dry_cells_do_not_set_the_timestep():
    """THE FINDING. A 1 km cell and a NaN cell, both dry, must be ignored; the
    wet minimum sets dt. This is the test that would have caught the 150 s."""
    dx = np.full((3, 3), 80.0e3)
    dx[0, 0] = 1.0e3          # the polar-convergence cell: dry
    dx[0, 1] = np.nan         # dry, and not even finite
    dx[1, 1] = 23.3e3         # smallest WET cell
    wet = np.ones_like(dx, dtype=bool)
    wet[0, 0] = False
    wet[0, 1] = False
    dt, prov = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                               u_max=0.0)
    assert prov["dx_min"] == pytest.approx(23.3e3)
    assert prov["dx_min_index"] == (1, 1)
    assert dt == pytest.approx(0.6 * 23.3e3 / 4.0)
    # and the dry 1 km cell would have cost a factor of 23
    assert dt / (0.6 * 1.0e3 / 4.0) == pytest.approx(23.3, rel=1e-9)


def test_all_dry_mask_raises():
    dx = np.full((3, 3), 80.0e3)
    with pytest.raises(ValueError, match="no cells"):
        derive_timestep(dx, np.zeros_like(dx, dtype=bool), c_courant=0.6,
                        c1_baroclinic=4.0, u_max=0.0)


def test_non_finite_dx_at_a_wet_cell_raises_but_is_tolerated_when_dry():
    dx = np.full((3, 3), 80.0e3)
    dx[1, 1] = np.nan
    wet = np.ones_like(dx, dtype=bool)
    with pytest.raises(ValueError, match="non-finite"):
        derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0, u_max=0.0)
    wet[1, 1] = False
    dt, _ = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                            u_max=0.0)
    assert np.isfinite(dt)


def test_non_positive_dx_at_a_wet_cell_raises():
    dx = np.full((3, 3), 80.0e3)
    dx[0, 2] = 0.0
    with pytest.raises(ValueError, match="non-positive"):
        derive_timestep(dx, np.ones_like(dx, dtype=bool), c_courant=0.6,
                        c1_baroclinic=4.0, u_max=0.0)


def test_cap_binds_and_is_reported():
    dx = np.full((2, 2), 80.0e3)
    wet = np.ones_like(dx, dtype=bool)
    dt, prov = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                               u_max=0.0, dt_cap=600.0)
    assert dt == 600.0
    assert prov["binding_constraint"] == "dt_cap"
    assert prov["courant_resulting"] < 0.6
    # a cap above the CFL bound must not bind
    dt2, prov2 = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                                 u_max=0.0, dt_cap=1.0e6)
    assert prov2["binding_constraint"] == "courant"
    assert dt2 == pytest.approx(prov2["dt_courant"])


def test_ties_are_counted_and_the_index_is_correct():
    dx = np.full((2, 3), 80.0e3)
    dx[0, 1] = 30.0e3
    dx[1, 2] = 30.0e3
    wet = np.ones_like(dx, dtype=bool)
    _, prov = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                              u_max=0.0)
    assert prov["n_binding_cells"] == 2
    assert dx[prov["dx_min_index"]] == prov["dx_min"]


def test_matches_a_vectorised_recomputation_on_random_meshes():
    """Property: the returned dt is the minimum of the per-cell CFL bound."""
    rng = np.random.default_rng(20260911)
    for _ in range(20):
        dx = rng.uniform(5.0e3, 120.0e3, size=(7, 11))
        wet = rng.random(dx.shape) > 0.3
        if not wet.any():
            continue
        dt, _ = derive_timestep(dx, wet, c_courant=0.55, c1_baroclinic=3.0,
                                u_max=1.0)
        expect = np.min(0.55 * dx[wet] / 4.0)
        assert dt == pytest.approx(expect, rel=1e-12)


def test_invalid_coefficients_and_shapes_raise():
    dx = np.full((2, 2), 80.0e3)
    wet = np.ones_like(dx, dtype=bool)
    with pytest.raises(ValueError, match="shape mismatch"):
        derive_timestep(dx, np.ones((3, 3), dtype=bool), c_courant=0.6,
                        c1_baroclinic=4.0, u_max=0.0)
    with pytest.raises(ValueError, match="c_courant"):
        derive_timestep(dx, wet, c_courant=0.0, c1_baroclinic=4.0, u_max=0.0)
    with pytest.raises(ValueError, match=">= 0"):
        derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=-1.0, u_max=0.0)
    with pytest.raises(ValueError, match="must be > 0"):
        derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=0.0, u_max=0.0)


def test_it_is_a_setup_time_helper_and_refuses_to_be_traced():
    """Tracing must fail loudly rather than silently producing a traced dt."""
    import jax
    dx = np.full((2, 2), 80.0e3)
    wet = np.ones_like(dx, dtype=bool)
    with pytest.raises(Exception):
        jax.jit(lambda d: derive_timestep(d, wet, c_courant=0.6,
                                          c1_baroclinic=4.0, u_max=0.0)[0])(dx)


def test_provenance_is_loggable():
    import json
    dx = np.full((2, 2), 80.0e3)
    dt, prov = derive_timestep(dx, np.ones_like(dx, dtype=bool),
                               c_courant=0.6, c1_baroclinic=4.0, u_max=0.0)
    assert isinstance(dt, float)
    json.dumps({k: (list(v) if isinstance(v, tuple) else v)
                for k, v in prov.items()})


# --- codex review 9705616, defects it found in the selector -----------------

def test_non_finite_coefficients_raise():
    """codex [MEDIUM]: non-finite coefficients were accepted silently."""
    dx = np.full((2, 2), 80.0e3)
    wet = np.ones_like(dx, dtype=bool)
    for kw in ({"c_courant": np.nan}, {"c1_baroclinic": np.inf},
               {"u_max": np.nan}):
        base = {"c_courant": 0.6, "c1_baroclinic": 4.0, "u_max": 0.0}
        base.update(kw)
        with pytest.raises(ValueError, match="finite"):
            derive_timestep(dx, wet, **base)


def test_bad_caps_raise_instead_of_falling_through():
    """codex [MEDIUM]: a NaN cap fell through the comparison and silently left
    the Courant value in place; a non-positive cap was accepted."""
    dx = np.full((2, 2), 80.0e3)
    wet = np.ones_like(dx, dtype=bool)
    for cap in (np.nan, 0.0, -10.0, np.inf):
        with pytest.raises(ValueError, match="dt_cap"):
            derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                            u_max=0.0, dt_cap=cap)


def test_binding_cell_count_is_none_when_the_cap_binds():
    """codex [MEDIUM]: n_binding_cells reported spacing minima even when the
    cap was what bound, which is a provenance value inconsistent with dt."""
    dx = np.full((2, 2), 80.0e3)
    wet = np.ones_like(dx, dtype=bool)
    _, capped = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                                u_max=0.0, dt_cap=600.0)
    assert capped["binding_constraint"] == "dt_cap"
    assert capped["n_binding_cells"] is None
    assert capped["n_dx_min_cells"] == 4
    _, free = derive_timestep(dx, wet, c_courant=0.6, c1_baroclinic=4.0,
                              u_max=0.0)
    assert free["n_binding_cells"] == 4
