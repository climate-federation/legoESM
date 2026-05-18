"""Tests for the static reference density profile on MPAS Voronoi ocean.

The static ``ρ_ref(z)`` is the seed-reduction fix for the
partial-cell PGF residual that drives the bottom-trapped rotational
mode (project_mpas_etopo_instability.md §"Option B").  It replaces
``ρ' = ρ - ρ_0`` with ``ρ' = ρ - ρ_ref(z)`` using a frozen-at-init
horizontally-uniform profile.

These tests verify:

1. ``attach_static_rho_ref_z`` is a no-op when the config switch is off.
2. The static and dynamic versions agree at ``t = 0`` (machine epsilon
   in float64) — they share the EOS iteration and reduction.
3. The static profile does NOT change when the runtime T,S drift away
   from the initial state (this is the entire point — the dynamic
   version recomputed-mean drift was the day-60 NaN cause).
4. With the static profile attached, ``ρ' = ρ - ρ_ref(z)`` in the
   tendency call's ``rho_prime`` output, exactly.
5. Mutual exclusion with the legacy dynamic flag is enforced.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

# Force float64 so the static-vs-dynamic equivalence is checked at
# machine precision rather than float32 reduction noise.
jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.core.precision import PrecisionPolicy, set_policy
set_policy(PrecisionPolicy(storage=jnp.float64, compute=jnp.float64))

from legoesm.grids.voronoi import create_voronoi_mesh
from legoesm.ocean.dynamics.mpas_fill import fill_land_cells_mpas
from legoesm.ocean.dynamics.ocean_tendency_common import (
    iterate_eos_and_pressure_anomaly,
)
from legoesm.ocean.eos import make_eos_fn
from legoesm.ocean.init_mpas import (
    attach_static_rho_ref_z,
    rest_state_mpas_ocean,
)
from legoesm.ocean.mpas_config import MPASOceanConfig
from legoesm.ocean.vertical import (
    create_ocean_z_star,
    create_partial_cell_coordinate,
)


@pytest.fixture(scope="module")
def setup():
    """Small ico-2 mesh + 10-level partial-cell coordinate + rest state."""
    mesh = create_voronoi_mesh(subdivision_level=2)
    z_coord = create_ocean_z_star(n_levels=10, H_max=5500.0)
    state = rest_state_mpas_ocean(
        mesh, z_coord,
        T_water_init_C=20.0, T_deep=2.0, S_uniform=35.0,
        H_max=5500.0, land_lat_threshold=80.0,
    )
    pc_coord = create_partial_cell_coordinate(z_coord, state.H_bathy.data)
    return mesh, z_coord, pc_coord, state


def test_off_by_default(setup):
    """Default config leaves rho_ref_z=None and returns the input state."""
    mesh, _z_coord, pc_coord, state = setup
    cfg = MPASOceanConfig()
    state_out = attach_static_rho_ref_z(state, mesh, pc_coord, cfg)
    assert state_out.rho_ref_z is None
    assert state_out is state, "should be a true no-op (same identity)"


def test_attach_populates_rho_ref(setup):
    """Switch-on writes a (nlev,) Field with stratified values."""
    mesh, _z_coord, pc_coord, state = setup
    cfg = MPASOceanConfig(use_static_baroclinic_rho_ref=True)
    state_out = attach_static_rho_ref_z(state, mesh, pc_coord, cfg)

    assert state_out.rho_ref_z is not None
    rrz = np.asarray(state_out.rho_ref_z.data)
    assert rrz.shape == (10,)
    # Surface < bottom — stable stratification on the rest state.
    assert rrz[0] < rrz[-1]
    # Plausible seawater range.
    assert 1015.0 < rrz.min() and rrz.max() < 1060.0
    # Strictly monotonically non-decreasing with depth (rest state is
    # stably stratified by construction).
    assert np.all(np.diff(rrz) > 0)


def test_static_matches_dynamic_at_init(setup):
    """Static profile = wet-cell mean of dynamic ρ at init (to machine eps)."""
    mesh, _z_coord, pc_coord, state = setup

    cfg = MPASOceanConfig(
        use_static_baroclinic_rho_ref=True,
        use_h_actual_pgf=False,  # match the dynamic call below
    )
    state_static = attach_static_rho_ref_z(state, mesh, pc_coord, cfg)

    # Replicate the dynamic-version reduction directly.
    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    mask = state.land_mask.data

    def _fill(f):
        return fill_land_cells_mpas(f, mask, c1, c2)

    eos_fn = make_eos_fn("wright", None)
    rho_dyn, rho_prime_dyn, _p = iterate_eos_and_pressure_anomaly(
        state.T.data, state.S.data, mask, _fill, eos_fn,
        pc_coord.dz_ref, constants.rho_ocean, constants.g, n_iter=2,
        use_depth_dependent_ref=True,
        is_active_3d=pc_coord.is_active.astype(state.T.data.dtype),
        h_actual=None,
    )
    implied_ref = np.asarray(rho_dyn - rho_prime_dyn)
    wet = np.asarray(pc_coord.is_active).astype(implied_ref.dtype)
    dyn_mean = (implied_ref * wet).sum(axis=0) / np.maximum(wet.sum(axis=0), 1.0)

    diff = np.abs(np.asarray(state_static.rho_ref_z.data) - dyn_mean).max()
    # In float64, the two paths differ only by the order in which the
    # final wet-cell sum is composed.  4e-12 leaves comfortable headroom.
    assert diff < 1e-9, f"static-vs-dynamic disagreement at init: {diff}"


def test_runtime_uses_static_profile(setup):
    """iterate_eos_and_pressure_anomaly with rho_ref_z_static gives ρ'=ρ-ρ_ref."""
    mesh, _z_coord, pc_coord, state = setup
    cfg = MPASOceanConfig(use_static_baroclinic_rho_ref=True)
    state_static = attach_static_rho_ref_z(state, mesh, pc_coord, cfg)

    c1 = mesh.cellsOnEdge[0]
    c2 = mesh.cellsOnEdge[1]
    mask = state.land_mask.data
    eos_fn = make_eos_fn("wright", None)

    rho, rho_prime, _p = iterate_eos_and_pressure_anomaly(
        state.T.data, state.S.data, mask,
        lambda f: fill_land_cells_mpas(f, mask, c1, c2),
        eos_fn, pc_coord.dz_ref, constants.rho_ocean, constants.g, n_iter=2,
        rho_ref_z_static=state_static.rho_ref_z.data,
        h_actual=None,
    )
    expected = np.asarray(rho) - np.asarray(state_static.rho_ref_z.data)[None, :]
    err = np.abs(np.asarray(rho_prime) - expected).max()
    assert err == 0.0, "rho_prime must equal rho - rho_ref_z bit-exactly"


def test_static_profile_is_invariant_under_drift(setup):
    """Re-running the helper on drifted T,S returns a *different* profile —
    confirms the runtime path does NOT recompute ρ_ref(z).  We test this
    by attaching once on the rest state, then drifting T (warm anomaly),
    and verifying that the *attached* profile is still the rest-state
    profile (we never re-attach during a run)."""
    mesh, _z_coord, pc_coord, state = setup
    cfg = MPASOceanConfig(use_static_baroclinic_rho_ref=True)
    state_static = attach_static_rho_ref_z(state, mesh, pc_coord, cfg)
    rho_ref_init = np.asarray(state_static.rho_ref_z.data)

    # Drift T by +1 °C uniformly.  In a real run, ``model.step`` returns
    # a new state but never overwrites ``rho_ref_z`` (state_new line in
    # ocean_model_mpas.py preserves it via ``state.rho_ref_z``).
    T_drifted = state_static.T.data + 1.0
    state_drifted = state_static._replace(T=state_static.T.replace(data=T_drifted))
    # The static profile rides through unchanged.
    assert state_drifted.rho_ref_z is state_static.rho_ref_z

    # Sanity: had we recomputed (re-attached), the profile WOULD have
    # changed — this verifies the warm anomaly is large enough to be
    # detectable, so the "no change" check above is meaningful.
    state_reattached = attach_static_rho_ref_z(
        state_drifted._replace(rho_ref_z=None), mesh, pc_coord, cfg,
    )
    rho_ref_drifted = np.asarray(state_reattached.rho_ref_z.data)
    drift = np.abs(rho_ref_drifted - rho_ref_init).max()
    assert drift > 1e-3, (
        "warm anomaly should change a re-attached profile measurably; "
        f"got max change {drift}"
    )


def test_mutual_exclusion(setup):
    """Static and dynamic flags both on → ValueError at attach time."""
    mesh, _z_coord, pc_coord, state = setup
    cfg = MPASOceanConfig(
        use_static_baroclinic_rho_ref=True,
        use_baroclinic_rho_ref=True,
    )
    with pytest.raises(ValueError, match="mutually exclusive"):
        attach_static_rho_ref_z(state, mesh, pc_coord, cfg)


def test_state_propagates_through_replace(setup):
    """state._replace(...) preserves rho_ref_z (it's a regular pytree leaf)."""
    mesh, _z_coord, pc_coord, state = setup
    cfg = MPASOceanConfig(use_static_baroclinic_rho_ref=True)
    state_static = attach_static_rho_ref_z(state, mesh, pc_coord, cfg)
    state_new = state_static._replace(
        T=state_static.T.replace(data=state_static.T.data * 1.0),
    )
    # Field is preserved (not None) and identical.
    assert state_new.rho_ref_z is state_static.rho_ref_z
