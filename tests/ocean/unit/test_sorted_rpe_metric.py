"""Discrimination tests for the sorted-RPE (RPE_mov) spurious-mixing metric.

The metric is ``run_ocean_test_matrix._compute_sorted_rpe`` -- the number the
lock-exchange matrix gate ``RPE_rel_mixing_sign`` PASSes/FAILs on. Each test
here fails for a specific broken implementation:

- ``test_homogenising_layers_raises_sorted_rpe``: a densest-at-SURFACE
  packing (the sign inversion this metric replaced) makes homogenisation
  LOWER the value; a constant-returning implementation returns 0 change.
  Level-swap invariance alone cannot catch either (codex 2026-08-09).
- ``test_rearrangement_invariance``: sorted RPE depends only on the
  (rho, vol) multiset, so an adiabatic rearrangement must not move it. The
  rearrangement is a LONGITUDE roll: cell area varies with latitude only and
  the z levels are stretched, so a lon roll is the volume-preserving
  permutation (a level swap is NOT -- it re-pairs rho with different dz).
- ``test_moving_volumes_enter_the_metric``: RPE_mov weights parcels with the
  MOVING z-star volume area*h(eta); a fixed-reference-volume (dz_ref)
  implementation is invariant to eta and fails this test. Moving volumes are
  the 2026-08-10 retraction: on the two bounded-front arms (FESOM FCT,
  tripole implicit_cn) fixed volumes gave a NEGATIVE drift while moving
  volumes gave the physical positive sign.
"""

from __future__ import annotations

import importlib.util
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

_REPO = Path(__file__).resolve().parents[3]


@pytest.fixture(scope="module")
def matrix_mod():
    path = _REPO / "scripts" / "matrix" / "run_ocean_test_matrix.py"
    sys.path.insert(0, str(_REPO / "scripts" / "matrix"))
    spec = importlib.util.spec_from_file_location("_rm_rpe_test", path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules["_rm_rpe_test"] = mod
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def latlon_setup(matrix_mod):
    m = matrix_mod
    tc = SimpleNamespace(grid_type="latlon", resolution="8x16",
                         case="lock_exchange", run_kwargs={})
    grid, z_coord, _cfg, _model, _ck, _lon, _lat = m._create_ocean_setup(
        tc, nlev=6, H_max=20.0)
    state = m._create_rest_state(tc, grid, z_coord, H_max=20.0)
    return m, tc, grid, z_coord, state


def _with_stratified_T(state):
    """Stable stratification: T decreases with depth, uniform horizontally."""
    import jax.numpy as jnp
    T = np.asarray(state.T.data)
    prof = 20.0 - 2.0 * np.arange(T.shape[-1])
    T_new = np.broadcast_to(prof, T.shape).astype(T.dtype)
    return state._replace(T=state.T.replace(data=jnp.asarray(T_new)))


def test_homogenising_layers_raises_sorted_rpe(latlon_setup):
    import jax.numpy as jnp
    m, tc, grid, z_coord, state = latlon_setup
    st = _with_stratified_T(state)
    rpe0 = m._compute_sorted_rpe(st, tc.grid_type, grid, z_coord)

    T = np.asarray(st.T.data).copy()
    # VOLUME-weighted homogenisation (levels are stretched, so an arithmetic
    # mean would not be a pure mixing event -- codex 2026-08-10).
    w = np.asarray(z_coord.dz_ref)[2:4]
    T[..., 2:4] = np.sum(T[..., 2:4] * w, axis=-1, keepdims=True) / w.sum()
    st_mixed = st._replace(T=st.T.replace(data=jnp.asarray(T)))
    rpe1 = m._compute_sorted_rpe(st_mixed, tc.grid_type, grid, z_coord)

    assert rpe1 > rpe0, (
        f"homogenising two stably-stratified layers must RAISE sorted RPE "
        f"(mixing sign); got {rpe1 - rpe0:.3e}")
    assert (rpe1 - rpe0) / abs(rpe0) > 1e-10


def test_rearrangement_invariance(latlon_setup):
    import jax.numpy as jnp
    m, tc, grid, z_coord, state = latlon_setup
    st = _with_stratified_T(state)
    # Lon-dependent front so the roll below is a non-trivial rearrangement.
    T = np.asarray(st.T.data).copy()
    n_lon = T.shape[1]
    T += 5.0 * (np.arange(n_lon) < n_lon // 2)[np.newaxis, :, np.newaxis]
    st = st._replace(T=st.T.replace(data=jnp.asarray(T)))
    rpe0 = m._compute_sorted_rpe(st, tc.grid_type, grid, z_coord)

    # Roll along LONGITUDE: area varies with latitude only, dz with level
    # only, so this permutes parcels among equal-volume slots -- the
    # (rho, vol) multiset is unchanged and sorted RPE must not move.
    st_roll = st._replace(T=st.T.replace(
        data=jnp.asarray(np.roll(T, 3, axis=1))))
    rpe_roll = m._compute_sorted_rpe(st_roll, tc.grid_type, grid, z_coord)

    assert abs(rpe_roll - rpe0) / abs(rpe0) < 1e-12


def test_pack_sorted_rpe_kernel_discrimination():
    """Production kernel (legoesm.ocean.rpe.pack_sorted_rpe): mixing must
    RAISE the value. A densest-at-SURFACE packing (the pre-2026-08-10
    compute_rpe inline block) makes mixing LOWER it and fails here."""
    from legoesm.ocean.rpe import pack_sorted_rpe
    rho = np.array([1027.0, 1022.0])
    vol = np.array([50.0, 50.0])
    r0 = pack_sorted_rpe(rho, vol, total_area=10.0)
    rho_mixed = np.array([1024.5, 1024.5])
    r1 = pack_sorted_rpe(rho_mixed, vol, total_area=10.0)
    assert r1 > r0
    # Two-layer analytic check: dRPE = g*V1*V2*(z_top-z_bot)*(rho1-rho2)/(V1+V2)
    from legoesm import constants
    expected = constants.g * (50.0 * 50.0 / 100.0) * 5.0 * 5.0
    assert abs((r1 - r0) - expected) / expected < 1e-12


def test_moving_volumes_enter_the_metric(latlon_setup):
    import jax.numpy as jnp
    m, tc, grid, z_coord, state = latlon_setup
    st = _with_stratified_T(state)
    # T front in lon + eta bump CORRELATED with it: the warm parcels gain
    # volume, the cold ones lose it, so the (rho, vol) multiset -- and hence
    # RPE_mov -- must change. (An eta bump over horizontally-uniform T is
    # degenerate: equal-and-opposite volume changes within each density
    # class cancel and even the moving-volume metric legitimately stays put.)
    T = np.asarray(st.T.data).copy()
    n_lon = T.shape[1]
    warm = (np.arange(n_lon) < n_lon // 2)
    T += 5.0 * warm[np.newaxis, :, np.newaxis]
    st = st._replace(T=st.T.replace(data=jnp.asarray(T)))
    rpe0 = m._compute_sorted_rpe(st, tc.grid_type, grid, z_coord)

    eta = np.asarray(st.eta.data)
    bump = 0.5 * np.where(warm, 1.0, -1.0)[np.newaxis, :]
    st_eta = st._replace(eta=st.eta.replace(
        data=jnp.asarray(np.broadcast_to(bump, eta.shape).astype(eta.dtype))))
    rpe_eta = m._compute_sorted_rpe(st_eta, tc.grid_type, grid, z_coord)

    assert abs(rpe_eta - rpe0) / abs(rpe0) > 1e-8, (
        "an eta perturbation must move RPE_mov (moving z-star volumes); "
        "invariance means the implementation regressed to fixed dz_ref "
        "volumes")
