"""Pytree-invariant tests for SpectralPlanePhysicsState +
SpectralPlanePhysicsTendencies (iter-264).

iter-241 cherry-picked these NamedTuple classes from unmerged
feature/crm-plane-spectral branch (commit edbae138) into
``src/legoesm/core/state.py``. They are exercised indirectly by
``test_spectral_plane_dycore.py`` via the factory + by
``test_run_rcemip_long_cross_grid_smoke.py`` via the subprocess
driver.

iter-264 adds explicit pytree invariants so a future regression
that:
- accidentally converts the NamedTuple to a dataclass (breaks
  JAX's auto-pytree registration);
- adds a non-pytree field that breaks tree_flatten;
- drops a field;
- reorders fields;
fires loudly at the unit level rather than surfacing as a cryptic
JIT/AD error deep in the spectral wrapper.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm.core.field import Field
from legoesm.core.state import (
    SpectralPlanePhysicsState,
    SpectralPlanePhysicsTendencies,
)


def _make_dummy_field(shape, dtype, name, dims, units):
    return Field(jnp.zeros(shape, dtype=dtype), name=name,
                 dims=dims, units=units)


def _build_dummy_state(ny=4, nx=4, nlev=3, n_tracers=2):
    """rfft2 of a face/cell field has shape (ny, nx//2+1) complex.
    Tendencies share the same pytree shape."""
    nx_r = nx // 2 + 1
    cplx = jnp.complex128
    shp_full = (ny, nx_r, nlev)
    shp_half = (ny, nx_r, nlev + 1)
    shp_tracers = (ny, nx_r, nlev, n_tracers)
    return SpectralPlanePhysicsState(
        u_hat=_make_dummy_field(
            shp_full, cplx, "u_hat", ("ny", "nx_r", "nlev"), "m/s",
        ),
        v_hat=_make_dummy_field(
            shp_full, cplx, "v_hat", ("ny", "nx_r", "nlev"), "m/s",
        ),
        w_hat=_make_dummy_field(
            shp_half, cplx, "w_hat", ("ny", "nx_r", "nlev_half"), "m/s",
        ),
        theta_prime_hat=_make_dummy_field(
            shp_full, cplx, "theta_prime_hat",
            ("ny", "nx_r", "nlev"), "K",
        ),
        rho_prime_hat=_make_dummy_field(
            shp_full, cplx, "rho_prime_hat",
            ("ny", "nx_r", "nlev"), "kg/m^3",
        ),
        # phis is REAL (physical-space static), not spectral.
        phis=_make_dummy_field(
            (ny, nx), jnp.float64, "phis", ("ny", "nx"), "m^2/s^2",
        ),
        tracers_hat=_make_dummy_field(
            shp_tracers, cplx, "tracers_hat",
            ("ny", "nx_r", "nlev", "tracer"), "kg/kg",
        ),
    )


def _build_dummy_tendencies(ny=4, nx=4, nlev=3, n_tracers=2):
    nx_r = nx // 2 + 1
    cplx = jnp.complex128
    shp_full = (ny, nx_r, nlev)
    shp_half = (ny, nx_r, nlev + 1)
    shp_tracers = (ny, nx_r, nlev, n_tracers)
    return SpectralPlanePhysicsTendencies(
        du_hat_dt=_make_dummy_field(
            shp_full, cplx, "du_hat_dt", ("ny", "nx_r", "nlev"),
            "m/s^2",
        ),
        dv_hat_dt=_make_dummy_field(
            shp_full, cplx, "dv_hat_dt", ("ny", "nx_r", "nlev"),
            "m/s^2",
        ),
        dw_hat_dt=_make_dummy_field(
            shp_half, cplx, "dw_hat_dt",
            ("ny", "nx_r", "nlev_half"), "m/s^2",
        ),
        dtheta_prime_hat_dt=_make_dummy_field(
            shp_full, cplx, "dtheta_prime_hat_dt",
            ("ny", "nx_r", "nlev"), "K/s",
        ),
        drho_prime_hat_dt=_make_dummy_field(
            shp_full, cplx, "drho_prime_hat_dt",
            ("ny", "nx_r", "nlev"), "kg/m^3/s",
        ),
        dphis_dt=_make_dummy_field(
            (ny, nx), jnp.float64, "dphis_dt", ("ny", "nx"),
            "m^2/s^3",
        ),
        dtracers_hat_dt=_make_dummy_field(
            shp_tracers, cplx, "dtracers_hat_dt",
            ("ny", "nx_r", "nlev", "tracer"), "kg/kg/s",
        ),
    )


def test_spectral_state_is_namedtuple_pytree():
    """SpectralPlanePhysicsState must be a NamedTuple + auto-register
    as a JAX pytree. tree_flatten/tree_unflatten round-trip MUST
    preserve identity."""
    state = _build_dummy_state()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, SpectralPlanePhysicsState)
    # Field-by-field identity through round-trip.
    for fld in SpectralPlanePhysicsState._fields:
        orig = getattr(state, fld).data
        rec = getattr(restored, fld).data
        assert orig.shape == rec.shape, (
            f"field {fld!r} shape changed: {orig.shape} → {rec.shape}"
        )
        assert orig.dtype == rec.dtype, (
            f"field {fld!r} dtype changed: {orig.dtype} → {rec.dtype}"
        )


def test_spectral_tendencies_is_namedtuple_pytree():
    """Mirror check for SpectralPlanePhysicsTendencies."""
    tend = _build_dummy_tendencies()
    leaves, treedef = jax.tree_util.tree_flatten(tend)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, SpectralPlanePhysicsTendencies)
    for fld in SpectralPlanePhysicsTendencies._fields:
        orig = getattr(tend, fld).data
        rec = getattr(restored, fld).data
        assert orig.shape == rec.shape
        assert orig.dtype == rec.dtype


def test_spectral_state_tree_map_preserves_shape():
    """jax.tree_util.tree_map across the state MUST visit every
    Field's .data array + produce a new state with the same pytree
    structure. Tests the SSP-RK3 averaging contract relies on
    (state_n+1 = tree_map(lambda x, y: 0.5*x + 0.5*y, state_a, state_b)).
    """
    state = _build_dummy_state()
    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, state)
    assert isinstance(doubled, SpectralPlanePhysicsState)
    for fld in SpectralPlanePhysicsState._fields:
        orig = getattr(state, fld).data
        new = getattr(doubled, fld).data
        # All-zero input doubled is still all-zero; the contract is
        # the SHAPE/dtype invariance + the visit cardinality.
        assert orig.shape == new.shape
        assert orig.dtype == new.dtype


_EXPECTED_STATE_FIELDS = (
    "u_hat", "v_hat", "w_hat", "theta_prime_hat", "rho_prime_hat",
    "phis", "tracers_hat",
)
_EXPECTED_TENDENCIES_FIELDS = (
    "du_hat_dt", "dv_hat_dt", "dw_hat_dt", "dtheta_prime_hat_dt",
    "drho_prime_hat_dt", "dphis_dt", "dtracers_hat_dt",
)


def test_spectral_state_field_order_locked():
    """iter-241 cherry-pick contract: field NAMES + ORDER MUST
    match the edbae138 original. A field reorder breaks pytree
    operations that rely on positional unpacking (e.g. SSP-RK3
    stage averaging).
    """
    assert SpectralPlanePhysicsState._fields == _EXPECTED_STATE_FIELDS, (
        f"SpectralPlanePhysicsState fields = {SpectralPlanePhysicsState._fields}, "
        f"expected {_EXPECTED_STATE_FIELDS}"
    )


def test_spectral_tendencies_field_order_locked():
    """Mirror check for tendencies. Tendency field names follow
    the ``d<field>_dt`` convention with one entry per state field."""
    assert SpectralPlanePhysicsTendencies._fields == _EXPECTED_TENDENCIES_FIELDS, (
        f"SpectralPlanePhysicsTendencies fields = {SpectralPlanePhysicsTendencies._fields}, "
        f"expected {_EXPECTED_TENDENCIES_FIELDS}"
    )


def test_spectral_state_and_tendencies_field_count_matches():
    """Tendencies + state must have the SAME pytree shape (one
    tendency per state field) so jax.tree_util.tree_map(state, tend)
    works. The names differ (state.u_hat vs tend.du_hat_dt) but
    the count + iteration order must align."""
    assert len(SpectralPlanePhysicsState._fields) == \
           len(SpectralPlanePhysicsTendencies._fields), (
        f"state has {len(SpectralPlanePhysicsState._fields)} fields, "
        f"tendencies has {len(SpectralPlanePhysicsTendencies._fields)}"
    )
