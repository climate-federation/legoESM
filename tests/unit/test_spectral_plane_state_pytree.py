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
import numpy as np

from legoesm.core.field import Field
from legoesm.core.state import (
    SpectralPlanePhysicsState,
    SpectralPlanePhysicsTendencies,
)


def _make_field_with_sentinel(shape, dtype, name, dims, units, sentinel):
    """Fill the field with a UNIQUE nonzero sentinel value so a
    round-trip that swaps leaves of identical shape gets detected.

    iter-265 (Codex iter-264 round-1 HIGH#1): the pre-iter-265
    helper used jnp.zeros which made all spectral fields
    bit-identical — a bad unflatten that swapped u_hat ↔ v_hat
    passed silently because both arrays were all-zero.
    """
    if jnp.issubdtype(dtype, jnp.complexfloating):
        arr = jnp.full(shape, sentinel + 1j * sentinel, dtype=dtype)
    else:
        arr = jnp.full(shape, sentinel, dtype=dtype)
    return Field(arr, name=name, dims=dims, units=units)


def _build_dummy_state(ny=4, nx=4, nlev=3, n_tracers=2):
    """rfft2 of a face/cell field has shape (ny, nx//2+1) complex.
    Tendencies share the same pytree shape. Each field gets a
    UNIQUE sentinel value (iter-265 HIGH#1 fix) so a bad unflatten
    that swaps same-shaped leaves trips the round-trip check.
    """
    nx_r = nx // 2 + 1
    cplx = jnp.complex128
    shp_full = (ny, nx_r, nlev)
    shp_half = (ny, nx_r, nlev + 1)
    shp_tracers = (ny, nx_r, nlev, n_tracers)
    return SpectralPlanePhysicsState(
        u_hat=_make_field_with_sentinel(
            shp_full, cplx, "u_hat",
            ("ny", "nx_r", "nlev"), "m/s", sentinel=1.0,
        ),
        v_hat=_make_field_with_sentinel(
            shp_full, cplx, "v_hat",
            ("ny", "nx_r", "nlev"), "m/s", sentinel=2.0,
        ),
        w_hat=_make_field_with_sentinel(
            shp_half, cplx, "w_hat",
            ("ny", "nx_r", "nlev_half"), "m/s", sentinel=3.0,
        ),
        theta_prime_hat=_make_field_with_sentinel(
            shp_full, cplx, "theta_prime_hat",
            ("ny", "nx_r", "nlev"), "K", sentinel=4.0,
        ),
        rho_prime_hat=_make_field_with_sentinel(
            shp_full, cplx, "rho_prime_hat",
            ("ny", "nx_r", "nlev"), "kg/m^3", sentinel=5.0,
        ),
        # phis is REAL (physical-space static), not spectral.
        phis=_make_field_with_sentinel(
            (ny, nx), jnp.float64, "phis",
            ("ny", "nx"), "m^2/s^2", sentinel=6.0,
        ),
        tracers_hat=_make_field_with_sentinel(
            shp_tracers, cplx, "tracers_hat",
            ("ny", "nx_r", "nlev", "tracer"), "kg/kg", sentinel=7.0,
        ),
    )


def _build_dummy_tendencies(ny=4, nx=4, nlev=3, n_tracers=2):
    nx_r = nx // 2 + 1
    cplx = jnp.complex128
    shp_full = (ny, nx_r, nlev)
    shp_half = (ny, nx_r, nlev + 1)
    shp_tracers = (ny, nx_r, nlev, n_tracers)
    return SpectralPlanePhysicsTendencies(
        du_hat_dt=_make_field_with_sentinel(
            shp_full, cplx, "du_hat_dt",
            ("ny", "nx_r", "nlev"), "m/s^2", sentinel=10.0,
        ),
        dv_hat_dt=_make_field_with_sentinel(
            shp_full, cplx, "dv_hat_dt",
            ("ny", "nx_r", "nlev"), "m/s^2", sentinel=20.0,
        ),
        dw_hat_dt=_make_field_with_sentinel(
            shp_half, cplx, "dw_hat_dt",
            ("ny", "nx_r", "nlev_half"), "m/s^2", sentinel=30.0,
        ),
        dtheta_prime_hat_dt=_make_field_with_sentinel(
            shp_full, cplx, "dtheta_prime_hat_dt",
            ("ny", "nx_r", "nlev"), "K/s", sentinel=40.0,
        ),
        drho_prime_hat_dt=_make_field_with_sentinel(
            shp_full, cplx, "drho_prime_hat_dt",
            ("ny", "nx_r", "nlev"), "kg/m^3/s", sentinel=50.0,
        ),
        dphis_dt=_make_field_with_sentinel(
            (ny, nx), jnp.float64, "dphis_dt",
            ("ny", "nx"), "m^2/s^3", sentinel=60.0,
        ),
        dtracers_hat_dt=_make_field_with_sentinel(
            shp_tracers, cplx, "dtracers_hat_dt",
            ("ny", "nx_r", "nlev", "tracer"), "kg/kg/s",
            sentinel=70.0,
        ),
    )


def test_spectral_state_is_namedtuple_pytree():
    """SpectralPlanePhysicsState must be a NamedTuple + auto-register
    as a JAX pytree. tree_flatten/tree_unflatten round-trip MUST
    preserve VALUE identity (iter-265 HIGH#1: unique sentinels per
    field so a bad unflatten that swaps same-shape leaves is
    detected — shape/dtype-only checks would pass silently)."""
    state = _build_dummy_state()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, SpectralPlanePhysicsState)
    # Field-by-field VALUE identity through round-trip — sentinels
    # are unique per field so swaps trip the equality check.
    for fld in SpectralPlanePhysicsState._fields:
        orig = np.asarray(getattr(state, fld).data)
        rec = np.asarray(getattr(restored, fld).data)
        assert orig.shape == rec.shape, (
            f"field {fld!r} shape changed: {orig.shape} → {rec.shape}"
        )
        assert orig.dtype == rec.dtype, (
            f"field {fld!r} dtype changed: {orig.dtype} → {rec.dtype}"
        )
        np.testing.assert_array_equal(orig, rec, err_msg=(
            f"field {fld!r} VALUES changed after round-trip — "
            f"leaf-order or treedef regression."
        ))


def test_spectral_tendencies_is_namedtuple_pytree():
    """Mirror check for SpectralPlanePhysicsTendencies."""
    tend = _build_dummy_tendencies()
    leaves, treedef = jax.tree_util.tree_flatten(tend)
    restored = jax.tree_util.tree_unflatten(treedef, leaves)
    assert isinstance(restored, SpectralPlanePhysicsTendencies)
    for fld in SpectralPlanePhysicsTendencies._fields:
        orig = np.asarray(getattr(tend, fld).data)
        rec = np.asarray(getattr(restored, fld).data)
        assert orig.shape == rec.shape
        assert orig.dtype == rec.dtype
        np.testing.assert_array_equal(orig, rec, err_msg=(
            f"tendency {fld!r} VALUES changed after round-trip."
        ))


def test_spectral_state_tree_map_actually_visits_each_leaf():
    """jax.tree_util.tree_map across the state MUST visit every
    Field's .data array + produce a new state with the same pytree
    structure. Tests the SSP-RK3 averaging contract relies on
    (state_n+1 = tree_map(lambda x, y: 0.5*x + 0.5*y, state_a, state_b)).

    iter-265 (Codex iter-264 round-1 HIGH#2): the pre-iter-265
    version used all-zero sentinels + lambda x: 2*x, which gave
    2*0=0 — a no-op traversal or one that treated Field as opaque
    leaf passed silently. iter-265 uses unique nonzero sentinels +
    asserts new == 2*original element-wise.
    """
    state = _build_dummy_state()
    doubled = jax.tree_util.tree_map(lambda x: 2.0 * x, state)
    assert isinstance(doubled, SpectralPlanePhysicsState)
    for fld in SpectralPlanePhysicsState._fields:
        orig = np.asarray(getattr(state, fld).data)
        new = np.asarray(getattr(doubled, fld).data)
        assert orig.shape == new.shape
        assert orig.dtype == new.dtype
        np.testing.assert_array_equal(new, 2.0 * orig, err_msg=(
            f"tree_map(lambda x: 2*x) did not double field "
            f"{fld!r} — either the field is being treated as an "
            f"opaque leaf or tree_map is no-op'ing."
        ))


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
    match the edbae138 original (feature/crm-plane-spectral branch
    commit). A field reorder breaks pytree operations that rely on
    positional unpacking (e.g. SSP-RK3 stage averaging).

    iter-265 (Codex iter-264 round-1 MEDIUM#2): failure message
    now names edbae138 as the source-of-truth so a future
    refactor-triggered failure points the dev at the right upstream
    commit.
    """
    assert SpectralPlanePhysicsState._fields == _EXPECTED_STATE_FIELDS, (
        f"SpectralPlanePhysicsState._fields = "
        f"{SpectralPlanePhysicsState._fields}, expected "
        f"{_EXPECTED_STATE_FIELDS} (cherry-picked from "
        f"feature/crm-plane-spectral commit edbae138 — see "
        f"iter-241 in CRM_implementation.md). A field reorder "
        f"breaks SSP-RK3 positional unpacking."
    )


def test_spectral_tendencies_field_order_locked():
    """Mirror check for tendencies. Tendency field names follow
    the ``d<field>_dt`` convention with one entry per state field.
    """
    assert SpectralPlanePhysicsTendencies._fields == _EXPECTED_TENDENCIES_FIELDS, (
        f"SpectralPlanePhysicsTendencies._fields = "
        f"{SpectralPlanePhysicsTendencies._fields}, expected "
        f"{_EXPECTED_TENDENCIES_FIELDS} (cherry-picked from "
        f"feature/crm-plane-spectral commit edbae138 — see "
        f"iter-241 in CRM_implementation.md)."
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
