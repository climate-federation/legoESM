"""End-to-end reachability of Bechtold's RCAPQADV correction through the
hydrostatic convection bridge (``use_ifs_cape_qadv`` + PhysicsState carry).

The RCAPQADV CAPE-advection correction needs the dynamics (large-scale
advective) T/q tendencies — the IFS ``PTENTA``/``PTENQA`` analog.  A driver
stashes them in ``PhysicsState.dyn_tendency_T`` / ``dyn_tendency_qv`` before
the convection call; the bridge threads them to the leaf and, when the flag is
on but the carry is absent, raises at trace time so the flag can never be
silently inert.  These tests pin that contract at the bridge (production) level,
complementing the leaf-level pins in ``tests/unit/test_bechtold.py``.
"""

from __future__ import annotations

from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.convection.config import ConvectionConfig  # noqa: E402
from legoesm.atmosphere.physics.convection.integration import (  # noqa: E402
    make_convection_physics,
)
from legoesm.core.field import Field  # noqa: E402
from legoesm.core.state import HydrostaticState  # noqa: E402
from legoesm.grids.cubed_sphere import create_cubed_sphere  # noqa: E402
from legoesm.grids.vertical import make_hybrid_levels  # noqa: E402

_NLEV = 12
_DT = 600.0


def _setup():
    grid = create_cubed_sphere(4)
    sigma = make_hybrid_levels(_NLEV)
    s2 = grid.grid_lat.shape
    s3 = (*s2, _NLEV)
    # Steep-lapse, SUB-SATURATED conditionally-unstable column (surface at
    # index -1): CAPE ~ 70 J/kg (convection fires) with a column-mean
    # saturation fraction ~ 0.41, well below the ZSATFR = 0.94 gate, so the
    # RCAPQADV advection branch is OPEN (a humid convecting column would shut
    # it — the physically-correct behaviour verified in the SCM-RCE work).
    z = np.linspace(0, 1, _NLEV)[::-1]
    Tprof = 302.0 - 80.0 * z
    T = jnp.asarray(np.broadcast_to(Tprof, s3).copy())
    qprof = 8e-3 * np.exp(-3.5 * z)
    q_v = jnp.asarray(np.broadcast_to(qprof, s3).copy())
    u = jnp.zeros(s3)
    v = jnp.zeros(s3)
    d3 = ("face", "x", "y", "level")
    d2 = ("face", "x", "y")
    state = HydrostaticState(
        u=Field(u, name="u", dims=d3, units="m/s"),
        v=Field(v, name="v", dims=d3, units="m/s"),
        T=Field(T, name="T", dims=d3, units="K"),
        p_s=Field(jnp.full(s2, 1.0e5), name="p_s", dims=d2, units="Pa"),
        phis=Field(jnp.zeros(s2), name="phis", dims=d2, units="m2/s2"),
        tracers={"q_v": Field(q_v, name="q_v", dims=d3, units="kg/kg")},
    )
    ncol = int(np.prod(s2))
    return grid, sigma, state, ncol


def _phys_state(ncol, *, dyn_T=None, dyn_qv=None):
    return SimpleNamespace(
        conv_prog_profile=jnp.zeros((ncol, _NLEV)),
        conv_stoch_state=jnp.zeros((ncol,)),
        prng_key=jax.random.PRNGKey(0),
        col_index=jnp.arange(ncol, dtype=jnp.int32),
        dyn_tendency_T=dyn_T,
        dyn_tendency_qv=dyn_qv,
    )


def _bechtold_cfg(**overrides):
    cfg = ConvectionConfig(scheme="bechtold")
    return cfg._replace(bechtold=cfg.bechtold._replace(**overrides))


def test_qadv_off_ignores_dyn_tendency_carry():
    """With ``use_ifs_cape_qadv=False`` (default) a supplied dyn-tendency carry
    must be ignored — the bridge is bit-identical to no carry."""
    grid, sigma, state, ncol = _setup()
    fn = make_convection_physics(_bechtold_cfg(), model_type="hydrostatic", dt=_DT)
    base = fn(state, grid, sigma, _phys_state(ncol))
    dynT = jnp.full((ncol, _NLEV), 3e-5)
    dynq = jnp.full((ncol, _NLEV), 2e-8)
    withcarry = fn(state, grid, sigma,
                   _phys_state(ncol, dyn_T=dynT, dyn_qv=dynq))
    assert bool(jnp.all(base[0].dT_dt.data == withcarry[0].dT_dt.data))


def test_qadv_on_without_carry_raises():
    """Flag on but no dyn-tendency carry -> loud trace-time error (the
    silently-inert case the guard exists to prevent)."""
    grid, sigma, state, ncol = _setup()
    fn = make_convection_physics(
        _bechtold_cfg(use_ifs_cape_qadv=True), model_type="hydrostatic", dt=_DT)
    with pytest.raises(ValueError, match="dyn_tendency"):
        fn(state, grid, sigma, _phys_state(ncol))


def test_qadv_on_with_carry_is_reachable_and_takes_effect():
    """Flag on + a dyn-tendency carry present -> the correction runs and the
    carry reaches the leaf: a nonzero moisture-advection tendency moves the
    diagnosed convective heating relative to the SAME flag-on run with a ZERO
    carry (both flag-on, so this isolates the tendency's effect from the gate
    / closure baseline)."""
    grid, sigma, state, ncol = _setup()
    on = make_convection_physics(
        _bechtold_cfg(use_ifs_cape_qadv=True), "hydrostatic", _DT)
    zero = _phys_state(ncol, dyn_T=jnp.zeros((ncol, _NLEV)),
                       dyn_qv=jnp.zeros((ncol, _NLEV)))
    # Moistening large-scale advection (ZDQCV > 0) in the sensitive band.
    carry = _phys_state(ncol, dyn_T=jnp.full((ncol, _NLEV), 2e-5),
                        dyn_qv=jnp.full((ncol, _NLEV), 1e-7))
    out_zero = on(state, grid, sigma, zero)
    out_carry = on(state, grid, sigma, carry)
    assert bool(jnp.all(jnp.isfinite(out_carry[0].dT_dt.data)))
    assert bool(jnp.any(
        out_carry[0].dT_dt.data != out_zero[0].dT_dt.data)), (
        "the dynamics-tendency carry had no effect through the bridge — it is "
        "not reaching the RCAPQADV closure in the leaf"
    )


def test_qadv_bridge_is_ad_safe_through_dyn_tendency():
    """Gradient of the convective heating w.r.t. the dynamics moisture
    tendency is finite (the carry participates in a differentiable closure)."""
    grid, sigma, state, ncol = _setup()
    on = make_convection_physics(
        _bechtold_cfg(use_ifs_cape_qadv=True), "hydrostatic", _DT)
    dynT = jnp.full((ncol, _NLEV), 2e-5)

    def loss(dynq):
        ps = _phys_state(ncol, dyn_T=dynT, dyn_qv=dynq)
        return jnp.sum(on(state, grid, sigma, ps)[0].dT_dt.data ** 2)

    g = jax.grad(loss)(jnp.full((ncol, _NLEV), 1e-8))
    assert bool(jnp.all(jnp.isfinite(g)))
