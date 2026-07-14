"""FV3_3D iter 602: PE total-energy-conserving correction test.

Tests
-----

1. ``test_te_correction_pe_reduces_drift`` — after correction,
   TE(corrected) ≈ TE(state_old).
2. ``test_te_correction_pe_no_op_when_states_match``.
3. ``test_te_correction_pe_only_adjusts_T``.
"""
from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.dynamics.gcm.primitive_eq_cdgrid import (
    hydrostatic_to_fv3,
)
from legoesm.atmosphere.held_suarez import held_suarez_init
from legoesm.diagnostics import (
    apply_te_correction_pe,
    te_drift_pe,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _build_pe(n=8, nlev=5, T_delta=0.0):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    if T_delta != 0.0:
        state = state._replace(
            T=state.T.replace(data=state.T.data + T_delta),
        )
    return grid, coord, state


def test_te_correction_pe_reduces_drift(capsys):
    grid, coord, state_old = _build_pe(T_delta=0.0)
    _, _, state_new = _build_pe(T_delta=10.0)
    drift_before = te_drift_pe(state_old, state_new, grid, coord)
    state_corr = apply_te_correction_pe(state_old, state_new, grid, coord)
    drift_after = te_drift_pe(state_old, state_corr, grid, coord)
    with capsys.disabled():
        print(f"\n[iter-602] PE TE drift before: {drift_before:.3e}")
        print(f"[iter-602] PE TE drift after:  {drift_after:.3e}")
        if abs(drift_before) > 0:
            ratio = abs(drift_after) / abs(drift_before)
            print(f"[iter-602] reduction factor: {ratio:.3e}")
    assert abs(drift_after) < abs(drift_before) * 1e-6, (
        f"correction should reduce PE drift by ≥6 orders: "
        f"before={drift_before:.3e}, after={drift_after:.3e}"
    )


def test_te_correction_pe_no_op_when_states_match():
    grid, coord, state = _build_pe(T_delta=5.0)
    state_corr = apply_te_correction_pe(state, state, grid, coord)
    dT = float(jnp.abs(state_corr.T.data - state.T.data).max())
    assert dT < 1e-8, f"identical states should be no-op; max|ΔT|={dT}"


def test_te_correction_pe_only_adjusts_T():
    grid, coord, state_old = _build_pe(T_delta=0.0)
    _, _, state_new = _build_pe(T_delta=10.0)
    state_corr = apply_te_correction_pe(state_old, state_new, grid, coord)
    # u_d, v_d, p_s, phis unchanged
    assert jnp.array_equal(state_corr.u_d.data, state_new.u_d.data)
    assert jnp.array_equal(state_corr.v_d.data, state_new.v_d.data)
    assert jnp.array_equal(state_corr.p_s.data, state_new.p_s.data)
    assert jnp.array_equal(state_corr.phis.data, state_new.phis.data)
    # T changed
    assert not jnp.array_equal(state_corr.T.data, state_new.T.data)


def test_te_correction_pe_is_jit_and_grad_safe():
    """PR B #1: the correction's docstring claims 'Differentiable end-to-end'.
    The old Newton loop used a host ``float()`` energy + a data-dependent
    ``break``, so it could NOT be jit/grad'd. The fixed-iteration lax loop must
    now compile under jit and yield finite reverse-mode gradients."""
    grid, coord, state_old = _build_pe(T_delta=0.0)
    _, _, state_new = _build_pe(T_delta=10.0)

    jitted = jax.jit(lambda s: apply_te_correction_pe(state_old, s, grid, coord))
    out = jitted(state_new)
    assert jnp.all(jnp.isfinite(out.T.data))

    def loss(T_new):
        sn = state_new._replace(T=state_new.T.replace(data=T_new))
        return jnp.sum(apply_te_correction_pe(state_old, sn, grid, coord).T.data)

    g = jax.grad(loss)(state_new.T.data)
    assert jnp.all(jnp.isfinite(g)), "grad through PE TE correction is non-finite"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "grad is identically zero"
