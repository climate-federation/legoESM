"""FV3_3D iter 604: PE AAM stack — static / drift / correction.

Mirror of NH AAM iter 583/587/588.

Tests
-----

1. ``test_aam_from_pe_state_finite_positive``.
2. ``test_aam_drift_pe_zero_for_identical``.
3. ``test_apply_aam_correction_pe_reduces_drift``.
4. ``test_apply_aam_correction_pe_only_adjusts_u_d_v_d``.
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
from legoesm.atmosphere.forcing.idealized.held_suarez import held_suarez_init
from legoesm.diagnostics import (
    aam_drift_pe,
    aam_from_pe_state,
    apply_aam_correction_pe,
)
from legoesm.grids.cubed_sphere import create_cubed_sphere
from legoesm.grids.cubed_sphere_cdgrid import create_cubed_sphere_cdgrid
from legoesm.grids.vertical import standard_hybrid_levels


def _build_pe(n=8, nlev=5, u_amp=0.0):
    grid = create_cubed_sphere(n, use_duogrid=True)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    coord = standard_hybrid_levels(nlev)
    state_hs = held_suarez_init(grid, coord)
    state = hydrostatic_to_fv3(state_hs, cdgrid)
    if u_amp != 0.0:
        state = state._replace(
            u_d=state.u_d.replace(data=state.u_d.data + u_amp),
        )
    return grid, coord, state


def test_aam_from_pe_state_finite_positive():
    grid, coord, state = _build_pe()
    col, total = aam_from_pe_state(state, grid, coord)
    assert col.shape == (6, 8, 8)
    assert np.isfinite(total)
    assert total > 0, f"AAM at HS rest > 0; got {total}"


def test_aam_drift_pe_zero_for_identical():
    grid, coord, state = _build_pe()
    drift = aam_drift_pe(state, state, grid, coord)
    assert abs(drift) < 1e-3, f"identical states drift ≈ 0; got {drift}"


def test_apply_aam_correction_pe_reduces_drift(capsys):
    grid, coord, state_old = _build_pe(u_amp=0.0)
    _, _, state_new = _build_pe(u_amp=1.0)
    drift_before = aam_drift_pe(state_old, state_new, grid, coord)
    state_corr = apply_aam_correction_pe(state_old, state_new, grid, coord)
    drift_after = aam_drift_pe(state_old, state_corr, grid, coord)
    with capsys.disabled():
        print(f"\n[iter-604] PE AAM drift before: {drift_before:.3e}")
        print(f"[iter-604] PE AAM drift after:  {drift_after:.3e}")
        if abs(drift_before) > 0:
            ratio = abs(drift_after) / abs(drift_before)
            print(f"[iter-604] reduction factor: {ratio:.3e}")
    assert abs(drift_after) < abs(drift_before) * 1e-4, (
        f"correction should reduce PE AAM drift by ≥4 orders; "
        f"before={drift_before:.3e}, after={drift_after:.3e}"
    )


def test_apply_aam_correction_pe_only_adjusts_u_d_v_d():
    grid, coord, state_old = _build_pe(u_amp=0.0)
    _, _, state_new = _build_pe(u_amp=1.0)
    state_corr = apply_aam_correction_pe(
        state_old, state_new, grid, coord,
    )
    # T, p_s, phis unchanged
    assert jnp.array_equal(state_corr.T.data, state_new.T.data)
    assert jnp.array_equal(state_corr.p_s.data, state_new.p_s.data)
    assert jnp.array_equal(state_corr.phis.data, state_new.phis.data)
    # u_d, v_d changed
    assert not jnp.array_equal(state_corr.u_d.data, state_new.u_d.data)
    assert not jnp.array_equal(state_corr.v_d.data, state_new.v_d.data)


def test_apply_aam_correction_pe_is_jit_and_grad_safe():
    """PR B #2: the correction advertised 'Differentiable end-to-end' but the
    old Newton loop used a host ``float()`` AAM + a data-dependent ``break`` on
    a traced scalar, so it could not be jit/grad'd. The fixed-iteration lax loop
    must now compile under jit and yield finite reverse-mode gradients."""
    grid, coord, state_old = _build_pe(u_amp=0.0)
    _, _, state_new = _build_pe(u_amp=5.0)

    jitted = jax.jit(lambda s: apply_aam_correction_pe(state_old, s, grid, coord))
    out = jitted(state_new)
    assert jnp.all(jnp.isfinite(out.u_d.data))

    def loss(u_new):
        sn = state_new._replace(u_d=state_new.u_d.replace(data=u_new))
        corr = apply_aam_correction_pe(state_old, sn, grid, coord)
        return jnp.sum(corr.u_d.data ** 2 + corr.v_d.data ** 2)

    g = jax.grad(loss)(state_new.u_d.data)
    assert jnp.all(jnp.isfinite(g)), "grad through PE AAM correction is non-finite"
    assert float(jnp.max(jnp.abs(g))) > 0.0, "grad is identically zero"
