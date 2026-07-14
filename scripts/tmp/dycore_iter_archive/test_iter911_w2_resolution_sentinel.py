"""Iter-911 regression sentinel: pin the iter-910 resolution-sweep
W2 v_ll_Linf measurements at C16 and C24 (production matrix config).

iter-910 measured:
  C16: 3.5367e-01 m/s  (dt=675s, 128 steps)
  C24: 1.8311e-01 m/s  (dt=450s, 192 steps)
  C36: 1.3188e-01 m/s  (dt=300s, 288 steps)  [pinned by iter-768]
  C48: 1.2488e-01 m/s  (dt=225s, 384 steps)  [too expensive for tests]

This sentinel pins C16 and C24 with ±5 % tolerance.  Any future iter
that changes the resolution-convergence behavior at low n will fire
this test, alerting the editor that iter-910's "post-iter-893 W2 is
1st-order resolution-refinable" finding may need re-measurement.

C16 runs in ~30 s (128 RK3 steps at C16 grid).  C24 runs in ~60 s.
Both are fast enough for routine CI inclusion.
"""
import os
os.environ.setdefault("JAX_ENABLE_X64", "1")

import numpy as np
import jax
import jax.numpy as jnp
import pytest

jax.config.update("jax_enable_x64", True)


def _div_damp_cube(n, ref_n=48, ref_coeff=1.5e7):
    return ref_coeff * (ref_n / n) ** 2


@pytest.fixture(scope="module")
def _w2_v_ll_linf_c16():
    """Module-scoped cache of the C16 W2 trajectory.  Iter-911b
    (Codex iter-911 stop-time fix): pre-iter-911b the third sentinel
    re-ran the C16 + C24 trajectories, doubling CI time.  Caching at
    module scope makes each trajectory run exactly once per test
    file invocation."""
    return _run_w2_v_ll_linf(n=16)


@pytest.fixture(scope="module")
def _w2_v_ll_linf_c24():
    """Module-scoped cache of the C24 W2 trajectory."""
    return _run_w2_v_ll_linf(n=24)


def _run_w2_v_ll_linf(n: int) -> float:
    """Run W2 1-day at the iter-892/iter-893 production matrix config
    with CFL-preserving dt = 300 * (36/n) and return v_ll_Linf."""
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.cubed_sphere_cdgrid import (
        cell_centre_angles_from_4edge)
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig, FV3EdgeShallowWaterModel,
        FV3EdgeShallowWaterState)
    from tests.atmosphere.shallow_water.test_cases.williamson import (
        williamson_test2)
    from legoesm.grids.regridding import (
        get_cubedsphere_to_latlon_weights, apply_cubedsphere_to_latlon)

    dt = 300.0 * (36 / n)
    n_steps = int(86400 / dt)
    grid = create_cubed_sphere(n=n, use_duogrid=False)
    sw = williamson_test2(grid)
    u0 = 2.0 * np.pi * float(grid.radius) / (12.0 * 86400.0)
    weights = get_cubedsphere_to_latlon_weights(n, n_lon=360, n_lat=181)

    cfg = CDGridShallowWaterConfig(
        hyperdiff_coeff=0.0,
        div_damp=8.0 * _div_damp_cube(n),
        boundary_fix=True,
        damp_v=0.06,
        nord_v=2,
        apply_fortran_xppm_boundary=True,
    )
    model = FV3EdgeShallowWaterModel(grid, config=cfg)
    cdg = model.cdgrid
    u_d = cdg.cos_angle_edge_x * (u0 * jnp.cos(cdg.lat_edge_x))
    v_d = -cdg.sin_angle_edge_y * (u0 * jnp.cos(cdg.lat_edge_y))
    state = FV3EdgeShallowWaterState(
        h=sw.h.data, u_d=u_d, v_d=v_d, h_s=sw.h_s.data)
    model.set_initial_mass(state)

    for _ in range(n_steps):
        state = model.step(state, dt)

    ca, sa = cell_centre_angles_from_4edge(cdg)
    u_cc = 0.5 * (np.asarray(state.u_d)[:, :, :-1]
                   + np.asarray(state.u_d)[:, :, 1:])
    v_cc = 0.5 * (np.asarray(state.v_d)[:, :-1, :]
                   + np.asarray(state.v_d)[:, 1:, :])
    v_north = np.asarray(sa) * u_cc + np.asarray(ca) * v_cc
    v_ll = apply_cubedsphere_to_latlon(v_north, weights)
    return float(np.max(np.abs(v_ll)))


def test_iter911_w2_c16_v_ll_linf_matches_iter910(_w2_v_ll_linf_c16):
    """iter-910 measured C16 W2 v_ll_Linf = 0.3537 m/s.  Pin within
    ±5 % tolerance."""
    iter910_value = 3.5367e-01
    tol_pct = 0.05  # 5 %
    measured = _w2_v_ll_linf_c16
    assert np.isfinite(measured), (
        f"C16 W2 v_ll_Linf is NaN — production path destabilized.")
    rel_err = abs(measured - iter910_value) / iter910_value
    assert rel_err < tol_pct, (
        f"C16 W2 v_ll_Linf = {measured:.4e} drifted by {rel_err*100:.2f}% "
        f"from iter-910's 0.3537 m/s.  Tolerance ±5 %.  Either the "
        f"production matrix changed (intentional?) or the resolution-"
        f"convergence behavior was inadvertently altered.  If the change "
        f"is intentional, re-baseline by updating this test plus the "
        f"iter-910 doc-entry table.")


def test_iter911_w2_c24_v_ll_linf_matches_iter910(_w2_v_ll_linf_c24):
    """iter-910 measured C24 W2 v_ll_Linf = 0.1831 m/s.  Pin within
    ±5 % tolerance.  C24 takes ~60 s — runs slower than C16."""
    iter910_value = 1.8311e-01
    tol_pct = 0.05
    measured = _w2_v_ll_linf_c24
    assert np.isfinite(measured), (
        f"C24 W2 v_ll_Linf is NaN — production path destabilized.")
    rel_err = abs(measured - iter910_value) / iter910_value
    assert rel_err < tol_pct, (
        f"C24 W2 v_ll_Linf = {measured:.4e} drifted by {rel_err*100:.2f}% "
        f"from iter-910's 0.1831 m/s.  See iter-911 docstring for "
        f"re-baselining guidance.")


def test_iter911_w2_resolution_convergence_holds_c16_to_c24(
        _w2_v_ll_linf_c16, _w2_v_ll_linf_c24):
    """The post-iter-893 baseline shows resolution refinement: C16 ->
    C24 should reduce v_ll_Linf by ~50 % (iter-910: 0.3537 -> 0.1831,
    -48 %).  Pin the convergence ratio with ±10 % tolerance.

    A future change that turned this into a non-monotone or
    resolution-invariant pattern would fire this sentinel — reverting
    to the pre-iter-893 "resolution-invariant" regime would be a
    serious W2 regression.

    Iter-911b (Codex iter-911 stop-time fix): uses the module-scoped
    fixtures `_w2_v_ll_linf_c16` and `_w2_v_ll_linf_c24` so the
    trajectories run exactly ONCE for the whole test file (was: this
    test re-ran both trajectories, doubling CI time).
    """
    iter910_c16 = 3.5367e-01
    iter910_c24 = 1.8311e-01
    iter910_ratio = iter910_c16 / iter910_c24  # ≈ 1.93

    c16 = _w2_v_ll_linf_c16
    c24 = _w2_v_ll_linf_c24
    assert np.isfinite(c16) and np.isfinite(c24)
    measured_ratio = c16 / c24
    rel = abs(measured_ratio - iter910_ratio) / iter910_ratio
    assert rel < 0.10, (
        f"C16/C24 W2 ratio = {measured_ratio:.3f} drifted by "
        f"{rel*100:.2f}% from iter-910's {iter910_ratio:.3f}.  The "
        f"resolution-convergence pattern has changed; verify whether "
        f"iter-893's resolution-refinable property still holds.")
