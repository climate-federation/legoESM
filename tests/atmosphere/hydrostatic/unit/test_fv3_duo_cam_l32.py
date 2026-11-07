"""Rung 4a: the FV3 duo on CAM6's L32 hybrid table (route A, M4).

The step reads ak/bk/ptop generically; the L32 deck is not
Fortran-certified, so this file gates what the table changes: the
pure-pressure top (bk == 0 for 15 interfaces, no ``ks`` branch), the
column view's hybrid coordinate against the native pressures, and a
short DCMIP16 baroclinic-wave run at km=32 staying finite and sane.
"""

from __future__ import annotations

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402

from legoesm import constants  # noqa: E402
from legoesm.grids.factory import create_fv3_duo_grid  # noqa: E402

N, NG, KM = 12, 3, 32
DT = 1920.0
CI = slice(NG, NG + N)


@pytest.fixture(autouse=True)
def _drop_compiled_graphs():
    yield
    jax.clear_caches()


@pytest.fixture(scope="module")
def grid():
    return create_fv3_duo_grid(N, NG)


def test_table_is_cam6_l32_and_refuses_other_km():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import duo_eta_table
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI
    ak, bk, ptop = duo_eta_table("cam6_l32", 32)
    np.testing.assert_array_equal(ak, np.asarray(CAM6_L32_HYAI) * constants.p_ref)
    np.testing.assert_array_equal(bk, np.asarray(CAM6_L32_HYBI))
    assert ptop == ak[0] and np.isclose(ptop, 225.52, atol=0.01)
    assert (bk[:15] == 0.0).all() and bk[15] > 0.0 and bk[-1] == 1.0
    assert (np.diff(ak + bk * 5e4) > 0.0).all()      # positive dp at 500 hPa
    for km in (5, 10, 31):
        with pytest.raises(ValueError, match="32-level"):
            duo_eta_table("cam6_l32", km)
    with pytest.raises(ValueError, match="analytic"):
        duo_eta_table("nope", 32)


def test_pure_pressure_top_is_ps_invariant_without_a_ks_branch():
    """pe = ak + bk*ps on the 15 pure-pressure interfaces does not move
    when ps moves by +-10 hPa (GLM 2026-09-27: the property FV3's ``ks``
    protects), through the lane's own p_var_hydrostatic."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import duo_eta_table
    from legoesm.core.fv3_dynamics import p_var_hydrostatic
    from legoesm.grids.fv3_native_gridstruct import FV3_KAPPA
    ak, bk, ptop = duo_eta_table("cam6_l32", 32)
    m = N + 2 * NG
    rng = np.random.default_rng(0)
    ps0 = 1.0e5 + 1.0e3 * rng.standard_normal((6, m, m))
    pes = []
    for dps in (0.0, 1.0e3, -1.0e3):
        ps = ps0 + dps
        delp = (np.diff(ak)[None, None, None, :]
                + np.diff(bk)[None, None, None, :] * ps[..., None])
        pr = p_var_hydrostatic(jnp.asarray(delp), ptop=ptop, akap=FV3_KAPPA,
                               n=N, ng=NG, km=KM)
        pes.append(np.asarray(pr["pe"]))
    for pe, dps in zip(pes[1:], (1.0e3, -1.0e3)):
        # interfaces 0..14 are pure pressure (bk == 0); 15 is the first
        # hybrid one and must move by bk[15]*dps (codex 2026-09-27)
        np.testing.assert_allclose(pe[:, :, :15, :], pes[0][:, :, :15, :],
                                   rtol=1e-13, atol=0)
        d15 = pe[:, 1:N + 1, 15, 1:N + 1] - pes[0][:, 1:N + 1, 15, 1:N + 1]
        np.testing.assert_allclose(d15, bk[15] * dps, rtol=1e-10)


@pytest.fixture(scope="module")
def l32(grid):
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_column import (
        FV3DuoColumnModel)
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    dyn = FV3DuoDynamicsModel(
        grid, FV3DuoConfig(km=KM, n_split=8, eta="cam6_l32", moist=True))
    return dyn, FV3DuoColumnModel(dyn)


def test_dcmip16_on_l32_runs_two_days_finite_and_sane(l32):
    """DCMIP16 baroclinic wave at km=32 with the 225 Pa top: two days
    (90 steps at dt=1920 s, C12) finite, temperatures physical, winds
    bounded, total mass conserved to roundoff; the column view's hybrid
    coordinate reproduces the native interface pressures throughout."""
    dyn, col = l32
    ic = dyn.dcmip16_initial_state(do_pert=True)
    q0 = ic["q"][0]                       # the column model's three slots
    ic = {**ic, "q": [q0, jnp.zeros_like(q0), jnp.zeros_like(q0)]}
    area = np.stack([dyn.grid.ctx_np["gs6"][t]["area"][CI, CI]
                     for t in range(6)])

    def mass(b):
        return float((np.asarray(b["state"]["delp"])[:, CI, CI].sum(-1)
                      * area).sum())
    m0 = mass(ic)
    b = ic
    n_steps = int(2 * 86400 / DT)
    for s in range(n_steps):
        b = dyn.step(b, DT)
        if s in (0, n_steps // 2, n_steps - 1):
            st = col.from_bundle(b)
            pe = np.transpose(np.asarray(b["press"]["pe"])[:, 1:N + 1, :, 1:N + 1],
                              (0, 1, 3, 2)).reshape(-1, KM + 1)
            ph = np.asarray(col.sigma_coord.pressure_at_half(st.p_s.data))
            np.testing.assert_allclose(ph, pe, rtol=1e-12, atol=0)
    pt = np.asarray(b["state"]["pt"])[:, CI, CI]
    pt0 = np.asarray(ic["state"]["pt"])[:, CI, CI]
    assert np.isfinite(pt).all()
    # not an identity stepper: the wave moved the temperature field
    assert np.abs(pt - pt0).max() > 0.1     # MEASURED 0.61 K at 2 d (GPU job 10061987)
    # the DCMIP16 profile is 149.0 K in the 2.3 hPa top layer (IC, MEASURED)
    assert 140.0 < pt.min() and pt.max() < 350.0, (pt.min(), pt.max())
    st = col.from_bundle(b)
    spd = np.hypot(np.asarray(st.u.data), np.asarray(st.v.data))
    assert np.isfinite(spd).all() and spd.max() < 120.0, spd.max()
    assert abs(mass(b) - m0) / m0 < 1e-12
