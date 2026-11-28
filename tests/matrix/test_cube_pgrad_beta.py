"""Cube FV3 duo lane: production PG off-centering ``beta`` (FV3
grad1_p_update, dyn_core.F90:1528; beta_d = 0 on it == 1, :440-444).
USER APPROVED 2026-11-28 as the cube default (0.4).

SW lane (km=1, C12, via the matrix adapter) and the km>1 hydrostatic
model.  Each gate fails when the corresponding piece is reverted:
* the it == 1 weight-zero rule (n_split = 1 => beta has no effect);
* beta reaches the step (n_split = 7 => it does);
* gravity-wave energy damping per step is reduced, mass exact;
* the production defaults (adapter + FV3DuoConfig) are 0.4, the oracle
  deck 0.0, and NH refuses an explicit beta > 0 (split_p_grad unported).
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp  # noqa: E402

A = pytest.importorskip("scripts.matrix._cube_duo_adapters")
import legoesm.core.fv3_duo_stepper as D  # noqa: E402

N = 12


@pytest.fixture(scope="module")
def sw():
    ns = A.build_duo_sw(8, N)
    jctx = D.build_jax_duo_stepper_context(ns.ctx)
    cfg = D.SW_CFG_CASE8          # the adapter's case-8 deck (d4_bg 0.12)
    return ns, jctx, cfg, D.make_full_acoustic_step_sixface_jit()


def _outer(sw, s, beta, n_split):
    ns, jctx, cfg, jstep = sw
    return D.advance_duo_outer_step(jctx, s, ns.dt, n_split, d_ext=0.0,
                                    sw_cfg=cfg, step_fn=jstep, beta=beta)


def _same(a, b):
    return all(np.array_equal(np.asarray(a[k]), np.asarray(b[k]))
               for k in ("delp", "pt", "u", "v"))


def test_first_substep_has_zero_weight(sw):
    """n_split = 1: every step is it == 1 (beta_d = 0) => bitwise beta=0."""
    s0 = sw[0].state
    assert _same(_outer(sw, s0, 0.4, 1), _outer(sw, s0, 0.0, 1))


def test_beta_reaches_the_step(sw):
    s0 = sw[0].state
    a, b = _outer(sw, s0, 0.4, 7), _outer(sw, s0, 0.0, 7)
    assert set(a) == set(b)                          # no du/dv leak
    assert float(np.abs(np.asarray(a["u"]) - np.asarray(b["u"])).max()) > 1e-6


def test_sw_step_refuses_a_misstacked_pg_carry(sw):
    ns, jctx, cfg, jstep = sw
    u, v = jnp.asarray(ns.state["u"]), jnp.asarray(ns.state["v"])
    bad = (jnp.concatenate([u, u[:1]]), jnp.concatenate([v, v[:1]]))
    with pytest.raises(ValueError, match="stacked like the winds"):
        D.full_acoustic_step_sixface(jctx, ns.state, ns.dt / 7, d_ext=0.0,
                                     sw_cfg=cfg, entry_ascalar=False,
                                     beta=0.4, beta_d=0.4, pg_prev=bad)


def _energy_mass(ns, s):
    h = ns.h(s)
    ue, vn = ns.winds(s)
    a = np.asarray(ns.area)
    hb = (a * h).sum() / a.sum()
    e = (a * h * (ue ** 2 + vn ** 2) / 2).sum() + (a * ns.grav * (h - hb) ** 2 / 2).sum()
    return float(e), float((a * h).sum())


def test_gravity_wave_damping_reduced_mass_exact(sw):
    """Standing P6 gravity mode (non-rotating, rest + 10 m on 10 km):
    the acoustic C/D predictor damps (KE+APE); beta = 0.4 must lose < 0.6x
    the beta = 0 loss over 6 outer steps (measured 2026-11-28, C12:
    beta 0 -> 7.96 %, beta 0.4 -> 3.19 %).  Mass exact both ways."""
    from scipy.special import eval_legendre
    ns = sw[0]
    lat = np.stack([np.asarray(g["agrid_lat"]) for g in ns.ctx["gs6"]])
    s0 = dict(ns.state)
    s0["u"] = jnp.zeros_like(s0["u"])
    s0["v"] = jnp.zeros_like(s0["v"])
    s0["delp"] = jnp.asarray(ns.grav * (1.0e4 + 10.0 * eval_legendre(6, np.sin(lat))))
    e0, m0 = _energy_mass(ns, s0)
    loss = {}
    for beta in (0.0, 0.4):
        s = s0
        for _ in range(6):
            s = _outer(sw, s, beta, 7)
        e, m = _energy_mass(ns, s)
        loss[beta] = 1.0 - e / e0
        # mass: conserved to 1e-13 relative (measured 0.0)
        assert abs(m / m0 - 1.0) < 1e-13, (beta, m / m0 - 1.0)
    assert loss[0.0] > 0.03                         # non-vacuity: it damps
    assert 0.0 <= loss[0.4] < 0.6 * loss[0.0], loss  # less damping, no gain


def test_matrix_adapter_default_is_production_beta(sw):
    """The adapter's own step closure runs beta = 0.4 (not only its note)."""
    assert D.CUBE_PGRAD_BETA == 0.4
    ns = sw[0]
    assert "beta=0.4" in ns.note
    got = ns.step(ns.state, ns.dt)
    assert _same(got, _outer(sw, ns.state, 0.4, ns.n_split))
    assert not _same(got, _outer(sw, ns.state, 0.0, ns.n_split))


def test_fv3duoconfig_beta_resolution():
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        ORACLE_DAMPING, FV3DuoConfig, resolve_pgrad_beta)
    assert resolve_pgrad_beta(FV3DuoConfig()) == 0.4
    assert resolve_pgrad_beta(FV3DuoConfig(hydrostatic=False)) == 0.0
    assert resolve_pgrad_beta(FV3DuoConfig(**ORACLE_DAMPING)) == 0.0
    assert FV3DuoConfig._fields[-1] == "beta"       # appended last


def test_nh_refuses_explicit_beta():
    from legoesm.core.fv3_acoustic_3d import acoustic_substep_3d
    with pytest.raises(NotImplementedError, match="split_p_grad"):
        acoustic_substep_3d(None, {"delp": jnp.zeros(1)}, 1.0, 3,
                            first_substep=True, ptop=1.0, akap=1.0,
                            cp_air=1.0, hydrostatic=False, nh={}, dp0=0,
                            beta=0.4)


@pytest.mark.parametrize("n_split, equal", [(1, True), (2, False)])
def test_hydro_model_default_beta(n_split, equal):
    """km>1 hydrostatic model: the DEFAULT config runs beta = 0.4 (moves the
    step at n_split = 2) and the it == 1 sub-step has zero weight
    (n_split = 1 => bitwise the beta = 0 step)."""
    from legoesm.atmosphere.dynamics.gcm.fv3_duo_dynamics import (
        FV3DuoConfig, FV3DuoDynamicsModel)
    from legoesm.grids.factory import create_fv3_duo_grid
    grid = create_fv3_duo_grid(N, 3)
    out = []
    for kw in ({}, {"beta": 0.0}):
        m = FV3DuoDynamicsModel(grid, FV3DuoConfig(km=5, n_split=n_split, **kw))
        s = m.dcmip16_initial_state()
        out.append(m.step(s, 120.0)["state"])
        jax.clear_caches()
    assert _same(out[0], out[1]) is equal
