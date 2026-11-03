"""IFS 49r1 CLOUDSC port: conservation, smooth forms, gradients (no oracle needed).

Uses the oracle fixture's input columns (oracle_fixtures/ifs_cloudsc_49r1.npz) as
realistic states.  Budgets per column, from the scheme's own outputs:

* water:   sum_k (dq/dt + sum_c dq_c/dt) dp/g + P_sfc - D = 0
* enthalpy (liquid-water temperature, K kg m-2 s-1):
  sum_k (dT/dt - Lv/cp (dq_l+dq_r)/dt - Ls/cp (dq_i+dq_s)/dt) dp/g
      - (Lv/cp P_liq + Ls/cp P_ice)_sfc + (Lv/cp D_l + Ls/cp D_i) = 0
with P the surface precipitation flux and D the convective detrainment CLOUDSC takes
in (an external source, sections 3.2).
"""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pytest

import legoesm.atmosphere.physics.microphysics.ifs_cloudsc as M

ROOT = Path(__file__).resolve().parents[4]


def _load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


T = _load("t_cloudsc_oracle", ROOT / "tests/atmosphere/test_ifs_cloudsc_oracle.py")
_ARMS = ("base", "detr")


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(T._FIX))


def budgets(cfg, th, x, out, dt):
    g = th.RG
    dp = np.asarray(x.paph[:, 1:] - x.paph[:, :-1])
    k0 = cfg.ncldtop - 1
    tq, tt = np.asarray(out.tend_q), np.asarray(out.tend_t)
    tc = np.asarray(out.tend_cld)
    psfc_l, psfc_i = np.asarray(out.fplsl)[:, -1], np.asarray(out.fplsn)[:, -1]
    # detrainment actually taken in: weight = PLUDE_out / (PLUDE_in dt g/dp), levels k0..nlev-2
    pin = np.asarray(x.plude) * dt * g / dp
    w = np.where(pin > 0, np.asarray(out.plude) / np.where(pin > 0, pin, 1.0), 0.0)
    w[:, :k0] = 0.0
    w[:, -1] = 0.0
    dl = np.sum(w * np.asarray(x.pludeli[..., 0]), axis=1)
    di = np.sum(w * np.asarray(x.pludeli[..., 1]), axis=1)
    assert np.all(np.asarray(x.psnde) == 0)
    water_lev = (tq + tc.sum(-1)) * dp / g
    h_lev = (tt - th.RALVDCP * (tc[..., 0] + tc[..., 2]) - th.RALSDCP * (tc[..., 1] + tc[..., 3])) * dp / g
    scale_w = np.sum(np.abs(water_lev), 1) + np.abs(psfc_l + psfc_i) + dl + di + 1e-30
    scale_h = np.sum(np.abs(h_lev), 1) + th.RALSDCP * (np.abs(psfc_l) + np.abs(psfc_i) + dl + di) + 1e-30
    res_w = water_lev.sum(1) + psfc_l + psfc_i - dl - di
    res_h = h_lev.sum(1) - th.RALVDCP * psfc_l - th.RALSDCP * psfc_i + th.RALVDCP * dl + th.RALSDCP * di
    above_w = water_lev[:, :k0].sum(1)
    above_h = h_lev[:, :k0].sum(1)
    return res_w, res_h, above_w, above_h, scale_w, scale_h


@pytest.mark.parametrize("hard", [True, False])
@pytest.mark.parametrize("arm", _ARMS)
def test_water_and_enthalpy_conserved(fix, arm, hard):
    cfg, th, x, dt = T.setup(fix, arm)
    out = M.cloudsc(cfg, th, x, dt, hard=hard)
    res_w, res_h, above_w, above_h, sw, sh = budgets(cfg, th, x, out, dt)
    # 49r1 defect, kept for fidelity: section 2 evaporates small condensate on EVERY level
    # into q/T tendencies, but above NCLDTOP the condensate tendency is never written, so
    # the column gains that water.  Report it; the rest of the column must close.
    assert np.max(np.abs(res_w - above_w) / sw) < 1e-12
    assert np.max(np.abs(res_h - above_h) / sh) < 1e-12


def test_above_ncldtop_water_defect_matches_independent_reference(fix):
    """49r1 defect (kept, vote V23a): above NCLDTOP the section-2 tidy adds the evaporated
    condensate to q but writes no condensate tendency.  Independent numpy reference of the
    tidy (cloudsc.F90:1127-1175) on the levels above NCLDTOP; zero on these columns."""
    cfg, th, x, dt = T.setup(fix, "base")
    out = M.cloudsc(cfg, th, x, dt, hard=True)
    _, _, above_w, _, _, _ = budgets(cfg, th, x, out, dt)
    k0 = cfg.ncldtop - 1
    c = np.asarray(x.clv[:, :k0, :4]) + dt * np.asarray(x.tend_cld[:, :k0, :4])
    a = np.asarray(x.a[:, :k0]) + dt * np.asarray(x.tend_a[:, :k0])
    moved = np.zeros(c.shape[:2])
    small = (c[..., 0] + c[..., 1] < cfg.rlmin) | (a < cfg.ramin)
    for m in (0, 1):
        moved += np.where(small, c[..., m], 0.0)
        c[..., m] = np.where(small, 0.0, c[..., m])
    for m in range(4):
        moved += np.where(c[..., m] < cfg.rlmin, c[..., m], 0.0)
    dp = np.asarray(x.paph[:, 1:k0 + 1] - x.paph[:, :k0])
    ref = np.sum(moved / dt * dp / th.RG, axis=1)
    assert np.allclose(above_w, ref, rtol=1e-12, atol=1e-30)
    assert np.all(ref == 0)       # size on the oracle columns: none (reported in the PR)


# --- smooth forms ---------------------------------------------------------------
_W = 0.01


@pytest.mark.parametrize("op", ["pos", "max", "min", "abs", "gt", "lt"])
def test_smooth_primitive_c1_and_exact_outside_window(op):
    s, h = M._Sm(False), M._Sm(True)
    f = {"pos": lambda m, x: m.pos(x, _W), "max": lambda m, x: m.max(x, 0.3, _W),
         "min": lambda m, x: m.min(x, 0.3, _W), "abs": lambda m, x: m.abs(x, _W),
         "gt": lambda m, x: m.gt(x, 0.3, _W), "lt": lambda m, x: m.lt(x, 0.3, _W)}[op]
    c = 0.0 if op in ("pos", "abs") else 0.3
    xs = jnp.linspace(c - 3 * _W, c + 3 * _W, 601)
    g = jax.vmap(jax.grad(lambda x: f(s, x)))(xs)
    assert np.all(np.isfinite(np.asarray(g)))
    d = np.asarray(xs) - c
    inside = {"gt": np.abs(d) < 0.4 * _W, "lt": np.abs(d) < 0.4 * _W,
              "abs": (np.abs(d) > 0.1 * _W) & (np.abs(d) < 0.9 * _W)}.get(
                  op, (d > 0.1 * _W) & (d < 0.9 * _W))
    assert np.all(np.abs(np.asarray(g))[inside] > 0)          # nonzero across the transition
    far = np.abs(np.asarray(xs) - c) > 1.01 * _W
    assert np.array_equal(np.asarray(f(s, xs))[far], np.asarray(f(h, xs), np.float64)[far])
    vals = np.asarray(f(s, xs))
    assert np.max(np.abs(np.diff(vals))) < 0.1                 # continuous on the sample


def test_smooth_scheme_gradient_finite_and_matches_fd(fix):
    cfg, th, x, dt = T.setup(fix, "detr", slice(0, 4))

    def loss(t, q, a):
        o = M.cloudsc(cfg, th, x._replace(t=t, q=q, a=a), dt, hard=False)
        return (jnp.sum(o.tend_t) * 1e3 + jnp.sum(o.tend_q) * 1e8 + jnp.sum(o.tend_a) * 1e3
                + jnp.sum(o.fplsl[:, -1] + o.fplsn[:, -1]) * 1e5)

    g = jax.grad(loss, argnums=(0, 1, 2))(x.t, x.q, x.a)
    for gi in g:
        assert np.all(np.isfinite(np.asarray(gi)))
    assert all(np.any(np.asarray(gi) != 0) for gi in g)
    rng = np.random.default_rng(0)
    # steps well inside the narrowest smooth windows (0.01 K, 1e-10 kg/kg, 1e-10 fraction)
    for i, (arr, scale) in enumerate(((x.t, 1e-6), (x.q, 1e-14), (x.a, 1e-13))):
        d = jnp.asarray(rng.standard_normal(arr.shape)) * scale
        args = [x.t, x.q, x.a]
        plus = list(args); plus[i] = args[i] + d
        minus = list(args); minus[i] = args[i] - d
        fd = (loss(*plus) - loss(*minus)) / 2.0
        ad = float(jnp.sum(g[i] * d))
        assert abs(fd - ad) <= 2e-2 * max(abs(ad), abs(fd)) + 1e-12, (i, fd, ad)


def test_smooth_close_to_hard(fix):
    cfg, th, x, dt = T.setup(fix, "base")
    hs = M.cloudsc(cfg, th, x, dt, hard=True)
    ss = M.cloudsc(cfg, th, x, dt, hard=False)
    # loose: the smooth forms differ from the oracle only inside 0.01-wide windows
    assert np.max(np.abs(np.asarray(ss.fplsn[:, -1] - hs.fplsn[:, -1]))) <= 0.1 * np.max(np.abs(np.asarray(hs.fplsn[:, -1]))) + 1e-12
