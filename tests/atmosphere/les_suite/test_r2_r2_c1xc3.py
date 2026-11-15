"""Round-2 offspring r2_c1xc3 (memory EDMF-TKE + buoyancy-barrier return downdrafts):
x64 column budgets, air-mass consistency of the return stream, the exact memory
kernel, the CloudBench surface-law protocol and finite gradients."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from legoesm.atmosphere.les_suite import candidate as cmod
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite"
         / "cosci_candidates" / "round2" / "r2_c1xc3.py")
_UW, _VW = -0.09, 0.02
_TOL = 1e-12          # relative round-off tolerance (x64)


def _module():
    spec = importlib.util.spec_from_file_location("_r2_c1xc3_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _column(kind="dry", uw_s=_UW, vw_s=_VW, z0_m=1e-3, w_qt_s=None, q_uniform=None,
            u_uniform=None):
    """'dry': the round-1 surface-law test column (16 levels, 1600 m, inversion at
    600 m). 'moist': 24 levels to 3000 m, near-saturated mixed layer, saturating plume."""
    if kind == "dry":
        n, top = 16, 1600.0
    else:
        n, top = 24, 3000.0
    z_half = jnp.linspace(0.0, top, n + 1)
    z_full = 0.5 * (z_half[1:] + z_half[:-1])
    rho_full = 1.2 * jnp.exp(-z_full / 8000.0)
    p_half = 1.0e5 * jnp.exp(-z_half / 8000.0)
    p_full = 1.0e5 * jnp.exp(-z_full / 8000.0)
    exner = exner_function(p_full)
    if kind == "dry":
        thl = 300.0 + 0.004 * jnp.maximum(z_full - 600.0, 0.0)
        qt = 0.01 - 2e-6 * z_full
        u = 5.0 + 0.002 * z_full
        v = jnp.sin(z_full / 300.0)
        wth, wq = 0.05, 1e-4
    else:
        # BOMEX-like (Siebesma et al. 2003 initial profiles, piecewise linear)
        zk = jnp.asarray([0.0, 520.0, 1480.0, 2000.0, 3000.0])
        thl = jnp.interp(z_full, zk, jnp.asarray([298.7, 298.7, 302.4, 308.2, 311.85]))
        qt = 1e-3 * jnp.interp(z_full, zk, jnp.asarray([17.0, 16.3, 10.7, 4.2, 3.0]))
        u = -8.75 + 1.8e-3 * jnp.maximum(z_full - 700.0, 0.0)
        v = 0.5 * jnp.sin(z_full / 700.0)
        wth, wq = 8e-3, 5.2e-5
    if q_uniform is not None:
        qt = jnp.full_like(z_full, q_uniform)
    if u_uniform is not None:
        u = jnp.full_like(z_full, u_uniform[0])
        v = jnp.full_like(z_full, u_uniform[1])
    theta, q_v, q_c = saturation_adjust(thl, qt, exner, p_full)
    return cmod.column_inputs(
        z_full=z_full, z_half=z_half, rho_full=rho_full,
        rho_half=cmod.half_level_average(rho_full, rho_full[0]),
        mass=(p_half[:-1] - p_half[1:]) / constants.g, exner=exner,
        exner_half=exner_function(p_half), p_full=p_full, p_half=p_half,
        theta_l=thl, q_t=qt, u=u, v=v, w_theta_s=wth,
        w_qt_s=wq if w_qt_s is None else w_qt_s, uw_s=uw_s, vw_s=vw_s, dt=600.0,
        temp=theta * exner, q_v=q_v, q_c=q_c, z0_m=z0_m, z0h_m=z0_m,
        sfc_drag_most=1.0)


def _cand():
    return cmod.load_candidate(_PATH)


def _advance(inp, fa, fb):
    """The column after the harness applies the fluxes (enthalpy form, no KE heat),
    with theta, q_v, q_c re-derived by saturation adjustment."""
    t = cmod.candidate_tendencies("both", fa, fb, inp, heat_form="enthalpy",
                                  ke_heating=False)
    thl, qt = inp.theta_l + inp.dt * t.dtheta, inp.q_t + inp.dt * t.dqt
    theta, q_v, q_c = saturation_adjust(thl, qt, inp.exner, inp.p_full)
    return inp._replace(theta_l=thl, q_t=qt, u=inp.u + inp.dt * t.du,
                        v=inp.v + inp.dt * t.dv, temp=theta * inp.exner, q_v=q_v, q_c=q_c)


def _two_steps(cand, params, inp, return_inputs=False):
    """Two calls: the second on the column the first one advanced, with its aux
    (exercises the memory, which is inert on a frozen column)."""
    fa, fb, aux1 = cand.fluxes(params, inp, cand.init_aux(inp))
    inp2 = _advance(inp, fa, fb)
    out = cand.fluxes(params, inp2, aux1)
    return (out, inp2) if return_inputs else out


COLUMNS = {"dry_law": lambda: _column("dry"),
           "dry_les": lambda: _column("dry", uw_s=jnp.nan, vw_s=jnp.nan),
           "moist_les": lambda: _column("moist", uw_s=jnp.nan, vw_s=jnp.nan, z0_m=1e-4),
           "moist_law": lambda: _column("moist", z0_m=1e-4)}


@pytest.mark.parametrize("col", list(COLUMNS))
def test_column_budgets_close(col):
    """in - out - d(storage) = 0 for theta_l (enthalpy form), q_t, u, v and total
    enthalpy + latent + KE, through the harness flux->tendency conversion."""
    cand = _cand()
    (fa, fb, _), inp = _two_steps(cand, cand.default_params(), COLUMNS[col](),
                                  return_inputs=True)
    (wt, wq), (uw, vw) = fa, fb
    assert float(wt[0]) == float(inp.w_theta_s) and float(wq[0]) == float(inp.w_qt_s)
    for f in (wt, wq, uw, vw):
        assert float(f[-1]) == 0.0 and np.all(np.isfinite(np.asarray(f)))
    td = cmod.candidate_tendencies("both", fa, fb, inp, heat_form="enthalpy",
                                   ke_heating=True)
    tn = cmod.candidate_tendencies("both", fa, fb, inp, heat_form="enthalpy",
                                   ke_heating=False)
    m, rh = inp.mass, inp.rho_half
    checks = [(m * inp.exner * tn.dtheta, rh[0] * inp.exner_half[0] * wt[0]),
              (m * tn.dqt, rh[0] * wq[0]),
              (m * tn.du, rh[0] * uw[0]),
              (m * tn.dv, rh[0] * vw[0])]
    for dstore, inflow in checks:   # top outflow is exactly 0 (asserted above)
        scale = float(jnp.sum(jnp.abs(dstore))) + abs(float(inflow))
        assert abs(float(jnp.sum(dstore) - inflow)) <= _TOL * scale
    bud = cmod.column_energy_budget(td, fa, fb, inp, "both")
    assert abs(float(bud.residual)) <= _TOL * (abs(float(bud.rate))
                                              + abs(float(bud.boundary_input)) + 1.0)


@pytest.mark.parametrize("col", list(COLUMNS))
def test_returned_flux_reproduces_the_solved_state(col):
    """The harness' forward-Euler flux-form step (heat_form='enthalpy') lands on the
    candidate's implicit solution: the flux reported is the flux applied."""
    mod, cand = _module(), _cand()
    inp = COLUMNS[col]()
    p = cand.default_params()
    fa, fb, _aux, st = mod.fluxes_and_state(p, inp, cand.init_aux(inp))
    tn = cmod.candidate_tendencies("both", fa, fb, inp, heat_form="enthalpy",
                                   ke_heating=False)
    for name, d in (("theta_l", tn.dtheta), ("q_t", tn.dqt), ("u", tn.du), ("v", tn.dv)):
        old = getattr(inp, name)
        new = old + inp.dt * d
        scale = float(jnp.max(jnp.abs(st[name] - old))) + 1e-300
        np.testing.assert_allclose(np.asarray(new), np.asarray(st[name]),
                                   rtol=0.0, atol=1e-9 * scale + _TOL * float(jnp.max(jnp.abs(old))))


@pytest.mark.parametrize("kind", ["dry", "moist"])
def test_uniform_fields_have_zero_flux(kind):
    """Air-mass consistency: updraft Mm, return R and subsidence D = Mm - R carry no
    net air across any face, and the Gregory return term is zero-sum, so a uniform
    q_t (zero surface moisture flux) and uniform winds (zero passed stress) have
    exactly zero flux everywhere while the plume and the return stream are active."""
    mod, cand = _module(), _cand()
    inp = _column(kind, uw_s=0.0, vw_s=0.0, w_qt_s=0.0, q_uniform=0.008,
                  u_uniform=(4.0, -1.0), z0_m=1e-3)
    p = cand.default_params()
    k = mod.kernel(p, inp, cand.init_aux(inp))
    assert float(jnp.max(k["R"])) > 1e-2 * float(jnp.max(k["Mm"])) > 0.0   # non-vacuous
    (_wt, wq), (uw, vw), _ = cand.fluxes(p, inp, cand.init_aux(inp))
    mmax = float(jnp.max(k["Mm"]) / inp.rho_half[0])
    assert float(jnp.max(jnp.abs(wq))) <= _TOL * 10.0 * mmax * 0.008
    assert float(jnp.max(jnp.abs(uw))) <= _TOL * 10.0 * mmax * 4.0
    assert float(jnp.max(jnp.abs(vw))) <= _TOL * 10.0 * mmax * 4.0


@pytest.mark.parametrize("col", list(COLUMNS))
def test_return_stream_bounds(col):
    mod, cand = _module(), _cand()
    inp = COLUMNS[col]()
    p = cand.default_params()
    k = mod.kernel(p, inp, cand.init_aux(inp))
    A, R, D, Mm = (np.asarray(k[x]) for x in ("A", "R", "D", "Mm"))
    assert np.all(A >= 0.0) and np.all(A <= 1.0)
    assert np.all(Mm >= 0.0) and float(Mm[0]) == 0.0 and float(Mm[-1]) == 0.0
    assert np.all(R <= float(p["alpha_s"]) * Mm[1:-1] * (1 + 1e-14))
    assert np.all(D >= -1e-14 * Mm[1:-1].max())


def test_memory_is_the_exact_exponential_kernel():
    mod, cand = _module(), _cand()
    inp = COLUMNS["moist_les"]()
    p = cand.default_params()
    aux0 = cand.init_aux(inp)
    k0 = mod.kernel(p, inp, aux0)
    np.testing.assert_array_equal(np.asarray(k0["M"]), np.asarray(k0["M_star"]))  # cold start
    prev = 0.5 * k0["M_star"]
    aux1 = dict(cand.fluxes(p, inp, aux0)[2], M=prev)
    k1 = mod.kernel(p, inp, aux1)
    dz = inp.z_half[1:] - inp.z_half[:-1]
    t_top = jnp.sum(k1["Pi_f"][1:] * dz / jnp.sqrt(k1["w2_f"][1:] + mod._W_D ** 2))
    np.testing.assert_allclose(float(k1["tau"]),
                               float(p["c_tau"] * t_top + mod._TAU_MIN), rtol=1e-14)
    assert float(t_top) > 0.0
    want = k1["M_star"] + (prev - k1["M_star"]) * jnp.exp(-inp.dt / k1["tau"])
    np.testing.assert_allclose(np.asarray(k1["M"]), np.asarray(want), rtol=1e-14, atol=0.0)
    lo, hi = jnp.minimum(prev, k1["M_star"]), jnp.maximum(prev, k1["M_star"])
    assert bool(jnp.all((k1["M"] >= lo) & (k1["M"] <= hi)))
    assert float(jnp.max(jnp.abs(k1["M"] - k1["M_star"]))) > 0.0


def test_regimes_are_exercised():
    """The test columns reach the parts the gradient test needs (non-vacuity)."""
    mod, cand = _module(), _cand()
    p = cand.default_params()
    km = mod.kernel(p, COLUMNS["moist_les"](), cand.init_aux(COLUMNS["moist_les"]()))
    live = np.asarray(km["M_star"]) > 0.0
    assert np.any(np.asarray(km["qcu_f"])[live] > 1e-5)          # condensing plume
    assert float(jnp.max(km["f_c"])) > 0.05                       # partial cloudiness
    kd = mod.kernel(p, COLUMNS["dry_law"](), cand.init_aux(COLUMNS["dry_law"]()))
    assert float(jnp.max(kd["N2f"])) > 0.0                        # stable faces


@pytest.mark.parametrize("col", list(COLUMNS))
def test_rain_is_passive_in_the_thermodynamics(col):
    """q_t includes rain (contract). A uniform rain field added to q_t (q_v, q_c,
    temp unchanged) must change nothing: the thermodynamics see q_v + q_c only, and
    a uniform shift of q_t has zero flux."""
    cand = _cand()
    p = cand.default_params()
    inp = COLUMNS[col]()
    wet = inp._replace(q_t=inp.q_t + 0.004)
    a = _leaves(cand.fluxes(p, inp, cand.init_aux(inp)))
    b = _leaves(cand.fluxes(p, wet, cand.init_aux(wet)))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_allclose(x, y, rtol=1e-9, atol=1e-12 * (np.max(np.abs(x)) + 1e-300))


@pytest.mark.parametrize("col", ["dry_les", "moist_law"])
@pytest.mark.parametrize("profile", ["cell0", "random"])
def test_nonuniform_rain_leaves_plume_thermodynamics_unchanged(col, profile):
    """Rain added non-uniformly to q_t (q_v, q_c, temp unchanged) changes no plume
    or thermodynamic diagnostic: plume parcels saturate their own non-precipitating
    water, never entrained rain."""
    mod, cand = _module(), _cand()
    p = cand.default_params()
    inp = COLUMNS[col]()
    n = inp.q_t.shape[0]
    if profile == "cell0":
        q_r = jnp.zeros(n).at[0].set(0.004)
    else:
        q_r = 0.004 * jnp.asarray(np.random.default_rng(0).random(n))
    wet = inp._replace(q_t=inp.q_t + q_r)
    k0 = mod.kernel(p, inp, cand.init_aux(inp))
    k1 = mod.kernel(p, wet, cand.init_aux(wet))
    for name in ("w2_f", "Pi_f", "M_star", "qcu_f", "f_c", "a_th", "a_q", "N2f", "A",
                 "Bs", "bmat", "sigma_w", "Mm", "delta"):
        x, y = np.asarray(k0[name]), np.asarray(k1[name])
        np.testing.assert_allclose(y, x, rtol=1e-10, atol=1e-12 * (np.max(np.abs(x)) + 1e-300),
                                   err_msg=name)
    assert float(jnp.max(k0["M_star"])) > 0.0


@pytest.mark.parametrize("col", list(COLUMNS))
@pytest.mark.parametrize("profile", ["cell0", "random"])
def test_nonuniform_rain_leaves_tke_production_unchanged(col, profile):
    """Rain added non-uniformly to q_t (q_v, q_c, temp unchanged) leaves the new TKE
    unchanged: the buoyancy production's nonlocal water flux carries no rain."""
    mod, cand = _module(), _cand()
    p = cand.default_params()
    inp = COLUMNS[col]()
    n = inp.q_t.shape[0]
    if profile == "cell0":
        q_r = jnp.zeros(n).at[0].set(0.004)
    else:
        q_r = 0.004 * jnp.asarray(np.random.default_rng(1).random(n))
    wet = inp._replace(q_t=inp.q_t + q_r)
    e0 = np.asarray(mod.fluxes(p, inp, cand.init_aux(inp))[2]["e"])
    e1 = np.asarray(mod.fluxes(p, wet, cand.init_aux(wet))[2]["e"])
    assert np.max(np.abs(e0 - cand.init_aux(inp)["e"])) > 1e-6
    np.testing.assert_allclose(e1, e0, rtol=1e-10, atol=1e-14)


def test_law_column_with_missing_roughness_has_finite_gradients():
    """A law column may carry NaN roughness; the discarded own law must not put
    0*NaN into the gradient, and the result equals that with finite roughness."""
    cand = _cand()
    p0 = cand.default_params()
    nan_z0 = _column("dry", z0_m=jnp.nan)
    g = jax.grad(_loss(cand, nan_z0))(p0)
    assert all(np.isfinite(float(v)) for v in g.values())
    g_ref = jax.grad(_loss(cand, _column("dry", z0_m=1e-3)))(p0)
    for name in p0:
        assert float(g[name]) == float(g_ref[name]), name


@pytest.mark.parametrize("col", list(COLUMNS))
def test_total_water_stays_non_negative(col):
    """The M-matrix claim is not made (deviation 11); measured: q_t >= 0 after the
    implicit solve on every test column (nothing is clipped)."""
    mod, cand = _module(), _cand()
    inp = COLUMNS[col]()
    st = mod.fluxes_and_state(cand.default_params(), inp, cand.init_aux(inp))[3]
    assert float(jnp.min(st["q_t"])) >= 0.0


def test_calm_cooling_law_column_is_finite():
    """Zero passed stress (u* = 0) over a cooling surface: softplus(B_s/B_d)
    underflows, the plume is dead; values, aux and every gradient stay finite."""
    cand = _cand()
    inp = _column("dry", uw_s=0.0, vw_s=0.0, w_qt_s=0.0)._replace(
        w_theta_s=jnp.asarray(-0.05))
    p0 = cand.default_params()
    for leaf in _leaves(_two_steps(cand, p0, inp)):
        assert np.all(np.isfinite(leaf))
    g = jax.grad(_loss(cand, inp))(p0)
    assert all(np.isfinite(float(v)) for v in g.values())


# --- CloudBench surface-law protocol ------------------------------------------------

def test_candidate_uses_the_passed_stress():
    cand = _cand()
    for col in ("dry_law", "moist_law"):
        inp = COLUMNS[col]()
        _fa, (uw, vw), _ = _two_steps(cand, cand.default_params(), inp)
        np.testing.assert_allclose([float(uw[0]), float(vw[0])], [_UW, _VW],
                                   rtol=1e-12, atol=0.0)


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


@pytest.mark.parametrize("kind", ["dry", "moist"])
def test_law_column_ignores_the_own_surface_law(kind):
    """On a law column nothing depends on what enters only the own law: roughness
    and beta_g (gust speed, own drag)."""
    cand = _cand()
    p = cand.default_params()
    a = _leaves(_two_steps(cand, p, _column(kind, z0_m=1e-3)))
    b = _leaves(_two_steps(cand, p, _column(kind, z0_m=0.1)))
    c = _leaves(_two_steps(cand, dict(p, beta_g=jnp.asarray(0.4)), _column(kind, z0_m=1e-3)))
    for x, y, z in zip(a, b, c, strict=True):
        np.testing.assert_array_equal(x, y)
        np.testing.assert_array_equal(x, z)


@pytest.mark.parametrize("kind", ["dry", "moist"])
def test_nan_stress_keeps_the_own_law(kind, monkeypatch):
    inp = _column(kind, uw_s=jnp.nan, vw_s=jnp.nan)
    cand = _cand()
    got = _leaves(_two_steps(cand, cand.default_params(), inp))
    monkeypatch.setattr(cmod, "surface_stress", lambda _x, uw, vw, us: (uw, vw, us))
    cand_p = _cand()
    ref = _leaves(_two_steps(cand_p, cand_p.default_params(), inp))
    for x, y in zip(got, ref, strict=True):
        np.testing.assert_array_equal(x, y)
    _fa, (uw, _vw), _ = _two_steps(cand, cand.default_params(), inp)
    assert float(uw[0]) * float(inp.u[0]) < 0.0          # own drag opposes the wind
    # control: the patch bites (with it a passed stress is no longer used)
    _fa, (uw, _vw), _ = _two_steps(cand_p, cand_p.default_params(), _column(kind))
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW)
    # the own law responds to roughness on the NaN path
    rough = _leaves(_two_steps(cand, cand.default_params(),
                               _column(kind, uw_s=jnp.nan, vw_s=jnp.nan, z0_m=0.1)))
    assert any(not np.array_equal(x, y) for x, y in zip(got, rough, strict=True))


# --- gradients ----------------------------------------------------------------------

def _loss(cand, inp):
    def loss(p):
        (wt, wq), (uw, vw), aux = _two_steps(cand, p, inp)
        return (jnp.sum((wt / 0.05) ** 2) + jnp.sum((wq / 1e-4) ** 2)
                + jnp.sum((uw / 0.1) ** 2 + (vw / 0.1) ** 2) + jnp.sum(aux["e"]))
    return loss


@pytest.mark.parametrize("col", list(COLUMNS))
def test_gradient_is_finite_and_no_parameter_is_inert(col):
    cand = _cand()
    inp = COLUMNS[col]()
    p0 = cand.default_params()
    loss = _loss(cand, inp)
    L = float(loss(p0))
    g = jax.grad(loss)(p0)
    assert all(np.isfinite(float(v)) for v in g.values())
    law = col.endswith("_law")
    for name, v in g.items():
        if law and name == "beta_g":
            assert float(v) == 0.0          # own-law only: frozen out on CloudBench
            continue
        if col.startswith("dry") and name in ("eps_c", "delta_c"):
            continue                        # condensing-plume terms: moist columns only
        if name == "alpha_s":
            # KNOWN NEAR-INERT (open question, not hidden): on these dry-CBL and
            # BOMEX-like columns the air the plume detrains is positively buoyant when
            # brought down, the barrier closes (A ~ 1e-16 there) and R/Mm ~ 1e-10, so
            # d loss/d alpha_s is ~1e-11 relative. Only non-zero is asserted.
            assert float(v) != 0.0
            continue
        assert abs(float(v) * float(p0[name])) > 1e-8 * L, name


@pytest.mark.parametrize("col", list(COLUMNS))
def test_gradient_matches_finite_differences(col):
    cand = _cand()
    inp = COLUMNS[col]()
    p0 = cand.default_params()
    loss = _loss(cand, inp)
    g = jax.grad(loss)(p0)
    for name in p0:
        if name == "alpha_s" or (col.startswith("dry") and name in ("eps_c", "delta_c")):
            continue    # round-off-level gradients (near-inert, see the inert test)
        h = 1e-6 * abs(float(p0[name]))
        lp = float(loss(dict(p0, **{name: p0[name] + h})))
        lm = float(loss(dict(p0, **{name: p0[name] - h})))
        fd = (lp - lm) / (2 * h)
        np.testing.assert_allclose(float(g[name]), fd, rtol=1e-4,
                                   atol=1e-6 * abs(float(loss(p0))) / abs(float(p0[name])),
                                   err_msg=name)
