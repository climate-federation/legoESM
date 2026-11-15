"""Round-2 candidate C1-mem-R (r2_c1mem_r): x64 conservation, memory, Lindzen bound,
moist regime, CloudBench surface-law protocol and gradients."""
from __future__ import annotations

import importlib.util
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest
from jax.test_util import check_grads
from legoesm.atmosphere.les_suite import candidate as cmod
from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
from legoesm.atmosphere.physics._shared import exner_function

from legoesm import constants

jax.config.update("jax_enable_x64", True)

_PATH = (Path(__file__).resolve().parents[3] / "scripts" / "validate" / "les_suite"
         / "cosci_candidates" / "round2" / "r2_c1mem_r.py")
_UW, _VW = -0.09, 0.02
N = 24


def _module():
    spec = importlib.util.spec_from_file_location("_r2_c1mem_r_test", _PATH)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def _column(uw_s=jnp.nan, vw_s=jnp.nan, z0_m=0.1, dt=600.0):
    """Well-mixed moist layer (cloud above its LCL) to 1300 m, inversion to 1500 m,
    stable free troposphere above; sheared winds; heated, moistened surface."""
    z_half = jnp.linspace(0.0, 3000.0, N + 1)
    z_full = 0.5 * (z_half[1:] + z_half[:-1])
    rho_full = 1.2 * jnp.exp(-z_full / 8000.0)
    p_half = 1.0e5 * jnp.exp(-z_half / 8000.0)
    p_full = 1.0e5 * jnp.exp(-z_full / 8000.0)
    exner = exner_function(p_full)
    theta_l = (298.0 + 0.04 * jnp.clip(z_full - 1300.0, 0.0, 200.0)
               + 0.004 * jnp.maximum(z_full - 1500.0, 0.0))
    q_t = jnp.where(z_full < 1300.0, 0.016, 0.006 - 1e-6 * (z_full - 1300.0))
    theta, q_v, q_c = saturation_adjust(theta_l, q_t, exner, p_full)
    return cmod.column_inputs(
        z_full=z_full, z_half=z_half, rho_full=rho_full,
        rho_half=cmod.half_level_average(rho_full, rho_full[0]),
        mass=(p_half[:-1] - p_half[1:]) / constants.g, exner=exner,
        exner_half=exner_function(p_half), p_full=p_full, p_half=p_half,
        theta_l=theta_l, q_t=q_t, u=5.0 + 0.003 * z_full, v=2.0 * jnp.sin(z_full / 500.0),
        w_theta_s=0.06, w_qt_s=5e-5, uw_s=uw_s, vw_s=vw_s, dt=dt,
        temp=theta * exner, q_v=q_v, q_c=q_c, z0_m=z0_m, z0h_m=z0_m, sfc_drag_most=1.0)


def _params(mod, **over):
    p = {k: jnp.asarray(float(v[0])) for k, v in mod.PARAMS.items()}
    p.update({k: jnp.asarray(v) for k, v in over.items()})
    return p


def _warm_aux(mod, p, inp, frac=0.5):
    """aux after one call, with the memory M set off its target (memory active)."""
    d = mod.closure(p, inp, mod.init_aux(inp))
    return dict(d["aux"], M=frac * d["M_star"])


def _leaves(tree):
    return [np.asarray(x) for x in jax.tree_util.tree_leaves(tree)]


# ---------------------------------------------------------------- conservation (x64)

@pytest.mark.parametrize("law", [False, True])
def test_returned_flux_reproduces_the_implicit_solve(law):
    """X + dt*tendency(returned flux) == the solved state for theta_l (enthalpy form,
    the harness' scored heat form), q_t, u, v: the flux the harness applies is the
    flux the solve applied (incl. implicit upwind MF and the implicit surface drag)."""
    mod = _module()
    inp = _column(_UW, _VW) if law else _column()
    p = _params(mod)
    d = mod.closure(p, inp, _warm_aux(mod, p, inp))
    (wt, wq), (uw, vw) = d["flux_scalar"], d["flux_momentum"]
    tend = [cmod.heat_flux_to_theta_tendency(wt, inp, "enthalpy"),
            cmod.flux_to_tendency(wq, inp.rho_half, inp.mass),
            cmod.flux_to_tendency(uw, inp.rho_half, inp.mass),
            cmod.flux_to_tendency(vw, inp.rho_half, inp.mass)]
    for i, (x0, td) in enumerate(zip([inp.theta_l, inp.q_t, inp.u, inp.v], tend)):
        got = np.asarray(x0 + inp.dt * td)
        ref = np.asarray(d["X_new"][i])
        scale = float(jnp.max(jnp.abs(x0)))
        np.testing.assert_allclose(got, ref, rtol=0.0, atol=1e-13 * scale)


@pytest.mark.parametrize("law", [False, True])
def test_column_budget_closes(law):
    """sum m w (X_new - X) = dt rho_0 w_0 F(0): surface flux in, F(n) = 0, for all four
    fields (theta_l in enthalpy weights m Pi, Pi_half); scalar F(0) is the prescribed
    flux bit for bit; momentum F(0) is the passed stress (law column) or the applied
    implicit drag -C U_1^{n+1} (LES column)."""
    mod = _module()
    inp = _column(_UW, _VW) if law else _column()
    p = _params(mod)
    d = mod.closure(p, inp, _warm_aux(mod, p, inp))
    (wt, wq), (uw, vw) = d["flux_scalar"], d["flux_momentum"]
    assert float(wt[0]) == float(inp.w_theta_s) and float(wq[0]) == float(inp.w_qt_s)
    for f in (wt, wq, uw, vw):
        assert float(f[-1]) == 0.0
    if law:
        assert (float(uw[0]), float(vw[0])) == (_UW, _VW)
        assert float(d["C_drag"]) == 0.0
    else:
        assert float(d["C_drag"]) > 0.0
        np.testing.assert_allclose(float(uw[0]), -float(d["C_drag"] * d["X_new"][2, 0]),
                                   rtol=1e-14)
    w_c = [inp.exner, 1.0, 1.0, 1.0]
    w_0 = [inp.exner_half[0], 1.0, 1.0, 1.0]
    for i, (x0, f) in enumerate(zip([inp.theta_l, inp.q_t, inp.u, inp.v], [wt, wq, uw, vw])):
        lhs = float(jnp.sum(inp.mass * w_c[i] * (d["X_new"][i] - x0)))
        rhs = float(inp.dt * inp.rho_half[0] * w_0[i] * f[0])
        roundoff = 1e-13 * float(jnp.sum(inp.mass * w_c[i] * jnp.abs(x0)))
        assert abs(lhs - rhs) <= roundoff + 1e-12 * abs(rhs), (i, lhs, rhs)


def test_harness_total_energy_budget_closes():
    """Harness enthalpy + KE budget with KE dissipation returned as heat (scored setting).
    This is the harness identity on the returned fluxes (it closes for any finite flux
    profile); it guards finiteness and the contract shapes, not a candidate piece."""
    mod = _module()
    inp = _column()
    p = _params(mod)
    fa, fb, _ = mod.fluxes(p, inp, _warm_aux(mod, p, inp))
    tend = cmod.candidate_tendencies("both", fa, fb, inp, heat_form="enthalpy",
                                     ke_heating=True)
    b = cmod.column_energy_budget(tend, fa, fb, inp, "both")
    assert abs(float(b.residual)) <= 1e-10 * abs(float(b.boundary_input)) + 1e-9


def test_wave_kernel_is_zero_sum_and_active():
    """The wave flux has F_W(0) = 0 and F_W(n) = 0 exactly, so it only relocates
    momentum; non-vacuous: the kernel carries flux on this column. (The column-sum
    line is the flux-form telescoping identity given those end values; the end-value
    and activity checks are the candidate-specific part.)"""
    mod = _module()
    inp = _column()
    p = _params(mod)
    d = mod.closure(p, inp, _warm_aux(mod, p, inp))
    fw = np.asarray(d["F_wave"])
    assert np.all(fw[0] == 0.0) and np.all(fw[-1] == 0.0)
    assert np.max(np.abs(fw)) > 1e-8
    col = cmod.flux_to_tendency(jnp.asarray(fw[:, 0]), inp.rho_half, inp.mass)
    assert abs(float(jnp.sum(inp.mass * col))) <= 1e-14 * float(
        jnp.sum(inp.mass * jnp.abs(col))) + 1e-300


# ---------------------------------------------------------------- Lindzen saturation

def test_lindzen_attenuation_bounds_and_never_reemits():
    """Synthetic paths: capped |B P| <= cap (to the smoothmin's 1.0 upper factor),
    non-increasing along each path, and no attenuation where |B| << cap."""
    mod = _module()
    n_j, n_f = 3, 8
    jj = jnp.arange(n_j)[:, None]
    ff = jnp.arange(n_f)[None, :]
    up = ff > jj + 2
    # |B| decays away from each launch in both directions (as the WKB kernel does)
    log_B = -0.1 * jnp.abs(ff - (jj + 2.5)).astype(float)
    has = jnp.ones((n_j, n_f), bool)
    log_cap = jnp.broadcast_to(jnp.log(jnp.array([5, 5, 5, 5, 0.1, 5, 5, 5.0])),
                               (n_j, n_f))
    log_P = mod._lindzen_log_attenuation(log_B, has, log_cap, up)
    mag = np.asarray(jnp.exp(log_B + log_P))
    cap = np.asarray(jnp.exp(log_cap))
    assert np.all(mag <= cap * (1.0 + 1e-12))
    assert np.all(np.asarray(log_P) <= 0.0)
    for j in range(n_j):
        upf = np.asarray(up[j])
        path = mag[j, upf]
        assert np.all(np.diff(path) <= 1e-15)              # upward, no re-emission
        dn = mag[j, ~upf][::-1]
        assert np.all(np.diff(dn) <= 1e-15)                # downward (from launch)
    # saturated at face 4 for the up-going launch 0: above it stays <= 0.1
    assert mag[0, 5] <= 0.1 and mag[0, 3] > 0.9 * np.exp(float(log_B[0, 3]))


def test_lindzen_attenuation_does_not_compound_with_the_number_of_faces():
    """|B| = cap on every face of a long path: every face gets the pointwise smooth
    cap 2^(-1/4) (the compounding form gave (1 + q)^(-1/4) after q faces)."""
    mod = _module()
    n_f = 16
    up = jnp.arange(n_f)[None, :] > 0
    zero = jnp.zeros((1, n_f))
    log_P = mod._lindzen_log_attenuation(zero, jnp.ones((1, n_f), bool), zero, up)
    np.testing.assert_allclose(np.exp(np.asarray(log_P)[0, 1:]), 2.0 ** -0.25,
                               rtol=1e-14)


def test_lindzen_log_norm_survives_fp32_underflow():
    """A nonzero wave flux whose squared norm underflows in fp32 keeps its log norm
    (so its Lindzen cap still applies); B = 0 is flagged."""
    mod = _module()
    B = jnp.asarray([[2e-23, 0.0], [2e-23, 1e-23], [0.0, 0.0]], jnp.float32)
    assert float(jnp.sum(B[0] ** 2)) == 0.0                    # the underflow is real
    log_B, has = mod._log_norm(B)
    np.testing.assert_allclose(np.asarray(log_B[:2]),
                               [np.log(2e-23), np.log(np.sqrt(5.0) * 1e-23)], rtol=1e-6)
    assert list(np.asarray(has)) == [True, True, False]
    # gradient d log|B|/dB = B/|B|^2, finite in fp32 for tiny |B| (no b_max^-2 term)
    g = jax.grad(lambda b: jnp.sum(mod._log_norm(b)[0]))(B)
    ref = np.asarray([[1.0 / 2e-23, 0.0], [2e-23 / 5e-46, 1e-23 / 5e-46], [0.0, 0.0]])
    assert np.all(np.isfinite(np.asarray(g)))
    np.testing.assert_allclose(np.asarray(g), ref, rtol=1e-5)


def test_lindzen_bound_holds_in_the_column():
    mod = _module()
    inp = _column()
    p = _params(mod, phi0=0.6, lambda_w=500.0)
    d = mod.closure(p, inp, _warm_aux(mod, p, inp))
    mag2 = np.sum(np.asarray(d["F_wave_launch"]) ** 2, axis=-1)
    on = mag2 > 0.0
    log_mag = 0.5 * np.log(np.where(on, mag2, 1.0))
    assert np.all(log_mag[on] <= np.asarray(d["log_cap"])[on] + 1e-12)
    # the cap's constant, recomputed: rho k U_+^3/(2 N_+), k = 2 pi/lambda_w
    k = 2.0 * np.pi / 500.0
    ref = (np.log(np.asarray(inp.rho_half) * k / 2.0)[None, :] - np.asarray(d["logN_face"])[None, :]
           + 3.0 * np.log(np.asarray(d["U_plus_face"])))
    np.testing.assert_allclose(np.asarray(d["log_cap"]), ref, rtol=1e-13)
    assert float(jnp.min(d["wave_log_P"])) < -0.1          # non-vacuous: the cap binds


# ---------------------------------------------------------------- memory

def test_memory_is_the_exact_exponential_and_dt_consistent():
    """On a law column M* does not depend on the TKE, so two dt/2 calls with frozen
    M*, tau equal one dt call (exact kernel; a backward-Euler filter fails this), and
    M is a convex combination of M^n and M*."""
    mod = _module()
    p = _params(mod)
    full = _column(_UW, _VW, dt=600.0)
    half = _column(_UW, _VW, dt=300.0)
    aux = _warm_aux(mod, p, full)
    d1 = mod.closure(p, full, aux)
    dh = mod.closure(p, half, aux)
    dh2 = mod.closure(p, half, dh["aux"])
    np.testing.assert_allclose(np.asarray(dh["M_star"]), np.asarray(d1["M_star"]),
                               rtol=0, atol=0)
    np.testing.assert_allclose(np.asarray(dh2["M"]), np.asarray(d1["M"]), rtol=1e-13,
                               atol=1e-16)
    lo = np.minimum(aux["M"], d1["M_star"])
    hi = np.maximum(aux["M"], d1["M_star"])
    M = np.asarray(d1["M"])
    assert np.all(M >= lo - 1e-16) and np.all(M <= hi + 1e-16)
    assert M[0] == 0.0 and M[-1] == 0.0 and np.max(M) > 0.0
    keep = np.exp(-600.0 / float(d1["tau"]))
    np.testing.assert_allclose(M - np.asarray(d1["M_star"]),
                               (np.asarray(aux["M"]) - np.asarray(d1["M_star"])) * keep,
                               rtol=1e-12, atol=1e-16)


def test_memory_free_control_is_safe(monkeypatch):
    """c_tau = 0 with tau_min = 0 (the meta-review's memory-free control): M = M*,
    finite gradient."""
    mod = _module()
    monkeypatch.setattr(mod, "_TAU_MIN_S", 0.0)
    inp = _column()
    p = _params(mod, c_tau=0.0)
    aux = _warm_aux(mod, p, inp)
    d = mod.closure(p, inp, aux)
    assert float(d["tau"]) == 0.0
    np.testing.assert_array_equal(np.asarray(d["M"]), np.asarray(d["M_star"]))
    g = jax.grad(lambda q: jnp.sum(mod.closure(q, inp, aux)["M"]))(p)
    assert all(np.isfinite(float(v)) for v in g.values())


# ---------------------------------------------------------------- moist regime, anchor

def test_moist_n_eff_is_zero_in_well_mixed_cloud_and_dry_below_cloud():
    """In the well-mixed (uniform theta_l, q_t) layer N_eff^2 = 0 whatever f_c, while
    the grid-mean dry N^2 (from theta_v) is positive inside the cloud."""
    mod = _module()
    inp = _column()
    d = mod.closure(_params(mod), inp, mod.init_aux(inp))
    zf = np.asarray(inp.z_half[1:-1])
    mixed = zf < 1250.0
    assert np.all(np.asarray(d["N2_eff"])[mixed] == 0.0)
    theta = inp.temp / inp.exner
    thv = theta * (1.0 + (1.0 / constants.epsilon - 1.0) * inp.q_v - inp.q_c)
    n2_dry = np.asarray(constants.g / thv[0] * jnp.diff(thv) / jnp.diff(inp.z_full))
    fc = np.asarray(d["f_c"])
    cloudy = mixed & (fc > 0.9)
    assert cloudy.any() and np.all(n2_dry[cloudy] > 0.0)
    assert (mixed & (fc < 0.1)).any()


def _theta_v_of(thl, qt, exner, p):
    th, qv, qc = saturation_adjust(thl, qt, exner, p)
    return th * (1.0 + (1.0 / constants.epsilon - 1.0) * qv - qc)


def test_buoyancy_coefficients_match_the_model_saturation_adjustment():
    """A_th, A_q at a fully cloudy and a fully clear face against central differences
    of theta_v(theta_l, q_t) from the model's own saturation adjustment at the face
    state (rtol 2e-2, vote V17: linearisation at face-mean T)."""
    mod = _module()
    inp = _column()
    d = mod.closure(_params(mod), inp, mod.init_aux(inp))
    fc = np.asarray(d["f_c"])
    picks = {"cloudy": int(np.argmax(fc > 0.999)), "clear": int(np.argmax(fc < 1e-3))}
    assert fc[picks["cloudy"]] > 0.999 and fc[picks["clear"]] < 1e-3
    for _name, i in picks.items():
        thl = 0.5 * (inp.theta_l[i] + inp.theta_l[i + 1])
        qt = 0.5 * (inp.q_t[i] + inp.q_t[i + 1])
        ex, pp = inp.exner_half[i + 1], inp.p_half[i + 1]
        dth, dq = 1e-3, 1e-6
        a_th = (_theta_v_of(thl + dth, qt, ex, pp) - _theta_v_of(thl - dth, qt, ex, pp)) / (2 * dth)
        a_q = (_theta_v_of(thl, qt + dq, ex, pp) - _theta_v_of(thl, qt - dq, ex, pp)) / (2 * dq)
        np.testing.assert_allclose(float(d["A_th"][i]), float(a_th), rtol=2e-2)
        np.testing.assert_allclose(float(d["A_q"][i]), float(a_q), rtol=2e-2)


def test_c_n_anchors_the_zless_balance_at_ri_eq():
    mod = _module()
    p = _params(mod)
    inp = _column()
    c_N = float(mod.closure(p, inp, mod.init_aux(inp))["c_N"])
    ri, ck = float(p["Ri_eq"]), float(p["c_k"])
    pr = float(p["Pr_0"]) + ri / float(p["R_finf"])
    np.testing.assert_allclose(c_N ** 2 / ri * (1.0 - ri / pr), ck ** 2, rtol=1e-14)


# ---------------------------------------------------------------- CloudBench protocol

def _run(inp, patch=None):
    cand = cmod.load_candidate(_PATH)
    return cand.fluxes(cand.default_params(), inp, cand.init_aux(inp))


def test_candidate_uses_the_passed_stress():
    _fa, (uw, vw), _aux = _run(_column(_UW, _VW))
    assert (float(uw[0]), float(vw[0])) == (_UW, _VW)


def test_law_column_ignores_the_own_surface_layer():
    """On a law column nothing depends on what enters only the own law (roughness;
    beta_g enters only the own gust speed): fluxes and aux bit-identical."""
    a = _leaves(_run(_column(_UW, _VW, z0_m=1e-3)))
    b = _leaves(_run(_column(_UW, _VW, z0_m=0.1)))
    for x, y in zip(a, b, strict=True):
        np.testing.assert_array_equal(x, y)
    mod = _module()
    inp = _column(_UW, _VW)
    l1 = _leaves(mod.fluxes(_params(mod, beta_g=0.3), inp, mod.init_aux(inp)))
    l2 = _leaves(mod.fluxes(_params(mod, beta_g=1.5), inp, mod.init_aux(inp)))
    for x, y in zip(l1, l2, strict=True):
        np.testing.assert_array_equal(x, y)


def test_nan_stress_keeps_the_own_law_bit_identically(monkeypatch):
    inp = _column()
    got = _leaves(_run(inp))
    monkeypatch.setattr(cmod, "surface_stress", lambda _x, uw, vw, us: (uw, vw, us))
    ref = _leaves(_run(inp))
    for x, y in zip(got, ref, strict=True):
        np.testing.assert_array_equal(x, y)
    _fa, (uw, _vw), _aux = _run(inp)
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW)
    # control: the patch bites (with it, a passed stress is no longer used)
    _fa, (uw, _vw), _aux = _run(_column(_UW, _VW))
    assert abs(float(uw[0]) - _UW) > 1e-3 * abs(_UW)


# ---------------------------------------------------------------- gradients

def _loss(mod, inp, aux):
    def f(p):
        (wt, wq), (uw, vw), a = mod.fluxes(p, inp, aux)
        return (jnp.sum(wt[1:] ** 2) * 1e2 + jnp.sum(wq[1:] ** 2) * 1e8
                + jnp.sum(uw ** 2 + vw ** 2) + jnp.sum(a["e"] ** 2) + jnp.sum(a["M"] ** 2))
    return f


def test_every_parameter_carries_gradient_on_an_les_column():
    mod = _module()
    inp = _column()
    p = _params(mod)
    g = jax.grad(_loss(mod, inp, _warm_aux(mod, p, inp)))(p)
    bad = {k: float(v) for k, v in g.items() if not (np.isfinite(float(v)) and float(v) != 0.0)}
    assert not bad, bad


def test_law_column_with_nan_roughness_and_calm_cooled_surface_is_differentiable():
    """Contract allows z0 = NaN on a law column; a zero passed stress with strong
    surface cooling makes sigma_w = 0. Values and gradients stay finite."""
    mod = _module()
    for inp in (_column(_UW, _VW, z0_m=jnp.nan), _column(_UW, _VW, z0_m=62.5),
                _column(0.0, 0.0)._replace(w_theta_s=jnp.asarray(-0.05),
                                           w_qt_s=jnp.asarray(0.0))):
        p = _params(mod)
        d = mod.closure(p, inp, mod.init_aux(inp))
        assert all(np.all(np.isfinite(x)) for x in _leaves(d["flux_momentum"]))
        g = jax.grad(_loss(mod, inp, _warm_aux(mod, p, inp)))(p)
        assert all(np.isfinite(float(v)) for v in g.values()), g
    assert float(d["sigma_w"]) == 0.0           # the second column hits the singular case


def test_gradient_through_a_law_column_is_finite_and_beta_g_inert():
    mod = _module()
    inp = _column(_UW, _VW)
    p = _params(mod)
    g = jax.grad(_loss(mod, inp, _warm_aux(mod, p, inp)))(p)
    assert all(np.isfinite(float(v)) for v in g.values())
    assert float(g["beta_g"]) == 0.0


@pytest.mark.parametrize("name", ["Ri_eq", "c_sig", "c_tau", "phi0"])
def test_check_grads_through_the_smooth_switches(name):
    mod = _module()
    inp = _column()
    p = _params(mod)
    aux = _warm_aux(mod, p, inp)
    loss = _loss(mod, inp, aux)

    def f(x):
        return loss(dict(p, **{name: x}))
    check_grads(f, (p[name],), order=1, modes=["rev"], eps=1e-6, atol=1e-4, rtol=1e-4)
