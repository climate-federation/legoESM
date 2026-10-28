"""Prognostic CLUBB core vs the Fortran CLUBB core (oracle fixture).

Fixture ``oracle_fixtures/clubb_core_cam.npz`` was produced by RUNNING the
Fortran ``advance_clubb_core`` of larson-group/clubb_release @
54403a04390fc4a0d4865357556e77c1ed6ad513 (constants patched to legoESM's;
CAM's cam_development pins clubb_release tag clubb_4ncar_20260109_ddf5110 and
CESM2.1's CLUBB_CESM vendor_clubb_r8099_n03 carries the same C7 damping, rtp2
clip, nu grid scaling and eddy-scalar rtm/thlm diffusion) with legoESM's CAM
parameters and flag table.  Generator: scripts/validate/oracle/clubb_core/
(build.sh; run.sh = dump_cam_settings.py + gen_clubb_core_fixture.py).

Seven benchmark columns (BOMEX, DYCOMS-II RF01, ARM, Wangara, M-PACE B, and
BOMEX/ARM on 100 m / 160 m grids) plus two dry-upper-troposphere variants
(the large-rtp2 clip binds there), each N consecutive 300 s Fortran steps
s0..sN.  Each of our steps starts from the oracle's s_n and is compared with
s_{n+1}.  The carried PDF-closure outputs of s_n were computed by the oracle at
the end of step n on that same state and boundary condition, so the
closure-placement difference (ours recomputes the closure at step start,
CAM's ipdf_call_placement = 2 carries it) does not enter; it is a separate,
open finding.  Lscale is our compute_mixing_length on s_n with the oracle's
inputs (thvm from the carried rcm), as the Fortran computes it inside the step.

Metric: per field, max |ours - oracle| / max |oracle| over the column and the
step (field-normalised, the tier-1 audit metric); bound 1e-10 in fp64 (user
decision 2026-10-19).  The smooth up2/vp2 cap does not bind on these
columns (measured: identical errors at cap width 10 and 0), so this is also the production configuration.  The l_implemented
fixture below checks the cap at its hard limit (width 0) and pins the
default width's per-case gap.

``oracle_fixtures/clubb_core_cam_implemented.npz`` (``run.sh implemented``):
the same generator with every recorded step run as CAM runs it,
``l_implemented = .true.`` (clubb_intr.F90:113 cam_cesm2_1_rel_60): BOMEX,
DYCOMS-II RF01 and M-PACE B carry subsidence ``wm``, ARM none, and
``bomex_bigup2`` (scaled up2/vp2 and winds) hits the up2/vp2 <= 1000 cap.
"""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.turbulence import clubb as C  # noqa: E402, N812

_FIX = Path(__file__).parent / "oracle_fixtures" / "clubb_core_cam.npz"
_FIX_IMPL = (Path(__file__).parent / "oracle_fixtures"
             / "clubb_core_cam_implemented.npz")
_GEN_DIR = (Path(__file__).resolve().parents[2]
            / "scripts" / "validate" / "oracle" / "clubb_core")
_TOL = 1.0e-10
_CFG = C.CLUBBConfig(prognostic=True)
# Hard limit of the smooth up2/vp2 cap (RULE 5): the reference's exact min().
_CFG_HARD = _CFG._replace(up2_vp2_cap_width_m2s2=0.0)
_CARRIED = ("rcm", "cloud_frac", "wpthvp", "wp2thvp", "rtpthvp", "thlpthvp",
            "Lscale", "Kh_zm")


def _module(name):
    spec = importlib.util.spec_from_file_location(name, _GEN_DIR / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def fix():
    return dict(np.load(_FIX))


@pytest.fixture(scope="module")
def fix_impl():
    return dict(np.load(_FIX_IMPL))


def _cases(fx):
    return [str(c) for c in fx["meta/cases"]]


def _inputs(fx, case):
    g = lambda k: jnp.asarray(fx[f"{case}/{k}"])  # noqa: E731
    gr = C.make_clubb_grid(g("zm"), g("zt"))
    forcing = C.CLUBBForcing(
        rtm=g("rtm_forcing"), thlm=g("thlm_forcing"), um=g("um_forcing"),
        vm=g("vm_forcing"), wprtp=g("wprtp_forcing"), wpthlp=g("wpthlp_forcing"),
        rtp2=g("rtp2_forcing"), thlp2=g("thlp2_forcing"),
        rtpthlp=g("rtpthlp_forcing"))
    env = dict(exner_zt=g("exner"), p_in_Pa_zt=g("p_in_Pa"), p_sfc=g("p_sfc"),
               thv_ds_zt=g("thv_ds_zt"), thv_ds_zm=g("thv_ds_zm"),
               rho_ds_zm=g("rho_ds_zm"), rho_ds_zt=g("rho_ds_zt"),
               invrs_rho_ds_zm=g("invrs_rho_ds_zm"),
               invrs_rho_ds_zt=g("invrs_rho_ds_zt"), wm_zt=g("wm_zt"),
               wm_zm=g("wm_zm"), sfc_elevation=g("sfc_elevation"),
               fcor=g("fcor"), ug=g("ug"), vg=g("vg"))
    return gr, forcing, env


def _state(fx, case, n):
    st = C.CLUBBMomentState(**{k: jnp.asarray(fx[f"{case}/s{n}/{k}"])
                               for k in C.CLUBBMomentState._fields})
    return st, jnp.asarray(fx[f"{case}/s{n}/rcm"])


def _step(st, rcm, forcing, env, gr, cfg, extra=(), diag_extra=(),
          l_implemented=False, host_dx=None):
    """One step of ours from the oracle state: Lscale/N^2 as the Fortran forms
    them inside advance_clubb_core, the core, then the post-advance closure.
    ``extra`` adds post-advance closure keys, ``diag_extra`` the core's own
    diagnostics (as ``diag/<key>``); ``l_implemented``/``host_dx`` as the
    fixture's Fortran call (host_dx = host_dy there)."""
    p = cfg.params
    ep1 = (1.0 - constants.epsilon) / constants.epsilon
    ep2 = 1.0 / constants.epsilon
    thv = env["thv_ds_zt"]
    thvm = (st.thlm + ep1 * thv * st.rtm
            + (constants.L_v / (constants.c_pd * env["exner_zt"]) - ep2 * thv) * rcm)
    ng = st.thlm.shape[0]
    zeros = jnp.zeros_like(st.thlm)
    brunt = C.calc_brunt_vaisala_freq_sqd(
        st.thlm, env["exner_zt"], st.rtm, zeros, env["p_in_Pa_zt"], zeros,
        p.bv_efold, cfg.T0, gr)[0]
    em = jnp.maximum(0.5 * (st.wp2 + st.up2 + st.vp2), cfg.tke_min)
    Lscale, _, _ = C.compute_mixing_length(
        thvm, st.thlm, st.rtm, em,
        C.set_Lscale_max(l_implemented, host_dx, host_dx, ng),
        env["p_in_Pa_zt"], env["exner_zt"], thv, jnp.full((ng,), p.mu),
        C.derive_lmin(p.lmin_coef), l_implemented, gr)
    new, diag = C.advance_clubb_core(
        st, forcing, Lscale=Lscale, brunt_vaisala_freq_sqd=brunt, dt=300.0,
        gr=gr, config=cfg, l_implemented=l_implemented, **env)
    dpost = C.compute_clubb_diagnostics(
        new.wp2, new.wp3, new.up2, new.vp2, new.thlp2, new.rtp2, new.wpthlp,
        new.wprtp, Lscale, brunt, gr, cfg)
    pdf = C.compute_pdf_closure(
        dpost, new.wp2, new.wp3, new.rtp2, new.thlp2, new.rtpthlp, new.up2,
        new.vp2, new.wprtp, new.wpthlp, new.upwp, new.vpwp, env["wm_zt"],
        env["wm_zm"], new.rtm, new.thlm, new.um, new.vm, env["exner_zt"],
        env["p_in_Pa_zt"], env["p_sfc"], env["thv_ds_zt"], env["thv_ds_zm"],
        gr, cfg)
    out = new._asdict()
    out.update({k: pdf[k] for k in _CARRIED[:6] + tuple(extra)}, Lscale=Lscale,
               Kh_zm=diag["Kh_zm"])
    out.update({f"diag/{k}": diag[k] for k in diag_extra})
    return out


def max_rel_errors(fx, case, cfg=_CFG, l_implemented=None):
    """Field -> worst field-normalised error over the case's N steps.
    ``l_implemented`` defaults to the fixture's (``meta/l_implemented``)."""
    gr, forcing, env = _inputs(fx, case)
    fx_impl = bool(fx.get("meta/l_implemented", False))
    impl = fx_impl if l_implemented is None else l_implemented
    host_dx = jnp.asarray(fx[f"{case}/host_dx"]) if fx_impl else None
    worst = {}
    for n in range(int(fx["meta/n_steps"])):
        st, rcm = _state(fx, case, n)
        out = _step(st, rcm, forcing, env, gr, cfg, l_implemented=impl,
                    host_dx=host_dx)
        for k, v in out.items():
            ref = fx[f"{case}/s{n + 1}/{k}"]
            v = np.asarray(v)
            # A non-finite output or reference is a failure, never a NaN that
            # max() drops.
            err = (np.max(np.abs(v - ref)) / max(np.max(np.abs(ref)), 1e-300)
                   if np.all(np.isfinite(v)) and np.all(np.isfinite(ref)) else np.inf)
            worst[k] = max(worst.get(k, 0.0), float(err))
    return worst


def test_fixture_was_generated_with_the_current_settings(fix):
    """The oracle ran with exactly the parameters and flag table in the code."""
    dump = _module("dump_cam_settings")
    params = {k: float(v) for k, v in C.CLUBBParams()._asdict().items()}
    assert json.loads(str(fix["meta/cam_params"])) == params
    flags = dump.parse_flag_table(Path(C.__file__).read_text())
    assert json.loads(str(fix["meta/cam_flags"])) == flags
    assert flags["l_do_expldiff_rtm_thlm"] is True


def test_generator_records_our_state_fields():
    gen = _module("gen_clubb_core_fixture")
    assert tuple(gen.STATE) == C.CLUBBMomentState._fields
    assert tuple(gen.CARRIED) == _CARRIED


def test_nu_scaling_reproduces_the_oracle_nu(fix):
    """calc_derived_params' nu_vert_res_dep, from the grid alone."""
    for case in _cases(fix):
        if case.endswith("_dryut"):
            continue
        gr, _, _ = _inputs(fix, case)
        p = _CFG.params
        mult = float(C.nu_vert_res_dep_factor(p.mult_coef, gr)[0])
        ours = np.array([p.nu1, p.nu2, p.nu6, p.nu8, p.nu9, p.nu10]) * mult
        np.testing.assert_allclose(ours, fix[f"{case}/nu_vert_res_dep"],
                                   rtol=1e-14, atol=0.0)


@pytest.mark.parametrize("case", [
    "bomex", "dycoms2_rf01", "arm", "wangara", "mpace_b", "bomex_dz100",
    "arm_dz160", "bomex_dryut", "arm_dryut"])
def test_core_matches_fortran(fix, case):
    worst = max_rel_errors(fix, case)
    bad = {k: v for k, v in worst.items() if not v <= _TOL}
    assert not bad, f"{case}: fields above {_TOL}: {bad}"


@pytest.mark.parametrize("case, undo", [
    # each override reproduces the pre-fix behaviour exactly on that column
    ("bomex_dryut", {"rtp2_clip_coef": 1.0e30}),     # no large-rtp2 clip
    ("bomex", {"C7_Lscale0": C.CLUBBParams().C7}),  # damping toward C7 = none
    ("bomex_dz100", {"mult_coef": 0.0}),             # nu not grid-scaled
    ("arm", {"c_K10h": 0.0}),                        # no eddy-scalar diffusion (wm = 0)
])
def test_fixture_exercises_each_fix(fix, case, undo):
    """Non-vacuity: undoing one fix on its column breaks the oracle match."""
    worst = max_rel_errors(fix, case, _CFG._replace(params=_CFG.params._replace(**undo)))
    assert max(worst.values()) > 1.0e3 * _TOL, (case, undo, worst)


def test_gradient_is_finite_on_the_fixture_columns(fix):
    """jax.grad of a scalar of the step w.r.t. the start state and a forcing."""
    for case in ("bomex_dz100", "arm_dryut"):
        gr, forcing, env = _inputs(fix, case)
        st, rcm = _state(fix, case, 1)

        def loss(thlm, rtm, wp2, thlm_forcing):
            out = _step(st._replace(thlm=thlm, rtm=rtm, wp2=wp2), rcm,
                        forcing._replace(thlm=thlm_forcing), env, gr, _CFG)
            return (jnp.sum(out["thlm"]) + 1e3 * jnp.sum(out["rtm"])
                    + jnp.sum(out["wp2"]) + jnp.sum(out["rtp2"]) * 1e6)

        grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
            st.thlm, st.rtm, st.wp2, forcing.thlm)
        for g in grads:
            assert np.all(np.isfinite(np.asarray(g))), case
        assert any(float(jnp.max(jnp.abs(g))) > 0.0 for g in grads)


# ---- l_implemented = .true. (CAM) fixture: wm from the host, up2/vp2 cap ----

_IMPL_CASES = ("bomex", "dycoms2_rf01", "mpace_b", "arm", "bomex_bigup2")


def test_implemented_fixture_records_the_cam_call(fix_impl):
    assert bool(fix_impl["meta/l_implemented"]) is True
    assert tuple(_cases(fix_impl)) == _IMPL_CASES
    for k in ("meta/cam_params", "meta/cam_flags"):
        assert str(fix_impl[k]) == str(np.load(_FIX)[k])


@pytest.mark.parametrize("case", _IMPL_CASES)
def test_core_matches_fortran_implemented(fix_impl, case):
    """At the up2/vp2 cap's hard limit (width 0) every column matches."""
    worst = max_rel_errors(fix_impl, case, _CFG_HARD)
    bad = {k: v for k, v in worst.items() if not v <= _TOL}
    assert not bad, f"{case}: fields above {_TOL}: {bad}"


@pytest.mark.parametrize("case", ["bomex", "dycoms2_rf01", "mpace_b", "arm"])
def test_implemented_fixture_exercises_l_implemented(fix_impl, case):
    """Non-vacuity: the standalone path (mean advection of the means and, on
    every column, the Coriolis term kept) misses the CAM-call oracle."""
    worst = max_rel_errors(fix_impl, case, _CFG_HARD, l_implemented=False)
    assert max(worst.values()) > 1.0e3 * _TOL, (case, worst)


def test_bigup2_column_hits_the_up2_vp2_cap(fix_impl):
    """The oracle's up2/vp2 <= 1000 m2/s2 cap binds on every recorded step."""
    for n in range(1, int(fix_impl["meta/n_steps"]) + 1):
        for k in ("up2", "vp2"):
            ref = fix_impl[f"bomex_bigup2/s{n}/{k}"]
            assert np.max(ref) == C._UP2_VP2_MAX and np.sum(ref == C._UP2_VP2_MAX) > 0


# Measured KNOWN DIFFERENCE of the smooth up2/vp2 cap at the default width
# (10 m2/s2), per l_implemented fixture column: field -> worst field-normalised
# error vs the oracle (max_rel_errors).  Fields absent here stay <= _TOL.  Only
# bomex_bigup2 reaches the cap; the other columns are unchanged from width 0.
# Measured 2026-10-28 (SLURM CPU, fp64) at the cap_up2_vp2 smooth_min form.
_DEFAULT_GAPS = {
    "bomex": {}, "dycoms2_rf01": {}, "mpace_b": {}, "arm": {},
    "bomex_bigup2": {
        "wp2": 6.069554338e-04, "wp3": 1.392288763e-03, "up2": 6.36533013e-03,
        "vp2": 4.597006535e-03, "wprtp": 7.416669653e-06,
        "wpthlp": 1.143293102e-05, "upwp": 1.710857469e-04,
        "vpwp": 1.36211683e-04, "rcm": 7.605580347e-07,
        "cloud_frac": 7.466844415e-07, "wpthvp": 6.145381119e-05,
        "wp2thvp": 4.250272206e-04, "rtpthvp": 1.379084444e-06,
        "thlpthvp": 8.177945709e-07,
    },
}


@pytest.mark.parametrize("case", _IMPL_CASES)
def test_up2_vp2_cap_default_gap_per_case(fix_impl, case):
    """At the default smooth cap width each column's recorded gap holds."""
    assert _CFG.up2_vp2_cap_width_m2s2 == 10.0
    worst = max_rel_errors(fix_impl, case)
    gaps = _DEFAULT_GAPS[case]
    over = {k: v for k, v in worst.items() if not v <= _TOL}
    assert set(over) == set(gaps), (case, over)
    for k, g in gaps.items():
        np.testing.assert_allclose(worst[k], g, rtol=1e-6, atol=1e-12,
                                   err_msg=f"{case}/{k}")


def test_up2_vp2_cap_gradient_at_default():
    """check_grads through the smooth cap at the default width, on both sides
    of 1000 m2/s2; the derivative is finite and non-zero everywhere."""
    from jax.test_util import check_grads
    w = _CFG.up2_vp2_cap_width_m2s2
    x = jnp.linspace(900.0, 1100.0, 9)
    f = lambda v: C.cap_up2_vp2(v, w)  # noqa: E731
    check_grads(f, (x,), order=2, modes=("fwd", "rev"))
    d = jax.vmap(jax.grad(f))(x)
    assert np.all(np.isfinite(np.asarray(d))) and np.all(np.asarray(d) > 0.0)


def test_up2_vp2_cap_hard_limit_and_negative_width():
    x = jnp.array([0.0, 4.0e-4, 999.0, 1000.0, 1500.0])
    np.testing.assert_array_equal(np.asarray(C.cap_up2_vp2(x, 0.0)),
                                  np.minimum(np.asarray(x), C._UP2_VP2_MAX))
    with pytest.raises(ValueError, match="up2_vp2_cap_width_m2s2"):
        C.cap_up2_vp2(x, -1.0)


def test_gradient_is_finite_on_the_implemented_columns(fix_impl):
    """jax.grad through the l_implemented step, incl. the capped column."""
    for case in ("mpace_b", "bomex_bigup2"):
        gr, forcing, env = _inputs(fix_impl, case)
        st, rcm = _state(fix_impl, case, 1)
        host_dx = jnp.asarray(fix_impl[f"{case}/host_dx"])

        def loss(thlm, rtm, up2, wm_zt):
            out = _step(st._replace(thlm=thlm, rtm=rtm, up2=up2), rcm, forcing,
                        dict(env, wm_zt=wm_zt), gr, _CFG, l_implemented=True,
                        host_dx=host_dx)
            return (jnp.sum(out["thlm"]) + 1e3 * jnp.sum(out["rtm"])
                    + jnp.sum(out["up2"]) + jnp.sum(out["wpthlp"]))

        grads = jax.grad(loss, argnums=(0, 1, 2, 3))(
            st.thlm, st.rtm, st.up2, env["wm_zt"])
        for g in grads:
            assert np.all(np.isfinite(np.asarray(g))), case
        assert all(float(jnp.max(jnp.abs(g))) > 0.0 for g in grads), case
