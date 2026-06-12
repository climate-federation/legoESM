"""Tests for the CLUBB core diagnostics bundle (now in ``clubb.py``).

The constituents (Skw/sigma_sqd_w/em/tau/C6-C7) are each independently
parity-tested; this validates the thin orchestration: the right keys, finite
outputs, correct shapes, and jit/grad cleanliness.
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    CLUBBForcing,
    CLUBBMomentState,
    advance_clubb_core,
    compute_clubb_diagnostics,
    compute_pdf_closure,
    init_clubb_moments,
    pack_clubb_moments,
    unpack_clubb_moments,
)
from legoesm.atmosphere.physics.turbulence.clubb_grid import make_clubb_grid  # noqa: E402


def _gr(ng=2, nzt=12):
    nzm = nzt + 1
    zm = jnp.asarray(np.tile(np.linspace(0.0, 3000.0, nzm), (ng, 1)))
    zt = 0.5 * (zm[:, 1:] + zm[:, :-1])
    return make_clubb_grid(zm, zt), ng, nzm


def _inputs(gr, ng, nzm, seed=0):
    nzt = nzm - 1
    rng = np.random.default_rng(seed)

    def zm(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzm)))

    return dict(
        wp2=jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm))),
        wp3=jnp.asarray(0.1 * rng.standard_normal((ng, nzt))),
        up2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        vp2=jnp.asarray(0.3 + 0.3 * rng.random((ng, nzm))),
        thlp2=jnp.asarray(0.05 + 0.05 * rng.random((ng, nzm))),
        rtp2=jnp.asarray(1e-6 + 1e-6 * rng.random((ng, nzm))),
        wpthlp=zm(1e-2), wprtp=zm(1e-4),
        Lscale=jnp.asarray(50.0 + 200.0 * rng.random((ng, nzt))),
        brunt_vaisala_freq_sqd=zm(1e-4),
        gr=gr, config=CLUBBConfig(),
    )


def test_diagnostics_keys_and_shapes():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    out = compute_clubb_diagnostics(**_inputs(gr, ng, nzm))
    expected = {"Skw_zm", "Skw_zt", "wp2_zt", "wp3_zm", "wp3_on_wp2",
                "wp3_on_wp2_zt", "gamma_Skw", "sigma_sqd_w", "em", "sqrt_em_zt",
                "Lscale_zm", "Kh_zt", "Kh_zm", "C6rt_Skw_fnc", "C6thl_Skw_fnc",
                "C7_Skw_fnc", "invrs_tau_C1_zm", "invrs_tau_C4_zm",
                "invrs_tau_C6_zm", "invrs_tau_C14_zm", "invrs_tau_xp2_zm",
                "invrs_tau_wp3_zt"}
    assert expected <= set(out)
    for k, v in out.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    # zm-level fields are (ng, nzm); zt-level are (ng, nzt)
    assert out["Skw_zm"].shape == (ng, nzm) and out["Skw_zt"].shape == (ng, nzt)
    assert out["sigma_sqd_w"].shape == (ng, nzm)
    assert out["invrs_tau_wp3_zt"].shape == (ng, nzt)
    assert out["Kh_zt"].shape == (ng, nzt) and out["Kh_zm"].shape == (ng, nzm)
    # sigma_sqd_w in (0, 1); Kh >= 0
    s = np.asarray(out["sigma_sqd_w"])
    assert np.all((s >= 0.0) & (s < 1.0))
    assert np.all(np.asarray(out["Kh_zt"]) >= 0.0) and np.all(np.asarray(out["Kh_zm"]) >= 0.0)


def _pdf_inputs(gr, ng, nzm, seed=3):
    """Post-advance moment state + means/thermo for ``compute_pdf_closure``."""
    nzt = nzm - 1
    rng = np.random.default_rng(seed)
    diag = compute_clubb_diagnostics(**_inputs(gr, ng, nzm, seed=seed))
    kw = _inputs(gr, ng, nzm, seed=seed)

    def zt(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzt)))

    return dict(
        diag=diag, wp2=kw["wp2"], wp3=kw["wp3"], rtp2=kw["rtp2"],
        thlp2=kw["thlp2"], rtpthlp=jnp.asarray(1e-7 * rng.standard_normal((ng, nzm))),
        up2=kw["up2"], vp2=kw["vp2"], wprtp=kw["wprtp"], wpthlp=kw["wpthlp"],
        upwp=jnp.asarray(1e-2 * rng.standard_normal((ng, nzm))),
        vpwp=jnp.asarray(1e-2 * rng.standard_normal((ng, nzm))),
        wm_zt=zt(1e-3), rtm=zt(1e-3, 8e-3), thlm=zt(1.0, 295.0),
        um=zt(2.0, 5.0), vm=zt(2.0),
        exner_zt=jnp.asarray(0.9 + 0.05 * rng.random((ng, nzt))),
        p_in_Pa_zt=jnp.asarray(7e4 + 2e4 * rng.random((ng, nzt))),
        thv_ds_zt=zt(1.0, 300.0), gr=gr, config=CLUBBConfig(),
    )


def test_pdf_closure_keys_shapes_finite():
    gr, ng, nzm = _gr()
    nzt = nzm - 1
    out = compute_pdf_closure(**_pdf_inputs(gr, ng, nzm))
    expected = {"wpthvp", "wp2thvp", "rtpthvp", "thlpthvp", "rc_coef_zm",
                "cloud_frac", "rcm", "wprcp", "rtprcp", "thlprcp", "uprcp",
                "vprcp", "wp4_zm", "wp2up2_zm", "wp2vp2_zm", "wpup2", "wpvp2",
                "wp2rtp", "wp2thlp", "wp2up", "wprtp2", "wpthlp2", "wprtpthlp"}
    assert expected <= set(out)
    for k, v in out.items():
        assert np.all(np.isfinite(np.asarray(v))), k
    # buoyancy fluxes on zm except wp2thvp (zt); cloud_frac/rcm on zt
    assert out["wpthvp"].shape == (ng, nzm)
    assert out["wp2thvp"].shape == (ng, nzt)
    assert out["cloud_frac"].shape == (ng, nzt) and out["rcm"].shape == (ng, nzt)
    # cloud fraction in [0, 1]; cloud water non-negative
    cf = np.asarray(out["cloud_frac"])
    assert np.all((cf >= 0.0) & (cf <= 1.0))
    assert np.all(np.asarray(out["rcm"]) >= 0.0)


def test_pdf_closure_jit_and_grad():
    gr, ng, nzm = _gr()
    kw = _pdf_inputs(gr, ng, nzm)

    def loss(wp2):
        out = compute_pdf_closure(**dict(kw, wp2=wp2))
        return jnp.sum(out["wpthvp"] ** 2) + jnp.sum(out["wp4_zm"] ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wp2"])))


def _core_state_env(gr, ng, nzm, seed=7):
    """A physically-plausible start-of-step moment state + host env for one core step."""
    nzt = nzm - 1
    rng = np.random.default_rng(seed)

    def zt(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ng, nzt)))

    def zm_pos(lo, hi):
        return jnp.asarray(lo + (hi - lo) * rng.random((ng, nzm)))

    state = CLUBBMomentState(
        rtm=zt(1e-3, 9e-3), thlm=zt(2.0, 298.0), um=zt(2.0, 4.0), vm=zt(2.0),
        wp2=zm_pos(0.05, 0.6), wp3=jnp.asarray(0.05 * rng.standard_normal((ng, nzt))),
        up2=zm_pos(0.1, 0.4), vp2=zm_pos(0.1, 0.4),
        wprtp=jnp.asarray(1e-4 * rng.standard_normal((ng, nzm))),
        wpthlp=jnp.asarray(1e-2 * rng.standard_normal((ng, nzm))),
        upwp=jnp.asarray(1e-2 * rng.standard_normal((ng, nzm))),
        vpwp=jnp.asarray(1e-2 * rng.standard_normal((ng, nzm))),
        rtp2=zm_pos(1e-8, 2e-6), thlp2=zm_pos(1e-3, 0.1),
        rtpthlp=jnp.asarray(1e-6 * rng.standard_normal((ng, nzm))))
    zeros_zt, zeros_zm = jnp.zeros((ng, nzt)), jnp.zeros((ng, nzm))
    forcing = CLUBBForcing(
        rtm=zeros_zt, thlm=zeros_zt, um=zeros_zt, vm=zeros_zt, wprtp=zeros_zm,
        wpthlp=zeros_zm, rtp2=zeros_zm, thlp2=zeros_zm, rtpthlp=zeros_zm)
    rho = jnp.asarray(1.0 - 0.05 * np.linspace(0, 1, nzm)[None, :] + 0.0 * rng.random((ng, nzm)))
    env = dict(
        Lscale=jnp.asarray(50.0 + 150.0 * rng.random((ng, nzt))),
        brunt_vaisala_freq_sqd=jnp.asarray(1e-4 + 1e-4 * rng.random((ng, nzm))),
        exner_zt=jnp.asarray(0.9 + 0.05 * rng.random((ng, nzt))),
        p_in_Pa_zt=jnp.asarray(7e4 + 2e4 * rng.random((ng, nzt))),
        thv_ds_zt=zt(1.0, 300.0), thv_ds_zm=jnp.asarray(300.0 + rng.random((ng, nzm))),
        rho_ds_zm=rho, rho_ds_zt=jnp.asarray(0.5 * (rho[:, 1:] + rho[:, :-1])),
        invrs_rho_ds_zm=1.0 / rho,
        invrs_rho_ds_zt=jnp.asarray(1.0 / (0.5 * (rho[:, 1:] + rho[:, :-1]))),
        wm_zt=zeros_zt, wm_zm=zeros_zm, sfc_elevation=jnp.zeros((ng,)),
        fcor=jnp.full((ng,), 1e-4), ug=zt(1.0, 5.0), vg=zt(1.0),
        dt=300.0, gr=gr, config=CLUBBConfig())
    return state, forcing, env


def test_pack_unpack_clubb_moments_roundtrip():
    """pack/unpack is a lossless round-trip and the packed array has the
    PhysicsState-carry shape (ncol, 15, nzm)."""
    gr, ng, nzm = _gr()
    nlev = nzm - 1
    state = _core_state_env(gr, ng, nzm)[0]
    arr = pack_clubb_moments(state)
    assert arr.shape == (ng, 15, nzm)
    back = unpack_clubb_moments(arr)
    for name in CLUBBMomentState._fields:
        np.testing.assert_array_equal(
            np.asarray(getattr(back, name)), np.asarray(getattr(state, name)))
    # zt-level fields keep nlev length; zm-level fields keep nzm.
    assert back.rtm.shape == (ng, nlev) and back.wp2.shape == (ng, nzm)
    assert back.wp3.shape == (ng, nlev)
    # init_clubb_moments also round-trips.
    m0 = init_clubb_moments(ng, nlev, CLUBBConfig())
    back0 = unpack_clubb_moments(pack_clubb_moments(m0))
    np.testing.assert_array_equal(np.asarray(back0.wp2), np.asarray(m0.wp2))


def test_advance_clubb_core_conserves_thlm_rtm():
    """Truth-tier check: with ZERO surface flux (flux BCs at the surface level
    zeroed) and ZERO forcing and wm=0, the flux-form scalar advances must
    CONSERVE the column-integrated rho_ds-weighted thlm and rtm across steps. A
    spurious source in the assembly (wrong field fed to an advance, a
    non-telescoping flux divergence) would break this even though the per-step
    finiteness/shape tests pass. (The mean advances inside advance_clubb_core run
    on the carried state directly — no reset, unlike clubb_step.)"""
    gr, ng, nzm = _gr()
    state, forcing, env = _core_state_env(gr, ng, nzm)
    # Zero the surface (index 0, ascending) flux BCs so no surface source enters.
    state = state._replace(
        wprtp=state.wprtp.at[:, 0].set(0.0),
        wpthlp=state.wpthlp.at[:, 0].set(0.0),
        upwp=state.upwp.at[:, 0].set(0.0),
        vpwp=state.vpwp.at[:, 0].set(0.0))
    w = np.asarray(env["rho_ds_zt"]) * np.asarray(gr.dzt)   # mass weight (zt)

    def col_int(field_zt):
        return np.sum(w * np.asarray(field_zt), axis=1)

    thlm0, rtm0 = col_int(state.thlm), col_int(state.rtm)
    s = state
    for _ in range(5):
        s, _ = advance_clubb_core(s, forcing, **env)
    # Relative drift over 5 steps must be at round-off (flux-form conservation).
    rel_thlm = np.max(np.abs(col_int(s.thlm) - thlm0) / np.abs(thlm0))
    rel_rtm = np.max(np.abs(col_int(s.rtm) - rtm0) / np.abs(rtm0))
    assert rel_thlm < 1e-9, f"thlm not conserved: rel drift {rel_thlm:.2e}"
    assert rel_rtm < 1e-9, f"rtm not conserved: rel drift {rel_rtm:.2e}"


def test_advance_clubb_core_one_step():
    gr, ng, nzm = _gr()
    state, forcing, env = _core_state_env(gr, ng, nzm)
    new_state, diags = advance_clubb_core(state, forcing, **env)
    # Every prognostic field stays finite and keeps its shape.
    for name, v in new_state._asdict().items():
        arr = np.asarray(v)
        assert np.all(np.isfinite(arr)), name
        assert arr.shape == np.asarray(getattr(state, name)).shape, name
    # Positive-definite variances stay non-negative after the advance+clips.
    for name in ("wp2", "up2", "vp2", "rtp2", "thlp2"):
        assert np.all(np.asarray(getattr(new_state, name)) >= 0.0), name
    # Cloud diagnostics physical.
    cf = np.asarray(diags["cloud_frac"])
    assert np.all((cf >= 0.0) & (cf <= 1.0)) and np.all(np.asarray(diags["rcm"]) >= 0.0)


def test_advance_clubb_core_jit_and_grad():
    gr, ng, nzm = _gr()
    state, forcing, env = _core_state_env(gr, ng, nzm)

    def loss(thlm):
        s = state._replace(thlm=thlm)
        new_state, _ = advance_clubb_core(s, forcing, **env)
        return jnp.sum(new_state.thlm ** 2) + jnp.sum(new_state.wp2 ** 2)

    assert jnp.isfinite(jax.jit(loss)(state.thlm))
    g = jax.grad(loss)(state.thlm)
    assert jnp.all(jnp.isfinite(g))


def test_pdf_closure_buoyancy_uses_raw_not_floored_variance():
    """Regression (codex): the scalar-variance floor (``rt_tol^2``/``thl_tol^2``)
    feeds ONLY the ADG1 driver, never the buoyancy assembly. In a cloud-free
    (``rcm=0``) column the x'thv' fluxes reduce to closed form in the RAW
    (un-floored) regridded variances; pinning them there proves the floor does
    not leak a tolerance-level covariance into rtpthvp/thlpthvp."""
    from legoesm.atmosphere.physics.turbulence.clubb_grid import zm2zt, zt2zm
    from legoesm.atmosphere.physics.turbulence.clubb_pdf_moments import _EP1

    gr, ng, nzm = _gr()
    nzt = nzm - 1
    kw = _pdf_inputs(gr, ng, nzm)
    # Sub-tolerance scalar variances (< rt_tol^2 / thl_tol^2) and a strongly
    # subsaturated, warm/dry column -> rcm = 0 (no cloud-water buoyancy term).
    kw["rtp2"] = jnp.full((ng, nzm), 1e-20)
    kw["thlp2"] = jnp.full((ng, nzm), 1e-6)   # thl_tol^2 = 1e-4 -> floored if leaked
    kw["rtpthlp"] = jnp.zeros((ng, nzm))
    kw["rtm"] = jnp.full((ng, nzt), 1e-4)     # very dry
    kw["thlm"] = jnp.full((ng, nzt), 320.0)   # warm
    kw["p_in_Pa_zt"] = jnp.full((ng, nzt), 9e4)
    kw["exner_zt"] = jnp.full((ng, nzt), 1.0)
    kw["thv_ds_zt"] = jnp.full((ng, nzt), 320.0)
    # Recompute the diagnostics consistent with the variance override.
    kw["diag"] = compute_clubb_diagnostics(
        wp2=kw["wp2"], wp3=kw["wp3"], up2=kw["up2"], vp2=kw["vp2"],
        thlp2=kw["thlp2"], rtp2=kw["rtp2"], wpthlp=kw["wpthlp"], wprtp=kw["wprtp"],
        Lscale=jnp.full((ng, nzt), 100.0),
        brunt_vaisala_freq_sqd=jnp.full((ng, nzm), 1e-4), gr=gr, config=CLUBBConfig())

    out = compute_pdf_closure(**kw)
    assert np.allclose(np.asarray(out["rcm"]), 0.0), "column must be cloud-free"

    # Closed form with RAW (un-floored) regridded variances, top zm level zeroed.
    rtp2_zt = zm2zt(kw["rtp2"], gr)
    thlp2_zt = zm2zt(kw["thlp2"], gr)
    rtpthlp_zt = zm2zt(kw["rtpthlp"], gr)
    # Cloud-water flux terms (rc_coef * x'rc') vanish at rcm = 0.
    rtpthvp_zt = rtpthlp_zt + _EP1 * kw["thv_ds_zt"] * rtp2_zt
    thlpthvp_zt = thlp2_zt + _EP1 * kw["thv_ds_zt"] * rtpthlp_zt
    k_ub = nzm - 1
    exp_rtpthvp = zt2zm(rtpthvp_zt, gr).at[:, k_ub].set(0.0)
    exp_thlpthvp = zt2zm(thlpthvp_zt, gr).at[:, k_ub].set(0.0)
    np.testing.assert_allclose(np.asarray(out["rtpthvp"]), np.asarray(exp_rtpthvp),
                               rtol=0, atol=1e-12)
    np.testing.assert_allclose(np.asarray(out["thlpthvp"]), np.asarray(exp_thlpthvp),
                               rtol=0, atol=1e-12)
    # And the floored value would be visibly different (thl_tol^2 = 1e-4 >> 1e-6).
    floored_thlpthvp = zt2zm(
        jnp.maximum(thlp2_zt, CLUBBConfig().thl_tol ** 2)
        + _EP1 * kw["thv_ds_zt"] * rtpthlp_zt, gr).at[:, k_ub].set(0.0)
    assert not np.allclose(np.asarray(out["thlpthvp"]), np.asarray(floored_thlpthvp))


_CLUBB_JAX_ROOT = Path(__file__).resolve().parents[2].parent / "CLUBB-JAX"


@pytest.mark.skipif(not (_CLUBB_JAX_ROOT / "clubb_jax").exists(),
                    reason="CLUBB-JAX reference tree not present")
def test_sigma_sqd_w_cam_form_matches_reference():
    """Audit (iter 45): the CAM rt/thl-only ``compute_sigma_sqd_w`` used inside
    ``compute_clubb_diagnostics`` is bit-exact to the full CLUBB-JAX reference
    invoked with ``l_predict_upwp_vpwp=False`` (the CAM default), proving the
    omitted up2/vp2/upwp/vpwp correlation terms are correctly absent."""
    if str(_CLUBB_JAX_ROOT) not in sys.path:
        sys.path.insert(0, str(_CLUBB_JAX_ROOT))
    import clubb_jax.src.CLUBB_core.sigma_sqd_w_module as R  # noqa: N812
    from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig
    from legoesm.atmosphere.physics.turbulence.clubb_helpers import compute_sigma_sqd_w

    gr, ng, nzm = _gr()
    cfg = CLUBBConfig()
    rng = np.random.default_rng(45)
    gamma = jnp.asarray(0.2 + 0.2 * rng.random((ng, nzm)))
    wp2 = jnp.asarray(0.2 + 0.5 * rng.random((ng, nzm)))
    thlp2 = jnp.asarray(0.05 + 0.05 * rng.random((ng, nzm)))
    rtp2 = jnp.asarray(1e-6 + 1e-6 * rng.random((ng, nzm)))
    wpthlp = jnp.asarray(1e-2 * rng.standard_normal((ng, nzm)))
    wprtp = jnp.asarray(1e-4 * rng.standard_normal((ng, nzm)))
    # Dummy momentum-flux/variance fields the reference ignores when the flag is off.
    dummy = jnp.asarray(rng.standard_normal((ng, nzm)))

    mine = compute_sigma_sqd_w(
        gamma, wp2, thlp2, rtp2, wpthlp, wprtp, gr,
        w_tol=cfg.w_tol, thl_tol=cfg.thl_tol, rt_tol=cfg.rt_tol)
    ref = R.compute_sigma_sqd_w(
        gamma, wp2, thlp2, rtp2, dummy, dummy, wpthlp, wprtp, dummy, dummy,
        False, gr)
    np.testing.assert_array_equal(np.asarray(mine), np.asarray(ref))


def test_diagnostics_jit_and_grad():
    gr, ng, nzm = _gr()
    kw = _inputs(gr, ng, nzm)

    def loss(wp2):
        out = compute_clubb_diagnostics(**dict(kw, wp2=wp2))
        return (jnp.sum(out["sigma_sqd_w"] ** 2) + jnp.sum(out["invrs_tau_C6_zm"] ** 2)
                + jnp.sum(out["C6rt_Skw_fnc"] ** 2))

    assert jnp.isfinite(jax.jit(loss)(kw["wp2"]))
    assert jnp.all(jnp.isfinite(jax.grad(loss)(kw["wp2"])))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
