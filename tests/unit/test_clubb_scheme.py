"""End-to-end tests for the runnable CLUBB turbulence scheme (``clubb.py``).

Confirms ``scheme="clubb"`` dispatches, runs through the column interface, is
JIT/grad-clean, and behaves physically (positive diffusivities, more vigorous
mixing in unstable than stable columns). The full prognostic moment coupling is
deepened in later phases (see ``PORT_CLUBB.md``); this locks the runnable entry.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm.atmosphere.physics.turbulence.clubb import (  # noqa: E402
    clubb_step,
    clubb_turbulence,
    clubb_turbulence_prognostic,
    integrate_clubb_column,
)
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.clubb_core import (  # noqa: E402
    CLUBBMomentState,
    init_clubb_moments,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulenceConfig  # noqa: E402
from legoesm.atmosphere.physics.turbulence.integration import get_turbulence_fn  # noqa: E402
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput  # noqa: E402

from legoesm import constants  # noqa: E402


def _column(ncol=4, nlev=24, dtheta_dz=4e-3, seed=0):
    """Synthetic top-down column; dtheta_dz controls static stability."""
    rng = np.random.default_rng(seed)
    p_half = np.linspace(2.0e4, 1.0e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_full = jnp.asarray(np.tile(np.linspace(15000.0, 50.0, nlev), (ncol, 1)))
    z_half = jnp.asarray(np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1)))
    exner = (p_full / constants.p_ref) ** constants.kappa
    # theta increasing upward (height increases with decreasing index) by dtheta_dz.
    z = np.asarray(z_full)
    theta = 290.0 + dtheta_dz * z
    T = jnp.asarray(theta * np.asarray(exner))
    u = jnp.asarray(8.0 + 4.0 * rng.standard_normal((ncol, nlev)))
    v = jnp.asarray(2.0 * rng.standard_normal((ncol, nlev)))
    q_v = jnp.asarray(2e-3 + 4e-3 * rng.random((ncol, nlev)))
    p_full = jnp.asarray(p_full)
    p_half = jnp.asarray(p_half)
    rho = p_full / (constants.R_d * T)
    return dict(u=u, v=v, T=T, q_v=q_v, tke=jnp.full((ncol, nlev), 0.4),
                p_full=p_full, p_half=p_half, z_full=z_full, z_half=z_half,
                T_sfc=T[:, -1] + 1.0, q_sfc=q_v[:, -1], rho=rho, dt=300.0,
                config=CLUBBConfig())


def test_dispatch_selects_clubb():
    name, fn, cfg = get_turbulence_fn(TurbulenceConfig(scheme="clubb"))
    assert name == "clubb"
    assert fn is clubb_turbulence
    assert isinstance(cfg, CLUBBConfig)


def test_runs_and_outputs_are_physical():
    out, wp2_new = clubb_turbulence(**_column())
    assert isinstance(out, TurbulenceOutput)
    ncol, nlev = _column()["T"].shape
    for arr in (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt, out.Km, out.Kh):
        assert arr.shape == (ncol, nlev)
        assert jnp.all(jnp.isfinite(arr))
    for arr in (out.shflx, out.lhflx, out.ustar, out.h_pbl):
        assert arr.shape == (ncol,)
        assert jnp.all(jnp.isfinite(arr))
    # Positive diffusivities and TKE.
    assert jnp.all(out.Km > 0.0) and jnp.all(out.Kh > 0.0)
    assert jnp.all(wp2_new > 0.0)
    assert wp2_new.shape == (ncol, nlev)


def test_unstable_mixes_more_than_stable():
    """Unstable stratification -> larger eddy diffusivity than strongly stable."""
    out_unstable, _ = clubb_turbulence(**_column(dtheta_dz=-2e-3))
    out_stable, _ = clubb_turbulence(**_column(dtheta_dz=1.5e-2))
    assert float(jnp.mean(out_unstable.Km)) > float(jnp.mean(out_stable.Km))


def test_custom_config_is_used():
    """A custom CLUBBConfig threads through TurbulenceConfig.clubb."""
    cfg = CLUBBConfig()
    name, fn, scheme_cfg = get_turbulence_fn(TurbulenceConfig(scheme="clubb", clubb=cfg))
    assert scheme_cfg is cfg


def test_jit_and_grad():
    kw = _column()

    def loss(T):
        out, wp2 = clubb_turbulence(**dict(kw, T=T))
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(out.Km) + jnp.sum(wp2)

    assert jnp.isfinite(jax.jit(loss)(kw["T"]))
    g = jax.grad(loss)(kw["T"])
    assert g.shape == kw["T"].shape and jnp.all(jnp.isfinite(g))


def _init_moments(ncol, nlev, seed=1):
    """A plausible start-of-step CLUBBMomentState on the CLUBB grids.

    Means on zt (nlev) are placeholders — clubb_step overwrites them from the
    live column; moments/fluxes on zm (nlev+1), wp3 on zt.
    """
    rng = np.random.default_rng(seed)
    nzm = nlev + 1

    def zt(s=1.0, b=0.0):
        return jnp.asarray(b + s * rng.standard_normal((ncol, nlev)))

    def zm_pos(lo, hi):
        return jnp.asarray(lo + (hi - lo) * rng.random((ncol, nzm)))

    return CLUBBMomentState(
        rtm=zt(1e-3, 9e-3), thlm=zt(2.0, 298.0), um=zt(2.0, 6.0), vm=zt(2.0),
        wp2=zm_pos(0.05, 0.6), wp3=jnp.asarray(0.05 * rng.standard_normal((ncol, nlev))),
        up2=zm_pos(0.1, 0.4), vp2=zm_pos(0.1, 0.4),
        wprtp=jnp.asarray(1e-4 * rng.standard_normal((ncol, nzm))),
        wpthlp=jnp.asarray(1e-2 * rng.standard_normal((ncol, nzm))),
        upwp=jnp.asarray(1e-2 * rng.standard_normal((ncol, nzm))),
        vpwp=jnp.asarray(1e-2 * rng.standard_normal((ncol, nzm))),
        rtp2=zm_pos(1e-8, 2e-6), thlp2=zm_pos(1e-3, 0.1),
        rtpthlp=jnp.asarray(1e-6 * rng.standard_normal((ncol, nzm))))


def test_clubb_step_one_prognostic_step():
    """clubb_step advances the full moment set and returns finite, correctly
    shaped top-down tendencies; the carried moments stay finite/shape-stable."""
    kw = _column(ncol=4, nlev=24)
    kw.pop("tke")
    # Real model columns are grid-consistent (thermo level = midpoint of the
    # bracketing momentum half-levels); _column uses independent linspaces, so
    # rebuild z_full as the half-level midpoint to avoid spurious zt2zm extrapolation.
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    moments = _init_moments(ncol, nlev)
    du, dv, dT, dq, new_moments, diags = clubb_step(moments=moments, **kw)
    for name, t in (("du", du), ("dv", dv), ("dT", dT), ("dq", dq)):
        assert t.shape == (ncol, nlev) and np.all(np.isfinite(np.asarray(t))), name
    for name, v in new_moments._asdict().items():
        arr = np.asarray(v)
        assert np.all(np.isfinite(arr)), name
        assert arr.shape == np.asarray(getattr(moments, name)).shape, name
    # positive-definite variances survive
    for name in ("wp2", "up2", "vp2", "rtp2", "thlp2"):
        assert np.all(np.asarray(getattr(new_moments, name)) >= 0.0), name
    cf = np.asarray(diags["cloud_frac"])
    assert np.all((cf >= 0.0) & (cf <= 1.0)) and np.all(np.asarray(diags["rcm"]) >= 0.0)


def test_clubb_step_surface_drag_decelerates():
    """Surface drag sign check: with a UNIFORM positive zonal wind (no internal
    shear, so the only momentum sink is the surface stress BC u'w'_sfc=tau_x/rho,
    NEGATIVE for u>0), CLUBB must DECELERATE the near-surface wind (du_dt<0). The
    earlier -tau_x/rho sign bug would instead accelerate it."""
    kw = _column(ncol=4, nlev=24)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    moments = _init_moments(ncol, nlev)
    kw["u"] = jnp.full((ncol, nlev), 12.0)   # uniform -> no shear, only drag
    kw["v"] = jnp.zeros((ncol, nlev))
    du, _, _, _, new_moments, _ = clubb_step(moments=moments, **kw)
    # Direct sign guard: the persisted surface (index 0, ascending) u'w' must be
    # NEGATIVE — this fails if upwp_sfc is flipped back to -tau_x/rho. (The
    # advance_windm mean-wind drag uses sqrt(upwp^2+vpwp^2), so du<0 alone is
    # sign-insensitive and cannot guard the fix.)
    assert np.all(np.asarray(new_moments.upwp)[:, 0] < 0.0)
    # Physical consequence: drag decelerates the near-surface wind (top-down last).
    assert np.all(np.asarray(du)[:, -1] < 0.0)


def test_clubb_step_jit_and_grad():
    kw = _column(ncol=3, nlev=20)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    moments = _init_moments(ncol, nlev)

    def loss(T):
        du, dv, dT, dq, ns, _ = clubb_step(moments=moments, **dict(kw, T=T))
        return jnp.sum(dT ** 2) + jnp.sum(ns.wp2 ** 2) + jnp.sum(du ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["T"]))
    g = jax.grad(loss)(kw["T"])
    assert g.shape == kw["T"].shape and jnp.all(jnp.isfinite(g))


def _scm_column(ncol=2, nlev=24, dtheta_dz=4e-3, seed=0):
    """A grid-consistent top-down column for the prognostic SCM driver."""
    kw = _column(ncol=ncol, nlev=nlev, dtheta_dz=dtheta_dz, seed=seed)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    for k in ("rho", "config", "dt"):
        kw.pop(k)
    return kw


def test_integrate_clubb_column_multistep_stable():
    """The prognostic CLUBB column integrates many steps stably: the carried
    moments stay finite, variances non-negative, and bounded — the genuine
    multi-step test of the higher-order moment closure (moments persist+evolve)."""
    kw = _scm_column(ncol=2, nlev=24)
    u_f, v_f, T_f, q_f, m_f, diags = integrate_clubb_column(
        **kw, dt=150.0, nsteps=40, config=CLUBBConfig())
    for name, arr in (("u", u_f), ("T", T_f), ("q_v", q_f)):
        a = np.asarray(arr)
        assert np.all(np.isfinite(a)), name
    for name, vv in m_f._asdict().items():
        assert np.all(np.isfinite(np.asarray(vv))), name
    for name in ("wp2", "up2", "vp2", "rtp2", "thlp2"):
        assert np.all(np.asarray(getattr(m_f, name)) >= 0.0), name
    # Mean state stays physical: q_v >= 0 (positivity clip), T strictly positive.
    assert np.all(np.asarray(q_f) >= 0.0)
    assert np.all(np.asarray(T_f) > 100.0)
    # TKE stays bounded (no blow-up) and physical over the run.
    assert float(np.max(np.asarray(m_f.wp2))) < CLUBBConfig().wp2_max
    # Per-step cloud diagnostics are physical throughout.
    cf = np.asarray(diags["cloud_frac"])
    assert cf.shape[0] == 40 and np.all((cf >= 0.0) & (cf <= 1.0))


def test_integrate_clubb_column_dry_stress_stays_physical():
    """A long (~3 h) near-dry, weakly-stratified run stays physical WITH the
    host-numerical-diffusion stand-in (default): wp2 bounded, T physical, q_v>=0.
    Root-caused iter 49-51: the bare driver (nu=0) grows grid-scale 2dz noise the
    coupled dynamical core would damp; a tiny host diffusion removes it (the
    closure itself is conservative + parity-faithful)."""
    kw = _scm_column(ncol=2, nlev=24, dtheta_dz=2e-3)
    kw["q_v"] = jnp.full_like(kw["q_v"], 1e-5)   # near-dry
    kw["q_sfc"] = jnp.full_like(kw["q_sfc"], 1e-5)
    _, _, T_f, q_f, m_f, _ = integrate_clubb_column(
        **kw, dt=200.0, nsteps=60, config=CLUBBConfig())
    assert np.all(np.isfinite(np.asarray(T_f))) and np.all(np.asarray(T_f) > 100.0)
    assert np.all(np.asarray(q_f) >= 0.0)
    assert np.all(np.isfinite(np.asarray(m_f.wp2)))
    assert float(np.max(np.asarray(m_f.wp2))) < 50.0


def test_integrate_clubb_column_host_diffusion_controls_grid_noise():
    """Root-cause guard: the dry-regime instability is grid-scale (2dz) noise the
    host numerical diffusion damps. Bare (nu=0) grows wp2 large; the default
    stand-in keeps it bounded. Pins the root cause + the fix (not a closure bug)."""
    def _run(nu):
        kw = _scm_column(ncol=1, nlev=24, dtheta_dz=2e-3)
        kw["q_v"] = jnp.full_like(kw["q_v"], 1e-5)
        kw["q_sfc"] = jnp.full_like(kw["q_sfc"], 1e-5)
        _, _, _, _, m_f, _ = integrate_clubb_column(
            **kw, dt=200.0, nsteps=60, config=CLUBBConfig(),
            host_numerical_diffusion=nu)
        return float(np.max(np.asarray(m_f.wp2)))
    bare, damped = _run(0.0), _run(0.05)
    # Bare driver: wp2 grows far above the tke_min (1e-6) rest floor (grid-scale
    # instability). Host-diffusion stand-in: bounded small. Diffusion cuts it >5x.
    assert bare > 1.0
    assert damped < 0.2
    assert bare > 5.0 * damped


def test_host_diffusion_conserves_sum_and_preserves_positivity():
    """The flux-form host diffusion (as used in integrate_clubb_column) must
    conserve the column sum exactly AND keep a non-negative (incl. near-zero,
    O(1e-5)) input non-negative for nu<=0.5 — so it never creates water mass
    (codex: the moisture-positivity clip is applied to the CLUBB tendency BEFORE
    this diffusion, which then conserves the clipped sum)."""
    rng = np.random.default_rng(0)
    nlev = 24

    def diffuse(f, nu):
        flux = nu * (f[:, 1:] - f[:, :-1])
        return f.at[:, :-1].add(flux).at[:, 1:].add(-flux)

    for nu in (0.05, 0.25, 0.5):
        # near-dry profile with sharp 2dz structure that would undershoot if the
        # operator weren't monotone.
        q = jnp.asarray(1e-5 * (1.0 + 0.9 * np.sin(np.arange(nlev)))[None, :]
                        + 1e-6 * rng.random((1, nlev)))
        qd = diffuse(q, nu)
        np.testing.assert_allclose(float(jnp.sum(qd)), float(jnp.sum(q)),
                                   rtol=0, atol=1e-18)   # column sum conserved
        assert np.all(np.asarray(qd) >= 0.0)             # positivity preserved
        # T-like field: sum conserved (general field, not just non-negative).
        Tf = jnp.asarray(250.0 + 40.0 * rng.random((1, nlev)))
        np.testing.assert_allclose(float(jnp.sum(diffuse(Tf, nu))),
                                   float(jnp.sum(Tf)), rtol=1e-14, atol=0)


def test_integrate_clubb_column_develops_tke_when_unstable():
    """Surface heating (T_sfc > T) into an unstable column must SUSTAIN/grow
    turbulence: the column-max wp2 after the run exceeds the rest-state floor."""
    kw = _scm_column(ncol=2, nlev=24, dtheta_dz=1e-3)
    # Strong surface heating.
    kw["T_sfc"] = kw["T"][:, -1] + 6.0
    _, _, _, _, m_f, _ = integrate_clubb_column(
        **kw, dt=150.0, nsteps=30, config=CLUBBConfig())
    assert float(np.max(np.asarray(m_f.wp2))) > 10.0 * CLUBBConfig().tke_min


def test_integrate_clubb_column_jit_and_grad():
    kw = _scm_column(ncol=2, nlev=16)

    def loss(T):
        _, _, T_f, _, m_f, _ = integrate_clubb_column(
            **dict(kw, T=T), dt=150.0, nsteps=8, config=CLUBBConfig())
        return jnp.sum(T_f ** 2) + jnp.sum(m_f.wp2 ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["T"]))
    g = jax.grad(loss)(kw["T"])
    assert g.shape == kw["T"].shape and jnp.all(jnp.isfinite(g))


def test_integrate_clubb_column_returns_consistent_means():
    """The returned CLUBBMomentState means must match the returned mean state:
    rtm == flip(q_v), um == flip(u), vm == flip(v) (codex consistency guard)."""
    from legoesm.atmosphere.physics.turbulence.clubb_grid import flip_vertical
    kw = _scm_column(ncol=2, nlev=20)
    u_f, v_f, T_f, q_f, m_f, _ = integrate_clubb_column(
        **kw, dt=150.0, nsteps=12, config=CLUBBConfig())
    np.testing.assert_allclose(np.asarray(m_f.rtm), np.asarray(flip_vertical(q_f)))
    np.testing.assert_allclose(np.asarray(m_f.um), np.asarray(flip_vertical(u_f)))
    np.testing.assert_allclose(np.asarray(m_f.vm), np.asarray(flip_vertical(v_f)))
    # q_v positivity holds in the returned mean AND the moment-state mean.
    assert np.all(np.asarray(q_f) >= 0.0) and np.all(np.asarray(m_f.rtm) >= 0.0)


def test_init_clubb_moments_shapes_and_floors():
    m = init_clubb_moments(3, 20, CLUBBConfig())
    assert m.rtm.shape == (3, 20) and m.wp2.shape == (3, 21) and m.wp3.shape == (3, 20)
    assert np.all(np.asarray(m.wp2) == CLUBBConfig().tke_min)
    assert np.allclose(np.asarray(m.rtp2), CLUBBConfig().rt_tol ** 2)


def test_prognostic_dispatch_selects_prognostic_fn():
    """scheme='clubb' with prognostic=True dispatches clubb_turbulence_prognostic
    and routes its carry to the PhysicsState.clubb_moments slot."""
    from legoesm.atmosphere.physics.turbulence.integration import (
        get_turbulence_fn,
        turbulence_carry_field,
    )
    name, fn, cfg = get_turbulence_fn(
        TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True)))
    assert name == "clubb" and fn is clubb_turbulence_prognostic
    assert turbulence_carry_field(name, cfg) == "clubb_moments"
    # default (diagnostic) clubb still routes to tke + clubb_turbulence.
    name2, fn2, cfg2 = get_turbulence_fn(TurbulenceConfig(scheme="clubb"))
    assert fn2 is clubb_turbulence and turbulence_carry_field(name2, cfg2) == "tke"


def test_clubb_turbulence_prognostic_carry_roundtrip_multistep():
    """The prognostic scheme entry carries the packed CLUBBMomentState
    (ncol,15,nlev+1) in/out of the tke-slot interface and runs stably multi-step
    in a moist column (the model carry path; no host diffusion needed here)."""
    from legoesm.atmosphere.physics.turbulence.clubb_core import (
        init_clubb_moments,
        pack_clubb_moments,
    )
    from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
    kw = _column(ncol=3, nlev=24, dtheta_dz=4e-3)   # moist, stably stratified
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    cfg = CLUBBConfig(prognostic=True)
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg))
    assert carry.shape == (ncol, 15, nlev + 1)
    u, v, T, q = kw["u"], kw["v"], kw["T"], kw["q_v"]
    dt = 150.0
    for _ in range(10):
        out, carry = clubb_turbulence_prognostic(
            u, v, T, q, carry, kw["p_full"], kw["p_half"], kw["z_full"],
            kw["z_half"], kw["T_sfc"], kw["q_sfc"], kw["rho"], dt, cfg)
        assert isinstance(out, TurbulenceOutput)
        assert carry.shape == (ncol, 15, nlev + 1)
        u = u + dt * out.du_dt
        v = v + dt * out.dv_dt
        T = T + dt * out.dT_dt
        q = jnp.maximum(q + dt * out.dq_v_dt, 0.0)
    assert np.all(np.isfinite(np.asarray(carry)))
    assert np.all(np.isfinite(np.asarray(T))) and np.all(np.asarray(T) > 100.0)
    # wp2 (slot 4) evolved away from the init floor (the moments are prognostic).
    wp2_final = np.asarray(carry)[:, 4, :]
    assert float(np.max(wp2_final)) > CLUBBConfig().tke_min


def test_prognostic_clubb_subcycling():
    """CAM runs CLUBB at clubb_timestep, sub-cycling it within the host dt.
    n_sub=1 (dt<=clubb_dt) is bit-identical to a single clubb_step; n_sub>1
    (dt>clubb_dt) runs n_sub sub-steps and returns the NET tendency + sub-cycled
    moments — finite, and the moments evolve."""
    from legoesm.atmosphere.physics.turbulence.clubb_core import (
        init_clubb_moments,
        pack_clubb_moments,
    )
    kw = _column(ncol=2, nlev=24, dtheta_dz=4e-3)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg))
    args = (kw["u"], kw["v"], kw["T"], kw["q_v"], carry, kw["p_full"],
            kw["p_half"], kw["z_full"], kw["z_half"], kw["T_sfc"], kw["q_sfc"],
            kw["rho"])

    # n_sub=1: dt == clubb_dt → bit-identical to a single clubb_step.
    out1, c1 = clubb_turbulence_prognostic(*args, 300.0, cfg)
    from legoesm.atmosphere.physics.turbulence.clubb_core import unpack_clubb_moments
    du, dv, dT, dq, m_new, _ = clubb_step(
        kw["u"], kw["v"], kw["T"], kw["q_v"], unpack_clubb_moments(carry),
        kw["p_full"], kw["p_half"], kw["z_full"], kw["z_half"], kw["T_sfc"],
        kw["q_sfc"], kw["rho"], 300.0, cfg)
    np.testing.assert_array_equal(np.asarray(out1.dT_dt), np.asarray(dT))
    np.testing.assert_array_equal(np.asarray(out1.du_dt), np.asarray(du))

    # n_sub=6: dt=1800 > clubb_dt=300 → sub-cycled, finite, moments evolve.
    out6, c6 = clubb_turbulence_prognostic(*args, 1800.0, cfg)
    assert c6.shape == (ncol, 15, nlev + 1)
    for t in (out6.du_dt, out6.dT_dt, out6.dq_v_dt):
        assert np.all(np.isfinite(np.asarray(t)))
    assert np.all(np.isfinite(np.asarray(c6)))
    assert not np.allclose(np.asarray(c6), np.asarray(carry))   # moments evolved


def test_prognostic_clubb_subcycling_raw_moisture_contract():
    """Codex: the sub-cycled moisture tendency must be the RAW integrated CLUBB
    tendency (no internal q_v positivity clip folded in), so dq_v_dt = (q_final -
    q_initial)/dt where q evolves by the unclipped CLUBB tendency — the SAME
    coupling contract as n_sub=1. Forced with a near-dry column + large dt so any
    clip would show up as a contract divergence (positivity is the host's job)."""
    from legoesm.atmosphere.physics._shared import virtual_temperature
    from legoesm.atmosphere.physics.turbulence.clubb_core import (
        init_clubb_moments,
        pack_clubb_moments,
        unpack_clubb_moments,
    )
    kw = _column(ncol=2, nlev=24, dtheta_dz=4e-3)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    kw["q_v"] = jnp.full_like(kw["q_v"], 1e-6)       # near-dry: any clip would bite
    kw["q_sfc"] = jnp.full_like(kw["q_sfc"], 1e-6)
    ncol, nlev = kw["T"].shape
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg))
    base = (kw["u"], kw["v"], kw["T"], kw["q_v"], carry, kw["p_full"],
            kw["p_half"], kw["z_full"], kw["z_half"], kw["T_sfc"], kw["q_sfc"], kw["rho"])

    out, _ = clubb_turbulence_prognostic(*base, 1500.0, cfg)   # n_sub=5

    # Reference: manually sub-cycle WITHOUT any clip and compare the net dq_v_dt.
    n_sub, dt = 5, 1500.0
    dt_sub = dt / n_sub
    u_c, v_c, T_c, q_c, m_c = (kw["u"], kw["v"], kw["T"], kw["q_v"],
                               unpack_clubb_moments(carry))
    for _ in range(n_sub):
        tv = jnp.maximum(virtual_temperature(T_c, q_c), cfg.T0 * 0.5)
        rho_c = kw["p_full"] / (constants.R_d * tv)
        du, dv, dT, dq, m_c, _ = clubb_step(
            u_c, v_c, T_c, q_c, m_c, kw["p_full"], kw["p_half"], kw["z_full"],
            kw["z_half"], kw["T_sfc"], kw["q_sfc"], rho_c, dt_sub, cfg)
        u_c, v_c, T_c, q_c = (u_c + dt_sub * du, v_c + dt_sub * dv,
                              T_c + dt_sub * dT, q_c + dt_sub * dq)   # NO clip
    np.testing.assert_allclose(np.asarray(out.dq_v_dt),
                               np.asarray((q_c - kw["q_v"]) / dt), rtol=1e-10, atol=0)


def test_prognostic_clubb_blocked_on_non_persisting_drivers():
    """Prognostic CLUBB needs a driver that persists PhysicsState; the
    nonhydrostatic CD-grid + spectral PE factories drop it, so they fail fast
    (like MYNN-2.5). Hydrostatic + mpas persist → allowed."""
    from legoesm.atmosphere.physics.turbulence.integration import make_turbulence_physics
    tc = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True))
    for mt in ("nonhydrostatic", "spectral_pe"):
        with pytest.raises(NotImplementedError, match="prognostic CLUBB"):
            make_turbulence_physics(tc, mt, 300.0)
    # hydrostatic builds fine.
    assert make_turbulence_physics(tc, "hydrostatic", 300.0) is not None
    # diagnostic clubb is allowed on all (stateless wp2 in tke).
    tc_diag = TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig())
    assert make_turbulence_physics(tc_diag, "nonhydrostatic", 300.0) is not None


def test_read_turb_carry_fails_fast_on_wrong_clubb_shape():
    """_read_turb_carry raises (no silent resize/retrace) when phys_state carries
    a wrong-shaped clubb_moments slot (PhysicsState not init'd for prognostic)."""
    from types import SimpleNamespace
    from legoesm.atmosphere.physics.turbulence.integration import _read_turb_carry
    bad = SimpleNamespace(clubb_moments=jnp.zeros((4, 1, 1)))   # minimal placeholder
    with pytest.raises(ValueError, match="prognostic CLUBB expects"):
        _read_turb_carry(bad, "clubb_moments", 4, 24, CLUBBConfig(prognostic=True),
                         jnp.float64)
    # phys_state=None seeds fresh at the correct shape (one-off call).
    seeded = _read_turb_carry(None, "clubb_moments", 4, 24,
                              CLUBBConfig(prognostic=True), jnp.float64)
    assert seeded.shape == (4, 15, 25)


def test_prognostic_clubb_cloud_fraction_responds_to_moisture():
    """The distinctive CLUBB capability vs clubb_lite is the ADG1 double-Gaussian
    assumed-PDF CLOUD closure. Physics check: the diagnosed cloud fraction must
    increase monotonically as the column moistens toward/above saturation — a
    near-dry column forms essentially no cloud; a supersaturated column forms
    cloud. (RH set with the model's q_sat; CLUBB's own Flatau saturation drives
    the closure, so the monotone response is the robust signal.)"""
    from legoesm.thermo import saturation_mixing_ratio
    base = _scm_column(ncol=2, nlev=24, dtheta_dz=3e-3)
    qsat = saturation_mixing_ratio(base["T"], base["p_full"])
    cfg = CLUBBConfig()

    def total_cloud(rh):
        kw = dict(base)
        kw["q_v"] = rh * qsat
        kw["q_sfc"] = rh * qsat[:, -1]
        _, _, _, _, _, diags = integrate_clubb_column(
            **kw, dt=120.0, nsteps=15, config=cfg)
        cf = np.asarray(diags["cloud_frac"])
        assert np.all((cf >= 0.0) & (cf <= 1.0))   # always a valid fraction
        return float(np.sum(cf))

    dry, moist, supersat = total_cloud(0.4), total_cloud(0.85), total_cloud(1.25)
    assert supersat >= moist >= dry           # monotone in column moisture
    assert supersat > 0.0                      # supersaturated column DOES cloud
    assert dry < 0.05 * max(supersat, 1.0)     # near-dry column ~ cloud-free


def test_prognostic_clubb_convective_bl_physics():
    """Idealized boundary-layer physics check (beyond runs-without-error): in a
    moist, stably-stratified column, STRONG surface heating must drive a
    convective response — the prognostic closure develops substantially more
    turbulence (column-integrated wp2) than the same column with NO surface
    heating, and the heated case has an UPWARD buoyancy flux (wpthvp>0) in the
    lower BL (buoyancy production of TKE). This is the canonical turbulence-
    scheme sanity check, on the genuinely-prognostic path."""
    base = _scm_column(ncol=2, nlev=30, dtheta_dz=3e-3)   # stably stratified
    cfg = CLUBBConfig()

    def run(extra_heating):
        kw = dict(base)
        kw["T_sfc"] = base["T"][:, -1] + extra_heating
        _, _, _, _, m_f, diags = integrate_clubb_column(
            **kw, dt=120.0, nsteps=60, config=cfg)   # ~2 h
        col_tke = float(np.sum(np.asarray(m_f.wp2)))
        return col_tke, np.asarray(diags["wpthvp"])   # (nsteps, ncol, nzm)

    tke_heated, wpthvp_heated = run(6.0)     # strong surface heating
    tke_calm, _ = run(0.0)                    # no surface heating

    # Convective forcing develops markedly more TKE than the unheated column.
    assert tke_heated > 3.0 * tke_calm
    # Upward buoyancy flux somewhere in the lower BL of the heated case (the
    # buoyancy production that sustains convective TKE). Lower BL = top-down
    # ascending zm: low indices are the surface side.
    final_wpthvp = wpthvp_heated[-1]          # (ncol, nzm) last step
    lower_bl = final_wpthvp[:, :final_wpthvp.shape[1] // 2]
    assert float(np.max(lower_bl)) > 0.0


def test_prognostic_clubb_runs_in_combined_physics_pipeline():
    """END-TO-END: scheme='clubb', prognostic=True runs through the real
    combined-physics pipeline (make_physics → hydrostatic physics_fn) on a
    cubed-sphere state, carrying PhysicsState.clubb_moments across TWO steps —
    the moments persist and evolve, tendencies stay finite. This is the
    'legoESM can be run+tested with the prognostic clubb scheme' check."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.held_suarez import held_suarez_init

    n, nlev = 3, 10
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    tracers = {"q_v": Field(5e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                            dims=("face", "x", "y", "level"), units="kg/kg")}
    state = state._replace(tracers=tracers)

    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"))
    ncol = 6 * n * n
    phys_state = init_physics_state(ncol, nlev, cfg)
    assert phys_state.clubb_moments.shape == (ncol, 15, nlev + 1)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)

    moments0 = np.asarray(phys_state.clubb_moments).copy()
    for _ in range(2):
        tend, phys_state = physics_fn(state, grid, sigma, phys_state)
        assert np.all(np.isfinite(np.asarray(tend.dT_dt.data)))
        assert np.all(np.isfinite(np.asarray(tend.du_dt.data)))
        assert phys_state.clubb_moments.shape == (ncol, 15, nlev + 1)
        assert np.all(np.isfinite(np.asarray(phys_state.clubb_moments)))
    # The carried moments evolved away from the rest-state init (genuinely prognostic).
    assert not np.allclose(np.asarray(phys_state.clubb_moments), moments0)


def test_prognostic_clubb_pipeline_multistep_stable():
    """Production-viability: the prognostic carry stays BOUNDED + finite over many
    combined-physics steps on a realistic (smooth) cubed-sphere profile — i.e. the
    moment closure reaches a stable quasi-equilibrium with the column, it does not
    blow up. Repeated physics calls evolve PhysicsState.clubb_moments while the
    (smooth) mean state is held fixed."""
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate
    from legoesm.atmosphere.held_suarez import held_suarez_init

    n, nlev = 2, 8
    grid = create_cubed_sphere(n)
    sigma = create_sigma_coordinate(nlev)
    state = held_suarez_init(grid, sigma)
    state = state._replace(tracers={
        "q_v": Field(5e-3 * jnp.ones((6, n, n, nlev)), name="q_v",
                     dims=("face", "x", "y", "level"), units="kg/kg")})
    cfg = PhysicsConfig(
        radiation=RadiationConfig(scheme="none"),
        convection=ConvectionConfig(scheme="none"),
        turbulence=TurbulenceConfig(scheme="clubb", clubb=CLUBBConfig(prognostic=True)),
        microphysics=MicrophysicsConfig(scheme="none"),
        gravity_wave_drag=GravityWaveDragConfig(scheme="none"))
    ncol = 6 * n * n
    phys_state = init_physics_state(ncol, nlev, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    step = jax.jit(lambda ps: physics_fn(state, grid, sigma, ps))

    wp2_max_hist = []
    for _ in range(15):
        _, phys_state = step(phys_state)
        cm = np.asarray(phys_state.clubb_moments)
        assert np.all(np.isfinite(cm))
        wp2_max_hist.append(float(np.max(cm[:, 4, :])))   # wp2 slot
    # No blow-up: wp2 stays well-bounded across the run.
    assert max(wp2_max_hist) < CLUBBConfig().wp2_max
    # Quasi-equilibrium: the last few steps don't keep growing super-linearly.
    assert wp2_max_hist[-1] < 5.0 * max(wp2_max_hist[:3] + [1e-3])


def test_clubb_turbulence_prognostic_jit_and_grad():
    from legoesm.atmosphere.physics.turbulence.clubb_core import (
        init_clubb_moments,
        pack_clubb_moments,
    )
    kw = _column(ncol=2, nlev=16, dtheta_dz=4e-3)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    cfg = CLUBBConfig(prognostic=True)
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg))

    def loss(T):
        out, c = clubb_turbulence_prognostic(
            kw["u"], kw["v"], T, kw["q_v"], carry, kw["p_full"], kw["p_half"],
            kw["z_full"], kw["z_half"], kw["T_sfc"], kw["q_sfc"], kw["rho"], 150.0, cfg)
        return jnp.sum(out.dT_dt ** 2) + jnp.sum(c ** 2)

    assert jnp.isfinite(jax.jit(loss)(kw["T"]))
    g = jax.grad(loss)(kw["T"])
    assert g.shape == kw["T"].shape and jnp.all(jnp.isfinite(g))


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
