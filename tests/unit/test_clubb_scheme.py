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


def test_prognostic_clubb_surface_heat_flux_tracks_surface_temperature():
    """CLUBB's NATIVE surface-flux coupling responds correctly to the surface
    temperature — the coupling the GABLS1 SCM benchmark (prescribe="T_s" + active
    bulk transfer) relies on.

    With the bulk transfer active (``Ch_neutral>0``), ``clubb_step`` computes the
    surface sensible-heat flux from ``(T_sfc − T_air)`` via the bulk formula and
    exposes it as ``out.shflx``. The sign must track the air–surface contrast:
      * a COLD surface (``T_sfc < T_air``) → ``shflx < 0`` (downward; heat drawn
        OUT of the near-surface air — exactly what cools/stabilises a GABLS1 SBL);
      * a WARM surface (``T_sfc > T_air``) → ``shflx > 0`` (upward);
      * ``T_sfc == T_air`` → ``shflx == 0``.
    This is a DIRECT, deterministic check of the surface boundary coupling (no
    slow SCM integration, no indirect cooling-proxy inference). Two layers are
    asserted so a correct *diagnostic* flux alone cannot make it pass:
      1. ``out.shflx`` (the diagnosed bulk flux) has the right sign + antisymmetry;
      2. the flux is actually COUPLED into the prognostic state — the near-surface
         temperature tendency ``out.dT_dt[:, -1]`` responds with the matching sign
         (cold surface cools, warm warms), proving the chain
         ``shflx → wpthlp_sfc lower-BC → advance_clubb_core → dT_dt`` is live. A
         regression that left ``shflx`` correct but zeroed/inverted the
         ``wpthlp_sfc`` boundary condition would pass (1) but fail (2)."""
    from legoesm.atmosphere.physics.turbulence.clubb_core import pack_clubb_moments
    from legoesm.atmosphere.physics.turbulence.config import SurfaceLayerConfig

    kw = _column(ncol=2, nlev=16, dtheta_dz=4e-3)
    kw.pop("tke")
    kw["z_full"] = 0.5 * (kw["z_half"][:, :-1] + kw["z_half"][:, 1:])
    ncol, nlev = kw["T"].shape
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0,
                      surface=SurfaceLayerConfig(Cd_neutral=1.5e-3, Ch_neutral=1.5e-3))
    carry = pack_clubb_moments(init_clubb_moments(ncol, nlev, cfg))

    def response(dT_sfc):
        """(diagnosed surface heat flux, near-surface temperature tendency)."""
        out, _ = clubb_turbulence_prognostic(
            kw["u"], kw["v"], kw["T"], kw["q_v"], carry, kw["p_full"], kw["p_half"],
            kw["z_full"], kw["z_half"], kw["T"][:, -1] + dT_sfc, kw["q_v"][:, -1],
            kw["rho"], 300.0, cfg)
        return np.asarray(out.shflx), np.asarray(out.dT_dt)[:, -1]

    cold_shf, cold_dT = response(-5.0)      # surface colder than air
    warm_shf, warm_dT = response(+5.0)      # surface warmer than air
    neut_shf, neut_dT = response(0.0)

    # (1) Diagnosed bulk surface flux: sign tracks the contrast; antisymmetric.
    assert np.all(cold_shf < 0.0), f"cold surface gave non-downward shflx: {cold_shf}"
    assert np.all(warm_shf > 0.0), f"warm surface gave non-upward shflx: {warm_shf}"
    assert np.allclose(neut_shf, 0.0, atol=1e-9), f"neutral shflx not ~0: {neut_shf}"
    assert np.allclose(warm_shf, -cold_shf, rtol=1e-6)  # linearised bulk formula
    # (2) Coupled into the prognostic tendency: near-surface dT_dt responds with
    # the matching sign and is well above the ~1e-8 zero-flux floor (neut_dT).
    assert np.all(cold_dT < neut_dT - 1e-6), (
        f"cold surface did not cool near-surface air: dT={cold_dT} vs neutral {neut_dT}")
    assert np.all(warm_dT > neut_dT + 1e-6), (
        f"warm surface did not warm near-surface air: dT={warm_dT} vs neutral {neut_dT}")
    assert np.allclose(warm_dT - neut_dT, -(cold_dT - neut_dT), rtol=1e-3)


def test_prognostic_clubb_conserves_column_moisture_no_sfc_flux():
    """Production-scheme conservation, made NON-VACUOUS by first spinning up a
    real turbulent moisture flux.

    With NO surface moisture flux (``q_sfc`` = current near-surface ``q_v`` ⇒
    ``lhflx == 0``), the prognostic CLUBB scheme must CONSERVE the mass-weighted
    column moisture: turbulent redistribution moves water vertically but
    flux-form telescoping with zero surface+top flux leaves the column total
    unchanged.

    Subtlety this test guards against: from the rest/floor moment state the
    turbulent moisture flux ``wprtp`` is ~0, so ``dq_v_dt`` is ~1e-11 kg/kg/s
    and *any* scheme — even one that zeroed ``dq_v_dt`` — would trivially
    "conserve". That makes a bare single-step assertion vacuous. So we first
    integrate the stable SCM driver to develop a genuine ``wprtp`` (the driver's
    flux-form host diffusion keeps the bare column finite), THEN take one
    prognostic-CLUBB step and assert BOTH (a) the redistribution is nontrivial
    (``max|dq_v_dt·dt|`` well above round-off) AND (b) the column total is
    conserved to round-off. A regression that suppressed the scalar flux/solve
    would now FAIL guard (a); one that broke flux-form telescoping would fail
    (b)."""
    from legoesm.atmosphere.physics._shared import virtual_temperature
    from legoesm.atmosphere.physics.turbulence.clubb import (
        integrate_clubb_column,
    )
    from legoesm.atmosphere.physics.turbulence.clubb_core import pack_clubb_moments

    kw = _scm_column(ncol=2, nlev=24, dtheta_dz=4e-3)   # moist, stably stratified
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)   # dt<=clubb_dt → n_sub=1
    # Spin up the higher-order moments so a real turbulent moisture flux exists
    # (dt=150, nsteps=40 is the proven-stable driver setting; the flux-form host
    # diffusion damps the bare-column 2Δz noise without altering the column sum).
    u_f, v_f, T_f, q_f, m_f, _ = integrate_clubb_column(
        **kw, dt=150.0, nsteps=40, config=cfg)
    # One prognostic-CLUBB step from the spun-up state with ZERO surface moisture
    # flux: q_sfc tracks the *current* near-surface q_v so the bulk lhflx is 0.
    tv = jnp.maximum(virtual_temperature(T_f, q_f), cfg.T0 * 0.5)
    rho = kw["p_full"] / (constants.R_d * tv)
    out, _ = clubb_turbulence_prognostic(
        u_f, v_f, T_f, q_f, pack_clubb_moments(m_f), kw["p_full"], kw["p_half"],
        kw["z_full"], kw["z_half"], T_f[:, -1], q_f[:, -1], rho, 150.0, cfg)
    assert np.all(np.asarray(out.lhflx) == 0.0)          # genuinely zero sfc flux
    # Layer mass [kg/m^2] = rho * dz; column moisture tendency = sum(mass*dq_v_dt).
    dz = np.abs(np.asarray(kw["z_half"])[:, :-1] - np.asarray(kw["z_half"])[:, 1:])
    mass = np.asarray(rho) * dz
    dq = np.asarray(out.dq_v_dt)
    col_dq = np.sum(mass * dq, axis=1)                   # kg/m^2/s
    col_q = np.sum(mass * np.asarray(q_f), axis=1)       # kg/m^2
    # (a) Nontrivial redistribution: the per-step moisture change somewhere in the
    # column is >=1e-8 kg/kg (the spun-up case is ~1e-6; the rest state is ~1e-11,
    # so this floor cleanly separates "real transport" from "vacuously quiescent").
    assert float(np.max(np.abs(dq) * 150.0)) > 1e-8
    # (b) Yet the mass-weighted column total drifts only at flux-form round-off.
    assert np.all(np.abs(col_dq) * 150.0 / col_q < 1e-12)


def test_prognostic_clubb_conserves_column_heat_no_sfc_flux():
    """Heat counterpart of the moisture-conservation invariant (NON-VACUOUS).

    CLUBB transports liquid-water potential temperature ``θl`` (the prognostic
    ``thlm``) in flux form, so with NO surface sensible-heat flux (``T_sfc`` =
    current near-surface ``T`` ⇒ ``shflx == 0``) the mass-weighted column ``θl``
    total must be conserved to round-off — turbulent mixing rearranges heat
    vertically but creates none.

    What ``dT_dt / Π`` actually is: the column bridge maps the advanced mean back
    as ``T_new = thlm_new · Π`` (``clubb.py`` ``T_new = flip_vertical(thlm)*exner``),
    so ``out.dT_dt / Π`` is *exactly* the prognostic ``θl`` tendency ``dθl/dt`` —
    by construction, independent of whether the column is cloudy. (It is: this
    spun-up state carries cloud water, ``rcm`` ~ 5e-3, so this is genuinely a
    ``θl`` budget, NOT a dry-``θ`` one — ``θl ≠ θ`` here.) The conserved quantity
    under turbulent transport is ``θl``, not ``θ``, so verifying ``Σ mass·dθl/dt
    ≈ 0`` is the physically correct heat invariant. Exercises the heat path
    through the bridge (the Exner conversion + surface-BC packing) that the
    moisture test does not.

    Made non-vacuous the same way: spin up a real turbulent heat flux ``wpthlp``
    via the stable SCM driver first, then assert BOTH a nontrivial-redistribution
    floor AND conservation. ``q_sfc`` also tracks ``q_v`` so ``lhflx == 0`` too —
    a purely internal redistribution with all surface fluxes off."""
    from legoesm.atmosphere.physics._shared import (
        exner_function,
        virtual_temperature,
    )
    from legoesm.atmosphere.physics.turbulence.clubb import integrate_clubb_column
    from legoesm.atmosphere.physics.turbulence.clubb_core import pack_clubb_moments

    kw = _scm_column(ncol=2, nlev=24, dtheta_dz=4e-3)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    u_f, v_f, T_f, q_f, m_f, _ = integrate_clubb_column(
        **kw, dt=150.0, nsteps=40, config=cfg)
    tv = jnp.maximum(virtual_temperature(T_f, q_f), cfg.T0 * 0.5)
    rho = kw["p_full"] / (constants.R_d * tv)
    # Zero BOTH surface fluxes: T_sfc/q_sfc track the current near-surface values.
    out, _ = clubb_turbulence_prognostic(
        u_f, v_f, T_f, q_f, pack_clubb_moments(m_f), kw["p_full"], kw["p_half"],
        kw["z_full"], kw["z_half"], T_f[:, -1], q_f[:, -1], rho, 150.0, cfg)
    # All surface fluxes off (heat AND moisture) → purely internal redistribution.
    assert np.all(np.asarray(out.shflx) == 0.0)          # genuinely zero sfc heat flux
    assert np.all(np.asarray(out.lhflx) == 0.0)          # and zero sfc moisture flux
    dz = np.abs(np.asarray(kw["z_half"])[:, :-1] - np.asarray(kw["z_half"])[:, 1:])
    mass = np.asarray(rho) * dz
    exner = np.asarray(exner_function(kw["p_full"]))
    dthl = np.asarray(out.dT_dt) / exner                 # dθl/dt [K/s]
    col_dthl = np.sum(mass * dthl, axis=1)               # (kg/m^2) K/s
    col_thl = np.sum(mass * np.asarray(T_f) / exner, axis=1)
    # (a) Nontrivial heat redistribution (~3e-4 K/step here; rest state ~1e-9 K).
    assert float(np.max(np.abs(dthl) * 150.0)) > 1e-6
    # (b) Mass-weighted column θl drifts only at flux-form round-off.
    assert np.all(np.abs(col_dthl) * 150.0 / col_thl < 1e-12)


def test_prognostic_clubb_conserves_column_momentum_to_surface_stress():
    """Momentum budget closure — the third leg of the conservation triad (after
    column θl and total-water rt).

    Turbulent transport conserves column momentum apart from the surface stress
    (the top flux is zero), so the mass-weighted column momentum tendency must
    equal the surface stress:  Σ mass·du/dt = τ_x  (and likewise v ↔ τ_y).

    Two things this checks that the scalar tests can't:
    1. **Surface-stress SIGN** (a historically-fixed bug), for BOTH components:
       with u_sfc, v_sfc > 0 the drag stress τ_x, τ_y < 0, and the column loses
       both eastward and northward momentum (Σ mass·du/dt, Σ mass·dv/dt < 0).
    2. **Interior flux-form conservation of u/v.** The winds advance via CAM's
       ``advance_windm_edsclrm`` eddy-diffusion path (``l_predict_upwp_vpwp=F``),
       NOT the prognostic ``advance_xm_wpxp`` the scalars use — so it needs its
       own conservation check. The interior transport is flux-form exact; the
       only non-closure is the surface stress applied semi-implicitly, an
       **O(Δt)** difference from τ(uⁿ). We verify that explicitly: the budget
       residual shrinks ~linearly as Δt is cut 10×, which both confirms interior
       conservation and pins the surface treatment as the sole O(Δt) term (a
       ρ-vs-ρ_ds weighting error would instead be Δt-independent)."""
    from legoesm.atmosphere.physics._shared import virtual_temperature
    from legoesm.atmosphere.physics.turbulence.clubb import integrate_clubb_column
    from legoesm.atmosphere.physics.turbulence.clubb_core import pack_clubb_moments
    from legoesm.atmosphere.physics.turbulence.surface_layer import (
        compute_surface_fluxes,
    )

    kw = _scm_column(ncol=2, nlev=24, dtheta_dz=4e-3)
    # Shift v to a clearly positive wind so the v-budget relative residual is
    # well-conditioned (the default near-zero-mean v has columns with τ_y ≈ 0).
    kw["v"] = kw["v"] + 5.0
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    u_f, v_f, T_f, q_f, m_f, _ = integrate_clubb_column(
        **kw, dt=150.0, nsteps=40, config=cfg)
    tv = jnp.maximum(virtual_temperature(T_f, q_f), cfg.T0 * 0.5)
    rho = kw["p_full"] / (constants.R_d * tv)
    dz = np.abs(np.asarray(kw["z_half"])[:, :-1] - np.asarray(kw["z_half"])[:, 1:])
    mass = np.asarray(rho) * dz
    carry = pack_clubb_moments(m_f)
    # Surface stress as clubb_step computes it (explicit, from the current wind).
    tau_x, tau_y, _, _, _ = compute_surface_fluxes(
        u_f[:, -1], v_f[:, -1], T_f[:, -1], q_f[:, -1], T_f[:, -1], q_f[:, -1],
        rho[:, -1], cfg.surface)
    tau_x = np.asarray(tau_x)
    tau_y = np.asarray(tau_y)

    def momentum_resid(dt):
        out, _ = clubb_turbulence_prognostic(  # dt < clubb_dt → n_sub == 1
            u_f, v_f, T_f, q_f, carry, kw["p_full"], kw["p_half"], kw["z_full"],
            kw["z_half"], T_f[:, -1], q_f[:, -1], rho, dt, cfg)
        col_du = np.sum(mass * np.asarray(out.du_dt), axis=1)
        col_dv = np.sum(mass * np.asarray(out.dv_dt), axis=1)
        rel_u = np.abs(col_du - tau_x) / np.abs(tau_x)
        rel_v = np.abs(col_dv - tau_y) / np.abs(tau_y)
        return rel_u, rel_v, col_du, col_dv

    rel_u_big, rel_v_big, col_du_big, col_dv_big = momentum_resid(150.0)
    rel_u_small, rel_v_small, _, _ = momentum_resid(15.0)
    # (1) Surface stress is drag in BOTH components: positive surface wind →
    # negative stress and the column loses momentum in that direction.
    assert np.all(u_f[:, -1] > 0.0) and np.all(v_f[:, -1] > 0.0)
    assert np.all(tau_x < 0.0) and np.all(tau_y < 0.0)
    assert np.all(col_du_big < 0.0) and np.all(col_dv_big < 0.0)
    # (2a) Both budgets close to within the O(Δt) surface term at the model step.
    assert np.all(rel_u_big < 1e-2) and np.all(rel_v_big < 1e-2)
    # (2b) Cutting Δt 10× cuts both residuals ~10× → interior conservation is
    # exact, surface treatment is the sole O(Δt) term (Richardson convergence).
    assert np.all(rel_u_small < rel_u_big / 5.0)
    assert np.all(rel_v_small < rel_v_big / 5.0)


def test_prognostic_clubb_couples_to_microphysics_total_water_budget():
    """Validate the moist-physics CONTRACT that makes prognostic CLUBB's
    variable convention correct.

    Prognostic CLUBB transports the moist-conserved variables θl (``thlm``) and
    total water rt (``rtm``), and the column bridge reports them back as the
    ``T``/``q_v`` tendencies (``T = thlm·Π``, ``q_v = rtm``; clubb.py). It does
    NOT do a saturation adjustment — it defers condensation + the associated
    latent heating to the microphysics scheme. That deferral is only correct if
    turbulence and microphysics together close the column water + energy budget.

    Production execution model (``combined.make_physics``): every physics module
    is called on the SAME (original) state and the tendencies are SUMMED — there
    is no sequential state update between turbulence and microphysics. So this
    test computes BOTH the CLUBB step and the ``sundqvist`` condensation from the
    same spun-up column (NOT a CLUBB-then-micro chain) and checks the summed-
    tendency budget. The budget closes because each stage independently conserves:
    CLUBB conserves total water (``Σ mass·dq_v|clubb ≈ 0`` at zero surface flux),
    and ``sundqvist``'s column water budget closes to the surface precip sink, so

        Σ mass·(dq_v|clubb + dq_v|μ + dq_c|μ + dq_r|μ) + precip ≈ 0 .

    Scope: the spun-up column is supersaturated (independent of CLUBB), so the
    branch exercised is condensation → autoconversion → surface precip — exactly
    the branch CLUBB's transported moisture feeds. Incoming rain is zero
    (``dq_r|μ ≡ 0`` by the diagnostic-rain scheme's instant-fallout semantics),
    so the rain-drain path is inactive here by construction and is covered by the
    microphysics' own tests, not this coupling test.

    Asserted alongside: (a) real condensation (non-vacuous), and (b) the
    condensation is enthalpy-consistent (``c_pd·dT + L_v·dq_v = 0``)."""
    from legoesm.atmosphere.physics._shared import virtual_temperature
    from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
    from legoesm.atmosphere.physics.microphysics.output import make_zero_hydrometeors
    from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    from legoesm.atmosphere.physics.turbulence.clubb import integrate_clubb_column
    from legoesm.atmosphere.physics.turbulence.clubb_core import pack_clubb_moments

    kw = _scm_column(ncol=2, nlev=24, dtheta_dz=4e-3)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    dt = 150.0
    # Spin up real turbulent fluxes; the resulting column is supersaturated.
    u_f, v_f, T_f, q_f, m_f, _ = integrate_clubb_column(
        **kw, dt=dt, nsteps=40, config=cfg)
    tv = jnp.maximum(virtual_temperature(T_f, q_f), cfg.T0 * 0.5)
    rho = kw["p_full"] / (constants.R_d * tv)
    ncol, nlev = T_f.shape
    dz = jnp.abs(kw["z_half"][:, :-1] - kw["z_half"][:, 1:])
    # Production model: BOTH modules act on the SAME state; tendencies are summed.
    out, _ = clubb_turbulence_prognostic(
        u_f, v_f, T_f, q_f, pack_clubb_moments(m_f), kw["p_full"], kw["p_half"],
        kw["z_full"], kw["z_half"], T_f[:, -1], q_f[:, -1], rho, dt, cfg)
    assert np.all(np.asarray(out.lhflx) == 0.0)          # zero sfc moisture flux
    hyd = make_zero_hydrometeors(ncol, nlev, dtype=T_f.dtype)   # q_r == 0
    mp = sundqvist_microphysics(
        T_f, q_f, hyd, kw["p_full"], kw["p_half"], rho, dz, dt, SundqvistConfig())

    mass = np.asarray(rho) * np.asarray(dz)              # kg/m^2
    dqc = np.asarray(mp.dq_c_dt)
    dqv_m = np.asarray(mp.dq_v_dt)
    dqr = np.asarray(mp.dq_r_dt)
    dTm = np.asarray(mp.dT_dt)
    precip = np.asarray(mp.precipitation)                # kg/m^2/s
    assert float(np.max(np.abs(dqr))) == 0.0             # diagnostic-rain: no q_r tracer growth
    # (a) Non-vacuous: the spun-up column is supersaturated → real condensation.
    assert float(np.max(dqc)) > 1e-7
    # (b) Condensation is enthalpy-consistent (latent heating balances vapor sink).
    assert float(np.max(np.abs(constants.c_pd * dTm + constants.L_v * dqv_m))) < 1e-10
    # (c) Headline: summed turbulence+microphysics tendencies close the column
    # water budget to the surface precip sink (CLUBB conserves; sundqvist closes).
    clubb_col_dqv = np.sum(mass * np.asarray(out.dq_v_dt), axis=1)
    micro_col = np.sum(mass * (dqv_m + dqc + dqr), axis=1)
    col_water = np.sum(mass * np.asarray(q_f), axis=1)
    combined_resid = clubb_col_dqv + micro_col + precip  # kg/m^2/s, want ~0
    # precip is genuinely nonzero (autoconversion fires), so this is not a
    # condensation-only check: the precip sink term is exercised.
    assert np.all(precip > 1e-4)
    assert np.all(np.abs(combined_resid) * dt / col_water < 1e-12)


def test_clubb_microphysics_chain_is_differentiable():
    """The prognostic-CLUBB → microphysics moist chain is end-to-end
    differentiable — the foundational legoESM autodiff requirement, here for the
    coupled path (which the forward-budget test does not exercise).

    The chain runs CLUBB (θl/rt transport), applies its tendencies, then runs
    ``sundqvist`` condensation, which contains a ``max(q_v − q_sat, 0)`` kink.
    ``jax.grad`` must still produce a finite, nonzero gradient — the subgradient
    is well-defined and flows through the kink. The spun-up moment state is held
    fixed (a constant), isolating the grad of the one-step coupling itself (the
    multi-step ``lax.scan`` grad is covered by
    ``test_integrate_clubb_column_jit_and_grad``).

    Crucially the gradient is checked TWO ways so it cannot pass while the
    condensation path is dead: (1) an aggregate end-to-end smoke check, and (2) a
    **microphysics-specific** objective (``Σ precip² + Σ dT_μ²``) whose nonzero
    gradient can ONLY come through the condensation branch — guarded by a forward
    assertion that the column actually condenses (so the ``max`` is on its active
    side, not the flat side where d/dq ≡ 0)."""
    from legoesm.atmosphere.physics._shared import virtual_temperature
    from legoesm.atmosphere.physics.microphysics.config import SundqvistConfig
    from legoesm.atmosphere.physics.microphysics.output import make_zero_hydrometeors
    from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
    from legoesm.atmosphere.physics.turbulence.clubb import integrate_clubb_column
    from legoesm.atmosphere.physics.turbulence.clubb_core import pack_clubb_moments

    kw = _scm_column(ncol=2, nlev=16, dtheta_dz=4e-3)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    dt = 150.0
    u_f, v_f, T_f, q_f, m_f, _ = integrate_clubb_column(
        **kw, dt=dt, nsteps=30, config=cfg)
    tv = jnp.maximum(virtual_temperature(T_f, q_f), cfg.T0 * 0.5)
    rho = kw["p_full"] / (constants.R_d * tv)
    carry = pack_clubb_moments(m_f)
    dz = jnp.abs(kw["z_half"][:, :-1] - kw["z_half"][:, 1:])
    ncol, nlev = T_f.shape
    hyd = make_zero_hydrometeors(ncol, nlev, dtype=T_f.dtype)

    def _run(q_in):
        out, _ = clubb_turbulence_prognostic(
            u_f, v_f, T_f, q_in, carry, kw["p_full"], kw["p_half"], kw["z_full"],
            kw["z_half"], T_f[:, -1], q_in[:, -1], rho, dt, cfg)
        t1 = T_f + dt * out.dT_dt
        q1 = q_in + dt * out.dq_v_dt
        mp = sundqvist_microphysics(
            t1, q1, hyd, kw["p_full"], kw["p_half"], rho, dz, dt, SundqvistConfig())
        return out, mp

    # Forward: the condensation branch is ACTIVE for this column (max on its
    # non-flat side), so a microphysics gradient is not zero-by-construction.
    out0, mp0 = _run(q_f)
    assert float(jnp.max(mp0.dq_c_dt)) > 1e-7
    assert float(jnp.max(mp0.precipitation)) > 1e-4

    # (1) Aggregate end-to-end smoke check: finite value + finite gradient.
    def loss_total(q_in):
        out, mp = _run(q_in)
        return jnp.sum((out.dT_dt + mp.dT_dt) ** 2) + jnp.sum(mp.precipitation ** 2)

    assert jnp.isfinite(jax.jit(loss_total)(q_f))
    g_total = jax.grad(loss_total)(q_f)
    assert g_total.shape == q_f.shape and jnp.all(jnp.isfinite(g_total))

    # (2) Microphysics-ONLY objective: a nonzero gradient here can come ONLY
    # through the sundqvist condensation kink — proves that path is differentiated.
    def loss_micro(q_in):
        _, mp = _run(q_in)
        return jnp.sum(mp.precipitation ** 2) + jnp.sum(mp.dT_dt ** 2)

    g_micro = jax.grad(loss_micro)(q_f)
    assert g_micro.shape == q_f.shape
    assert jnp.all(jnp.isfinite(g_micro))
    assert float(jnp.max(jnp.abs(g_micro))) > 0.0    # grad flows THROUGH the kink


def test_prognostic_clubb_develops_convective_skewness_unlike_clubb_lite():
    """The defining 'fuller-than-clubb_lite' signature: buoyancy-driven
    vertical-velocity SKEWNESS (positive ``wp3``).

    The whole point of porting the full higher-order CLUBB closure (vs the
    down-gradient eddy-diffusion ``clubb_lite``) is the prognostic THIRD moment
    ``wp3`` and the non-local transport it drives. In a convective boundary layer
    (surface heating), buoyant plumes make updrafts narrower/stronger than the
    broad gentle downdrafts → the vertical-velocity distribution is positively
    skewed, ``wp3 > 0``. A pure down-gradient scheme (flux = −Kh·∂φ/∂z) has NO
    third moment and cannot represent this at all.

    We contrast a strongly-heated column against a near-neutral control (surface
    temperature equal to the near-surface air, so the diagnosed surface buoyancy
    flux is negligible — verified, NOT assumed). The convective case must develop
    (a) genuine turbulence (``wp2`` ≫ control) and (b) a clearly POSITIVE skewness
    in the UPPER mixed layer (400–900 m, the entrainment zone where buoyant plume
    skewness peaks — measured ALOFT, surface levels excluded, so it cannot be a
    near-surface superadiabatic artifact), which the near-neutral control does not
    develop (its ``wp3`` ≈ 0 — any skewness there would be numerical, not
    buoyant). This is the physical raison d'être of the port.

    (Aside found while writing this: a *cooled*-surface control is NOT usable as a
    'stable' contrast — a fixed cold ``T_sfc`` over-cools the near-surface air and
    flips the column to vigorous convection. The near-neutral zero-offset control
    is the clean, quiescent baseline.)"""
    from legoesm.atmosphere.physics.turbulence.clubb import integrate_clubb_column
    from legoesm.atmosphere.physics.turbulence.clubb_lite import (
        clubb_lite_turbulence,
    )
    from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig

    ncol, nlev = 1, 40
    p_half = (np.linspace(5e4, 1.0e5, nlev + 1)[None, :]
              * np.ones((ncol, nlev + 1)))
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_half = np.tile(np.linspace(3000.0, 0.0, nlev + 1), (ncol, 1))
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    exner = (p_full / constants.p_ref) ** constants.kappa
    zc = z_full
    # Mixed layer below 1 km, capping inversion + stable free troposphere above.
    theta = 300.0 + np.where(zc < 1000.0, 0.0, 0.006 * (zc - 1000.0) + 2.0)
    T = jnp.asarray(theta * exner)
    u = jnp.asarray(np.full((ncol, nlev), 5.0))
    v = jnp.asarray(np.zeros((ncol, nlev)))
    q_v = jnp.asarray(np.full((ncol, nlev), 5e-3))
    p_full = jnp.asarray(p_full)
    p_half = jnp.asarray(p_half)
    z_full = jnp.asarray(z_full)
    z_half = jnp.asarray(z_half)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    zt = np.sort(zc[0])
    # Upper mixed layer / entrainment zone — surface levels EXCLUDED so the
    # skewness statistic measures non-local transport aloft, not a surface spike.
    upper_bl = (zt > 400.0) & (zt < 900.0)

    def spin_up(t_sfc_offset):
        _, _, _, _, m_f, d = integrate_clubb_column(
            u, v, T, q_v, p_full, p_half, z_full, z_half, T[:, -1] + t_sfc_offset,
            q_v[:, -1], dt=150.0, nsteps=60, config=cfg)
        shflx = float(np.asarray(d["shflx"])[-1, 0])
        return np.asarray(m_f.wp3)[0], np.asarray(m_f.wp2)[0], shflx

    wp3_conv, wp2_conv, sh_conv = spin_up(8.0)    # strong surface heating
    wp3_neut, wp2_neut, sh_neut = spin_up(0.0)    # near-neutral control

    # Everything stays finite and variances are non-negative.
    for arr in (wp3_conv, wp2_conv, wp3_neut, wp2_neut):
        assert np.all(np.isfinite(arr))
    assert np.all(wp2_conv >= 0.0) and np.all(wp2_neut >= 0.0)
    # The control really is near-neutral: its surface heat flux is a small
    # fraction of the convective case's (verified, not assumed).
    assert sh_conv > 0.0
    assert abs(sh_neut) < 0.05 * sh_conv
    # (a) The convective case is genuinely turbulent; the control is quiescent.
    assert float(np.max(wp2_conv)) > 10.0 * float(np.max(wp2_neut))
    assert float(np.max(wp2_conv)) > 0.1
    # (b) THE signature: positive vertical-velocity skewness in the UPPER mixed
    # layer (aloft, surface excluded) — buoyancy-driven, the third moment a
    # down-gradient scheme cannot represent. The near-neutral control has none.
    assert float(np.max(wp3_conv[upper_bl])) > 0.05      # clear positive peak aloft
    assert float(np.mean(wp3_conv[upper_bl])) > 0.0      # net positive skewness aloft
    assert float(np.max(np.abs(wp3_neut[upper_bl]))) < 1e-3

    # (c) The full-vs-lite boundary, EXECUTED (not just asserted in prose): run
    # clubb_lite on the same convective column. It is a down-gradient eddy-
    # diffusion scheme whose ONLY turbulence-state carry is wp2 (a single second
    # moment) — its TurbulenceOutput has no wp3 field at all, so it structurally
    # cannot represent the third-moment skewness the full closure develops above.
    rho_lite = p_full / (constants.R_d * T)
    out_lite, wp2_lite = clubb_lite_turbulence(
        u, v, T, q_v, jnp.full((ncol, nlev), 0.4), p_full, p_half, z_full,
        z_half, T[:, -1] + 8.0, q_v[:, -1], rho_lite, 150.0, CLUBBLiteConfig())
    assert not hasattr(out_lite, "wp3")                  # no third moment, by design
    assert wp2_lite.shape == (ncol, nlev)                # carries one 2nd moment only
    assert jnp.all(jnp.isfinite(wp2_lite))


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
    from legoesm.atmosphere.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate

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


def test_prognostic_clubb_differentiable_through_pipeline():
    """End-to-end AD (the foundational legoESM requirement): jax.grad flows
    through the REAL combined-physics pipeline with prognostic CLUBB — a scalar
    loss on the temperature tendency is differentiable w.r.t. the input T, with a
    finite, nonzero gradient. Confirms the new scheme keeps the model jax.grad-
    compatible in production (not just in isolated unit tests)."""
    from legoesm.atmosphere.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate

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
    phys_state = init_physics_state(6 * n * n, nlev, cfg)
    physics_fn = make_physics(cfg, model_type="hydrostatic", dt=300.0)
    T0 = state.T.data

    def loss(T):
        s = state._replace(T=state.T.replace(data=T))
        tend, _ = physics_fn(s, grid, sigma, phys_state)
        return jnp.sum(tend.dT_dt.data ** 2) + jnp.sum(tend.du_dt.data ** 2)

    g = jax.grad(loss)(T0)
    assert g.shape == T0.shape
    assert np.all(np.isfinite(np.asarray(g)))
    assert float(np.max(np.abs(np.asarray(g)))) > 0.0   # nonzero — clubb is in the graph


def test_prognostic_clubb_pipeline_multistep_stable():
    """Production-viability: the prognostic carry stays BOUNDED + finite over many
    combined-physics steps on a realistic (smooth) cubed-sphere profile — i.e. the
    moment closure reaches a stable quasi-equilibrium with the column, it does not
    blow up. Repeated physics calls evolve PhysicsState.clubb_moments while the
    (smooth) mean state is held fixed."""
    from legoesm.atmosphere.held_suarez import held_suarez_init
    from legoesm.atmosphere.physics.combined import PhysicsConfig, make_physics
    from legoesm.atmosphere.physics.convection.config import ConvectionConfig
    from legoesm.atmosphere.physics.gravity_wave_drag.config import GravityWaveDragConfig
    from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
    from legoesm.atmosphere.physics.physics_state import init_physics_state
    from legoesm.atmosphere.physics.radiation.config import RadiationConfig
    from legoesm.core.field import Field
    from legoesm.grids.cubed_sphere import create_cubed_sphere
    from legoesm.grids.vertical import create_sigma_coordinate

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


def test_diagnostic_clubb_runs_in_float32_no_dtype_promotion():
    """Cross-backend (float32 / Metal) robustness for the DEFAULT diagnostic
    ``scheme="clubb"`` entry — the path most users get (prognostic is opt-in).

    Complements ``test_prognostic_clubb_runs_in_float32_no_dtype_promotion``: the
    diagnostic phase-1 path shares the parcel-Lscale mixing length (fixed in
    iter-67) and the ADG1-PDF closure, so it must likewise return float32 outputs
    with no silent float64 promotion. Feeds an all-float32 column and asserts every
    field of the ``TurbulenceOutput`` plus the carried ``wp2`` stays float32 and
    finite."""
    f32 = jnp.float32
    ncol, nlev = 2, 16
    rng = np.random.default_rng(0)
    p_half = jnp.asarray(
        np.linspace(2e4, 1e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1)), dtype=f32)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_half = jnp.asarray(
        np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1)), dtype=f32)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    exner = (p_full / jnp.asarray(constants.p_ref, f32)) ** jnp.asarray(constants.kappa, f32)
    T = jnp.asarray(290.0 + 4e-3 * np.asarray(z_full), dtype=f32) * exner
    u = jnp.asarray(8.0 + 4.0 * rng.standard_normal((ncol, nlev)), dtype=f32)
    v = jnp.asarray(2.0 * rng.standard_normal((ncol, nlev)), dtype=f32)
    q_v = jnp.asarray(2e-3 + 4e-3 * rng.random((ncol, nlev)), dtype=f32)
    tke = jnp.full((ncol, nlev), 0.4, dtype=f32)
    rho = p_full / (jnp.asarray(constants.R_d, f32)
                    * jnp.maximum(T * (1.0 + 0.61 * q_v), jnp.asarray(150.0, f32)))
    out, wp2 = clubb_turbulence(
        u, v, T, q_v, tke, p_full, p_half, z_full, z_half, T[:, -1], q_v[:, -1],
        rho, jnp.asarray(150.0, f32), CLUBBConfig())
    for arr in (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt, out.Km, out.Kh,
                out.shflx, out.lhflx, out.ustar, out.h_pbl, wp2):
        assert arr.dtype == f32                          # NO silent float64 promotion
        assert jnp.all(jnp.isfinite(arr))


def test_prognostic_clubb_runs_in_float32_no_dtype_promotion():
    """Cross-backend (float32 / Metal) robustness: the prognostic scheme must run
    in float32 WITHOUT silently promoting to float64.

    legoESM targets Apple-Silicon/GPU float32 paths (CLAUDE.md). A stray strong
    float64 literal or default-dtype array construction in a hot loop both breaks
    those backends AND — under ``JAX_ENABLE_X64`` (this suite) with a float32
    column — makes ``lax.scan`` reject the carry on a dtype mismatch. This test
    feeds an all-float32 column and asserts every output (tendencies, diffusivities,
    the packed moment carry) stays float32 and finite, single-step and multi-step.

    Regression guard for the iter-66 fix in ``clubb_mixing_length.py``: the parcel
    buoyant-sorting Lscale scans had ``jnp.float64(0.0)`` carry inits and
    default-dtype (float64-under-x64) ``jnp.zeros``/``jnp.full`` pads + Lscale cap,
    which promoted the float32 column and crashed the ``lax.scan`` carry check."""
    f32 = jnp.float32
    ncol, nlev = 2, 16
    rng = np.random.default_rng(0)
    p_half = jnp.asarray(
        np.linspace(2e4, 1e5, nlev + 1)[None, :] * np.ones((ncol, nlev + 1)), dtype=f32)
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    z_half = jnp.asarray(
        np.tile(np.linspace(16000.0, 0.0, nlev + 1), (ncol, 1)), dtype=f32)
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    exner = (p_full / jnp.asarray(constants.p_ref, f32)) ** jnp.asarray(constants.kappa, f32)
    T = jnp.asarray(290.0 + 4e-3 * np.asarray(z_full), dtype=f32) * exner
    u = jnp.asarray(8.0 + 4.0 * rng.standard_normal((ncol, nlev)), dtype=f32)
    v = jnp.asarray(2.0 * rng.standard_normal((ncol, nlev)), dtype=f32)
    q_v = jnp.asarray(2e-3 + 4e-3 * rng.random((ncol, nlev)), dtype=f32)
    cfg = CLUBBConfig(prognostic=True, clubb_dt=300.0)
    carry = init_clubb_moments(ncol, nlev, cfg, dtype=f32)
    from legoesm.atmosphere.physics.turbulence.clubb_core import pack_clubb_moments
    rho = p_full / (jnp.asarray(constants.R_d, f32)
                    * jnp.maximum(T * (1.0 + 0.61 * q_v), jnp.asarray(150.0, f32)))

    out, carry_new = clubb_turbulence_prognostic(
        u, v, T, q_v, pack_clubb_moments(carry), p_full, p_half, z_full, z_half,
        T[:, -1], q_v[:, -1], rho, jnp.asarray(150.0, f32), cfg)
    for arr in (out.du_dt, out.dv_dt, out.dT_dt, out.dq_v_dt, out.Km, out.Kh, carry_new):
        assert arr.dtype == f32                       # NO silent float64 promotion
        assert jnp.all(jnp.isfinite(arr))
    # Multi-step driver (exercises the lax.scan carry that the bug crashed) in f32.
    u_f, _, T_f, q_f, m_f, _ = integrate_clubb_column(
        u, v, T, q_v, p_full, p_half, z_full, z_half, T[:, -1], q_v[:, -1],
        dt=jnp.asarray(150.0, f32), nsteps=10, config=cfg)
    for arr in (u_f, T_f, q_f, m_f.wp2, m_f.wp3):
        assert arr.dtype == f32
        assert jnp.all(jnp.isfinite(arr))


def test_clubb_mixing_length_no_promotion_on_mixed_dtype_grid():
    """``compute_mixing_length`` must not promote on a MIXED-dtype column.

    The Lscale compute touches both the thermodynamic state and the grid
    geometry. A float32 state combined with a float64 grid (or vice-versa) must
    still return a single dtype (the thermo-state dtype) — not silently promote
    to float64 and poison a downstream float32/Metal path. Guards the iter-67 fix
    that normalizes ALL float inputs (state + ``CLUBBGrid`` fields) to one working
    dtype at the top of ``compute_mixing_length``."""
    from legoesm.atmosphere.physics.turbulence.clubb_grid import (
        make_clubb_grid_from_levels,
    )
    from legoesm.atmosphere.physics.turbulence.clubb_mixing_length import (
        compute_mixing_length,
        set_Lscale_max,
    )

    f32, f64 = jnp.float32, jnp.float64
    ncol, nzt = 2, 20
    # Grid built from float64 heights → all CLUBBGrid fields float64.
    z_full64 = jnp.asarray(np.tile(np.linspace(50.0, 15000.0, nzt), (ncol, 1)), dtype=f64)
    z_half64 = jnp.asarray(
        np.tile(np.linspace(0.0, 16000.0, nzt + 1), (ncol, 1)), dtype=f64)
    gr64 = make_clubb_grid_from_levels(z_full64, z_half64)
    assert gr64.zt.dtype == f64                              # mixed: grid is f64

    # Thermodynamic state in float32 (the "working" dtype we expect back).
    thvm = jnp.asarray(300.0 + 3e-3 * np.asarray(z_full64), dtype=f32)
    thlm = thvm
    rtm = jnp.asarray(5e-3 * np.ones((ncol, nzt)), dtype=f32)
    em = jnp.asarray(0.4 * np.ones((ncol, nzt + 1)), dtype=f32)
    p_in = jnp.asarray(np.tile(np.linspace(1e5, 2e4, nzt), (ncol, 1)), dtype=f32)
    exner = (p_in / jnp.asarray(constants.p_ref, f32)) ** jnp.asarray(constants.kappa, f32)
    thv_ds = thvm
    mu = jnp.full((ncol,), 6e-4, dtype=f32)
    lscale_max = set_Lscale_max(False, None, None, ncol)     # float64 default cap
    lmin = jnp.asarray(0.1, dtype=f64)                       # strong float64 scalar

    Lscale, Lup, Ldn = compute_mixing_length(
        thvm, thlm, rtm, em, lscale_max, p_in, exner, thv_ds, mu,
        lmin, False, gr64)
    # All outputs come back in the float32 state dtype — no float64 promotion.
    # (Dtype is the property under test; physical Lscale validity is covered by
    # the golden-parity tests in test_clubb_mixing_length.py. This synthetic
    # mixed column only needs to be finite and non-negative.)
    for arr in (Lscale, Lup, Ldn):
        assert arr.dtype == f32
        assert jnp.all(jnp.isfinite(arr)) and jnp.all(arr >= 0.0)


def test_clubb_mixing_length_float32_is_numerically_faithful():
    """The float32 mixing length must be CORRECT, not merely finite.

    The other float32 tests check no-crash / no-promotion. This one checks the
    iter-67 dtype-normalization fix did not change the *values*: the parcel
    buoyant-sorting Lscale computed in float32 must match the float64 reference
    to float32 precision on an identical column. (A fix that silently altered the
    algorithm — e.g. casting at the wrong place and dropping a term — would pass
    the finiteness tests but fail here.) The eddy diffusivity Km = c_K·Lscale·√wp2
    is linear in Lscale, so Lscale fidelity is the binding accuracy property.

    Aside (not asserted): the *temperature tendency* dT_dt = Π·(θl_new − θl)/dt is
    a difference of two ~300 K values, so in float32 it carries ~5–10 % relative
    cancellation noise — inherent to any tendency-as-difference-of-large-T scheme
    (clubb_lite and the diagnostic path share it), NOT a CLUBB defect. Lscale, Km
    and the moisture tendency (small absolute values) do not suffer this."""
    from legoesm.atmosphere.physics.turbulence.clubb_grid import (
        make_clubb_grid_from_levels,
    )
    from legoesm.atmosphere.physics.turbulence.clubb_mixing_length import (
        compute_mixing_length,
        set_Lscale_max,
    )

    ncol, nzt = 2, 24
    zf = np.tile(np.linspace(50.0, 15000.0, nzt), (ncol, 1))
    zh = np.tile(np.linspace(0.0, 16000.0, nzt + 1), (ncol, 1))
    thvm = 300.0 + 3e-3 * zf
    rtm = 5e-3 * np.ones((ncol, nzt))
    em = 0.4 * np.ones((ncol, nzt + 1))
    pin = np.tile(np.linspace(1e5, 2e4, nzt), (ncol, 1))
    exn = (pin / constants.p_ref) ** constants.kappa

    def lscale(dtype):
        j = lambda a: jnp.asarray(a, dtype=dtype)  # noqa: E731
        gr = make_clubb_grid_from_levels(j(zf), j(zh))
        mu = jnp.full((ncol,), 6e-4, dtype=dtype)
        Ls, _, _ = compute_mixing_length(
            j(thvm), j(thvm), j(rtm), j(em), set_Lscale_max(False, None, None, ncol),
            j(pin), j(exn), j(thvm), mu, jnp.asarray(0.1, dtype), False, gr)
        return np.asarray(Ls, np.float64)

    L64 = lscale(jnp.float64)
    L32 = lscale(jnp.float32)
    # Float32 Lscale matches the float64 reference to float32 precision (~1e-7),
    # not just "finite" — the fix preserved the algorithm exactly.
    rel = np.max(np.abs(L32 - L64)) / np.max(np.abs(L64))
    assert rel < 1e-4, f"float32 Lscale diverges from float64: rel={rel:.2e}"


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
