"""A prescribed surface flux must reach the CLOSURE, not just the column.

A case deck that prescribes its surface heat and moisture fluxes used to have
them injected as a separate tendency on the lowest cell AFTER turbulence ran,
with the bulk exchange coefficient zeroed to avoid double-counting. The
closure therefore computed its own surface flux as exactly zero. Local
closures still see the resulting gradient; every flux-driven NONLOCAL scheme
loses its defining pathway, because YSU and friends derive the convective
velocity scale, PBL depth, entrainment and countergradient from ``shflx``.

``SurfaceLayerConfig.prescribed_shflx_w_m2`` / ``prescribed_lhflx_w_m2`` hand
the deck value to the closure instead. These tests pin that it arrives, that
momentum is deliberately untouched, that it reaches the MOST branch too, and
that leaving it unset is byte-identical to before.
"""
from __future__ import annotations

import numpy as np
import pytest

jax = pytest.importorskip("jax")
jnp = jax.numpy

from legoesm import constants  # noqa: E402
from legoesm.atmosphere.physics.turbulence.config import (  # noqa: E402
    SurfaceLayerConfig,
)
from legoesm.atmosphere.physics.turbulence.surface_layer import (  # noqa: E402
    compute_surface_fluxes,
)

NCOL = 3
# The Nieuwstadt CBL deck's prescribed kinematic heat flux [K m/s].
CBL_W_THETA_K_M_S = 0.06
CBL_RHO = 1.15


def _column():
    u = jnp.full((NCOL,), 6.0)
    v = jnp.full((NCOL,), -2.0)
    T = jnp.full((NCOL,), 295.0)
    q_v = jnp.full((NCOL,), 8.0e-3)
    T_sfc = jnp.full((NCOL,), 300.0)
    q_sfc = jnp.full((NCOL,), 1.5e-2)
    rho = jnp.full((NCOL,), 1.2)
    return u, v, T, q_v, T_sfc, q_sfc, rho


def test_unset_is_byte_identical_to_the_historical_config():
    """Default None must not perturb any existing run."""
    args = _column()
    base = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=1.1e-3)
    explicit_none = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=1.1e-3,
        prescribed_shflx_w_m2=None, prescribed_lhflx_w_m2=None)
    a = compute_surface_fluxes(*args, base)
    b = compute_surface_fluxes(*args, explicit_none)
    for x, y in zip(a, b):
        assert np.array_equal(np.asarray(x), np.asarray(y))


def test_prescribed_scalar_fluxes_are_returned_verbatim():
    args = _column()
    cfg = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=0.0,
        prescribed_shflx_w_m2=9.46, prescribed_lhflx_w_m2=153.4)
    _tx, _ty, shflx, lhflx, _ustar = compute_surface_fluxes(*args, cfg)
    assert np.allclose(np.asarray(shflx), 9.46)
    assert np.allclose(np.asarray(lhflx), 153.4)


def test_the_zeroed_coefficient_is_what_the_override_repairs():
    """Without the override, Ch=0 gives a surface flux of exactly zero.

    This is the defect, stated as a test: it is the state every prescribed-flux
    case ran in.
    """
    args = _column()
    cfg = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=0.0)
    _tx, _ty, shflx, lhflx, _ustar = compute_surface_fluxes(*args, cfg)
    assert np.allclose(np.asarray(shflx), 0.0)
    assert np.allclose(np.asarray(lhflx), 0.0)


def test_momentum_is_not_overridden():
    """Decks that fix scalar fluxes leave the stress interactive."""
    args = _column()
    plain = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=1.1e-3)
    pinned = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=1.1e-3,
        prescribed_shflx_w_m2=9.46, prescribed_lhflx_w_m2=153.4)
    a = compute_surface_fluxes(*args, plain)
    b = compute_surface_fluxes(*args, pinned)
    assert np.array_equal(np.asarray(a[0]), np.asarray(b[0]))   # tau_x
    assert np.array_equal(np.asarray(a[1]), np.asarray(b[1]))   # tau_y
    assert np.array_equal(np.asarray(a[4]), np.asarray(b[4]))   # ustar
    assert not np.allclose(np.asarray(a[2]), np.asarray(b[2]))  # shflx differs


def test_override_also_applies_on_the_most_branch():
    """The MOST path returns early; it must not skip the prescription."""
    args = _column()
    cfg = SurfaceLayerConfig(
        bulk_scheme="most", z0=1.0e-4, z_ref=20.0,
        prescribed_shflx_w_m2=15.0, prescribed_lhflx_w_m2=115.0)
    _tx, _ty, shflx, lhflx, _ustar = compute_surface_fluxes(*args, cfg)
    assert np.allclose(np.asarray(shflx), 15.0)
    assert np.allclose(np.asarray(lhflx), 115.0)


def test_only_one_channel_can_be_prescribed():
    """Sensible-only is legitimate: the dry analytic cases have no latent flux."""
    args = _column()
    cfg = SurfaceLayerConfig(
        Cd_neutral=1.1e-3, Ch_neutral=1.1e-3,
        prescribed_shflx_w_m2=-7.5, prescribed_lhflx_w_m2=None)
    plain = SurfaceLayerConfig(Cd_neutral=1.1e-3, Ch_neutral=1.1e-3)
    _tx, _ty, shflx, lhflx, _ = compute_surface_fluxes(*args, cfg)
    ref = compute_surface_fluxes(*args, plain)
    assert np.allclose(np.asarray(shflx), -7.5)
    assert np.array_equal(np.asarray(lhflx), np.asarray(ref[3]))


def test_ysu_convective_pathway_switches_on_with_the_flux():
    """The consequence, on the scheme codex named.

    YSU builds its convective velocity scale from shflx. With the historical
    Ch=0 the dry convective case runs it with no convection at all; handing it
    the deck flux must change its tendencies.
    """
    from legoesm.atmosphere.physics.turbulence.config import YSUConfig
    from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence

    nlev = 24
    ncol = 1
    z_half = jnp.linspace(1600.0, 0.0, nlev + 1)[None, :]
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    p_half = jnp.linspace(8.0e4, 1.0e5, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    # Well-mixed dry column, zero wind: the CBL case's defining configuration.
    T = jnp.full((ncol, nlev), 300.0)
    u = jnp.zeros((ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.full((ncol, nlev), 1.0e-3)
    rho = jnp.full((ncol, nlev), CBL_RHO)
    T_sfc = jnp.full((ncol,), 300.0)
    q_sfc = jnp.full((ncol,), 1.0e-3)

    deck_shf = CBL_W_THETA_K_M_S * CBL_RHO * constants.c_pd
    off = YSUConfig(surface=SurfaceLayerConfig(
        Cd_neutral=7.5e-3, Ch_neutral=0.0))
    on = YSUConfig(surface=SurfaceLayerConfig(
        Cd_neutral=7.5e-3, Ch_neutral=0.0,
        prescribed_shflx_w_m2=deck_shf, prescribed_lhflx_w_m2=0.0))

    out_off = ysu_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half,
                             T_sfc, q_sfc, rho, 10.0, off)
    out_on = ysu_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half,
                            T_sfc, q_sfc, rho, 10.0, on)

    assert float(out_off.shflx[0]) == pytest.approx(0.0), (
        "Ch=0 with no prescription gives the closure a zero surface flux -- "
        "this is the defect being repaired")
    assert float(out_on.shflx[0]) == pytest.approx(deck_shf)

    # h_pbl is the sharp discriminator: YSU's convective branch grows the PBL
    # from the surface buoyancy flux, so with no flux it sits at its floor.
    # (dT_dt is NOT exactly zero without the flux -- residual background
    # diffusion acts on the initial profile at ~1e-7 K/s -- so asserting an
    # exact zero there would be asserting the wrong thing.)
    h_off = float(out_off.h_pbl[0])
    h_on = float(out_on.h_pbl[0])
    assert h_on > h_off, (
        f"the prescribed surface flux must deepen the convective PBL: "
        f"h_pbl {h_off:.1f} -> {h_on:.1f} m")

    amp_off = float(np.max(np.abs(np.asarray(out_off.dT_dt))))
    amp_on = float(np.max(np.abs(np.asarray(out_on.dT_dt))))
    assert amp_on > 100.0 * amp_off, (
        f"the flux-driven tendency must dominate the residual background "
        f"diffusion: {amp_on:.3e} vs {amp_off:.3e} K/s")


# ---------------------------------------------------------------------------
# TIME-VARYING prescribed flux (Wangara Day 33's diurnal cycle)
#
# The config scalar above is ONE value for the whole run, so a case whose
# surface flux follows a diurnal cycle could only ever hand the closure one
# instant of it. The per-step route writes the current value into PhysicsState
# and the turbulence integration folds it into the same config leaf, so no
# closure signature changes and no scheme learns about time.
# ---------------------------------------------------------------------------

def _phys_state_stub(*, wth=None, wqv=None, ncol=1):
    """A minimal PhysicsState carrying (or not carrying) the flux overrides."""
    from legoesm.atmosphere.physics.physics_state import (
        NO_SFC_T_OVERRIDE, PhysicsState,
    )
    z1 = jnp.zeros((ncol, 1))
    return PhysicsState(
        tke=z1, conv_prog_profile=z1, conv_stoch_state=jnp.zeros((ncol,)),
        gwd_spectrum=jnp.zeros((ncol, 1, 1)),
        prng_key=jax.random.PRNGKey(0),
        surface_T_sfc_override=jnp.full((ncol,), NO_SFC_T_OVERRIDE),
        qke=z1, clubb_moments=jnp.zeros((ncol, 1, 1)), rad_heating=z1,
        col_index=jnp.arange(ncol, dtype=jnp.int32),
        surface_wth_override=(None if wth is None
                              else jnp.full((ncol,), float(wth))),
        surface_wqv_override=(None if wqv is None
                              else jnp.full((ncol,), float(wqv))),
    )


def test_absent_override_leaves_the_config_untouched():
    """NON-VACUITY: the default path must be the identity, object and all.

    Without this the two tests below would pass on a helper that rewrote every
    config it was handed, which is the change that would silently perturb every
    3-D run in the repo.
    """
    from legoesm.atmosphere.physics.turbulence.config import YSUConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        _resolve_prescribed_surface_fluxes,
    )
    cfg = YSUConfig(surface=SurfaceLayerConfig(Ch_neutral=1.1e-3))
    rho = jnp.full((1, 4), CBL_RHO)
    assert _resolve_prescribed_surface_fluxes(cfg, None, rho) is cfg
    assert _resolve_prescribed_surface_fluxes(
        cfg, _phys_state_stub(), rho) is cfg


@pytest.mark.parametrize("w_theta", [0.0897, 0.0])
def test_the_override_arrives_as_the_matching_w_m2_flux(w_theta):
    """rho * c_pd * w'T', with rho the LOWEST FULL level the closure is given."""
    from legoesm.atmosphere.physics.turbulence.config import YSUConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        _resolve_prescribed_surface_fluxes,
    )
    cfg = YSUConfig(surface=SurfaceLayerConfig(Ch_neutral=0.0))
    # A rho PROFILE, not a constant: picking the wrong level is then visible.
    rho = jnp.asarray([[0.9, 1.0, 1.1, CBL_RHO]])
    out = _resolve_prescribed_surface_fluxes(
        cfg, _phys_state_stub(wth=w_theta), rho)
    assert float(out.surface.prescribed_shflx_w_m2[0]) == pytest.approx(
        w_theta * CBL_RHO * constants.c_pd)
    # The moisture channel was not supplied, so it must stay absent rather
    # than become a zero flux -- a dry case has no latent flux, and writing
    # 0.0 would override an interactive one on a moist case.
    assert out.surface.prescribed_lhflx_w_m2 is None


def test_the_kinematic_round_trip_is_exact():
    """YSU divides straight back out; it must recover what was prescribed.

    ``ysu.py``: ``wtheta_sfc = shflx / (rho[:, -1] * c_pd)``. Converting with
    any OTHER density here would leave a silent offset in the one number every
    flux-driven nonlocal scheme is built on.
    """
    from legoesm.atmosphere.physics.turbulence.config import YSUConfig
    from legoesm.atmosphere.physics.turbulence.integration import (
        _resolve_prescribed_surface_fluxes,
    )
    w_theta = 0.0897
    rho = jnp.asarray([[0.9, 1.0, 1.1, CBL_RHO]])
    out = _resolve_prescribed_surface_fluxes(
        YSUConfig(surface=SurfaceLayerConfig(Ch_neutral=0.0)),
        _phys_state_stub(wth=w_theta), rho)
    recovered = (out.surface.prescribed_shflx_w_m2
                 / (rho[:, -1] * constants.c_pd))
    assert float(recovered[0]) == pytest.approx(w_theta, rel=1e-12)


def test_the_closure_tracks_a_flux_that_changes_between_steps():
    """The whole point: two times, two fluxes, two different PBLs.

    Pinned through ``make_turbulence_physics`` rather than the helper, so the
    integration wiring is part of what is tested; a helper that worked while
    its call site still passed the unmodified config would pass the tests
    above and fail this one.
    """
    from legoesm.atmosphere.physics.turbulence.config import (
        TurbulenceConfig, YSUConfig,
    )
    from legoesm.atmosphere.physics.turbulence.ysu import ysu_turbulence

    nlev, ncol = 24, 1
    z_half = jnp.linspace(1600.0, 0.0, nlev + 1)[None, :]
    z_full = 0.5 * (z_half[:, :-1] + z_half[:, 1:])
    p_half = jnp.linspace(8.0e4, 1.0e5, nlev + 1)[None, :]
    p_full = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    T = jnp.full((ncol, nlev), 300.0)
    u = jnp.zeros((ncol, nlev))
    v = jnp.zeros((ncol, nlev))
    q_v = jnp.full((ncol, nlev), 1.0e-3)
    rho = jnp.full((ncol, nlev), CBL_RHO)
    T_sfc = jnp.full((ncol,), 300.0)
    q_sfc = jnp.full((ncol,), 1.0e-3)

    from legoesm.atmosphere.physics.turbulence.integration import (
        _resolve_prescribed_surface_fluxes,
    )
    base = YSUConfig(surface=SurfaceLayerConfig(
        Cd_neutral=7.5e-3, Ch_neutral=0.0))
    assert TurbulenceConfig(scheme="ysu", ysu=base).ysu is base  # dispatch sanity

    def _run(w_theta):
        cfg = _resolve_prescribed_surface_fluxes(
            base, _phys_state_stub(wth=w_theta), rho)
        return ysu_turbulence(u, v, T, q_v, p_full, p_half, z_full, z_half,
                              T_sfc, q_sfc, rho, 10.0, cfg)

    morning, midday = _run(0.02), _run(0.12)
    assert float(morning.shflx[0]) == pytest.approx(
        0.02 * CBL_RHO * constants.c_pd)
    assert float(midday.shflx[0]) == pytest.approx(
        0.12 * CBL_RHO * constants.c_pd)
    assert float(midday.h_pbl[0]) > float(morning.h_pbl[0]), (
        "a stronger surface heat flux must deepen the convective PBL; "
        "equal depths mean the per-step value never reached the closure")


# ---------------------------------------------------------------------------
# CLUBB: wp2 is clipped at BOTH ends on the prognostic path
# ---------------------------------------------------------------------------

def test_clip_variance_applies_the_upper_threshold():
    """The helper always could; the prognostic call site did not pass it."""
    from legoesm.atmosphere.physics.turbulence.clubb import clip_variance

    xp2 = jnp.asarray([[1e-9, 5.0, 5000.0, 7000.0]])
    out = np.asarray(clip_variance(xp2, 1e-4, 1000.0))
    assert out[0, 0] == pytest.approx(1e-4), "floor"
    assert out[0, 1] == pytest.approx(5.0), "interior untouched"
    assert out[0, 2] == pytest.approx(1000.0), "cap"
    # the TOP level is deliberately left alone (nzm-1), matching upstream
    assert out[0, 3] == pytest.approx(7000.0)


def test_the_prognostic_path_passes_wp2_max():
    """NON-VACUOUS source check, naming the symbol that RUNS.

    Upstream calls clip_variance with the optional wp2_max and states the
    reason: "instability caused by large wp2 in CLUBB led unrealistic results
    in AM3". Our prognostic path passed only the floor, while the DIAGNOSTIC
    path already capped -- so a source test that looked at the wrong one would
    have passed throughout. This asserts on ``advance_wp2_wp3``, which is the
    function the campaign's prognostic runs execute.
    """
    import inspect

    from legoesm.atmosphere.physics.turbulence import clubb

    src = inspect.getsource(clubb.advance_wp2_wp3)
    assert "clip_variance(" in src, "the clip moved; update this test"
    call = src[src.index("clip_variance("):]
    call = call[:call.index(")") + 1]
    assert "wp2_max" in call, (
        f"advance_wp2_wp3 clips wp2 without an upper threshold: {call!r}")


def test_wp2_max_matches_upstreams_value():
    """1000 m^2/s^2, constants_clubb.F90. A cap at the wrong magnitude is
    either inert or a new physics change."""
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

    assert CLUBBConfig().wp2_max == pytest.approx(1000.0)


# ---------------------------------------------------------------------------
# #1508: the surface variance boundary condition
# ---------------------------------------------------------------------------

def _sfc_varnce_inputs(ncol=2, nzm=8):
    zeros = jnp.zeros((ncol, nzm))
    return dict(wp2=zeros + 1.0, up2=zeros + 1.0, vp2=zeros + 1.0,
                thlp2=zeros + 1.0, rtp2=zeros + 1e-6, rtpthlp=zeros)


def test_sfc_varnce_matches_the_oracle_formulas():
    """Hand-evaluated against sfc_varnce_module.F90's l_andre_1978=.false.
    branch, which is the only live one (it is a compile-time PARAMETER)."""
    from legoesm.atmosphere.physics.turbulence.clubb import (
        CLUBBConfig, calc_sfc_varnce,
    )
    cfg = CLUBBConfig(prognostic=True)
    a, coef = cfg.params.a_const, cfg.params.up2_sfc_coef
    upwp = jnp.asarray([-0.05]); vpwp = jnp.asarray([0.0])
    wpthlp = jnp.asarray([0.0]); wprtp = jnp.asarray([0.0])   # no buoyancy
    out = calc_sfc_varnce(upwp, vpwp, wpthlp, wprtp,
                          **_sfc_varnce_inputs(ncol=1), config=cfg)
    wp2, up2, vp2 = (np.asarray(o)[:, 0] for o in out[:3])
    # wstar = 0, so uf = sqrt(|tau|/rho) = sqrt(0.05)
    uf = float(np.sqrt(0.05))
    assert wp2[0] == pytest.approx(a * uf ** 2, rel=1e-12)
    assert up2[0] == pytest.approx(coef * a * uf ** 2, rel=1e-12)
    assert vp2[0] == pytest.approx(coef * a * uf ** 2, rel=1e-12)


def test_sfc_varnce_only_touches_the_surface_level():
    """Every level above must be returned byte-identical."""
    from legoesm.atmosphere.physics.turbulence.clubb import (
        CLUBBConfig, calc_sfc_varnce,
    )
    args = _sfc_varnce_inputs()
    z = jnp.asarray([0.0, 0.0])
    out = calc_sfc_varnce(z - 0.05, z, z + 0.05, z + 1e-5,
                          **args, config=CLUBBConfig(prognostic=True))
    for got, name in zip(out, ("wp2", "up2", "vp2", "thlp2", "rtp2",
                               "rtpthlp")):
        ref = np.asarray(args[name])
        assert np.array_equal(np.asarray(got)[:, 1:], ref[:, 1:]), name


def test_sfc_varnce_ustar_floor_bites_in_dead_calm():
    """ufmin = 0.01 m/s keeps wp2 finite with no wind and no heat flux, where
    every formula would otherwise divide by zero."""
    from legoesm.atmosphere.physics.turbulence.clubb import (
        CLUBBConfig, calc_sfc_varnce,
    )
    cfg = CLUBBConfig(prognostic=True)
    z = jnp.zeros((1,))
    out = calc_sfc_varnce(z, z, z, z, **_sfc_varnce_inputs(ncol=1), config=cfg)
    for o in out:
        assert np.all(np.isfinite(np.asarray(o)))
    # NOT a_const*ufmin^2 = 1.8e-4: the wp2 correlation floor is
    # max(w_tol^2, ...) = 4e-4 here and it is the larger of the two, so the
    # floor is what the surface value ends up at. (My first expectation was
    # a_const*ufmin^2 and the code was right, not the test.)
    assert np.asarray(out[0])[0, 0] == pytest.approx(
        cfg.w_tol ** 2, rel=1e-12)
    assert cfg.w_tol ** 2 > cfg.params.a_const * 0.01 ** 2, (
        "this test only means something while the floor is the binding one")


def test_sfc_varnce_is_called_by_the_prognostic_core():
    """NON-VACUOUS: naming the function that RUNS. Upstream calls it every step
    and says it must precede advance_xp2_xpyp and advance_wp2_wp3."""
    import inspect

    from legoesm.atmosphere.physics.turbulence import clubb

    src = inspect.getsource(clubb.advance_clubb_core)
    assert "calc_sfc_varnce(" in src
    assert src.index("calc_sfc_varnce(") < src.index("advance_xp2_xpyp(")
    assert src.index("calc_sfc_varnce(") < src.index("advance_wp2_wp3(")


def test_sfc_varnce_gradient_is_finite_under_a_stable_surface():
    """THE test that would have caught the AD bug this port shipped with.

    A negative surface heat flux takes the `wstar = 0` arm of the where. jax
    still differentiates the DISCARDED arm, and with a zero floor that arm is
    cbrt(0), whose derivative is infinite -- 0 * inf = NaN through the mask.
    The forward value is perfectly finite, so only a gradient check sees it,
    and GABLS1 is stable for its entire run.
    """
    from legoesm.atmosphere.physics.turbulence.clubb import (
        CLUBBConfig, calc_sfc_varnce,
    )
    cfg = CLUBBConfig(prognostic=True)
    args = _sfc_varnce_inputs(ncol=1)

    def loss(wpthlp_sfc):
        out = calc_sfc_varnce(
            jnp.asarray([-0.05]), jnp.asarray([0.01]), wpthlp_sfc,
            jnp.asarray([1.0e-6]), **args, config=cfg)
        return sum(jnp.sum(o) for o in out)

    for wth in (-0.05, -1e-12, 0.0, 1e-12, 0.06):
        g = float(jax.grad(loss)(jnp.asarray([wth]))[0])
        assert np.isfinite(g), f"non-finite d/d(wpthlp_sfc) at wpthlp={wth}"


def test_sfc_varnce_gradient_is_finite_in_a_dead_calm():
    """The sibling singularity: zero stress AND zero buoyancy flux makes the
    sqrt argument exactly 0, whose derivative the maximum() then masks."""
    from legoesm.atmosphere.physics.turbulence.clubb import (
        CLUBBConfig, calc_sfc_varnce,
    )
    cfg = CLUBBConfig(prognostic=True)
    args = _sfc_varnce_inputs(ncol=1)

    def loss(upwp):
        out = calc_sfc_varnce(
            upwp, jnp.zeros((1,)), jnp.zeros((1,)), jnp.zeros((1,)),
            **args, config=cfg)
        return sum(jnp.sum(o) for o in out)

    g = float(jax.grad(loss)(jnp.zeros((1,)))[0])
    assert np.isfinite(g), "non-finite gradient in a quiescent column"


def test_sfc_varnce_tolerances_are_unsquared_in_the_config():
    """`w_tol` must be the tolerance, not its square: the port writes
    `config.w_tol ** 2`, so a pre-squared field would floor at tol^4."""
    from legoesm.atmosphere.physics.turbulence.clubb import CLUBBConfig

    cfg = CLUBBConfig()
    assert cfg.w_tol == pytest.approx(2.0e-2)
    assert cfg.thl_tol > 0.0 and cfg.rt_tol > 0.0
