"""The DINO surface restoring runs NEMO's EXPLICIT form, not an implicit-Euler one.

WHAT NEMO DOES, as compiled (``cfgs/DINO/BLD/ppsrc/nemo/``, ``nn_forcingtype=4``):

    sfx  = rn_srp * ( ts(:,:,1,jp_sal,Kbb) - S* ) * tmask         usrdef_sbc.f90:223
    qtot = rn_trp * ( ts(:,:,1,jp_tem,Kbb) - T* ) - emp*...       :272-273  (emp = 0)
    qns  = ( qtot - zqsr_dayMean ) * tmask                        :294
    sbc_tsc(jp_tem) = r1_rho0_rcp * qns ; sbc_tsc(jp_sal) = r1_rho0 * sfx
                                                                  trasbc.f90:153-154
    pts(:,:,1,jn,Krhs) += zfact*(sbc_tsc_b + sbc_tsc)
                          / ( e3t_3d(:,:,1) * (1 + r3t(:,:,Kmm)*tmask) )
                                                                  trasbc.f90:169-170

There is no damping denominator anywhere in that chain.  legoESM applied the
analytical implicit-Euler form (``tau -> tau + dt``) until PR #1728; the
resulting level-0 deficit was measured as exactly that factor on the salt row
(predicted 0.998999633, measured 0.998999633).

Each test below is a synthetic-violation check: flipping the statement back to
``implicit=True`` turns it red.
"""
from __future__ import annotations

import ast
import os

import numpy as np
import pytest

from legoesm.core.precision import PrecisionPolicy, set_policy

DT = 2700.0                      # NEMO DINO rn_Dt


@pytest.fixture(scope="module", autouse=True)
def _fp64():
    set_policy(PrecisionPolicy.fp64())


# ---------------------------------------------------------------- the module
def test_explicit_and_implicit_restoring_are_the_two_distinct_statements():
    """An INDEPENDENT transcription of both forms, on synthetic arrays.

    This pins what the two flag values mean, so the card-level test below is
    about which one the card selects rather than about the algebra.
    """
    import jax.numpy as jnp
    from legoesm.ocean.physics.surface_forcing.config import RestoringConfig
    from legoesm.ocean.physics.surface_forcing.restoring import (
        restoring_surface_forcing,
    )

    rng = np.random.default_rng(20260911)
    T = jnp.asarray(rng.uniform(-1.0, 28.0, (5, 4, 6)))
    S = jnp.asarray(rng.uniform(33.0, 37.5, (5, 4, 6)))
    T_star = jnp.asarray(rng.uniform(-1.0, 28.0, (5, 4)))
    S_star = jnp.asarray(rng.uniform(33.0, 37.5, (5, 4)))
    tau_T, tau_S = 1.038121e6, 2.696309e6

    class _Grid:
        grid_lat = np.zeros((5, 4))

    def _run(implicit):
        return restoring_surface_forcing(
            T, S, _Grid(),
            RestoringConfig(tau_T=tau_T, tau_S=tau_S, T_star_array=T_star,
                            S_star_array=S_star, subtract_qsr=False,
                            implicit=implicit),
            dt=DT)

    expl, impl = _run(False), _run(True)
    # NEMO's form: the flux is evaluated and applied, with no dt in sight.
    assert np.array_equal(np.asarray(expl.dS_dt[..., 0]),
                          np.asarray((S_star - S[..., 0]) / tau_S))
    assert np.array_equal(np.asarray(expl.dT_dt[..., 0]),
                          np.asarray((T_star - T[..., 0]) / tau_T))
    # The implicit form is the same thing with tau -> tau + dt.
    assert np.array_equal(np.asarray(impl.dS_dt[..., 0]),
                          np.asarray((S_star - S[..., 0]) / (tau_S + DT)))
    # ... and the two are genuinely different objects, so a test that cannot
    # tell them apart is broken rather than lucky.
    assert not np.array_equal(np.asarray(expl.dS_dt), np.asarray(impl.dS_dt))


# ------------------------------------------------------------------ the card
def _dino_card(recipe="nemo_dino_kamm_mlf"):
    from legoesm.ocean.experiments import dino as dm
    cfg = dm.dino_config_for_recipe(recipe)
    grid = dm.dino_lat_lon_grid(cfg)
    z = dm.dino_lat_lon_vertical(grid, cfg)
    state = dm.dino_lat_lon_state(grid, z, cfg)
    forcing = dm.dino_lat_lon_surface_forcing_arrays(grid, cfg)
    return dm, cfg, grid, z, state, forcing


def test_the_dino_applicator_selects_nemos_explicit_form():
    """The shipped applicator's level-0 SALT rate is ``(S* - S)/tau_S``.

    Salt is the clean test because it has no solar member: the whole level-0
    salt rate is the restoring statement, so the implicit denominator cannot
    hide behind ``Q_sr``.  The card runs ``surface_flux_divisor='nemo_live'``,
    and from rest ``eta = 0`` so the live stretch is exactly 1 and the scalar
    ``tau_S`` is the card's own value.
    """
    from legoesm.ocean.physics.surface_forcing.config import (
        tau_from_flux_coefficient,
    )
    dm, cfg, grid, z, state, forcing = _dino_card()
    _, (_, dS) = dm.apply_dino_lat_lon_surface_forcing(
        state, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    tau_S = float(tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0,
                                            float(z.dz_ref[0])))
    S0 = np.asarray(state.S.data)[..., 0]
    S_star = np.asarray(forcing["S_star_2d"])
    wet = np.asarray(state.land_mask.data) > 0.5
    got = np.asarray(dS)[..., 0][wet]
    expl = ((S_star - S0) / tau_S)[wet]
    impl = ((S_star - S0) / (tau_S + DT))[wet]
    scale = float(np.abs(expl).max())
    assert scale > 0.0
    assert float(np.abs(got - expl).max()) / scale < 1e-12, (
        "the applicator is NOT on NEMO's explicit form")
    # NON-VACUITY: the implicit form is 1.0e-03 away, thousands of times the
    # tolerance above, so the assertion above could have failed.
    assert float(np.abs(got - impl).max()) / scale > 1e-4


def test_the_implicit_form_can_no_longer_be_reached_through_the_dino_card():
    """This test used to SIZE the move away from the implicit denominator as
    exactly ``dt/(tau_S+dt)``.  That sizing is now impossible to perform, and
    the reason is the stronger statement: the DINO applicator hands the
    restoring module NEMO's own association (``nemo_trasbc=``), which has no
    damping denominator anywhere in ``usrdef_sbc.f90`` or ``trasbc.f90``, so
    asking for the implicit form through this card RAISES instead of silently
    returning it.  A knob whose wrong setting is unreachable is a better
    guarantee than a knob whose wrong setting merely measures differently.

    The original sizing survives, on synthetic arrays and with no card
    involved, in ``test_explicit_and_implicit_restoring_are_the_two_distinct_
    statements`` above.
    """
    import legoesm.ocean.physics.surface_forcing.restoring as restmod
    dm, cfg, grid, z, state, forcing = _dino_card()
    real = restmod.restoring_surface_forcing

    def _implicit_arm(T, S, g, c, **kw):
        return real(T, S, g, c._replace(implicit=True), **kw)

    restmod.restoring_surface_forcing = _implicit_arm
    try:
        with pytest.raises(ValueError, match="implicit=True"):
            dm.apply_dino_lat_lon_surface_forcing(
                state, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    finally:
        restmod.restoring_surface_forcing = real

    # Non-vacuity: the UNPATCHED card must still run, or the raise above
    # would be indistinguishable from the card being broken outright.
    _, (_, dS) = dm.apply_dino_lat_lon_surface_forcing(
        state, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    assert float(np.abs(np.asarray(dS)).max()) > 0.0


def test_the_box_heat_budget_mirror_did_not_stay_on_the_implicit_form():
    """``box_heat_budget`` re-implements the applicator's restoring, so it is a
    diagnostic that can silently stop describing the model it budgets.

    This reads the AST of that ONE call's ``implicit=`` keyword -- not a string
    grep and not ``inspect.getsource`` of a wrapper.  What it CANNOT see: any
    other field drifting between the two builders.  That is stated rather than
    implied.
    """
    import legoesm.ocean.fidelity.box_heat_budget as bh
    src = ast.parse(open(bh.__file__).read())
    found = [kw.value.value
             for node in ast.walk(src)
             if isinstance(node, ast.Call)
             and getattr(node.func, "id", None) == "RestoringConfig"
             for kw in node.keywords
             if kw.arg == "implicit" and isinstance(kw.value, ast.Constant)]
    assert found == [False], (
        f"box_heat_budget's RestoringConfig implicit= keywords are {found}; "
        "the applicator runs NEMO's explicit form and the mirror must too")


def test_the_card_sweep_is_committed_and_runs_every_caller():
    """Rule 12's sweep exists, and it enumerates callers rather than one card."""
    here = os.path.dirname(os.path.abspath(__file__))
    repo = os.path.abspath(os.path.join(here, "..", "..", ".."))
    p = os.path.join(repo, "scripts", "validate", "ocean_fidelity", "dino_1226",
                     "surface_restoring_card_sweep.py")
    assert os.path.exists(p)
    src = open(p).read()
    from legoesm.ocean.experiments.dino import DINO_RECIPES
    assert "DINO_RECIPES" in src and "neverworld2_lite" in src, (
        "the sweep must cover every caller of the shared applicator")
    assert len(DINO_RECIPES) >= 7


# ------------------------------------------------- NEMO's ASSOCIATION, not
# ------------------------------------------------- merely NEMO's algebra
def _nemo_dz0_live(dm, cfg, z, state):
    """The divisor trasbc.f90:170 uses, resolved from the card (Rule 10)."""
    dz0 = float(np.asarray(z.dz_ref)[0])
    divisor = getattr(cfg, "surface_flux_divisor", "static")
    if divisor == "static":
        return dz0
    if divisor == "nemo_live":
        from legoesm.ocean.eos import nemo_r3t_stretch
        return dz0 * nemo_r3t_stretch(z, state.eta.data, state.H_bathy.data)
    raise AssertionError(f"unhandled surface_flux_divisor {divisor!r}")


def test_the_salt_restoring_is_nemos_association_bit_for_bit():
    """The DINO card's level-0 salt tendency must equal an INDEPENDENT
    transcription of NEMO's three statements with ZERO cells unequal.

    ``sfx = ( rn_srp*( ts(:,:,1,jp_sal,Kbb) - zsstar ) )*tmask``
                                                    usrdef_sbc.f90:388
    ``sbc_tsc(:,:,jp_sal) = r1_rho0*sfx``           trasbc.f90:153
    ``pts(:,:,1,Krhs) += zfact*( sbc_tsc_b + sbc_tsc )/e3t(1)``
                                                    trasbc.f90:169-170
    with ``zfact = 1`` and ``sbc_tsc_b = 0`` at ``kt = nit000``
    (``trasbc.f90:139-143``).  The salt row is the clean one: no solar member,
    no concentration/dilution member (``rn_emp_prop = 0``), so what is left is
    exactly the association.
    """
    # fp64 comes from the module-scoped autouse fixture
    dm, cfg, grid, z, state, forcing = _dino_card()
    _, (_, dS) = dm.apply_dino_lat_lon_surface_forcing(
        state, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    S0 = np.asarray(state.S.data)[..., 0]
    S_star = np.asarray(forcing["S_star_2d"])
    mask = np.asarray(state.land_mask.data)
    dz0_live = np.asarray(_nemo_dz0_live(dm, cfg, z, state))
    sfx = (-cfg.A_S) * (S0 - S_star)
    want = (1.0 / cfg.rho_0) * sfx / dz0_live * mask
    got = np.asarray(dS)[..., 0]
    bad = int((got != want).sum())
    assert bad == 0, (
        f"{bad} of {got.size} level-0 salt cells differ from NEMO's "
        f"association; max|d| = {float(np.abs(got - want).max()):.3e}")


def test_the_timescale_association_would_not_pass_that_test():
    """Non-vacuity for the test above: the SAME algebra written as
    ``-(S - S*)/tau`` disagrees in the last bits, which is the entire reason
    the association is transcribed rather than re-derived.  If this arm also
    matched, the assertion above would be measuring nothing."""
    # fp64 comes from the module-scoped autouse fixture
    from legoesm.ocean.physics.surface_forcing.config import (
        tau_from_flux_coefficient,
    )
    dm, cfg, grid, z, state, forcing = _dino_card()
    S0 = np.asarray(state.S.data)[..., 0]
    S_star = np.asarray(forcing["S_star_2d"])
    dz0_live = np.asarray(_nemo_dz0_live(dm, cfg, z, state))
    nemo = (1.0 / cfg.rho_0) * ((-cfg.A_S) * (S0 - S_star)) / dz0_live
    tau_S = np.asarray(
        tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0, dz0_live))
    timescale = -(S0 - S_star) / tau_S
    differ = int((nemo != timescale).sum())
    assert differ > 0, (
        "the two associations agree bit-for-bit on every cell, so the "
        "bit-for-bit test above cannot fail and proves nothing")
    assert float(np.abs(nemo - timescale).max()) < 1e-15 * float(
        np.abs(nemo).max() + 1e-300) * 1e3, (
        "the two arms differ by more than rounding -- that is a physics "
        "change, not an association change, and the wrong thing was edited")


def test_nemo_trasbc_refuses_the_two_shapes_it_cannot_reproduce():
    """The NEMO association folds ``zqsr_dayMean`` INSIDE ``qns``
    (usrdef_sbc.f90:438) and has no damping denominator anywhere, so the two
    configurations it cannot express must RAISE rather than silently return
    the timescale form."""
    # fp64 comes from the module-scoped autouse fixture
    import jax.numpy as jnp
    from legoesm.ocean.physics.surface_forcing.config import RestoringConfig
    from legoesm.ocean.physics.surface_forcing.restoring import (
        restoring_surface_forcing,
    )

    class _G:
        grid_lat = np.zeros((3, 4))

    T = jnp.zeros((3, 4, 2))
    S = jnp.zeros((3, 4, 2))
    tup = (-40.0, -3.858e-3, 1.0 / (1026.0 * 3991.0), 1.0 / 1026.0, 10.0)
    base = dict(T_star_array=np.zeros((3, 4)), S_star_array=np.zeros((3, 4)),
                tau_T=1.0e6, tau_S=1.0e6)
    with pytest.raises(ValueError, match="implicit=True"):
        restoring_surface_forcing(
            T, S, _G(), RestoringConfig(subtract_qsr=True, implicit=True,
                                        **base),
            sw_down=np.zeros((3, 4)), dt=1.0, rho_0=1026.0, c_p=3991.0,
            dz_0=10.0, nemo_trasbc=tup)
    with pytest.raises(ValueError, match="subtract_qsr=True"):
        restoring_surface_forcing(
            T, S, _G(), RestoringConfig(subtract_qsr=False, implicit=False,
                                        **base),
            rho_0=1026.0, c_p=3991.0, dz_0=10.0, nemo_trasbc=tup)



def _seasonal_targets(dm, cfg, forcing, z, state):
    """``(T_star, Q_sr)`` at ``t = DT``, the way the applicator resolves them.

    ``ln_ann_cyc = .true.`` on this card (``namelist_cfg:35``), so both are
    recomputed per step from the 360-day-year phase; the static arrays in
    ``forcing`` are the annual means and restoring toward them would be a
    different experiment."""
    import jax.numpy as jnp
    T_star = forcing["T_star_2d"]
    Q_sr = forcing["Q_sr_2d"]
    if getattr(cfg, "forcing_annual_cycle", False):
        lat1 = forcing["lat_deg_1d"]
        shape = T_star.shape
        T_star = jnp.broadcast_to(
            dm.dino_T_star_seasonal(lat1, DT, cfg)[:, None], shape)
        Q_sr = jnp.broadcast_to(
            dm.dino_Q_sr_seasonal(lat1, DT, cfg)[:, None], shape)
    return np.asarray(T_star), np.asarray(Q_sr)

def test_the_heat_restoring_is_nemos_association_bit_for_bit():
    """The T twin of the salt test above, and it exists because a diff
    reviewer PROVED the salt test alone was not enough: with the temperature
    anomaly INVERTED (``T_star - T``, i.e. Newtonian damping turned into
    anti-restoring) the entire 50-test DINO fidelity selection still passed and
    the surface gate's printed verdict was byte-identical.  The T half of the
    new block had zero bit-for-bit coverage.

    NEMO's statement, with ``rn_emp_prop = 0`` so the concentration/dilution
    member is identically zero:

        qns = ( rn_trp*( ts(:,:,1,jp_tem,Kbb) - ztstar )
                - zqsr_dayMean ) * tmask          usrdef_sbc.f90:436-438
        sbc_tsc(jp_tem) = r1_rho0_rcp * qns       trasbc.f90:152
        pts(:,:,1,Krhs) += sbc_tsc / e3t(:,:,1)   trasbc.f90:169-170  (kt=nit000)

    The card's level-0 tendency is that PLUS ``tra_qsr``'s own level-0
    penetration, so the solar member is taken from the model's own
    ``shortwave_penetration_tendency`` -- this test is about the RESTORING's
    association, and separating the two would require re-deriving Jerlov.
    WHAT IT CANNOT SEE, written down rather than discovered later: the solar
    member's own association (that is ``traqsr.F90``, a different routine and
    the next statement), and the divisor's VALUE -- from rest the live stretch
    is exactly 1, so a static divisor would pass.
    """
    # fp64 comes from the module-scoped autouse fixture
    import jax.numpy as jnp
    from legoesm.ocean.physics.shortwave_penetration import (
        ShortwavePenetrationConfig, shortwave_penetration_tendency,
    )
    dm, cfg, grid, z, state, forcing = _dino_card()
    _, (dT, _) = dm.apply_dino_lat_lon_surface_forcing(
        state, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)

    T0 = np.asarray(state.T.data)[..., 0]
    # ln_ann_cyc: T* and Q_sr are TIME-DEPENDENT on this card, so the static
    # arrays are NOT what the applicator restores toward.  The salt twin above
    # needs no such branch -- S* carries no annual cycle.
    T_star, Q_sr = _seasonal_targets(dm, cfg, forcing, z, state)
    mask = np.asarray(state.land_mask.data)
    dz0_live = np.asarray(_nemo_dz0_live(dm, cfg, z, state))

    qns = (-cfg.A_theta) * (T0 - T_star) - Q_sr          # usrdef_sbc.f90:436-438
    nemo_T = (1.0 / (cfg.rho_0 * cfg.c_p)) * qns / dz0_live   # trasbc.f90:152,:170

    # THE SOLAR MEMBER MUST BE BUILT THE WAY THE CARD BUILDS IT.  This used to
    # branch on `ladder == "nemo_live"` and fall through to the STATIC ladder
    # for anything else, so the moment the card moved to "nemo_2bd" the test
    # rebuilt a different solar member than the applicator runs and went red
    # on 3450 level-0 cells at 8.5e-22 -- a stale string compare, not a
    # defect in the applicator.  A diff reviewer caught it.  The branch now
    # ENUMERATES every value the applicator accepts and RAISES on a new one,
    # so the next ladder cannot silently reintroduce the same failure.
    ladder = getattr(cfg, "shortwave_penetration_ladder", "static")
    sw_kw = {}
    if ladder == "static":
        z_half_stretch = None
    elif ladder in ("nemo_live", "nemo_2bd"):
        from legoesm.ocean.eos import nemo_r3t_stretch
        z_half_stretch = nemo_r3t_stretch(z, state.eta.data,
                                          state.H_bathy.data)
        if ladder == "nemo_2bd":
            from legoesm.ocean.physics.shortwave_penetration import (
                nemo_qsr_ext_lev)
            wet3 = jnp.asarray(z.is_active) > 0.5
            sw_kw["nemo_2bd_levels"] = nemo_qsr_ext_lev(
                z, wet3, rdt=2.0 * DT, rho_0=cfg.rho_0, c_sw=cfg.c_p)
            sw_kw["cell_wet"] = wet3
    else:
        raise AssertionError(
            f"this test does not know how to build the solar member for "
            f"shortwave_penetration_ladder={ladder!r}; add it here rather "
            "than letting the test rebuild a different one")
    sw = np.asarray(shortwave_penetration_tendency(
        sw_down=jnp.asarray(Q_sr),
        z_coord_dz_ref=z.dz_ref, z_coord_z_half_ref=z.z_half_ref,
        jacobian=jnp.ones_like(state.eta.data),
        config=ShortwavePenetrationConfig(water_type=cfg.jerlov_water_type),
        rho_0=cfg.rho_0, c_sw=cfg.c_p, z_half_stretch=z_half_stretch,
        **sw_kw))

    want = (sw[..., 0] + nemo_T) * mask
    got = np.asarray(dT)[..., 0]
    bad = int((got != want).sum())
    assert bad == 0, (
        f"{bad} of {got.size} level-0 heat cells differ from NEMO's "
        f"association; max|d| = {float(np.abs(got - want).max()):.3e}")


def test_the_inverted_heat_anomaly_would_not_pass_that_test():
    """Non-vacuity for the T test, planting the exact defect the reviewer got
    past everything else: NEMO writes ``rn_trp*( ts - ztstar )`` with
    ``rn_trp`` NEGATIVE (namelist_cfg:37, "must be negative"), so the
    restoring member has the sign of ``-(T - T*)``.  Writing ``(T* - T)``
    with the same negative coefficient inverts it into anti-restoring, and
    the assertion above must be able to tell."""
    # fp64 comes from the module-scoped autouse fixture
    dm, cfg, grid, z, state, forcing = _dino_card()
    T0 = np.asarray(state.T.data)[..., 0]
    T_star, Q_sr = _seasonal_targets(dm, cfg, forcing, z, state)
    dz0_live = np.asarray(_nemo_dz0_live(dm, cfg, z, state))
    r1 = 1.0 / (cfg.rho_0 * cfg.c_p)
    right = r1 * ((-cfg.A_theta) * (T0 - T_star) - Q_sr) / dz0_live
    flipped = r1 * ((-cfg.A_theta) * (T_star - T0) - Q_sr) / dz0_live
    differ = int((right != flipped).sum())
    assert differ > 0, (
        "the inverted anomaly is bit-identical to the correct one on every "
        "cell, so the bit-for-bit heat test cannot see a sign flip")
    # and it is a PHYSICS-sized difference, not a rounding one -- if this were
    # small the test above would be pinning noise.
    scale = float(np.abs(right).max())
    assert float(np.abs(right - flipped).max()) > 1e-3 * scale


# ----------------------------------------------- decision 32: the TIME LEVEL
def test_the_restoring_flux_reads_the_before_tracer_not_the_now_tracer():
    """NEMO evaluates the restoring flux on ``Kbb``; legoESM read ``Nnn``.

    ``usrdef_sbc.f90:388`` (salt) and ``:436-438`` (heat) read
    ``ts(ji,jj,1,jn,Kbb)`` and nothing else, and ``sbc`` is reached as
    ``CALL sbc( kstp, Nbb, Nnn )`` (``stpmlf.f90:173``), so that ``Kbb`` IS
    the step's before level.

    The check is a DISPLACEMENT, not an inspection: the same state is scored
    twice, once with ``T_before``/``S_before`` unset and once with them set to
    a field that differs by a known constant.  If the applicator read the NOW
    tracer the two would be identical, which is exactly the synthetic
    violation this test exists to catch.
    """
    dm, cfg, grid, z, state, forcing = _dino_card()
    dT = 0.37                       # K, and psu for the salt twin
    dS = -0.11
    _, rate_now = dm.apply_dino_lat_lon_surface_forcing(
        state, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    displaced = state._replace(
        T_before=state.T.replace(data=state.T.data + dT),
        S_before=state.S.replace(data=state.S.data + dS))
    _, rate_bef = dm.apply_dino_lat_lon_surface_forcing(
        displaced, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)

    t_now = np.asarray(rate_now[0])[..., 0]
    t_bef = np.asarray(rate_bef[0])[..., 0]
    s_now = np.asarray(rate_now[1])[..., 0]
    s_bef = np.asarray(rate_bef[1])[..., 0]
    wet = np.asarray(state.land_mask.data) > 0.5
    assert wet.sum() > 1000
    assert not np.array_equal(t_now[wet], t_bef[wet]), (
        "displacing T_before by 0.37 K left the heat rate unchanged: the "
        "applicator is still reading the NOW tracer (usrdef_sbc.f90:436)")
    assert not np.array_equal(s_now[wet], s_bef[wet]), (
        "displacing S_before by 0.11 psu left the salt rate unchanged: the "
        "applicator is still reading the NOW tracer (usrdef_sbc.f90:388)")

    # And the SIZE is the statement's own: NEMO's flux is linear in the
    # tracer, so a constant displacement moves the rate by exactly
    # r1_rho0_rcp*rn_trp*dT/e3t(1) (heat) and r1_rho0*rn_srp*dS/e3t(1) (salt).
    dz0 = float(np.asarray(z.dz_ref)[0])
    want_t = (1.0 / (cfg.rho_0 * cfg.c_p)) * (-cfg.A_theta) * dT / dz0
    want_s = (1.0 / cfg.rho_0) * (-cfg.A_S) * dS / dz0
    got_t = (t_bef - t_now)[wet]
    got_s = (s_bef - s_now)[wet]
    assert np.allclose(got_t, want_t, rtol=1e-12, atol=0.0), (
        f"heat rate moved by {got_t.mean():.6e} K/s, NEMO's statement says "
        f"{want_t:.6e}")
    assert np.allclose(got_s, want_s, rtol=1e-12, atol=0.0), (
        f"salt rate moved by {got_s.mean():.6e} psu/s, NEMO's statement says "
        f"{want_s:.6e}")


def test_a_state_with_no_before_level_is_byte_unchanged_by_decision_32():
    """Rule 12: a card whose outer integrator carries no before level.

    ``T_before``/``S_before`` default to ``None`` (``state.py:607-608``) and
    only the leap-frog integrator populates them, so every non-leap-frog
    caller of the shared applicator must be BIT-IDENTICAL.  This pins that as
    a measurement rather than an argument, on the real applicator.
    """
    dm, cfg, grid, z, state, forcing = _dino_card()
    assert state.T_before is None and state.S_before is None
    _, rate = dm.apply_dino_lat_lon_surface_forcing(
        state, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    # The NOW tracer is the fallback, so setting the before level to the NOW
    # field reproduces it BIT for BIT -- the Euler-start case NEMO has at
    # kt=nit000, where usr_def_istate has filled both slots identically.
    same = state._replace(T_before=state.T, S_before=state.S)
    _, rate2 = dm.apply_dino_lat_lon_surface_forcing(
        same, forcing, z, cfg, DT, t_seconds=DT, return_rate=True)
    for a, b, tag in ((rate[0], rate2[0], "T"), (rate[1], rate2[1], "S")):
        d = np.abs(np.asarray(a) - np.asarray(b))
        assert d.max() == 0.0, (
            f"{tag}: the None fallback is not the NOW tracer (max {d.max():.3e})")
