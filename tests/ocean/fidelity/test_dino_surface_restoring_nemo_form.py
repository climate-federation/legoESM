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


def test_the_move_away_from_the_implicit_form_is_exactly_dt_over_tau_plus_dt():
    """Size check: the statement removed is a known scalar factor, not a
    reformulation.  If the two arms differ by anything other than
    ``dt/(tau_S+dt)`` the flag reached something besides the restoring term."""
    from legoesm.ocean.physics.surface_forcing.config import (
        tau_from_flux_coefficient,
    )
    import legoesm.ocean.physics.surface_forcing.restoring as restmod
    dm, cfg, grid, z, state, forcing = _dino_card()
    real = restmod.restoring_surface_forcing

    def _implicit_arm(T, S, g, c, **kw):
        return real(T, S, g, c._replace(implicit=True), **kw)

    kw = dict(t_seconds=DT, return_rate=True)
    _, (_, dS_a) = dm.apply_dino_lat_lon_surface_forcing(
        state, forcing, z, cfg, DT, **kw)
    restmod.restoring_surface_forcing = _implicit_arm
    try:
        _, (_, dS_b) = dm.apply_dino_lat_lon_surface_forcing(
            state, forcing, z, cfg, DT, **kw)
    finally:
        restmod.restoring_surface_forcing = real
    tau_S = float(tau_from_flux_coefficient(cfg.A_S, cfg.rho_0, 1.0,
                                            float(z.dz_ref[0])))
    a, b = np.asarray(dS_a)[..., 0], np.asarray(dS_b)[..., 0]
    scale = float(np.abs(a).max())
    assert scale > 0.0
    measured = float(np.abs(a - b).max()) / scale
    predicted = DT / (tau_S + DT)
    assert abs(measured - predicted) <= 1e-9 * predicted, (
        f"measured {measured:.9e} vs dt/(tau+dt) {predicted:.9e}")


# ------------------------------------------------- the diagnostic that mirrors it
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
