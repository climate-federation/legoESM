"""Leuning (1995) stomatal conductance: kernel, dispatch, and quadratic closure.

Covers the new ``leuning_gs`` kernel (single home, ``land/stomata.py``) against
an independent scalar oracle of Leuning 1995 eq. 8, the big-leaf and two-leaf
dispatch branches, CLI selectability, and the analytic surface-VPD quadratic
used by the CLM-ML backend branches (root of
``gs^2 + gs*(gbv*fD - g0 - A') - gbv*(g0*fD + A') = 0`` must satisfy the
implicit ``gs = g0 + A'/(1 + Ds/D0)`` with ``Ds = gbv*Da/(gbv+gs)``, and reduce
to the explicit kernel as gbv -> inf).
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm.land import stomata as sto
from legoesm.land.stomata import StomataConfig, leuning_gs


def test_leuning_gs_matches_scalar_oracle():
    A, D, Cs, gamma, a1, d0, g0 = 12.0, 1.2, 400.0, 42.0, 9.0, 1.5, 0.01
    got = float(leuning_gs(jnp.asarray(A), jnp.asarray(D), jnp.asarray(Cs),
                           jnp.asarray(gamma), a1, d0, g0))
    want = g0 + a1 * A / ((Cs - gamma) * (1.0 + D / d0))
    assert got == pytest.approx(want, rel=1e-12)


def test_leuning_gs_monotone_and_floored():
    A = jnp.asarray(10.0)
    gs_wet = float(leuning_gs(A, jnp.asarray(0.5), jnp.asarray(400.0),
                              jnp.asarray(42.0), 9.0, 1.5, 0.01))
    gs_dry = float(leuning_gs(A, jnp.asarray(3.0), jnp.asarray(400.0),
                              jnp.asarray(42.0), 9.0, 1.5, 0.01))
    assert gs_dry < gs_wet  # stomata close with VPD
    gs_dark = float(leuning_gs(jnp.asarray(-2.0), jnp.asarray(1.0),
                               jnp.asarray(400.0), jnp.asarray(42.0),
                               9.0, 1.5, 0.01))
    assert gs_dark == pytest.approx(0.01)  # g0 floor at negative A
    g = jax.grad(lambda d: leuning_gs(A, d, jnp.asarray(400.0),
                                      jnp.asarray(42.0), 9.0, 1.5, 0.01))(1.0)
    assert np.isfinite(float(g)) and float(g) < 0.0


def test_bigleaf_dispatch_accepts_leuning_and_rejects_typo():
    cfg = StomataConfig(enabled=True, stomata_model="leuning")
    out = sto.solve_coupled_farquhar_ci(
        T_leaf=jnp.full((2,), 298.15), sw_down=jnp.full((2,), 600.0),
        co2_ppmv=400.0, q_air=jnp.full((2,), 0.008),
        p_surface=jnp.full((2,), 101325.0), LAI=jnp.full((2,), 3.0),
        beta_soil=jnp.ones((2,)), config=cfg)
    assert np.all(np.isfinite(np.asarray(out.gs)))
    assert float(out.gs[0]) > 0.0
    with pytest.raises(ValueError, match="stomata_model"):
        sto.solve_coupled_farquhar_ci(
            T_leaf=jnp.full((1,), 298.15), sw_down=jnp.full((1,), 600.0),
            co2_ppmv=400.0, q_air=jnp.full((1,), 0.008),
            p_surface=jnp.full((1,), 101325.0), LAI=jnp.full((1,), 3.0),
            beta_soil=jnp.ones((1,)), config=StomataConfig(stomata_model="luening"))


def test_two_leaf_gs_branch_uses_kernel():
    from legoesm.land.canopy.energy_balance import _compute_gs_and_ci
    from legoesm.land.canopy.photosynthesis import co2_compensation_point
    An = jnp.asarray(10.0); Ca = jnp.asarray(400.0); Tf = jnp.asarray(298.15)
    _, gs_ms, _ = _compute_gs_and_ci(
        An, jnp.asarray(0.7), jnp.asarray(1200.0), Ca, Tf,
        jnp.asarray(101325.0), jnp.asarray(9.0), jnp.asarray(0.01),
        "leuning", d0_leuning_kpa=1.5)
    assert np.isfinite(float(gs_ms)) and float(gs_ms) > 0.0
    # Missing D0 with leuning is a loud error, not a silent default.
    with pytest.raises(ValueError, match="d0_leuning_kpa"):
        _compute_gs_and_ci(
            An, jnp.asarray(0.7), jnp.asarray(1200.0), Ca, Tf,
            jnp.asarray(101325.0), jnp.asarray(9.0), jnp.asarray(0.01),
            "leuning")
    # CanopyConfig accepts the new model and still cross-checks the P-model g1.
    from legoesm.land.canopy.config import CanopyConfig
    CanopyConfig(stomatal_model="leuning").validate()
    # leuning + g1_source='p_model' validates (phydro's slope mapping covers
    # it); the least-cost-only combination is refused at the consumers.
    CanopyConfig(stomatal_model="leuning", g1_source="p_model").validate()


def test_clm_ml_quadratic_closes_the_implicit_leuning_equation():
    """The backend's derived quadratic (see MLLeafPhotosynthesisMod gs_type==3)
    must return the root of the IMPLICIT Leuning equation with the boundary
    layer folded into the surface VPD, and reduce to the explicit kernel as
    gbv -> infinity."""
    rng = np.random.default_rng(0)
    for _ in range(50):
        g0 = rng.uniform(0.005, 0.05)
        a1 = rng.uniform(2.0, 15.0)
        d0_pa = rng.uniform(500.0, 3000.0)
        an = rng.uniform(0.5, 30.0)
        cs_minus_cp = rng.uniform(100.0, 400.0)
        gbv = rng.uniform(0.2, 5.0)
        da = rng.uniform(0.0, 4000.0)
        fD = 1.0 + da / d0_pa
        aprime = a1 * an / cs_minus_cp
        bq = gbv * fD - g0 - aprime
        cq = -gbv * (g0 * fD + aprime)
        gs = (-bq + math.sqrt(bq * bq - 4.0 * cq)) / 2.0
        # implicit closure: gs == g0 + A'/(1 + Ds/D0), Ds = gbv*Da/(gbv+gs)
        ds = gbv * da / (gbv + gs)
        resid = gs - (g0 + aprime / (1.0 + ds / d0_pa))
        assert abs(resid) < 1e-10 * max(gs, 1.0)
    # gbv -> inf limit equals the explicit kernel form
    gbv = 1e8
    bq = gbv * fD - g0 - aprime
    cq = -gbv * (g0 * fD + aprime)
    gs_inf = (-bq + math.sqrt(bq * bq - 4.0 * cq)) / 2.0
    assert gs_inf == pytest.approx(g0 + aprime / fD, rel=1e-6)


def test_clm_ml_config_and_map_accept_leuning():
    from legoesm.land.canopy.config import (
        CLM_ML_STOMATAL_GS_TYPE, CLMMLCanopyConfig)
    assert CLM_ML_STOMATAL_GS_TYPE["leuning"] == 3
    cfg = CLMMLCanopyConfig(stomatal_model="leuning")
    assert cfg.a1_leuning == 9.0 and cfg.d0_leuning_kpa == 1.5
    from legoesm.land.canopy.clm_ml_interface import _apply_stomatal_model
    import legoesm.land.canopy.clm_ml_backend.multilayer_canopy.MLLeafPhotosynthesisMod as _photo
    _apply_stomatal_model(cfg)
    assert _photo.gs_type == 3
    assert _photo.leuning_a1 == 9.0
    assert _photo.leuning_d0_pa == 1500.0
    # restore default wue so process-global state does not leak to other tests
    _apply_stomatal_model(CLMMLCanopyConfig())
    assert _photo.gs_type == 2
