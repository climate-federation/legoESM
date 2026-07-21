"""SIMTOP-style TOPMODEL runoff FAITHFULNESS + conservation tests.

``land/topmodel_runoff.py`` implements a SIMTOP-style TOPMODEL runoff (Niu et al.
2005; the response laws used by CLM4.5).  The existing ``test_topmodel_runoff.py``
is behavioral — deeper-when-drier, decay-with-depth, conservation via
``allclose`` — it never pins the closed FORMS against a separate reference nor
records the numeric departures from CLM4.5.

What these tests establish, in order of authority:

1. TRUTH TIER — the DEFINING water-budget closure of the full step
   (``P_input == dW/dt + evap + runoff``, ``runoff == r_surf + r_base``) to a
   tight pin on every ``limit_evaporation=True`` path (Hortonian, Dunne-overflow,
   evap-cap, baseflow-cap, and the ``infiltration_excess=False`` branch) — the
   setting under which the module promises closure.  This outranks form-matching
   per the CLAUDE.md truth-tier doctrine.

2. WHOLE-STEP COMPONENT PIN — a SEPARATE scalar reference implementation of the
   entire partition (``_partition_oracle``, plain Python, no jnp; re-derived from
   the module's physical spec) pins all FIVE outputs
   ``(W_new, evap, runoff, r_surf, r_base)`` across both evaporation-limiting
   settings — so a coherently mis-split budget (e.g. baseflow dropped but water
   retained in ``W_new``, which conservation alone cannot catch) is caught by the
   surface/base decomposition.  It is not a fully independent oracle (it mirrors
   the algorithm's operation order), so it is complementary to — not a substitute
   for — the physical conservation pin above, which is algorithm-agnostic.

3. RESPONSE-LAW FORMS — ``f_sat(z_wt)`` and ``q_drai(z_wt)`` pinned to high
   precision (rel 1e-13) for BOTH the default config and a deliberately
   ASYMMETRIC non-default config (so an implementation that hard-codes the
   defaults instead of reading ``config`` fails).

Departures from CLM4.5 (documented, and pinned as canaries — NOT silent):
  * f_sat decay convention: CLM4.5 §7 eq. 7.4 writes ``f_sat = f_max*exp(-0.5*
    f_over_CLM*z)`` with ``f_over_CLM = 0.5 m^-1`` (effective decay 0.25 m^-1).
    legoESM parameterizes the EFFECTIVE decay directly, ``f_sat =
    f_max*exp(-f_over*z)``; its default ``f_over = 0.5`` is therefore a factor of
    TWO steeper than CLM's effective decay.  ``f_over``/``f_drai`` are TUNABLE
    (tier-2) knobs, so the numeric defaults are legoESM choices, not CLM values.
    ``test_fsat_departs_from_clm_by_factor_two`` pins this factor explicitly.
  * baseflow omits CLM's frozen-soil impedance ``(1 - f_ice)`` (this is an
    unfrozen-slab specialization); ``q_drai_max`` default is a legoESM choice,
    not CLM's ``~5.5e-3`` value.
  * ``z_wt = z_wt_max*(1 - W/W_max)`` is a legoESM slab CLOSURE mapping the
    single-bucket store onto a water-table depth (there is no layer-resolved soil
    column here).  It is NOT claimed literature-faithful — only internally
    consistent — and is tier-0/excluded in ``__param_spec__``.  The gSAM on-disk
    ``SLM/runoff.f90`` is a DIFFERENT scheme (2D shallow-water flood routing),
    NOT SIMTOP, so it is not the oracle either.

Reference coefficients: the ``_O_*`` literals (legoESM effective-rate
convention) are canaried against ``TopmodelConfig`` (so they cannot be sourced
from the module); the raw CLM value is a separate ``_CLM_*`` literal used only to
pin the departure.
"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import pytest

_ENTRY_X64 = jax.config.read("jax_enable_x64")


@pytest.fixture(autouse=True)
def _force_x64():
    """Per-test float64 for the high-precision pins; restore the process-entry
    state in finally so selecting a single test never leaks x64 into another
    module."""
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", _ENTRY_X64)


from legoesm import constants                                          # noqa: E402
from legoesm.land.topmodel_runoff import (                            # noqa: E402
    water_table_depth, saturated_area_fraction, baseflow,
    partition_topmodel_runoff, TopmodelConfig,
)

# legoESM SIMTOP-style oracle literals (EFFECTIVE-decay convention; canaried
# against TopmodelConfig below).
_O_F_MAX = 0.38          # max saturated area fraction [-]
_O_F_OVER = 0.5          # EFFECTIVE saturated-fraction decay [1/m]  (f_sat=f_max*exp(-f_over*z))
_O_Q_DRAI_MAX = 1.0e-3   # max baseflow [kg/m^2/s]   (legoESM default, NOT CLM's ~5.5e-3)
_O_F_DRAI = 2.5          # baseflow decay [1/m]
_O_Z_WT_MAX = 5.0        # slab water-table depth at zero storage [m]
_O_RHO_WATER = 1000.0    # constants.rho_water [kg/m^3] (used by the Green-Ampt infil cap)

# CLM4.5 §7 eq. 7.4 RAW coefficient — f_sat = f_max*exp(-0.5*f_over_CLM*z).  Used
# ONLY to pin the documented factor-2 departure of legoESM's effective decay.
_CLM_F_OVER_RAW = 0.5    # [1/m]; CLM effective decay = 0.5 * 0.5 = 0.25 /m

# Deliberately ASYMMETRIC non-default config: every field distinct so a form pin
# that passes here cannot be an implementation that ignores `config`.
_CFG_A = TopmodelConfig(f_max=0.271, f_over=0.83, q_drai_max=3.3e-4, f_drai=1.7, z_wt_max=3.4)


def _zwt_oracle(W, W_max, cfg):
    return cfg.z_wt_max * (1.0 - min(max(W / max(W_max, 1e-6), 0.0), 1.0))


def _fsat_oracle(z_wt, cfg):
    return cfg.f_max * math.exp(-cfg.f_over * max(z_wt, 0.0))


def _baseflow_oracle(z_wt, cfg):
    return cfg.q_drai_max * math.exp(-cfg.f_drai * max(z_wt, 0.0))


def _partition_oracle(W, P, evap, dt, W_max, Kinf, suction, cfg,
                      infil_excess=True, limit_evap=True):
    """Separate scalar reference implementation of the FULL step, re-derived from
    the physical spec (plain Python, no jnp): Dunne fraction sheds all rain;
    Hortonian sheds ``P - infil_cap`` over the unsaturated fraction; baseflow
    drains the exp law capped by availability; any excess over ``W_max`` becomes
    overflow.  Mirrors the algorithm's operation order (so it catches
    split/relabel/order/JAX-specific regressions, complementary to the
    algorithm-agnostic conservation pin).  Returns
    ``(W_new, evap_actual, runoff, r_surf, r_base)``."""
    rho = _O_RHO_WATER
    z = _zwt_oracle(W, W_max, cfg)
    f_sat = _fsat_oracle(z, cfg)
    deficit = min(max(1.0 - W / max(W_max, 1e-6), 0.0), 1.0)
    infil_cap = Kinf * rho * (1.0 + suction * deficit)
    q_horton = max(P - infil_cap, 0.0) if infil_excess else 0.0
    r_surf = f_sat * P + (1.0 - f_sat) * q_horton
    infiltration = P - r_surf
    evap_a = min(evap, max(W / dt + infiltration, 0.0)) if limit_evap else evap
    demand = _baseflow_oracle(z, cfg)
    avail = max(W / dt + infiltration - evap_a, 0.0)
    r_base = min(demand, avail)
    W_unc = W + dt * (infiltration - evap_a - r_base)
    overflow = max(W_unc - W_max, 0.0) / dt
    W_new = min(max(W_unc, 0.0), W_max)
    r_surf = r_surf + overflow
    return W_new, evap_a, r_surf + r_base, r_surf, r_base


def _a(x):
    return jnp.array(float(x))


_CFG = TopmodelConfig()


# --- response-law forms (default AND asymmetric non-default config) ------------

@pytest.mark.parametrize("cfg", [_CFG, _CFG_A], ids=["default", "asymmetric"])
@pytest.mark.parametrize("W,W_max", [(30.0, 100.0), (0.0, 100.0), (100.0, 100.0),
                                     (120.0, 100.0), (-10.0, 100.0)])
def test_water_table_depth_form(cfg, W, W_max):
    """z_wt = z_wt_max*(1 - clip(W/W_max,0,1)): clamped to [0, z_wt_max]
    (W>=W_max -> 0; W<=0 -> z_wt_max).  legoESM slab closure, internal
    consistency (not a literature-faithfulness claim)."""
    got = float(water_table_depth(_a(W), _a(W_max), cfg))
    assert got == pytest.approx(_zwt_oracle(W, W_max, cfg), rel=1e-13, abs=1e-15)


@pytest.mark.parametrize("cfg", [_CFG, _CFG_A], ids=["default", "asymmetric"])
@pytest.mark.parametrize("z_wt", [0.0, 1.0, 3.0, 5.0])
def test_saturated_area_fraction_form(cfg, z_wt):
    """f_sat = f_max*exp(-f_over*z_wt) [effective-decay convention], high precision."""
    got = float(saturated_area_fraction(_a(z_wt), cfg))
    assert got == pytest.approx(_fsat_oracle(z_wt, cfg), rel=1e-13, abs=0.0)


@pytest.mark.parametrize("cfg", [_CFG, _CFG_A], ids=["default", "asymmetric"])
@pytest.mark.parametrize("z_wt", [0.0, 1.0, 3.0, 5.0])
def test_baseflow_form(cfg, z_wt):
    """q_drai = q_drai_max*exp(-f_drai*z_wt), high precision."""
    got = float(baseflow(_a(z_wt), cfg))
    assert got == pytest.approx(_baseflow_oracle(z_wt, cfg), rel=1e-13, abs=0.0)


def test_topmodel_constants_match_config():
    c = TopmodelConfig()
    assert c.f_max == _O_F_MAX == 0.38
    assert c.f_over == _O_F_OVER == 0.5
    assert c.q_drai_max == _O_Q_DRAI_MAX == 1.0e-3
    assert c.f_drai == _O_F_DRAI == 2.5
    assert c.z_wt_max == _O_Z_WT_MAX == 5.0
    assert constants.rho_water == _O_RHO_WATER == 1000.0


def test_fsat_departs_from_clm_by_factor_two():
    """DOCUMENTED DEPARTURE (canary): legoESM's f_sat effective decay is a factor
    of TWO steeper than CLM4.5's.  CLM4.5 §7 eq. 7.4 is f_sat = f_max*exp(-0.5*
    f_over_CLM*z) with f_over_CLM=0.5 (effective 0.25/m); legoESM uses
    f_sat = f_max*exp(-f_over*z) with f_over=0.5 (effective 0.5/m).  This test
    fails LOUDLY if either convention is silently changed."""
    z = 2.0
    lego = float(saturated_area_fraction(_a(z), _CFG))
    clm = _O_F_MAX * math.exp(-0.5 * _CLM_F_OVER_RAW * z)     # CLM raw convention
    assert lego == pytest.approx(_O_F_MAX * math.exp(-_O_F_OVER * z), rel=1e-13)  # legoESM form
    assert lego != pytest.approx(clm, rel=1e-6)               # NOT CLM's convention
    # the ratio is exactly exp(-0.5*f_over_CLM*z) = exp(-0.25*z): a 2x-steeper decay
    assert lego / clm == pytest.approx(math.exp(-0.5 * _CLM_F_OVER_RAW * z), rel=1e-12)


# --- whole-step component pin (catches coherently mis-split budgets) -----------

@pytest.mark.parametrize("nm,W,P,evap,W_max,Kinf,suction,cfg,infx,lim", [
    ("horton",   50.0,   5.0e-2, 0.0,   100.0, 1e-5, 0.0, _CFG,   True,  True),   # Hortonian active, uncapped baseflow
    ("overflow", 99.5,   5.0e-2, 1e-4,  100.0, 1e-5, 3.0, _CFG,   True,  True),   # Dunne overflow branch
    ("cap",      0.0099, 0.0,    0.0,   0.01,  1e-5, 3.0, _CFG,   True,  True),   # baseflow availability cap binds
    ("evaplim",  0.5,    0.0,    1e-3,  100.0, 1e-5, 3.0, _CFG,   True,  True),   # evaporation cap binds
    ("booloff",  50.0,   5.0e-2, 0.0,   100.0, 1e-5, 0.0, _CFG,   False, True),   # infiltration_excess=False path
    ("evapfree", 0.5,    0.0,    1e-3,  100.0, 1e-5, 3.0, _CFG,   True,  False),  # limit_evaporation=False path
    ("nondef",   40.0,   2.0e-3, 5e-5,  120.0, 2e-5, 1.5, _CFG_A, True,  True),   # asymmetric config
])
def test_whole_step_matches_reference_impl(nm, W, P, evap, W_max, Kinf, suction, cfg, infx, lim):
    """All FIVE outputs pinned to the separate scalar reference implementation of
    the whole step.  Pins the surface/base SPLIT, not just the conserved sum, so a
    dropped/relabelled mechanism is caught where conservation alone would not."""
    out = partition_topmodel_runoff(_a(W), _a(P), _a(evap), 3600.0, _a(W_max),
                                    _a(Kinf), _a(suction), cfg, infx, lim)
    orc = _partition_oracle(W, P, evap, 3600.0, W_max, Kinf, suction, cfg, infx, lim)
    names = ("W_new", "evap", "runoff", "r_surf", "r_base")
    for k in range(5):
        assert float(out[k]) == pytest.approx(orc[k], rel=1e-12, abs=1e-13), \
            f"{nm}: {names[k]} {float(out[k])} != oracle {orc[k]}"


# --- conservation truth tier (every limit_evaporation=True case) ---------------

@pytest.mark.parametrize("W,P,evap,Kinf,suction,infx", [
    (30.0, 5.0e-4, 1.0e-4, 1e-5, 3.0, True),    # unsaturated, moderate rain
    (95.0, 2.0e-3, 5.0e-5, 1e-5, 3.0, True),    # near-saturated, heavy rain
    (99.5, 5.0e-2, 1.0e-4, 1e-5, 3.0, True),    # overfills W_max -> Dunne overflow branch
    (0.5,  0.0,    1.0e-3, 1e-5, 3.0, True),    # evap-cap binds (evap_demand >> W/dt)
    (60.0, 0.0,    1.0e-4, 1e-5, 3.0, True),    # no rain, draining
    (50.0, 5.0e-2, 0.0,    1e-5, 0.0, False),   # infiltration_excess=False path
])
def test_column_water_conservation(W, P, evap, Kinf, suction, infx):
    """DEFINING water-budget closure P == dW/dt + evap + runoff to a tight pin on
    each covered limit_evaporation=True case (values are O(1e-2), so a dropped
    flux O(>=1e-6) is far above the 1e-12 residual bound); runoff splits exactly
    into surface + base.  (limit_evaporation=False does not promise closure — evap
    is unthrottled — so it is covered by the whole-step component pin, not here.)"""
    dt, W_max = 3600.0, 100.0
    W_new, ev, runoff, r_surf, r_base = partition_topmodel_runoff(
        _a(W), _a(P), _a(evap), dt, _a(W_max), _a(Kinf), _a(suction), _CFG, infx, True)
    residual = float(P) - ((float(W_new) - W) / dt + float(ev) + float(runoff))
    assert residual == pytest.approx(0.0, abs=1e-12)
    assert float(runoff) == pytest.approx(float(r_surf) + float(r_base), rel=1e-12, abs=1e-15)


def test_baseflow_availability_cap_binds():
    """The baseflow cap must BIND, not just be present: a thin near-full bucket
    (small W_max, W~=W_max so z_wt~=0 -> demand ~ q_drai_max) has far less stored
    water than the unconstrained demand would drain, so runoff_base is limited to
    the available W/dt (< demand) and W_new stays >= 0.  With a fat W_max the
    diagnostic z_wt~=z_wt_max at low storage suppresses the demand exponentially,
    so the cap never engages there — this thin-bucket case exercises the guard."""
    dt, W_max = 3600.0, 0.01           # thin bucket (kg/m^2)
    W = 0.99 * W_max                   # 99% full -> z_wt ~= 0.05 m -> demand ~ q_drai_max
    W_new, ev, runoff, r_surf, r_base = partition_topmodel_runoff(
        _a(W), _a(0.0), _a(0.0), dt, _a(W_max), _a(1e-5), _a(3.0), _CFG)
    z_wt = float(water_table_depth(_a(W), _a(W_max), _CFG))
    demand = _baseflow_oracle(z_wt, _CFG)
    assert demand > W / dt                                     # the cap genuinely binds
    assert float(W_new) >= 0.0
    assert float(r_base) < demand                              # delivered below demand
    assert float(r_base) == pytest.approx(W / dt, rel=1e-9)    # drains exactly what remains


def test_dunne_overflow_saturates_storage():
    """The Dunne-overflow branch: heavy rain on a near-full bucket drives
    W_unclamped past W_max, so W_new pins to W_max exactly and the overflow shows
    up as an explicit POSITIVE surface-runoff component above the pre-overflow
    surface runoff."""
    dt, W_max, W, P = 3600.0, 100.0, 99.5, 5.0e-2
    W_new, ev, runoff, r_surf, r_base = partition_topmodel_runoff(
        _a(W), _a(P), _a(1e-4), dt, _a(W_max), _a(1e-5), _a(3.0), _CFG)
    z = float(water_table_depth(_a(W), _a(W_max), _CFG))
    f_sat = _fsat_oracle(z, _CFG)
    r_surf_pre = f_sat * P + (1.0 - f_sat) * max(P - 1e-5 * _O_RHO_WATER * (1.0 + 3.0 * (1.0 - W / W_max)), 0.0)
    assert float(W_new) == W_max                               # storage saturates exactly (clip is exact)
    assert float(r_surf) > r_surf_pre                          # overflow added a positive component


def test_evaporation_cap_binds():
    """The evaporation throttle must BIND: an almost-empty column with a large
    evap demand can only evaporate the water it has, so evap_actual < demand and
    equals W/dt (no infiltration), leaving W_new = 0."""
    dt = 3600.0
    W, demand = 0.5, 1.0e-3
    W_new, ev, runoff, r_surf, r_base = partition_topmodel_runoff(
        _a(W), _a(0.0), _a(demand), dt, _a(100.0), _a(1e-5), _a(3.0), _CFG)
    assert float(ev) < demand                                  # cap binds
    assert float(ev) == pytest.approx(W / dt, rel=1e-9)        # evaporates exactly what remains
    assert float(W_new) == pytest.approx(0.0, abs=1e-12)


def test_fsat_and_baseflow_decay_monotonically():
    """Non-vacuity: both f_sat and q_drai strictly decrease as the water table
    deepens (drier column runs off less)."""
    zs = [0.0, 1.0, 2.0, 4.0]
    fs = [float(saturated_area_fraction(_a(z), _CFG)) for z in zs]
    qb = [float(baseflow(_a(z), _CFG)) for z in zs]
    assert all(fs[k] > fs[k + 1] for k in range(3))
    assert all(qb[k] > qb[k + 1] for k in range(3))


# --- AD-safety -----------------------------------------------------------------

def test_topmodel_grad_finite_x64_and_float32():
    """grad of total runoff wrt storage W is finite in x64 AND float32, both at an
    interior unsaturated point (all caps inactive -> smooth, d(runoff)/dW>0 since
    wetter -> higher f_sat -> more Dunne runoff) and on the ACTIVE availability-cap
    branch (thin bucket -> runoff_base = min(demand, avail) selects `avail = W/dt`,
    which is smooth and W-dependent) where autodiff must still return a finite,
    nonzero slope rather than NaN."""
    dt = 3600.0

    def _runoff_interior(W):
        return partition_topmodel_runoff(
            W, _a(5.0e-4), _a(1.0e-4), dt, _a(100.0), _a(1e-5), _a(3.0), _CFG)[2]

    def _runoff_capped(W):   # thin bucket: min() selects the avail branch at the eval point
        return partition_topmodel_runoff(
            W, _a(0.0), _a(0.0), dt, _a(0.01), _a(1e-5), _a(3.0), _CFG)[2]

    def _check():
        g_int = jax.grad(_runoff_interior)(_a(40.0))
        assert bool(jnp.isfinite(g_int)) and float(g_int) > 0.0   # smooth interior, positive
        g_cap = jax.grad(_runoff_capped)(_a(0.0099))
        # avail branch selected -> runoff_base = W/dt -> d/dW = 1/dt, finite & nonzero
        assert bool(jnp.isfinite(g_cap)) and float(g_cap) > 0.0

    _check()
    jax.config.update("jax_enable_x64", False)
    _check()   # autouse fixture restores the entry state afterwards
