"""Direct tests for the Emanuel CONVECT downdraft port.

The decisive test here is the ORIENTATION one.  The oracle is surface-first
and this repo is surface-last, so a flipped port produces a fully finite,
plausibly-shaped, completely wrong answer — no shape check and no norm would
catch it.  ``test_rain_falls_downward`` is built so that a correct port passes
and an index flip fails: condensate is detrained at exactly one level and the
rain water must appear BELOW it.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

jax.config.update("jax_enable_x64", True)

from legoesm import constants
from legoesm.atmosphere.physics.convection._emanuel_downdraft import (
    _taper_mass_flux_to_surface,
    emanuel_downdraft,
)

_NLEV = 20

#: CONVECT v4.3c published defaults (convect43c.f lines 186-201).
_ORACLE = dict(
    sigd=0.05, sigs=0.12, omtrain=50.0, omtsnow=5.5, coeffr=1.0, coeffs=0.8,
    freeze_transition_K=1.0, inertia_scale_hPa=20.0, taper_p_fraction=0.949,
    dhdp_min=10.0, ep_gate_threshold=1.0e-4, ep_gate_width=1.0e-5,
)


def _column(*, ep_level: int | None = None, ep_value: float = 0.5,
            force_top_source: bool = True):
    """One idealized tropical column, SURFACE-FIRST (index 0 = surface).

    ``ep_level`` places ALL the precipitation efficiency at a single level so
    the shaft has one unambiguous source; ``None`` leaves the column with no
    detrained condensate at all.
    """
    p_sfc = 101_480.0
    p_full = jnp.linspace(p_sfc, 10_000.0, _NLEV)[None, :]
    # Half levels: the interface at the BOTTOM of each level.
    p_half = jnp.concatenate(
        [p_full[:, :1], 0.5 * (p_full[:, :-1] + p_full[:, 1:])], axis=-1)
    z = jnp.linspace(0.0, 16_000.0, _NLEV)[None, :]
    T = 300.0 - 6.5e-3 * z
    qs = 0.022 * jnp.exp(-z / 3000.0)
    q = 0.8 * qs
    lv = jnp.full_like(T, constants.L_v)
    cpn = jnp.full_like(T, constants.c_pd)
    gz = constants.g * z
    # The oracle's H is DRY static energy with a moisture-weighted heat
    # capacity (convect43c.f line 353) -- NO Lv*q term.  Building a true moist
    # static energy here would silently disagree with production and change
    # DHDP, hence the whole downdraft mass flux.
    h_dry_static = cpn * T + gz

    clw = jnp.full_like(T, 2.0e-3)
    m_profile = jnp.full_like(T, 0.02)
    ment = jnp.zeros((1, _NLEV, _NLEV))
    elij = jnp.zeros((1, _NLEV, _NLEV))
    if ep_level is None:
        ep = jnp.zeros_like(T)
    else:
        ep = jnp.zeros_like(T).at[:, ep_level].set(ep_value)
        if force_top_source:
            # The whole-shaft gate reads ep at cloud top, so a source anywhere
            # normally needs a non-zero top value or nothing runs.  Tests that
            # need a SINGLE unambiguous source open the gate instead (see
            # _OPEN_GATE) rather than adding a second one.
            ep = ep.at[:, -1].set(max(ep_value, 1.0e-2))
    return dict(T=T, q=q, qs=qs, p_full=p_full, p_half=p_half,
                h_dry_static=h_dry_static, gz=gz, lv=lv, cpn=cpn,
                m_profile=m_profile, ment=ment, elij=elij, clw=clw, ep=ep)


def test_no_detrained_condensate_gives_exactly_zero():
    """The contract's idealized test: no precipitation source, no downdraft."""
    out = emanuel_downdraft(**_column(ep_level=None), **_ORACLE)
    for name in ("mp", "evap", "water", "dT_dt", "dq_v_dt"):
        arr = np.asarray(getattr(out, name))
        assert np.all(np.isfinite(arr)), f"{name} is not finite"
        assert np.max(np.abs(arr)) == 0.0, f"{name} is non-zero: {arr.max()}"
    assert float(out.precip_mm_day[0]) == 0.0


def test_rain_falls_downward_not_upward():
    """ORIENTATION GATE — the test a surface-first/surface-last swap fails.

    ONE source level, and the whole-shaft gate forced open so no second source
    is needed at the top.  Rain falls, so the support of the rain-water field
    must be exactly the source level and everything BELOW it, and exactly zero
    above.  An earlier version of this test seeded a second source at the top
    to satisfy the gate, which let a reversed sweep populate the levels below
    from that source and pass (codex review, 2026-08-14).
    """
    src = 12
    col = _column(ep_level=src, force_top_source=False)
    out = emanuel_downdraft(**col, **{**_ORACLE, **_OPEN_GATE})
    water = np.asarray(out.water)[0]
    assert water[src] > 0.0, "no rain water at the source level"
    assert np.all(water[:src] > 0.0), "rain did not reach the levels below"
    assert np.max(water[src + 1:]) == 0.0, (
        "rain water appears ABOVE the only source — the level ordering is "
        "inverted")


#: Gate wide open: threshold below any ep, so a test can use ONE source level.
_OPEN_GATE = dict(ep_gate_threshold=-1.0, ep_gate_width=1.0e-3)


def test_top_level_downdraft_humidity_uses_the_oracle_initialisation():
    """``IF(I.EQ.INB)GOTO 400`` leaves QP(INB) at its INITIALISATION Q(INB-1)
    (convect43c.f line 501), not at the incoming carry Q(INB).  The two differ
    by one level and the wrong one still runs, still looks sane, and feeds a
    shifted QP(I+1) into every level below."""
    col = _column(ep_level=12)
    # Make the two candidate values unmistakably different.
    q = np.asarray(col["q"]).copy()
    q[0, -1] = 1.0e-6
    q[0, -2] = 5.0e-3
    col["q"] = jnp.asarray(q)
    out = emanuel_downdraft(**col, **_ORACLE)
    assert float(out.qp[0, -1]) == pytest.approx(q[0, -2], rel=1e-9)


def test_the_smooth_branches_recover_the_oracle_in_the_zero_width_limit():
    """The two deliberate departures are bounded, not open-ended.

    Driving the phase-switch width to zero must reproduce the hard
    ``IF(T(I).GT.273.0)`` selection: every level of this column is warmer than
    273 K except the top few, so the fall speed must collapse onto OMTRAIN
    below the freezing level and OMTSNOW above it.
    """
    col = _column(ep_level=12)
    out = emanuel_downdraft(**col, **{**_ORACLE, "freeze_transition_K": 1.0e-4})
    T = np.asarray(col["T"])[0]
    wt = np.asarray(out.wt)[0]
    warm = T > 273.0
    assert np.allclose(wt[warm], _ORACLE["omtrain"], rtol=1e-6)
    assert np.allclose(wt[~warm], _ORACLE["omtsnow"], rtol=1e-6)


def test_mass_flux_is_zero_at_the_surface_and_non_negative():
    out = emanuel_downdraft(**_column(ep_level=12), **_ORACLE)
    mp = np.asarray(out.mp)[0]
    assert mp[0] == 0.0, "the oracle leaves MP at the lowest level at zero"
    assert np.all(mp >= 0.0), "a downdraft mass flux magnitude went negative"


def test_evaporation_moistens_and_cools():
    """Sign convention, stated in the contract and checked here.

    Evaporation adds vapour and removes heat in the layer it occurs in.  The
    vapour tendency also carries a mass-flux divergence, so the check is made
    on the term that is unambiguously evaporative: where evap is largest,
    dT_dt must be negative.
    """
    out = emanuel_downdraft(**_column(ep_level=12), **_ORACLE)
    evap = np.asarray(out.evap)[0]
    dT = np.asarray(out.dT_dt)[0]
    k = int(np.argmax(evap))
    assert evap[k] > 0.0
    assert dT[k] < 0.0, "evaporation warmed the layer"


def test_a_saturated_environment_still_evaporates():
    """A saturated ENVIRONMENT does not stop the shaft, and that is correct.

    The ventilation factor is driven by ``QS(I) - 0.5*(Q(I) + QP(I+1))``
    (convect43c.f line 754): the deficit of the MIXTURE of environmental and
    downdraft air, not of the environment alone.  The shaft descends carrying
    air from aloft, so ``QP`` sits below the local ``QS`` and evaporation
    continues even at ``Q == QS``.

    Written as an explicit expectation because the obvious guess -- saturate
    the column, evaporation stops -- is wrong, and an earlier version of this
    file (and of the module's physics contract) asserted it.
    """
    col = _column(ep_level=12)
    col["q"] = col["qs"]
    out = emanuel_downdraft(**col, **_ORACLE)
    assert np.max(np.asarray(out.evap)) > 0.0


def test_supersaturating_the_mixture_floors_evaporation_at_zero():
    """The guarantee that DOES hold: ``AFAC=MAX(AFAC,0.0)`` (line 755).

    Push the environment far enough above saturation that the mixture's
    deficit is negative at every level, and the floor must give exactly zero
    -- never a negative 'evaporation' that would condense rain out of thin
    air and reverse the sign of the tendencies.
    """
    col = _column(ep_level=12)
    col["q"] = col["qs"] * 3.0
    out = emanuel_downdraft(**col, **_ORACLE)
    evap = np.asarray(out.evap)
    assert np.max(np.abs(evap)) == 0.0
    assert np.min(evap) >= 0.0


def test_gradients_are_finite_through_the_whole_sweep():
    """A NaN gradient here is a bug for every trainer, and the guarded square
    root and MP divisions are exactly where one would appear."""
    col = _column(ep_level=12)

    def loss(q):
        out = emanuel_downdraft(**{**col, "q": q}, **_ORACLE)
        return jnp.sum(out.dq_v_dt) + jnp.sum(out.dT_dt)

    g = np.asarray(jax.grad(loss)(col["q"]))
    assert np.all(np.isfinite(g)), "non-finite gradient through the downdraft"
    assert np.max(np.abs(g)) > 0.0, "the sweep is not connected to its input"


def test_gradient_is_finite_on_a_column_with_no_source():
    """The zero-discriminant case: sqrt(0) has an infinite derivative, and a
    non-convecting column is where the trainer spends most of its time."""
    col = _column(ep_level=None)

    def loss(q):
        out = emanuel_downdraft(**{**col, "q": q}, **_ORACLE)
        return jnp.sum(out.dq_v_dt) + jnp.sum(out.dT_dt)

    assert np.all(np.isfinite(np.asarray(jax.grad(loss)(col["q"]))))


def test_surface_taper_is_linear_to_zero_and_anchored_at_the_band_top():
    """The JTT construct, checked against its closed form directly."""
    p_sfc = 100_000.0
    p = jnp.asarray([[p_sfc, 99_000.0, 96_000.0, 94_000.0, 80_000.0]])
    mp = jnp.asarray([[7.0, 3.0, 5.0, 9.0, 11.0]])
    out = np.asarray(_taper_mass_flux_to_surface(
        mp, p, taper_p_fraction=0.949))[0]
    # Band is p > 94_900: indices 0,1,2 -> anchor jtt = 2.
    anchor = 5.0
    assert out[2] == pytest.approx(anchor), "the anchor level must be a no-op"
    expected1 = anchor * (p_sfc - 99_000.0) / (p_sfc - 96_000.0)
    assert out[1] == pytest.approx(expected1)
    assert out[3] == pytest.approx(9.0), "a level above the band was tapered"
    assert out[4] == pytest.approx(11.0)
    # Index 0 is left to the caller's surface mask, matching the oracle's
    # ``IF(I.EQ.1)GOTO 360``.
    assert out[0] == pytest.approx(7.0)


def test_taper_survives_two_levels_at_equal_pressure():
    """GLM's edge case: the oracle divides by ``P(1)-P(JTT)`` and produces NaN
    when the two lowest levels share a pressure."""
    p = jnp.asarray([[100_000.0, 100_000.0, 80_000.0]])
    mp = jnp.asarray([[2.0, 3.0, 4.0]])
    out = np.asarray(_taper_mass_flux_to_surface(
        mp, p, taper_p_fraction=0.949))
    assert np.all(np.isfinite(out)), "degenerate pressures produced NaN"


# --------------------------------------------------------------------------- #
# Integration through the PUBLIC Emanuel entry point.
#
# The isolated tests above exercise a helper.  They cannot catch the failure
# mode the review actually found: a correct helper that production never
# calls.  These do.
# --------------------------------------------------------------------------- #
def _production_column(nlev: int = 30):
    """Surface-LAST column for the public scheme (index -1 is the surface)."""
    from legoesm.grids.vertical import create_sigma_coordinate

    p_sfc = 101_480.0
    sigma = create_sigma_coordinate(nlev)
    sigma_full = np.asarray(sigma.sigma_full, dtype=float)
    p_full = jnp.asarray(sigma_full * p_sfc)[None, :]
    p_half = jnp.asarray(
        np.asarray(sigma.sigma_half, dtype=float) * p_sfc)[None, :]
    z = -(constants.R_d * 290.0 / constants.g) * np.log(
        np.maximum(sigma_full, 1e-6))
    T = jnp.asarray(np.maximum(302.0 - 8.0e-3 * z, 200.0))[None, :]
    from legoesm.thermo import saturation_mixing_ratio
    q_v = 0.85 * saturation_mixing_ratio(T, p_full)
    return T, q_v, p_full, p_half


def _emanuel_tendencies(*, downdraft: bool):
    from legoesm.atmosphere.physics.convection.config import EmanuelConfig
    from legoesm.atmosphere.physics.convection.emanuel import (
        emanuel_convection,
    )

    T, q_v, p_full, p_half = _production_column()
    cfg = EmanuelConfig(use_genuine_mixing=True,
                        enable_unsaturated_downdraft=downdraft)
    # ``conv_prog_profile`` is the carried cloud-base mass flux; a zero
    # profile is the cold start the scheme is documented to accept.
    cbmf = jnp.zeros_like(T)
    out, _carry = emanuel_convection(
        T, q_v, p_full, p_half, cbmf, 600.0, cfg)
    return out


def test_the_flag_actually_changes_the_production_tendencies():
    """THE integration gate.  The port was committed unwired once, and every
    isolated test still passed; only a comparison through the public entry
    point can see that."""
    off = _emanuel_tendencies(downdraft=False)
    on = _emanuel_tendencies(downdraft=True)
    dT_off = np.asarray(off.dT_dt)
    dT_on = np.asarray(on.dT_dt)
    assert np.all(np.isfinite(dT_on)), "the wired downdraft produced non-finite T"
    assert np.max(np.abs(dT_on - dT_off)) > 0.0, (
        "enable_unsaturated_downdraft changed nothing in production — the "
        "flag is advertising a scheme it does not run")


def test_the_downdraft_is_refused_on_the_legacy_surrogate():
    """It is driven by the buoyancy-sort mixing matrix, which the surrogate
    does not produce; running silently would be different physics under the
    same flag."""
    from legoesm.atmosphere.physics.convection.config import EmanuelConfig
    from legoesm.atmosphere.physics.convection.emanuel import (
        emanuel_convection,
    )

    T, q_v, p_full, p_half = _production_column()
    cfg = EmanuelConfig(use_genuine_mixing=False,
                        enable_unsaturated_downdraft=True)
    with pytest.raises(ValueError, match="requires use_genuine_mixing"):
        emanuel_convection(
            T, q_v, p_full, p_half, jnp.zeros_like(T), 600.0, cfg)
