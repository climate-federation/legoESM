"""Scheme-level oracle-faithfulness tests for Zhang-McFarlane deep convection.

Complements ``tests/unit/test_zm_dilute_parcel.py`` (which pins the DILUTE-CAPE
helper against its Raymond-Blyth/E3SM ``parcel_dilute`` properties) by pinning,
at the ``zhang_mcfarlane_convection`` SCHEME level:

FAITHFUL (matched to the E3SM/CAM ``zm_conv.F90`` oracle):
  * the scheme USES the dilute CAPE by default (E3SM oracle defaults), so the
    signature ZM property propagates: dilute CAPE < undilute CAPE on an unstable
    column, and the scheme routes CAPE through the ported ``dilute_parcel_cape``;
  * the dilute-parcel oracle constants (``use_dilute_cape``/``dmpdz``/
    ``tiedke_add``) hold their E3SM/CAM defaults (canary).

DEPARTURE (SURROGATE, locked + labeled):
  * the cloud-base mass-flux closure is the documented first-order CAPE-relaxation
    SURROGATE (NOT the ZM95 cloud-work closure):
        M_b_eq = sigmoid(s·Δ) · rho_BL · softplus(s·Δ)/(s · g · tau_cape),
        M_b    = clip( (M_b_old + r·M_b_eq)/(1 + r), 0, M_b_max ),  r = dt/max(tau_cape, 1e-30),
    with Δ = CAPE - cape_threshold, s = cape_sharpness, rho_BL the dry BL density.
    Pinned EXACTLY against an INDEPENDENT reimplementation (sigmoid trigger,
    softplus positive-part, dry-rho, implicit-Euler carry, M_b_max clip), so a
    silent change to any factor trips it — including a regime where the trigger
    weight is < 1 (sigmoid actually applied), the carry term, and an active clip;
  * the surrogate closure knobs (``cape_threshold``/``cape_sharpness``/
    ``M_b_max``/``tau_cape``) are NOT dilute-parcel oracle constants — pinned in a
    separately labeled DEPARTURE canary.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)
from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry
from legoesm.atmosphere.physics.convection._zm_dilute import dilute_parcel_cape
from legoesm.atmosphere.physics.thermodynamics import parcel_profile_and_cape


@pytest.fixture(autouse=True)
def _enable_x64():
    prev = jax.config.jax_enable_x64
    jax.config.update("jax_enable_x64", True)
    try:
        yield
    finally:
        jax.config.update("jax_enable_x64", prev)


def _column(nlev=30, T_sfc=302.0, lapse=8.5, rh=0.9, p_s=1.0e5, p_top=5.0e3):
    """Conditionally-unstable moist tropical column, shape (1, nlev)."""
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = (sigma * p_s)[None, :]
    p_half = jnp.concatenate(
        [jnp.full((1, 1), p_top * 0.5),
         0.5 * (p_full[:, :-1] + p_full[:, 1:]),
         jnp.full((1, 1), p_s)],
        axis=1,
    )
    H = 8000.0
    z = -H * jnp.log(p_full / p_s)
    T = jnp.maximum(T_sfc - lapse * 1e-3 * z, 200.0)
    q_v = rh * saturation_mixing_ratio(T, p_full)
    u = jnp.full_like(T, 5.0)
    v = jnp.zeros_like(T)
    return T, q_v, p_full, p_half, u, v


def _independent_mb(cape, T_sfc, p_sfc, M_b_old, cfg, dt):
    """Independent NumPy reimplementation of the DOCUMENTED surrogate closure.

    Uses ONLY the documented form and returns ``(clipped, unclipped, weight)``:
      * ``weight``   = sigmoid(s·Δ)                         [trigger]
      * ``pos``      = softplus(s·Δ)/s                      [(Δ)_+]
      * ``rho_BL``   = p_sfc/(R_d·clip(T_sfc,1,None))       [dry ideal gas]
      * ``M_b_eq``   = weight·rho_BL·pos/(g·tau_cape)
      * ``unclipped``= (M_b_old + r·M_b_eq)/(1+r), r = dt/max(tau_cape,1e-30)  [implicit Euler]
      * ``clipped``  = clip(unclipped, 0, M_b_max)
    Mirrors ``cape_trigger``/``smooth_positive_part``/``compute_rho`` exactly.
    """
    cape = np.asarray(cape, dtype=np.float64)
    T_sfc = np.asarray(T_sfc, dtype=np.float64)
    p_sfc = np.asarray(p_sfc, dtype=np.float64)
    x = cfg.cape_sharpness * (cape - cfg.cape_threshold)
    weight = 1.0 / (1.0 + np.exp(-x))                       # sigmoid trigger
    pos = np.logaddexp(0.0, x) / cfg.cape_sharpness         # softplus(s·Δ)/s
    rho_BL = p_sfc / (constants.R_d * np.clip(T_sfc, 1.0, None))
    M_b_eq = weight * rho_BL * pos / (constants.g * cfg.tau_cape)
    r = dt / max(cfg.tau_cape, 1e-30)
    unclipped = (M_b_old + r * M_b_eq) / (1.0 + r)
    clipped = np.clip(unclipped, 0.0, cfg.M_b_max)
    return clipped, unclipped, weight


# ===========================================================================
# FAITHFUL: the scheme uses the dilute-CAPE (E3SM) faithfully
# ===========================================================================
def test_faithful_default_is_e3sm_dilute_parcel_setup():
    """Canary: the DILUTE-PARCEL oracle constants are the E3SM/CAM defaults."""
    c = ZhangMcFarlaneConfig()
    assert c.use_dilute_cape is True
    assert c.dmpdz == -1.0e-3        # E3SM/CAM fractional entrainment [1/m]
    assert c.tiedke_add == 0.5       # oracle buoyancy offset [K]


def test_departure_default_surrogate_closure_controls():
    """Canary: the SURROGATE closure knobs hold their documented defaults.

    These are NOT dilute-parcel oracle constants — they parameterize the departure
    CAPE-relaxation closure (which is itself a surrogate, not the ZM95 closed
    form). ``cape_threshold=70`` J/kg is the classic ZM95 trigger VALUE, but it is
    a closure/trigger setting here, not a dilute-parcel constant.
    """
    c = ZhangMcFarlaneConfig()
    assert c.cape_threshold == 70.0  # ZM95 trigger value [J/kg] (closure setting)
    assert c.cape_sharpness == 0.1   # smooth-trigger sharpness [1/(J/kg)]
    assert c.M_b_max == 0.05         # surrogate mass-flux cap [kg/m^2/s]
    assert c.tau_cape == 3600.0      # CAPE-relaxation timescale [s]


def test_faithful_scheme_dilute_cape_less_than_undilute():
    """The signature ZM property propagates to the scheme: dilute < undilute.

    Also pins that the scheme's CAPE diagnostic IS the faithful dilute helper
    (and the undilute path is the undilute moist adiabat), so this certifies the
    scheme ROUTES CAPE through the oracle-ported ``dilute_parcel_cape``, not just
    that some lower number appears.
    """
    cfg = ZhangMcFarlaneConfig()
    T, q, pf, ph, u, v = _column()
    cpp = jnp.zeros_like(T)
    out_dil, _ = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=cfg)
    out_und, _ = zhang_mcfarlane_convection(
        T, q, pf, ph, u, v, cpp, dt=300.0,
        config=ZhangMcFarlaneConfig(use_dilute_cape=False),
    )
    # both columns are genuinely unstable (positive CAPE) so the comparison bites
    assert float(out_dil.cape[0]) > 0.0
    assert float(out_und.cape[0]) > 0.0

    # scheme CAPE == the faithful dilute helper (same geometry the scheme builds)
    _, _, z = compute_column_geometry(T, pf, ph, q_v=q)
    dhelper = dilute_parcel_cape(
        T, q, pf, ph, z, dmpdz=cfg.dmpdz, tiedke_add=cfg.tiedke_add,
        tp_fac=cfg.tp_fac, tpert=cfg.parcel_tpert, pbl_top_pa=cfg.pbl_top_pa,
    ).cape
    assert np.allclose(np.asarray(out_dil.cape), np.asarray(dhelper), rtol=1e-9, atol=1e-6)
    _, uhelper = parcel_profile_and_cape(T, pf, ph, q_v=q)
    assert np.allclose(np.asarray(out_und.cape), np.asarray(uhelper), rtol=1e-9, atol=1e-6)

    assert float(out_dil.cape[0]) < float(out_und.cape[0])
    # entrainment removes a large fraction of the undilute CAPE
    assert float(out_dil.cape[0]) < 0.85 * float(out_und.cape[0])


# ===========================================================================
# DEPARTURE: the cloud-base M_b closure is the documented surrogate
# ===========================================================================
def test_departure_mb_closure_is_documented_surrogate_form():
    """Cold-start M_b == the independent surrogate closure (exact FORM pin).

    Independent reimplementation (:func:`_independent_mb`) of the DOCUMENTED
    surrogate
        M_b_eq = sigmoid(s·Δ)·rho_BL·softplus(s·Δ)/(s·g·tau),
        M_b    = clip((M_b_old + r·M_b_eq)/(1+r), 0, M_b_max)
    with ``r = dt/max(tau_cape, 1e-30)`` and M_b_old = 0 (cold start). Evaluated
    on TWO columns: a strongly-unstable tropical column (trigger weight ~1,
    appreciable M_b) and a WEAKLY-unstable column whose trigger weight is ~0.02.
    On the latter, dropping the sigmoid factor scales M_b by ~1/weight (~50×), so
    the exact-form pin AND an explicit ``not allclose`` vs the trigger-removed
    closure certify the sigmoid factor is APPLIED; the test also asserts an
    explicit finite gap (> 1e-12, far above the 1e-14 atol) for that column so
    the check is non-vacuous rather than relying on a universal-magnitude claim.
    (Carry term + active clip are pinned in
    :func:`test_departure_mb_closure_carry_and_clip`.)
    """
    cfg = ZhangMcFarlaneConfig()
    dt = 300.0
    got_vals = []
    sigmoid_region_tested = False
    # (strongly-unstable, weakly-unstable) columns -> (weight~1, weight<<1)
    columns = [
        dict(T_sfc=302.0),                             # CAPE ~ 4000, weight ~ 1
        dict(T_sfc=298.0, lapse=6.5, rh=0.67),         # CAPE ~ 30, weight ~ 0.02
    ]
    for col_kw in columns:
        T, q, pf, ph, u, v = _column(**col_kw)
        cpp = jnp.zeros_like(T)                      # M_b_old = 0 (cold start)
        out, cpp_new = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=dt, config=cfg)
        cape = np.asarray(out.cape)                  # dilute CAPE (scheme output)
        T_sfc_col = np.asarray(T)[:, -1]
        p_sfc_col = np.asarray(pf)[:, -1]
        expected, unclipped, weight = _independent_mb(cape, T_sfc_col, p_sfc_col, 0.0, cfg, dt)
        got = np.asarray(cpp_new)[:, -1]
        assert np.allclose(got, expected, rtol=1e-9, atol=1e-14)
        assert np.all(unclipped < cfg.M_b_max)       # uncapped regime (FORM, not clip)
        got_vals.append(float(got[0]))

        # Sigmoid factor genuinely applied: where the trigger weight < 1,
        # dropping it (softplus-only) must change the answer.
        if float(weight[0]) < 0.98:
            sigmoid_region_tested = True
            pos = np.logaddexp(0.0, cfg.cape_sharpness * (cape - cfg.cape_threshold)) / cfg.cape_sharpness
            rho_BL = p_sfc_col / (constants.R_d * np.clip(T_sfc_col, 1.0, None))
            M_b_eq_nosig = rho_BL * pos / (constants.g * cfg.tau_cape)  # trigger removed
            r = dt / max(cfg.tau_cape, 1e-30)
            got_nosig = np.clip((0.0 + r * M_b_eq_nosig) / (1.0 + r), 0.0, cfg.M_b_max)
            assert not np.allclose(got, got_nosig, rtol=1e-6, atol=1e-14)
            # concrete margin: trigger removal scales M_b by ~1/weight, giving a
            # gap far above the atol floor for this weakly-unstable column.
            assert float(np.max(np.abs(got - got_nosig))) > 1e-12
    assert max(got_vals) > 0.0            # at least one column fired
    assert sigmoid_region_tested         # non-vacuous sigmoid-applied check


def test_departure_mb_closure_carry_and_clip():
    """Pins the implicit-Euler CARRY term and the active M_b_max CLIP.

    (a) Non-zero M_b_old: production == the independent formula WITH the carry
        term, and differs materially from the cold-start (M_b_old=0) result.
    (b) Active clip: with a tiny M_b_max the unclipped surrogate exceeds the cap,
        so production must equal M_b_max exactly (not the unclipped value).
    """
    T, q, pf, ph, u, v = _column()
    dt = 300.0
    T_sfc_col = np.asarray(T)[:, -1]
    p_sfc_col = np.asarray(pf)[:, -1]

    # (a) carry term ------------------------------------------------------
    cfg = ZhangMcFarlaneConfig()
    _, cpp0 = zhang_mcfarlane_convection(
        T, q, pf, ph, u, v, jnp.zeros_like(T), dt=dt, config=cfg,
    )
    M_b_old = 0.01
    cpp_in = jnp.zeros_like(T).at[:, -1].set(M_b_old)
    out1, cpp1 = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp_in, dt=dt, config=cfg)
    cape = np.asarray(out1.cape)
    exp_carry, unclipped, _ = _independent_mb(cape, T_sfc_col, p_sfc_col, M_b_old, cfg, dt)
    got1 = np.asarray(cpp1)[:, -1]
    assert np.allclose(got1, exp_carry, rtol=1e-9, atol=1e-14)
    assert np.all(unclipped < cfg.M_b_max)       # this case is uncapped
    # the carry term materially changes the answer vs cold start
    assert abs(float(got1[0]) - float(np.asarray(cpp0)[:, -1][0])) > 1e-4

    # (b) active clip -----------------------------------------------------
    cfg_cap = cfg._replace(M_b_max=1.0e-5)
    out2, cpp2 = zhang_mcfarlane_convection(
        T, q, pf, ph, u, v, jnp.zeros_like(T), dt=dt, config=cfg_cap,
    )
    cape2 = np.asarray(out2.cape)
    exp_cap, unclipped2, _ = _independent_mb(cape2, T_sfc_col, p_sfc_col, 0.0, cfg_cap, dt)
    got2 = np.asarray(cpp2)[:, -1]
    assert np.all(unclipped2 > cfg_cap.M_b_max)  # clip is genuinely active
    assert np.allclose(got2, cfg_cap.M_b_max, rtol=1e-9, atol=1e-14)
    assert np.allclose(got2, exp_cap, rtol=1e-9, atol=1e-14)


def test_departure_mb_bounded_in_range():
    """Surrogate closure stays in [0, M_b_max] across a range of columns.

    (Monotonicity of M_b in surface temperature is NOT asserted: warmer surface
    raises CAPE but lowers rho_BL, so M_b_eq = sigmoid(sΔ)*rho_BL*softplus(sΔ)/
    (s*g*tau) need not be monotone in T. The EXACT closure form is pinned in
    ``test_departure_mb_closure_is_documented_surrogate_form``; here we only lock
    the M_b_max bound + non-negativity, which the clip guarantees.)
    """
    cfg = ZhangMcFarlaneConfig()

    def _mb(T_sfc):
        T, q, pf, ph, u, v = _column(T_sfc=T_sfc)
        cpp = jnp.zeros_like(T)
        _, cpp_new = zhang_mcfarlane_convection(T, q, pf, ph, u, v, cpp, dt=300.0, config=cfg)
        return float(np.asarray(cpp_new)[0, -1])

    for T_sfc in (296.0, 300.0, 304.0, 308.0):
        mb = _mb(T_sfc)
        assert 0.0 <= mb <= cfg.M_b_max


def test_trigger_off_below_threshold_cold_start_gives_near_zero_massflux():
    """Cold-start (M_b_old=0) stable column (CAPE <= threshold) -> M_b ~ 0.

    Documented as a COLD-START test: with a non-zero prior M_b the implicit
    relaxation would retain flux even below threshold. Here M_b_old = 0, and the
    near-zero result is additionally pinned to the exact surrogate value from
    :func:`_independent_mb` (not merely ``< 1e-6``), plus near-zero tendencies.
    """
    # Isothermal-ish, dry, cold column: no CAPE.
    nlev = 30
    p_s, p_top = 1.0e5, 5.0e3
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = (sigma * p_s)[None, :]
    p_half = jnp.concatenate(
        [jnp.full((1, 1), p_top * 0.5),
         0.5 * (p_full[:, :-1] + p_full[:, 1:]),
         jnp.full((1, 1), p_s)], axis=1,
    )
    T = jnp.full((1, nlev), 250.0)          # cold, stable
    q = jnp.full((1, nlev), 1e-5)           # very dry
    u = jnp.zeros_like(T)
    v = jnp.zeros_like(T)
    cpp = jnp.zeros_like(T)                  # M_b_old = 0 (cold start)
    cfg = ZhangMcFarlaneConfig()
    dt = 300.0
    out, cpp_new = zhang_mcfarlane_convection(
        T, q, p_full, p_half, u, v, cpp, dt=dt, config=cfg,
    )
    # precondition: this column is stable (CAPE at/below the trigger threshold)
    assert float(np.asarray(out.cape)[0]) <= cfg.cape_threshold
    # exact surrogate value (tiny) — pins the form, not just a loose bound
    expected, _, _ = _independent_mb(
        np.asarray(out.cape), np.asarray(T)[:, -1], np.asarray(p_full)[:, -1], 0.0, cfg, dt,
    )
    got = np.asarray(cpp_new)[:, -1]
    assert np.allclose(got, expected, rtol=1e-9, atol=1e-14)
    assert float(got[0]) < 1e-6
    assert float(np.max(np.abs(np.asarray(out.dT_dt)))) < 1e-6
    assert float(np.max(np.abs(np.asarray(out.dq_v_dt)))) < 1e-6
    assert float(np.max(np.abs(np.asarray(out.dq_c_conv_dt)))) < 1e-6
