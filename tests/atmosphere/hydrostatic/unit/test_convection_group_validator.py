"""Cross-scheme truth-tier validator for the atm-convection group.

Validates every convection scheme in the "atm-convection" group against
five truth tiers, uniformly:

  1. UNITS  — outputs finite, correct shapes, CAPE in physical range
              (proxy for dimensional consistency; the schemes draw every
              constant from ``legoesm.constants`` and saturation from
              ``legoesm.thermo`` — re-derivation is forbidden by the AST
              ratchets, so the runtime check here is finiteness + range).
  2. SIGNS  — a destabilized (CAPE-positive) column heats the free
              troposphere (column-integrated c_p dT > 0) and the
              convective cloud-water source is non-negative everywhere.
  3. CONSERV— column total water ``∫(dq_v + dq_c) dp/g`` ≈ 0 and, for the
              schemes that advertise MSE conservation, column moist
              static energy ``∫(c_p dT + L_v dq_v) dp/g`` ≈ 0.
  4. DIFF   — ``jax.grad`` of a scalar column loss w.r.t. a tunable
              config field (or the input state) is finite, non-NaN, and
              non-zero; jit matches eager.
  5. IDEAL  — a stable (sub-adiabatic, dry) column produces ~zero
              tendencies (no spurious convection), and a destabilized
              column produces a clearly non-zero response.

Schemes covered: sbm, dca (manabe + ahmed_neelin), kuo, zhang_mcfarlane,
emanuel, kain_fritsch, bechtold, and the shared _triggers primitives.

Run:
    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python -m pytest \
        tests/atmosphere/hydrostatic/unit/test_convection_group_validator.py -q
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import pytest

from legoesm import constants
from legoesm.atmosphere.physics.convection.config import (
    AhmedNeelinDCAConfig,
    BechtoldConfig,
    DCAConfig,
    EmanuelConfig,
    KainFritschConfig,
    KuoConfig,
    SBMConfig,
    ZhangMcFarlaneConfig,
)
from legoesm.atmosphere.physics.convection.sbm import sbm_convection
from legoesm.atmosphere.physics.convection.dca import (
    ahmed_neelin_dca,
    dca_convection,
)
from legoesm.atmosphere.physics.convection.kuo import kuo_convection
from legoesm.atmosphere.physics.convection.zhang_mcfarlane import (
    zhang_mcfarlane_convection,
)
from legoesm.atmosphere.physics.convection.emanuel import emanuel_convection
from legoesm.atmosphere.physics.convection.kain_fritsch import (
    kain_fritsch_convection,
)
from legoesm.atmosphere.physics.convection.bechtold import bechtold_convection


jax.config.update("jax_enable_x64", True)

DT = 300.0
G = constants.g
C_PD = constants.c_pd
L_V = constants.L_v


# ---------------------------------------------------------------------------
# Shared column builders
# ---------------------------------------------------------------------------

def _column(
    ncol: int = 2,
    nlev: int = 24,
    *,
    T_sfc: float = 301.0,
    q_sfc: float = 17.0e-3,
    lapse_rate: float = 7.5,   # K/km — conditionally unstable
    p_s: float = 1.0e5,
    p_top: float = 5.0e3,
    rh_scale_m: float = 3000.0,
):
    """Surface-last conditionally-unstable column (CAPE-positive)."""
    sigma = jnp.linspace(p_top / p_s, 1.0, nlev)
    p_full = sigma[None, :] * jnp.full((ncol, 1), p_s)
    p_half_inner = 0.5 * (p_full[:, :-1] + p_full[:, 1:])
    p_half = jnp.concatenate(
        [
            jnp.full((ncol, 1), p_top * 0.5),
            p_half_inner,
            jnp.full((ncol, 1), p_s),
        ],
        axis=1,
    )
    z_full = -8500.0 * jnp.log(p_full / p_s)
    T = jnp.full((ncol,), T_sfc)[:, None] - lapse_rate * 1e-3 * z_full
    q = q_sfc * jnp.exp(-z_full / rh_scale_m)
    u = jnp.broadcast_to(jnp.linspace(0.0, 25.0, nlev)[None, :], (ncol, nlev))
    v = jnp.zeros_like(u)
    return T, q, p_full, p_half, u, v


def _stable_dry_column(ncol: int = 2, nlev: int = 24):
    """Strongly stable, dry column — convection should stay quiescent.

    Near-isothermal (almost no lapse rate) and very dry so no scheme finds
    positive CAPE.
    """
    return _column(
        ncol=ncol, nlev=nlev,
        T_sfc=288.0, q_sfc=1.0e-3, lapse_rate=2.0, rh_scale_m=1500.0,
    )


# Schemes whose parcel needs a level of neutral buoyancy inside the column
# (KF: cloud-top search; CAM6 ZM: buoyan_dilute tentative-top search).
_LNB_SCHEMES = ("kain_fritsch", "zhang_mcfarlane")


def _capped_column(
    ncol: int = 2,
    nlev: int = 24,
    *,
    tropopause_pa: float = 1.5e4,
    **kw,
):
    """``_column`` with an isothermal stratosphere above ``tropopause_pa``.

    The default :func:`_column` applies a single constant lapse rate all the
    way to ``p_top`` (50 hPa) — i.e. it has NO tropopause, so a moist-adiabatic
    parcel stays positively buoyant at every level up to the model top
    (``T_parcel − T_env`` never turns negative above the LFC).

    That is fine for the buoyancy/CAPE-driven schemes that key only off the
    integrated CAPE (ZM, Emanuel, Bechtold), but it is ILL-POSED for Kain-
    Fritsch specifically: KF is the only scheme here that builds its cloud
    geometry around a finite Level of Neutral Buoyancy via
    ``_plume.compute_lfc_lnb``.  With no upper buoyancy cap that LNB has no
    positive→negative crossing to localize, so it collapses onto the LFC
    (``cloud_depth ≈ 0``); KF's cloud-top detrainment then dumps the updraft
    right at cloud base, the plume makes ~no cloud water, and the net column
    signal is dominated by sub-cloud downdraft re-evaporation — net COOLING
    and MOISTENING regardless of the deep/shallow branch or trigger path.

    The fix is the physical one: give KF a column with a real tropopause.  KF
    then nets the expected heating/drying robustly (``+700..800 W/m²``,
    ``dq_v < 0``) across tropopause heights 100–200 hPa.  This is NOT a relaxed
    gate — the sign thresholds below are unchanged; only the test column is
    made well-posed for a scheme that requires a finite LNB.  (The shared
    ``compute_lfc_lnb`` LNB-collapse on an uncapped tower is a narrow but real
    diagnostic soft-spot, reported separately; it does not bite on any
    physically-capped sounding.)
    """
    T, q, p_full, p_half, u, v = _column(ncol=ncol, nlev=nlev, **kw)
    # Isothermal cap: freeze T at its tropopause value above ``tropopause_pa``.
    p_s = p_full[:, -1:]
    z_full = -8500.0 * jnp.log(p_full / p_s)
    z_trop = -8500.0 * jnp.log(tropopause_pa / p_s)
    T_trop = jax.vmap(
        lambda z_col, T_col, zt: jnp.interp(zt[0], z_col[::-1], T_col[::-1])
    )(z_full, T, z_trop)
    T = jnp.where(p_full < tropopause_pa, T_trop[:, None], T)
    return T, q, p_full, p_half, u, v


def _col_int(field, dp):
    """Column integral ∫ field dp / g  (mass-weighted), shape (ncol,)."""
    return jnp.sum(field * dp, axis=-1) / G


def _dp(p_half):
    return p_half[:, 1:] - p_half[:, :-1]


# ---------------------------------------------------------------------------
# Per-scheme thin call wrappers → ConvectionOutput
# ---------------------------------------------------------------------------

def _run_sbm(T, q, pf, ph, **kw):
    return sbm_convection(T, q, pf, ph, DT, SBMConfig(**kw))


def _run_dca_manabe(T, q, pf, ph, **kw):
    return dca_convection(T, q, pf, ph, DT, DCAConfig(variant="manabe", **kw))


def _run_dca_ahmed(T, q, pf, ph, **kw):
    return ahmed_neelin_dca(T, q, pf, ph, DT, AhmedNeelinDCAConfig(**kw))


def _run_kuo(T, q, pf, ph, mc=None, **kw):
    return kuo_convection(
        T, q, pf, ph, DT, KuoConfig(**kw), moisture_convergence=mc,
    )


def _run_zm(T, q, pf, ph, u, v, **kw):
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = zhang_mcfarlane_convection(
        T, q, pf, ph, u, v, cpp, DT,
        ZhangMcFarlaneConfig(**{"land_fraction": "none", **kw}),  # ocean columns
    )
    return out


def _run_emanuel(T, q, pf, ph, **kw):
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    out, _ = emanuel_convection(T, q, pf, ph, cpp, DT, EmanuelConfig(**kw))
    return out


def _run_kf(T, q, pf, ph, w=None, **kw):
    ncol, nlev = T.shape
    if w is None:
        w = jnp.zeros((ncol, nlev))
    cpp = jnp.zeros((ncol, nlev))
    out, _ = kain_fritsch_convection(
        T, q, pf, ph, w, cpp, DT, KainFritschConfig(**kw),
    )
    return out


def _run_bechtold(T, q, pf, ph, u, v, mc=None, **kw):
    ncol, nlev = T.shape
    cpp = jnp.zeros((ncol, nlev))
    stoch = jnp.zeros((ncol,))
    out, _, _ = bechtold_convection(
        T, q, pf, ph, u, v, cpp, stoch, None, DT, BechtoldConfig(**kw),
        moisture_convergence=mc,
    )
    return out


# A registry mapping scheme-name → a no-arg callable on the shared columns.
def _all_schemes(T, q, pf, ph, u, v, w, mc):
    return {
        "sbm": lambda: _run_sbm(T, q, pf, ph),
        "dca_manabe": lambda: _run_dca_manabe(T, q, pf, ph),
        "dca_ahmed_neelin": lambda: _run_dca_ahmed(T, q, pf, ph),
        "kuo": lambda: _run_kuo(T, q, pf, ph, mc=mc),
        "zhang_mcfarlane": lambda: _run_zm(T, q, pf, ph, u, v),
        "emanuel": lambda: _run_emanuel(T, q, pf, ph),
        "kain_fritsch": lambda: _run_kf(T, q, pf, ph, w=w),
        "bechtold": lambda: _run_bechtold(T, q, pf, ph, u, v, mc=mc),
    }


SCHEME_NAMES = [
    "sbm", "dca_manabe", "dca_ahmed_neelin", "kuo",
    "zhang_mcfarlane", "emanuel", "kain_fritsch", "bechtold",
]


# ===========================================================================
# Tier 1 — UNITS / finiteness / shapes / CAPE range
# ===========================================================================

@pytest.mark.parametrize("name", SCHEME_NAMES)
def test_tier1_outputs_finite_and_shaped(name):
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)               # gentle resolved ascent
    mc = jnp.full((ncol, nlev), 2.0e-6)            # +moisture convergence
    out = _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()
    assert out.dT_dt.shape == (ncol, nlev)
    assert out.dq_v_dt.shape == (ncol, nlev)
    assert out.dq_c_conv_dt.shape == (ncol, nlev)
    assert out.cape.shape == (ncol,)
    assert out.convective_mask.shape == (ncol,)
    for fld in (out.dT_dt, out.dq_v_dt, out.dq_c_conv_dt,
                out.cape, out.convective_mask):
        assert jnp.all(jnp.isfinite(fld)), f"{name}: non-finite output"
    # convective_mask is a smooth 0-1 indicator.
    assert jnp.all(out.convective_mask >= -1e-9)
    assert jnp.all(out.convective_mask <= 1.0 + 1e-6)
    # CAPE diagnostic in a physical range (ahmed_neelin overloads cape→B_L
    # [m/s²], a small number, so only sanity-check finiteness/upper bound).
    if name == "dca_ahmed_neelin":
        assert jnp.all(jnp.abs(out.cape) < 10.0)   # |B_L| [m/s²]
    else:
        assert jnp.all(out.cape >= -1e-6)
        # Upper bound: saturated-from-base parcel CAPE (sbm/dca_manabe use
        # the legacy saturated moist adiabat) runs ~3x the dilute / virtual-T
        # CAPE the mass-flux schemes use; 50 kJ/kg is a generous physical
        # ceiling that still catches a runaway/NaN.
        assert jnp.all(out.cape < 5.0e4)           # < 50 kJ/kg


# ===========================================================================
# Tier 2 — SIGNS
# ===========================================================================

@pytest.mark.parametrize("name", SCHEME_NAMES)
def test_tier2_cloud_water_source_non_negative(name):
    """The convective condensate source handed to microphysics must be >= 0
    for EVERY scheme (documented invariant of ConvectionOutput)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)
    mc = jnp.full((ncol, nlev), 2.0e-6)
    out = _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()
    assert jnp.all(out.dq_c_conv_dt >= -1e-12), \
        f"{name}: negative convective cloud-water source"


@pytest.mark.parametrize(
    "name",
    # Schemes that actively heat a CAPE-positive column. Kuo is driven by
    # moisture convergence; the others by CAPE/buoyancy.
    ["sbm", "dca_manabe", "dca_ahmed_neelin", "kuo",
     "zhang_mcfarlane", "emanuel", "kain_fritsch", "bechtold"],
)
def test_tier2_destabilized_column_net_heats(name):
    """A CAPE-positive (or moisture-convergent, for Kuo) column should
    produce net column heating: ∫ c_p dT/dt dp/g > 0."""
    # KF and CAM6 ZM need a finite LNB (tropopause); the bare _column has
    # none — see _capped_column (ZM's buoyan_dilute finds no tentative cloud
    # top on a column buoyant up to the model top, so CAPE = 0 there).
    T, q, pf, ph, u, v = (
        _capped_column() if name in _LNB_SCHEMES else _column()
    )
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)
    mc = jnp.full((ncol, nlev), 3.0e-6)
    out = _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()
    dp = _dp(ph)
    col_heat = _col_int(C_PD * out.dT_dt, dp)       # W/m²
    assert jnp.all(col_heat > 0.0), \
        f"{name}: expected net column heating, got {col_heat}"


@pytest.mark.parametrize(
    "name",
    # Schemes whose net column vapor change should be drying (vapor → cloud)
    # on a convecting column.  Kuo's net dq includes the −ptenq removal of
    # the large-scale source it consumes, which can be a net moistening of
    # the prognostic vapor; tested separately.
    ["zhang_mcfarlane", "emanuel", "kain_fritsch", "bechtold",
     "dca_ahmed_neelin"],
)
def test_tier2_mass_flux_schemes_net_dry(name):
    """Mass-flux / buoyancy schemes remove vapor (it becomes cloud water):
    ∫ dq_v/dt dp/g <= 0 on a convecting column."""
    # KF and CAM6 ZM need a finite LNB (tropopause); the bare _column has
    # none — see _capped_column (ZM's buoyan_dilute finds no tentative cloud
    # top on a column buoyant up to the model top, so CAPE = 0 there).
    T, q, pf, ph, u, v = (
        _capped_column() if name in _LNB_SCHEMES else _column()
    )
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)
    mc = jnp.full((ncol, nlev), 3.0e-6)
    out = _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()
    dp = _dp(ph)
    col_dqv = _col_int(out.dq_v_dt, dp)             # kg/m²/s
    assert jnp.all(col_dqv <= 1e-10), \
        f"{name}: expected net column drying, got {col_dqv}"


# ===========================================================================
# Tier 3 — CONSERVATION
# ===========================================================================

@pytest.mark.parametrize(
    "name",
    # Schemes that claim exact column total-water conservation
    # (vapor removed reappears as cloud water in the column).
    # NB: emanuel is intentionally EXCLUDED here. With use_genuine_mixing=True
    # (the default) it is a PRECIPITATING scheme — the net vapor removal is the
    # reference EP*CLW precipitation sink (emanuel.py:464-498, 534-549), NOT
    # reinserted as in-column cloud water — so it does not close column total
    # water in-scheme. Its behavioral contract is gated in the tier1/tier2 lists.
    ["sbm", "dca_manabe", "dca_ahmed_neelin"],
)
def test_tier3_total_water_conserved(name):
    """∫(dq_v + dq_c) dp/g ≈ 0: every kg of vapor removed reappears as
    in-column cloud water (nothing precipitates in-scheme)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    mc = jnp.full((ncol, nlev), 3.0e-6)
    out = _all_schemes(T, q, pf, ph, u, v, None, mc)[name]()
    dp = _dp(ph)
    net_water = _col_int(out.dq_v_dt + out.dq_c_conv_dt, dp)  # kg/m²/s
    # Normalize by the column drying magnitude to get a relative residual.
    scale = _col_int(jnp.abs(out.dq_v_dt), dp) + 1e-15
    rel = jnp.abs(net_water) / scale
    assert jnp.all(rel < 1e-3), \
        f"{name}: total-water residual rel={rel}, abs={net_water}"


@pytest.mark.parametrize(
    "name",
    # Schemes that advertise column MSE conservation of the adjustment.
    # emanuel excluded — it precipitates by design (see the tier3 total-water
    # note above); the condensate carries latent heat out of the vapor-only
    # reservoir, so the vapor-MSE residual is nonzero for the genuine scheme.
    ["dca_manabe", "dca_ahmed_neelin"],
)
def test_tier3_column_mse_conserved(name):
    """∫(c_p dT + L_v dq_v) dp/g ≈ 0: column moist static energy of the
    vapor reservoir is conserved by the adjustment (heating == latent heat
    of the condensed vapor)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    mc = jnp.full((ncol, nlev), 3.0e-6)
    out = _all_schemes(T, q, pf, ph, u, v, None, mc)[name]()
    dp = _dp(ph)
    mse_tend = _col_int(C_PD * out.dT_dt + L_V * out.dq_v_dt, dp)  # W/m²
    heat = _col_int(jnp.abs(C_PD * out.dT_dt), dp) + 1e-9
    rel = jnp.abs(mse_tend) / heat
    assert jnp.all(rel < 5e-3), \
        f"{name}: MSE residual rel={rel}, abs={mse_tend} W/m²"


def test_tier3_zm_total_water_closed_in_scheme():
    """The CAM6 ZM port closes column total water in-scheme (vapour + cloud +
    explicit rain), like KF (2026-07-13) and Bechtold (2026-07-22); the former
    strict-xfail ("ZM leaks in-scheme") is retired with the surrogate scheme.
    """
    T, q, pf, ph, u, v = _capped_column()
    ncol, nlev = T.shape
    mc = jnp.full((ncol, nlev), 3.0e-6)
    dp = _dp(ph)
    for name in ("zhang_mcfarlane",):
        out = _all_schemes(T, q, pf, ph, u, v, None, mc)[name]()
        # CAM6 ZM emits its rain explicitly (dq_r_conv_dt): vapour + cloud +
        # rain closes the column water budget.
        net = _col_int(out.dq_v_dt + out.dq_c_conv_dt + out.dq_r_conv_dt, dp)
        scale = _col_int(jnp.abs(out.dq_v_dt), dp) + 1e-15
        assert jnp.all(jnp.abs(net) / scale < 1e-3), \
            f"{name}: total water not closed: {net}"


def test_tier3_kf_precip_efficiency_leak_removed():
    """REGRESSION (was a BUG — kain_fritsch.py; FIXED 2026-07-13, coupled
    water+energy budget).  KF-Eta previously let the downdraft re-evaporate its
    CONDLOAD fallout as a ``+dq_v_dt`` source with NO matching vapor sink or
    latent warming for the CONDENSATION that produced that precip — water and
    energy appeared from nowhere: ``net(dq_v+dq_c)=+3.16e-4`` kg/m^2/s spurious
    water SOURCE (130x ZM) and ``~-1500`` W/m^2 of phantom COOLING on this
    deep-tropical column (convection cooling+moistening a conditionally-unstable
    column — backwards).

    The fix books the condensation explicitly (see ``kain_fritsch.py`` "column
    condensation water+energy budget"): the ``+L_v/c_p`` latent warming at the
    condensation levels, a POSITIVITY-SAFE vapor sink of the same column
    magnitude ``precip_col`` distributed over the post-transport vapor (so no
    level goes negative), and the non-re-evaporated condensate routed to CLOUD
    water ``dq_c_conv_dt`` (ZM/Emanuel convention: microphysics owns precip).
    The kernel is switched to the conservative ``implicit_flux`` solve (residual
    telescopes to machine precision) and its detrained-cloud latent is released
    as warming so convection is MSE ``h=c_p T+L_v q_v`` conserving in-scheme.

    Result on this column: column TOTAL WATER ``∫(dq_v+dq_c)`` and column MSE
    ``∫(c_p dT + L_v dq_v)`` both close to MACHINE PRECISION, and the net column
    response flips to the physically-correct WARM+DRY (``+L_v*`` net condensate)."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    mc = jnp.full((ncol, nlev), 3.0e-6)
    dp = _dp(ph)
    sch = _all_schemes(T, q, pf, ph, u, v, None, mc)
    out_kf = sch["kain_fritsch"]()
    Tc, qc, pfc, phc, uc, vc = _capped_column()
    out_zm = _all_schemes(Tc, qc, pfc, phc, uc, vc, None, mc)["zhang_mcfarlane"]()
    dp_zm = _dp(phc)
    net_kf = _col_int(out_kf.dq_v_dt + out_kf.dq_c_conv_dt, dp)
    net_zm = _col_int(out_zm.dq_v_dt + out_zm.dq_c_conv_dt + out_zm.dq_r_conv_dt, dp_zm)
    # (1) Column TOTAL WATER closes to machine precision (was +3.16e-4 leak);
    #     the CAM6 ZM port closes its own budget (vapour + cloud + rain) too.
    assert jnp.all(jnp.abs(net_kf) < 1.0e-8), \
        f"KF column-water residual {net_kf} not machine-zero — budget leak"
    assert jnp.all(jnp.abs(net_zm) < 1.0e-8), \
        f"ZM column-water residual {net_zm} not machine-zero — budget leak"
    # (2) Column MSE closes to machine precision (was -728 W/m^2 sink): the
    #     +L_v condensation warming offsets the -L_v vapor sink; the -L_v re-evap
    #     cooling offsets the +L_v re-evap moistening.  This is the ENERGY half
    #     of the fix — a water-only patch would still leave this ~-728.
    mse_kf = _col_int(constants.c_pd * out_kf.dT_dt
                      + constants.L_v * out_kf.dq_v_dt, dp)
    assert jnp.all(jnp.abs(mse_kf) < 1.0e-6), \
        f"KF column MSE residual {mse_kf} W/m^2 not conserved — energy leak"
    # (3) Net column response is now WARM+DRY (deep convection), not the old
    #     cool+moisten: heating > 0 and vapor tendency < 0.
    heat_kf = _col_int(constants.c_pd * out_kf.dT_dt, dp)
    dry_kf = _col_int(out_kf.dq_v_dt, dp)
    assert jnp.all(heat_kf > 0.0), f"KF net column heating {heat_kf} W/m^2 <= 0"
    assert jnp.all(dry_kf < 0.0), f"KF net column drying {dry_kf} >= 0"


def test_tier3_kf_mse_conserved_on_capped_finite_lnb_column():
    """KF MSE closure must hold on the FINITE-LNB (capped) column too — KF's
    real production regime (a real tropopause), where the plume detrains genuine
    cloud.  The implicit_flux kernel books ``dq_v -= dq_c`` for that detrained
    cloud with NO sensible heat (it defers the latent), which left column MSE
    ``h=c_p T + L_v q_v`` short by ~-549 W/m^2 here.  KF now RELEASES the
    detrained-cloud latent as warming (h-conserving in-scheme, ZM convention),
    so both column total water and MSE close to machine precision on the capped
    column, not only the uncapped one (codex review-2 #2)."""
    T, q, pf, ph, u, v = _capped_column()
    dp = _dp(ph)
    out = _run_kf(T, q, pf, ph)
    water = _col_int(out.dq_v_dt + out.dq_c_conv_dt, dp)
    mse = _col_int(constants.c_pd * out.dT_dt + constants.L_v * out.dq_v_dt, dp)
    assert jnp.all(jnp.abs(water) < 1.0e-8), \
        f"KF capped-column water residual {water} not machine-zero"
    assert jnp.all(jnp.abs(mse) < 1.0e-5), \
        f"KF capped-column MSE residual {mse} W/m^2 — detrained-cloud latent leak"


@pytest.mark.parametrize("dt", [300.0, 1800.0, 3600.0])
@pytest.mark.parametrize("q_scale", [1.0, 0.1, 0.02])
def test_tier3_kf_vapor_stays_nonnegative(dt, q_scale):
    """KF must never drive column vapor negative after a step: the CONDLOAD
    condensation sink is drawn from the vapor the updraft ingests, distributed
    over the available (post-transport) moisture and moisture-limited so
    ``precip*dt <= column vapor``; a final CONSERVATIVE VAPOR RELOCATION fills any
    residual kernel-transport over-dry from the surplus deposited elsewhere.
    Probe the production dt range AND the drier / longer-dt corner (q_scale=0.02,
    dt=3600) that actually exercises the relocation clamp (codex review-2 #1: the
    earlier -cond_rate sink drove q_v to -4e-4; review-4: cover the relocation
    path on a nonzero-vapor column).  ``dt`` is threaded into the KF call so its
    internal moisture limiter / clamp see the same step used for the check."""
    T, q, pf, ph, u, v = _column()
    q = q * q_scale
    ncol, nlev = T.shape
    dp = _dp(ph)
    out, _ = kain_fritsch_convection(
        T, q, pf, ph, jnp.zeros((ncol, nlev)), jnp.zeros((ncol, nlev)),
        dt, KainFritschConfig(),
    )
    q_new = q + out.dq_v_dt * dt
    assert jnp.all(jnp.isfinite(out.dq_v_dt)) and jnp.all(jnp.isfinite(out.dT_dt))
    assert jnp.all(q_new >= -1.0e-12), \
        f"KF drove q_v negative (min={jnp.min(q_new)}) at dt={dt}, q_scale={q_scale}"
    # On a nonzero-vapor column the vapor relocation is conservative: column
    # total water stays closed (not the degenerate q_v==0 residual regime).
    water = _col_int(out.dq_v_dt + out.dq_c_conv_dt, dp)
    assert jnp.all(jnp.abs(water) < 1.0e-5), \
        f"KF column-water residual {water} at dt={dt}, q_scale={q_scale}"


def test_tier3_sbm_conserves_water_but_not_instantaneous_mse():
    """SBM (Betts-Miller) conserves column total water EXACTLY (the cloud-water
    source is the rescaled column-net drying) but does NOT instantaneously
    conserve column MSE: it relaxes T→T_ref and q→q_ref INDEPENDENTLY, so
    c_p dT + L_v dq ≠ 0 per step.  The enthalpy correction (sbm.py Newton
    iteration) is applied to the REFERENCE PROFILE, not the tendencies — this
    is the documented BM design, so SBM is excluded from the MSE tier.  We
    pin both facts here (codex review-1 #5)."""
    T, q, pf, ph, u, v = _column()
    out = _run_sbm(T, q, pf, ph)
    dp = _dp(ph)
    # total water conserved to machine precision
    net_water = _col_int(out.dq_v_dt + out.dq_c_conv_dt, dp)
    assert jnp.all(jnp.abs(net_water) < 1e-12), \
        f"SBM total-water not conserved: {net_water}"
    # MSE residual is small but non-zero (~1-3% of column heating) — NOT
    # conserved instantaneously, as the BM design implies.
    mse = _col_int(C_PD * out.dT_dt + L_V * out.dq_v_dt, dp)
    heat = _col_int(jnp.abs(C_PD * out.dT_dt), dp) + 1e-9
    rel = jnp.abs(mse) / heat
    assert jnp.all(rel > 5e-3), \
        f"SBM unexpectedly MSE-conserving (rel={rel}); revisit exclusion"
    assert jnp.all(rel < 0.1), \
        f"SBM MSE residual implausibly large (rel={rel})"


def test_tier3_kuo_consumes_supplied_convergence():
    """Kuo's vapor tendency includes the −ptenq removal of the large-scale
    source it consumes; column total water (vapor+cloud) plus the consumed
    convergence should balance.  Here we check the documented invariant:
    with zero supplied convergence Kuo is exactly quiescent."""
    T, q, pf, ph, u, v = _column()
    out = _run_kuo(T, q, pf, ph, mc=None)
    assert jnp.allclose(out.dT_dt, 0.0)
    assert jnp.allclose(out.dq_v_dt, 0.0)
    assert jnp.allclose(out.dq_c_conv_dt, 0.0)
    assert jnp.allclose(out.convective_mask, 0.0)


# ===========================================================================
# Tier 4 — DIFFERENTIABILITY
# ===========================================================================

def _scalar_loss(out):
    """Smooth scalar functional of the tendencies for grad probing."""
    return (
        jnp.sum(out.dT_dt ** 2)
        + jnp.sum(out.dq_v_dt ** 2)
        + jnp.sum(out.dq_c_conv_dt ** 2)
    )


# (scheme-name, (x0, builder-keyed-on-a-tunable-float)) — grad w.r.t. that
# float.  We deliberately probe an ALWAYS-LIVE rate / slope prefactor for
# every scheme (the timescale or slope that multiplies the whole tendency),
# NOT a CAPE *threshold*.  A CAPE threshold's gradient is correctly zero
# wherever the trigger sigmoid is saturated (huge-CAPE column) — that is not
# a dead-gradient bug, just a saturated sigmoid.  The grad-AT-threshold
# behaviour is already covered by each scheme's own test suite
# (e.g. test_zm_grad_through_cape_threshold_finite_at_threshold).  Probing a
# rate prefactor is both the physically meaningful trainability test and
# robust to the operating point.
_GRAD_CASES = {
    "sbm": (7200.0, lambda T, q, pf, ph, u, v, w, mc, x:
            _scalar_loss(_run_sbm(T, q, pf, ph, tau_c=x))),                 # tau_c [s]
    "dca_manabe": (10.0, lambda T, q, pf, ph, u, v, w, mc, x:
                   _scalar_loss(_run_dca_manabe(
                       T, q, pf, ph, instability_blend_sharpness=x))),       # blend sharpness
    "dca_ahmed_neelin": (0.6, lambda T, q, pf, ph, u, v, w, mc, x:
                         _scalar_loss(_run_dca_ahmed(T, q, pf, ph, a_mm_per_hr=x))),  # slope a
    "zhang_mcfarlane": (3600.0, lambda T, q, pf, ph, u, v, w, mc, x:
                        _scalar_loss(_run_zm(T, q, pf, ph, u, v, tau=x))),  # tau [s]
    "emanuel": (1.0, lambda T, q, pf, ph, u, v, w, mc, x:
                _scalar_loss(_run_emanuel(T, q, pf, ph, alpha_closure=x))),   # CBMF closure slope
    "kain_fritsch": (1800.0, lambda T, q, pf, ph, u, v, w, mc, x:
                     _scalar_loss(_run_kf(T, q, pf, ph, w=w,
                                          cape_consumption_time=x))),         # TIMEC [s]
    "bechtold": (3600.0, lambda T, q, pf, ph, u, v, w, mc, x:
                 _scalar_loss(_run_bechtold(T, q, pf, ph, u, v, mc=mc, tau_bl=x))),  # tau_bl [s]
}


# KF (TIMEC) and Bechtold (tau_bl) feed a CAPE/tau cloud-base mass flux that
# is hard-clipped at ``M_b_max`` (a CFL/stability cap).  In the deep-CAPE
# regime that clip is ACTIVE, which makes the relaxation-timescale gradient
# exactly zero (see test_tier4_relaxation_tau_grad_dies_under_M_b_cap_KNOWN
# below — a documented trainability limitation, NOT broken AD plumbing).  To
# verify the AD path itself is sound we probe these two at a milder
# (sub-cap) column where the clip is inactive and the gradient is alive.
_GRAD_SUBCAP = {"kain_fritsch", "bechtold"}


def _grad_column(name):
    if name in _GRAD_SUBCAP:
        return _column(T_sfc=295.0, q_sfc=11.0e-3, lapse_rate=6.2, rh_scale_m=2500.0)
    if name in _LNB_SCHEMES:
        return _capped_column()
    return _column()


@pytest.mark.parametrize("name", list(_GRAD_CASES))
def test_tier4_grad_through_tunable_finite_nonzero(name):
    T, q, pf, ph, u, v = _grad_column(name)
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)
    mc = jnp.full((ncol, nlev), 3.0e-6)
    x0, fn = _GRAD_CASES[name]
    g = jax.grad(lambda x: fn(T, q, pf, ph, u, v, w, mc, x))(x0)
    assert jnp.isfinite(g), f"{name}: non-finite grad {g}"
    assert g != 0.0, f"{name}: dead gradient w.r.t. rate/slope tunable"


@pytest.mark.xfail(
    strict=True,
    reason=(
        "KNOWN LIMITATION (not a crash): KF (and Bechtold) diagnose a "
        "CAPE/tau cloud-base mass flux that is HARD-clipped at M_b_max "
        "(~0.05 kg/m2/s) as a CFL/stability cap. In the deep-CAPE regime the "
        "clip is active (KF M_b_closure=0.88 >> 0.05), so jax.grad of the "
        "tendencies w.r.t. the relaxation timescale (TIMEC / tau_bl) is "
        "exactly 0 — the timescale is UNTRAINABLE precisely where convection "
        "fires. (NOTE: ZM uses the SAME hard-cap pattern but on this column "
        "its dL/dtau_cape ~= -2e-10 is non-zero — not clipped here — so ZM is "
        "NOT claimed dead; codex review-1 #3.) A smooth cap (softplus "
        "saturation instead of jnp.clip) would keep the gradient alive; "
        "deferred because it is a shared multi-scheme numerics change needing "
        "RCE revalidation of the cap behaviour."
    ),
)
def test_tier4_relaxation_tau_grad_dies_under_M_b_cap_KNOWN():
    """# BUG (trainability): KF AND Bechtold tau gradients are killed by the
    M_b_max clip in the deep-CAPE regime (asserted for both; ZM is NOT — its
    grad is alive on this column — so ZM is deliberately excluded)."""
    T, q, pf, ph, u, v = _column()              # high-CAPE → clip active
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)
    mc = jnp.full((ncol, nlev), 3.0e-6)

    def kf_loss(tau):
        out, _ = kain_fritsch_convection(
            T, q, pf, ph, w, jnp.zeros((ncol, nlev)), DT,
            KainFritschConfig(cape_consumption_time=tau),
        )
        return _scalar_loss(out)

    def be_loss(tau):
        out, _, _ = bechtold_convection(
            T, q, pf, ph, u, v, jnp.zeros((ncol, nlev)), jnp.zeros((ncol,)),
            None, DT, BechtoldConfig(tau_bl=tau), moisture_convergence=mc,
        )
        return _scalar_loss(out)

    g_kf = jax.grad(kf_loss)(1800.0)
    g_be = jax.grad(be_loss)(3600.0)
    # strict-xfail: BOTH are expected to be exactly 0 under the active clip,
    # so this "alive" assertion fails (documenting the limitation for both).
    assert g_kf != 0.0 and g_be != 0.0, (
        f"tau gradients should be alive; KF={g_kf}, Bechtold={g_be} "
        "(both dead under the M_b_max clip)"
    )


def test_tier4_kuo_grad_through_entrainment():
    """Kuo grad w.r.t. entrainment with supplied convergence is finite."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    mc = jnp.full((ncol, nlev), 3.0e-6)

    def loss(eps):
        return _scalar_loss(_run_kuo(T, q, pf, ph, mc=mc, entrainment=eps))

    g = jax.grad(loss)(1.0e-4)
    assert jnp.isfinite(g)


@pytest.mark.parametrize("name", SCHEME_NAMES)
def test_tier4_grad_through_state_finite(name):
    """Grad of the column loss w.r.t. the temperature field is finite (no
    NaN/Inf from clip/where/scan branches), even in the quiescent / stable
    limit."""
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)
    mc = jnp.full((ncol, nlev), 3.0e-6)

    def loss(Tx):
        return _scalar_loss(_all_schemes(Tx, q, pf, ph, u, v, w, mc)[name]())

    g = jax.grad(loss)(T)
    assert jnp.all(jnp.isfinite(g)), f"{name}: non-finite dL/dT"


@pytest.mark.parametrize("name", SCHEME_NAMES)
def test_tier4_jit_matches_eager(name):
    T, q, pf, ph, u, v = _column()
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.05)
    mc = jnp.full((ncol, nlev), 3.0e-6)
    eager = _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()

    def f(T, q, pf, ph, u, v, w, mc):
        return _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()

    jitted = jax.jit(f)(T, q, pf, ph, u, v, w, mc)
    assert jnp.allclose(eager.dT_dt, jitted.dT_dt, atol=1e-10, rtol=1e-7)
    assert jnp.allclose(eager.dq_v_dt, jitted.dq_v_dt, atol=1e-10, rtol=1e-7)


# ===========================================================================
# Tier 5 — IDEALIZED fidelity
# ===========================================================================

@pytest.mark.parametrize(
    "name",
    # CAPE-gated schemes should be quiescent on a stable, dry column.
    # bechtold is now INCLUDED: its spurious-heating runaway was fixed by the
    # launch-aware mass-flux cap ``cape_weight**2 * M_b_max`` (see the dedicated
    # regression test below + bechtold_convection).
    ["sbm", "dca_manabe", "zhang_mcfarlane", "emanuel", "kain_fritsch", "bechtold"],
)
def test_tier5_stable_dry_column_quiescent(name):
    """A strongly stable, dry column has no positive CAPE — convective
    heating should be negligible (< 1 W/m² column-integrated)."""
    T, q, pf, ph, u, v = _stable_dry_column()
    ncol, nlev = T.shape
    w = jnp.zeros((ncol, nlev))
    mc = jnp.zeros((ncol, nlev))
    out = _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()
    dp = _dp(ph)
    col_heat = jnp.abs(_col_int(C_PD * out.dT_dt, dp))    # |W/m²|
    assert jnp.all(col_heat < 1.0), \
        f"{name}: spurious convection on stable dry column: {col_heat} W/m²"


def test_tier5_bechtold_stable_dry_column_quiescent():
    """REGRESSION (was a confirmed BUG; NOW FIXED): Bechtold heated a stable,
    DRY, ZERO-CAPE column by ~3900 W/m^2.  Root cause: its deep/shallow plume
    has entrainment epsilon (1.75e-3..3.5e-3 /m) >> detrainment delta
    (7.5e-5 /m), so the bare budget ``M_u = M_b*exp(∫(eps-delta)dz)`` ran away
    by ~1e13x from a CAPE-gated launch (~2.8e-10) up to ~2700 kg/m2/s even
    though T_u-T<0 at EVERY level — then saturated at the *constant* M_b_max
    clip, ERASING the launch-time cape_weight gate.  FIX (validator codex review
    round-2): cap the relaxed ``M_u_new`` at ``cape_weight**2 * M_b_max`` instead
    of the absolute ``M_b_max`` (bechtold.py).  This carries the CAPE launch
    gate through the downstream transport: a zero-CAPE column (cape_weight ~9e-4
    here, the soft floor of the CAPE trigger) is capped ~10^6x lower so it stays
    quiescent (~4e-3 W/m^2, below the <1 W/m^2 bar; the square is what collapses
    that soft trigger floor — a bare cape_weight*M_b_max still left ~4.6 W/m^2),
    while a genuinely convecting column (cape_weight -> 1, so cape_weight**2 -> 1)
    keeps the full legacy M_b_max bound — byte-identical to the pre-fix scheme,
    so the closed column-MSE budget on a convecting column is untouched."""
    T, q, pf, ph, u, v = _stable_dry_column()
    ncol, nlev = T.shape
    out = _run_bechtold(T, q, pf, ph, u, v, mc=jnp.zeros((ncol, nlev)))
    dp = _dp(ph)
    col_heat = jnp.abs(_col_int(C_PD * out.dT_dt, dp))
    assert jnp.all(col_heat < 1.0), \
        f"bechtold: spurious convection on stable dry column: {col_heat} W/m²"


def test_tier5_ahmed_neelin_below_critical_quiescent():
    """Ahmed-Neelin precip is P = a (B_L − B_c)+; on a dry/stable column B_L
    sits well below B_c so the heating is ~zero."""
    T, q, pf, ph, u, v = _stable_dry_column()
    out = _run_dca_ahmed(T, q, pf, ph)
    dp = _dp(ph)
    col_heat = jnp.abs(_col_int(C_PD * out.dT_dt, dp))
    assert jnp.all(col_heat < 1.0)


@pytest.mark.parametrize("name", SCHEME_NAMES)
def test_tier5_destabilized_response_nonzero(name):
    """The flip side of quiescence: the destabilized / convergent column
    must produce a clearly non-zero response."""
    T, q, pf, ph, u, v = _capped_column() if name in _LNB_SCHEMES else _column()
    ncol, nlev = T.shape
    w = jnp.full((ncol, nlev), 0.1)
    mc = jnp.full((ncol, nlev), 5.0e-6)
    out = _all_schemes(T, q, pf, ph, u, v, w, mc)[name]()
    assert jnp.max(jnp.abs(out.dT_dt)) > 1e-8, f"{name}: dead on unstable col"


def test_tier5_kuo_quiescent_without_convergence():
    """Canonical Kuo: zero large-scale convergence → exactly zero
    tendencies (physically correct single-column behaviour)."""
    T, q, pf, ph, u, v = _column()
    out = _run_kuo(T, q, pf, ph, mc=None)
    assert jnp.max(jnp.abs(out.dT_dt)) == 0.0
    assert jnp.max(jnp.abs(out.dq_v_dt)) == 0.0


# ===========================================================================
# _triggers — shared smooth primitives
# ===========================================================================

def test_triggers_smooth_max_upper_bounds_and_diff():
    from legoesm.core.smooth import smooth_max, smooth_min
    from legoesm.atmosphere.physics.convection._triggers import (
        smooth_positive_part, cape_trigger,
    )
    a = jnp.array([1.0, -2.0, 3.5])
    b = jnp.array([0.5, 1.0, 3.5])
    sm = smooth_max(a, b, sharpness=50.0)
    assert jnp.all(sm >= jnp.maximum(a, b) - 1e-6)        # upper bound
    smin = smooth_min(a, b, sharpness=50.0)
    assert jnp.all(smin <= jnp.minimum(a, b) + 1e-6)
    # positive-part >= 0 and ~max(x,0) for large sharpness
    spp = smooth_positive_part(jnp.array([-1.0, 0.0, 2.0]), sharpness=50.0)
    assert jnp.all(spp >= 0.0)
    assert jnp.isclose(spp[-1], 2.0, atol=1e-2)
    # cape_trigger monotone in cape, finite grad at the threshold.
    g = jax.grad(lambda c: cape_trigger(c, 100.0, 0.02))(100.0)
    assert jnp.isfinite(g) and g > 0.0


def test_triggers_lowest_crossing_index_finite_grad():
    from legoesm.atmosphere.physics.convection._triggers import (
        smooth_lowest_crossing_index,
    )
    # profile increasing upward (surface-last): crosses threshold once.
    prof = jnp.array([[10.0, 8.0, 6.0, 4.0, 2.0, 0.0]])   # top→surface
    idx = smooth_lowest_crossing_index(prof, 5.0, sharpness=5.0)
    assert jnp.all(jnp.isfinite(idx))

    def loss(p):
        return jnp.sum(smooth_lowest_crossing_index(p, 5.0, 5.0))

    g = jax.grad(loss)(prof)
    assert jnp.all(jnp.isfinite(g))
