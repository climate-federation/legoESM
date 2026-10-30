"""Deep Convective Adjustment (DCA) scheme.

Two variants, routed by ``DCAConfig.variant``:

* ``"manabe"`` (default): Manabe moist convective adjustment, a port of
  GFDL ``atmos_phys`` tag ``2021.02`` (commit
  46817a12232ed54d1a5d5a4394a695621f575812)
  ``atmos_param/moist_conv/moist_conv.F90`` (``moist_conv``, coldT = F, no
  cloud-tracer detrainment, namelist defaults).  Layer pairs where BOTH levels
  are supersaturated and the lapse rate exceeds the moist adiabat by more than
  ``ALTOL`` form unstable blocks; each block is set to the moist adiabat
  conserving moist enthalpy with the saturation humidity linearised in T
  (block solve), ``ITSMOD = 30`` iterations per tolerance, ``ALTOL`` doubling
  from 0.02 K while <= 0.10 K.  Unsaturated layers are never adjusted.  The
  oracle's quirks are kept: within an iteration the column is finished as
  soon as the block search finds no further block, so a column is re-tested
  only when its lowest block reaches the bottom two levels.  Precipitation is
  ``max(column drying, 0)``.  Saturation curve and constants are legoESM's
  (the oracle fixture was produced with them).  The existing smooth CAPE gate
  (``cape_threshold``, ``cape_sharpness``; no counterpart in the oracle) still
  scales the result; ``cape_threshold < 0`` with a hard sharpness is the
  oracle.  Matched to the RUNNING oracle in
  ``tests/atmosphere/test_dca_moist_conv_oracle.py`` (fixture from
  ``scripts/validate/oracle/dca_moist_conv/``).  Piecewise differentiable
  (hard switches as ``jnp.where``; fixed 90-iteration ``lax.scan``).

* ``"ahmed_neelin"``: the Ahmed-Neelin-Adames (2020) lower-tropospheric-
  buoyancy (B_L) precipitation-buoyancy closure (``ahmed_neelin_dca``).
  The operative closure is the empirical eq-(8) precipitation–buoyancy
  relation ``P = a·(B_L − B_c)+``; the implied column latent heating is
  partitioned heating-up / drying-down along the eq-(41) direction
  (latent cooling cancels sensible heating per level), which conserves
  column moist static energy and drives ``B_L`` toward ``B_c``.  The
  column relaxation time scale is emergent (the paper's nominal ≈2 h is
  derived from observational EOF structures unavailable in-model).  See
  :func:`ahmed_neelin_dca`.

Not ported (remainder): the oracle's ``coldT`` path (sublimation latent heat
and snow) and its stratiform-cloud detrainment (``beta``, cloud tracers).
There is no dry convective adjustment here; the gray RCE column refuses one.

References
----------
- GFDL atmos_phys 2021.02, atmos_param/moist_conv/moist_conv.F90.
- Manabe, S., Smagorinsky, J., & Strickler, R. F. (1965).
  Simulated climatology of a general circulation model with a
  hydrological cycle. Mon. Wea. Rev., 93, 769-798.
- Ahmed, F., Adames, A. F., & Neelin, J. D. (2020). Deep convective
  adjustment of temperature and moisture. J. Atmos. Sci., 77, 2163-2186.

"""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.thermo import (
    saturation_specific_humidity,
    saturation_specific_humidity_dT,
    saturation_vapor_pressure,
)
from legoesm.atmosphere.physics.thermodynamics import compute_cape
from legoesm.atmosphere.physics.convection.config import (
    AhmedNeelinDCAConfig,
    DCAConfig,
)
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection._triggers import cape_trigger
from legoesm.atmosphere.physics._shared import safe_divide

__physics_contract__ = {
    "summary": (
        "Deep convective adjustment (variant dispatch): Manabe moist "
        "convective adjustment (port of GFDL atmos_phys 2021.02 moist_conv: "
        "supersaturated unstable blocks set to the moist adiabat, moist "
        "enthalpy conserved, excess vapour precipitated) or the "
        "Ahmed-Neelin-Adames (2020) lower-tropospheric-buoyancy precipitation "
        "closure; the dried vapor is handed out as cloud water."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa", "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s", "dq_c_conv_dt": "kg/kg/s",
        "cape": "J/kg (B_L in m/s^2 for the ahmed_neelin variant)",
        "convective_mask": "1 (0-1 convective indicator)",
    },
    "sign_convention": (
        "Warms and dries where it stabilizes; the condensed vapor becomes "
        "cloud water (dq_c_conv_dt>=0, handed to microphysics) whose column "
        "integral is the oracle precipitation max(-int(dq_v) dp/g, 0); each "
        "Manabe block conserves int(c_pd dT + L_v dq_v) dp exactly (the new "
        "humidity is q_sat + dq_sat/dT * (T_new - T) and the block temperature "
        "solves the enthalpy balance with it); surface at the last vertical "
        "index."
    ),
    "conserves": ["energy", "moisture"],
    "differentiable": True,
    "reference": (
        "Manabe, Smagorinsky & Strickler (1965), Mon. Wea. Rev. 93, 769-798; "
        "Ahmed, Adames & Neelin (2020), J. Atmos. Sci. 77, 2163-2186"
    ),
    "idealized_test": (
        "tests/atmosphere/test_dca_moist_conv_oracle.py: 23 columns vs the "
        "running GFDL moist_conv to 1e-10 with the CAPE gate off; unsaturated "
        "columns are untouched."
    ),
}


# --- GFDL atmos_phys 2021.02 moist_conv namelist defaults (moist_conv.F90 l.38-42) ---
_MCA_HC = 1.00          # HC: relative humidity at which saturation is assumed
_MCA_TOL_MIN_K = 0.02   # TOLmin: initial lapse-rate tolerance ALTOL [K]
_MCA_TOL_MAX_K = 0.10   # TOLmax: largest ALTOL before giving up [K]
_MCA_ITSMOD = 30        # ITSMOD: iterations per tolerance
# ALTOL doubles from TOLmin while <= TOLmax: 0.02, 0.04, 0.08 -> three stages.
_MCA_N_TOL_STAGES = 1 + int(math.floor(math.log2(_MCA_TOL_MAX_K / _MCA_TOL_MIN_K)))

_INERT_MANABE_FIELDS = ("n_iterations", "mixing_fraction", "instability_blend_sharpness")


def _es_des(T):
    """``lookup_es_des``: e_s and de_s/dT of legoESM's Tetens curve."""
    return jax.jvp(saturation_vapor_pressure, (T,), (jnp.ones_like(T),))


def _compute_qs(T, p):
    """``compute_qs(T, p, qs, dqsdT, hc, esat)`` with legoESM's q_sat curve."""
    qs = saturation_specific_humidity(T, p)
    dqs = saturation_specific_humidity_dT(T, p)
    es = saturation_vapor_pressure(T) * _MCA_HC
    return qs, dqs, es


def _alrm(T, pf, ph, HL):
    """Moist-adiabatic temperature difference between adjacent levels, ALRM(k)."""
    rd, rv, cp = constants.R_d, constants.R_v, constants.c_pd
    d622 = rd / rv
    rocp = rd / cp
    delpop = (pf[1:] - pf[:-1]) / ph[1:-1]
    thaf = 0.50 * (T[:-1] + T[1:])
    es, des = _es_des(thaf)
    esm = _MCA_HC * es
    esd = _MCA_HC * des
    return (rocp * delpop * thaf * (ph[1:-1] + d622 * HL * esm / thaf / rd)
            / (ph[1:-1] + d622 * HL * esd / cp))


def _ivf(T, Qdif, Qsat, ALRM, altol):
    """IVF(k) = 1 where levels k and k+1 are both supersaturated and the
    layer is unstable beyond the tolerance; IVF(KX) = 0."""
    sat = Qdif * Qsat > 0.0
    ivf = sat[:-1] & sat[1:] & ((T[1:] - T[:-1]) > (ALRM + altol))
    return jnp.concatenate([ivf, jnp.zeros((1,), bool)])


def _row_seq_sums(mask, vals):
    """out[k] = sum over kk (in increasing kk) of vals[kk] where mask[k, kk]."""
    def add(c, x):
        m, v = x
        return c + jnp.where(m, v, 0.0), None
    return jax.lax.scan(add, jnp.zeros(mask.shape[0], vals.dtype), (mask.T, vals))[0]


def _moist_conv_column(Tin, Qin, pf, ph):
    """``moist_conv`` for one column (coldT = F, no cloud tracers, Lbot absent):
    returns the adjustments Tdel, Qdel and Rain [kg/m2] (before ``dtinv``)."""
    cp, HL = constants.c_pd, constants.L_v
    d378 = 1.0 - constants.R_d / constants.R_v
    kx = Tin.shape[0]
    lev = jnp.arange(kx)
    zero = jnp.zeros((), Tin.dtype)
    pdelta = ph[1:] - ph[:-1]
    Qsat, Esdv, Esat = _compute_qs(Tin, pf)
    Qdif = jnp.maximum(0.0, Qin - Qsat)
    ALRM = _alrm(Tin, pf, ph, HL)
    IVF = _ivf(Tin, Qdif, Qsat, ALRM, _MCA_TOL_MIN_K)

    def iteration(state, s):
        Temp, Qmix, Qsat, Esdv, Esat, Qdif, ALRM, IVF, active = state
        it = s % _MCA_ITSMOD + 1
        altol = _MCA_TOL_MIN_K * 2.0 ** (s // _MCA_ITSMOD)
        # Unstable blocks KTOP..KBOT: a run of IVF = 1 at KTOP..KBOT-1 plus
        # KBOT, the first level below with IVF = 0.  Blocks are disjoint and
        # each solve reads only its own levels, so all are solved at once.
        in_blk = IVF | jnp.concatenate([jnp.zeros((1,), bool), IVF[:-1]])

        def first_zero(c, x):
            k, v = x
            c = jnp.where(v, c, k)
            return c, c
        kbot = jax.lax.scan(first_zero, kx - 1, (lev[::-1], IVF[::-1]))[1][::-1]
        C = jnp.where(pf - d378 * Esat <= 0.0, 0.0, Esdv)
        alrm_k = jnp.concatenate([ALRM, jnp.zeros((1,), ALRM.dtype)])
        sum0 = _row_seq_sums((lev[None, :] >= lev[:, None]) & (lev[None, :] <= kbot[:, None] - 1),
                             alrm_k)
        t1 = pdelta * ((cp + HL * C) * (Temp + sum0) + HL * (Qmix - Qsat))
        t2 = pdelta * (cp + HL * C)
        prev_same = jnp.concatenate([jnp.zeros((1,), bool), in_blk[:-1] & (kbot[:-1] == kbot[1:])])
        start = in_blk & ~prev_same

        def block_sums(c, x):
            st, ib, a, b = x
            s1 = jnp.where(st, zero, c[0]) + jnp.where(ib, a, zero)
            s2 = jnp.where(st, zero, c[1]) + jnp.where(ib, b, zero)
            return (s1, s2), (s1, s2)
        S1, S2 = jax.lax.scan(block_sums, (zero, zero), (start, in_blk, t1, t2))[1]
        ta_bot = (S1 / jnp.where(in_blk, S2, 1.0))[kbot]
        Ta = jnp.where(lev == kbot, ta_bot, ta_bot - sum0)
        Qa = Qsat + C * (Ta - Temp)
        upd = active & in_blk
        Temp_n = jnp.where(upd, Ta, Temp)
        Qmix_n = jnp.where(upd, Qa, Qmix)
        qs, dqs, es = _compute_qs(Temp_n, pf)
        Qsat_n = jnp.where(upd, qs, Qsat)
        Esdv_n = jnp.where(upd, dqs, Esdv)
        Esat_n = jnp.where(upd, es, Esat)
        Qdif_n = jnp.where(upd, jnp.maximum(0.0, Qmix_n - Qsat_n), Qdif)
        alrm_new = _alrm(Temp_n, pf, ph, HL)
        ALRM_n = jnp.where(upd[:-1] & (lev[:-1] < kbot[:-1]), alrm_new, ALRM)
        # The block search ends the column unless the last block reached
        # KBOT >= MXLEV1 (kstart > MXLEV1): only then the column is re-tested.
        fall = IVF[kx - 2] | IVF[kx - 3]
        retest = active & fall & (it < _MCA_ITSMOD)
        ALRM_r = jnp.where(retest, alrm_new, ALRM_n)
        IVF_r = jnp.where(retest, _ivf(Temp_n, Qdif_n, Qsat_n, alrm_new, altol), IVF)
        last_stage = s // _MCA_ITSMOD == _MCA_N_TOL_STAGES - 1
        active_n = active & fall & jnp.where(it < _MCA_ITSMOD, jnp.any(IVF_r), ~last_stage)
        return (Temp_n, Qmix_n, Qsat_n, Esdv_n, Esat_n, Qdif_n, ALRM_r, IVF_r, active_n), None

    state = (Tin, Qin, Qsat, Esdv, Esat, Qdif, ALRM, IVF, jnp.any(IVF))
    state = jax.lax.scan(iteration, state, jnp.arange(_MCA_ITSMOD * _MCA_N_TOL_STAGES))[0]
    Tdel = state[0] - Tin
    Qdel = state[1] - Qin

    def add(c, x):
        return c + x, None
    rain = jax.lax.scan(add, zero, (ph[:-1] - ph[1:]) * Qdel * (1.0 / constants.g))[0]
    return Tdel, Qdel, jnp.maximum(rain, 0.0)


def _manabe_dca_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: DCAConfig = DCAConfig(),
) -> ConvectionOutput:
    """Manabe moist convective adjustment (GFDL ``moist_conv``) tendencies.

    Parameters
    ----------
    T, q_v : jax.Array
        Temperature [K] and water-vapour specific humidity [kg/kg] at full
        levels, shape (ncol, nlev), level 0 = model top.
    p_full, p_half : jax.Array
        Full- and half-level pressure [Pa], shapes (ncol, nlev), (ncol, nlev+1).
    dt : float
        Model time step [s]; adjustments are returned per second (``dtinv``).
    config : DCAConfig
        ``cape_threshold`` / ``cape_sharpness`` set the existing CAPE gate
        (no oracle counterpart; ``cape_threshold < 0`` with a hard sharpness
        disables it).  ``n_iterations``, ``mixing_fraction`` and
        ``instability_blend_sharpness`` belonged to the replaced pairwise
        sweep and must stay at their defaults.

    Returns
    -------
    ConvectionOutput
    """
    traced_inert = []
    for name in _INERT_MANABE_FIELDS:
        v = getattr(config, name)
        if isinstance(v, jax.core.Tracer):
            traced_inert.append(jnp.any(v != DCAConfig._field_defaults[name]))
        elif np.any(np.asarray(v) != DCAConfig._field_defaults[name]):
            raise ValueError(
                f"DCAConfig.{name} belonged to the replaced pairwise sweep; the "
                "GFDL moist_conv port has no counterpart, so the field is inert "
                f"and must stay at its default {DCAConfig._field_defaults[name]}")
    dtype = jnp.result_type(T, q_v, p_full, p_half)
    T, q_v, p_full, p_half = (jnp.asarray(x, dtype) for x in (T, q_v, p_full, p_half))
    Tdel, Qdel, rain = jax.vmap(_moist_conv_column)(T, q_v, p_full, p_half)
    dtinv = 1.0 / dt
    dT, dq, rain = Tdel * dtinv, Qdel * dtinv, rain * dtinv
    # Existing CAPE gate (no counterpart in moist_conv): R_d * positive part of
    # the adjustment's warming, integrated in dp/p.
    cape = compute_cape(T, T + Tdel, p_full, p_half)
    if traced_inert:
        # traced config (jit argument): a non-default inert field cannot raise,
        # so it poisons the outputs instead of being ignored
        cape = jnp.where(jnp.any(jnp.stack(traced_inert)), jnp.nan, cape)
    cape_gate = cape_trigger(cape, config.cape_threshold, config.cape_sharpness)
    dT_dt = cape_gate[:, None] * dT
    dq_v_dt = cape_gate[:, None] * dq
    # Condensate source: local drying rescaled so its column integral is the
    # oracle Rain (AD-safe divide, issue #249).
    dp = p_half[:, 1:] - p_half[:, :-1]
    local_cond = jnp.maximum(-dq_v_dt, 0.0)
    col_local_cond = jnp.sum(local_cond * dp / constants.g, axis=1, keepdims=True)
    dq_c_conv_dt = local_cond * safe_divide(
        (cape_gate * rain)[:, None], col_local_cond, eps=1e-20)
    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=cape_gate,
        precip_frozen_sfc=jnp.zeros_like(cape),  # coldT = F: liquid only
    )


# ===========================================================================
# Ahmed-Neelin-Adames (2020) lower-tropospheric-buoyancy (B_L) closure
# ===========================================================================
#
# Reference: Ahmed, Adames & Neelin (2020), J. Atmos. Sci. 77, 2163-2186
# (ANA20).  Equation numbers below refer to that paper.
#
# Moist enthalpy (ANA20, after eq 6):  e = T + (L_v/c_p)·q   [K]
#   (q is specific humidity; e* uses saturation specific humidity q*).
# Exner function:                       Π(p) = (p/p0)^κ        [-]
# Buoyancy (eq 7):
#   B_L = (g·Π_L/e_L*)·[ w_b·(e_B/Π_B) + w_L·(e_L/Π_L) − e_L*/Π_L ]  [m/s²]
# Precipitation (eq 8):  P = a·(B_L − B_c)·H(B_L − B_c)               [kg/m²/s]


def _exner(p: jax.Array) -> jax.Array:
    """Exner function Π(p) = (p/p_ref)^κ [-]; p_ref, κ from constants."""
    return (p / constants.p_ref) ** constants.kappa


def _layer_membership(
    p_full: jax.Array,
    p_top,
    p_bot,
    edge_width_pa: float,
) -> jax.Array:
    """Smooth (sigmoid) 0–1 membership weight for a pressure layer.

    Returns ~1 for ``p_top < p < p_bot`` and ~0 outside, with sigmoid
    transitions of half-width ``edge_width_pa`` at each edge.  A hard
    boolean pressure mask kills ``jax.grad`` through the layer boundaries;
    the product of two sigmoids keeps the layer averages differentiable.

    Parameters
    ----------
    p_full : jax.Array
        Full-level pressure [Pa], shape (ncol, nlev).
    p_top, p_bot : float
        Top (lower pressure) and bottom (higher pressure) of the layer
        [Pa], with ``p_top < p_bot``.
    edge_width_pa : float
        Sigmoid transition half-width [Pa].

    Returns
    -------
    jax.Array
        Membership weight in [0, 1], shape (ncol, nlev).
    """
    # below_bot ≈ 1 where p <= p_bot (inside, on the high-pressure side)
    below_bot = jax.nn.sigmoid((p_bot - p_full) / edge_width_pa)
    # above_top ≈ 1 where p >= p_top (inside, on the low-pressure side)
    above_top = jax.nn.sigmoid((p_full - p_top) / edge_width_pa)
    return below_bot * above_top


def _layer_average(
    field: jax.Array,
    dp: jax.Array,
    membership: jax.Array,
    eps: float = 1.0,
) -> jax.Array:
    """Mass-weighted layer average of ``field`` over a smooth layer.

    ⟨X⟩ = Σ_k X_k · w_k · Δp_k / Σ_k w_k · Δp_k, where ``w_k`` is the
    smooth layer membership and ``Δp_k`` the layer thickness [Pa].
    AD-safe: the denominator carries the same membership so it never
    vanishes when the layer is populated, and an ``eps`` [Pa] floor guards
    the empty-layer limit.

    Parameters
    ----------
    field : jax.Array
        Field to average [arbitrary unit], shape (ncol, nlev).
    dp : jax.Array
        Layer thickness [Pa], shape (ncol, nlev), positive.
    membership : jax.Array
        Smooth 0–1 layer membership, shape (ncol, nlev).
    eps : float
        Denominator floor [Pa] for the empty-layer limit.

    Returns
    -------
    jax.Array
        Mass-weighted layer mean [same unit as field], shape (ncol,).
    """
    w = membership * dp
    num = jnp.sum(field * w, axis=-1)
    den = jnp.sum(w, axis=-1)
    return num / jnp.clip(den, eps, None)


def _compute_BL(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    dp: jax.Array,
    cfg: AhmedNeelinDCAConfig,
) -> tuple[jax.Array, jax.Array, jax.Array, jax.Array, jax.Array]:
    """Lower-tropospheric buoyancy B_L (ANA20 eq 7) and its building blocks.

    Parameters
    ----------
    T : jax.Array
        Temperature [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Full-level pressure [Pa], shape (ncol, nlev).
    dp : jax.Array
        Layer thickness [Pa], shape (ncol, nlev), positive.
    cfg : AhmedNeelinDCAConfig
        Ahmed-Neelin configuration.

    Returns
    -------
    BL : jax.Array
        Lower-tropospheric buoyancy [m/s²], shape (ncol,).
    member_bl : jax.Array
        Boundary-layer membership, shape (ncol, nlev).
    member_lft : jax.Array
        Lower-free-troposphere membership, shape (ncol, nlev).
    Pi_L : jax.Array
        LFT-averaged Exner function [-], shape (ncol,).
    e_L_star : jax.Array
        LFT-averaged saturation moist enthalpy [K], shape (ncol,).
    """
    Lv_cp = constants.L_v / constants.c_pd  # [K per (kg/kg)]

    # Moist enthalpy fields (ANA20 after eq 6).  ``q_sat`` is the saturation
    # MIXING RATIO, matching the model's ``q_v`` convention: legoESM
    # initialises and carries ``q_v`` as a mixing ratio (it is built from
    # ``saturation_specific_humidity``), so e = T + (L_v/c_p)·q_v and
    # e* = T + (L_v/c_p)·q_sat use the SAME water variable.  (ANA20 writes
    # ``q`` loosely as "specific humidity"; at tropical q≈20 g/kg the
    # mixing-ratio vs specific-humidity difference is ~2% in e and far
    # smaller in B_L, but using one consistent variable avoids a spurious
    # systematic offset between e and e* — codex review-1 finding.)
    q_sat = saturation_specific_humidity(T, p_full)  # [kg/kg] mixing ratio
    e = T + Lv_cp * q_v          # [K]
    e_star = T + Lv_cp * q_sat   # [K]
    Pi = _exner(p_full)          # [-]

    # Surface-aware layer edges.  ANA20 defines the BL as "surface → 850 hPa"
    # (Δp_B ≈ 150 hPa).  Over high topography / low surface pressure a FIXED
    # 850-hPa BL top can sit above the surface, leaving the BL empty and the
    # ratios e_B/Π_B meaningless (codex review-1 finding).  We therefore
    # clamp the BL top to stay a minimum depth below the surface pressure
    # (``p_s − layer_floor``) and the LFT top a minimum depth below the BL
    # top, so both layers always retain mass and degrade gracefully on a
    # shallow column.  On a standard p_s ≈ 1000 hPa column the clamps are
    # inactive and the edges are exactly the config 850 / 500 hPa.
    # Surface pressure ≈ bottom interface = lowest full level + half its
    # thickness (p_full ordered top→bottom, so index −1 is the surface
    # layer).  Avoids changing ``_compute_BL``'s signature for all callers.
    p_s = (p_full[:, -1:] + 0.5 * dp[:, -1:])            # (ncol, 1) [Pa]
    layer_floor = cfg.layer_min_depth_pa                 # min layer depth [Pa]
    p_bl_top = jnp.minimum(cfg.p_bl_top_pa, p_s - layer_floor)        # (ncol,1)
    p_lft_top = jnp.minimum(cfg.p_lft_top_pa, p_bl_top - layer_floor)  # (ncol,1)
    # BL high-pressure edge left open above the surface so the lowest model
    # levels are always counted.
    p_surface_ceiling = p_s + 10.0 * cfg.layer_edge_width_pa
    member_bl = _layer_membership(
        p_full, p_bl_top, p_surface_ceiling, cfg.layer_edge_width_pa,
    )
    member_lft = _layer_membership(
        p_full, p_lft_top, p_bl_top, cfg.layer_edge_width_pa,
    )

    # Layer-averaged moist enthalpies and Exner functions.
    e_B = _layer_average(e, dp, member_bl)            # [K]
    e_L = _layer_average(e, dp, member_lft)           # [K]
    e_L_star = _layer_average(e_star, dp, member_lft)  # [K]
    Pi_B = _layer_average(Pi, dp, member_bl)          # [-]
    Pi_L = _layer_average(Pi, dp, member_lft)         # [-]

    # ANA20 eq (7):
    #   B_L = (g·Π_L/e_L*)·[ w_b·(e_B/Π_B) + w_L·(e_L/Π_L) − e_L*/Π_L ]
    bracket = (
        cfg.w_b * (e_B / Pi_B)
        + cfg.w_l * (e_L / Pi_L)
        - e_L_star / Pi_L
    )
    BL = (constants.g * Pi_L / e_L_star) * bracket  # [m/s²]
    return BL, member_bl, member_lft, Pi_L, e_L_star


def _a_si(cfg: AhmedNeelinDCAConfig) -> float:
    """Slope ``a`` of the P–B_L line in SI mass-flux units.

    ANA20 Table 1 reports ``a`` in mm h⁻¹ per (m s⁻²).  Convert to
    [kg m⁻² s⁻¹ per (m s⁻²)] using water density ρ_w = 1000 kg/m³ and
    3600 s/h: ``a_SI = a_mm_per_hr · ρ_w / (1000 · 3600)`` since 1 mm of
    water = 1 kg/m² (ρ_w · 1e-3 m).
    """
    rho_w = constants.rho_water  # density of liquid water [kg/m³]
    # 1 mm/h of rain = (rho_w * 1e-3 m) / 3600 s = rho_w / 3.6e6 kg/m²/s
    return cfg.a_mm_per_hr * rho_w / 3.6e6  # coeff-ok: mm/hr -> kg/m^2/s


def _precip_from_BL(
    BL: jax.Array,
    cfg: AhmedNeelinDCAConfig,
) -> jax.Array:
    """Precipitation P = a·(B_L − B_c)·H(B_L − B_c) (ANA20 eq 8).

    The Heaviside is smoothed to a softplus so the ramp is differentiable
    at B_c:  ``softplus(s·x)/s → max(x, 0)`` as ``s → ∞``.  Output is a
    non-negative mass flux [kg/m²/s].

    Parameters
    ----------
    BL : jax.Array
        Lower-tropospheric buoyancy [m/s²], shape (ncol,).
    cfg : AhmedNeelinDCAConfig
        Configuration (slope ``a``, critical ``B_c``, sharpness).

    Returns
    -------
    jax.Array
        Precipitation [kg/m²/s], shape (ncol,), >= 0.
    """
    s = cfg.precip_heaviside_sharpness
    excess = jax.nn.softplus(s * (BL - cfg.b_c)) / s  # smooth max(BL-Bc, 0)
    return _a_si(cfg) * excess


def ahmed_neelin_dca(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: AhmedNeelinDCAConfig = AhmedNeelinDCAConfig(),
) -> ConvectionOutput:
    """Ahmed-Neelin-Adames (2020) B_L convective-adjustment tendencies.

    The OPERATIVE closure is the empirical precipitation-buoyancy relation
    eq (8) — the central result the paper validates against TRMM/ERA data
    (Fig 3).  The column heating and drying are set so that the
    column-integrated precipitation equals
    ``P = a·(B_L − B_c)·H(B_L − B_c)`` (eq 8), moved along the eq-(41)
    direction that conserves column moist static energy:

    * **Precipitation (eq 8, OPERATIVE).** ``P = a·(B_L − B_c)+`` with the
      empirical slope ``a`` (config ``a_mm_per_hr``) and critical buoyancy
      ``B_c`` (config ``b_c``).  The Heaviside is softplus-smoothed for
      differentiability.  The implied column latent heating is
      ``Q̂_c = L_v·P`` (ANA20 eq 11), so ``∫ c_p·dT/dt dp/g = L_v·P``
      holds to machine precision.

    * **Adjustment direction (eq 41).** The heating is moved along the
      second eigenvector of the linearised system (slope −1 in the
      moist-enthalpy q̂–T̂ plane): per level the latent cooling exactly
      cancels the sensible heating, ``L_v·dq_v/dt + c_p·dT/dt = 0``.  Hence
      column moist static energy is conserved during the adjustment, and
      drying coincides with warming.

    * **Adjustment time scale (eqs 38-42).** The buoyancy excess relaxes
      toward zero with an EMERGENT time scale set by eq (8) and the
      column's own buoyancy sensitivity to the heating direction.  ANA20
      report ``τ_c ≈ 2 h`` (eq 42); that value is itself DERIVED from
      ``a`` together with the observational EOF vertical structures
      (``L_{TL}``, ``L_{qL}``, …; eqs 25-27) that are not available
      in-model, so we surface the emergent τ as a diagnostic rather than
      imposing it.  See `config.tau_adjust_s` — it is retained for
      reference / optional rescaling but is NOT used to set the operative
      rate (which is eq 8).

    Conventions: ``dq_v_dt < 0`` (drying), ``dT_dt > 0`` (latent heating)
    where convecting; ``q_v`` stays non-negative (the per-step adjustment
    is capped at the available vapor, preserving the column MSE balance).

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).  (legoESM
        carries ``q_v`` as a mixing ratio; ``e`` and ``e*`` use the same
        variable — see `_compute_BL`.)
    p_full : jax.Array
        Full-level pressure [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Half-level pressure [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : AhmedNeelinDCAConfig
        Ahmed-Neelin configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.  ``cape`` carries the
        diagnostic ``B_L`` [m/s²] (not Joules) for this scheme;
        ``convective_mask`` is the smooth precipitating indicator.
    """
    ncol, nlev = T.shape
    dp = p_half[:, 1:] - p_half[:, :-1]            # (ncol, nlev) [Pa] > 0
    g = constants.g
    Lv = constants.L_v
    c_pd = constants.c_pd

    # --- B_L and the OPERATIVE precipitation (ANA20 eqs 7, 8) -------------
    BL, member_bl, member_lft, _Pi_L, _e_L_star = _compute_BL(
        T, q_v, p_full, dp, config,
    )                                              # BL: (ncol,) [m/s²]
    precip = _precip_from_BL(BL, config)           # (ncol,) [kg/m²/s] >= 0  (eq 8)
    # Implied column-integrated latent heating Q̂_c = L_v·P (ANA20 eq 11).
    Qc_col = Lv * precip                           # (ncol,) [W/m²]

    # --- Vertical structure: BL+LFT mass shape ---------------------------
    # ``member_union`` is the union of the BL and LFT memberships (the
    # layers that define B_L).  ``shape_k`` [1/Pa] is normalised so that
    # Σ_k shape_k·Δp_k = 1 on a populated column; it vanishes on a
    # degenerate all-stratosphere column.
    member_union = member_bl + member_lft           # (ncol, nlev) [-]
    member_mass = jnp.sum(member_union * dp, axis=-1)  # (ncol,) [Pa]
    shape = member_union * safe_divide(
        jnp.ones_like(member_mass), member_mass, eps=1.0,
    )[:, None]                                       # (ncol, nlev) [1/Pa]

    # --- Heating from eq (8), distributed; drying along eq (41) ----------
    # Heating per level [K/s]:  dT_dt_k = (Q̂_c / c_p) · g · shape_k.
    #   Σ_k c_p·dT_dt_k·Δp_k/g = Q̂_c·Σ_k shape_k·Δp_k = Q̂_c   (eq 11). ✓
    heating_rate = (Qc_col / c_pd)[:, None] * g * shape   # (ncol, nlev) [K/s]
    # Slope −1 in moist-enthalpy units (eq 41): per level latent cooling
    # cancels sensible heating ⇒ column MSE conserved and
    # ∫ L_v·dq dp/g = −∫ c_p·dT dp/g = −L_v·P exactly.
    drying_rate = -(c_pd / Lv) * heating_rate        # (ncol, nlev) [kg/kg/s]

    # --- Positivity: cap the per-step drying at the available vapor ------
    # A single step must not drive q_v negative.  Scale the WHOLE column's
    # heating+drying by one factor so the MSE balance (eq 41) and the
    # latent-heating ↔ precip closure stay exact under the cap.
    max_drying = -q_v / dt                           # (ncol, nlev) <= 0
    frac_level = jnp.where(
        drying_rate < max_drying,                    # would over-dry
        safe_divide(max_drying, drying_rate, eps=1e-30, fill=1.0),
        jnp.ones_like(drying_rate),
    )
    frac = jnp.min(frac_level, axis=-1, keepdims=True)  # (ncol, 1) in (0, 1]

    dT_dt = heating_rate * frac                      # (ncol, nlev) [K/s]
    dq_v_dt = drying_rate * frac                     # (ncol, nlev) [kg/kg/s]

    # Diagnostic precipitation P [kg/m²/s] = column latent heating / L_v
    # (equals eq-(8) P before the positivity cap engages).
    precip = jnp.sum(c_pd * dT_dt * dp, axis=-1) / (g * Lv)  # (ncol,) >= 0

    # Convective condensate source for the cloud-water bucket: the column
    # net drying becomes cloud water (microphysics owns surface precip).
    # By construction dq_v_dt <= 0, so -dq_v_dt >= 0 everywhere.
    dq_c_conv_dt = jnp.maximum(-dq_v_dt, 0.0)        # (ncol, nlev) [kg/kg/s]

    # Smooth precipitating indicator in [0, 1].
    convective_mask = jax.nn.sigmoid(
        config.precip_heaviside_sharpness * (BL - config.b_c)
    )                                                # (ncol,)
    del precip  # microphysics owns the surface precip diagnostic

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=BL,                                     # B_L [m/s²] diagnostic
        convective_mask=convective_mask,
        precip_frozen_sfc=jnp.zeros_like(BL),  # L_v-only: liquid at sfc (F29)
    )


def dca_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    dt: float,
    config: DCAConfig = DCAConfig(),
) -> ConvectionOutput:
    """Compute Deep Convective Adjustment tendencies (variant dispatch).

    Routes on ``config.variant``:

    * ``"manabe"`` (default): :func:`_manabe_dca_convection` — the
      Manabe moist convective adjustment (port of GFDL ``moist_conv``).
    * ``"ahmed_neelin"``: :func:`ahmed_neelin_dca` — the
      Ahmed-Neelin-Adames (2020) B_L precipitation-buoyancy closure,
      using ``config.ahmed_neelin``.

    Dispatch is on a static Python string (the config ``variant`` field),
    so only the selected branch is traced — no ``jnp.where`` over both
    schemes.  Unknown variants raise ``ValueError`` (no silent fallback).

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor specific humidity [kg/kg], shape (ncol, nlev).
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    dt : float
        Model time step [s].
    config : DCAConfig
        Convection configuration (selects the variant).

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    """
    if config.variant == "manabe":
        return _manabe_dca_convection(T, q_v, p_full, p_half, dt, config)
    elif config.variant == "ahmed_neelin":
        return ahmed_neelin_dca(
            T, q_v, p_full, p_half, dt, config.ahmed_neelin,
        )
    else:
        raise ValueError(
            f"Unknown DCAConfig.variant: {config.variant!r}. "
            f"Choose 'manabe' or 'ahmed_neelin'."
        )
