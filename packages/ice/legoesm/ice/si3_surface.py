"""SI3 ice-atmosphere surface statements, NEMO-literal: the one shared implementation.

Transcribes what NEMO 5.0.2 executes in ``ice_sbc_tau`` / ``ice_sbc_flx``:
``blk_ice_1`` (constant coefficients), the stress ``lbc_lnk``, ``ice_alb``
(pond and no-pond arms), ``blk_ice_2`` and ``ice_flx_other``.  Called by the
ORCA2 rung (O7, any category count, ORCA2 halo grid) and by the C1D column
gate (one category, explicit neighbours).  Citations are to
``ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/`` (``-fdefault-real-8 -O3``, no FMA):
every bare literal is binary64 and every operation is rounded in source order
(``nemo_source_round``).  ``exp``/``log``/``log10``/``pow`` follow the
precision policy (glibc arm under ``transcendentals="libm"``).  The
saturation, ``theta_exner``, ``blk_ice_1`` and turbulent statements live in
``legoesm.thermo`` / ``legoesm.core.bulk_flux`` (the core package cannot
import ice) and are called from here.

Exact-input contract: every physical array is required (``None`` raises) and
every deck scalar comes from :class:`SI3SurfaceConfig`, which has no defaults
and is built only from NEMO namelists.

Non-differentiable statements (``IF``/``MIN``/``MAX`` switches, square roots
at zero, ``pow`` at zero) keep NEMO's forward bits; only their gradients are
guarded.  ``h_s`` zero tests read the IEEE bit pattern (NEMO's scalar
comparison keeps a positive subnormal nonzero); XLA CPU ``jit`` still flushes
that operand, the D5 platform floor (DECISION 158).
"""

from __future__ import annotations

from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
from legoesm import constants
from legoesm.core import transcendentals
from legoesm.core.bulk_flux import (
    NemoSI3BlkIce1,
    nemo_si3_blk_ice_1_cst,
    nemo_si3_ice_turbulent,
    sqrt_zero_subgradient,
)
from legoesm.core.nemo_namelist import overlay, parse_namelist
from legoesm.core.source_rounding import nemo_source_round as sr
from legoesm.ice.constants_config import NEMO_SI3_CONSTANTS_CONFIG
from legoesm.ice.dynamics import SI3ORCA2FoldProgram
from legoesm.ice.si3_orca2_state import (
    SI3ORCA2StateConfig,
    si3_lbc_lnk_orca2,
    si3_var_snwfra_3d,
)

__param_spec__ = {
    "SI3SurfaceConfig": {
        "scheme_key": "ice.si3_surface",
        "excluded": {
            name: "oracle identity: NEMO namelist value"
            for name in (
                "rn_Cd_ia", "rn_Ce_ia", "rn_Ch_ia", "rn_pfac", "rn_efac",
                "rn_snwblow", "rn_alb_sdry", "rn_alb_smlt", "rn_alb_idry",
                "rn_alb_imlt", "rn_alb_dpnd", "rn_alb_lpnd", "rn_alb_hpiv",
                "rn_Cd_io", "rDt_ice",
            )
        },
        "params": {},
    },
}

# --- NEMO source literals (ORCA2_SAS_ICE_O2_REF ppsrc; binary64 under -fdefault-real-8) ---
_THIN_ICE_ALBEDO = 0.18          # icealb.f90:219,221
_ALB_THIN_BREAK_M = 0.05         # icealb.f90:184-185,218,220
_ALB_SNOW_DECAY_DRY_M = 0.02     # icealb.f90:186
_ALB_SNOW_DECAY_MELT_M = 0.03    # icealb.f90:187
_POND_DEPTH_SCALE_M = 0.05       # icealb.f90:183
_POND_LID_ALBEDO_M = 0.02        # icealb.f90:231
_ALB_CLOUD_QUAD = 0.1010         # icealb.f90:240
_ALB_CLOUD_LINEAR = 0.1933       # icealb.f90:241
_ALB_CLOUD_OFFSET = 0.0148       # icealb.f90:241
_SW_TRANS_CLEAR = 0.18           # sbcblk.f90:1320
_SW_TRANS_CLOUDY = 0.35          # sbcblk.f90:1320
_SW_THIN_ICE_M = 0.1             # sbcblk.f90:1325
_SW_THIN_ICE_INV_M = 10.0        # sbcblk.f90:1326
_EPSI10 = 1.0e-10                # par_ice.f90:135
_ZFRIC_UMIN = 0.0                # icesbc.f90:379
_ZCH = 0.0057                    # icesbc.f90:380
_SUPERCOOL_VOLUME_M = 20.0       # icesbc.f90:440
_LANDFAST_SPEED = 5.0e-3         # icesbc.f90:472
_AMAX_MARGIN = 0.001             # icesbc.f90:472
_THICK_ICE_M = 3.0               # icesbc.f90:472,476
# Compile-time intrinsic fold (gfortran/MPFR, correctly rounded; glibc agrees):
_LOG_ALB_THIN_BREAK = float.fromhex("-0x1.7f7427b73e391p+1")  # LOG(0.05_wp), icealb.f90:184

# --- NEMO PARAMETER constants (sbc_phy.f90:42-90; phycst.f90; eosbn2.f90:1944-1945) ---
_RT0 = constants.T_freeze                     # phycst.f90:50 rt0
_ALBO = constants.albedo_ocean_nemo           # sbc_phy.f90:49 albo
_EMISS_I = constants.emissivity_ice_nemo      # sbc_phy.f90:67
_STEFAN = constants.sigma_sb_nemo             # phycst.f90:66
_ICE = NEMO_SI3_CONSTANTS_CONFIG
_RALB_OCE = _ICE.lead_albedo                  # phycst.f90:77
_RHO0 = constants.rho_ocean_nemo              # eosbn2.f90:1944
_RCP = _ICE.c_ocean                           # eosbn2.f90:1945
_R1_RHO0 = 1.0 / _RHO0                        # eosbn2.f90:2378


class SI3SurfaceConfig(NamedTuple):
    """Deck scalars for the SI3 surface statements; no defaults."""

    rn_Cd_ia: float
    rn_Ce_ia: float
    rn_Ch_ia: float
    rn_pfac: float
    rn_efac: float
    rn_snwblow: float
    nn_snwfra: int
    rn_alb_sdry: float
    rn_alb_smlt: float
    rn_alb_idry: float
    rn_alb_imlt: float
    rn_alb_dpnd: float
    rn_alb_lpnd: float
    rn_alb_hpiv: float
    ld_pnd_alb: bool
    nn_qtrice: int
    ln_icedyn: bool
    rn_Cd_io: float
    rDt_ice: float
    nn_hls: int
    ln_blk: bool
    ln_mixcpl: bool
    ln_Cx_ice_cst: bool
    ln_Cx_ice_frm: bool
    ln_Cx_ice_AN05: bool
    ln_Cx_ice_LU12: bool
    ln_Cx_ice_LG15: bool
    nn_flxdist: int
    ln_cndflx: bool
    ln_leadhfx: bool
    ln_icedO: bool
    ln_icedH: bool

    @classmethod
    def from_resolved_namelists(cls, dyn: Path, ice: Path) -> "SI3SurfaceConfig":
        """From a run's ``output.namelist.dyn`` / ``output.namelist.ice`` (what NEMO read)."""
        return cls._from_values({**parse_namelist(dyn, "oce"), **parse_namelist(ice, "ice")})

    @classmethod
    def from_input_namelists(
        cls, namelist_ref: Path, namelist_cfg: Path, namelist_ice_ref: Path, namelist_ice_cfg: Path,
    ) -> "SI3SurfaceConfig":
        """From a deck's input files, merged as NEMO reads them: each ``_cfg`` key replaces its ``_ref`` key."""
        return cls._from_values(overlay(
            parse_namelist(namelist_ref, "oce"), parse_namelist(namelist_cfg, "oce"),
            parse_namelist(namelist_ice_ref, "ice"), parse_namelist(namelist_ice_cfg, "ice"),
        ))

    @classmethod
    def _from_values(cls, values: dict) -> "SI3SurfaceConfig":
        def get(ns, group, key, kind):
            if (ns, group, key) not in values:
                raise ValueError(f"namelist lacks &{group} {key}; no default")
            tokens = values[(ns, group, key)]
            if len(tokens) != 1 or not isinstance(tokens[0], (bool, float)):
                raise ValueError(f"&{group} {key} is not one scalar: {tokens!r}")
            value = tokens[0]
            if kind is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"&{group} {key} is not a logical: {value!r}")
                return value
            if isinstance(value, bool):
                raise ValueError(f"&{group} {key} is a logical, expected {kind.__name__}")
            if kind is int:
                if value != int(value):
                    raise ValueError(f"&{group} {key} is not an integer: {value!r}")
                return int(value)
            return value

        dyn = lambda g, k, t=float: get("oce", g, k, t)  # noqa: E731
        ice = lambda g, k, t=float: get("ice", g, k, t)  # noqa: E731
        # icethd_pnd.f90:1480-1481: no pond scheme forces ln_pnd_alb off after the echo.
        ld_pnd_alb = ice("namthd_pnd", "ln_pnd_alb", bool) and ice("namthd_pnd", "ln_pnd", bool)
        return cls(
            rn_Cd_ia=dyn("namsbc_blk", "rn_cd_ia"),
            rn_Ce_ia=dyn("namsbc_blk", "rn_ce_ia"),
            rn_Ch_ia=dyn("namsbc_blk", "rn_ch_ia"),
            rn_pfac=dyn("namsbc_blk", "rn_pfac"),
            rn_efac=dyn("namsbc_blk", "rn_efac"),
            rn_snwblow=ice("namsbc", "rn_snwblow"),
            nn_snwfra=ice("namsbc", "nn_snwfra", int),
            rn_alb_sdry=ice("namalb", "rn_alb_sdry"),
            rn_alb_smlt=ice("namalb", "rn_alb_smlt"),
            rn_alb_idry=ice("namalb", "rn_alb_idry"),
            rn_alb_imlt=ice("namalb", "rn_alb_imlt"),
            rn_alb_dpnd=ice("namalb", "rn_alb_dpnd"),
            rn_alb_lpnd=ice("namalb", "rn_alb_lpnd"),
            rn_alb_hpiv=ice("namalb", "rn_alb_hpiv"),
            ld_pnd_alb=ld_pnd_alb,
            nn_qtrice=ice("namsbc", "nn_qtrice", int),
            ln_icedyn=ice("nampar", "ln_icedyn", bool),
            rn_Cd_io=ice("namsbc", "rn_cd_io"),
            # icestp.f90:418: rDt_ice = REAL(nn_fsbc) * rn_Dt
            rDt_ice=float(dyn("namsbc", "nn_fsbc", int)) * dyn("namdom", "rn_dt"),
            nn_hls=dyn("nammpp", "nn_hls", int),
            ln_blk=dyn("namsbc", "ln_blk", bool),
            ln_mixcpl=dyn("namsbc", "ln_mixcpl", bool),
            ln_Cx_ice_cst=dyn("namsbc_blk", "ln_cx_ice_cst", bool),
            ln_Cx_ice_frm=dyn("namsbc_blk", "ln_cx_ice_frm", bool),
            ln_Cx_ice_AN05=dyn("namsbc_blk", "ln_cx_ice_an05", bool),
            ln_Cx_ice_LU12=dyn("namsbc_blk", "ln_cx_ice_lu12", bool),
            ln_Cx_ice_LG15=dyn("namsbc_blk", "ln_cx_ice_lg15", bool),
            nn_flxdist=ice("namsbc", "nn_flxdist", int),
            ln_cndflx=ice("namsbc", "ln_cndflx", bool),
            ln_leadhfx=ice("namthd", "ln_leadhfx", bool),
            ln_icedO=ice("namthd", "ln_icedo", bool),
            ln_icedH=ice("namthd", "ln_icedh", bool),
        )


def _validate(config: SI3SurfaceConfig) -> None:
    if not isinstance(config, SI3SurfaceConfig):
        raise TypeError("SI3 surface requires an explicit SI3SurfaceConfig")
    refused = {
        "ln_blk=F (only the bulk formulation is transcribed)": not config.ln_blk,
        "ln_mixcpl=T": config.ln_mixcpl,
        "non-constant ice-air coefficients": (
            not config.ln_Cx_ice_cst or config.ln_Cx_ice_frm or config.ln_Cx_ice_AN05
            or config.ln_Cx_ice_LU12 or config.ln_Cx_ice_LG15
        ),
        "nn_flxdist != -1": config.nn_flxdist != -1,
        "ln_cndflx=T": config.ln_cndflx,
        "nn_qtrice != 0": config.nn_qtrice != 0,
        "ln_leadhfx/ln_icedO/ln_icedH=F": not (config.ln_leadhfx and config.ln_icedO and config.ln_icedH),
    }
    bad = [name for name, hit in refused.items() if hit]
    if bad:
        raise ValueError(f"SI3 surface arm not transcribed: {bad}")


def _req(name: str, value) -> jax.Array:
    if value is None:
        raise ValueError(f"SI3 surface requires the NEMO array {name!r}; no default")
    return jnp.asarray(value, dtype=jnp.float64)


def _fmin(a, b):
    """Fortran ``MIN(a, b)`` as compiled: ``b`` only when strictly smaller."""
    return jnp.where(b < a, b, a)


def _fmax(a, b):
    """Fortran ``MAX(a, b)`` as compiled: ``b`` only when strictly larger."""
    return jnp.where(b > a, b, a)


def _sum0(terms):
    """Fortran ``SUM(x(:))`` over the category axis: ``+0`` then each term in order."""
    # Materialized seed: XLA otherwise folds 0 + x to x, which keeps a -0 term's sign.
    acc = sr(jnp.zeros(terms.shape[:-1], dtype=terms.dtype))
    for jl in range(terms.shape[-1]):
        acc = sr(acc + terms[..., jl])
    return acc


def _ieee_is_zero(value):
    """``value == 0._wp`` on the bit pattern (a positive subnormal stays nonzero)."""
    bits = jax.lax.bitcast_convert_type(value, jnp.uint64)
    return (bits << jnp.uint64(1)) == jnp.uint64(0)


def _ieee_le_zero(value):
    """``value <= 0._wp`` on the bit pattern: either zero, or sign bit set and not NaN."""
    return _ieee_is_zero(value) | (jnp.signbit(value) & ~jnp.isnan(value))


def si3_blk_ice_1(*, wndi, wndj, theta_air_zt, q_air_zt, slp, tm_su, rhoa, config) -> NemoSI3BlkIce1:
    """``blk_ice_1`` constant-coefficient arm (``sbcblk.f90:1081-1098,1136-1139``), ``zsipt`` included."""
    _validate(config)
    return nemo_si3_blk_ice_1_cst(
        _req("wndi", wndi), _req("wndj", wndj), _req("theta_air_zt", theta_air_zt),
        _req("q_air_zt", q_air_zt), _req("slp", slp), _req("tm_su", tm_su), _req("rhoa", rhoa),
        config.rn_Cd_ia, config.rn_Ch_ia, config.rn_Ce_ia,
    )


def si3_ice_sbc_tau_lbc(utau_interior, vtau_interior, fold: SI3ORCA2FoldProgram):
    """``CALL lbc_lnk('icesbc', utau_ice, 'T', -1., vtau_ice, 'T', -1.)`` (``icesbc.f90:167``).

    The exchange overwrites every halo cell (cyclic seam, fold, closed south),
    so the halo content before the call does not reach the result.
    """
    if not isinstance(fold, SI3ORCA2FoldProgram):
        raise TypeError("SI3 surface requires an explicit SI3ORCA2FoldProgram")
    h = fold.nn_hls
    out = []
    for name, value in (("utau_ice", utau_interior), ("vtau_ice", vtau_interior)):
        interior = _req(name, value)
        full = jnp.pad(interior, ((h, h), (h, h)))
        out.append(si3_lbc_lnk_orca2(full, "T", -1.0, nn_hls=h, ldfull=fold.nfd_full_when_absent))
    return tuple(out)


def si3_ice_alb(*, t_su, h_i, h_s, a_ip_eff, h_ip, h_il, cloud_fra, smask0, config):
    """``ice_alb`` (``icealb.f90:183-244``); pond arm when ``ld_pnd_alb``, else the no-pond arm.

    Arrays carry the category axis last: ``(..., jpl)``; ``cloud_fra``/``smask0`` are ``(...)``.
    """
    _validate(config)
    pt_su, ph_ice, ph_snw = _req("t_su", t_su), _req("h_i", h_i), _req("h_s", h_s)
    pafrac_pnd, ph_pnd, ph_lid = _req("a_ip_eff", a_ip_eff), _req("h_ip", h_ip), _req("h_il", h_il)
    cloud = _req("cloud_fra", cloud_fra)[..., None]
    mask = _req("smask0", smask0)[..., None]
    z1_href_pnd = 1.0 / _POND_DEPTH_SCALE_M                                         # :183
    zhpiv = sr(transcendentals.log(jnp.asarray(config.rn_alb_hpiv)))               # :189
    z1_c1 = sr(1.0 / sr(zhpiv - _LOG_ALB_THIN_BREAK))                              # :184
    z1_c2 = 1.0 / _ALB_THIN_BREAK_M                                                 # :185
    z1_c3 = 1.0 / _ALB_SNOW_DECAY_DRY_M                                             # :186
    z1_c4 = 1.0 / _ALB_SNOW_DECAY_MELT_M                                            # :187
    snw3 = ph_snw.reshape((-1,) + ph_snw.shape[-2:]) if ph_snw.ndim != 3 else ph_snw
    za_s_fra = si3_var_snwfra_3d(                                                   # :191
        snw3, SI3ORCA2StateConfig.for_orca2_sas_ice(nn_snwfra=config.nn_snwfra, nn_hls=0)
    ).reshape(ph_snw.shape)
    if config.ld_pnd_alb:
        zafrac_pnd = _fmin(pafrac_pnd, sr(1.0 - za_s_fra))                          # :201
        zalb_ice = jnp.full_like(ph_ice, config.rn_alb_idry)                        # :212
    else:
        zafrac_pnd = jnp.zeros_like(ph_ice)                                         # :203
        melt = _ieee_is_zero(ph_snw) & (pt_su >= _RT0)                              # :214
        zalb_ice = jnp.where(melt, config.rn_alb_imlt, config.rn_alb_idry)          # :214-215
    zafrac_ice = _fmax(0.0, sr(sr(1.0 - zafrac_pnd) - za_s_fra))                    # :205
    mid = (_ALB_THIN_BREAK_M < ph_ice) & (ph_ice <= config.rn_alb_hpiv)            # :218
    log_h = sr(transcendentals.log(jnp.where(mid, ph_ice, 1.0)))  # gradient-only safe argument
    alb_mid = sr(zalb_ice + sr(sr(sr(_THIN_ICE_ALBEDO - zalb_ice) * z1_c1) * sr(zhpiv - log_h)))  # :219
    alb_thin = sr(_RALB_OCE + sr(sr((_THIN_ICE_ALBEDO - _RALB_OCE) * z1_c2) * ph_ice))           # :221
    zalb_ice = jnp.where(mid, alb_mid, jnp.where(ph_ice <= _ALB_THIN_BREAK_M, alb_thin, zalb_ice))
    snw_dry = sr(config.rn_alb_sdry - sr(sr(config.rn_alb_sdry - zalb_ice)
                 * sr(transcendentals.exp(sr(-sr(ph_snw * z1_c3))))))                          # :226
    snw_melt = sr(config.rn_alb_smlt - sr(sr(config.rn_alb_smlt - zalb_ice)
                  * sr(transcendentals.exp(sr(-sr(ph_snw * z1_c4))))))                         # :228
    zalb_snw = jnp.where(pt_su < _RT0, snw_dry, snw_melt)                                      # :225
    pnd_decay = sr(transcendentals.exp(sr(-sr(ph_pnd * z1_href_pnd))))
    pnd_dark = sr(config.rn_alb_dpnd - sr(sr(config.rn_alb_dpnd - zalb_ice) * pnd_decay))       # :232
    pnd_lid = sr(config.rn_alb_lpnd - sr(sr(config.rn_alb_lpnd - zalb_ice) * pnd_decay))        # :234
    zalb_pnd = jnp.where(ph_lid <= _POND_LID_ALBEDO_M, pnd_dark, pnd_lid)                      # :231
    zalb_os = sr(sr(sr(sr(za_s_fra * zalb_snw) + sr(zafrac_pnd * zalb_pnd))
                    + sr(zafrac_ice * zalb_ice)) * mask)                                       # :238
    poly = sr(-sr(sr(_ALB_CLOUD_QUAD * zalb_os) * zalb_os))
    poly = sr(sr(poly + sr(_ALB_CLOUD_LINEAR * zalb_os)) - _ALB_CLOUD_OFFSET)
    zalb_cs = sr(zalb_os - sr(poly * mask))                                                    # :240-241
    return sr(sr(sr(1.0 - cloud) * zalb_cs) + sr(cloud * zalb_os))                              # :244


class SI3BlkIce2(NamedTuple):
    qsr_ice: jax.Array
    qns_ice: jax.Array
    dqns_ice: jax.Array
    qla_ice: jax.Array
    dqla_ice: jax.Array
    evap_ice: jax.Array
    devap_ice: jax.Array
    tprecip: jax.Array
    sprecip: jax.Array
    emp_oce: jax.Array
    emp_ice: jax.Array
    emp_tot: jax.Array
    qemp_oce: jax.Array
    qemp_ice: jax.Array
    qns_tot: jax.Array
    qsr_tot: jax.Array
    qprec_ice: jax.Array
    qevap_ice: jax.Array
    qtr_ice_top: jax.Array
    # Statement intermediates, exposed for the oracle (sbcblk.f90:1214-1216,1224-1225,1235,1237,1250,1251,1278,1286)
    zcptrain: jax.Array
    zcptsnw: jax.Array
    zcptn: jax.Array
    zsq: jax.Array
    zsipt: jax.Array
    z_qlw: jax.Array
    z_dqlw: jax.Array
    z_qsb: jax.Array
    z_dqsb: jax.Array
    zevap: jax.Array
    zsnw: jax.Array


def si3_blk_ice_2(
    *, t_su, h_s, h_i, alb_ice, theta_air_zt, q_air_zt, slp, qlw, precip, snow, qsr,
    rhoa, wndm_ice, Ch_ice, Ce_ice, theta_zu_i, q_zu_i, smask0, sst_m, emp,
    at_i_b, a_i_b, qns_oce, qsr_oce, cloud_fra, config,
) -> SI3BlkIce2:
    """``blk_ice_2`` (``sbcblk.f90:1210-1331``) with ``ice_var_snwblow`` (``icevar.f90:1636``).

    Category arrays are ``(..., jpl)``; the rest ``(...)``.  ``q_air_zt`` is an
    argument of the call but is read only through ``q_zu_i`` on this arm.
    """
    _validate(config)
    zst, phs, phi, palb = _req("t_su", t_su), _req("h_s", h_s), _req("h_i", h_i), _req("alb_ice", alb_ice)
    ptair, pslp = _req("theta_air_zt", theta_air_zt), _req("slp", slp)
    _req("q_air_zt", q_air_zt)
    pdqlw, pprec, psnow = _req("qlw", qlw), _req("precip", precip), _req("snow", snow)
    qsr2, rho, wndm = _req("qsr", qsr), _req("rhoa", rhoa), _req("wndm_ice", wndm_ice)
    ch, ce = _req("Ch_ice", Ch_ice), _req("Ce_ice", Ce_ice)
    th_zu, q_zu = _req("theta_zu_i", theta_zu_i), _req("q_zu_i", q_zu_i)
    mask, sst, emp2 = _req("smask0", smask0), _req("sst_m", sst_m), _req("emp", emp)
    atb, aib = _req("at_i_b", at_i_b), _req("a_i_b", a_i_b)
    qns_o, qsr_o, cloud = _req("qns_oce", qns_oce), _req("qsr_oce", qsr_oce), _req("cloud_fra", cloud_fra)
    rcpi, rlfus, rhos = _ICE.c_ice, _ICE.latent_fusion, _ICE.rho_snow

    zcoef_dqlw = (4.0 * _EMISS_I) * _STEFAN                                         # :1210
    zztmp = sr(1.0 / sr(1.0 - _ALBO))                                               # :1211
    zcptrain = sr(sr(sr(ptair - _RT0) * _RCP) * mask)                               # :1214
    zcptsnw = sr(sr(sr(_fmin(ptair, _RT0) - _RT0) * rcpi) * mask)                   # :1215
    zcptn = sr(sr(sst * _RCP) * mask)                                               # :1216

    def x(a):  # 2D field read inside the category loop
        return a[..., None]

    turb = nemo_si3_ice_turbulent(zst, x(pslp), x(rho), x(wndm), x(ch), x(ce), x(th_zu), x(q_zu))  # :1224-1256
    qsr_ice = sr(sr(zztmp * sr(1.0 - palb)) * x(qsr2))                              # :1231
    zst3 = sr(sr(zst * zst) * zst)                                                  # :1234
    z_qlw = sr(sr(_EMISS_I * sr(x(pdqlw) - sr(sr(_STEFAN * zst) * zst3))) * x(mask))  # :1235
    z_dqlw = sr(zcoef_dqlw * zst3)                                                  # :1237
    qla_ice, dqla_ice = turb.qla_ice, turb.dqla_ice
    qns_ice = sr(sr(z_qlw - turb.z_qsb) - qla_ice)                                  # :1263
    dqns_ice = -sr(sr(z_dqlw + turb.z_dqsb) + dqla_ice)                             # :1265

    z1_rlsub = 1.0 / constants.L_sub_nemo                                           # :1271
    tprecip = sr(sr(pprec * config.rn_pfac) * mask)                                 # :1274
    sprecip = sr(sr(psnow * config.rn_pfac) * mask)                                 # :1275
    zevap = sr(emp2 + tprecip)                                                      # :1278
    evap_ice = sr(sr(config.rn_efac * qla_ice) * z1_rlsub)                          # :1280
    devap_ice = sr(sr(config.rn_efac * dqla_ice) * z1_rlsub)                        # :1281
    lead = sr(1.0 - atb)
    zsnw = sr(1.0 - sr(transcendentals.pow(lead, config.rn_snwblow)))               # :1286, icevar.f90:1636
    rain = sr(tprecip - sprecip)
    emp_oce = sr(sr(sr(lead * zevap) - rain) - sr(sprecip * sr(1.0 - zsnw)))        # :1289
    emp_ice = sr(_sum0(sr(aib * evap_ice)) - sr(sprecip * zsnw))                    # :1290
    emp_tot = sr(emp_oce + emp_ice)                                                 # :1291
    snow_heat = sr(zcptsnw - rlfus)
    qemp_oce = sr(sr(-sr(sr(lead * zevap) * zcptn) + sr(rain * zcptrain))
                  + sr(sr(sprecip * sr(1.0 - zsnw)) * snow_heat))                   # :1294-1296
    qemp_ice = sr(sr(sprecip * zsnw) * snow_heat)                                   # :1297
    qns_tot = sr(sr(sr(sr(lead * qns_o) + _sum0(sr(aib * qns_ice))) + qemp_ice) + qemp_oce)  # :1300-1301
    qsr_tot = sr(sr(lead * qsr_o) + _sum0(sr(aib * qsr_ice)))                       # :1302
    qprec_ice = sr(rhos * snow_heat)                                                # :1305
    qevap_ice = jnp.zeros_like(qsr_ice)                                             # :1309

    ztri = x(sr(sr(_SW_TRANS_CLEAR * sr(1.0 - cloud)) + sr(_SW_TRANS_CLOUDY * cloud)))  # :1320
    thin = sr(ztri + sr(sr(1.0 - ztri) * sr(1.0 - sr(phi * _SW_THIN_ICE_INV_M))))   # :1326
    ztmp = jnp.where(phi < _SW_THIN_ICE_M, thin, ztri)                              # :1325
    qtr_ice_top = jnp.where(_ieee_le_zero(phs), sr(qsr_ice * ztmp), 0.0)            # :1323-1329
    return SI3BlkIce2(
        qsr_ice=qsr_ice, qns_ice=qns_ice, dqns_ice=dqns_ice, qla_ice=qla_ice, dqla_ice=dqla_ice,
        evap_ice=evap_ice, devap_ice=devap_ice, tprecip=tprecip, sprecip=sprecip,
        emp_oce=emp_oce, emp_ice=emp_ice, emp_tot=emp_tot, qemp_oce=qemp_oce, qemp_ice=qemp_ice,
        qns_tot=qns_tot, qsr_tot=qsr_tot, qprec_ice=qprec_ice, qevap_ice=qevap_ice,
        qtr_ice_top=qtr_ice_top, zcptrain=zcptrain, zcptsnw=zcptsnw, zcptn=zcptn, zsq=turb.zsq,
        zsipt=turb.zsipt, z_qlw=z_qlw, z_dqlw=z_dqlw, z_qsb=turb.z_qsb, z_dqsb=turb.z_dqsb,
        zevap=zevap, zsnw=zsnw,
    )


class SI3FlxOther(NamedTuple):
    qsb_ice_bot: jax.Array
    fhld: jax.Array
    qlead: jax.Array


def si3_ice_flx_other(
    *, u_ice, u_ice_w, v_ice, v_ice_s, ssu_m, ssu_m_w, ssv_m, ssv_m_s, utau, vtau, smask0,
    at_i, at_i_b, vt_i, qsr_oce, qns_oce, qemp_oce, frq_m, e3t_m, sst_m, t_bo, rn_amax_2d, config,
) -> SI3FlxOther:
    """``ice_flx_other`` (``icesbc.f90:385-482``), ``ln_leadhfx/ln_icedO/ln_icedH = T``.

    Pointwise: the ``(ji-1,jj)``/``(ji,jj-1)`` neighbours are explicit
    arguments (``_w``/``_s``); see :func:`orca2_flx_other_neighbours`.
    Velocities are read only when ``ln_icedyn``; ``utau``/``vtau`` only when not.
    """
    _validate(config)
    mask, at, atb, vt = _req("smask0", smask0), _req("at_i", at_i), _req("at_i_b", at_i_b), _req("vt_i", vt_i)
    qsr_o, qns_o, qemp_o = _req("qsr_oce", qsr_oce), _req("qns_oce", qns_oce), _req("qemp_oce", qemp_oce)
    frq, e3t, sst, t_bo2, amax = (_req("frq_m", frq_m), _req("e3t_m", e3t_m), _req("sst_m", sst_m),
                                  _req("t_bo", t_bo), _req("rn_amax_2d", rn_amax_2d))
    if config.ln_icedyn:                                                            # :385-400
        ui, uiw, vi, vis = _req("u_ice", u_ice), _req("u_ice_w", u_ice_w), _req("v_ice", v_ice), _req("v_ice_s", v_ice_s)
        uo, uow, vo, vos = _req("ssu_m", ssu_m), _req("ssu_m_w", ssu_m_w), _req("ssv_m", ssv_m), _req("ssv_m_s", ssv_m_s)
        zu_io, zu_iom1 = sr(ui - uo), sr(uiw - uow)                                 # :387-388
        zv_io, zv_iom1 = sr(vi - vo), sr(vis - vos)                                 # :389-390
        zdrag = config.rn_Cd_io                                                     # :394
        uu = sr(sr(zu_io * zu_io) + sr(zu_iom1 * zu_iom1))
        vv = sr(sr(zv_io * zv_io) + sr(zv_iom1 * zv_iom1))
        zfric = sr(sr(zdrag * sr(0.5 * sr(uu + vv))) * mask)                        # :396-397
        us, vs = sr(uiw + ui), sr(vis + vi)
        zvel = sr(0.5 * sqrt_zero_subgradient(sr(sr(us * us) + sr(vs * vs))))       # :398-399
    else:                                                                           # :401-406
        tu, tv = _req("utau", utau), _req("vtau", vtau)
        zfric = sr(sr(_R1_RHO0 * sqrt_zero_subgradient(sr(sr(tu * tu) + sr(tv * tv)))) * mask)
        zvel = jnp.zeros_like(mask)

    rdt = config.rDt_ice
    r1_dt = 1.0 / rdt                                                               # icestp.f90:419
    zswitch = jnp.where(at >= _EPSI10, mask, 0.0)                                   # :412-414
    lead = sr(1.0 - atb)
    zqld = sr(sr(mask * rdt) * sr(sr(sr(sr(lead * qsr_o) * frq) + sr(lead * qns_o)) + qemp_o))  # :417-419
    sst_k = sr(sst + _RT0)
    zqfr = sr(sr(sr(sr(_RHO0 * _RCP) * e3t) * sr(t_bo2 - sst_k)) * mask)            # :423
    zqfr_neg = _fmin(zqfr, 0.0)                                                     # :424
    zqfr_pos = _fmax(zqfr, 0.0)                                                     # :425
    zfric_u = _fmax(sqrt_zero_subgradient(zfric), _ZFRIC_UMIN)                      # :429
    qsb = sr(sr(sr(sr(sr(zswitch * _RHO0) * _RCP) * _ZCH) * zfric_u) * sr(sst_k - t_bo2))  # :430
    cap = -sr(sr(zqfr_neg * r1_dt) / _fmax(at, _EPSI10))
    qsb = sr(zswitch * jnp.where(cap <= qsb, cap, qsb))  # MINSD keeps RHS on tie  # :436
    supercool = (sr(t_bo2 - sst_k) > 0.0) & (vt >= _SUPERCOOL_VOLUME_M)             # :440
    zqfr = jnp.where(supercool, 0.0, zqfr)
    zqfr_pos = jnp.where(supercool, 0.0, zqfr_pos)
    qsb = jnp.where(supercool, 0.0, qsb)
    excess = sr(zqld - zqfr)
    cooling = excess < 0.0                                                          # :450
    fhld_warm = sr(zswitch * _fmax(0.0, sr(sr(sr(zqld - zqfr_pos) * r1_dt) / _fmax(at, _EPSI10))))  # :463
    fhld = jnp.where(cooling, 0.0, fhld_warm)                                       # :451,463
    qlead = jnp.where(cooling, _fmin(0.0, excess), 0.0)                             # :457,464
    landfast = (zvel <= _LANDFAST_SPEED) & (at >= sr(amax - _AMAX_MARGIN)) & (vt >= _THICK_ICE_M)  # :472
    full = (at >= (1.0 - _EPSI10)) & (vt >= _THICK_ICE_M)                           # :476
    qlead = jnp.where(landfast | full, 0.0, qlead)
    return SI3FlxOther(qsb_ice_bot=qsb, fhld=fhld, qlead=qlead)


def orca2_flx_other_neighbours(*, u_ice, v_ice, ssu_m, ssv_m, nn_hls) -> dict:
    """Interior and ``(ji-1,jj)``/``(ji,jj-1)`` slices of full-halo ORCA2 velocity arrays (``icesbc.f90:387-390``)."""
    ui, vi, uo, vo = _req("u_ice", u_ice), _req("v_ice", v_ice), _req("ssu_m", ssu_m), _req("ssv_m", ssv_m)
    h = nn_hls
    ni, nj = ui.shape[0] - 2 * h, ui.shape[1] - 2 * h

    def c(a):
        return a[h:h + ni, h:h + nj]

    def w(a):
        return a[h - 1:h - 1 + ni, h:h + nj]

    def s(a):
        return a[h:h + ni, h - 1:h - 1 + nj]

    return dict(u_ice=c(ui), u_ice_w=w(ui), v_ice=c(vi), v_ice_s=s(vi),
                ssu_m=c(uo), ssu_m_w=w(uo), ssv_m=c(vo), ssv_m_s=s(vo))


class SI3SurfaceOutputs(NamedTuple):
    blk_ice_1: NemoSI3BlkIce1
    utau_ice: jax.Array   # full halo, after icesbc.f90:167
    vtau_ice: jax.Array
    alb_ice: jax.Array
    blk_ice_2: SI3BlkIce2
    flx_other: SI3FlxOther


def si3_surface_step(
    *, wndi, wndj, theta_air_zt, q_air_zt, slp, tm_su, rhoa, t_su, h_i, h_s, a_ip_eff, h_ip,
    h_il, cloud_fra, smask0, qlw, precip, snow, qsr, sst_m, emp, at_i_b, a_i_b, qns_oce,
    qsr_oce, u_ice, v_ice, ssu_m, ssv_m, utau, vtau, at_i, vt_i, frq_m, e3t_m, t_bo,
    rn_amax_2d, config, fold,
) -> SI3SurfaceOutputs:
    """The four routines on the ORCA2 grid, in NEMO's call order (``icestp.f90:203,250``; ``icesbc.f90:144-167,209-254``).

    Internal products (stress, coefficients, ``wndm_ice``, albedo,
    ``qemp_oce``) feed forward; every other array is the NEMO value at its call.
    ``u_ice``/``v_ice``/``ssu_m``/``ssv_m`` are full-halo; the rest interior.
    """
    if fold.nn_hls != config.nn_hls:
        raise ValueError("fold program and deck disagree on nn_hls")
    b1 = si3_blk_ice_1(wndi=wndi, wndj=wndj, theta_air_zt=theta_air_zt, q_air_zt=q_air_zt,
                       slp=slp, tm_su=tm_su, rhoa=rhoa, config=config)
    utau_ice, vtau_ice = si3_ice_sbc_tau_lbc(b1.utau_ice, b1.vtau_ice, fold)
    alb = si3_ice_alb(t_su=t_su, h_i=h_i, h_s=h_s, a_ip_eff=a_ip_eff, h_ip=h_ip, h_il=h_il,
                      cloud_fra=cloud_fra, smask0=smask0, config=config)
    b2 = si3_blk_ice_2(t_su=t_su, h_s=h_s, h_i=h_i, alb_ice=alb, theta_air_zt=theta_air_zt,
                       q_air_zt=q_air_zt, slp=slp, qlw=qlw, precip=precip, snow=snow, qsr=qsr,
                       rhoa=rhoa, wndm_ice=b1.wndm_ice, Ch_ice=b1.Ch_ice, Ce_ice=b1.Ce_ice,
                       theta_zu_i=b1.theta_zu_i, q_zu_i=b1.q_zu_i, smask0=smask0, sst_m=sst_m,
                       emp=emp, at_i_b=at_i_b, a_i_b=a_i_b, qns_oce=qns_oce, qsr_oce=qsr_oce,
                       cloud_fra=cloud_fra, config=config)
    other = si3_ice_flx_other(**orca2_flx_other_neighbours(u_ice=u_ice, v_ice=v_ice, ssu_m=ssu_m,
                                                            ssv_m=ssv_m, nn_hls=config.nn_hls),
                              utau=utau, vtau=vtau, smask0=smask0, at_i=at_i, at_i_b=at_i_b, vt_i=vt_i,
                              qsr_oce=qsr_oce, qns_oce=qns_oce, qemp_oce=b2.qemp_oce,
                              frq_m=frq_m, e3t_m=e3t_m, sst_m=sst_m, t_bo=t_bo,
                              rn_amax_2d=rn_amax_2d, config=config)
    return SI3SurfaceOutputs(b1, utau_ice, vtau_ice, alb, b2, other)


__all__ = (
    "SI3BlkIce2", "SI3FlxOther", "SI3SurfaceConfig", "SI3SurfaceOutputs",
    "orca2_flx_other_neighbours", "si3_blk_ice_1", "si3_blk_ice_2", "si3_ice_alb",
    "si3_ice_flx_other", "si3_ice_sbc_tau_lbc", "si3_surface_step",
)
