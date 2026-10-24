"""Selectable full ORCA2 SI3 state, category bounds, aggregation, halos, and restart I/O.

Implements the rung-O1 (`DECISION 159`) full NEMO 5.0.2 SI3 state/restart
schema (`jpl = 5` categories, `nlay_i = 10` ice + `nlay_s = 5` snow layers of
energy and salt, age, melt pond and lid volumes, the five Prather moments of
every extensive variable, and the HFN category bounds) while generalizing
lane 3's one-category `3+3` SI3 column state without changing any existing
caller.

NEMO 5.0.2 oracle source citations:
- `src/ICE/icestp.F90:96-394` (`ice_stp`, `ice_init`, `store_fields`)
- `src/ICE/icerst.F90:111-378` (`ice_rst_write`, `ice_rst_read`)
- `src/ICE/icevar.F90:105-428`, `:466-590`, `:1564-1587` (`ice_var_agg`,
  `ice_var_glo2eqv`, `ice_var_salprof`, `ice_var_snwfra_3d`)
- `src/ICE/iceitd.F90:715-780` (`ice_itd_init` HFN category bounds)
- `src/ICE/icedyn_adv_pra.F90:36-46`, `:438-490`, `:1145-1503` (`adv_pra_init`,
  `adv_pra_rst`, Prather moment `lbc_lnk` halo calls)
- `src/ICE/iceupdate.F90:189-194`, `:438-485` (`snwice_mass` update and
  `update_rst`)
- `src/OCE/LBC/lbc_lnk_pt2pt_generic.h90:99-243` and
  `src/OCE/LBC/lbc_nfd_generic.h90:22-191` (ORCA2 cyclic east-west seam and
  `c_NFtype = 'T'` north fold)
"""

from __future__ import annotations

from pathlib import Path
import re
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm import constants as canonical_constants
from legoesm.core import transcendentals
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.constants_config import (
    NEMO_SI3_CONSTANTS_CONFIG,
    IceConstantsConfig,
)
from legoesm.ocean.constants_config import NEMO_CONSTANTS_CONFIG
from legoesm.ocean.eos import nemo_eos_fzp
from legoesm.ocean.fidelity.nemo_recipe import nemo_ddpdd_sum_2d

if TYPE_CHECKING:
    from legoesm.ice.state import SI3ColumnState

Field = jax.Array

_EPSI20 = 1.0e-20
_ZSI0 = 3.5
_ZSI1 = 4.5
_HI_MAX_SENTINEL = 999.99

# Fixed NEMO 5.0.2 SI3 coefficients.  These are oracle-source constants, not
# legoESM tuning parameters: iceitd.F90:721-761, icevar.F90:477-532 and
# icevar.F90:1578-1601 in the shipped source tree.
_HFN_BISECTION_LOWER_M = 0.05
_HFN_ALPHA = 0.05
_SALPROF_CONSTANT_G_PER_KG = 2.30
_SALPROF_EXPONENT_NUMERATOR = 0.407
_SALPROF_EXPONENT_OFFSET = 0.573
_SALPROF_AMPLITUDE = 1.6
_SNOW_FRACTION_EXPONENT = -0.2
_LANE3_REFERENCE_SALINITY = 34.0
_ORCA2_RN_HIMAX_M = 99.0

__param_spec__ = {
    "SI3ORCA2StateConfig": {
        "scheme_key": "ice.si3_orca2_state",
        "excluded": {
            "rn_icesal": "oracle identity: NEMO namelist_ice_ref constant",
            "rn_sal_gd": "oracle identity: NEMO namelist_ice_ref constant",
            "rn_sal_fl": "oracle identity: NEMO namelist_ice_ref constant",
            "rn_simax": "oracle identity: NEMO namelist_ice_ref constant",
            "rn_simin": "oracle identity: NEMO namelist_ice_ref constant",
            "rn_sinew": "oracle identity: NEMO namelist_ice_ref constant",
            "rn_himean": "oracle identity: NEMO namelist_ice_cfg HFN mean thickness",
            "rn_himax": "oracle identity: NEMO namelist_ice_ref maximum ice thickness",
            "rn_apnd_min": "oracle identity: NEMO namelist_ice_cfg pond parameter",
            "rn_apnd_max": "oracle identity: NEMO namelist_ice_cfg pond parameter",
            "rn_pnd_hl_min": "oracle identity: NEMO namelist_ice_ref lid threshold",
            "rn_pnd_hl_max": "oracle identity: NEMO namelist_ice_ref lid threshold",
            "rDt_ice": "oracle identity: NEMO ORCA2_SAS_ICE ice timestep [s]",
        },
        "params": {},
    },
}


class SI3ORCA2StateConfig(NamedTuple):
    """Selectable full SI3 state configuration (`nampar`, `namitd`, `namthd_sal`, `namthd_pnd`).

    Defaults reproduce NEMO 5.0.2 `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg`
    (`cfgs/SHARED/namelist_ice_ref`) with the rung-O1 overlay
    (`ln_icedyn = False`, `ln_icethd = False`, `DECISION 159`).
    """

    jpl: int = 5
    nlay_i: int = 10
    nlay_s: int = 5
    nn_icesal: int = 4
    rn_icesal: float = 4.0
    rn_sal_gd: float = 5.0
    rn_sal_fl: float = 2.0
    rn_simax: float = 35.0
    rn_simin: float = 0.1
    rn_sinew: float = 0.75
    rn_himean: float = 2.0
    rn_himax: float = _ORCA2_RN_HIMAX_M
    opt_hips: int = 0
    ln_pnd: bool = True
    ln_pnd_LEV: bool = True
    ln_pnd_TOPO: bool = False
    ln_pnd_lids: bool = True
    ln_pnd_alb: bool = True
    rn_apnd_min: float = 0.15
    rn_apnd_max: float = 0.80
    rn_pnd_hl_min: float = 0.005
    rn_pnd_hl_max: float = 0.015
    nn_snwfra: int = 2
    ln_icedyn: bool = False
    ln_icethd: bool = False
    ln_dynadv_pra: bool = True
    nn_fsbc: int = 1
    rDt_ice: float = 10800.0
    nn_hls: int = 2
    ln_cpl: bool = False
    ice_constants: IceConstantsConfig = NEMO_SI3_CONSTANTS_CONFIG

    @classmethod
    def for_orca2_sas_ice(cls, **overrides: Any) -> "SI3ORCA2StateConfig":
        """Return the NEMO 5.0.2 `ORCA2_SAS_ICE` rung-O1 state configuration."""
        base = cls()
        return base._replace(**overrides) if overrides else base

    @classmethod
    def for_lane3_c1d(cls, **overrides: Any) -> "SI3ORCA2StateConfig":
        """Return the 1-category `3+3` lane-3 `C1D_OMIP_L3` state configuration."""
        base = cls(
            jpl=1,
            nlay_i=3,
            nlay_s=3,
            nn_icesal=2,
            ln_pnd=False,
            ln_pnd_LEV=False,
            ln_pnd_TOPO=False,
            ln_pnd_lids=False,
            ln_pnd_alb=False,
            rDt_ice=3600.0,
            nn_hls=0,
        )
        return base._replace(**overrides) if overrides else base


class SI3CategoryBounds(NamedTuple):
    """SI3 ITD category thickness bounds (`src/ICE/iceitd.F90:715-780`).

    Attributes:
        hi_max: Category upper bounds `hi_max(0:jpl)` of shape `(jpl + 1,)` [m].
        hi_mean: Category midpoints `hi_mean(1:jpl)` of shape `(jpl,)` [m].
    """

    hi_max: Field
    hi_mean: Field


class SI3PratherMoments(NamedTuple):
    """Five second-order Prather advection moments (`src/ICE/icedyn_adv_pra.F90:36-46`)."""

    sx: Field
    sy: Field
    sxx: Field
    syy: Field
    sxy: Field

    @classmethod
    def zeros(cls, shape: tuple[int, ...], dtype: Any = jnp.float64) -> "SI3PratherMoments":
        z = jnp.zeros(shape, dtype=dtype)
        return cls(sx=z, sy=z, sxx=z, syy=z, sxy=z)


class SI3AllPratherMoments(NamedTuple):
    """All 11 Prather moment families (`55` arrays, `src/ICE/icedyn_adv_pra.F90:36-46`).

    3D `(jpi, jpj, jpl)` families: `ice`, `sn`, `a`, `age`, `sal`, `ap`, `vp`, `vl`.
    4D `(jpi, jpj, nlay_s, jpl)` family: `c0` (snow enthalpy).
    4D `(jpi, jpj, nlay_i, jpl)` families: `e` (ice enthalpy), `si` (ice layer salt).
    """

    ice: SI3PratherMoments
    sn: SI3PratherMoments
    a: SI3PratherMoments
    age: SI3PratherMoments
    sal: SI3PratherMoments
    ap: SI3PratherMoments
    vp: SI3PratherMoments
    vl: SI3PratherMoments
    c0: SI3PratherMoments
    e: SI3PratherMoments
    si: SI3PratherMoments

    @classmethod
    def zeros(
        cls,
        jpi: int,
        jpj: int,
        jpl: int = 5,
        nlay_i: int = 10,
        nlay_s: int = 5,
        dtype: Any = jnp.float64,
    ) -> "SI3AllPratherMoments":
        s3 = (jpi, jpj, jpl)
        s4_s = (jpi, jpj, nlay_s, jpl)
        s4_i = (jpi, jpj, nlay_i, jpl)
        return cls(
            ice=SI3PratherMoments.zeros(s3, dtype=dtype),
            sn=SI3PratherMoments.zeros(s3, dtype=dtype),
            a=SI3PratherMoments.zeros(s3, dtype=dtype),
            age=SI3PratherMoments.zeros(s3, dtype=dtype),
            sal=SI3PratherMoments.zeros(s3, dtype=dtype),
            ap=SI3PratherMoments.zeros(s3, dtype=dtype),
            vp=SI3PratherMoments.zeros(s3, dtype=dtype),
            vl=SI3PratherMoments.zeros(s3, dtype=dtype),
            c0=SI3PratherMoments.zeros(s4_s, dtype=dtype),
            e=SI3PratherMoments.zeros(s4_i, dtype=dtype),
            si=SI3PratherMoments.zeros(s4_i, dtype=dtype),
        )


class SI3EquivalentVars(NamedTuple):
    """Intensive equivalent variables from `ice_var_glo2eqv` (`src/ICE/icevar.F90:255-428`)."""

    h_i: Field
    h_s: Field
    o_i: Field
    h_ip: Field
    h_il: Field
    a_ip_frac: Field
    a_ip_eff: Field
    s_i: Field
    sz_i: Field
    t_i: Field
    t_s: Field
    a_i: Field
    v_s: Field

    @classmethod
    def zeros(
        cls,
        jpi: int,
        jpj: int,
        jpl: int = 5,
        nlay_i: int = 10,
        nlay_s: int = 5,
        dtype: Any = jnp.float64,
    ) -> "SI3EquivalentVars":
        s3 = (jpi, jpj, jpl)
        s4_i = (jpi, jpj, nlay_i, jpl)
        s4_s = (jpi, jpj, nlay_s, jpl)
        z3 = jnp.zeros(s3, dtype=dtype)
        return cls(
            h_i=z3,
            h_s=z3,
            o_i=z3,
            h_ip=z3,
            h_il=z3,
            a_ip_frac=z3,
            a_ip_eff=z3,
            s_i=z3,
            sz_i=jnp.zeros(s4_i, dtype=dtype),
            t_i=jnp.zeros(s4_i, dtype=dtype),
            t_s=jnp.zeros(s4_s, dtype=dtype),
            a_i=z3,
            v_s=z3,
        )


class SI3AggregatedVars(NamedTuple):
    """Category-aggregated variables from `ice_var_agg` (`src/ICE/icevar.F90:105-252`)."""

    vt_i: Field
    vt_s: Field
    at_i: Field
    at_ip: Field
    vt_ip: Field
    vt_il: Field
    ato_i: Field
    et_s: Field
    et_i: Field
    at_ip_eff: Field
    tm_su: Field
    st_i: Field
    hm_i: Field
    hm_s: Field
    tm_si: Field
    om_i: Field
    sm_i: Field
    tm_i: Field
    tm_s: Field
    hm_ip: Field
    hm_il: Field

    @classmethod
    def zeros(cls, jpi: int, jpj: int, dtype: Any = jnp.float64) -> "SI3AggregatedVars":
        z2 = jnp.zeros((jpi, jpj), dtype=dtype)
        return cls(
            vt_i=z2,
            vt_s=z2,
            at_i=z2,
            at_ip=z2,
            vt_ip=z2,
            vt_il=z2,
            ato_i=z2,
            et_s=z2,
            et_i=z2,
            at_ip_eff=z2,
            tm_su=z2,
            st_i=z2,
            hm_i=z2,
            hm_s=z2,
            tm_si=z2,
            om_i=z2,
            sm_i=z2,
            tm_i=z2,
            tm_s=z2,
            hm_ip=z2,
            hm_il=z2,
        )


class SI3State(NamedTuple):
    """Selectable full multi-category, multi-layer SI3 state (`src/ICE/ice.F90`, `icerst.F90`).

    Generalizes `SI3ColumnState` (`jpl = 1, nlay_i = 3, nlay_s = 3`) to the full
    ORCA2 SI3 state (`jpl = 5, nlay_i = 10, nlay_s = 5`) with all five Prather
    moments of every extensive variable, HFN category bounds, intensive
    equivalent variables, and category-aggregated diagnostics.
    """

    v_i: Field
    v_s: Field
    sv_i: Field
    a_i: Field
    t_su: Field
    u_ice: Field
    v_ice: Field
    oa_i: Field
    a_ip: Field
    v_ip: Field
    v_il: Field
    e_s: Field
    e_i: Field
    szv_i: Field
    cnd_ice: Field
    t1_ice: Field
    t_si: Field
    snwice_mass: Field
    snwice_mass_b: Field
    snwice_fmass: Field
    prather: SI3AllPratherMoments
    bounds: SI3CategoryBounds
    equivalent: SI3EquivalentVars
    aggregated: SI3AggregatedVars

    @classmethod
    def zeros(
        cls,
        jpi: int,
        jpj: int,
        config: SI3ORCA2StateConfig | None = None,
        dtype: Any = jnp.float64,
    ) -> "SI3State":
        cfg = config if config is not None else SI3ORCA2StateConfig()
        s2 = (jpi, jpj)
        s3 = (jpi, jpj, cfg.jpl)
        s4_s = (jpi, jpj, cfg.nlay_s, cfg.jpl)
        s4_i = (jpi, jpj, cfg.nlay_i, cfg.jpl)
        z2 = jnp.zeros(s2, dtype=dtype)
        z3 = jnp.zeros(s3, dtype=dtype)
        bounds = si3_hfn_category_bounds(
            jpl=cfg.jpl,
            rn_himean=cfg.rn_himean,
            opt_hips=cfg.opt_hips,
            rn_himax=cfg.rn_himax,
        )
        return cls(
            v_i=z3,
            v_s=z3,
            sv_i=z3,
            a_i=z3,
            t_su=z3,
            u_ice=z2,
            v_ice=z2,
            oa_i=z3,
            a_ip=z3,
            v_ip=z3,
            v_il=z3,
            e_s=jnp.zeros(s4_s, dtype=dtype),
            e_i=jnp.zeros(s4_i, dtype=dtype),
            szv_i=jnp.zeros(s4_i, dtype=dtype),
            cnd_ice=z3,
            t1_ice=z3,
            t_si=z3,
            snwice_mass=z2,
            snwice_mass_b=z2,
            snwice_fmass=z2,
            prather=SI3AllPratherMoments.zeros(
                jpi, jpj, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s, dtype=dtype
            ),
            bounds=bounds,
            equivalent=SI3EquivalentVars.zeros(
                jpi, jpj, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s, dtype=dtype
            ),
            aggregated=SI3AggregatedVars.zeros(jpi, jpj, dtype=dtype),
        )

    @classmethod
    def from_column_state(
        cls,
        col: SI3ColumnState,
        tmask: Field | None = None,
        sss_m: Field | float = _LANE3_REFERENCE_SALINITY,
        config: SI3ORCA2StateConfig | None = None,
    ) -> "SI3State":
        """Promote a lane-3 1-category `SI3ColumnState` into `SI3State` (`jpl = 1`)."""

        def _unwrap(f: Any) -> jax.Array:
            return jnp.asarray(f.data if hasattr(f, "data") else f, dtype=jnp.float64)

        a_i_raw = _unwrap(col.concentration)
        if a_i_raw.ndim == 0:
            a_i_2d = a_i_raw.reshape(1, 1)
        elif a_i_raw.ndim == 1:
            a_i_2d = a_i_raw.reshape(-1, 1)
        else:
            a_i_2d = a_i_raw
        jpi, jpj = a_i_2d.shape
        e_ice_raw = _unwrap(col.e_ice)
        e_snow_raw = _unwrap(col.e_snow)
        nlay_i = int(e_ice_raw.shape[-1])
        nlay_s = int(e_snow_raw.shape[-1])
        cfg = (
            config
            if config is not None
            else SI3ORCA2StateConfig.for_lane3_c1d(nlay_i=nlay_i, nlay_s=nlay_s)
        )
        dtype = jnp.float64

        def _to_3d(x: Any) -> Field:
            return _unwrap(x).reshape(jpi, jpj, 1)

        def _to_4d(x: Any, nlay: int) -> Field:
            return _unwrap(x).reshape(jpi, jpj, nlay, 1)

        a_i = _to_3d(col.concentration)
        h_i = _to_3d(col.h_ice)
        h_s = _to_3d(col.h_snow)
        t_su = _to_3d(col.T_surface)
        s_i = _to_3d(col.S_bulk)
        sz_i = _to_4d(col.S_layers, nlay_i)
        o_i = _to_3d(col.age_volume)

        v_i = nemo_source_round(h_i * a_i)
        v_s = nemo_source_round(h_s * a_i)
        sv_i = nemo_source_round(s_i * v_i)
        oa_i = nemo_source_round(o_i * a_i)
        e_i_vol = _to_4d(col.e_ice, nlay_i)
        e_s_vol = _to_4d(col.e_snow, nlay_s)
        e_i = nemo_source_round(e_i_vol * (v_i[..., None, :] / float(nlay_i)))
        e_s = nemo_source_round(e_s_vol * (v_s[..., None, :] / float(nlay_s)))
        szv_i = nemo_source_round(sz_i * (v_i[..., None, :] / float(nlay_i)))

        z2 = jnp.zeros((jpi, jpj), dtype=dtype)
        z3 = jnp.zeros((jpi, jpj, 1), dtype=dtype)
        interior_3d = _interior_mask_2d(jpi, jpj, cfg.nn_hls)[:, :, None]
        t_si_init = jnp.where(interior_3d, jnp.full_like(z3, cfg.ice_constants.T0), 0.0)
        bounds = si3_hfn_category_bounds(
            jpl=1,
            rn_himean=cfg.rn_himean,
            opt_hips=cfg.opt_hips,
            rn_himax=cfg.rn_himax,
        )
        eqv = si3_var_glo2eqv(
            v_i=v_i,
            v_s=v_s,
            sv_i=sv_i,
            a_i=a_i,
            oa_i=oa_i,
            a_ip=z3,
            v_ip=z3,
            v_il=z3,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            kn=2,
            hi_max_top=bounds.hi_max[-1],
            sss_m=sss_m,
            config=cfg,
        )
        agg = si3_var_agg(
            v_i=v_i,
            v_s=eqv.v_s,
            sv_i=sv_i,
            a_i=eqv.a_i,
            oa_i=oa_i,
            a_ip=z3,
            v_ip=z3,
            v_il=z3,
            t_su=t_su,
            t_si=t_si_init,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            a_ip_eff=eqv.a_ip_eff,
            t_i=eqv.t_i,
            t_s=eqv.t_s,
            kn=2,
            config=cfg,
        )
        tmask_arr = (
            jnp.asarray(tmask, dtype=dtype).reshape(jpi, jpj)
            if tmask is not None
            else jnp.ones((jpi, jpj), dtype=dtype)
        )
        snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
            vt_i=agg.vt_i,
            vt_s=agg.vt_s,
            vt_ip=agg.vt_ip,
            vt_il=agg.vt_il,
            snwice_mass_prev=z2,
            tmask=tmask_arr,
            config=cfg,
            is_init=True,
        )
        return cls(
            v_i=v_i,
            v_s=eqv.v_s,
            sv_i=sv_i,
            a_i=eqv.a_i,
            t_su=t_su,
            u_ice=z2,
            v_ice=z2,
            oa_i=oa_i,
            a_ip=z3,
            v_ip=z3,
            v_il=z3,
            e_s=e_s,
            e_i=e_i,
            szv_i=szv_i,
            cnd_ice=z3,
            # iceistate.F90:134-135 initializes the coupling diagnostic from
            # the first ice layer, not from rt0.
            t1_ice=eqv.t_i[:, :, 0, :],
            t_si=t_si_init,
            snwice_mass=snwice_mass,
            snwice_mass_b=snwice_mass_b,
            snwice_fmass=snwice_fmass,
            prather=SI3AllPratherMoments.zeros(
                jpi, jpj, jpl=1, nlay_i=nlay_i, nlay_s=nlay_s, dtype=dtype
            ),
            bounds=bounds,
            equivalent=eqv,
            aggregated=agg,
        )

    def to_column_state(self, squeeze_1d: bool = False) -> SI3ColumnState:
        """Project a `jpl == 1` `SI3State` back to lane-3 `SI3ColumnState`."""
        from legoesm.ice.state import SI3ColumnState

        if self.a_i.shape[-1] != 1:
            raise ValueError(
                f"to_column_state requires jpl == 1, got shape {self.a_i.shape}"
            )

        def _s3(x: Field) -> Field:
            out = x[..., 0]
            if squeeze_1d and out.shape == (1, 1):
                return out.reshape((1,))
            if squeeze_1d and out.shape[1] == 1:
                return out[:, 0]
            return out

        def _s4(x: Field) -> Field:
            out = x[..., 0]
            if squeeze_1d and out.shape[:2] == (1, 1):
                return out.reshape((1, out.shape[2]))
            if squeeze_1d and out.shape[1] == 1:
                return out[:, 0, :]
            return out

        nlay_i = self.e_i.shape[2]
        nlay_s = self.e_s.shape[2]
        v_i_pos = self.v_i[..., None, :] > _EPSI20
        v_s_pos = self.v_s[..., None, :] > _EPSI20
        v_i_safe = jnp.where(v_i_pos, self.v_i[..., None, :], 1.0)
        v_s_safe = jnp.where(v_s_pos, self.v_s[..., None, :], 1.0)
        e_ice_vol = jnp.where(
            v_i_pos,
            nemo_source_round(nemo_source_round(self.e_i / v_i_safe) * float(nlay_i)),
            0.0,
        )
        e_snow_vol = jnp.where(
            v_s_pos,
            nemo_source_round(nemo_source_round(self.e_s / v_s_safe) * float(nlay_s)),
            0.0,
        )

        return SI3ColumnState(
            concentration=_s3(self.a_i),
            h_ice=_s3(self.equivalent.h_i),
            h_snow=_s3(self.equivalent.h_s),
            T_surface=_s3(self.t_su),
            e_ice=_s4(e_ice_vol),
            e_snow=_s4(e_snow_vol),
            S_bulk=_s3(self.equivalent.s_i),
            S_layers=_s4(self.equivalent.sz_i),
            age_volume=_s3(self.equivalent.o_i),
        )


def _hfn_bisection_primal(
    jpl: int,
    rn_himean_val: Field,
    opt_hips: int,
) -> tuple[Field, Field]:
    """Execute NEMO's exact `ice_itd_init` category-bound computation (`src/ICE/iceitd.F90:721-764`)."""
    sr = nemo_source_round
    rn_himean_f64 = jnp.asarray(rn_himean_val, dtype=jnp.float64)
    if opt_hips == 1:
        hi_max_list = [jnp.asarray(0.0, dtype=jnp.float64)]
        hi_mean_list = []
        znum = sr(
            sr(float(jpl) * sr(3.0 * float(jpl) + 1.0)) * rn_himean_f64
        )
        zden = jnp.asarray(1.0, dtype=jnp.float64)
        denom_const = float(jpl * (jpl + 1))
        for jl in range(1, jpl + 1):
            zden = sr(zden + sr(sr(6.0 * float(jl - 1)) / float(jpl)))
            h_prev = hi_max_list[-1]
            h_curr = sr(h_prev + sr(znum / sr(zden * denom_const)))
            hi_max_list.append(h_curr)
            hi_mean_list.append(sr(sr(h_curr + h_prev) * 0.5))
        hi_max_list[-1] = jnp.asarray(_HI_MAX_SENTINEL, dtype=jnp.float64)
        return jnp.stack(hi_max_list), jnp.stack(hi_mean_list)

    if opt_hips != 2:
        raise ValueError(f"Unsupported opt_hips={opt_hips}; expected 1 or 2")

    zden = jnp.asarray(float(jpl * jpl + 1), dtype=jnp.float64)
    rjpl = jnp.asarray(float(jpl), dtype=jnp.float64)

    def _eval_from_zhmax(zhmax: Field) -> tuple[Field, Field, Field]:
        h_prev = jnp.asarray(0.0, dtype=jnp.float64)
        zhm = jnp.asarray(0.0, dtype=jnp.float64)
        h_max_vals = [h_prev]
        h_mean_vals = []
        zhmax_sq = sr(zhmax * zhmax)
        for jl in range(1, jpl + 1):
            znum = jnp.asarray(2.0 * float(jl) - 1.0, dtype=jnp.float64)
            rad = sr(sr(h_prev * h_prev) + sr(sr(zhmax_sq * znum) / zden))
            h_curr = sr(jnp.sqrt(rad))
            h_m = sr(sr(h_curr + h_prev) * 0.5)
            zhm = sr(zhm + sr(h_m / rjpl))
            h_max_vals.append(h_curr)
            h_mean_vals.append(h_m)
            h_prev = h_curr
        return jnp.stack(h_max_vals), jnp.stack(h_mean_vals), zhm

    init_z1 = jnp.asarray(_HFN_BISECTION_LOWER_M, dtype=jnp.float64)
    init_z2 = jnp.asarray(100.0, dtype=jnp.float64)
    init_zhm = jnp.asarray(0.0, dtype=jnp.float64)
    init_zhmax = jnp.asarray(0.0, dtype=jnp.float64)
    init_hi_max = jnp.zeros((jpl + 1,), dtype=jnp.float64)
    init_hi_mean = jnp.zeros((jpl,), dtype=jnp.float64)

    def _cond(carry: tuple[Field, Field, Field, Field, Field, Field]) -> Field:
        z1, z2, zhm, _zhmax, _hmax, _hmean = carry
        return (jnp.abs(sr(zhm - rn_himean_f64)) > 1.0e-10) & (
            jnp.abs(sr(z2 - z1)) > 1.0e-10
        )

    def _body(
        carry: tuple[Field, Field, Field, Field, Field, Field],
    ) -> tuple[Field, Field, Field, Field, Field, Field]:
        z1, z2, _zhm, _zhmax, _hmax, _hmean = carry
        zhmax = sr(sr(z1 + z2) * 0.5)
        h_max_arr, h_mean_arr, zhm = _eval_from_zhmax(zhmax)
        zf0 = sr(z1 - rn_himean_f64)
        zf1 = sr(zhm - rn_himean_f64)
        prod_pos = sr(zf0 * zf1) > 0.0
        new_z1 = jnp.where(prod_pos, zhmax, z1)
        new_z2 = jnp.where(prod_pos, z2, zhmax)
        return new_z1, new_z2, zhm, zhmax, h_max_arr, h_mean_arr

    _z1, _z2, _zhm, _zhmax, hi_max_raw, hi_mean = jax.lax.while_loop(
        _cond,
        _body,
        (init_z1, init_z2, init_zhm, init_zhmax, init_hi_max, init_hi_mean),
    )
    hi_max = hi_max_raw.at[jpl].set(jnp.asarray(_HI_MAX_SENTINEL, dtype=jnp.float64))
    return hi_max, hi_mean


def si3_hfn_category_bounds(
    jpl: int = 5,
    rn_himean: float | Field = 2.0,
    opt_hips: int = 0,
    rn_himax: float | Field = _ORCA2_RN_HIMAX_M,
) -> SI3CategoryBounds:
    """Compute SI3 category thickness bounds `hi_max(0:jpl)` and `hi_mean(1:jpl)`.

    Source citation: `src/ICE/iceitd.F90:736-761` (compiled
    `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceitd.f90:736-761`).
    Supports end-to-end `jax.jit`, `jax.jvp`, and `jax.grad` with respect to
    `rn_himean`.
    """
    if opt_hips in (1, 2):
        @jax.custom_jvp
        def _solve(himean_arg: Field) -> SI3CategoryBounds:
            hi_max, hi_mean = _hfn_bisection_primal(jpl, himean_arg, opt_hips)
            return SI3CategoryBounds(hi_max=hi_max, hi_mean=hi_mean)

        @_solve.defjvp
        def _solve_jvp(
            primals: tuple[Field], tangents: tuple[Field]
        ) -> tuple[SI3CategoryBounds, SI3CategoryBounds]:
            (himean_p,) = primals
            (himean_t,) = tangents
            primal_out = _solve(himean_p)
            safe_himean = jnp.where(jnp.abs(himean_p) > _EPSI20, himean_p, 1.0)
            scale = himean_t / safe_himean
            d_hi_max = (primal_out.hi_max * scale).at[0].set(0.0).at[jpl].set(0.0)
            d_hi_mean = primal_out.hi_mean * scale
            return primal_out, SI3CategoryBounds(hi_max=d_hi_max, hi_mean=d_hi_mean)

        return _solve(jnp.asarray(rn_himean, dtype=jnp.float64))

    if opt_hips != 0:
        raise ValueError(f"Unsupported opt_hips={opt_hips}; expected 0 (NEMO 5.0.2 np_cathfn), 1, or 2")

    sr = nemo_source_round
    himean_f64 = jnp.asarray(rn_himean, dtype=jnp.float64)
    himax_f64 = jnp.asarray(rn_himax, dtype=jnp.float64)
    zalpha = jnp.asarray(_HFN_ALPHA, dtype=jnp.float64)
    inv_zalpha = jnp.asarray(
        np.float64(np.float32(1.0)) / np.float64(_HFN_ALPHA), dtype=jnp.float64
    )
    zhmax = sr(3.0 * himean_f64)
    zhmax_p1_pow = sr(jnp.power(sr(zhmax + 1.0), zalpha))

    hi_max_list = [jnp.asarray(0.0, dtype=jnp.float64)]
    for jl in range(1, jpl + 1):
        znum = sr(float(jpl) * zhmax_p1_pow)
        zden = sr(sr(float(jpl - jl) * zhmax_p1_pow) + float(jl))
        hi_max_jl = sr(sr(jnp.power(sr(znum / zden), inv_zalpha)) - 1.0)
        hi_max_list.append(hi_max_jl)

    hi_max_raw = jnp.stack(hi_max_list)
    hi_mean = sr(sr(hi_max_raw[1:] + hi_max_raw[:-1]) * 0.5)
    hi_max = hi_max_raw.at[jpl].set(himax_f64)
    return SI3CategoryBounds(hi_max=hi_max, hi_mean=hi_mean)


def fortran_sum_categories(arr: Field, axis: int = -1) -> Field:
    """Sequential category summation matching Fortran `SUM(..., dim=...)` rounding."""
    sr = nemo_source_round
    n = arr.shape[axis]
    if n == 0:
        return jnp.sum(arr, axis=axis)
    acc = jnp.take(arr, 0, axis=axis)
    for idx in range(1, n):
        acc = sr(acc + jnp.take(arr, idx, axis=axis))
    return acc


def _interior_mask_2d(jpi: int, jpj: int, nn_hls: int, dtype: Any = jnp.float64) -> Field:
    """Return 1.0 on NEMO `A2D(0)` interior `(Nis0:Nie0, Njs0:Nje0)` and 0.0 on halos."""
    if nn_hls <= 0 or (2 * nn_hls >= jpi) or (2 * nn_hls >= jpj):
        return jnp.ones((jpi, jpj), dtype=dtype)
    mask = jnp.zeros((jpi, jpj), dtype=dtype)
    return mask.at[nn_hls : jpi - nn_hls, nn_hls : jpj - nn_hls].set(1.0)


def si3_var_salprof(
    s_i: Field,
    sss_m: Field | float | None = None,
    config: SI3ORCA2StateConfig | None = None,
) -> tuple[Field, Field]:
    """Compute SI3 ice salinity profile `sz_i` (`src/ICE/icevar.F90:466-534`).

    Source citation: `src/ICE/icevar.F90:477-532` (compiled
    `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:491-545`).
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    sr = nemo_source_round
    dtype = s_i.dtype
    jpi, jpj, jpl = s_i.shape
    nlay_i = cfg.nlay_i
    r1_nlay_i = sr(jnp.asarray(1.0 / float(nlay_i), dtype=dtype))

    if sss_m is None:
        sss_2d = jnp.full((jpi, jpj), canonical_constants.S_ocean_ref, dtype=dtype)
    else:
        sss_2d = jnp.asarray(sss_m, dtype=dtype)
        if sss_2d.ndim == 0:
            sss_2d = jnp.full((jpi, jpj), sss_2d, dtype=dtype)

    if cfg.nn_icesal == 1:
        s_out = jnp.full_like(s_i, cfg.rn_icesal)
        sz_out = jnp.full((jpi, jpj, nlay_i, jpl), cfg.rn_icesal, dtype=dtype)
        return s_out, sz_out

    if cfg.nn_icesal in (2, 4):
        z1_dS = sr(jnp.asarray(1.0 / (_ZSI1 - _ZSI0), dtype=dtype))
        zalpha = jnp.where(
            s_i >= _ZSI1,
            0.0,
            jnp.where(s_i <= _ZSI0, 1.0, sr(sr(_ZSI1 - s_i) * z1_dS)),
        )
        sss_3d = sss_2d[..., None]
        upper_sss = sr(sr(0.5 * cfg.rn_sinew) * sss_3d)
        lower_sss = sr(
            float(nlay_i) * jnp.minimum(cfg.rn_simin, sr(cfg.rn_sinew * sss_3d))
        )
        zalpha = jnp.where(s_i >= upper_sss, 0.0, zalpha)
        zalpha = jnp.where(s_i <= lower_sss, 0.0, zalpha)

        layers = []
        for jk in range(1, nlay_i + 1):
            jk_term = jnp.asarray(float(jk) - 0.5, dtype=dtype)
            term1 = sr(sr(sr(sr(zalpha * s_i) * 2.0) * jk_term) * r1_nlay_i)
            term2 = sr(sr(1.0 - zalpha) * s_i)
            layers.append(sr(term1 + term2))
        sz_out = jnp.stack(layers, axis=2)
        return s_i, sz_out

    if cfg.nn_icesal == 3:
        s_out = jnp.full_like(s_i, _SALPROF_CONSTANT_G_PER_KG)
        rpi = jnp.asarray(float(canonical_constants.PI), dtype=dtype)
        layers = []
        for jk in range(1, nlay_i + 1):
            ztmp1 = sr(jnp.asarray(float(jk) - 0.5, dtype=dtype) * r1_nlay_i)
            exponent = sr(
                _SALPROF_EXPONENT_NUMERATOR / sr(_SALPROF_EXPONENT_OFFSET + ztmp1)
            )
            ztmp2 = sr(
                _SALPROF_AMPLITUDE
                * sr(1.0 - jnp.cos(sr(rpi * jnp.power(ztmp1, exponent))))
            )
            layers.append(jnp.full((jpi, jpj, jpl), ztmp2, dtype=dtype))
        sz_out = jnp.stack(layers, axis=2)
        return s_out, sz_out

    raise ValueError(f"Unsupported nn_icesal={cfg.nn_icesal}")


def si3_var_snwfra_3d(
    h_s: Field,
    config: SI3ORCA2StateConfig | None = None,
) -> Field:
    """Compute snow coverage fraction `pa_s_fra` (`src/ICE/icevar.F90:1564-1587`).

    Source citation: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:1578-1601`.
    Only the interior `A2D(0)` (`Nis0:Nie0, Njs0:Nje0`) is populated by NEMO.
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    sr = nemo_source_round
    consts = cfg.ice_constants
    jpi, jpj, _jpl = h_s.shape
    interior = _interior_mask_2d(jpi, jpj, cfg.nn_hls, dtype=h_s.dtype)[..., None]

    if cfg.nn_snwfra == 0:
        raw = jnp.where(h_s > 0.0, 1.0, 0.0)
    elif cfg.nn_snwfra == 1:
        raw = sr(
            1.0
            - jnp.exp(sr(sr(_SNOW_FRACTION_EXPONENT * consts.rho_snow) * h_s))
        )
    elif cfg.nn_snwfra == 2:
        raw = sr(h_s / sr(h_s + canonical_constants.snow_cover_depth_nemo))
    else:
        raise ValueError(f"Unsupported nn_snwfra={cfg.nn_snwfra}")
    return raw * interior


def si3_var_glo2eqv(
    v_i: Field,
    v_s: Field,
    sv_i: Field,
    a_i: Field,
    oa_i: Field,
    a_ip: Field,
    v_ip: Field,
    v_il: Field,
    e_s: Field,
    e_i: Field,
    szv_i: Field,
    *,
    kn: int = 2,
    hi_max_top: float | Field = _HI_MAX_SENTINEL,
    sss_m: Field | float | None = None,
    config: SI3ORCA2StateConfig | None = None,
    _plant_wrong_formula: bool = False,
) -> SI3EquivalentVars:
    """Convert global extensive SI3 fields to intensive equivalents (`ice_var_glo2eqv`).

    Source citation: `src/ICE/icevar.F90:255-428` (compiled
    `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:269-442`).
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    sr = nemo_source_round
    consts = cfg.ice_constants
    dtype = v_i.dtype
    jpi, jpj, jpl = v_i.shape
    nlay_i = cfg.nlay_i
    nlay_s = cfg.nlay_s

    zhmax = jnp.asarray(hi_max_top, dtype=dtype)
    z1_hmax = sr(1.0 / zhmax)
    v_i_top = v_i[..., jpl - 1]
    a_i_top = a_i[..., jpl - 1]
    a_i_top_new = jnp.where(
        v_i_top > sr(zhmax * a_i_top),
        jnp.minimum(1.0, sr(v_i_top * z1_hmax)),
        a_i_top,
    )
    a_i_upd = a_i.at[..., jpl - 1].set(a_i_top_new)

    # Safe denominator for AD through jnp.where.
    a_i_safe = jnp.where(a_i_upd > _EPSI20, a_i_upd, 1.0)
    z1_a_i = jnp.where(a_i_upd > _EPSI20, sr(1.0 / a_i_safe), 0.0)
    h_i = sr(v_i * z1_a_i)
    h_s = sr(v_s * z1_a_i)
    o_i = sr(oa_i * z1_a_i)

    interior_3d = _interior_mask_2d(jpi, jpj, cfg.nn_hls, dtype=dtype)[..., None]
    if kn == 1 or cfg.ln_pnd:
        z1_hl = sr(
            jnp.asarray(1.0 / (cfg.rn_pnd_hl_max - cfg.rn_pnd_hl_min), dtype=dtype)
        )
        a_ip_safe = jnp.where(a_ip > _EPSI20, a_ip, 1.0)
        z1_a_ip = jnp.where(a_ip > _EPSI20, sr(1.0 / a_ip_safe), 0.0)
        h_ip = sr(v_ip * z1_a_ip)
        h_il = sr(v_il * z1_a_ip)

        a_ip_frac_full = jnp.where(a_i_upd > _EPSI20, sr(a_ip / a_i_safe), 0.0)
        a_ip_frac = a_ip_frac_full * interior_3d

        partial_eff = sr(
            sr(a_ip_frac * sr(cfg.rn_pnd_hl_max - h_il)) * z1_hl
        )
        a_ip_eff_raw = jnp.where(
            h_il <= cfg.rn_pnd_hl_min,
            a_ip_frac,
            jnp.where(h_il >= cfg.rn_pnd_hl_max, 0.0, partial_eff),
        )
        a_ip_eff_raw = jnp.where(h_ip < h_s, 0.0, a_ip_eff_raw)
        za_s_fra = si3_var_snwfra_3d(h_s, config=cfg)
        a_ip_eff = (
            jnp.minimum(a_ip_eff_raw, sr(1.0 - za_s_fra)) * interior_3d
        )
    else:
        z3 = jnp.zeros_like(v_i)
        h_ip = z3
        h_il = z3
        a_ip_frac = z3
        a_ip_eff = z3

    if sss_m is None:
        sss_2d = jnp.full((jpi, jpj), canonical_constants.S_ocean_ref, dtype=dtype)
    else:
        sss_2d = jnp.asarray(sss_m, dtype=dtype)
        if sss_2d.ndim == 0:
            sss_2d = jnp.full((jpi, jpj), sss_2d, dtype=dtype)
    s_no_ice = jnp.minimum(cfg.rn_simin, sr(cfg.rn_sinew * sss_2d))[..., None]
    v_i_safe = jnp.where(v_i > _EPSI20, v_i, 1.0)

    if cfg.nn_icesal in (1, 3):
        s_i, sz_i = si3_var_salprof(jnp.zeros_like(v_i), sss_m=sss_2d, config=cfg)
    elif cfg.nn_icesal == 2:
        s_i_raw = jnp.where(v_i > _EPSI20, sr(sv_i / v_i_safe), s_no_ice)
        s_i, sz_i = si3_var_salprof(s_i_raw, sss_m=sss_2d, config=cfg)
    elif cfg.nn_icesal == 4:
        r1_nlay_i = sr(jnp.asarray(1.0 / float(nlay_i), dtype=dtype))
        denom_4 = sr(v_i_safe[..., None, :] * r1_nlay_i)
        zs_i = sr(szv_i / denom_4)
        sz_i = jnp.where(
            (v_i > _EPSI20)[..., None, :],
            zs_i,
            jnp.broadcast_to(s_no_ice[..., None, :], szv_i.shape),
        )
        s_i_acc = jnp.zeros_like(v_i)
        for jk in range(nlay_i):
            s_i_acc = sr(s_i_acc + sr(sz_i[:, :, jk, :] * r1_nlay_i))
        s_i = s_i_acc
    else:
        raise ValueError(f"Unsupported nn_icesal={cfg.nn_icesal}")

    zlay_i = jnp.asarray(float(nlay_i), dtype=dtype)
    r1_rhoi = sr(jnp.asarray(1.0 / consts.rho_ice, dtype=dtype))
    r1_rcpi = sr(jnp.asarray(1.0 / consts.c_ice, dtype=dtype))
    rcp_minus_rcpi = sr(jnp.asarray(consts.c_ocean - consts.c_ice, dtype=dtype))
    rLfus = jnp.asarray(consts.latent_fusion, dtype=dtype)
    rTmlt = jnp.asarray(consts.liquidus_slope, dtype=dtype)
    rt0 = jnp.asarray(consts.T0, dtype=dtype)

    v_i_4d = v_i[..., None, :]
    v_i_4d_safe = jnp.where(v_i_4d > _EPSI20, v_i_4d, 1.0)
    ze_i = sr(sr(e_i / v_i_4d_safe) * zlay_i)
    ztmelts = sr(-sz_i * rTmlt)
    zbbb = sr(sr(sr(rcp_minus_rcpi * ztmelts) + sr(ze_i * r1_rhoi)) - rLfus)
    four_rcpi_rlfus = jnp.asarray(
        4.0 * consts.c_ice * consts.latent_fusion, dtype=dtype
    )
    disc_arg = jnp.maximum(
        sr(sr(zbbb * zbbb) - sr(four_rcpi_rlfus * ztmelts)),
        0.0,
    )
    disc_safe = jnp.where(disc_arg > 0.0, disc_arg, 1.0)
    zccc = jnp.where(disc_arg > 0.0, sr(jnp.sqrt(disc_safe)), 0.0)
    if _plant_wrong_formula:
        # Plant P5: flip sign of zccc in quadratic root.
        t_i_cel = jnp.maximum(
            -100.0, jnp.minimum(sr(sr(-sr(zbbb - zccc) * 0.5) * r1_rcpi), ztmelts)
        )
    else:
        t_i_cel = jnp.maximum(
            -100.0, jnp.minimum(sr(sr(-sr(zbbb + zccc) * 0.5) * r1_rcpi), ztmelts)
        )
    t_i = jnp.where(v_i_4d > _EPSI20, sr(t_i_cel + rt0), rt0)

    zlay_s = jnp.asarray(float(nlay_s), dtype=dtype)
    r1_rhos = sr(jnp.asarray(1.0 / consts.rho_snow, dtype=dtype))
    v_s_4d = v_s[..., None, :]
    v_s_4d_safe = jnp.where(v_s_4d > _EPSI20, v_s_4d, 1.0)
    ze_s = sr(sr(e_s / v_s_4d_safe) * zlay_s)
    t_s_cel = jnp.maximum(
        -100.0,
        jnp.minimum(sr(r1_rcpi * sr(sr(-r1_rhos * ze_s) + rLfus)), 0.0),
    )
    t_s = jnp.where(v_s_4d > _EPSI20, sr(rt0 + t_s_cel), rt0)

    return SI3EquivalentVars(
        h_i=h_i,
        h_s=h_s,
        o_i=o_i,
        h_ip=h_ip,
        h_il=h_il,
        a_ip_frac=a_ip_frac,
        a_ip_eff=a_ip_eff,
        s_i=s_i,
        sz_i=sz_i,
        t_i=t_i,
        t_s=t_s,
        a_i=a_i_upd,
        v_s=v_s,
    )


def si3_var_agg(
    v_i: Field,
    v_s: Field,
    sv_i: Field,
    a_i: Field,
    oa_i: Field,
    a_ip: Field,
    v_ip: Field,
    v_il: Field,
    t_su: Field,
    t_si: Field,
    e_s: Field,
    e_i: Field,
    szv_i: Field,
    a_ip_eff: Field,
    t_i: Field,
    t_s: Field,
    *,
    kn: int = 2,
    config: SI3ORCA2StateConfig | None = None,
    _plant_wrong_formula: bool = False,
) -> SI3AggregatedVars:
    """Aggregate SI3 category variables (`ice_var_agg`, `src/ICE/icevar.F90:105-252`).

    Source citation: `src/ICE/icevar.F90:105-252` (compiled
    `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:119-266`).
    Full arrays (`jpi, jpj`): `vt_i`, `vt_s`, `at_i`, `at_ip`, `vt_ip`, `vt_il`, `ato_i`.
    Reduced `A2D(0)` arrays (`Nis0:Nie0, Njs0:Nje0`, zero in halo ring):
    `et_s`, `et_i`, `at_ip_eff`, `tm_su`, `st_i`, `hm_i`, `hm_s`, `tm_si`,
    `om_i`, `sm_i`, `tm_i`, `tm_s`, `hm_ip`, `hm_il`.
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    sr = nemo_source_round
    consts = cfg.ice_constants
    dtype = v_i.dtype
    jpi, jpj, jpl = v_i.shape
    nlay_i = cfg.nlay_i
    nlay_s = cfg.nlay_s
    rt0 = jnp.asarray(consts.T0, dtype=dtype)
    interior = _interior_mask_2d(jpi, jpj, cfg.nn_hls, dtype=dtype)

    vt_i = fortran_sum_categories(v_i, axis=-1)
    vt_s = fortran_sum_categories(v_s, axis=-1)
    at_i = fortran_sum_categories(a_i, axis=-1)
    at_ip = fortran_sum_categories(a_ip, axis=-1)
    vt_ip = fortran_sum_categories(v_ip, axis=-1)
    vt_il = fortran_sum_categories(v_il, axis=-1)
    ato_i = sr(1.0 - at_i)

    # SUM(SUM(e_s(ji,jj,:,:), dim=2)) sums over jl (axis=-1) then jk (axis=-1).
    et_s = fortran_sum_categories(fortran_sum_categories(e_s, axis=-1), axis=-1) * interior
    et_i = fortran_sum_categories(fortran_sum_categories(e_i, axis=-1), axis=-1) * interior
    at_ip_eff = fortran_sum_categories(sr(a_ip_eff * a_i), axis=-1) * interior

    at_i_safe = jnp.where(at_i > _EPSI20, at_i, 1.0)
    tm_su_num = fortran_sum_categories(sr(t_su * a_i), axis=-1)
    tm_su_full = jnp.where(at_i <= _EPSI20, rt0, sr(tm_su_num / at_i_safe))
    tm_su = tm_su_full * interior

    if cfg.nn_icesal == 4:
        st_i = (
            fortran_sum_categories(fortran_sum_categories(szv_i, axis=-1), axis=-1)
            * interior
        )
    else:
        st_i = fortran_sum_categories(sv_i, axis=-1) * interior

    if kn <= 1:
        z2 = jnp.zeros((jpi, jpj), dtype=dtype)
        return SI3AggregatedVars(
            vt_i=vt_i,
            vt_s=vt_s,
            at_i=at_i,
            at_ip=at_ip,
            vt_ip=vt_ip,
            vt_il=vt_il,
            ato_i=ato_i,
            et_s=et_s,
            et_i=et_i,
            at_ip_eff=at_ip_eff,
            tm_su=tm_su,
            st_i=st_i,
            hm_i=z2,
            hm_s=z2,
            tm_si=z2,
            om_i=z2,
            sm_i=z2,
            tm_i=z2,
            tm_s=z2,
            hm_ip=z2,
            hm_il=z2,
        )

    z1_at_i = jnp.where(at_i > _EPSI20, sr(1.0 / at_i_safe), 0.0)
    vt_i_safe = jnp.where(vt_i > _EPSI20, vt_i, 1.0)
    z1_vt_i = jnp.where(vt_i > _EPSI20, sr(1.0 / vt_i_safe), 0.0)

    if _plant_wrong_formula:
        # Plant P5: unweighted category sum instead of dividing by at_i.
        hm_i = vt_i * interior
    else:
        hm_i = sr(vt_i * z1_at_i) * interior
    hm_s = sr(vt_s * z1_at_i) * interior
    tm_si_raw = sr(fortran_sum_categories(sr(t_si * a_i), axis=-1) * z1_at_i)
    om_i = sr(fortran_sum_categories(oa_i, axis=-1) * z1_at_i) * interior
    sm_i = sr(st_i * z1_vt_i) * interior

    r1_nlay_i = sr(jnp.asarray(1.0 / float(nlay_i), dtype=dtype))
    tm_i_acc = jnp.zeros((jpi, jpj), dtype=dtype)
    for jl in range(jpl):
        v_i_cat = v_i[:, :, jl]
        for jk in range(nlay_i):
            contrib = sr(sr(sr(r1_nlay_i * t_i[:, :, jk, jl]) * v_i_cat) / vt_i_safe)
            tm_i_acc = jnp.where(vt_i > _EPSI20, sr(tm_i_acc + contrib), rt0)

    r1_nlay_s = sr(jnp.asarray(1.0 / float(nlay_s), dtype=dtype))
    vt_s_safe = jnp.where(vt_s > _EPSI20, vt_s, 1.0)
    tm_s_acc = jnp.zeros((jpi, jpj), dtype=dtype)
    for jl in range(jpl):
        v_s_cat = v_s[:, :, jl]
        for jk in range(nlay_s):
            contrib = sr(sr(sr(r1_nlay_s * t_s[:, :, jk, jl]) * v_s_cat) / vt_s_safe)
            tm_s_acc = jnp.where(vt_s > _EPSI20, sr(tm_s_acc + contrib), rt0)

    no_ice = at_i <= _EPSI20
    tm_si = jnp.where(no_ice, rt0, tm_si_raw) * interior
    tm_i = jnp.where(no_ice, rt0, tm_i_acc) * interior
    tm_s = jnp.where(no_ice, rt0, tm_s_acc) * interior

    at_ip_safe = jnp.where(at_ip > _EPSI20, at_ip, 1.0)
    hm_ip = jnp.where(at_ip > _EPSI20, sr(vt_ip / at_ip_safe), 0.0) * interior
    hm_il = jnp.where(at_ip > _EPSI20, sr(vt_il / at_ip_safe), 0.0) * interior

    return SI3AggregatedVars(
        vt_i=vt_i,
        vt_s=vt_s,
        at_i=at_i,
        at_ip=at_ip,
        vt_ip=vt_ip,
        vt_il=vt_il,
        ato_i=ato_i,
        et_s=et_s,
        et_i=et_i,
        at_ip_eff=at_ip_eff,
        tm_su=tm_su,
        st_i=st_i,
        hm_i=hm_i,
        hm_s=hm_s,
        tm_si=tm_si,
        om_i=om_i,
        sm_i=sm_i,
        tm_i=tm_i,
        tm_s=tm_s,
        hm_ip=hm_ip,
        hm_il=hm_il,
    )


def si3_update_snwice_mass(
    vt_i: Field,
    vt_s: Field,
    vt_ip: Field,
    vt_il: Field,
    snwice_mass_prev: Field,
    tmask: Field,
    *,
    config: SI3ORCA2StateConfig | None = None,
    is_init: bool = False,
) -> tuple[Field, Field, Field]:
    """Update `snwice_mass`, `snwice_mass_b`, and `snwice_fmass` (`src/ICE/iceupdate.F90:189-194`).

    Source citation: `src/ICE/iceupdate.F90:189-194` and `:464-471`.
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    sr = nemo_source_round
    consts = cfg.ice_constants
    rhos_vt_s = sr(consts.rho_snow * vt_s)
    rhoi_vt_i = sr(consts.rho_ice * vt_i)
    rhow_pnd = sr(canonical_constants.rho_water * sr(vt_ip + vt_il))
    new_mass = sr(tmask * sr(sr(rhos_vt_s + rhoi_vt_i) + rhow_pnd))
    if is_init:
        return new_mass, new_mass, jnp.zeros_like(new_mass)
    r1_Dt_ice = sr(jnp.asarray(1.0 / cfg.rDt_ice, dtype=vt_i.dtype))
    fmass = sr(sr(new_mass - snwice_mass_prev) * r1_Dt_ice)
    return new_mass, snwice_mass_prev, fmass


def si3_lbc_lnk_orca2(
    field: Field,
    cd_nat: str = "T",
    psgn: float = 1.0,
    nn_hls: int = 2,
    ldfull: bool = True,
    *,
    _plant_wrong_nfd_sign: bool = False,
) -> Field:
    """Apply ORCA2 lateral boundary conditions (`lbc_lnk` + `lbc_nfd` for `c_NFtype == 'T'`).

    Implements:
    1. East-west cyclic periodicity (`src/OCE/LBC/lbc_lnk_pt2pt_generic.h90:204-218`):
       - west halo `0:ihls` <- `jpi - 2*ihls : jpi - ihls`
       - east halo `jpi - ihls : jpi` <- `ihls : 2*ihls`
    2. North-fold T-pivot reflection (`src/OCE/LBC/lbc_nfd_generic.h90:22-191`,
       `c_NFtype = 'T'`) for grid point natures `'T'`, `'U'`, `'V'`, `'F'`.
    3. South closed boundary (`src/OCE/LBC/lbc_lnk_pt2pt_generic.h90:245-260`):
       - south halo `0:ihls` set to `0.0`.
    """
    if nn_hls <= 0:
        return field
    ihls = int(nn_hls)
    ipi = int(field.shape[0])
    ipj = int(field.shape[1])
    ni0glo = ipi - 2 * ihls
    sign = -float(psgn) if _plant_wrong_nfd_sign else float(psgn)

    # 1. East-west cyclic seam (1-based: 1..ihls <- ipi-2*ihls+1..ipi-ihls;
    #    ipi-ihls+1..ipi <- ihls+1..2*ihls).
    out = field.at[:ihls, :].set(field[ipi - 2 * ihls : ipi - ihls, :])
    out = out.at[ipi - ihls :, :].set(out[ihls : 2 * ihls, :])

    # 2. North fold (c_NFtype == 'T', 1-based formulas from lbc_nfd_generic.h90:24-187).
    if cd_nat == "T":
        for jj in range(1, ihls + 1):
            ij1 = ipj - jj + 1
            ij2 = ipj - 2 * ihls + jj - 1
            src_i_1based = []
            for ji in range(1, ipi + 1):
                if ji <= ihls:
                    ii2 = 2 * ihls + 2 - ji
                elif ji == ihls + 1:
                    ii2 = ihls + 1
                elif ji <= ipi - ihls:
                    ii2 = ipi + 2 - ji
                elif ji == ipi - ihls + 1:
                    ii2 = ihls + 1
                else:
                    ii2 = 2 * (ipi - ihls + 1) - ji
                src_i_1based.append(ii2 - 1)
            src_idx = jnp.asarray(src_i_1based, dtype=jnp.int32)
            out = out.at[:, ij1 - 1].set(sign * out[src_idx, ij2 - 1])

        if ldfull:
            ij1 = ipj - ihls
            row = out[:, ij1 - 1]
            dst_list = []
            src_list = []
            for ji in range(1, ni0glo // 2):
                ii1 = ipi // 2 + ji + 1
                ii2 = ipi // 2 - ji + 1
                dst_list.append(ii1 - 1)
                src_list.append(ii2 - 1)
            for ji in range(1, ihls + 1):
                ii1 = ji
                ii2 = 2 * ihls + 2 - ji
                dst_list.append(ii1 - 1)
                src_list.append(ii2 - 1)
            dst_idx = jnp.asarray(dst_list, dtype=jnp.int32)
            src_idx = jnp.asarray(src_list, dtype=jnp.int32)
            row = row.at[dst_idx].set(sign * row[src_idx])
            out = out.at[:, ij1 - 1].set(row)

    elif cd_nat == "U":
        for jj in range(1, ihls + 1):
            ij1 = ipj - jj + 1
            ij2 = ipj - 2 * ihls + jj - 1
            src_i_1based = []
            for ji in range(1, ipi + 1):
                if ji <= ihls:
                    ii2 = 2 * ihls + 1 - ji
                elif ji <= ipi - ihls - 1:
                    ii2 = ipi + 1 - ji
                elif ji == ipi - ihls:
                    ii2 = ihls + 1
                else:
                    ii2 = 2 * (ipi - ihls) + 1 - ji
                src_i_1based.append(ii2 - 1)
            src_idx = jnp.asarray(src_i_1based, dtype=jnp.int32)
            out = out.at[:, ij1 - 1].set(sign * out[src_idx, ij2 - 1])

        if ldfull:
            ij1 = ipj - ihls
            row = out[:, ij1 - 1]
            dst_list = []
            src_list = []
            for ji in range(1, ni0glo // 2):
                ii1 = ipi // 2 + ji
                ii2 = ipi // 2 - ji + 1
                dst_list.append(ii1 - 1)
                src_list.append(ii2 - 1)
            for ji in range(1, ihls + 1):
                ii1 = ji
                ii2 = 2 * ihls + 1 - ji
                dst_list.append(ii1 - 1)
                src_list.append(ii2 - 1)
            ii1_end = ipi - ihls
            ii2_end = ihls + 1
            dst_list.append(ii1_end - 1)
            src_list.append(ii2_end - 1)
            dst_idx = jnp.asarray(dst_list, dtype=jnp.int32)
            src_idx = jnp.asarray(src_list, dtype=jnp.int32)
            row = row.at[dst_idx].set(sign * row[src_idx])
            out = out.at[:, ij1 - 1].set(row)

    elif cd_nat == "V":
        for jj in range(1, ihls + 2 if ldfull else ihls + 1):
            ij1 = ipj - jj + 1
            ij2 = ipj - 2 * ihls + jj - 2
            src_i_1based = []
            for ji in range(1, ipi + 1):
                if ji <= ihls:
                    ii2 = 2 * ihls + 2 - ji
                elif ji == ihls + 1:
                    ii2 = ihls + 1
                elif ji <= ipi - ihls:
                    ii2 = ipi + 2 - ji
                elif ji == ipi - ihls + 1:
                    ii2 = ihls + 1
                else:
                    ii2 = 2 * (ipi - ihls + 1) - ji
                src_i_1based.append(ii2 - 1)
            src_idx = jnp.asarray(src_i_1based, dtype=jnp.int32)
            out = out.at[:, ij1 - 1].set(sign * out[src_idx, ij2 - 1])
    elif cd_nat == "F":
        # lbcnfd.f90:722-747 (ORCA2_SAS_ICE_O2_REF ppsrc); the extra line
        # jj = ihls+1 is the mpp_nfd "full" line (lbcnfd.f90:1741-1766).
        for jj in range(1, ihls + 2 if ldfull else ihls + 1):
            ij1 = ipj - jj + 1
            ij2 = ipj - 2 * ihls + jj - 2
            src_i_1based = []
            for ji in range(1, ipi + 1):
                if ji <= ihls:
                    ii2 = 2 * ihls + 1 - ji
                elif ji <= ipi - ihls:
                    ii2 = ipi + 1 - ji
                else:
                    ii2 = 2 * (ipi - ihls) + 1 - ji
                src_i_1based.append(ii2 - 1)
            src_idx = jnp.asarray(src_i_1based, dtype=jnp.int32)
            out = out.at[:, ij1 - 1].set(sign * out[src_idx, ij2 - 1])
    else:
        raise ValueError(f"Unsupported grid nature cd_nat={cd_nat!r}")

    # 3. South closed boundary (rows 0:ihls zeroed).
    out = out.at[:, :ihls].set(0.0)
    return out


def _apply_prather_family_halos(
    fam: SI3PratherMoments,
    nn_hls: int,
    ldfull: bool = True,
    *,
    _plant_wrong_nfd_sign: bool = False,
) -> SI3PratherMoments:
    """Apply `lbc_lnk` to a 5-moment Prather family (`src/ICE/icedyn_adv_pra.F90:438-482`).

    First-order gradients `sx, sy` use `psgn = -1.0_wp`; second-order moments
    `sxx, syy, sxy` use `psgn = +1.0_wp` (`src/ICE/icedyn_adv_pra.F90:440-480`).
    """
    return SI3PratherMoments(
        sx=si3_lbc_lnk_orca2(
            fam.sx, "T", -1.0, nn_hls=nn_hls, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign
        ),
        sy=si3_lbc_lnk_orca2(
            fam.sy, "T", -1.0, nn_hls=nn_hls, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign
        ),
        sxx=si3_lbc_lnk_orca2(fam.sxx, "T", 1.0, nn_hls=nn_hls, ldfull=ldfull),
        syy=si3_lbc_lnk_orca2(fam.syy, "T", 1.0, nn_hls=nn_hls, ldfull=ldfull),
        sxy=si3_lbc_lnk_orca2(fam.sxy, "T", 1.0, nn_hls=nn_hls, ldfull=ldfull),
    )


def si3_apply_state_halos_orca2(
    state: SI3State,
    config: SI3ORCA2StateConfig | None = None,
    ldfull: bool = True,
    *,
    _plant_wrong_nfd_sign: bool = False,
) -> SI3State:
    """Apply ORCA2 cyclic seam and T-pivot north-fold halos to the full `SI3State`.

    Source citations:
    - `src/ICE/icerst.F90:281-282`, `:307-356` (`ice_rst_read` `lbc_lnk` calls)
    - `src/ICE/icedyn_adv_pra.F90:438-482` (`adv_pra` `lbc_lnk` calls)
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    h = cfg.nn_hls

    def _t_pos(arr: Field) -> Field:
        return si3_lbc_lnk_orca2(arr, "T", 1.0, nn_hls=h, ldfull=ldfull)

    u_ice = si3_lbc_lnk_orca2(
        state.u_ice, "U", -1.0, nn_hls=h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign
    )
    v_ice = si3_lbc_lnk_orca2(
        state.v_ice, "V", -1.0, nn_hls=h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign
    )
    pr = state.prather
    prather_halos = SI3AllPratherMoments(
        ice=_apply_prather_family_halos(pr.ice, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        sn=_apply_prather_family_halos(pr.sn, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        a=_apply_prather_family_halos(pr.a, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        age=_apply_prather_family_halos(pr.age, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        sal=_apply_prather_family_halos(pr.sal, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        ap=_apply_prather_family_halos(pr.ap, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        vp=_apply_prather_family_halos(pr.vp, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        vl=_apply_prather_family_halos(pr.vl, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        c0=_apply_prather_family_halos(pr.c0, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        e=_apply_prather_family_halos(pr.e, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
        si=_apply_prather_family_halos(pr.si, h, ldfull=ldfull, _plant_wrong_nfd_sign=_plant_wrong_nfd_sign),
    )
    return state._replace(
        v_i=_t_pos(state.v_i),
        v_s=_t_pos(state.v_s),
        sv_i=_t_pos(state.sv_i),
        a_i=_t_pos(state.a_i),
        t_su=_t_pos(state.t_su),
        u_ice=u_ice,
        v_ice=v_ice,
        oa_i=_t_pos(state.oa_i),
        a_ip=_t_pos(state.a_ip),
        v_ip=_t_pos(state.v_ip),
        v_il=_t_pos(state.v_il),
        e_s=_t_pos(state.e_s),
        e_i=_t_pos(state.e_i),
        szv_i=_t_pos(state.szv_i),
        snwice_mass=_t_pos(state.snwice_mass),
        snwice_mass_b=_t_pos(state.snwice_mass_b),
        snwice_fmass=_t_pos(state.snwice_fmass),
        prather=prather_halos,
    )


class SI3ORCA2Forcing(NamedTuple):
    """Boundary forcing for `SI3State` steps through `step_sea_ice` (`thermo_scheme='si3_orca2_o1'`)."""

    tmask: Field
    sss_m: Field | float | None = None
    evaporation: Field | None = None
    apply_halos: bool = False


def si3_o1_step(
    state: SI3State,
    tmask: Field,
    *,
    sss_m: Field | float | None = None,
    config: SI3ORCA2StateConfig | None = None,
    apply_halos: bool = False,
    _plant_stop_gradient: bool = False,
) -> SI3State:
    """Execute one O1 SI3 step (`ln_icedyn = False`, `ln_icethd = False`).

    Follows NEMO's exact statement order in `src/ICE/icestp.F90:148-224`:
    1. `store_fields` (`:148`, `:361-394`) and `diag_set0` (`:164`, `:488`)
    2. `CALL ice_var_glo2eqv(2)` (`:218`)
    3. `CALL ice_var_agg(2)` (`:219`)
    4. `CALL ice_update_flx(kt)` snow/ice mass update (`src/ICE/iceupdate.F90:189-194`)
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    work = jax.lax.stop_gradient(state) if _plant_stop_gradient else state
    if apply_halos:
        work = si3_apply_state_halos_orca2(work, config=cfg)

    # diag_set0 (`src/ICE/icestp.F90:164, 488`): t_si(A2D(0), 1:jpl) = rt0
    jpi, jpj, _ = work.v_i.shape
    interior_3d = _interior_mask_2d(jpi, jpj, cfg.nn_hls)[:, :, None]
    t_si_step = jnp.where(interior_3d, jnp.full_like(work.t_si, cfg.ice_constants.T0), 0.0)
    work = work._replace(t_si=t_si_step)

    eqv = si3_var_glo2eqv(
        v_i=work.v_i,
        v_s=work.v_s,
        sv_i=work.sv_i,
        a_i=work.a_i,
        oa_i=work.oa_i,
        a_ip=work.a_ip,
        v_ip=work.v_ip,
        v_il=work.v_il,
        e_s=work.e_s,
        e_i=work.e_i,
        szv_i=work.szv_i,
        kn=2,
        hi_max_top=work.bounds.hi_max[-1],
        sss_m=sss_m,
        config=cfg,
    )
    agg = si3_var_agg(
        v_i=work.v_i,
        v_s=eqv.v_s,
        sv_i=work.sv_i,
        a_i=eqv.a_i,
        oa_i=work.oa_i,
        a_ip=work.a_ip,
        v_ip=work.v_ip,
        v_il=work.v_il,
        t_su=work.t_su,
        t_si=work.t_si,
        e_s=work.e_s,
        e_i=work.e_i,
        szv_i=work.szv_i,
        a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        kn=2,
        config=cfg,
    )
    snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
        vt_i=agg.vt_i,
        vt_s=agg.vt_s,
        vt_ip=agg.vt_ip,
        vt_il=agg.vt_il,
        snwice_mass_prev=work.snwice_mass,
        tmask=tmask,
        config=cfg,
        is_init=False,
    )
    return work._replace(
        a_i=eqv.a_i,
        v_s=eqv.v_s,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        snwice_fmass=snwice_fmass,
        equivalent=eqv,
        aggregated=agg,
    )


_PRATHER_3D_FAMILIES: tuple[tuple[str, str], ...] = (
    ("ice", "ice"),
    ("sn", "sn"),
    ("a", "a"),
    ("age", "age"),
    ("sal", "sal"),
    ("ap", "ap"),
    ("vp", "vp"),
    ("vl", "vl"),
)
_PRATHER_4D_FAMILIES: tuple[tuple[str, str, str], ...] = (
    ("c0", "c0", "nlay_s"),
    ("e", "e", "nlay_i"),
    ("si", "si", "nlay_i"),
)


def si3_rst_to_netcdf_dict(
    state: SI3State,
    kt: int = 1,
    config: SI3ORCA2StateConfig | None = None,
    *,
    write_prather: bool | None = None,
) -> dict[str, np.ndarray]:
    """Serialize `SI3State` to NEMO's exact `ice_rst_write` + `update_rst` (+ `adv_pra_rst` when active) dict.

    Statement order matches:
    1. `update_rst('WRITE', kt)` (`src/ICE/iceupdate.F90:474-481`)
    2. `adv_pra_rst('WRITE', kt)` (`src/ICE/icedyn_adv_pra.F90:1383-1503`), called
       inside `ice_dyn_adv_pra` (`src/ICE/icedyn_adv_pra.F90:487`) only when
       `ln_icedyn` and `ln_dynadv_pra` are active (`src/ICE/icestp.F90:167`,
       `src/ICE/icedyn_adv.F90:141`).
    3. `ice_rst_write(kt)` (`src/ICE/icerst.F90:137-182`)

    Note: NetCDF variables in NEMO's IOM (`iom_nf90`) are stored in `(t, cat, y, x)`
    C-order (corresponding to Fortran `(x, y, cat, t)`). Here each spatial array is
    stored in Fortran-logical `(jpi, jpj, ...)` order so direct array comparisons are
    shape-preserving; `write_si3_netcdf_restart` / `read_si3_netcdf_restart` handle
    both native NEMO `(y, x)` and `(x, y)` conventions.
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    do_write_prather = (
        bool(cfg.ln_icedyn and cfg.ln_dynadv_pra)
        if write_prather is None
        else bool(write_prather)
    )
    iter_val = float(kt + cfg.nn_fsbc - 1)
    out: dict[str, np.ndarray] = {}

    # 1. update_rst ('WRITE', kt)
    out["snwice_mass"] = np.asarray(state.snwice_mass, dtype=np.float64)
    out["snwice_mass_b"] = np.asarray(state.snwice_mass_b, dtype=np.float64)

    # 2. adv_pra_rst ('WRITE', kt) — written in NEMO only when ice_dyn_adv_pra runs
    if do_write_prather:
        for attr_name, suffix in _PRATHER_3D_FAMILIES:
            if suffix == "sal" and cfg.nn_icesal == 4:
                continue
            if suffix in ("ap", "vp", "vl") and not (cfg.ln_pnd_LEV or cfg.ln_pnd_TOPO):
                continue
            fam: SI3PratherMoments = getattr(state.prather, attr_name)
            for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
                out[f"{prefix}{suffix}"] = np.asarray(
                    getattr(fam, prefix), dtype=np.float64
                )

        for attr_name, suffix, nlay_attr in _PRATHER_4D_FAMILIES:
            if suffix == "si" and cfg.nn_icesal != 4:
                continue
            nlay = getattr(cfg, nlay_attr)
            fam = getattr(state.prather, attr_name)
            for jk in range(1, nlay + 1):
                tag = f"_l{jk:02d}"
                for prefix in ("sx", "sy", "sxx", "syy", "sxy"):
                    arr4 = np.asarray(getattr(fam, prefix), dtype=np.float64)
                    out[f"{prefix}{suffix}{tag}"] = arr4[:, :, jk - 1, :]

    # 3. ice_rst_write(kt)
    out["nn_fsbc"] = np.asarray(float(cfg.nn_fsbc), dtype=np.float64)
    out["kt_ice"] = np.asarray(iter_val, dtype=np.float64)
    out["v_i"] = np.asarray(state.v_i, dtype=np.float64)
    out["v_s"] = np.asarray(state.v_s, dtype=np.float64)
    out["a_i"] = np.asarray(state.a_i, dtype=np.float64)
    out["t_su"] = np.asarray(state.t_su, dtype=np.float64)
    out["u_ice"] = np.asarray(state.u_ice, dtype=np.float64)
    out["v_ice"] = np.asarray(state.v_ice, dtype=np.float64)
    out["oa_i"] = np.asarray(state.oa_i, dtype=np.float64)
    out["a_ip"] = np.asarray(state.a_ip, dtype=np.float64)
    out["v_ip"] = np.asarray(state.v_ip, dtype=np.float64)
    out["v_il"] = np.asarray(state.v_il, dtype=np.float64)

    e_s_np = np.asarray(state.e_s, dtype=np.float64)
    for jk in range(1, cfg.nlay_s + 1):
        out[f"e_s_l{jk:02d}"] = e_s_np[:, :, jk - 1, :]

    e_i_np = np.asarray(state.e_i, dtype=np.float64)
    for jk in range(1, cfg.nlay_i + 1):
        out[f"e_i_l{jk:02d}"] = e_i_np[:, :, jk - 1, :]

    out["sv_i"] = np.asarray(state.sv_i, dtype=np.float64)
    szv_i_np = np.asarray(state.szv_i, dtype=np.float64)
    for jk in range(1, cfg.nlay_i + 1):
        out[f"szv_i_l{jk:02d}"] = szv_i_np[:, :, jk - 1, :]

    if cfg.ln_cpl:
        out["cnd_ice"] = np.asarray(state.cnd_ice, dtype=np.float64)
        out["t1_ice"] = np.asarray(state.t1_ice, dtype=np.float64)

    return out


def si3_rst_from_netcdf_dict(
    rst: Mapping[str, Any],
    tmask: Field,
    *,
    sss_m: Field | float | None = None,
    config: SI3ORCA2StateConfig | None = None,
    apply_halos: bool = True,
    ln_rstart: bool = True,
) -> SI3State:
    """Reconstruct `SI3State` in NEMO's exact `ice_init` restart-read sequence.

    Follows NEMO's statement order in `src/ICE/icestp.F90:281-300`:
    1. `diag_set0` (`src/ICE/icestp.F90:281, 488`): `t_si(A2D(0), 1:jpl) = rt0`
    2. `ice_itd_init` (`src/ICE/iceitd.F90:715-780`)
    3. `ice_rst_read` (`src/ICE/icerst.F90:200-378`): refuses if `v_i` is absent or
       if the number of snow + ice enthalpy layers in the restart does not equal
       `nlay_s + nlay_i` (`:241-255, 369`); uses `sv_i * r1_nlay_i` when `szv_i_l01`
       is absent (`:351`).
    4. `ice_var_glo2eqv(1)` (`src/ICE/icestp.F90:295`)
    5. `ice_var_agg(1)` (`src/ICE/icestp.F90:296`)
    6. `adv_pra_rst('READ')` (`src/ICE/icedyn_adv_pra.F90:546, 1203-1381`): reads
       Prather moments iff `ln_rstart` and `sxice` + `nlay_s + nlay_i` layer
       moments are present; zeroes them otherwise.
    7. `update_rst('READ')` (`src/ICE/iceupdate.F90:453-473`)
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    consts = cfg.ice_constants
    sr = nemo_source_round

    if "v_i" not in rst:
        raise ValueError(
            "ice_rst_read: v_i is needed to read the restart file (src/ICE/icerst.F90:369)"
        )
    # iom_nf90 writes no halo points: file dimensions are Ni_0 x Nj_0, while
    # SI3 allocates jpi x jpj arrays.  iom_get reads the payload into
    # Nis0:Nie0,Njs0:Nje0 and then calls lbc_lnk (iom.F90:1347-1432).  Accept
    # either that real NEMO layout or an already-expanded in-memory mapping,
    # but refuse every other shape instead of guessing an orientation.
    tmask_arr = jnp.asarray(tmask, dtype=jnp.float64)
    if tmask_arr.ndim != 2:
        raise ValueError(f"ice_rst_read: tmask must be 2-D, got {tmask_arr.shape}")
    full_xy = tuple(int(n) for n in tmask_arr.shape)
    halo = int(cfg.nn_hls)
    file_xy = (
        full_xy[0] - 2 * halo,
        full_xy[1] - 2 * halo,
    ) if halo > 0 else full_xy

    def _expand_file_halos(name: str, value: Any) -> jax.Array:
        arr = jnp.asarray(value, dtype=jnp.float64)
        if arr.ndim < 2:
            return arr
        two_dimensional = {
            "u_ice",
            "v_ice",
            "snwice_mass",
            "snwice_mass_b",
        }
        three_dimensional = {
            "v_i",
            "v_s",
            "sv_i",
            "a_i",
            "t_su",
            "oa_i",
            "a_ip",
            "v_ip",
            "v_il",
            "cnd_ice",
            "t1_ice",
        }
        is_layer_field = bool(
            re.fullmatch(r"(?:e_s|e_i|szv_i)_l\d{2}", name)
        )
        is_prather_field = bool(
            re.fullmatch(r"s(?:x|y|xx|yy|xy)[A-Za-z0-9_]*", name)
        )
        expected_tail: tuple[int, ...] | None
        if name in two_dimensional:
            expected_tail = ()
        elif name in three_dimensional or is_layer_field or is_prather_field:
            expected_tail = (cfg.jpl,)
        else:
            expected_tail = None
        if expected_tail is not None and tuple(int(n) for n in arr.shape[2:]) != expected_tail:
            expected_shape = ("x", "y") + expected_tail
            raise ValueError(
                f"ice_rst_read: {name} rank/shape {tuple(arr.shape)} does not match "
                f"NEMO field shape {expected_shape}"
            )
        spatial = tuple(int(n) for n in arr.shape[:2])
        if spatial == full_xy:
            return arr
        if spatial != file_xy or halo <= 0:
            raise ValueError(
                f"ice_rst_read: {name} spatial shape {spatial} is neither NEMO's "
                f"halo-free {file_xy} nor allocated {full_xy}"
            )
        expanded = jnp.zeros(full_xy + arr.shape[2:], dtype=arr.dtype)
        return expanded.at[
            halo : full_xy[0] - halo,
            halo : full_xy[1] - halo,
            ...,
        ].set(arr)

    rst_full = {
        name: (_expand_file_halos(name, value) if np.asarray(value).ndim >= 2 else value)
        for name, value in rst.items()
    }
    v_i = _expand_file_halos("v_i", rst["v_i"])
    jpi, jpj, jpl = v_i.shape
    if jpl != cfg.jpl:
        raise ValueError(
            f"ice_rst_read: number of categories in restart ({jpl}) != config.jpl ({cfg.jpl}) "
            "(src/ICE/icerst.F90:235-239)"
        )

    # Layer count check (`src/ICE/icerst.F90:241-255`): NEMO counts e_s_l01..99 and
    # e_i_l01..99 and calls ctl_stop if id_nlay != (nlay_s + nlay_i).
    id_nlay_s = sum(1 for jk in range(1, 100) if f"e_s_l{jk:02d}" in rst)
    id_nlay_i = sum(1 for jk in range(1, 100) if f"e_i_l{jk:02d}" in rst)
    id_nlay = id_nlay_s + id_nlay_i
    expected_nlay = cfg.nlay_s + cfg.nlay_i
    missing_e_s = [
        f"e_s_l{jk:02d}"
        for jk in range(1, cfg.nlay_s + 1)
        if f"e_s_l{jk:02d}" not in rst
    ]
    missing_e_i = [
        f"e_i_l{jk:02d}"
        for jk in range(1, cfg.nlay_i + 1)
        if f"e_i_l{jk:02d}" not in rst
    ]
    if id_nlay != expected_nlay or missing_e_s or missing_e_i:
        raise ValueError(
            f"ice_rst_read: number of layers in restart ({id_nlay}: "
            f"nlay_s={id_nlay_s}, nlay_i={id_nlay_i}) != nlay_s + nlay_i "
            f"({expected_nlay}: {cfg.nlay_s} + {cfg.nlay_i}); missing={missing_e_s + missing_e_i} "
            "(src/ICE/icerst.F90:241-255)"
        )

    v_s = jnp.asarray(rst_full["v_s"], dtype=jnp.float64)
    sv_i = jnp.asarray(rst_full["sv_i"], dtype=jnp.float64)
    a_i = jnp.asarray(rst_full["a_i"], dtype=jnp.float64)
    t_su = jnp.asarray(rst_full["t_su"], dtype=jnp.float64)
    u_ice = jnp.asarray(rst_full["u_ice"], dtype=jnp.float64)
    v_ice = jnp.asarray(rst_full["v_ice"], dtype=jnp.float64)

    if "oa_i" in rst_full:
        oa_i = jnp.asarray(rst_full["oa_i"], dtype=jnp.float64)
    else:
        oa_i = jnp.zeros_like(v_i)

    if "a_ip" in rst_full:
        a_ip = jnp.asarray(rst_full["a_ip"], dtype=jnp.float64)
        v_ip = jnp.asarray(rst_full["v_ip"], dtype=jnp.float64)
    else:
        a_ip = jnp.zeros_like(v_i)
        v_ip = jnp.zeros_like(v_i)

    # icerst.F90:318-325 tests v_il independently of the melt-pond fields and
    # namelist switches: if the restart variable exists, NEMO reads it.
    if "v_il" in rst_full:
        v_il = jnp.asarray(rst_full["v_il"], dtype=jnp.float64)
    else:
        v_il = jnp.zeros_like(v_i)

    e_s = jnp.stack(
        [
            jnp.asarray(rst_full[f"e_s_l{jk:02d}"], dtype=jnp.float64)
            for jk in range(1, cfg.nlay_s + 1)
        ],
        axis=2,
    )
    e_i = jnp.stack(
        [
            jnp.asarray(rst_full[f"e_i_l{jk:02d}"], dtype=jnp.float64)
            for jk in range(1, cfg.nlay_i + 1)
        ],
        axis=2,
    )

    if "szv_i_l01" in rst:
        szv_i = jnp.stack(
            [
                jnp.asarray(rst_full[f"szv_i_l{jk:02d}"], dtype=jnp.float64)
                for jk in range(1, cfg.nlay_i + 1)
            ],
            axis=2,
        )
    else:
        # `src/ICE/icerst.F90:351` and `src/ICE/icestp.F90:382`:
        # szv_i(:,:,jk,:) = sv_i(:,:,:) * r1_nlay_i where r1_nlay_i = 1._wp / REAL(nlay_i, wp)
        r1_nlay_i = sr(jnp.asarray(1.0 / float(cfg.nlay_i), dtype=jnp.float64))
        szv_i = jnp.broadcast_to(
            sr(sv_i[..., None, :] * r1_nlay_i),
            (jpi, jpj, cfg.nlay_i, jpl),
        )

    cnd_ice = (
        jnp.asarray(rst_full["cnd_ice"], dtype=jnp.float64)
        if ("cnd_ice" in rst_full and cfg.ln_cpl)
        else jnp.zeros_like(v_i)
    )
    t1_ice = (
        jnp.asarray(rst_full["t1_ice"], dtype=jnp.float64)
        if ("t1_ice" in rst_full and cfg.ln_cpl)
        # With ln_cpl=F, icerst.F90:327-338 never reads or initializes t1_ice.
        # Keep a deterministic carrier value, but callers/gates must treat it
        # as undefined (it is not a restart-certified field in this branch).
        else jnp.zeros_like(v_i)
    )
    # `diag_set0` (`src/ICE/icestp.F90:281, 488`): t_si(A2D(0), 1:jpl) = rt0
    interior_3d = _interior_mask_2d(jpi, jpj, cfg.nn_hls)[:, :, None]
    t_si = jnp.where(interior_3d, jnp.full_like(v_i, consts.T0), 0.0)

    # Prather moments (`adv_pra_rst('READ')`, src/ICE/icedyn_adv_pra.F90:546, 1217-1381)
    pr_zeros = SI3AllPratherMoments.zeros(
        jpi, jpj, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s
    )
    id0_pra = sum(1 for jk in range(1, 100) if f"sxc0_l{jk:02d}" in rst) + sum(
        1 for jk in range(1, 100) if f"sxe_l{jk:02d}" in rst
    )
    if ln_rstart and ("sxice" in rst) and (id0_pra == expected_nlay):
        fam_dict: dict[str, SI3PratherMoments] = {}
        for attr_name, suffix in _PRATHER_3D_FAMILIES:
            if f"sx{suffix}" in rst:
                fam_dict[attr_name] = SI3PratherMoments(
                    sx=jnp.asarray(rst_full[f"sx{suffix}"], dtype=jnp.float64),
                    sy=jnp.asarray(rst_full[f"sy{suffix}"], dtype=jnp.float64),
                    sxx=jnp.asarray(rst_full[f"sxx{suffix}"], dtype=jnp.float64),
                    syy=jnp.asarray(rst_full[f"syy{suffix}"], dtype=jnp.float64),
                    sxy=jnp.asarray(rst_full[f"sxy{suffix}"], dtype=jnp.float64),
                )
            else:
                fam_dict[attr_name] = getattr(pr_zeros, attr_name)

        for attr_name, suffix, nlay_attr in _PRATHER_4D_FAMILIES:
            nlay = getattr(cfg, nlay_attr)
            if f"sx{suffix}_l01" in rst:
                fam_dict[attr_name] = SI3PratherMoments(
                    sx=jnp.stack(
                        [
                            jnp.asarray(rst_full[f"sx{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    sy=jnp.stack(
                        [
                            jnp.asarray(rst_full[f"sy{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    sxx=jnp.stack(
                        [
                            jnp.asarray(rst_full[f"sxx{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    syy=jnp.stack(
                        [
                            jnp.asarray(rst_full[f"syy{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    sxy=jnp.stack(
                        [
                            jnp.asarray(rst_full[f"sxy{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                )
            else:
                fam_dict[attr_name] = getattr(pr_zeros, attr_name)
        prather = SI3AllPratherMoments(**fam_dict)
    else:
        prather = pr_zeros

    bounds = si3_hfn_category_bounds(
        jpl=cfg.jpl,
        rn_himean=cfg.rn_himean,
        opt_hips=cfg.opt_hips,
        rn_himax=cfg.rn_himax,
    )
    base_state = SI3State(
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        a_i=a_i,
        t_su=t_su,
        u_ice=u_ice,
        v_ice=v_ice,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        cnd_ice=cnd_ice,
        t1_ice=t1_ice,
        t_si=t_si,
        snwice_mass=jnp.zeros((jpi, jpj), dtype=jnp.float64),
        snwice_mass_b=jnp.zeros((jpi, jpj), dtype=jnp.float64),
        snwice_fmass=jnp.zeros((jpi, jpj), dtype=jnp.float64),
        prather=prather,
        bounds=bounds,
        equivalent=SI3EquivalentVars.zeros(
            jpi, jpj, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s
        ),
        aggregated=SI3AggregatedVars.zeros(jpi, jpj),
    )

    if apply_halos:
        base_state = si3_apply_state_halos_orca2(base_state, config=cfg)

    eqv = si3_var_glo2eqv(
        v_i=base_state.v_i,
        v_s=base_state.v_s,
        sv_i=base_state.sv_i,
        a_i=base_state.a_i,
        oa_i=base_state.oa_i,
        a_ip=base_state.a_ip,
        v_ip=base_state.v_ip,
        v_il=base_state.v_il,
        e_s=base_state.e_s,
        e_i=base_state.e_i,
        szv_i=base_state.szv_i,
        kn=1,
        hi_max_top=bounds.hi_max[-1],
        sss_m=sss_m,
        config=cfg,
    )
    agg = si3_var_agg(
        v_i=base_state.v_i,
        v_s=eqv.v_s,
        sv_i=base_state.sv_i,
        a_i=eqv.a_i,
        oa_i=base_state.oa_i,
        a_ip=base_state.a_ip,
        v_ip=base_state.v_ip,
        v_il=base_state.v_il,
        t_su=base_state.t_su,
        t_si=base_state.t_si,
        e_s=base_state.e_s,
        e_i=base_state.e_i,
        szv_i=base_state.szv_i,
        a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        kn=1,
        config=cfg,
    )
    if ln_rstart and "snwice_mass" in rst_full and "snwice_mass_b" in rst_full:
        snwice_mass = jnp.asarray(rst_full["snwice_mass"], dtype=jnp.float64)
        snwice_mass_b = jnp.asarray(rst_full["snwice_mass_b"], dtype=jnp.float64)
        snwice_fmass = jnp.zeros_like(snwice_mass)
        if apply_halos:
            # update_rst reads these after the earlier ice_rst_read halo pass;
            # iom_get itself applies lbc_lnk to each newly read 2-D field.
            snwice_mass = si3_lbc_lnk_orca2(
                snwice_mass, "T", 1.0, nn_hls=cfg.nn_hls, ldfull=True
            )
            snwice_mass_b = si3_lbc_lnk_orca2(
                snwice_mass_b, "T", 1.0, nn_hls=cfg.nn_hls, ldfull=True
            )
    else:
        snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
            vt_i=agg.vt_i,
            vt_s=agg.vt_s,
            vt_ip=agg.vt_ip,
            vt_il=agg.vt_il,
            snwice_mass_prev=jnp.zeros((jpi, jpj), dtype=jnp.float64),
            tmask=tmask_arr,
            config=cfg,
            is_init=True,
        )
    return base_state._replace(
        a_i=eqv.a_i,
        v_s=eqv.v_s,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        snwice_fmass=snwice_fmass,
        equivalent=eqv,
        aggregated=agg,
    )


def write_si3_netcdf_restart(
    path: str | Path,
    state: SI3State,
    *,
    kt: int = 1,
    config: SI3ORCA2StateConfig | None = None,
    write_prather: bool | None = None,
) -> Path:
    """Write a NEMO-compatible `*_restart_ice.nc` file using `netCDF4`."""
    import netCDF4

    cfg = config if config is not None else SI3ORCA2StateConfig()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rst_dict = si3_rst_to_netcdf_dict(
        state, kt=kt, config=cfg, write_prather=write_prather
    )
    jpi, jpj, jpl = state.v_i.shape
    halo = int(cfg.nn_hls)
    if halo > 0:
        if 2 * halo >= jpi or 2 * halo >= jpj:
            raise ValueError(
                f"Cannot strip nn_hls={halo} from restart grid {(jpi, jpj)}"
            )
        ni_0, nj_0 = jpi - 2 * halo, jpj - 2 * halo
    else:
        ni_0, nj_0 = jpi, jpj

    def _without_halos(arr: np.ndarray) -> np.ndarray:
        if arr.ndim < 2 or halo == 0:
            return arr
        if arr.shape[:2] != (jpi, jpj):
            raise ValueError(
                f"Restart field spatial shape {arr.shape[:2]} != state grid {(jpi, jpj)}"
            )
        return arr[halo : jpi - halo, halo : jpj - halo, ...]

    with netCDF4.Dataset(out_path, "w", format="NETCDF4_CLASSIC") as ds:
        # iom_nf90.F90:135-154,729-760 writes Ni_0 x Nj_0 payloads and records
        # zero file halos, even though the live Fortran arrays are jpi x jpj.
        ds.createDimension("x", ni_0)
        ds.createDimension("y", nj_0)
        ds.createDimension("numcat", jpl)
        ds.createDimension("time_counter", None)
        ds.setncattr("DOMAIN_size_local", np.asarray((ni_0, nj_0), dtype=np.int32))
        ds.setncattr("DOMAIN_halo_size_start", np.asarray((0, 0), dtype=np.int32))
        ds.setncattr("DOMAIN_halo_size_end", np.asarray((0, 0), dtype=np.int32))

        for name, val in rst_dict.items():
            arr = _without_halos(np.asarray(val, dtype=np.float64))
            if arr.ndim == 0:
                var = ds.createVariable(name, "f8", ())
                var[...] = arr
            elif arr.ndim == 2:
                # NEMO NetCDF C-order: (time_counter, y, x)
                var = ds.createVariable(name, "f8", ("time_counter", "y", "x"))
                var[0, :, :] = arr.T
            elif arr.ndim == 3:
                # NEMO NetCDF C-order: (time_counter, numcat, y, x)
                var = ds.createVariable(
                    name, "f8", ("time_counter", "numcat", "y", "x")
                )
                var[0, :, :, :] = np.transpose(arr, (2, 1, 0))
            else:
                raise ValueError(f"Unexpected restart rank for {name}: {arr.shape}")
    return out_path


def read_si3_netcdf_restart(
    path: str | Path,
    tmask: Field,
    *,
    sss_m: Field | float | None = None,
    config: SI3ORCA2StateConfig | None = None,
    apply_halos: bool = True,
    ln_rstart: bool = True,
) -> SI3State:
    """Read a NEMO `*_restart_ice.nc` file and reconstruct `SI3State`."""
    import netCDF4

    cfg = config if config is not None else SI3ORCA2StateConfig()
    rst_dict: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(Path(path), "r") as ds:
        if "x" not in ds.dimensions or "y" not in ds.dimensions:
            raise ValueError("NEMO ice restart is missing x/y dimensions")
        file_xy = (len(ds.dimensions["x"]), len(ds.dimensions["y"]))
        full_xy = tuple(int(n) for n in np.asarray(tmask).shape)
        halo = int(cfg.nn_hls)
        expected_no_halo = (
            full_xy[0] - 2 * halo,
            full_xy[1] - 2 * halo,
        ) if halo > 0 else full_xy
        if file_xy not in (expected_no_halo, full_xy):
            raise ValueError(
                f"NEMO ice restart x/y={file_xy} does not match halo-free "
                f"{expected_no_halo} or allocated {full_xy}"
            )
        for name, var in ds.variables.items():
            raw = np.asarray(var[...], dtype=np.float64)
            dims = tuple(var.dimensions)
            if "time_counter" in dims and dims[0] == "time_counter":
                raw = raw[0]
                dims = dims[1:]
            if raw.ndim == 0:
                rst_dict[name] = raw
            elif raw.ndim == 2:
                if dims == ("y", "x") or (
                    dims != ("x", "y") and raw.shape[0] != tmask.shape[0]
                ):
                    rst_dict[name] = raw.T
                else:
                    rst_dict[name] = raw
            elif raw.ndim == 3:
                if dims == ("numcat", "y", "x") or raw.shape[0] == cfg.jpl:
                    rst_dict[name] = np.transpose(raw, (2, 1, 0))
                else:
                    rst_dict[name] = raw
    return si3_rst_from_netcdf_dict(
        rst_dict,
        tmask=tmask,
        sss_m=sss_m,
        config=cfg,
        apply_halos=apply_halos,
        ln_rstart=ln_rstart,
    )


# ==============================================================================
# Rung O3: Full-state ORCA2 SI3 Prather advection (`icedyn_adv_pra.f90`,
# `icedyn_adv.f90`, `icedyn.f90`, `icestp.f90`).
# ==============================================================================

from legoesm.ice.transport import (  # noqa: E402
    si3_active_mask_2d,
    si3_adv_x,
    si3_adv_y,
    si3_hbig_pra,
    si3_hsnow_pra,
    si3_icemax2d_pra,
    si3_icemax3d_pra,
    si3_var_zapneg,
    si3_var_zapsmall,
)

# Fixed NEMO 5.0.2 SI3 Prather advection literals (`icedyn_adv_pra.f90:138-141`,
# `icedyn_adv.f90:92`, `icedyn.f90:155-171`).
_SI3_PRA_ZERO = 0.0
_SI3_PRA_ONE = 1.0
_SI3_PRA_ICE_PRESENCE = 1.0e-10
_SI3_PRA_CFL_TWO_CYCLE_THRESHOLD = 0.5
_SI3_PRA_CFL_THREE_CYCLE_THRESHOLD = 1.5
_si3_active_mask_2d = si3_active_mask_2d


class SI3ORCA2AdvectionGrid(NamedTuple):
    """ORCA2 horizontal C-grid metrics and surface masks for SI3 Prather advection.

    Source citations:
    - `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:133-150,233-261,370-397,440-443`
    - `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:222-223`
    """

    e1u: Field
    e2v: Field
    r1_e1u: Field
    r1_e2v: Field
    e2u: Field
    e1v: Field
    e1e2t: Field
    r1_e1e2t: Field
    tmask: Field
    umask: Field
    vmask: Field

    @classmethod
    def from_metrics(
        cls,
        *,
        e1u: Field,
        e2v: Field,
        e2u: Field,
        e1v: Field,
        e1e2t: Field,
        tmask: Field,
        umask: Field | None = None,
        vmask: Field | None = None,
        r1_e1u: Field | None = None,
        r1_e2v: Field | None = None,
        r1_e1e2t: Field | None = None,
    ) -> "SI3ORCA2AdvectionGrid":
        """Construct `SI3ORCA2AdvectionGrid` with source-rounded reciprocal metrics."""
        sr = nemo_source_round
        e1u_f = jnp.asarray(e1u, dtype=jnp.float64)
        e2v_f = jnp.asarray(e2v, dtype=jnp.float64)
        e2u_f = jnp.asarray(e2u, dtype=jnp.float64)
        e1v_f = jnp.asarray(e1v, dtype=jnp.float64)
        e1e2t_f = jnp.asarray(e1e2t, dtype=jnp.float64)
        tmask_f = jnp.asarray(tmask, dtype=jnp.float64)
        umask_f = (
            tmask_f if umask is None else jnp.asarray(umask, dtype=jnp.float64)
        )
        vmask_f = (
            tmask_f if vmask is None else jnp.asarray(vmask, dtype=jnp.float64)
        )
        r1_e1u_f = (
            sr(jnp.reciprocal(e1u_f))
            if r1_e1u is None
            else jnp.asarray(r1_e1u, dtype=jnp.float64)
        )
        r1_e2v_f = (
            sr(jnp.reciprocal(e2v_f))
            if r1_e2v is None
            else jnp.asarray(r1_e2v, dtype=jnp.float64)
        )
        r1_e1e2t_f = (
            sr(jnp.reciprocal(e1e2t_f))
            if r1_e1e2t is None
            else jnp.asarray(r1_e1e2t, dtype=jnp.float64)
        )
        return cls(
            e1u=e1u_f,
            e2v=e2v_f,
            r1_e1u=r1_e1u_f,
            r1_e2v=r1_e2v_f,
            e2u=e2u_f,
            e1v=e1v_f,
            e1e2t=e1e2t_f,
            r1_e1e2t=r1_e1e2t_f,
            tmask=tmask_f,
            umask=umask_f,
            vmask=vmask_f,
        )


class SI3AdvectionDiagnostics(NamedTuple):
    """Diagnostics produced by one ORCA2 SI3 Prather advection step (`icestp.f90`, `icedyn_adv.f90`, `icedyn_adv_pra.f90`)."""

    diag_adv_mass: Field
    diag_adv_salt: Field
    diag_adv_heat: Field
    wfx_res: Field
    hfx_res: Field
    sfx_res: Field
    diag_trp_vi: Field
    diag_trp_vs: Field
    diag_trp_ei: Field
    diag_trp_es: Field
    diag_trp_sv: Field
    diag_heat: Field
    diag_sice: Field
    diag_vice: Field
    diag_vsnw: Field
    diag_vpnd: Field
    diag_aice: Field
    ato_i_post_adv: Field
    ato_i_post_dyn: Field
    zcfl: Field
    icycle: Field


def _moments_to_tuple(
    fam: SI3PratherMoments,
) -> tuple[Field, Field, Field, Field, Field]:
    return fam.sx, fam.sy, fam.sxx, fam.syy, fam.sxy


def _tuple_to_moments(
    tup: tuple[Field, Field, Field, Field, Field],
) -> SI3PratherMoments:
    return SI3PratherMoments(
        sx=tup[0],
        sy=tup[1],
        sxx=tup[2],
        syy=tup[3],
        sxy=tup[4],
    )


def si3_dyn_adv_pra_orca2(
    kt: int,
    u_ice: Field,
    v_ice: Field,
    h_i: Field,
    h_s: Field,
    h_ip: Field,
    ato_i: Field,
    v_i: Field,
    v_s: Field,
    sv_i: Field,
    oa_i: Field,
    a_i: Field,
    a_ip: Field,
    v_ip: Field,
    v_il: Field,
    e_s: Field,
    e_i: Field,
    szv_i: Field,
    prather: SI3AllPratherMoments,
    grid: SI3ORCA2AdvectionGrid,
    *,
    config: SI3ORCA2StateConfig | None = None,
    hi_max_top: float | Field = _ORCA2_RN_HIMAX_M,
    subcycles: int | None = None,
    wfx_res: Field | None = None,
    hfx_res: Field | None = None,
    sfx_res: Field | None = None,
    diag_adv_mass: Field | None = None,
    diag_adv_salt: Field | None = None,
    diag_adv_heat: Field | None = None,
    _plant_invert_sweep_order: bool = False,
    _plant_perturb_hbig_cap: bool = False,
    _plant_omit_subcycle_lbc: bool = False,
    _plant_omit_donor_readjust: bool = False,
    _plant_corrupt_receiver_flux: bool = False,
    _plant_wrong_nfd_sign: bool = False,
) -> tuple[
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    SI3AllPratherMoments,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
    Field,
]:
    """Execute `ice_dyn_adv_pra` on ORCA2 (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn_adv_pra.f90:70-510`).

    Transcribes:
    - CFL evaluation and `icycle in {1, 2, 3}` (`:133-138`)
    - Face transport `zudy = pu_ice * e2u`, `zvdx = pv_ice * e1v` (`:149-150`)
    - Subcycle loop `DO jt = 1, icycle` (`:156-498`):
      - `zati1` (`:159`), `ihls` (`:161-163`)
      - 9-point maxima `icemax2D_pra` (`zh_i, zh_s, zh_ip, zs_i`) and
        `icemax3D_pra` (`ze_i, ze_s, zsz_i`) (`:172-213`)
      - Pre-advection `zdiag_adv_mass, zdiag_adv_heat, zdiag_adv_salt` (`:216-229`)
      - Extensive packing `z0* = p* * e1e2t` on `A2D(ihls+1)` (`:233-261`)
      - Alternating sweep `MOD((kt-1)/nn_fsbc, 2) == MOD(jt-1, 2)` (`:267-367`)
      - Intensive recovery `p* = z0* * r1_e1e2t * tmask` on `A2D(ihls)` (`:370-397`)
      - Post-advection `diag_adv_mass, diag_adv_heat, diag_adv_salt` (`:400-417`)
      - `Hbig_pra` (`:421-422`) and `Hsnow_pra` (`:425`) per category `jl = 1..jpl`
      - `ice_var_zapneg` (`:435`)
      - Open-water `pato_i` update (`:438-443`)
      - `lbc_lnk(..., ldfull = .TRUE.)` on all transported fields, their five
        Prather moments (`psgn = -1` for `sx, sy`; `+1` for `sxx, syy, sxy`),
        `pato_i`, and `wfx_res, hfx_res, sfx_res` (`:450-494`).
    """
    cfg = config if config is not None else SI3ORCA2StateConfig(ln_icedyn=True)
    if kt < 1 or cfg.nn_fsbc < 1:
        raise ValueError("si3_dyn_adv_pra_orca2 requires kt >= 1 and nn_fsbc >= 1")
    sr = nemo_source_round
    consts = cfg.ice_constants
    jpi, jpj, jpl = v_i.shape
    dtype = v_i.dtype
    nn_hls = int(cfg.nn_hls)
    interior_2d = _si3_active_mask_2d(jpi, jpj, nn_hls, 0)

    rDt_ice = jnp.asarray(cfg.rDt_ice, dtype=dtype)
    r1_Dt_ice = sr(_SI3_PRA_ONE / rDt_ice)
    rhoi_f = jnp.asarray(consts.rho_ice, dtype=dtype)
    rhos_f = jnp.asarray(consts.rho_snow, dtype=dtype)
    rhow_f = jnp.asarray(canonical_constants.rho_water, dtype=dtype)
    epsi10 = jnp.asarray(_SI3_PRA_ICE_PRESENCE, dtype=dtype)

    if wfx_res is None:
        wfx_res = jnp.zeros((jpi, jpj), dtype=dtype)
    if hfx_res is None:
        hfx_res = jnp.zeros((jpi, jpj), dtype=dtype)
    if sfx_res is None:
        sfx_res = jnp.zeros((jpi, jpj), dtype=dtype)
    if diag_adv_mass is None:
        diag_adv_mass = jnp.zeros((jpi, jpj), dtype=dtype)
    if diag_adv_salt is None:
        diag_adv_salt = jnp.zeros((jpi, jpj), dtype=dtype)
    if diag_adv_heat is None:
        diag_adv_heat = jnp.zeros((jpi, jpj), dtype=dtype)

    # CFL and subcycle count (`icedyn_adv_pra.f90:133-138`)
    cfl_u = jnp.max(sr(sr(jnp.abs(u_ice) * rDt_ice) * grid.r1_e1u))
    cfl_v = jnp.max(sr(sr(jnp.abs(v_ice) * rDt_ice) * grid.r1_e2v))
    zcfl = jnp.maximum(cfl_u, cfl_v)
    if subcycles is None:
        cfl_host = float(zcfl)
        icycle = (
            3
            if cfl_host > _SI3_PRA_CFL_THREE_CYCLE_THRESHOLD
            else 2
            if cfl_host > _SI3_PRA_CFL_TWO_CYCLE_THRESHOLD
            else 1
        )
    else:
        if subcycles not in (1, 2, 3):
            raise ValueError(f"subcycles must be 1, 2, or 3; got {subcycles}")
        icycle = int(subcycles)

    zdt = sr(rDt_ice / jnp.asarray(float(icycle), dtype=dtype))
    zudy = sr(u_ice * grid.e2u)
    zvdx = sr(v_ice * grid.e1v)

    pnd_active = bool(cfg.ln_pnd_LEV or cfg.ln_pnd_TOPO)
    sal4_active = cfg.nn_icesal == 4

    for jt in range(1, icycle + 1):
        # `zati1(:,:) = SUM( pa_i(Nis0:Nie0,Njs0:Nje0,:), dim=3 )` (`:159`)
        zati1 = jnp.where(
            interior_2d, fortran_sum_categories(a_i, axis=-1), _SI3_PRA_ZERO
        )
        ihls = 0 if icycle == 1 else max(0, nn_hls - jt)
        mask_ihls = _si3_active_mask_2d(jpi, jpj, nn_hls, ihls)
        mask_ihls_p1 = _si3_active_mask_2d(jpi, jpj, nn_hls, ihls + 1)

        # 9-point neighborhood maxima (`:172-213`)
        zhi_max = si3_icemax2d_pra(ihls, h_i, nn_hls=nn_hls)
        zhs_max = si3_icemax2d_pra(ihls, h_s, nn_hls=nn_hls)
        if pnd_active:
            zhip_max = si3_icemax2d_pra(ihls, h_ip, nn_hls=nn_hls)
        else:
            zhip_max = jnp.zeros_like(h_ip)

        vi_ge_eps = (v_i >= epsi10)[:, :, None, :]
        vi_safe_4d = jnp.where(vi_ge_eps, v_i[:, :, None, :], _SI3_PRA_ONE)
        ze_i = jnp.where(vi_ge_eps, sr(e_i / vi_safe_4d), _SI3_PRA_ZERO)

        vs_ge_eps = (v_s >= epsi10)[:, :, None, :]
        vs_safe_4d = jnp.where(vs_ge_eps, v_s[:, :, None, :], _SI3_PRA_ONE)
        ze_s = jnp.where(vs_ge_eps, sr(e_s / vs_safe_4d), _SI3_PRA_ZERO)

        zei_max = si3_icemax3d_pra(ihls, ze_i, nn_hls=nn_hls)
        zes_max = si3_icemax3d_pra(ihls, ze_s, nn_hls=nn_hls)

        if sal4_active:
            zsz_i = jnp.where(vi_ge_eps, sr(szv_i / vi_safe_4d), _SI3_PRA_ZERO)
            zszi_max = si3_icemax3d_pra(ihls, zsz_i, nn_hls=nn_hls)
            zsi_max = jnp.zeros_like(v_i)
        else:
            vi_ge_3d = v_i >= epsi10
            vi_safe_3d = jnp.where(vi_ge_3d, v_i, _SI3_PRA_ONE)
            zs_i = jnp.where(vi_ge_3d, sr(sv_i / vi_safe_3d), _SI3_PRA_ZERO)
            zsi_max = si3_icemax2d_pra(ihls, zs_i, nn_hls=nn_hls)
            zszi_max = jnp.zeros_like(szv_i)

        # Pre-advection diagnostics per category on `A2D(0)` (`:216-229`)
        zdiag_adv_mass = sr(
            sr(sr(sr(v_i * rhoi_f) + sr(v_s * rhos_f)) + sr(v_ip * rhow_f))
            + sr(v_il * rhow_f)
        )
        sum_ei_pre = fortran_sum_categories(e_i, axis=2)
        sum_es_pre = fortran_sum_categories(e_s, axis=2)
        zdiag_adv_heat = sr(sr(-sum_ei_pre) - sum_es_pre)
        if sal4_active:
            sum_szv_pre = fortran_sum_categories(szv_i, axis=2)
            zdiag_adv_salt = sr(sum_szv_pre * rhoi_f)
        else:
            zdiag_adv_salt = sr(sv_i * rhoi_f)

        # Pack extensives on `A2D(ihls+1)` (`:233-261`)
        m_p1_3d = mask_ihls_p1[..., None]
        m_p1_4d = mask_ihls_p1[:, :, None, None]
        e1e2t_3d = grid.e1e2t[..., None]
        e1e2t_4d = grid.e1e2t[:, :, None, None]

        zarea_init = jnp.where(mask_ihls_p1, grid.e1e2t, _SI3_PRA_ZERO)
        z0ice = jnp.where(m_p1_3d, sr(v_i * e1e2t_3d), _SI3_PRA_ZERO)
        z0snw = jnp.where(m_p1_3d, sr(v_s * e1e2t_3d), _SI3_PRA_ZERO)
        z0ai = jnp.where(m_p1_3d, sr(a_i * e1e2t_3d), _SI3_PRA_ZERO)
        z0oi = jnp.where(m_p1_3d, sr(oa_i * e1e2t_3d), _SI3_PRA_ZERO)
        z0es = jnp.where(m_p1_4d, sr(e_s * e1e2t_4d), _SI3_PRA_ZERO)
        z0ei = jnp.where(m_p1_4d, sr(e_i * e1e2t_4d), _SI3_PRA_ZERO)
        if sal4_active:
            z0si = jnp.where(m_p1_4d, sr(szv_i * e1e2t_4d), _SI3_PRA_ZERO)
            z0smi = jnp.zeros_like(sv_i)
        else:
            z0smi = jnp.where(m_p1_3d, sr(sv_i * e1e2t_3d), _SI3_PRA_ZERO)
            z0si = jnp.zeros_like(szv_i)
        if pnd_active:
            z0ap = jnp.where(m_p1_3d, sr(a_ip * e1e2t_3d), _SI3_PRA_ZERO)
            z0vp = jnp.where(m_p1_3d, sr(v_ip * e1e2t_3d), _SI3_PRA_ZERO)
            z0vl = jnp.where(m_p1_3d, sr(v_il * e1e2t_3d), _SI3_PRA_ZERO)
        else:
            z0ap = jnp.zeros_like(a_ip)
            z0vp = jnp.zeros_like(v_ip)
            z0vl = jnp.zeros_like(v_il)

        x_first = (((kt - 1) // cfg.nn_fsbc) % 2) == ((jt - 1) % 2)
        if _plant_invert_sweep_order:
            x_first = not x_first

        def _sweep_pair(
            content_in: Field,
            fam_in: SI3PratherMoments,
        ) -> tuple[Field, SI3PratherMoments]:
            m_tup = _moments_to_tuple(fam_in)
            if x_first:
                zarea_mid, c_mid, m_mid = si3_adv_x(
                    ihls,
                    zdt,
                    zudy,
                    _SI3_PRA_ONE,
                    zarea_init,
                    content_in,
                    m_tup,
                    e1e2t=grid.e1e2t,
                    tmask=grid.tmask,
                    nn_hls=nn_hls,
                    _plant_omit_donor_readjust=_plant_omit_donor_readjust,
                    _plant_corrupt_receiver_flux=_plant_corrupt_receiver_flux,
                )
                _, c_out, m_out = si3_adv_y(
                    ihls,
                    zdt,
                    zvdx,
                    _SI3_PRA_ZERO,
                    zarea_mid,
                    c_mid,
                    m_mid,
                    e1e2t=grid.e1e2t,
                    tmask=grid.tmask,
                    nn_hls=nn_hls,
                    _plant_omit_donor_readjust=_plant_omit_donor_readjust,
                    _plant_corrupt_receiver_flux=_plant_corrupt_receiver_flux,
                )
            else:
                zarea_mid, c_mid, m_mid = si3_adv_y(
                    ihls,
                    zdt,
                    zvdx,
                    _SI3_PRA_ONE,
                    zarea_init,
                    content_in,
                    m_tup,
                    e1e2t=grid.e1e2t,
                    tmask=grid.tmask,
                    nn_hls=nn_hls,
                    _plant_omit_donor_readjust=_plant_omit_donor_readjust,
                    _plant_corrupt_receiver_flux=_plant_corrupt_receiver_flux,
                )
                _, c_out, m_out = si3_adv_x(
                    ihls,
                    zdt,
                    zudy,
                    _SI3_PRA_ZERO,
                    zarea_mid,
                    c_mid,
                    m_mid,
                    e1e2t=grid.e1e2t,
                    tmask=grid.tmask,
                    nn_hls=nn_hls,
                    _plant_omit_donor_readjust=_plant_omit_donor_readjust,
                    _plant_corrupt_receiver_flux=_plant_corrupt_receiver_flux,
                )
            return c_out, _tuple_to_moments(m_out)

        z0ice, m_ice = _sweep_pair(z0ice, prather.ice)
        z0snw, m_sn = _sweep_pair(z0snw, prather.sn)
        z0ai, m_a = _sweep_pair(z0ai, prather.a)
        z0oi, m_age = _sweep_pair(z0oi, prather.age)
        z0es, m_c0 = _sweep_pair(z0es, prather.c0)
        z0ei, m_e = _sweep_pair(z0ei, prather.e)
        if sal4_active:
            z0si, m_si = _sweep_pair(z0si, prather.si)
            m_sal = prather.sal
        else:
            z0smi, m_sal = _sweep_pair(z0smi, prather.sal)
            m_si = prather.si
        if pnd_active:
            z0ap, m_ap = _sweep_pair(z0ap, prather.ap)
            z0vp, m_vp = _sweep_pair(z0vp, prather.vp)
            z0vl, m_vl = _sweep_pair(z0vl, prather.vl)
        else:
            m_ap = prather.ap
            m_vp = prather.vp
            m_vl = prather.vl

        # Recover intensives on `A2D(ihls)` (`:370-397`)
        m_ihls_3d = mask_ihls[..., None]
        m_ihls_4d = mask_ihls[:, :, None, None]
        r1_area_3d = grid.r1_e1e2t[..., None]
        r1_area_4d = grid.r1_e1e2t[:, :, None, None]
        tmask_3d = grid.tmask[..., None]
        tmask_4d = grid.tmask[:, :, None, None]

        v_i = jnp.where(m_ihls_3d, sr(sr(z0ice * r1_area_3d) * tmask_3d), v_i)
        v_s = jnp.where(m_ihls_3d, sr(sr(z0snw * r1_area_3d) * tmask_3d), v_s)
        oa_i = jnp.where(m_ihls_3d, sr(sr(z0oi * r1_area_3d) * tmask_3d), oa_i)
        a_i = jnp.where(m_ihls_3d, sr(sr(z0ai * r1_area_3d) * tmask_3d), a_i)
        e_s = jnp.where(m_ihls_4d, sr(sr(z0es * r1_area_4d) * tmask_4d), e_s)
        e_i = jnp.where(m_ihls_4d, sr(sr(z0ei * r1_area_4d) * tmask_4d), e_i)
        if sal4_active:
            szv_i = jnp.where(
                m_ihls_4d, sr(sr(z0si * r1_area_4d) * tmask_4d), szv_i
            )
        else:
            sv_i = jnp.where(
                m_ihls_3d, sr(sr(z0smi * r1_area_3d) * tmask_3d), sv_i
            )
        if pnd_active:
            a_ip = jnp.where(
                m_ihls_3d, sr(sr(z0ap * r1_area_3d) * tmask_3d), a_ip
            )
            v_ip = jnp.where(
                m_ihls_3d, sr(sr(z0vp * r1_area_3d) * tmask_3d), v_ip
            )
            v_il = jnp.where(
                m_ihls_3d, sr(sr(z0vl * r1_area_3d) * tmask_3d), v_il
            )

        # Post-advection `diag_adv_*` accumulation on `A2D(0)` (`:400-417`)
        for jl in range(jpl):
            mass_post = sr(
                sr(
                    sr(sr(v_i[:, :, jl] * rhoi_f) + sr(v_s[:, :, jl] * rhos_f))
                    + sr(v_ip[:, :, jl] * rhow_f)
                )
                + sr(v_il[:, :, jl] * rhow_f)
            )
            dmass = sr(sr(mass_post - zdiag_adv_mass[:, :, jl]) * r1_Dt_ice)
            diag_adv_mass = jnp.where(
                interior_2d, sr(diag_adv_mass + dmass), diag_adv_mass
            )

            sum_ei_post = fortran_sum_categories(e_i[:, :, :, jl], axis=2)
            sum_es_post = fortran_sum_categories(e_s[:, :, :, jl], axis=2)
            heat_post = sr(sr(-sum_ei_post) - sum_es_post)
            dheat = sr(sr(heat_post - zdiag_adv_heat[:, :, jl]) * r1_Dt_ice)
            diag_adv_heat = jnp.where(
                interior_2d, sr(diag_adv_heat + dheat), diag_adv_heat
            )

            if sal4_active:
                sum_szv_post = fortran_sum_categories(szv_i[:, :, :, jl], axis=2)
                salt_post = sr(sum_szv_post * rhoi_f)
            else:
                salt_post = sr(sv_i[:, :, jl] * rhoi_f)
            dsalt = sr(sr(salt_post - zdiag_adv_salt[:, :, jl]) * r1_Dt_ice)
            diag_adv_salt = jnp.where(
                interior_2d, sr(diag_adv_salt + dsalt), diag_adv_salt
            )

            # `Hbig_pra` (`:421-422`) and `Hsnow_pra` (`:425`) interleaved per category `jl`
            (
                v_i_jl,
                v_s_jl,
                a_i_jl,
                a_ip_jl,
                v_ip_jl,
                sv_i_jl,
                e_s_jl,
                e_i_jl,
                szv_i_jl,
                wfx_res,
                hfx_res,
                sfx_res,
            ) = si3_hbig_pra(
                ihls,
                rDt_ice,
                zhi_max[:, :, jl],
                zhs_max[:, :, jl],
                zhip_max[:, :, jl],
                zsi_max[:, :, jl],
                zes_max[:, :, :, jl],
                zei_max[:, :, :, jl],
                zszi_max[:, :, :, jl],
                v_i[:, :, jl],
                v_s[:, :, jl],
                a_i[:, :, jl],
                a_ip[:, :, jl],
                v_ip[:, :, jl],
                sv_i[:, :, jl],
                e_s[:, :, :, jl],
                e_i[:, :, :, jl],
                szv_i[:, :, :, jl],
                wfx_res,
                hfx_res,
                sfx_res,
                nn_hls=nn_hls,
                nn_icesal=cfg.nn_icesal,
                ln_pnd_LEV=cfg.ln_pnd_LEV,
                ln_pnd_TOPO=cfg.ln_pnd_TOPO,
                hi_max_top=hi_max_top,
                rhoi=consts.rho_ice,
                rhos=consts.rho_snow,
                _plant_perturb_hbig_cap=_plant_perturb_hbig_cap,
            )
            (
                v_i_jl,
                v_s_jl,
                a_i_jl,
                a_ip_jl,
                e_s_jl,
                wfx_res,
                hfx_res,
            ) = si3_hsnow_pra(
                ihls,
                rDt_ice,
                v_i_jl,
                v_s_jl,
                a_i_jl,
                a_ip_jl,
                e_s_jl,
                wfx_res,
                hfx_res,
                nn_hls=nn_hls,
                rho0=consts.rho_ocean,
                rhoi=consts.rho_ice,
                rhos=consts.rho_snow,
            )
            v_i = v_i.at[:, :, jl].set(v_i_jl)
            v_s = v_s.at[:, :, jl].set(v_s_jl)
            a_i = a_i.at[:, :, jl].set(a_i_jl)
            a_ip = a_ip.at[:, :, jl].set(a_ip_jl)
            v_ip = v_ip.at[:, :, jl].set(v_ip_jl)
            sv_i = sv_i.at[:, :, jl].set(sv_i_jl)
            e_s = e_s.at[:, :, :, jl].set(e_s_jl)
            e_i = e_i.at[:, :, :, jl].set(e_i_jl)
            szv_i = szv_i.at[:, :, :, jl].set(szv_i_jl)

        # `ice_var_zapneg` (`:435`)
        (
            v_i,
            v_s,
            sv_i,
            oa_i,
            a_i,
            a_ip,
            v_ip,
            v_il,
            e_s,
            e_i,
            szv_i,
            wfx_res,
            hfx_res,
            sfx_res,
        ) = si3_var_zapneg(
            ihls,
            rDt_ice,
            v_i,
            v_s,
            sv_i,
            oa_i,
            a_i,
            a_ip,
            v_ip,
            v_il,
            e_s,
            e_i,
            szv_i,
            wfx_res,
            hfx_res,
            sfx_res,
            nn_hls=nn_hls,
            nn_icesal=cfg.nn_icesal,
            rhoi=consts.rho_ice,
            rhos=consts.rho_snow,
            rhow=canonical_constants.rho_water,
        )

        # Open-water fraction `pato_i` update on `A2D(0)` (`:438-443`)
        zati2 = fortran_sum_categories(a_i, axis=-1)
        da = sr(zati2 - zati1)
        div_u = sr(zudy - jnp.roll(zudy, 1, axis=0))
        div_v = sr(zvdx - jnp.roll(zvdx, 1, axis=1))
        div_uv = sr(div_u + div_v)
        div_term = sr(sr(div_uv * grid.r1_e1e2t) * zdt)
        ato_i_upd = jnp.maximum(_SI3_PRA_ZERO, sr(sr(ato_i - da) - div_term))
        ato_i = jnp.where(interior_2d, ato_i_upd, ato_i)

        prather = SI3AllPratherMoments(
            ice=m_ice,
            sn=m_sn,
            a=m_a,
            age=m_age,
            sal=m_sal,
            ap=m_ap,
            vp=m_vp,
            vl=m_vl,
            c0=m_c0,
            e=m_e,
            si=m_si,
        )

        # Lateral boundary conditions `lbc_lnk(..., ldfull = .TRUE.)` (`:450-494`)
        if not _plant_omit_subcycle_lbc:
            def _t_lbc(arr: Field) -> Field:
                return si3_lbc_lnk_orca2(
                    arr, "T", _SI3_PRA_ONE, nn_hls=nn_hls, ldfull=True
                )

            def _fam_lbc(fam: SI3PratherMoments) -> SI3PratherMoments:
                return _apply_prather_family_halos(
                    fam,
                    nn_hls=nn_hls,
                    ldfull=True,
                    _plant_wrong_nfd_sign=_plant_wrong_nfd_sign,
                )

            v_i = _t_lbc(v_i)
            v_s = _t_lbc(v_s)
            sv_i = _t_lbc(sv_i)
            a_i = _t_lbc(a_i)
            oa_i = _t_lbc(oa_i)
            m_ice_h = _fam_lbc(prather.ice)
            m_sn_h = _fam_lbc(prather.sn)
            m_sal_h = _fam_lbc(prather.sal)
            m_a_h = _fam_lbc(prather.a)
            m_age_h = _fam_lbc(prather.age)

            if pnd_active:
                a_ip = _t_lbc(a_ip)
                v_ip = _t_lbc(v_ip)
                v_il = _t_lbc(v_il)
                m_ap_h = _fam_lbc(prather.ap)
                m_vp_h = _fam_lbc(prather.vp)
                m_vl_h = _fam_lbc(prather.vl)
            else:
                m_ap_h = prather.ap
                m_vp_h = prather.vp
                m_vl_h = prather.vl

            e_s = _t_lbc(e_s)
            e_i = _t_lbc(e_i)
            m_c0_h = _fam_lbc(prather.c0)
            m_e_h = _fam_lbc(prather.e)
            if sal4_active:
                szv_i = _t_lbc(szv_i)
                m_si_h = _fam_lbc(prather.si)
            else:
                m_si_h = prather.si

            ato_i = _t_lbc(ato_i)
            wfx_res = _t_lbc(wfx_res)
            hfx_res = _t_lbc(hfx_res)
            sfx_res = _t_lbc(sfx_res)

            prather = SI3AllPratherMoments(
                ice=m_ice_h,
                sn=m_sn_h,
                a=m_a_h,
                age=m_age_h,
                sal=m_sal_h,
                ap=m_ap_h,
                vp=m_vp_h,
                vl=m_vl_h,
                c0=m_c0_h,
                e=m_e_h,
                si=m_si_h,
            )

    return (
        ato_i,
        v_i,
        v_s,
        sv_i,
        oa_i,
        a_i,
        a_ip,
        v_ip,
        v_il,
        e_s,
        e_i,
        szv_i,
        prather,
        wfx_res,
        hfx_res,
        sfx_res,
        diag_adv_mass,
        diag_adv_salt,
        diag_adv_heat,
        zcfl,
        jnp.asarray(icycle, dtype=jnp.int32),
    )


def si3_dyn_orca2(
    state: SI3State,
    grid: SI3ORCA2AdvectionGrid,
    *,
    kt: int = 1,
    sss_m: Field | float | None = None,
    sst_m: Field | float | None = None,
    rn_uice: float | Field | None = None,
    rn_vice: float | Field | None = None,
    config: SI3ORCA2StateConfig | None = None,
    subcycles: int | None = None,
    _plant_invert_sweep_order: bool = False,
    _plant_perturb_hbig_cap: bool = False,
    _plant_omit_subcycle_lbc: bool = False,
    _plant_omit_donor_readjust: bool = False,
    _plant_corrupt_receiver_flux: bool = False,
    _plant_wrong_nfd_sign: bool = False,
) -> tuple[SI3State, SI3AdvectionDiagnostics]:
    """Execute NEMO SI3 `ice_dyn` (`ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icedyn.f90:126-265`).

    Follows NEMO's exact statement order:
    1. Pre-advection intensive thickness recovery (`icedyn.f90:156-170`)
    2. Optional `np_dynADV2D` velocity masking (`u_ice = rn_uice * umask`,
       `v_ice = rn_vice * vmask`, `icedyn.f90:222-223`)
    3. `ice_dyn_adv_pra` (`icedyn.f90:224`, `icedyn_adv.f90:149-150`,
       `icedyn_adv_pra.f90:70-510`)
    4. Transport diagnostics `diag_trp_ei, diag_trp_es, diag_trp_sv, diag_trp_vi, diag_trp_vs`
       on `A2D(0)` (`icedyn_adv.f90:156-164`)
    5. `ice_var_zapsmall` (`icedyn.f90:225`, `icevar.f90:615-737`)
    6. End-of-`ice_dyn` `lbc_lnk(..., ldfull = .TRUE.)` (`icedyn.f90:254-260`).
    """
    cfg = config if config is not None else SI3ORCA2StateConfig(ln_icedyn=True)
    sr = nemo_source_round
    consts = cfg.ice_constants
    jpi, jpj, _ = state.v_i.shape
    dtype = state.v_i.dtype
    nn_hls = int(cfg.nn_hls)
    interior_2d = _si3_active_mask_2d(jpi, jpj, nn_hls, 0)
    rDt_ice = jnp.asarray(cfg.rDt_ice, dtype=dtype)
    r1_Dt_ice = sr(_SI3_PRA_ONE / rDt_ice)
    epsi20 = jnp.asarray(_EPSI20, dtype=dtype)

    # Record before-step `*_b` on `A2D(0)` (`store_fields`, `icestp.f90:438-457`)
    a_i_b = state.a_i
    v_i_b = state.v_i
    v_s_b = state.v_s
    v_ip_b = state.v_ip
    v_il_b = state.v_il
    sv_i_b = state.sv_i
    e_s_b = state.e_s
    e_i_b = state.e_i
    szv_i_b = state.szv_i

    # 1. Pre-advection intensive thickness recovery (`icedyn.f90:156-170`)
    ai_ge_eps20 = state.a_i >= epsi20
    ai_safe = jnp.where(ai_ge_eps20, state.a_i, _SI3_PRA_ONE)
    h_i_pre = jnp.where(ai_ge_eps20, sr(state.v_i / ai_safe), _SI3_PRA_ZERO)
    h_s_pre = jnp.where(ai_ge_eps20, sr(state.v_s / ai_safe), _SI3_PRA_ZERO)

    aip_ge_eps20 = state.a_ip >= epsi20
    aip_safe = jnp.where(aip_ge_eps20, state.a_ip, _SI3_PRA_ONE)
    h_ip_pre = jnp.where(aip_ge_eps20, sr(state.v_ip / aip_safe), _SI3_PRA_ZERO)
    h_il_pre = jnp.where(aip_ge_eps20, sr(state.v_il / aip_safe), _SI3_PRA_ZERO)

    # 2. Velocity selection (`icedyn.f90:222-223`)
    if rn_uice is not None and rn_vice is not None:
        u_ice = sr(jnp.asarray(rn_uice, dtype=dtype) * grid.umask)
        v_ice = sr(jnp.asarray(rn_vice, dtype=dtype) * grid.vmask)
    else:
        u_ice = state.u_ice
        v_ice = state.v_ice

    # 3. `ice_dyn_adv_pra` (`icedyn_adv_pra.f90:70-510`)
    (
        ato_i_adv,
        v_i,
        v_s,
        sv_i,
        oa_i,
        a_i,
        a_ip,
        v_ip,
        v_il,
        e_s,
        e_i,
        szv_i,
        prather,
        wfx_res,
        hfx_res,
        sfx_res,
        diag_adv_mass,
        diag_adv_salt,
        diag_adv_heat,
        zcfl,
        icycle,
    ) = si3_dyn_adv_pra_orca2(
        kt=kt,
        u_ice=u_ice,
        v_ice=v_ice,
        h_i=h_i_pre,
        h_s=h_s_pre,
        h_ip=h_ip_pre,
        ato_i=state.aggregated.ato_i,
        v_i=state.v_i,
        v_s=state.v_s,
        sv_i=state.sv_i,
        oa_i=state.oa_i,
        a_i=state.a_i,
        a_ip=state.a_ip,
        v_ip=state.v_ip,
        v_il=state.v_il,
        e_s=state.e_s,
        e_i=state.e_i,
        szv_i=state.szv_i,
        prather=state.prather,
        grid=grid,
        config=cfg,
        hi_max_top=state.bounds.hi_max[-1],
        subcycles=subcycles,
        _plant_invert_sweep_order=_plant_invert_sweep_order,
        _plant_perturb_hbig_cap=_plant_perturb_hbig_cap,
        _plant_omit_subcycle_lbc=_plant_omit_subcycle_lbc,
        _plant_omit_donor_readjust=_plant_omit_donor_readjust,
        _plant_corrupt_receiver_flux=_plant_corrupt_receiver_flux,
        _plant_wrong_nfd_sign=_plant_wrong_nfd_sign,
    )

    # 4. Transport diagnostics `diag_trp_*` on `A2D(0)` (`icedyn_adv.f90:156-164`)
    dei = sr(e_i - e_i_b)
    des = sr(e_s - e_s_b)
    diag_trp_ei = jnp.where(
        interior_2d,
        sr(
            fortran_sum_categories(fortran_sum_categories(dei, axis=-1), axis=-1)
            * r1_Dt_ice
        ),
        _SI3_PRA_ZERO,
    )
    diag_trp_es = jnp.where(
        interior_2d,
        sr(
            fortran_sum_categories(fortran_sum_categories(des, axis=-1), axis=-1)
            * r1_Dt_ice
        ),
        _SI3_PRA_ZERO,
    )
    if cfg.nn_icesal == 4:
        dszv = sr(szv_i - szv_i_b)
        diag_trp_sv = jnp.where(
            interior_2d,
            sr(
                fortran_sum_categories(
                    fortran_sum_categories(dszv, axis=-1), axis=-1
                )
                * r1_Dt_ice
            ),
            _SI3_PRA_ZERO,
        )
    else:
        dsv = sr(sv_i - sv_i_b)
        diag_trp_sv = jnp.where(
            interior_2d,
            sr(fortran_sum_categories(dsv, axis=-1) * r1_Dt_ice),
            _SI3_PRA_ZERO,
        )
    dvi = sr(v_i - v_i_b)
    dvs = sr(v_s - v_s_b)
    diag_trp_vi = jnp.where(
        interior_2d,
        sr(fortran_sum_categories(dvi, axis=-1) * r1_Dt_ice),
        _SI3_PRA_ZERO,
    )
    diag_trp_vs = jnp.where(
        interior_2d,
        sr(fortran_sum_categories(dvs, axis=-1) * r1_Dt_ice),
        _SI3_PRA_ZERO,
    )

    # 5. `ice_var_zapsmall` (`icedyn.f90:225`, `icevar.f90:615-737`)
    sss_val = canonical_constants.S_ocean_ref if sss_m is None else sss_m
    sst_val = _SI3_PRA_ZERO if sst_m is None else sst_m
    (
        v_i,
        v_s,
        sv_i,
        oa_i,
        a_i,
        a_ip,
        v_ip,
        v_il,
        t_su,
        h_i_post,
        h_s_post,
        h_ip_post,
        h_il_post,
        e_s,
        e_i,
        szv_i,
        sz_i_post,
        t_i_post,
        t_s_post,
        at_i_post,
        vt_i_post,
        ato_i_dyn,
        wfx_res,
        hfx_res,
        sfx_res,
    ) = si3_var_zapsmall(
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        oa_i=oa_i,
        a_i=a_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        t_su=state.t_su,
        h_i=h_i_pre,
        h_s=h_s_pre,
        h_ip=h_ip_pre,
        h_il=h_il_pre,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        sz_i=state.equivalent.sz_i,
        t_i=state.equivalent.t_i,
        t_s=state.equivalent.t_s,
        at_i=state.aggregated.at_i,
        vt_i=state.aggregated.vt_i,
        ato_i=ato_i_adv,
        wfx_res=wfx_res,
        hfx_res=hfx_res,
        sfx_res=sfx_res,
        rDt_ice=rDt_ice,
        sss_m=sss_val,
        sst_m=sst_val,
        nn_hls=nn_hls,
        nn_icesal=cfg.nn_icesal,
        rn_simin=cfg.rn_simin,
        rn_sinew=cfg.rn_sinew,
        rt0=consts.T0,
        rhoi=consts.rho_ice,
        rhos=consts.rho_snow,
        rhow=canonical_constants.rho_water,
    )

    # 6. End-of-`ice_dyn` `lbc_lnk(..., ldfull = .TRUE.)` (`icedyn.f90:254-260`)
    if not _plant_omit_subcycle_lbc:
        def _t_pos(arr: Field) -> Field:
            return si3_lbc_lnk_orca2(
                arr, "T", _SI3_PRA_ONE, nn_hls=nn_hls, ldfull=True
            )

        a_i = _t_pos(a_i)
        v_i = _t_pos(v_i)
        v_s = _t_pos(v_s)
        sv_i = _t_pos(sv_i)
        oa_i = _t_pos(oa_i)
        t_su = _t_pos(t_su)
        if cfg.ln_pnd_LEV or cfg.ln_pnd_TOPO:
            a_ip = _t_pos(a_ip)
            v_ip = _t_pos(v_ip)
            v_il = _t_pos(v_il)
        e_i = _t_pos(e_i)
        e_s = _t_pos(e_s)
        szv_i = _t_pos(szv_i)

    # `diag_trends(1)` (`icestp.f90:215, 545-599`)
    rhoi_f = jnp.asarray(consts.rho_ice, dtype=dtype)
    rhos_f = jnp.asarray(consts.rho_snow, dtype=dtype)
    rhow_f = jnp.asarray(canonical_constants.rho_water, dtype=dtype)

    dei_trend = fortran_sum_categories(
        fortran_sum_categories(sr(e_i - e_i_b), axis=-1), axis=-1
    )
    des_trend = fortran_sum_categories(
        fortran_sum_categories(sr(e_s - e_s_b), axis=-1), axis=-1
    )
    diag_heat = jnp.where(
        interior_2d,
        sr(sr(-sr(dei_trend * r1_Dt_ice)) - sr(des_trend * r1_Dt_ice)),
        _SI3_PRA_ZERO,
    )
    dvi_trend = fortran_sum_categories(sr(v_i - v_i_b), axis=-1)
    diag_vice = jnp.where(
        interior_2d,
        sr(sr(dvi_trend * r1_Dt_ice) * rhoi_f),
        _SI3_PRA_ZERO,
    )
    dvs_trend = fortran_sum_categories(sr(v_s - v_s_b), axis=-1)
    diag_vsnw = jnp.where(
        interior_2d,
        sr(sr(dvs_trend * r1_Dt_ice) * rhos_f),
        _SI3_PRA_ZERO,
    )
    dpnd_diff = sr(sr(sr(v_ip + v_il) - v_ip_b) - v_il_b)
    dpnd_trend = fortran_sum_categories(dpnd_diff, axis=-1)
    diag_vpnd = jnp.where(
        interior_2d,
        sr(sr(dpnd_trend * r1_Dt_ice) * rhow_f),
        _SI3_PRA_ZERO,
    )
    if cfg.nn_icesal == 4:
        dsal_trend = fortran_sum_categories(
            fortran_sum_categories(sr(szv_i - szv_i_b), axis=-1), axis=-1
        )
    else:
        dsal_trend = fortran_sum_categories(sr(sv_i - sv_i_b), axis=-1)
    diag_sice = jnp.where(
        interior_2d,
        sr(sr(dsal_trend * r1_Dt_ice) * rhoi_f),
        _SI3_PRA_ZERO,
    )
    dai_trend = fortran_sum_categories(sr(a_i - a_i_b), axis=-1)
    diag_aice = jnp.where(
        interior_2d,
        sr(dai_trend * r1_Dt_ice),
        _SI3_PRA_ZERO,
    )

    eqv_dyn = state.equivalent._replace(
        h_i=h_i_post,
        h_s=h_s_post,
        h_ip=h_ip_post,
        h_il=h_il_post,
        sz_i=sz_i_post,
        t_i=t_i_post,
        t_s=t_s_post,
        a_i=a_i,
        v_s=v_s,
    )
    agg_dyn = state.aggregated._replace(
        at_i=at_i_post,
        vt_i=vt_i_post,
        ato_i=ato_i_dyn,
    )
    state_dyn = state._replace(
        v_i=v_i,
        v_s=v_s,
        sv_i=sv_i,
        a_i=a_i,
        t_su=t_su,
        u_ice=u_ice,
        v_ice=v_ice,
        oa_i=oa_i,
        a_ip=a_ip,
        v_ip=v_ip,
        v_il=v_il,
        e_s=e_s,
        e_i=e_i,
        szv_i=szv_i,
        prather=prather,
        equivalent=eqv_dyn,
        aggregated=agg_dyn,
    )
    diagnostics = SI3AdvectionDiagnostics(
        diag_adv_mass=diag_adv_mass,
        diag_adv_salt=diag_adv_salt,
        diag_adv_heat=diag_adv_heat,
        wfx_res=wfx_res,
        hfx_res=hfx_res,
        sfx_res=sfx_res,
        diag_trp_vi=diag_trp_vi,
        diag_trp_vs=diag_trp_vs,
        diag_trp_ei=diag_trp_ei,
        diag_trp_es=diag_trp_es,
        diag_trp_sv=diag_trp_sv,
        diag_heat=diag_heat,
        diag_sice=diag_sice,
        diag_vice=diag_vice,
        diag_vsnw=diag_vsnw,
        diag_vpnd=diag_vpnd,
        diag_aice=diag_aice,
        ato_i_post_adv=ato_i_adv,
        ato_i_post_dyn=ato_i_dyn,
        zcfl=zcfl,
        icycle=icycle,
    )
    return state_dyn, diagnostics


def si3_o3_step(
    state: SI3State,
    grid: SI3ORCA2AdvectionGrid,
    *,
    kt: int = 1,
    sss_m: Field | float | None = None,
    sst_m: Field | float | None = None,
    rn_uice: float | Field | None = None,
    rn_vice: float | Field | None = None,
    config: SI3ORCA2StateConfig | None = None,
    subcycles: int | None = None,
    apply_halos: bool = False,
    _plant_invert_sweep_order: bool = False,
    _plant_perturb_hbig_cap: bool = False,
    _plant_omit_subcycle_lbc: bool = False,
    _plant_omit_donor_readjust: bool = False,
    _plant_corrupt_receiver_flux: bool = False,
    _plant_wrong_nfd_sign: bool = False,
    _plant_stop_gradient: bool = False,
) -> tuple[SI3State, SI3AdvectionDiagnostics]:
    """Execute one Rung O3 ORCA2 SI3 Prather advection step (`ln_icedyn = True`, `ln_icethd = False`).

    Follows NEMO's exact statement order in `ORCA2_SAS_ICE_O1_REF/BLD/ppsrc/nemo/icestp.f90:181-272`:
    1. `store_fields` (`:195, 429-467`) and `diag_set0` (`:208, 470-542`)
    2. `CALL ice_dyn( kt, Kmm )` (`:211-212`, `icedyn.f90:126-265`,
       `icedyn_adv.f90:78-179`, `icedyn_adv_pra.f90:70-510`)
    3. `CALL diag_trends( 1 )` (`:215, 545-599`)
    4. `CALL ice_var_glo2eqv(2)` (`:226, 254`)
    5. `CALL ice_var_agg(2)` (`:227, 255`)
    6. `CALL ice_update_flx(kt)` snow/ice mass update (`:257`, `iceupdate.f90:203-208`).
    """
    cfg = config if config is not None else SI3ORCA2StateConfig(ln_icedyn=True)
    work = jax.lax.stop_gradient(state) if _plant_stop_gradient else state
    if apply_halos:
        work = si3_apply_state_halos_orca2(work, config=cfg)

    # `diag_set0` (`icestp.f90:208, 529`): `t_si(A2D(0), 1:jpl) = rt0`
    jpi, jpj, _ = work.v_i.shape
    interior_3d = _interior_mask_2d(jpi, jpj, cfg.nn_hls)[:, :, None]
    t_si_step = jnp.where(
        interior_3d,
        jnp.full_like(work.t_si, cfg.ice_constants.T0),
        _SI3_PRA_ZERO,
    )
    work = work._replace(t_si=t_si_step)

    # `CALL ice_dyn( kt, Kmm )` + `CALL diag_trends( 1 )`
    work_dyn, diagnostics = si3_dyn_orca2(
        work,
        grid,
        kt=kt,
        sss_m=sss_m,
        sst_m=sst_m,
        rn_uice=rn_uice,
        rn_vice=rn_vice,
        config=cfg,
        subcycles=subcycles,
        _plant_invert_sweep_order=_plant_invert_sweep_order,
        _plant_perturb_hbig_cap=_plant_perturb_hbig_cap,
        _plant_omit_subcycle_lbc=_plant_omit_subcycle_lbc,
        _plant_omit_donor_readjust=_plant_omit_donor_readjust,
        _plant_corrupt_receiver_flux=_plant_corrupt_receiver_flux,
        _plant_wrong_nfd_sign=_plant_wrong_nfd_sign,
    )
    if _plant_stop_gradient:
        work_dyn = jax.lax.stop_gradient(work_dyn)

    # `CALL ice_var_glo2eqv(2)` (`icestp.f90:226, 254`)
    eqv = si3_var_glo2eqv(
        v_i=work_dyn.v_i,
        v_s=work_dyn.v_s,
        sv_i=work_dyn.sv_i,
        a_i=work_dyn.a_i,
        oa_i=work_dyn.oa_i,
        a_ip=work_dyn.a_ip,
        v_ip=work_dyn.v_ip,
        v_il=work_dyn.v_il,
        e_s=work_dyn.e_s,
        e_i=work_dyn.e_i,
        szv_i=work_dyn.szv_i,
        kn=2,
        hi_max_top=work_dyn.bounds.hi_max[-1],
        sss_m=sss_m,
        config=cfg,
    )
    # `CALL ice_var_agg(2)` (`icestp.f90:227, 255`)
    agg = si3_var_agg(
        v_i=work_dyn.v_i,
        v_s=eqv.v_s,
        sv_i=work_dyn.sv_i,
        a_i=eqv.a_i,
        oa_i=work_dyn.oa_i,
        a_ip=work_dyn.a_ip,
        v_ip=work_dyn.v_ip,
        v_il=work_dyn.v_il,
        t_su=work_dyn.t_su,
        t_si=work_dyn.t_si,
        e_s=work_dyn.e_s,
        e_i=work_dyn.e_i,
        szv_i=work_dyn.szv_i,
        a_ip_eff=eqv.a_ip_eff,
        t_i=eqv.t_i,
        t_s=eqv.t_s,
        kn=2,
        config=cfg,
    )
    # `CALL ice_update_flx(kt)` snow/ice mass update (`icestp.f90:257`, `iceupdate.f90:203-208`)
    snwice_mass, snwice_mass_b, snwice_fmass = si3_update_snwice_mass(
        vt_i=agg.vt_i,
        vt_s=agg.vt_s,
        vt_ip=agg.vt_ip,
        vt_il=agg.vt_il,
        snwice_mass_prev=work.snwice_mass,
        tmask=grid.tmask,
        config=cfg,
        is_init=False,
    )
    out_state = work_dyn._replace(
        a_i=eqv.a_i,
        v_s=eqv.v_s,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        snwice_fmass=snwice_fmass,
        equivalent=eqv,
        aggregated=agg,
    )
    return out_state, diagnostics


__all__ = [
    "SI3ORCA2StateConfig",
    "SI3ORCA2Forcing",
    "SI3ORCA2AdvectionGrid",
    "SI3AdvectionDiagnostics",
    "SI3CategoryBounds",
    "SI3PratherMoments",
    "SI3AllPratherMoments",
    "SI3EquivalentVars",
    "SI3AggregatedVars",
    "SI3State",
    "fortran_sum_categories",
    "si3_hfn_category_bounds",
    "si3_var_salprof",
    "si3_var_snwfra_3d",
    "si3_var_glo2eqv",
    "si3_var_agg",
    "si3_update_snwice_mass",
    "si3_lbc_lnk_orca2",
    "si3_apply_state_halos_orca2",
    "si3_o1_step",
    "si3_dyn_adv_pra_orca2",
    "si3_dyn_orca2",
    "si3_o3_step",
    "si3_rst_to_netcdf_dict",
    "si3_rst_from_netcdf_dict",
    "write_si3_netcdf_restart",
    "read_si3_netcdf_restart",
]


__param_spec__["SI3O2InitConfig"] = {
    "scheme_key": "ice.si3_orca2_o2_init",
    "excluded": {
        "rn_thres_sst": "oracle identity: NEMO namini SST freezing threshold [degC]",
        "rn_hti_ini_n": "oracle identity: NEMO namini NH initial ice thickness [m]",
        "rn_hti_ini_s": "oracle identity: NEMO namini SH initial ice thickness [m]",
        "rn_hts_ini_n": "oracle identity: NEMO namini NH initial snow thickness [m]",
        "rn_hts_ini_s": "oracle identity: NEMO namini SH initial snow thickness [m]",
        "rn_ati_ini_n": "oracle identity: NEMO namini NH initial ice concentration [1]",
        "rn_ati_ini_s": "oracle identity: NEMO namini SH initial ice concentration [1]",
        "rn_smi_ini_n": "oracle identity: NEMO namini NH initial ice salinity [g/kg]",
        "rn_smi_ini_s": "oracle identity: NEMO namini SH initial ice salinity [g/kg]",
        "rn_tmi_ini_n": "oracle identity: NEMO namini NH initial ice temperature [K]",
        "rn_tmi_ini_s": "oracle identity: NEMO namini SH initial ice temperature [K]",
        "rn_tsu_ini_n": "oracle identity: NEMO namini NH initial surface temperature [K]",
        "rn_tsu_ini_s": "oracle identity: NEMO namini SH initial surface temperature [K]",
        "rn_tms_ini_n": "oracle identity: NEMO namini NH initial snow temperature [K]",
        "rn_tms_ini_s": "oracle identity: NEMO namini SH initial snow temperature [K]",
        "rn_apd_ini_n": "oracle identity: NEMO namini NH initial pond fraction [1]",
        "rn_apd_ini_s": "oracle identity: NEMO namini SH initial pond fraction [1]",
        "rn_hpd_ini_n": "oracle identity: NEMO namini NH initial pond depth [m]",
        "rn_hpd_ini_s": "oracle identity: NEMO namini SH initial pond depth [m]",
        "rn_hld_ini_n": "oracle identity: NEMO namini NH initial pond lid depth [m]",
        "rn_hld_ini_s": "oracle identity: NEMO namini SH initial pond lid depth [m]",
        "rn_amax_n": "oracle identity: NEMO nampar NH maximum ice concentration [1]",
        "rn_amax_s": "oracle identity: NEMO nampar SH maximum ice concentration [1]",
        "ln_pnd": "oracle identity: NEMO namthd_pnd melt-pond switch",
        "ln_pnd_lids": "oracle identity: NEMO namthd_pnd frozen-lid switch",
    },
    "params": {},
}


class SI3O2InitConfig(NamedTuple):
    """Rung-O2 `&namini` and `&nampar` scalar initial-state configuration.

    Source citations:
    - `src/ICE/iceistate.F90:453-534` (`ice_istate_init` `&namini` read)
    - `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:483-562`
    - `cfgs/ORCA2_SAS_ICE/EXPREF/namelist_ice_cfg:9-50`
    - `cfgs/SHARED/namelist_ice_ref:65-95`
    """

    ln_iceini: bool = True
    nn_iceini_file: int = 0
    rn_thres_sst: float = 2.0
    rn_hti_ini_n: float = 2.0
    rn_hti_ini_s: float = 1.0
    rn_hts_ini_n: float = 0.2
    rn_hts_ini_s: float = 0.2
    rn_ati_ini_n: float = 0.9
    rn_ati_ini_s: float = 0.9
    rn_smi_ini_n: float = 6.3
    rn_smi_ini_s: float = 6.3
    rn_tmi_ini_n: float = 270.0
    rn_tmi_ini_s: float = 270.0
    rn_tsu_ini_n: float = 270.0
    rn_tsu_ini_s: float = 270.0
    rn_tms_ini_n: float = 270.0
    rn_tms_ini_s: float = 270.0
    rn_apd_ini_n: float = 0.2
    rn_apd_ini_s: float = 0.2
    rn_hpd_ini_n: float = 0.05
    rn_hpd_ini_s: float = 0.05
    rn_hld_ini_n: float = 0.0
    rn_hld_ini_s: float = 0.0
    rn_amax_n: float = 0.997
    rn_amax_s: float = 0.997
    ln_pnd: bool = True
    ln_pnd_lids: bool = True
    ln_teos10: bool = True
    ln_ice_embd: bool = False
    ln_rstart: bool = False
    nn_hls: int = 2
    ice_constants: IceConstantsConfig = NEMO_SI3_CONSTANTS_CONFIG

    @classmethod
    def for_orca2_sas_ice(cls, **overrides: Any) -> "SI3O2InitConfig":
        """Return the NEMO 5.0.2 `ORCA2_SAS_ICE` rung-O2 `&namini` configuration."""
        base = cls()
        return base._replace(**overrides) if overrides else base

    @classmethod
    def for_orca2_omip_l4(cls, **overrides: Any) -> "SI3O2InitConfig":
        """Return the NEMO 5.0.2 `ORCA2_OMIP_L4` (`ln_eos80 = .true.`) `&namini` configuration."""
        base = cls(ln_teos10=False)
        return base._replace(**overrides) if overrides else base


class SI3O2ScalarInitialState(NamedTuple):
    """Rung-O2 scalar initial state and levitating-ice SSH adjustment (`src/ICE/iceistate.F90:83-443`).

    Source citations:
    - `src/ICE/iceistate.F90:124-126`, `:203-207`, `:262-304`, `:398-430`
    - `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:174-175`, `:252-256`,
      `:309-338`, `:440-460`
    """

    t_bo: Field
    ztfrz: Field
    zmsk: Field
    zht_i_ini: Field
    zht_s_ini: Field
    zat_i_ini: Field
    zsm_i_ini: Field
    ztm_i_ini: Field
    zt_su_ini: Field
    ztm_s_ini: Field
    zapnd_ini: Field
    zhpnd_ini: Field
    zhlid_ini: Field
    rn_amax_2d: Field
    vt_i: Field
    vt_s: Field
    at_i: Field
    at_ip: Field
    vt_ip: Field
    vt_il: Field
    snwice_mass: Field
    snwice_mass_b: Field
    area: Field
    zsshadj: Field
    ssh_kbb: Field
    ssh_kmm: Field
    o3: "SI3O3IstateFields"


class SI3ORCA2O2Forcing(NamedTuple):
    """Boundary forcing for `SI3State` steps through `step_sea_ice` (`thermo_scheme='si3_orca2_o2'`)."""

    tmask: Field
    sst_m: Field
    sss_m: Field
    ff_t: Field
    gphit: Field
    e1e2t: Field
    ssmask: Field
    smask0_i: Field
    ssh_before_kbb: Field
    ssh_before_kmm: Field
    apply_halos: bool


@jax.custom_jvp
def _safe_sqrt(x: Field) -> Field:
    """Exact `jnp.sqrt(x)` primal with finite zero tangent at `x <= 0.0`."""
    return jnp.sqrt(jnp.maximum(x, jnp.asarray(0.0, dtype=x.dtype)))


@_safe_sqrt.defjvp
def _safe_sqrt_jvp(primals: tuple[Field], tangents: tuple[Field]) -> tuple[Field, Field]:
    (x,) = primals
    (dx,) = tangents
    y = _safe_sqrt(x)
    safe_denom = jnp.where(y > 0.0, 2.0 * y, 1.0)
    dy = jnp.where(y > 0.0, dx / safe_denom, jnp.zeros_like(dx))
    return y, dy


def si3_dom_uniq_t_orca2(
    jpi: int = 184,
    jpj: int = 152,
    nn_hls: int = 2,
    *,
    _plant_omit_nfd_uniq: bool = False,
) -> Field:
    """Compute NEMO's `dom_uniq( puniq, 'T' )` uniqueness mask on ORCA2 (`src/OCE/DOM/domutl.F90:91-121`).

    Sets `1.0` on interior points `(Nis0:Nie0, Njs0:Nje0)` that are not
    overwritten by `lbc_lnk( 'dom_uniq', z2d, 'T', 1._wp )`, and `0.0` on
    halos and on the duplicated right half of the T-pivot north-fold row
    (`jj = Nje0 = jpj - nn_hls`, 1-based `ji = jpi / 2 + 2 .. jpi - nn_hls`).
    """
    h = int(nn_hls)
    ni0glo = int(jpi - 2 * h)
    ji_1based = jnp.arange(1, jpi + 1, dtype=jnp.float64)[:, None]
    jj_1based = jnp.arange(1, jpj + 1, dtype=jnp.float64)[None, :]
    mig0 = ji_1based - float(h)
    mjg0 = jj_1based - float(h)
    ztst = mig0 + (mjg0 - 1.0) * float(ni0glo)
    interior = _interior_mask_2d(jpi, jpj, h)
    z2d = ztst * interior
    if _plant_omit_nfd_uniq:
        return interior
    z2d_linked = si3_lbc_lnk_orca2(
        z2d, cd_nat="T", psgn=1.0, nn_hls=h, ldfull=True
    )
    return jnp.where((interior > 0.0) & (z2d_linked == ztst), 1.0, 0.0)


def si3_glob_2dsum_ddpdd(
    field_2d: Field,
    smask0_i: Field,
    nn_hls: int = 2,
) -> Field:
    """NEMO `glob_2Dsum` double-double Knuth two-sum reduction (`src/OCE/LBC/lib_mpp.F90:1244-1267`).

    Source citations:
    - `src/OCE/lib_fortran_generic.h90:90-193` (`local_2Dsum`, `glob_2Dsum`)
    - `ORCA2_OMIP_L4/BLD/ppsrc/nemo/lib_fortran.f90:1505-1531`
    - `src/OCE/LBC/lib_mpp.F90:1244-1267` (`SUBROUTINE DDPDD`)
    """
    h = int(nn_hls)
    jpi, jpj = field_2d.shape
    nis0, nie0 = h, jpi - h
    njs0, nje0 = h, jpj - h
    return nemo_ddpdd_sum_2d(
        jnp.transpose(field_2d[nis0:nie0, njs0:nje0]),
        jnp.transpose(smask0_i[nis0:nie0, njs0:nje0]),
    )


def si3_eos_fzp(
    sss_m: Field,
    *,
    ln_teos10: bool = True,
    kbnd: int = 2,
    nn_hls: int = 2,
    _plant_wrong_eos: bool = False,
) -> Field:
    """Compute NEMO's 2D surface freezing point `eos_fzp` (`src/OCE/TRA/eosbn2.F90:1650-1710`).

    Source citations:
    - `src/OCE/TRA/eosbn2.F90:69`, `:1650-1710` (`eos_fzp_2d_t`)
    - `ORCA2_OMIP_L4/BLD/ppsrc/nemo/eosbn2.f90:1787-1848`
    - `src/ICE/iceistate.F90:125`, `:263`
    """
    sr = nemo_source_round
    sss = jnp.asarray(sss_m, dtype=jnp.float64)
    dtype = sss.dtype
    use_teos10 = (not ln_teos10) if _plant_wrong_eos else ln_teos10

    if use_teos10:
        pfzp = nemo_eos_fzp(sss)
    else:
        # UNESCO / Millero EOS-80 polynomial (`eosbn2.F90:1691-1696`)
        zs = sr(_safe_sqrt(jnp.maximum(sss, jnp.asarray(0.0, dtype=dtype))))
        a0 = jnp.asarray(-0.0575, dtype=dtype)
        a1 = jnp.asarray(1.710523e-3, dtype=dtype)
        a2 = jnp.asarray(2.154996e-4, dtype=dtype)
        pfzp = sr(sr(sr(a0 + sr(a1 * zs)) - sr(a2 * sss)) * sss)

    jpi, jpj = sss.shape
    pad = max(int(nn_hls) - int(kbnd), 0)
    if pad > 0:
        bnd_mask = _interior_mask_2d(jpi, jpj, pad)
        pfzp = pfzp * bnd_mask
    return pfzp


# Fixed NEMO 5.0.2 SI3 small numbers (`ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/par_ice.f90:134-135`).
_EPSI06 = 1.0e-06
_EPSI10 = 1.0e-10


class SI3O3ITDFit(NamedTuple):
    """Per-category output of NEMO's `ice_var_itd_1cMc` (`icevar.f90:1182-1327`)."""

    h_i: Field
    h_s: Field
    a_i: Field
    t_i1: Field
    t_s1: Field
    t_su: Field
    s_i: Field
    a_ip: Field
    h_ip: Field
    h_il: Field


class SI3O3IstateFields(NamedTuple):
    """Rung-O3 post-fit `ice_istate` fields (`iceistate.f90:356-434`)."""

    fit: SI3O3ITDFit
    a_i: Field
    v_i: Field
    v_s: Field
    sv_i: Field
    t_su: Field
    s_i: Field
    sz_i: Field
    t_i: Field
    t_s: Field
    e_i: Field
    e_s: Field
    szv_i: Field
    a_ip: Field
    v_ip: Field
    v_il: Field
    a_ip_eff: Field
    t1_ice: Field
    zswitch: Field
    at_i: Field


def _safe_div(num: Field, den: Field, ok: Field) -> Field:
    """`num / den` where `ok`, else `0.0`; gradient-only guard on the untaken branch."""
    return jnp.where(ok, nemo_source_round(num / jnp.where(ok, den, 1.0)), 0.0)


def si3_var_itd_1cmc(
    *,
    phti: Field,
    phts: Field,
    pati: Field,
    ptmi: Field,
    ptms: Field,
    ptmsu: Field,
    psmi: Field,
    patip: Field,
    phtip: Field,
    phtil: Field,
    hi_max: Field,
    ice_constants: IceConstantsConfig,
    rho0: float,
    _plant_skip_snow_transfer: bool = False,
) -> SI3O3ITDFit:
    """Transcribe NEMO's 1-category -> jpl gamma-function fit `ice_var_itd_1cMc`.

    Source: `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/icevar.f90:1182-1327`
    (Abraham et al. 2015).  Elementwise over any leading shape; `hi_max` is
    the O1 HFN bound vector `hi_max(0:jpl)`.  Forward arithmetic is
    NEMO-literal in source order; the `WHERE`/`IF` switches are not
    differentiable across their thresholds, and divisions carry
    gradient-only safe denominators.
    """
    sr = nemo_source_round
    exp = transcendentals.exp
    phti = jnp.asarray(phti, dtype=jnp.float64)
    dtype = phti.dtype
    hm = jnp.asarray(hi_max, dtype=dtype)
    jpl = int(hm.shape[0]) - 1
    rhos = jnp.asarray(ice_constants.rho_snow, dtype=dtype)
    rhoi = jnp.asarray(ice_constants.rho_ice, dtype=dtype)
    rho0_v = jnp.asarray(rho0, dtype=dtype)
    r1_rho0 = sr(1.0 / rho0_v)
    r1_rhos = sr(1.0 / rhos)
    rhoi_m_rho0 = sr(rhoi - rho0_v)

    # :1220-1223
    z1_hti = _safe_div(jnp.ones_like(phti), phti, phti != 0.0)
    has_ice = phti > 0.0
    za = sr(pati * z1_hti)

    def _conc_term(h: Field) -> tuple[Field, Field]:
        # ( phti + 2.*h ) and EXP( -2.*h*z1_hti )
        lin = sr(phti + sr(2.0 * h))
        ex = exp(-sr(sr(2.0 * h) * z1_hti))
        return lin, ex

    def _vol_poly(h: Field) -> Field:
        # phti*phti + 2.*phti*h + 2.*h*h
        return sr(sr(sr(phti * phti) + sr(sr(2.0 * phti) * h)) + sr(sr(2.0 * h) * h))

    a_cats = []
    h_cats = []
    # :1227-1250  categories 1:jpl-1
    for jl in range(1, jpl):
        lin0, ex0 = _conc_term(hm[jl - 1])
        lin1, ex1 = _conc_term(hm[jl])
        pa = sr(za * sr(sr(lin0 * ex0) - sr(lin1 * ex1)))
        zv = sr(za * sr(sr(_vol_poly(hm[jl - 1]) * ex0) - sr(_vol_poly(hm[jl]) * ex1)))
        keep = pa > _EPSI06
        a_cats.append(jnp.where(has_ice & keep, pa, 0.0))
        h_cats.append(jnp.where(has_ice, _safe_div(zv, pa, keep), 0.0))
    # :1253-1271  last category, hi_max(jpl-1) to infinity
    lin0, ex0 = _conc_term(hm[jpl - 1])
    pa = sr(sr(za * lin0) * ex0)
    zv = sr(sr(za * _vol_poly(hm[jpl - 1])) * ex0)
    keep = pa > _EPSI06
    a_cats.append(jnp.where(has_ice & keep, pa, 0.0))
    h_cats.append(jnp.where(has_ice, _safe_div(zv, pa, keep), 0.0))

    # :1274-1286  snow, and snow-to-ice transfer of the excess load
    hs_cats = []
    for jl in range(1, jpl + 1):
        pa_l = a_cats[jl - 1]
        hi_l = h_cats[jl - 1]
        hs_l = sr(sr(hi_l * phts) * z1_hti)
        if _plant_skip_snow_transfer:
            hi_new, hs_new = hi_l, hs_l
        else:
            zdh = jnp.maximum(0.0, sr(sr(sr(rhos * hs_l) + sr(rhoi_m_rho0 * hi_l)) * r1_rho0))
            hi_new = jnp.minimum(hm[jl], sr(hi_l + zdh))
            hs_new = jnp.maximum(0.0, sr(hs_l - sr(sr(zdh * rhoi) * r1_rhos)))
        snow = pa_l > 0.0
        h_cats[jl - 1] = jnp.where(snow, hi_new, hi_l)
        hs_cats.append(jnp.where(snow, hs_new, 0.0))

    a_i = jnp.stack(a_cats, axis=-1)
    h_i = jnp.stack(h_cats, axis=-1)
    h_s = jnp.stack(hs_cats, axis=-1)

    # :1291-1296  temperature and salinity, every category
    def _bc(x: Field) -> Field:
        return jnp.broadcast_to(jnp.asarray(x, dtype=dtype)[..., None], a_i.shape)

    # :1299-1325  ponds
    zfra = _safe_div(patip, pati, pati != 0.0)[..., None]
    a_ip = sr(zfra * a_i)
    vi_ati = sr(phti * pati)
    vi_ok = vi_ati != 0.0
    vi_cat = sr(h_i * a_i)
    ip_ok = a_ip != 0.0
    zfra_p = _safe_div(sr(phtip * patip), vi_ati, vi_ok)[..., None]
    h_ip = _safe_div(sr(zfra_p * vi_cat), a_ip, ip_ok)
    zfra_l = _safe_div(sr(phtil * patip), vi_ati, vi_ok)[..., None]
    h_il = _safe_div(sr(zfra_l * vi_cat), a_ip, ip_ok)

    return SI3O3ITDFit(
        h_i=h_i, h_s=h_s, a_i=a_i,
        t_i1=_bc(ptmi), t_s1=_bc(ptms), t_su=_bc(ptmsu), s_i=_bc(psmi),
        a_ip=a_ip, h_ip=h_ip, h_il=h_il,
    )


def si3_o3_istate_fill_fields(
    *,
    zht_i_ini: Field,
    zht_s_ini: Field,
    zat_i_ini: Field,
    ztm_i_ini: Field,
    ztm_s_ini: Field,
    zt_su_ini: Field,
    zsm_i_ini: Field,
    zapnd_ini: Field,
    zhpnd_ini: Field,
    zhlid_ini: Field,
    tmask: Field,
    sss_m: Field,
    rn_amax_2d: Field,
    config: SI3ORCA2StateConfig,
    _plant_skip_snow_transfer: bool = False,
    _plant_skip_amax: bool = False,
) -> SI3O3IstateFields:
    """Rung O3: fit and fill `ice_istate`'s jpl categories and layers.

    Transcribes `ORCA2_SAS_ICE_O2_REF/BLD/ppsrc/nemo/iceistate.f90:356-434`:
    `ice_var_itd` (the `_1cMc` arm, selected because the 2-D inputs are
    passed row by row as rank-1 arrays and the outputs are rank-2), the
    layer copies, `zswitch`, the salinity clamp and `ice_var_salprof`, the
    extensive volumes/energies/salt, the pond volumes, `t1_ice`, and the
    `rn_amax_2d` concentration cap.  No default exists for any input.
    """
    cfg = config
    if not isinstance(cfg, SI3ORCA2StateConfig):
        raise TypeError("O3 requires an explicit SI3ORCA2StateConfig")
    if cfg.nn_icesal not in (2, 4):
        raise ValueError("O3 transcribes only the ORCA2_SAS_ICE nn_icesal=4 salinity arm")
    sr = nemo_source_round
    consts = cfg.ice_constants
    tm = jnp.asarray(tmask[..., 0] if jnp.ndim(tmask) == 3 else tmask, dtype=jnp.float64)
    dtype = tm.dtype
    jpi, jpj = tm.shape
    nlay_i, nlay_s = int(cfg.nlay_i), int(cfg.nlay_s)
    bounds = si3_hfn_category_bounds(
        jpl=cfg.jpl, rn_himean=cfg.rn_himean, opt_hips=cfg.opt_hips, rn_himax=cfg.rn_himax
    )

    # :356-363  CALL ice_var_itd( ... )  -> ice_var_itd_1cMc
    fit = si3_var_itd_1cmc(
        phti=zht_i_ini, phts=zht_s_ini, pati=zat_i_ini,
        ptmi=ztm_i_ini, ptms=ztm_s_ini, ptmsu=zt_su_ini, psmi=zsm_i_ini,
        patip=zapnd_ini, phtip=zhpnd_ini, phtil=zhlid_ini,
        hi_max=bounds.hi_max, ice_constants=consts,
        rho0=NEMO_CONSTANTS_CONFIG.rho_0,
        _plant_skip_snow_transfer=_plant_skip_snow_transfer,
    )
    # :364-373  copy layer 1 temperatures into every layer
    t_s = jnp.broadcast_to(fit.t_s1[:, :, None, :], (jpi, jpj, nlay_s, cfg.jpl))
    t_i = jnp.broadcast_to(fit.t_i1[:, :, None, :], (jpi, jpj, nlay_i, cfg.jpl))
    # :376-378
    zswitch = jnp.where(fortran_sum_categories(fit.a_i, axis=-1) > 0.0, tm, 0.0)
    # :381-386  salinity clamp then ice_var_salprof
    sss = jnp.asarray(sss_m, dtype=dtype)
    s_i = jnp.minimum(
        jnp.maximum(cfg.rn_simin, fit.s_i), sr(cfg.rn_sinew * sss)[..., None]
    )
    s_i, sz_i = si3_var_salprof(s_i, sss_m=sss, config=cfg)
    # :388-394
    v_i = sr(fit.h_i * fit.a_i)
    v_s = sr(fit.h_s * fit.a_i)
    sv_i = sr(s_i * v_i)
    rt0 = jnp.asarray(consts.T0, dtype=dtype)
    rhos = jnp.asarray(consts.rho_snow, dtype=dtype)
    rhoi = jnp.asarray(consts.rho_ice, dtype=dtype)
    rcpi = jnp.asarray(consts.c_ice, dtype=dtype)
    rcp = jnp.asarray(consts.c_ocean, dtype=dtype)
    rLfus = jnp.asarray(consts.latent_fusion, dtype=dtype)
    rTmlt = jnp.asarray(consts.liquidus_slope, dtype=dtype)
    r1_nlay_s = sr(1.0 / jnp.asarray(float(nlay_s), dtype=dtype))
    r1_nlay_i = sr(1.0 / jnp.asarray(float(nlay_i), dtype=dtype))
    zsw = zswitch[:, :, None, None]
    # :396-401
    e_s = sr(
        sr(sr(sr(zsw * v_s[:, :, None, :]) * r1_nlay_s) * rhos)
        * sr(sr(rcpi * sr(rt0 - t_s)) + rLfus)
    )
    # :403-414
    v_i4 = v_i[:, :, None, :]
    szv_i = sr(sr(sz_i * v_i4) * r1_nlay_i)
    ztmelts = sr(-sr(rTmlt * sz_i) + rt0)
    zfrac = sr(sr(ztmelts - rt0) / jnp.minimum(sr(t_i - rt0), -_EPSI10))
    e_i = sr(
        sr(sr(sr(zsw * v_i4) * r1_nlay_i) * rhoi)
        * sr(
            sr(sr(rcpi * sr(ztmelts - t_i)) + sr(rLfus * jnp.maximum(0.0, sr(1.0 - zfrac))))
            - sr(rcp * sr(ztmelts - rt0))
        )
    )
    # :416-421  ponds (a_ip_eff on A2D(0) only)
    interior = _interior_mask_2d(jpi, jpj, cfg.nn_hls, dtype=dtype)[..., None]
    a_ip_eff = _safe_div(fit.a_ip, fit.a_i, fit.a_i > _EPSI10) * interior
    v_ip = sr(fit.h_ip * fit.a_ip)
    v_il = sr(fit.h_il * fit.a_ip)
    # :424-427  coupled-run temperature, A2D(0)
    t1_ice = fit.t_i1 * interior
    # :430-434  concentration cap
    at_i = fortran_sum_categories(fit.a_i, axis=-1)
    amax = jnp.asarray(rn_amax_2d, dtype=dtype)
    over = (at_i > amax)[..., None]
    if _plant_skip_amax:
        a_i = fit.a_i
    else:
        a_i = jnp.where(over, _safe_div(sr(fit.a_i * amax[..., None]), at_i[..., None], over), fit.a_i)
    at_i = fortran_sum_categories(a_i, axis=-1)
    return SI3O3IstateFields(
        fit=fit, a_i=a_i, v_i=v_i, v_s=v_s, sv_i=sv_i, t_su=fit.t_su, s_i=s_i,
        sz_i=sz_i, t_i=t_i, t_s=t_s, e_i=e_i, e_s=e_s, szv_i=szv_i,
        a_ip=fit.a_ip, v_ip=v_ip, v_il=v_il, a_ip_eff=a_ip_eff, t1_ice=t1_ice,
        zswitch=zswitch, at_i=at_i,
    )


def si3_o2_scalar_initial_state(
    sst_m: Field,
    sss_m: Field,
    tmask: Field,
    ff_t: Field,
    gphit: Field,
    e1e2t: Field,
    *,
    ssmask: Field,
    smask0_i: Field,
    ssh_before_kbb: Field,
    ssh_before_kmm: Field,
    config: SI3O2InitConfig,
    state_config: SI3ORCA2StateConfig,
    _control_exact_fitted: tuple[Field, Field, Field, Field, Field, Field] | None = None,
    _plant_wrong_eos: bool = False,
    _plant_tbo_halo_leak: bool = False,
    _plant_pond_unscaled: bool = False,
    _plant_hemisphere_swap: bool = False,
    _plant_ssh_sign: bool = False,
    _plant_omit_nfd_uniq: bool = False,
    _plant_stop_gradient: bool = False,
) -> SI3O2ScalarInitialState:
    """Compute the Rung-O2 scalar initial state and levitating-ice SSH adjustment.

    Transcribes NEMO 5.0.2 `src/ICE/iceistate.F90`:
    1. `t_bo` (`:125-126`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/iceistate.f90:174-175`):
       `CALL eos_fzp( sss_m(:,:), t_bo(:,:), kbnd=0 )`
       `t_bo(:,:) = ( t_bo(:,:) + rt0 ) * smask0(:,:)`
    2. `rn_amax_2d` (`:203-207`, `iceistate.f90:252-256`):
       `WHERE( gphit(:,:) >= 0._wp ) ; rn_amax_2d = rn_amax_n ; ELSEWHERE ; rn_amax_2d = rn_amax_s`
    3. `ztfrz` and `zmsk` (`:263-266`, `iceistate.f90:310-313`):
       `CALL eos_fzp( sss_m(:,:), ztfrz(:,:), kbnd=nn_hls )`
       `WHERE( ( sst_m(:,:) - ztfrz(:,:) ) * tmask(:,:,1) >= rn_thres_sst ) ; zmsk = 0 ; ELSEWHERE ; zmsk = tmask(:,:,1)`
    4. 10-field hemisphere-dependent scalar state (`:275-303`, `iceistate.f90:320-338`):
       `zht_i_ini`, `zht_s_ini`, `zat_i_ini`, `zsm_i_ini`, `ztm_i_ini`,
       `zt_su_ini`, `ztm_s_ini`, `zapnd_ini`, `zhpnd_ini`, `zhlid_ini`, with
       `zapnd_ini(:,:) = zapnd_ini(:,:) * zat_i_ini(:,:)` (`:303`).
    5. Rung-O3 category fit and fill (`iceistate.f90:356-434`) through
       `si3_o3_istate_fill_fields`.  `_control_exact_fitted` (test-only)
       substitutes NEMO's six recorded `(v_i, v_s, a_i, a_ip, v_ip, v_il)`.
    6. Levitating-ice initial SSH branch (`:398-430`, `iceistate.f90:440-460`):
       `snwice_mass`, `snwice_mass_b`, `area`, `zsshadj`, `ssh_kbb`, `ssh_kmm`.
    """
    sr = nemo_source_round
    cfg = config
    if not isinstance(cfg, SI3O2InitConfig):
        raise TypeError("O2 scalar initialization requires an explicit SI3O2InitConfig")
    if not cfg.ln_iceini or cfg.nn_iceini_file != 0:
        raise ValueError(
            "O2 requires ORCA2_SAS_ICE namelist initialization "
            "(ln_iceini=True, nn_iceini_file=0)"
        )
    if not cfg.ln_teos10:
        raise ValueError("O2 ORCA2_SAS_ICE requires ln_teos10=True")
    if cfg.ln_ice_embd:
        raise ValueError("O2 owns only ORCA2_SAS_ICE's levitating-ice branch")
    if cfg.ln_rstart:
        raise ValueError("O2 ice_istate is not entered on a restart")
    nn_hls = int(cfg.nn_hls)
    consts = cfg.ice_constants
    rn_thres_sst = float(cfg.rn_thres_sst)

    sst = jnp.asarray(sst_m, dtype=jnp.float64)
    sss = jnp.asarray(sss_m, dtype=jnp.float64)
    if _plant_stop_gradient:
        sst = jax.lax.stop_gradient(sst)
        sss = jax.lax.stop_gradient(sss)
    tmask_2d = jnp.asarray(
        tmask[..., 0] if jnp.ndim(tmask) == 3 else tmask, dtype=jnp.float64
    )
    ff_t_2d = jnp.asarray(ff_t, dtype=jnp.float64)
    gphit_2d = jnp.asarray(gphit, dtype=jnp.float64)
    e1e2t_2d = jnp.asarray(e1e2t, dtype=jnp.float64)
    dtype = sst.dtype
    jpi, jpj = sst.shape

    interior = _interior_mask_2d(jpi, jpj, nn_hls)
    smask0 = sr(tmask_2d * interior)
    ssmask_2d = jnp.asarray(ssmask, dtype=dtype)
    smask0_i_2d = jnp.asarray(smask0_i, dtype=dtype)
    if _plant_omit_nfd_uniq:
        smask0_i_2d = ssmask_2d * _interior_mask_2d(jpi, jpj, nn_hls)

    rt0 = sr(jnp.asarray(consts.T0, dtype=dtype))

    # 1. t_bo (`src/ICE/iceistate.F90:125-126`): kbnd=0 on interior, masked by smask0
    t_bo_degc = si3_eos_fzp(
        sss,
        ln_teos10=cfg.ln_teos10,
        kbnd=nn_hls if _plant_tbo_halo_leak else 0,
        nn_hls=nn_hls,
        _plant_wrong_eos=_plant_wrong_eos,
    )
    if _plant_tbo_halo_leak:
        t_bo = sr(sr(t_bo_degc + rt0) * tmask_2d)
    else:
        t_bo = sr(sr(t_bo_degc + rt0) * smask0)

    # 2. rn_amax_2d (`src/ICE/iceistate.F90:203-207`)
    amax_n = jnp.asarray(cfg.rn_amax_n, dtype=dtype)
    amax_s = jnp.asarray(cfg.rn_amax_s, dtype=dtype)
    rn_amax_2d = jnp.where(gphit_2d >= 0.0, amax_n, amax_s)

    # 3. ztfrz and zmsk (`src/ICE/iceistate.F90:263-266`)
    ztfrz = si3_eos_fzp(
        sss,
        ln_teos10=cfg.ln_teos10,
        kbnd=nn_hls,
        nn_hls=nn_hls,
        _plant_wrong_eos=_plant_wrong_eos,
    )
    thres = sr(jnp.asarray(rn_thres_sst, dtype=dtype))
    sst_minus_tfrz = sr(sr(sst - ztfrz) * tmask_2d)
    zmsk = jnp.where(sst_minus_tfrz >= thres, 0.0, tmask_2d)

    # 4. Hemisphere-dependent scalar state (`src/ICE/iceistate.F90:275-303`)
    def _val(attr: str) -> Field:
        v = getattr(cfg, attr)
        out = jnp.asarray(v, dtype=dtype)
        return jax.lax.stop_gradient(out) if _plant_stop_gradient else out

    v_hti_n = _val("rn_hti_ini_n")
    v_hti_s = _val("rn_hti_ini_s")
    if _plant_hemisphere_swap:
        v_hti_n, v_hti_s = v_hti_s, v_hti_n
    v_hts_n = _val("rn_hts_ini_n")
    v_hts_s = _val("rn_hts_ini_s")
    v_ati_n = _val("rn_ati_ini_n")
    v_ati_s = _val("rn_ati_ini_s")
    v_smi_n = _val("rn_smi_ini_n")
    v_smi_s = _val("rn_smi_ini_s")
    v_tmi_n = _val("rn_tmi_ini_n")
    v_tmi_s = _val("rn_tmi_ini_s")
    v_tsu_n = _val("rn_tsu_ini_n")
    v_tsu_s = _val("rn_tsu_ini_s")
    v_tms_n = _val("rn_tms_ini_n")
    v_tms_s = _val("rn_tms_ini_s")
    v_apd_n = _val("rn_apd_ini_n")
    v_apd_s = _val("rn_apd_ini_s")
    v_hpd_n = _val("rn_hpd_ini_n")
    v_hpd_s = _val("rn_hpd_ini_s")
    v_hld_n = _val("rn_hld_ini_n")
    v_hld_s = _val("rn_hld_ini_s")

    is_nh = ff_t_2d >= 0.0
    zht_i_ini = sr(jnp.where(is_nh, v_hti_n, v_hti_s) * zmsk)
    zht_s_ini = sr(jnp.where(is_nh, v_hts_n, v_hts_s) * zmsk)
    zat_i_ini = sr(jnp.where(is_nh, v_ati_n, v_ati_s) * zmsk)
    zsm_i_ini = sr(jnp.where(is_nh, v_smi_n, v_smi_s) * zmsk)
    ztm_i_ini = sr(jnp.where(is_nh, v_tmi_n, v_tmi_s) * zmsk)
    zt_su_ini = sr(jnp.where(is_nh, v_tsu_n, v_tsu_s) * zmsk)
    ztm_s_ini = sr(jnp.where(is_nh, v_tms_n, v_tms_s) * zmsk)
    zapnd_raw = sr(jnp.where(is_nh, v_apd_n, v_apd_s) * zmsk)
    zapnd_ini = zapnd_raw if _plant_pond_unscaled else sr(zapnd_raw * zat_i_ini)
    zhpnd_ini = sr(jnp.where(is_nh, v_hpd_n, v_hpd_s) * zmsk)
    zhlid_ini = sr(jnp.where(is_nh, v_hld_n, v_hld_s) * zmsk)
    if not cfg.ln_pnd:
        zapnd_ini = jnp.zeros_like(zapnd_ini)
        zhpnd_ini = jnp.zeros_like(zhpnd_ini)
        zhlid_ini = jnp.zeros_like(zhlid_ini)
    elif not cfg.ln_pnd_lids:
        zhlid_ini = jnp.zeros_like(zhlid_ini)

    # 5. Rung-O3 fit and fill (`iceistate.f90:356-434`).
    if not isinstance(state_config, SI3ORCA2StateConfig):
        raise TypeError("O2 requires an explicit SI3ORCA2StateConfig for the O3 fit")
    o3 = si3_o3_istate_fill_fields(
        zht_i_ini=zht_i_ini, zht_s_ini=zht_s_ini, zat_i_ini=zat_i_ini,
        ztm_i_ini=ztm_i_ini, ztm_s_ini=ztm_s_ini, zt_su_ini=zt_su_ini,
        zsm_i_ini=zsm_i_ini, zapnd_ini=zapnd_ini, zhpnd_ini=zhpnd_ini,
        zhlid_ini=zhlid_ini, tmask=tmask_2d, sss_m=sss, rn_amax_2d=rn_amax_2d,
        config=state_config,
    )
    if _control_exact_fitted is None:
        fitted = (o3.v_i, o3.v_s, o3.a_i, o3.a_ip, o3.v_ip, o3.v_il)
    else:
        fitted = tuple(jnp.asarray(x, dtype=dtype) for x in _control_exact_fitted)
    if _plant_stop_gradient:
        fitted = tuple(jax.lax.stop_gradient(x) for x in fitted)
    v_i_fit, v_s_fit, a_i_fit, a_ip_fit, v_ip_fit, v_il_fit = fitted
    expected_shape = (jpi, jpj, state_config.jpl)
    for name, arr in zip(("v_i", "v_s", "a_i", "a_ip", "v_ip", "v_il"), fitted):
        if arr.shape != expected_shape:
            raise ValueError(f"O2 requires fitted {name} shape {expected_shape}, got {arr.shape}")
    vt_i = fortran_sum_categories(v_i_fit, axis=-1)
    vt_s = fortran_sum_categories(v_s_fit, axis=-1)
    at_i = fortran_sum_categories(a_i_fit, axis=-1)
    at_ip = fortran_sum_categories(a_ip_fit, axis=-1)
    vt_ip = fortran_sum_categories(v_ip_fit, axis=-1)
    vt_il = fortran_sum_categories(v_il_fit, axis=-1)
    mass_by_category = sr(
        sr(sr(consts.rho_snow * v_s_fit) + sr(consts.rho_ice * v_i_fit))
        + sr(canonical_constants.rho_water * sr(v_ip_fit + v_il_fit))
    )
    snwice_mass = sr(tmask_2d * fortran_sum_categories(mass_by_category, axis=-1))

    # 6. Levitating-ice SSH adjustment (`iceistate.F90:408-427`).
    r1_rho0 = sr(
        jnp.asarray(1.0, dtype=dtype)
        / jnp.asarray(NEMO_CONSTANTS_CONFIG.rho_0, dtype=dtype)
    )
    area = si3_glob_2dsum_ddpdd(e1e2t_2d * ssmask_2d, smask0_i_2d, nn_hls=nn_hls)
    mass_area = sr(sr(snwice_mass * r1_rho0) * e1e2t_2d)
    zsshadj = sr(
        si3_glob_2dsum_ddpdd(mass_area, smask0_i_2d, nn_hls=nn_hls) / area
    )
    ssh_delta = -zsshadj
    snwice_mass_b = snwice_mass
    ssh_b0 = jnp.asarray(ssh_before_kbb, dtype=dtype)
    ssh_m0 = jnp.asarray(ssh_before_kmm, dtype=dtype)
    if _plant_stop_gradient:
        ssh_b0 = jax.lax.stop_gradient(ssh_b0)
        ssh_m0 = jax.lax.stop_gradient(ssh_m0)
    if _plant_ssh_sign:
        ssh_delta = -ssh_delta
    ssh_kmm = jnp.where(ssmask_2d == 1.0, sr(ssh_m0 + ssh_delta), ssh_m0)
    ssh_kbb = jnp.where(ssmask_2d == 1.0, sr(ssh_b0 + ssh_delta), ssh_b0)

    return SI3O2ScalarInitialState(
        t_bo=t_bo,
        ztfrz=ztfrz,
        zmsk=zmsk,
        zht_i_ini=zht_i_ini,
        zht_s_ini=zht_s_ini,
        zat_i_ini=zat_i_ini,
        zsm_i_ini=zsm_i_ini,
        ztm_i_ini=ztm_i_ini,
        zt_su_ini=zt_su_ini,
        ztm_s_ini=ztm_s_ini,
        zapnd_ini=zapnd_ini,
        zhpnd_ini=zhpnd_ini,
        zhlid_ini=zhlid_ini,
        rn_amax_2d=rn_amax_2d,
        vt_i=vt_i,
        vt_s=vt_s,
        at_i=at_i,
        at_ip=at_ip,
        vt_ip=vt_ip,
        vt_il=vt_il,
        snwice_mass=snwice_mass,
        snwice_mass_b=snwice_mass_b,
        area=area,
        zsshadj=zsshadj,
        ssh_kbb=ssh_kbb,
        ssh_kmm=ssh_kmm,
        o3=o3,
    )


def si3_o2_step(
    state: SI3State,
    forcing: SI3ORCA2O2Forcing,
    *,
    config: SI3ORCA2StateConfig,
    o2_init_config: SI3O2InitConfig,
    _plant_stop_gradient: bool = False,
) -> tuple[SI3State, SI3O2ScalarInitialState]:
    """Compute the Rung-O2 scalar initial state + levitating-ice SSH and execute one O1/O2 step."""
    cfg = config
    init_cfg = o2_init_config
    o2_istate = si3_o2_scalar_initial_state(
        sst_m=forcing.sst_m,
        sss_m=forcing.sss_m,
        tmask=forcing.tmask,
        ff_t=forcing.ff_t,
        gphit=forcing.gphit,
        e1e2t=forcing.e1e2t,
        ssmask=forcing.ssmask,
        smask0_i=forcing.smask0_i,
        ssh_before_kbb=forcing.ssh_before_kbb,
        ssh_before_kmm=forcing.ssh_before_kmm,
        config=init_cfg,
        state_config=cfg,
        _plant_stop_gradient=_plant_stop_gradient,
    )
    new_state = si3_o1_step(
        state,
        forcing.tmask,
        sss_m=forcing.sss_m,
        config=cfg,
        apply_halos=forcing.apply_halos,
        _plant_stop_gradient=_plant_stop_gradient,
    )
    return new_state, o2_istate


__all__.extend((
    "SI3O2InitConfig",
    "SI3O2ScalarInitialState",
    "SI3ORCA2O2Forcing",
    "si3_dom_uniq_t_orca2",
    "si3_glob_2dsum_ddpdd",
    "si3_eos_fzp",
    "si3_o2_scalar_initial_state",
    "si3_o2_step",
    "SI3O3ITDFit",
    "SI3O3IstateFields",
    "si3_var_itd_1cmc",
    "si3_o3_istate_fill_fields",
))
