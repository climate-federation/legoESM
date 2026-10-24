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
from typing import TYPE_CHECKING, Any, Mapping, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from legoesm import constants as canonical_constants
from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.constants_config import (
    NEMO_SI3_CONSTANTS_CONFIG,
    IceConstantsConfig,
)

if TYPE_CHECKING:
    from legoesm.ice.state import SI3ColumnState

Field = jax.Array

_EPSI20 = 1.0e-20
_ZSI0 = 3.5
_ZSI1 = 4.5
_HI_MAX_SENTINEL = 999.99

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
    rn_himax: float = 99.0
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
        sss_m: Field | float = 34.0,
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
            t_si=z3,
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
            t1_ice=z3,
            t_si=z3,
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

    init_z1 = jnp.asarray(0.05, dtype=jnp.float64)
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
    rn_himax: float | Field = 99.0,
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
    zalpha = jnp.asarray(0.05, dtype=jnp.float64)
    inv_zalpha = jnp.asarray(np.float64(np.float32(1.0)) / np.float64(0.05), dtype=jnp.float64)
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
        s_out = jnp.full_like(s_i, 2.30)
        rpi = jnp.asarray(float(canonical_constants.PI), dtype=dtype)
        layers = []
        for jk in range(1, nlay_i + 1):
            ztmp1 = sr(jnp.asarray(float(jk) - 0.5, dtype=dtype) * r1_nlay_i)
            exponent = sr(0.407 / sr(0.573 + ztmp1))
            ztmp2 = sr(1.6 * sr(1.0 - jnp.cos(sr(rpi * jnp.power(ztmp1, exponent)))))
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
        raw = sr(1.0 - jnp.exp(sr(sr(-0.2 * consts.rho_snow) * h_s)))
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
    1. `store_fields` (`:148`, `:361-394`)
    2. `CALL ice_var_glo2eqv(2)` (`:218`)
    3. `CALL ice_var_agg(2)` (`:219`)
    4. `CALL ice_update_flx(kt)` snow/ice mass update (`src/ICE/iceupdate.F90:189-194`)
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    work = jax.lax.stop_gradient(state) if _plant_stop_gradient else state
    if apply_halos:
        work = si3_apply_state_halos_orca2(work, config=cfg)

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
) -> dict[str, np.ndarray]:
    """Serialize `SI3State` to NEMO's exact `ice_rst_write` + `update_rst` + `adv_pra_rst` dict.

    Statement order matches:
    1. `update_rst('WRITE', kt)` (`src/ICE/iceupdate.F90:474-481`)
    2. `adv_pra_rst('WRITE', kt)` (`src/ICE/icedyn_adv_pra.F90:1383-1503`)
    3. `ice_rst_write(kt)` (`src/ICE/icerst.F90:137-182`)

    Note: NetCDF variables in NEMO's IOM (`iom_nf90`) are stored in `(t, cat, y, x)`
    C-order (corresponding to Fortran `(x, y, cat, t)`). Here each spatial array is
    stored in Fortran-logical `(jpi, jpj, ...)` order so direct array comparisons are
    shape-preserving; `write_si3_netcdf_restart` / `read_si3_netcdf_restart` handle
    both native NEMO `(y, x)` and `(x, y)` conventions.
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    iter_val = float(kt + cfg.nn_fsbc - 1)
    out: dict[str, np.ndarray] = {}

    # 1. update_rst ('WRITE', kt)
    out["snwice_mass"] = np.asarray(state.snwice_mass, dtype=np.float64)
    out["snwice_mass_b"] = np.asarray(state.snwice_mass_b, dtype=np.float64)

    # 2. adv_pra_rst ('WRITE', kt)
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
) -> SI3State:
    """Reconstruct `SI3State` in NEMO's exact `ice_init` restart-read sequence.

    Follows NEMO's statement order in `src/ICE/icestp.F90:282-300`:
    1. `ice_itd_init` (`src/ICE/iceitd.F90:715-780`)
    2. `ice_rst_read` (`src/ICE/icerst.F90:200-378`)
    3. `ice_var_glo2eqv(1)` (`src/ICE/icestp.F90:295`)
    4. `ice_var_agg(1)` (`src/ICE/icestp.F90:296`)
    5. `adv_pra_rst('READ')` (`src/ICE/icedyn_adv_pra.F90:1203-1381`)
    6. `update_rst('READ')` (`src/ICE/iceupdate.F90:453-473`)
    """
    cfg = config if config is not None else SI3ORCA2StateConfig()
    consts = cfg.ice_constants
    sr = nemo_source_round

    v_i = jnp.asarray(rst["v_i"], dtype=jnp.float64)
    jpi, jpj, jpl = v_i.shape
    if jpl != cfg.jpl:
        raise ValueError(f"Restart jpl={jpl} does not match config.jpl={cfg.jpl}")

    v_s = jnp.asarray(rst["v_s"], dtype=jnp.float64)
    sv_i = jnp.asarray(rst["sv_i"], dtype=jnp.float64)
    a_i = jnp.asarray(rst["a_i"], dtype=jnp.float64)
    t_su = jnp.asarray(rst["t_su"], dtype=jnp.float64)
    u_ice = jnp.asarray(rst["u_ice"], dtype=jnp.float64)
    v_ice = jnp.asarray(rst["v_ice"], dtype=jnp.float64)

    if "oa_i" in rst:
        oa_i = jnp.asarray(rst["oa_i"], dtype=jnp.float64)
    else:
        oa_i = jnp.zeros_like(v_i)

    if "a_ip" in rst:
        a_ip = jnp.asarray(rst["a_ip"], dtype=jnp.float64)
        v_ip = jnp.asarray(rst["v_ip"], dtype=jnp.float64)
        if "v_il" in rst and cfg.ln_pnd_lids:
            v_il = jnp.asarray(rst["v_il"], dtype=jnp.float64)
        else:
            v_il = jnp.zeros_like(v_i)
    else:
        a_ip = jnp.zeros_like(v_i)
        v_ip = jnp.zeros_like(v_i)
        v_il = jnp.zeros_like(v_i)

    if "e_s_l01" in rst:
        e_s = jnp.stack(
            [
                jnp.asarray(rst[f"e_s_l{jk:02d}"], dtype=jnp.float64)
                for jk in range(1, cfg.nlay_s + 1)
            ],
            axis=2,
        )
    else:
        e_s_val = sr(consts.rho_snow * (consts.latent_fusion - consts.c_ice * -2.0))
        e_s = jnp.broadcast_to(
            sr(e_s_val * (v_s[..., None, :] / float(cfg.nlay_s))),
            (jpi, jpj, cfg.nlay_s, jpl),
        )

    if "e_i_l01" in rst:
        e_i = jnp.stack(
            [
                jnp.asarray(rst[f"e_i_l{jk:02d}"], dtype=jnp.float64)
                for jk in range(1, cfg.nlay_i + 1)
            ],
            axis=2,
        )
    else:
        e_i = jnp.zeros((jpi, jpj, cfg.nlay_i, jpl), dtype=jnp.float64)

    if "szv_i_l01" in rst:
        szv_i = jnp.stack(
            [
                jnp.asarray(rst[f"szv_i_l{jk:02d}"], dtype=jnp.float64)
                for jk in range(1, cfg.nlay_i + 1)
            ],
            axis=2,
        )
    else:
        szv_i = jnp.broadcast_to(
            sr(sv_i[..., None, :] / float(cfg.nlay_i)),
            (jpi, jpj, cfg.nlay_i, jpl),
        )

    cnd_ice = (
        jnp.asarray(rst["cnd_ice"], dtype=jnp.float64)
        if ("cnd_ice" in rst and cfg.ln_cpl)
        else jnp.zeros_like(v_i)
    )
    t1_ice = (
        jnp.asarray(rst["t1_ice"], dtype=jnp.float64)
        if ("t1_ice" in rst and cfg.ln_cpl)
        else jnp.full_like(v_i, consts.T0)
    )
    t_si = jnp.zeros_like(v_i)

    # Prather moments (`adv_pra_rst('READ')`, src/ICE/icedyn_adv_pra.F90:1203-1381)
    pr_zeros = SI3AllPratherMoments.zeros(
        jpi, jpj, jpl=cfg.jpl, nlay_i=cfg.nlay_i, nlay_s=cfg.nlay_s
    )
    if "sxice" in rst:
        fam_dict: dict[str, SI3PratherMoments] = {}
        for attr_name, suffix in _PRATHER_3D_FAMILIES:
            if f"sx{suffix}" in rst:
                fam_dict[attr_name] = SI3PratherMoments(
                    sx=jnp.asarray(rst[f"sx{suffix}"], dtype=jnp.float64),
                    sy=jnp.asarray(rst[f"sy{suffix}"], dtype=jnp.float64),
                    sxx=jnp.asarray(rst[f"sxx{suffix}"], dtype=jnp.float64),
                    syy=jnp.asarray(rst[f"syy{suffix}"], dtype=jnp.float64),
                    sxy=jnp.asarray(rst[f"sxy{suffix}"], dtype=jnp.float64),
                )
            else:
                fam_dict[attr_name] = getattr(pr_zeros, attr_name)

        for attr_name, suffix, nlay_attr in _PRATHER_4D_FAMILIES:
            nlay = getattr(cfg, nlay_attr)
            if f"sx{suffix}_l01" in rst:
                fam_dict[attr_name] = SI3PratherMoments(
                    sx=jnp.stack(
                        [
                            jnp.asarray(rst[f"sx{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    sy=jnp.stack(
                        [
                            jnp.asarray(rst[f"sy{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    sxx=jnp.stack(
                        [
                            jnp.asarray(rst[f"sxx{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    syy=jnp.stack(
                        [
                            jnp.asarray(rst[f"syy{suffix}_l{jk:02d}"], dtype=jnp.float64)
                            for jk in range(1, nlay + 1)
                        ],
                        axis=2,
                    ),
                    sxy=jnp.stack(
                        [
                            jnp.asarray(rst[f"sxy{suffix}_l{jk:02d}"], dtype=jnp.float64)
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
    tmask_arr = jnp.asarray(tmask, dtype=jnp.float64)
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
    if "snwice_mass" in rst and "snwice_mass_b" in rst:
        snwice_mass = jnp.asarray(rst["snwice_mass"], dtype=jnp.float64)
        snwice_mass_b = jnp.asarray(rst["snwice_mass_b"], dtype=jnp.float64)
        snwice_fmass = jnp.zeros_like(snwice_mass)
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
) -> Path:
    """Write a NEMO-compatible `*_restart_ice.nc` file using `netCDF4`."""
    import netCDF4

    cfg = config if config is not None else SI3ORCA2StateConfig()
    out_path = Path(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    rst_dict = si3_rst_to_netcdf_dict(state, kt=kt, config=cfg)
    jpi, jpj, jpl = state.v_i.shape

    with netCDF4.Dataset(out_path, "w", format="NETCDF4_CLASSIC") as ds:
        ds.createDimension("x", jpi)
        ds.createDimension("y", jpj)
        ds.createDimension("numcat", jpl)
        ds.createDimension("time_counter", None)

        for name, val in rst_dict.items():
            arr = np.asarray(val, dtype=np.float64)
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
) -> SI3State:
    """Read a NEMO `*_restart_ice.nc` file and reconstruct `SI3State`."""
    import netCDF4

    cfg = config if config is not None else SI3ORCA2StateConfig()
    rst_dict: dict[str, np.ndarray] = {}
    with netCDF4.Dataset(Path(path), "r") as ds:
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
    )


__all__ = [
    "SI3ORCA2StateConfig",
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
    "si3_rst_to_netcdf_dict",
    "si3_rst_from_netcdf_dict",
    "write_si3_netcdf_restart",
    "read_si3_netcdf_restart",
]
