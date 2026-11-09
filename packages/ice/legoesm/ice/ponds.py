"""Melt-pond tracking (CESM / topo-pond style).

State: per-category pond area fraction ``a_pond`` (of ice area)
and mean pond depth ``h_pond`` [m].  Volume: ``V_pond = a_pond ·
h_pond · a_ice`` (per unit grid-cell area).

Sources / sinks:

- **Melt + rain** on bare ice add water to pond volume (snow-on-ice
  intercepts precipitation; ponds appear only when the snow layer
  is depleted).
- **Drainage** to the ocean is a linear relaxation toward zero
  with timescale ``drainage_timescale``.
- **Refreezing** drains ponds when the air is colder than the
  refreeze threshold — water returns to the ice column.

These kernels are JAX-friendly and operate on arbitrary leading
spatial shape with optional trailing category axis.
"""

from __future__ import annotations

import jax.numpy as jnp

from legoesm.core.source_rounding import nemo_source_round
from legoesm.ice.config import MeltPondConfig, SI3PondConfig
from legoesm.ice.constants_config import IceConstantsConfig
from legoesm.ice.si3_thd_pnd import si3_ice_thd_pnd

# Canonical melt-pond defaults (single source of truth callers pass in).
_POND_DEFAULTS = MeltPondConfig()


def step_ponds(
    pond_area: jnp.ndarray,
    pond_depth: jnp.ndarray,
    *,
    melt_water_m: jnp.ndarray,
    rain_water_m: jnp.ndarray,
    ice_mask: jnp.ndarray,
    h_snow: jnp.ndarray,
    T_air: jnp.ndarray,
    dt: float,
    drainage_timescale: float,
    refreeze_threshold: float,
    pond_to_ice_max_area: float,
    depth_to_area_ratio: float,
    snow_block_threshold: float = _POND_DEFAULTS.snow_block_threshold,
    refreeze_width_K: float = _POND_DEFAULTS.refreeze_width_K,
    scheme: str = _POND_DEFAULTS.scheme,
    at_i: jnp.ndarray | None = None,
    a_i: jnp.ndarray | None = None,
    h_i: jnp.ndarray | None = None,
    t_su: jnp.ndarray | None = None,
    h_il: jnp.ndarray | None = None,
    dh_i_sum: jnp.ndarray | None = None,
    dh_s_sum: jnp.ndarray | None = None,
    sz_i: jnp.ndarray | None = None,
    t_i: jnp.ndarray | None = None,
    namthd_pnd: SI3PondConfig | None = None,
    rho_w: float | None = None,
    gravity: float | None = None,
    ice_constants: IceConstantsConfig | None = None,
    v_i: jnp.ndarray | None = None,
    a_ip: jnp.ndarray | None = None,
    v_ip: jnp.ndarray | None = None,
    v_il: jnp.ndarray | None = None,
    wfx_pnd: jnp.ndarray | None = None,
    nn_hls: int = 0,
    _return_si3_outputs: bool = False,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Advance pond area + depth one time step.

    Parameters
    ----------
    pond_area, pond_depth : arrays
        Current pond area fraction [0-1] (relative to ice area)
        and mean depth [m].
    melt_water_m : array
        Water input from surface ice / snow melt this step [m of
        liquid water].
    rain_water_m : array
        Water input from precipitation this step [m liquid].
    ice_mask : bool array
        True where ice exists.
    h_snow : array
        Current snow depth — ponds only collect melt once snow is
        gone.
    T_air : array
        Reference air temperature [K] (lowest atmospheric layer).
        Used to gate the refreezing path.
    dt : float
        Time step [s].
    drainage_timescale : float
        Linear drainage e-folding time [s].
    refreeze_threshold : float
        Air-T below which ponds refreeze [K].
    pond_to_ice_max_area : float
        Maximum pond area fraction.
    depth_to_area_ratio : float
        ``h_pond / a_pond`` ratio assumed for converting added
        volume into the (area, depth) split.
    refreeze_width_K : float
        Half-width [K] of the smooth linear ramp over which ponds
        transition from fully liquid to fully refrozen as ``T_air``
        crosses ``refreeze_threshold``.  Keeps the refreeze response
        differentiable; smaller → sharper (CICE-like) cutoff.

    Returns
    -------
    pond_area_new, pond_depth_new : arrays
        Updated pond state.
    drain_to_ocean_m : array
        Pond water lost to drainage this step [m of liquid water
        per unit ICE area] — pond state is stored relative to the
        ice area (``V_pond = pond_area * pond_depth``), so this is
        per-ice-area.  The caller multiplies by the ice concentration
        ``conc`` before routing it to the ocean freshwater channel.
    refreeze_volume_m : array
        Pond water that refroze back into ice [m of liquid water
        per unit ICE area] — caller converts it to an ice-thickness
        gain on the per-ice-area column ``h`` (white-ice gain; no FW
        exchange with the ocean).
    """
    if scheme not in ("cesm", "si3_lev"):
        raise ValueError(
            f"Unknown MeltPondConfig.scheme={scheme!r}; expected 'cesm' or 'si3_lev'"
        )
    if scheme == "si3_lev":
        if not isinstance(namthd_pnd, SI3PondConfig):
            raise ValueError(
                "step_ponds(scheme='si3_lev') requires MeltPondConfig.namthd_pnd "
                "to be an explicit SI3PondConfig; no defaults"
            )
        if any(
            x is None
            for x in (
                at_i,
                a_i,
                h_i,
                t_su,
                h_il,
                dh_i_sum,
                dh_s_sum,
                sz_i,
                t_i,
                rho_w,
                gravity,
                ice_constants,
            )
        ):
            raise ValueError(
                "step_ponds(scheme='si3_lev') requires at_i, a_i, h_i, t_su, "
                "h_il, melt totals, sz_i, t_i, rho_w, gravity, and ice_constants"
            )
        a_i_f64 = jnp.asarray(a_i, dtype=jnp.float64)
        pond_area_f64 = jnp.asarray(pond_area, dtype=jnp.float64)
        h_ip_in = jnp.asarray(pond_depth, dtype=jnp.float64)
        h_il_in = jnp.asarray(h_il, dtype=jnp.float64)
        h_i_in = jnp.asarray(h_i, dtype=jnp.float64)
        t_su_in = jnp.asarray(t_su, dtype=jnp.float64)
        dh_i_sum_in = jnp.asarray(dh_i_sum, dtype=jnp.float64)
        dh_s_sum_in = jnp.asarray(dh_s_sum, dtype=jnp.float64)
        at_i_in = jnp.asarray(at_i, dtype=jnp.float64)
        a_ip_in = (
            nemo_source_round(pond_area_f64 * a_i_f64)
            if a_ip is None
            else jnp.asarray(a_ip, dtype=jnp.float64)
        )
        v_i_in = (
            nemo_source_round(h_i_in * a_i_f64)
            if v_i is None
            else jnp.asarray(v_i, dtype=jnp.float64)
        )
        v_ip_in = (
            nemo_source_round(h_ip_in * a_ip_in)
            if v_ip is None
            else jnp.asarray(v_ip, dtype=jnp.float64)
        )
        v_il_in = (
            nemo_source_round(h_il_in * a_ip_in)
            if v_il is None
            else jnp.asarray(v_il, dtype=jnp.float64)
        )
        sz_i_f64 = jnp.asarray(sz_i, dtype=jnp.float64)
        t_i_f64 = jnp.asarray(t_i, dtype=jnp.float64)
        if sz_i_f64.ndim == h_ip_in.ndim:
            sz_i_f64 = sz_i_f64[..., None]
        if t_i_f64.ndim == h_ip_in.ndim:
            t_i_f64 = t_i_f64[..., None]
        squeeze_cat = a_i_f64.ndim == 2
        if squeeze_cat:
            a_i_3d = a_i_f64[..., None]
            v_i_3d = v_i_in[..., None]
            h_i_3d = h_i_in[..., None]
            t_su_3d = t_su_in[..., None]
            a_ip_3d = a_ip_in[..., None]
            v_ip_3d = v_ip_in[..., None]
            h_ip_3d = h_ip_in[..., None]
            v_il_3d = v_il_in[..., None]
            h_il_3d = h_il_in[..., None]
            a_ip_frac_3d = pond_area_f64[..., None]
            dh_i_sum_3d = dh_i_sum_in[..., None]
            dh_s_sum_3d = dh_s_sum_in[..., None]
            sz_i_4d = sz_i_f64[..., None]
            t_i_4d = t_i_f64[..., None]
            at_i_2d = at_i_in
        else:
            a_i_3d = a_i_f64
            v_i_3d = v_i_in
            h_i_3d = h_i_in
            t_su_3d = t_su_in
            a_ip_3d = a_ip_in
            v_ip_3d = v_ip_in
            h_ip_3d = h_ip_in
            v_il_3d = v_il_in
            h_il_3d = h_il_in
            a_ip_frac_3d = pond_area_f64
            dh_i_sum_3d = dh_i_sum_in
            dh_s_sum_3d = dh_s_sum_in
            if sz_i_f64.shape[2] == a_i_3d.shape[2] and sz_i_f64.shape[3] != a_i_3d.shape[2]:
                sz_i_4d = jnp.transpose(sz_i_f64, (0, 1, 3, 2))
                t_i_4d = jnp.transpose(t_i_f64, (0, 1, 3, 2))
            else:
                sz_i_4d = sz_i_f64
                t_i_4d = t_i_f64
            at_i_2d = at_i_in if at_i_in.ndim == 2 else None
        wfx_2d = (
            jnp.zeros(a_i_3d.shape[:2], dtype=jnp.float64)
            if wfx_pnd is None
            else jnp.asarray(wfx_pnd, dtype=jnp.float64)
        )
        dt_f64 = jnp.asarray(dt, dtype=jnp.float64)
        r1_dt = nemo_source_round(jnp.float64(1.0) / dt_f64)
        nml = {
            "ln_icedH": True,
            "ln_pnd": True,
            "ln_pnd_LEV": True,
            "rn_apnd_min": namthd_pnd.rn_apnd_min,
            "rn_apnd_max": namthd_pnd.rn_apnd_max,
            "rn_pnd_flush": namthd_pnd.rn_pnd_flush,
            "rn_himin": namthd_pnd.rn_himin,
            "ln_pnd_lids": namthd_pnd.ln_pnd_lids,
            "ln_pnd_rain": True,
            "ln_pnd_alb": True,
            "ln_pnd_CST": False,
            "ln_pnd_TOPO": False,
            "rn_apnd": 0.0,
            "rn_hpnd": 0.0,
            "rn_pnd_hl_min": 0.0,
            "rn_pnd_hl_max": 0.0,
            "ln_icediachk": False,
        }
        pnd_out = si3_ice_thd_pnd(
            a_i=a_i_3d,
            v_i=v_i_3d,
            h_i=h_i_3d,
            t_su=t_su_3d,
            a_ip=a_ip_3d,
            v_ip=v_ip_3d,
            h_ip=h_ip_3d,
            v_il=v_il_3d,
            h_il=h_il_3d,
            a_ip_frac=a_ip_frac_3d,
            dh_i_sum_2d=dh_i_sum_3d,
            dh_s_sum_2d=dh_s_sum_3d,
            t_i=t_i_4d,
            sz_i=sz_i_4d,
            wfx_pnd=wfx_2d,
            namelist=nml,
            rDt_ice=dt_f64,
            r1_Dt_ice=r1_dt,
            rho_w=rho_w,
            gravity=gravity,
            nn_hls=nn_hls,
            constants=ice_constants,
            _allow_general_shape=not _return_si3_outputs,
            _at_i_override=None if _return_si3_outputs else at_i_2d,
        )
        if _return_si3_outputs:
            return pnd_out
        a_ip_out = pnd_out.a_ip[..., 0] if squeeze_cat else pnd_out.a_ip
        h_ip_out = pnd_out.h_ip[..., 0] if squeeze_cat else pnd_out.h_ip
        h_il_out = pnd_out.h_il[..., 0] if squeeze_cat else pnd_out.h_il
        wfx_out = pnd_out.wfx_pnd if squeeze_cat else pnd_out.wfx_pnd[..., None]
        has_ice = a_i_f64 > 0.0
        safe_ai = jnp.where(has_ice, a_i_f64, jnp.ones_like(a_i_f64))
        pond_area_new = jnp.where(
            has_ice,
            nemo_source_round(a_ip_out / safe_ai),
            jnp.zeros_like(a_ip_out),
        )
        pond_depth_new = h_ip_out
        h_il_new = h_il_out
        rho_w_f64 = jnp.asarray(rho_w, dtype=jnp.float64)
        drain_cell_m = nemo_source_round(
            nemo_source_round(wfx_out * dt_f64) / rho_w_f64
        )
        drain_to_ocean_m = jnp.where(
            has_ice,
            nemo_source_round(drain_cell_m / safe_ai),
            jnp.zeros_like(pond_depth_new),
        )
        refreeze_m = jnp.zeros_like(pond_depth_new)
        return (
            pond_area_new.astype(pond_area.dtype),
            pond_depth_new.astype(pond_depth.dtype),
            drain_to_ocean_m.astype(pond_depth.dtype),
            refreeze_m.astype(pond_depth.dtype),
            h_il_new.astype(jnp.asarray(h_il).dtype),
        )
    if namthd_pnd is not None:
        raise ValueError(
            "step_ponds(scheme='cesm') does not accept namthd_pnd; "
            "set MeltPondConfig.scheme='si3_lev'"
        )
    # Input water this step.  Snow thicker than the configured
    # threshold blocks pond formation; thinner snow / deposition
    # patches do not.
    snow_mask = h_snow > snow_block_threshold
    input_m = jnp.where(
        ice_mask & (~snow_mask),
        jnp.maximum(melt_water_m + rain_water_m, 0.0),
        0.0,
    )

    # Current pond water volume per unit *ice* area (m³_water / m²_ice).
    V_pond = pond_area * pond_depth

    # Refreezing: when T_air < threshold, drain everything back to
    # ice.  Smooth ramp over ``refreeze_width_K`` for differentiability.
    refreeze_fraction = jnp.clip(
        (refreeze_threshold - T_air) / refreeze_width_K,
        0.0,
        1.0,
    )
    V_refreeze = V_pond * refreeze_fraction
    V_after_refreeze = jnp.maximum(V_pond - V_refreeze, 0.0)

    # Drainage: linear relaxation.
    drainage_fraction = 1.0 - jnp.exp(-dt / jnp.maximum(drainage_timescale, 1.0))
    V_drain = V_after_refreeze * drainage_fraction
    V_after_drain = jnp.maximum(V_after_refreeze - V_drain, 0.0)

    # Add new melt + rain water.
    V_new = V_after_drain + input_m

    # Recover (a_pond, h_pond) under the ratio constraint:
    #   h_pond = sqrt(depth_to_area_ratio · V_new)
    #   a_pond = V_new / max(h_pond, 1e-6)
    # Floor the sqrt argument at a tiny positive (not 0): at an empty pond
    # (V_new = 0 — the common no-melt / cold-season state) sqrt'(0) = inf gives
    # a NaN reverse-mode gradient through h_new.  The forward is unchanged for
    # any real pond (and at V_new=0 the downstream ``max(h_new, 1e-6)`` and
    # ``a_new = V_new/h_safe = 0`` are identical).  Mirrors the sea_ice.py
    # pond_depth fix (iter 2).
    h_new = jnp.sqrt(jnp.maximum(depth_to_area_ratio * V_new, 1e-14))
    h_safe = jnp.maximum(h_new, 1e-6)
    a_new = V_new / h_safe
    a_new = jnp.clip(a_new, 0.0, pond_to_ice_max_area)

    # If we hit the area cap, fold residual volume into depth.
    capped = a_new >= pond_to_ice_max_area
    a_new = jnp.where(capped, pond_to_ice_max_area, a_new)
    h_new = jnp.where(capped, V_new / jnp.maximum(a_new, 1e-6), h_new)

    return a_new, h_new, V_drain, V_refreeze
