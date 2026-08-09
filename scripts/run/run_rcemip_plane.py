"""RCEMIP1 (Wing et al. 2018) RCE harness on the plane NH dycore.

Reference: doi:10.5194/gmd-11-793-2018.

Scope
-----
RCEMIP-style radiative-convective equilibrium on the plane
non-hydrostatic dycore (PR2b-PR3d) wired with the same factory
dispatch the cubed-sphere / MPAS NH harnesses use:

- Bulk surface fluxes via :func:`legoesm.core.bulk_flux.simple_bulk_fluxes`
- Radiation via :func:`legoesm.atmosphere.physics.radiation.integration.make_radiation_physics`
  with ``model_type="plane"`` (selects gray or RRTMGP from the
  RadiationConfig scheme literal)
- Microphysics via
  :func:`legoesm.atmosphere.physics.microphysics.integration.make_microphysics_physics`
  with ``model_type="plane"`` (selects kessler, morrison, sundqvist,
  seifert_beheng, thompson, ml_emulator, or "none" from the
  MicrophysicsConfig scheme literal)
- Smagorinsky LES (PR3c, horizontal-only pilot)
- Hyperdiffusion (PR3a) + sponge (PR2d) + upwind advection (PR3b)
- ``n_tracers >= 3`` for q_v / q_c / q_r (PR3d)

CLI
---
.. code-block:: bash

   JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \\
       scripts/run_rcemip_plane.py \\
       --nx 16 --ny 16 --nlev 30 --dx 4000.0 --dt 6.0 \\
       --steps 50 --radiation gray --microphysics kessler \\
       --output results/rcemip_smoke
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np

from legoesm import constants
from legoesm.atmosphere.dynamics.gcm.compressible_euler import (
    CompressibleEulerConfig,
)
from legoesm.atmosphere.dynamics.les.compressible_euler_plane import (
    PlaneCompressibleEulerModel,
    compute_dry_mass_plane,
    make_flat_plane_terrain_metric,
    make_rest_state,
)
from legoesm.atmosphere.dynamics.shared.tracer_positivity import (
    apply_positive_filter_state,
)
from legoesm.atmosphere.idealized.land_rce import (
    apply_plane_tendency_forward_euler,
    land_surface_flux_tendencies,
    sum_plane_tendencies,
    update_land_slab_temperature,
)
from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
    make_wing2018_theta_ref_fn,
    wing2018_qv_profile,
    WING_Q_SFC_DEFAULT,
)
from legoesm.atmosphere.physics.microphysics.config import MicrophysicsConfig
from legoesm.atmosphere.physics.microphysics.integration import (
    make_microphysics_physics,
)
from legoesm.atmosphere.physics.radiation.config import RadiationConfig
from legoesm.atmosphere.physics.radiation.integration import (
    make_radiation_physics,
)
from legoesm.core.field import Field
from legoesm.core.state import PlaneNonHydrostaticTendencies
from legoesm.grids.plane import create_plane_grid
from legoesm.grids.vertical import (
    create_height_coordinate,
    create_stretched_height_coordinate,
)


# JAX x64 toggle happens at IMPORT TIME (before argparse). Two paths:
# 1. LEGOESM_RCEMIP_PLANE_FP32=1 in the env -> x64 stays OFF -> fp32
#    arithmetic stays fp32. This is the supported fp32 path.
# 2. Anything else -> x64 ON -> fp64 default (and fp32 arrays will
#    auto-promote to fp64 if mixed with any fp64 literal).
# Per codex iter-... HIGH#3: a previous --precision float32 flag was
# DEAD because the module-level toggle ran before argparse. We deleted
# the broken _enable_x64_if_needed shim and now require the env var.
# main() will refuse --precision float32 without the env var to make
# the contract explicit.
import os as _os
if _os.environ.get("LEGOESM_RCEMIP_PLANE_FP32") != "1":
    jax.config.update("jax_enable_x64", True)


# -------- RCEMIP1 IC (Wing 2018 Tab A1, simplified) -------- #


def _rcemip_theta_profile(z: jax.Array, T_sfc: float = 300.0,
                          q_sfc: float = WING_Q_SFC_DEFAULT) -> jax.Array:
    """RCEMIP/Wing 2018 θ(z) — delegates to the library `make_wing2018_theta_ref_fn`.

    CONV-TRIGGER #83 FIX (iter-61): the old local formula `θ = T_sfc + Γ·z`
    (Γ=6.7e-3) MIS-USED Γ — 6.7 K/km is the Wing 2018 *temperature* (virtual-T)
    lapse rate (Tab A1: `T_v = T_v0 − Γ·z`), NOT a θ gradient. Used as a θ gradient
    it gave an env T-lapse of ~−4.3 K/km (vs realistic tropical ~−6.5) ⇒ FAR too
    stable ⇒ near-zero deep CAPE ⇒ RCE stayed LAMINAR. The library function builds θ
    correctly from the Wing virtual-T profile + hydrostatic integration (T-lapse
    ~−6.1 K/km, isothermal stratosphere ⇒ θ rises) — conditionally unstable, the
    proper RCEMIP IC. VERIFIED: makes RCE DEEP-CONVECT — max|w| reaches ~5 m/s by
    ~9 sim-hours (vs laminar ~0.05 m/s with the buggy profile), mass-conserving with
    fix_mass=True (production). De-duplicates per the no-duplicate-numerics rule.

    ``q_sfc``: the Wing θ is built on a *virtual*-T hydrostatic base, so it
    MUST use the SAME surface humidity as :func:`_rcemip_qv_profile`. Threaded
    as an explicit arg (not the library default) so a single ``q_sfc`` drives
    both θ and q_v from one source.

    ``T_sfc`` IS NOT USED and is retained only because callers pass it: it is
    the prescribed SEA-surface temperature (the flux boundary condition), and
    ``T_v0`` is the surface AIR virtual temperature, which gSAM's RCE300
    sounding puts ~3 K BELOW the SST.  Deriving one from the other
    (``T_v0 = T_sfc·(1+0.608·q_sfc)``, as an earlier revision of this docstring
    claimed) overshoots the oracle by ~2.7 K; pinning ``T_v0`` to a fixed 295 K
    for every SST undershoots it by ~5 K and made the column supersaturated.
    ``WING_T_V0`` is now calibrated against the gSAM sounding — see the
    constant's note — and the SST enters only through ``--T-sfc``.
    """
    # SCOPE WARNING (#1507 codex P1): WING_T_V0 / WING_GAMMA / WING_Q_SFC_DEFAULT
    # are calibrated against the gSAM **RCE300** sounding, and they are MODULE
    # DEFAULTS -- every defaulted caller gets them, including the SCM campaign
    # and the RCE295 / RCE305 cases. Discarding T_sfc here is right (the SST is
    # a boundary condition, not the IC), but it means a non-300 K run silently
    # receives RCE300-calibrated ATMOSPHERIC coefficients. Say so once, loudly,
    # rather than let the calibration travel unannounced; pass explicit
    # T_v0/Gamma/q_sfc, or use --sounding, for the other SSTs.
    if abs(float(T_sfc) - 300.0) > 0.5:
        import warnings
        warnings.warn(
            f"RCEMIP analytic IC: T_sfc={float(T_sfc):.1f} K but the profile "
            f"coefficients (T_v0={WING_T_V0}, Gamma={WING_GAMMA}, "
            f"q_sfc={WING_Q_SFC_DEFAULT}) are calibrated against the gSAM "
            f"RCE300 sounding. The atmospheric profile is therefore RCE300's, "
            f"not this SST's. Pass explicit coefficients or --sounding for "
            f"RCE295/RCE305 (#1507).",
            stacklevel=2)
    del T_sfc  # documented above: the SST is a boundary condition, not the IC
    return make_wing2018_theta_ref_fn(q_sfc=float(q_sfc))(z)


def _rcemip_qv_profile(z: jax.Array,
                       q_sfc: float = WING_Q_SFC_DEFAULT) -> jax.Array:
    """RCEMIP/Wing 2018 q_v(z) — delegates to the library `wing2018_qv_profile`
    (CONV-TRIGGER #83: two-scale `exp(−z/z_q1)·exp(−(z/z_q2)²)` Wing profile,
    consistent with the virtual-T used by `_rcemip_theta_profile`; the old local
    single-scale `exp(−z/4km)` was an inconsistent simplification)."""
    return wing2018_qv_profile(z, q_sfc=float(q_sfc))


# -------- Surface-flux physics_fn (bulk_flux only) -------- #


def _make_surface_flux_physics(
    grid, height_coord, terrain_metric,
    T_sfc: float, p_sfc: float, wd: float = 0.0,
):
    """Lowest-level SAM ``oceflx`` surface fluxes (iter-5 SF fix).

    Returns a ``physics_fn(state, grid, hc, tm) ->
    PlaneNonHydrostaticTendencies`` whose only non-zero tendencies
    are momentum drag + sensible-heat + latent-heat at the lowest
    model level (``k = nlev - 1`` under top-down indexing).

    Uses the faithful SAM bulk scheme
    :func:`legoesm.core.bulk_flux.compute_sam_oceflx_fluxes`
    (iterative Monin–Obukhov, SAM Stanton/Dalton/cdn coefficients) with
    the SAM ocean surface humidity ``q_sfc = 0.981·qsat(SST)``
    (:func:`sam_ocean_surface_q`) — replacing the previous fixed-Cd/Ch
    ``simple_bulk_fluxes`` + hardcoded ``q_sfc = 0.018`` (audit SF-1/SF-2:
    the hardcoded value biased the latent-heat flux ~20 % low and broke
    the WISHE feedback). Gust ``wd = 0`` reproduces SAM's
    ``vmag = max(1, |U|)`` (NOT the Wing-2018 5 m/s gust floor — SF-3).
    """
    from legoesm.core.bulk_flux import (
        compute_sam_oceflx_fluxes, sam_ocean_surface_q,
    )

    # SAM ocean surface saturation humidity (salt-reduced) — constant in
    # this fixed-SST harness, computed once.
    q_sfc = float(sam_ocean_surface_q(jnp.asarray(T_sfc), p_sfc))

    def physics_fn(state, grid_in, hc_in, tm_in):
        ny, nx, nlev = state.theta_prime.data.shape
        n_tracers = state.tracers.data.shape[-1]
        rho_0 = hc_in.rho_ref
        theta_0 = hc_in.theta_ref
        theta_total = theta_0 + state.theta_prime.data
        rho_total = rho_0 + state.rho_prime.data

        k_sfc = nlev - 1
        u_lo = state.u.data[..., k_sfc]
        v_lo = state.v.data[..., k_sfc]
        rho_lo = rho_total[..., k_sfc]
        theta_lo = theta_total[..., k_sfc]
        pi_sfc = hc_in.exner_ref[k_sfc]
        if n_tracers > 0:
            q_lo = state.tracers.data[..., k_sfc, 0]
        else:
            q_lo = jnp.zeros_like(theta_lo)
        # Lowest-level height above the surface (SAM zbot).
        z_bot = hc_in.z_full[k_sfc]

        tau_x, tau_y, shflx, lhflx, _ = compute_sam_oceflx_fluxes(
            u_atm=u_lo, v_atm=v_lo, theta_atm=theta_lo, q_atm=q_lo,
            T_sfc=jnp.full_like(theta_lo, T_sfc),
            q_sfc=jnp.full_like(theta_lo, q_sfc),
            rho=rho_lo, z_bot=z_bot, exner_sfc=pi_sfc, wd=wd,
        )

        dz_sfc = hc_in.dz[k_sfc]
        du_sfc = tau_x / (rho_lo * dz_sfc)
        dv_sfc = tau_y / (rho_lo * dz_sfc)
        du_dt_data = jnp.zeros_like(state.u.data).at[..., k_sfc].set(du_sfc)
        dv_dt_data = jnp.zeros_like(state.v.data).at[..., k_sfc].set(dv_sfc)

        dT_sfc = shflx / (rho_lo * constants.c_pd * dz_sfc)
        dtheta_sfc = dT_sfc / pi_sfc
        dtheta_p_data = jnp.zeros_like(
            state.theta_prime.data
        ).at[..., k_sfc].set(dtheta_sfc)

        dtracers_data = jnp.zeros_like(state.tracers.data)
        if n_tracers > 0:
            dq_sfc = lhflx / (rho_lo * constants.L_v * dz_sfc)
            dtracers_data = dtracers_data.at[..., k_sfc, 0].add(dq_sfc)

        zeros_w = jnp.zeros_like(state.w.data)
        zeros_rho = jnp.zeros_like(state.rho_prime.data)
        zeros_phis = jnp.zeros_like(state.phis.data)

        return PlaneNonHydrostaticTendencies(
            du_dt=state.u.replace(data=du_dt_data),
            dv_dt=state.v.replace(data=dv_dt_data),
            dw_dt=state.w.replace(data=zeros_w),
            dtheta_prime_dt=state.theta_prime.replace(data=dtheta_p_data),
            drho_prime_dt=state.rho_prime.replace(data=zeros_rho),
            dphis_dt=state.phis.replace(data=zeros_phis),
            dtracers_dt=state.tracers.replace(data=dtracers_data),
        )

    return physics_fn


# -------- Physics composer -------- #


def _plane_radiation_physics_with_sst(radiation_config, grid, T_sfc):
    """Plane radiation physics_fn with the surface radiative boundary pinned to
    the prescribed SST (RAD-7).

    SAM's radiation uses the (fixed) sea-surface temperature for the surface
    longwave emission ``σ·ε·T_sfc⁴``; legoESM's radiation otherwise defaults to
    the lowest-level AIR temperature ``T[..., -1]`` (``integration.py`` line
    722), which drifts away from the SST as the column evolves and makes the
    surface LW flux un-SAM-faithful. Setting the static ``set_T_sfc_override``
    to a full ``(ncol,)`` SST field pins the surface boundary to the SST (the
    same hook the SCM/fixed-anchor path uses). Build-time set ⇒ the value is
    captured at trace time (fixed-SST CRM ⇒ static, JIT-safe).
    """
    rad_fn = make_radiation_physics(radiation_config, model_type="plane")
    ncol = int(grid.ny * grid.nx)
    # float64 = the CRM model/radiation dtype (codex iter-48 G: keep the
    # override in the same precision as T[..., -1] it replaces).
    rad_fn.set_T_sfc_override(jnp.full((ncol,), float(T_sfc), dtype=jnp.float64))
    return rad_fn


def _radiation_surface_emissivity(radiation_config: RadiationConfig | None) -> float:
    if radiation_config is None:
        return 1.0
    if radiation_config.scheme == "rrtmgp":
        return float(radiation_config.rrtmgp.sfc_emissivity)
    return float(radiation_config.gray.sfc_emissivity)


def _with_land_radiation_surface(
    radiation_config: RadiationConfig | None,
    albedo: float,
) -> RadiationConfig | None:
    """Return a radiation config whose radiative surface is land-like."""
    if radiation_config is None:
        return None
    if radiation_config.scheme == "rrtmgp":
        return radiation_config._replace(
            rrtmgp=radiation_config.rrtmgp._replace(
                sfc_albedo=albedo,
                sfc_albedo_direct=albedo,
            )
        )
    return radiation_config._replace(
        gray=radiation_config.gray._replace(sfc_albedo=albedo)
    )


def _make_land_radiation_refresh_fn(
    radiation_config: RadiationConfig | None,
    sfc_albedo: float,
    sfc_emissivity: float,
):
    """Build a dynamic-T_s plane radiation refresh for land mode.

    The returned callable takes ``(state, grid, hc, tm, T_sfc)`` and returns
    ``(tendency, surface_flux_diagnostics)``.  It is intentionally separate
    from the fixed-SST helper so the ocean path keeps using the existing static
    radiation override.
    """
    if radiation_config is None:
        return None

    from legoesm.atmosphere.physics.radiation.integration import (
        _call_radiation_backend,
        _compute_insolation,
        _get_grid_lat_lon,
        _validate_cloud_gate,
    )
    from legoesm.atmosphere.physics.thermodynamics import (
        pressure_from_eos,
        reconstruct_half_level_pressure_hydrostatic,
        sanitize_theta_rho,
    )

    _validate_cloud_gate(radiation_config)
    rrtmgp_solver = None
    ml_ozone_coefs = None
    if radiation_config.scheme == "rrtmgp":
        from legoesm.atmosphere.physics.radiation.rrtmgp.rrtmgp import RRTMGP

        RRTMGP.preload(radiation_config.rrtmgp)
        rrtmgp_solver = RRTMGP.from_legoesm_config(radiation_config.rrtmgp)
        if radiation_config.ozone.source == "ml":
            if not radiation_config.ozone.ml_weights_path:
                raise ValueError(
                    "OzoneProfileConfig.source='ml' requires ml_weights_path."
                )
            from legoesm.atmosphere.physics.radiation.ozone_ml import (
                load_ml_ozone_coefficients,
            )

            ml_ozone_coefs = load_ml_ozone_coefficients(
                radiation_config.ozone.ml_weights_path
            )

    def refresh_fn(state, grid, height_coord, terrain_metric, T_sfc):
        theta_p = state.theta_prime.data
        rho_p = state.rho_prime.data
        theta_0 = height_coord.theta_ref
        rho_0 = height_coord.rho_ref

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p, rho_0 + rho_p,
        )
        p = pressure_from_eos(rho_total, theta_total)
        exner = (p / constants.p_ref) ** constants.kappa
        T = theta_total * exner

        nlev = height_coord.n_levels
        shape_3d = theta_p.shape
        shape_w = state.w.data.shape
        shape_2d = state.phis.data.shape
        ny, nx = shape_2d
        ncol = ny * nx

        p_half = reconstruct_half_level_pressure_hydrostatic(
            p_full=p, rho_full=rho_total,
            z_half=terrain_metric.z_half_3d,
        )
        T_sfc = jnp.asarray(T_sfc, dtype=T.dtype)
        if T_sfc.shape != shape_2d:
            raise ValueError(
                f"land T_sfc shape {T_sfc.shape!r} must match {shape_2d!r}"
            )

        lat, lon = _get_grid_lat_lon(grid, shape_2d)
        insol, cos_sza, f_day, eccf = _compute_insolation(
            lat, radiation_config, lon=lon,
        )

        T_col = T.reshape(ncol, nlev)
        p_full_col = p.reshape(ncol, nlev)
        p_half_col = p_half.reshape(ncol, nlev + 1)
        T_sfc_col = T_sfc.reshape(ncol)
        lat_col = lat.reshape(ncol)
        lon_col = lon.reshape(ncol)
        insol_col = insol.reshape(ncol)
        cos_sza_col = (
            cos_sza.reshape(ncol) if cos_sza is not None else None
        )
        f_day_col = f_day.reshape(ncol) if f_day is not None else None

        n_tracers = state.tracers.data.shape[-1]
        if n_tracers > 0:
            q_v = jnp.clip(state.tracers.data[..., 0], 0.0, None)
        else:
            q_v = jnp.zeros_like(T)
        q_v_col = q_v.reshape(ncol, nlev)

        q_cloud_col = None
        q_ice_col = None
        if n_tracers > 1:
            q_cloud_col = jnp.clip(
                state.tracers.data[..., 1], 0.0, None,
            ).reshape(ncol, nlev)
        if n_tracers > 3:
            q_ice_col = jnp.clip(
                state.tracers.data[..., 3], 0.0, None,
            ).reshape(ncol, nlev)
        n_cloud_col = None
        n_ice_col = None
        if n_tracers > 8:
            n_cloud_col = jnp.clip(
                state.tracers.data[..., 6], 0.0, None,
            ).reshape(ncol, nlev)
            n_ice_col = jnp.clip(
                state.tracers.data[..., 8], 0.0, None,
            ).reshape(ncol, nlev)

        sfc_albedo_col = jnp.full(
            (ncol,), sfc_albedo, dtype=T_sfc_col.dtype,
        )
        sfc_emissivity_col = jnp.full(
            (ncol,), sfc_emissivity, dtype=T_sfc_col.dtype,
        )
        rad_out = _call_radiation_backend(
            radiation_config=radiation_config,
            eccf=eccf,
            T=T_col, p_full=p_full_col, p_half=p_half_col,
            sfc_temperature=T_sfc_col, lat=lat_col, q_v=q_v_col,
            insolation=insol_col, cos_sza=cos_sza_col,
            sfc_albedo_override=sfc_albedo_col,
            sfc_emissivity_override=sfc_emissivity_col,
            q_cloud=q_cloud_col, q_ice=q_ice_col,
            n_cloud=n_cloud_col, n_ice=n_ice_col, f_day=f_day_col,
            rrtmgp_solver=rrtmgp_solver, lon=lon_col,
            ml_ozone_coefs=ml_ozone_coefs,
        )

        dT_dt = rad_out.heating_rate.reshape(shape_3d)
        dtheta_prime_dt = dT_dt / jnp.clip(exner, 1e-6, None)
        dims_3d = ("y", "x", "z")
        dims_w = ("y", "x", "z_half")
        dims_2d = ("y", "x")
        dims_tr = ("y", "x", "z", "tracer")
        _sd = T.dtype
        _pd = state.phis.data.dtype

        tendencies = PlaneNonHydrostaticTendencies(
            du_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_sd), name="du_dt_rad_land",
                dims=dims_3d, units="m/s^2",
            ),
            dv_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_sd), name="dv_dt_rad_land",
                dims=dims_3d, units="m/s^2",
            ),
            dw_dt=Field(
                data=jnp.zeros(shape_w, dtype=_sd), name="dw_dt_rad_land",
                dims=dims_w, units="m/s^2",
            ),
            dtheta_prime_dt=Field(
                data=dtheta_prime_dt, name="dtheta_prime_dt_rad_land",
                dims=dims_3d, units="K/s",
            ),
            drho_prime_dt=Field(
                data=jnp.zeros(shape_3d, dtype=_sd),
                name="drho_prime_dt_rad_land",
                dims=dims_3d, units="kg/m^3/s",
            ),
            dphis_dt=Field(
                data=jnp.zeros(shape_2d, dtype=_pd), name="dphis_dt_rad_land",
                dims=dims_2d, units="m^2/s^3",
            ),
            dtracers_dt=Field(
                data=jnp.zeros_like(state.tracers.data),
                name="dtracers_dt_rad_land", dims=dims_tr, units="1/s",
            ),
        )
        surface_fluxes = {
            "sw_down": rad_out.sw_flux_down[:, -1].reshape(shape_2d),
            "lw_down": rad_out.lw_flux_down[:, -1].reshape(shape_2d),
            "sw_up": rad_out.sw_flux_up[:, -1].reshape(shape_2d),
            "lw_up": rad_out.lw_flux_up[:, -1].reshape(shape_2d),
        }
        return tendencies, surface_fluxes

    return refresh_fn


def make_rcemip_physics(
    grid, height_coord, terrain_metric,
    radiation_config: RadiationConfig | None,
    microphysics_config: MicrophysicsConfig | None,
    dt: float,
    surface_flux: bool = True,
    T_sfc: float = 300.0, p_sfc: float = 101480.0,
    ls_forcing_physics=None,
):
    """Compose RCEMIP physics_fn — surface + radiation + microphysics.

    Returns a single ``physics_fn(state, grid, hc, tm)`` that calls each
    branch and sums tendencies. Radiation is called every outer step
    here; for production with RRTMGP use
    :func:`make_rcemip_physics_gated_rad` which caches radiation across
    a configurable interval.

    ``ls_forcing_physics`` (optional): a plane physics_fn from
    :func:`legoesm.atmosphere.forcing.plane_large_scale_forcing
    .make_plane_ls_forcing_physics` adding SAM-style large-scale forcing
    (subsidence + advective tendencies + nudging). ``None`` (the RCE
    default) leaves the column free-running.
    """
    physics_fns = []
    if surface_flux:
        physics_fns.append(_make_surface_flux_physics(
            grid, height_coord, terrain_metric,
            T_sfc=T_sfc, p_sfc=p_sfc,
        ))
    if radiation_config is not None:
        physics_fns.append(_plane_radiation_physics_with_sst(
            radiation_config, grid, T_sfc,
        ))
    if microphysics_config is not None:
        physics_fns.append(make_microphysics_physics(
            microphysics_config, model_type="plane", dt=dt,
        ))
    if ls_forcing_physics is not None:
        physics_fns.append(ls_forcing_physics)

    def physics_fn(state, grid_in, hc_in, tm_in):
        tendencies = [
            fn(state, grid_in, hc_in, tm_in) for fn in physics_fns
        ]
        return sum_plane_tendencies(*tendencies)

    return physics_fn


def split_rad_from_other_physics(
    grid, height_coord, terrain_metric,
    radiation_config: RadiationConfig | None,
    microphysics_config: MicrophysicsConfig | None,
    dt: float,
    surface_flux: bool = True,
    T_sfc: float = 300.0, p_sfc: float = 101480.0,
    ls_forcing_physics=None,
):
    """Build TWO separate physics callables for the gated-radiation pattern.

    Returns ``(non_rad_physics_fn, rad_physics_fn_or_None)``:
    - non_rad_physics_fn(state) — runs every dycore outer step
      (surface fluxes + microphysics + optional large-scale forcing). Cheap.
    - rad_physics_fn(state) — runs only every ``radiation_interval``
      outer steps; returned tendency is cached + applied as forward
      Euler increments between refreshes. None if radiation_config is None.

    ``ls_forcing_physics`` (optional): SAM-style large-scale forcing
    physics_fn (subsidence + advective tendencies + nudging) appended to
    the cheap per-step branch; ``None`` (RCE default) leaves the column
    free-running. See :func:`make_rcemip_physics`.

    Both follow the standard ``physics_fn(state, grid, hc, tm) ->
    PlaneNonHydrostaticTendencies`` signature.
    """
    non_rad_fns = []
    if surface_flux:
        non_rad_fns.append(_make_surface_flux_physics(
            grid, height_coord, terrain_metric,
            T_sfc=T_sfc, p_sfc=p_sfc,
        ))
    if microphysics_config is not None:
        non_rad_fns.append(make_microphysics_physics(
            microphysics_config, model_type="plane", dt=dt,
        ))
    if ls_forcing_physics is not None:
        non_rad_fns.append(ls_forcing_physics)

    def non_rad_physics_fn(state, grid_in, hc_in, tm_in):
        if not non_rad_fns:
            return _zero_tendencies(state)
        tendencies = [
            fn(state, grid_in, hc_in, tm_in) for fn in non_rad_fns
        ]
        return sum_plane_tendencies(*tendencies)

    rad_physics_fn = None
    if radiation_config is not None:
        rad_physics_fn = _plane_radiation_physics_with_sst(
            radiation_config, grid, T_sfc,   # RAD-7: SST surface boundary
        )
    return non_rad_physics_fn, rad_physics_fn


def _zero_tendencies(state):
    """All-zero PlaneNonHydrostaticTendencies matching state's pytree shape."""
    return PlaneNonHydrostaticTendencies(
        du_dt=state.u.replace(data=jnp.zeros_like(state.u.data)),
        dv_dt=state.v.replace(data=jnp.zeros_like(state.v.data)),
        dw_dt=state.w.replace(data=jnp.zeros_like(state.w.data)),
        dtheta_prime_dt=state.theta_prime.replace(
            data=jnp.zeros_like(state.theta_prime.data)),
        drho_prime_dt=state.rho_prime.replace(
            data=jnp.zeros_like(state.rho_prime.data)),
        dphis_dt=state.phis.replace(data=jnp.zeros_like(state.phis.data)),
        dtracers_dt=state.tracers.replace(
            data=jnp.zeros_like(state.tracers.data)),
    )


def apply_radiation_forward_euler(state, rad_tend, dt):
    """Forward-Euler apply of cached radiation tendency over dt.

    Radiation tends to be slow (~K/day in tropos, ~10K/day at strato).
    Over a 5-min radiation interval, forward Euler error << RK3 dycore
    error on the same fields. Standard treatment in operational CRMs
    (SAM, WRF, CM1 all forward-Euler their radiation increment).

    Applies ONLY ``dtheta_prime_dt``: RRTMGP and gray radiation produce a
    pure heating rate (no direct q_v / condensate tendency), so the θ'
    increment IS the complete radiative effect. A future radiation package
    that also emitted a moisture/condensate tendency would need this
    extended (it would otherwise be silently dropped here).
    """
    new_theta_p = state.theta_prime.data + dt * rad_tend.dtheta_prime_dt.data
    return state._replace(
        theta_prime=state.theta_prime.replace(data=new_theta_p),
    )


# -------- IC + main -------- #


def emit_crm_profiles(state, height_coord, out_dir, label="run",
                      npz_name=None, verbose=True):
    """Report CRM convective magnitudes + vertical profiles from a final state
    via :func:`crm_comparison_profiles_plane` (w'^2(z), T/q_v ± std, cloud
    fraction, condensate, CWV, max|w|, precip proxy) and save an npz.  Shared
    by the GATE / LBA / RCE plane drivers for the docs' SAM comparison protocol
    (by MAGNITUDE + profile SHAPE, NOT snapshots)."""
    import numpy as _np
    from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
        crm_comparison_profiles_plane,
    )
    prof = crm_comparison_profiles_plane(state, height_coord)
    zf = _np.asarray(prof.z_full)
    zh = _np.asarray(prof.z_half)
    wvar = _np.asarray(prof.w_var_half)
    T = _np.asarray(prof.T)
    qv = _np.asarray(prof.q_v)
    cf = _np.asarray(prof.cloud_fraction)
    qc = _np.asarray(prof.q_cloud)
    qp = _np.asarray(prof.q_precip)
    if verbose:
        kpk = int(_np.argmax(wvar))
        print(f"\n=== {label} convective magnitudes + profiles "
              f"(vs SAM-{label} / obs) ===")
        print(f"  max|w| = {float(prof.max_w):6.2f} m/s   "
              f"peak w'^2 = {wvar[kpk]:.3f} m2/s2 @ {zh[kpk]/1e3:.1f} km")
        print(f"  CWV = {float(prof.cwv_mean):5.1f} mm   "
              f"precip(proxy) = {float(prof.precip_mean):.3e}   "
              f"peak cloud_frac = {_np.max(cf):.3f} @ "
              f"{zf[int(_np.argmax(cf))]/1e3:.1f} km")
        print("    z[km]   w'^2     T[K]   qv[g/kg]  cldfrac  qc[g/kg]  qp[g/kg]")
        for ztarg in (0.5, 1.0, 3.0, 6.0, 9.0, 12.0, 15.0):
            k = int(_np.argmin(_np.abs(zf - ztarg * 1e3)))
            kh = int(_np.argmin(_np.abs(zh - ztarg * 1e3)))
            print(f"  {zf[k]/1e3:6.1f}  {wvar[kh]:7.3f}  {T[k]:6.1f}  "
                  f"{qv[k]*1e3:7.2f}  {cf[k]:6.3f}  {qc[k]*1e3:7.3f}  "
                  f"{qp[k]*1e3:7.3f}")
    out_dir.mkdir(parents=True, exist_ok=True)
    _np.savez(
        out_dir / (npz_name or f"{label.lower()}_profiles.npz"),
        z_full=zf, z_half=zh, w_var_half=wvar, T=T, q_v=qv,
        cloud_fraction=cf, q_cloud=qc, q_precip=qp,
        max_w=float(prof.max_w), precip_mean=float(prof.precip_mean),
        cwv_mean=float(prof.cwv_mean),
    )
    return prof


from legoesm.atmosphere.dynamics.crm.sam_case_setup import (  # noqa: E402
    band_limited_seed_pattern as _band_limited_seed_pattern,
)


# Minimum vertical levels for the --sounding path. The gSAM tropopause cold
# point is a KINK, and linear reconstruction error at a kink is FIRST order in
# dz: 1.08 K at nlev=48 over H=20 km (job 9331622), acceptable at nlev=96. The
# driver default (--nlev 30) would ship a tropopause warm by kelvin, so the
# sounding path refuses it rather than quietly producing a bad column.
_SOUNDING_MIN_NLEV = 64


def build_sounding_height_coord(args, p_sfc_rcemip, dtype=jnp.float64):
    """Read a TABULATED SAM sounding and build the RCEMIP1 vertical column.

    gSAM's own RCEMIP1 deck ships one sounding per SST
    (``CASES/RCEMIP1/snd_rcemip_{295,300,305}s6.11.2``; ``snd`` is the 300 K
    one), and that file is NOT the Wing analytic profile — its tropospheric
    lapse is steeper and its stratosphere WARMS with height where the analytic
    form caps isothermally.  No choice of analytic constants can represent the
    second difference, so the faithful RCEMIP1 IC has to read the table, as
    BOMEX/RICO/DYCOMS/GATE already do.

    Everything here routes through the SHARED reader + setup — no second
    parser, no second IC builder — so the theta (potential temperature) and
    q [g/kg -> kg/kg] conventions are handled in exactly one place for every
    SAM-deck case::

        read_sam_snd -> extend_sounding_to_top -> build_sam_case_height_coord

    Module-level (rather than inline in ``main``) so the guards below are
    directly testable: a CLI round-trip only proves argparse stores a string.

    Returns ``(snd, height_coord)``; the sounding is returned already extended
    to cover the model top so the caller can build the IC from the SAME object.
    """
    from legoesm.atmosphere.dynamics.crm.sam_case_setup import (
        build_sam_case_height_coord,
    )
    from legoesm.atmosphere.forcing.sam_case_forcing import (
        extend_sounding_to_top,
        read_sam_snd,
    )
    snd_path = Path(args.sounding)
    if not snd_path.is_file():
        raise SystemExit(
            f"--sounding {snd_path}: file not found. gSAM's RCEMIP1 deck "
            "lives at $LEGOESM_GSAM_ROOT/CASES/RCEMIP1/ (e.g. "
            "snd_rcemip_300s6.11.2 for the RCE300 case).")
    # build_sam_case_height_coord only builds a STRETCHED column — the
    # SAM-faithful grid.  Refuse rather than silently ignore a requested
    # uniform grid, so the vertical grid a run used is never a surprise
    # (dispatch-hardening: no silent coercion).
    if not args.stretched_vertical:
        raise SystemExit(
            "--sounding requires --stretched-vertical: the tabulated-sounding "
            "path builds the SAM-faithful stretched column, and silently "
            "overriding a requested uniform grid would hide which grid a run "
            "used.")
    # The gSAM cold point is a sharp V at ~14.5 km.  Reconstruction error at a
    # kink is FIRST order in dz, so a coarse column moves the tropopause
    # temperature by whole kelvin — measured 1.08 K at nlev=48 over H=20 km.
    # The driver default is 30, which would silently ship a warm tropopause.
    if args.nlev < _SOUNDING_MIN_NLEV:
        raise SystemExit(
            f"--sounding with --nlev {args.nlev}: too coarse. The gSAM "
            f"tropopause cold point is a kink whose interpolation error is "
            f"first order in dz (measured 1.08 K at nlev=48, H=20 km), so a "
            f"column below {_SOUNDING_MIN_NLEV} levels misplaces it by "
            f"kelvin. Use --nlev {_SOUNDING_MIN_NLEV} or more.")
    snd = read_sam_snd(snd_path)
    # Cover the model top so theta_ref does not clamp constant (dry-neutral)
    # aloft; SAM extrapolates on the US-standard-atmosphere T ratio and
    # extend_sounding_to_top replicates that.
    snd = extend_sounding_to_top(snd, float(args.H))
    # The sounding carries its OWN surface pressure (RCEMIP1: 1014.8 mb,
    # identical to WING_P_SFC).  Assert rather than assume, so a deck with a
    # different p_sfc cannot silently disagree with the flux/radiation code
    # that still uses p_sfc_rcemip.
    p_sfc_snd = float(snd.pres0) * 100.0
    if abs(p_sfc_snd - p_sfc_rcemip) > 1.0:
        raise SystemExit(
            f"--sounding {snd_path}: surface pressure {p_sfc_snd:.1f} Pa "
            f"disagrees with the RCEMIP1 p_sfc {p_sfc_rcemip:.1f} Pa used "
            "by the surface-flux and radiation paths.")
    hc = build_sam_case_height_coord(
        snd, nlev=args.nlev, H=args.H, p_sfc_pa=p_sfc_snd,
        dz_sfc=args.dz_sfc, dtype=dtype,
    )
    print(f"  SOUNDING IC: {snd_path} "
          f"({np.asarray(snd.z).size} levels after top-extension, "
          f"p_sfc={p_sfc_snd/100.0:.1f} mb)")
    return snd, hc


def build_sounding_initial_state(snd, grid, hc, args, n_tracers,
                                 dtype=jnp.float64):
    """Plane IC from a tabulated sounding via the shared SAM-deck builder.

    ``band_noise`` is this driver's name for the builder's ``band_random``
    seed — the SAME band-limited generator (``run_rcemip_plane`` imports
    ``band_limited_seed_pattern`` from ``sam_case_setup``), just a different
    label on the CLI.
    """
    from legoesm.atmosphere.dynamics.crm.sam_case_setup import (
        build_sam_case_initial_state,
    )
    return build_sam_case_initial_state(
        snd, grid, hc, n_tracers=n_tracers,
        seed_amp=args.theta_noise_amp,
        seed_kind=("band_random" if args.seed_kind == "band_noise"
                   else args.seed_kind),
        seed_kmax=args.seed_kmax, dtype=dtype,
    )


def _build_rcemip_initial_state(grid, height_coord, dtype=jnp.float64,
                                theta_noise_amp=0.1, n_seed_lev=4,
                                n_tracers=3, seed_kind="smooth_k1",
                                seed_kmax=6, rng_seed=0,
                                q_sfc=WING_Q_SFC_DEFAULT):
    """RCEMIP1 IC: rest state + q_v profile + small theta noise.

    theta noise restricted to the bottom ``n_seed_lev`` levels (Wing
    2018 symmetry breaker) — applying noise everywhere causes
    spurious upper-tropospheric buoyancy gradients that NaN within
    ~10 steps at dx=2 km regardless of dt. Zero-mean horizontal so
    total energy is conserved at IC.

    ``seed_kind``: ``"smooth_k1"`` (default, legacy — a single k=1 cosine, the
    safe-but-laminar symmetry breaker) or ``"band_noise"`` (CONV-TRIGGER #83 — a
    band-limited random field k=1..``seed_kmax`` that seeds a CELL POPULATION so
    convection can actually organise; needs the vertical-w filter on, see #82).

    ``n_tracers``: 3 = q_v, q_c, q_r (Kessler/Sundqvist/Thompson layout
    head); 9 = full Morrison/Seifert-Beheng with N_c, N_r, N_i + ice
    classes. The microphysics integration validates the slot count.
    """
    rest = make_rest_state(grid, height_coord, dtype=dtype)
    ny, nx, nlev = rest.theta_prime.data.shape
    tracers = jnp.zeros((ny, nx, nlev, n_tracers), dtype=dtype)
    q_v = _rcemip_qv_profile(height_coord.z_full, q_sfc=q_sfc).astype(dtype)
    tracers = tracers.at[..., 0].set(
        jnp.broadcast_to(q_v, (ny, nx, nlev)),
    )
    n_seed_lev = min(n_seed_lev, nlev)
    if seed_kind == "band_noise":
        pattern_2d = jnp.asarray(
            _band_limited_seed_pattern(ny, nx, k_max=seed_kmax,
                                       rng_seed=rng_seed), dtype=dtype)
    elif seed_kind == "smooth_k1":
        # SMOOTH k=1 cosine (kx=ky=1): all energy at the lowest non-trivial
        # wavenumber so the seed is RESOLVED, not grid-scale (white noise NaNs
        # the rest-state IC where Smag K=0 at t=0). Safe but seeds only ONE
        # large circulation (laminar — see CONV-TRIGGER #83).
        ix = jnp.arange(nx, dtype=dtype)
        iy = jnp.arange(ny, dtype=dtype)
        pattern_2d = (jnp.cos(2 * jnp.pi * iy / ny)[:, None]
                      * jnp.cos(2 * jnp.pi * ix / nx)[None, :])
    else:
        raise ValueError(
            f"seed_kind={seed_kind!r} invalid; use 'smooth_k1' or 'band_noise'.")
    theta_noise = (theta_noise_amp * pattern_2d[:, :, None]
                    * jnp.ones((1, 1, n_seed_lev), dtype=dtype))
    # Subtract horizontal mean (already ~zero-mean; keep for degenerate grids).
    theta_noise = theta_noise - jnp.mean(theta_noise, axis=(0, 1),
                                         keepdims=True)
    theta_p = jnp.zeros_like(rest.theta_prime.data)
    # Bottom 4 levels in top-down indexing = LAST 4 array entries.
    theta_p = theta_p.at[..., -n_seed_lev:].set(theta_noise)
    # Hydrostatic-balance IC: set rho' = -rho_0 * theta'/theta_0 so the
    # initial pressure perturbation is zero (matches the warm-bubble
    # convention used by tests/validation/test_plane_nh_rising_thermal.py).
    # Without this, theta' alone breaks hydrostatic balance — pressure
    # imbalance triggers an acoustic shock at step 1 that compounds with
    # Smagorinsky + hyperdiff and NaN's within ~2 sim hours at dt=2s.
    rho_0 = jnp.asarray(height_coord.rho_ref, dtype=dtype)
    theta_0 = jnp.asarray(height_coord.theta_ref, dtype=dtype)
    rho_p = -rho_0 * theta_p / theta_0
    return rest._replace(
        theta_prime=rest.theta_prime.replace(data=theta_p),
        rho_prime=rest.rho_prime.replace(data=rho_p),
        tracers=rest.tracers.replace(data=tracers),
    )


def _add_vortex_seed(state, grid, height_coord, vmax, rmax, ztop, dtype=jnp.float64):
    """Superpose a weak CYCLONIC tangential-wind vortex at the domain centre — the
    standard rotating-RCE tropical-cyclone intensification seed (a weak initial
    vortex spins up via WISHE under an f-plane + warm SST).

    Tangential-wind profile is a smooth modified-Rankine
    ``V(r) = vmax · 2 r·rmax / (r² + rmax²)`` (0 at the centre, peak ``vmax`` at
    ``r=rmax``, ~1/r decay outside), tapered ``cos²`` in height from full strength
    at the surface to zero at ``ztop``. Cyclonic for f>0 (counter-clockwise):
    ``u' = −V·Δy/r``, ``v' = +V·Δx/r``. WIND-ONLY seed (no warm core / pressure
    perturbation imposed): a weak vortex is near-balanced and the dycore radiates
    the small imbalance as inertia-gravity waves within ~1 h; the warm core then
    develops self-consistently. θ', ρ', tracers untouched."""
    u = np.asarray(state.u.data)
    v = np.asarray(state.v.data)
    ny, nx, nlev = u.shape
    xc = np.asarray(grid.xc); yc = np.asarray(grid.yc)
    x0 = 0.5 * (xc[0] + xc[-1]); y0 = 0.5 * (yc[0] + yc[-1])
    dxr = (xc[None, :] - x0); dyr = (yc[:, None] - y0)          # (ny,nx)
    r = np.sqrt(dxr ** 2 + dyr ** 2)
    r_safe = np.where(r > 1.0, r, 1.0)
    Vt = vmax * 2.0 * r * rmax / (r ** 2 + rmax ** 2)          # modified-Rankine
    # RADIAL WINDOW: drive Vt→0 before the half-domain so the ~1/r tail does NOT
    # reach the periodic boundary and self-interact with its image vortex (codex
    # 2026-06-09). Flat to r0, cos² roll-off r0→r_cut, zero beyond. r_cut at 0.42·L
    # (L=domain width) keeps a buffer to the L/2 seam.
    L = float(xc[-1] - xc[0])
    r_cut = 0.42 * L; r0 = 0.55 * r_cut
    win = np.where(r <= r0, 1.0,
                   np.where(r >= r_cut, 0.0,
                            np.cos(0.5 * np.pi * (r - r0) / (r_cut - r0)) ** 2))
    Vt = Vt * win
    up = -Vt * dyr / r_safe                                     # cyclonic (NH f>0)
    vp = +Vt * dxr / r_safe
    z = np.asarray(height_coord.z_full)                        # (nlev,)
    taper = np.where(z < ztop, np.cos(0.5 * np.pi * z / ztop) ** 2, 0.0)
    u = u + (up[:, :, None] * taper[None, None, :]).astype(u.dtype)
    v = v + (vp[:, :, None] * taper[None, None, :]).astype(v.dtype)
    return state._replace(
        u=state.u.replace(data=jnp.asarray(u, dtype=dtype)),
        v=state.v.replace(data=jnp.asarray(v, dtype=dtype)),
    )


# RAD-2: perpetual fixed-zenith RCE insolation presets (S_0 [W/m²], cosθ).
# Verified against the gSAM CASES namelists:
#   "rcemip" — RCEMIP1/prm: doperpetual+dosolarconstant, solar_constant=551.58,
#              zenith_angle=42.05° (cos=0.7425) ⇒ ≈409.6 W/m². The faithful
#              DEFAULT for this RCEMIP1 plane harness.
#   "sam"    — generic SAM doperpetual RCE: zenith 51.7°, S_0=685 ⇒ ≈424.7 W/m²
#              (an older/other SAM RCE setup; NOT what RCEMIP1/prm uses).
#   "off"    — latitude daily-mean / perpetual-equinox + full S_0 (legacy).
# RRTMGP applies S_0·mu0 internally (cos_sza passed as mu0); verified TOA SW
# down = S_0·cosθ (no double-counting of the projection).
_RCE_INSOLATION = {
    "rcemip": (551.58, 0.7425),
    "sam": (685.0, 0.620),
}


def _build_radiation_config(
    scheme: str,
    update_interval_steps: int = 1,
    clouds: bool = True,
    insolation: str = "rcemip",
    t_sfc: float = 300.0,
) -> RadiationConfig | None:
    if scheme == "none":
        return None
    if scheme not in ("gray", "rrtmgp"):
        raise ValueError(
            f"Unknown --radiation: {scheme!r}; "
            f"choose from 'gray', 'rrtmgp', 'none'."
        )
    if insolation not in (*_RCE_INSOLATION, "off"):
        raise ValueError(
            f"Unknown --insolation: {insolation!r}; "
            f"choose from {(*_RCE_INSOLATION, 'off')}."
        )
    from legoesm.atmosphere.physics.radiation.config import RRTMGPConfig
    # RAD-2: fixed-zenith perpetual RCE insolation (SAM doperpetual) — a
    # uniform S_0·cosθ TOA flux with cosθ as the SW optical path, instead of
    # the latitude daily-mean cos(SZA). Reduced S_0 paired with the zenith.
    rce_cos = None
    s0 = None
    if insolation != "off":
        s0, rce_cos = _RCE_INSOLATION[insolation]
    # RAD-5/RAD-1: enable cloud-radiative coupling for the RRTMGP path so
    # resolved condensate (q_c slot 1, q_i slot 3) feeds LW/SW cloud optics.
    # Uses the CRM "resolved" cloud fraction (cf≈1 where condensate) + SAM's
    # ocean liquid r_eff = 14 µm. Gray has no cloud optics. NOTE (deferred
    # RAD-1-ice): ice r_eff stays 30 µm; SAM uses T-dependent 6–250 µm.
    if scheme == "rrtmgp":
        from legoesm.atmosphere.physics.clouds.config import CloudConfig
        rrtmgp_kwargs = {"include_clouds": clouds}
        if s0 is not None:
            rrtmgp_kwargs["S_0"] = s0
        # RAD-4: SAM RCEMIP trace-gas concentrations. gSAM (RCEMIP1/prm,
        # nxco2=1) uses the MLS-standard VMRs in RUNDATA/rrtmg_lw.nc
        # (AbsorberAmountMLS), verified by reading the file: CO2≈3.549e-4
        # (355 ppm, well-mixed), CH4 surface 1.7e-6 (1700 ppb), N2O surface
        # 3.2e-7 (320 ppb). We match what gSAM ACTUALLY runs (355), NOT the
        # Wing-2018 RCEMIP protocol value (348 ppm) — the oracle is gSAM.
        # legoESM's RRTMGP applies these well-mixed; SAM's CH4/N2O actually
        # DECREASE aloft (MLS profile), so the tropospheric value is a
        # well-mixed APPROXIMATION (radiatively-dominant layer; slightly
        # over-counts the stratosphere). Replaces legoESM's MODERN defaults
        # (415/1900/332); CO2 415→355 is ≈0.8 W/m² less LW forcing.
        if rce_cos is not None:
            rrtmgp_kwargs["co2_ppmv"] = 355.0
            rrtmgp_kwargs["ch4_ppbv"] = 1700.0
            rrtmgp_kwargs["n2o_ppbv"] = 320.0
        # RAD-3: SAM ocean surface albedo for the perpetual RCE, replacing the
        # flat 0.06 RRTMGP default. SAM splits the DIRECT beam (Briegleb
        # zenith-dependent ocean albedo, cam_rad_parameterizations.f90:albedo —
        # ≈0.033 at μ=0.7425) from the DIFFUSE field (adif=0.07, the RCEMIP
        # value, AAW 2017). The two-stream solver reflects each beam faithfully.
        if rce_cos is not None:
            from legoesm.atmosphere.physics.radiation.integration import (
                sam_ocean_albedo,
            )
            rrtmgp_kwargs["sfc_albedo_direct"] = float(
                sam_ocean_albedo(rce_cos, t_sfc)
            )
            rrtmgp_kwargs["sfc_albedo"] = 0.07     # SAM adif (RCEMIP diffuse)
        # RAD-4 (O3): use SAM's MLS ozone profile (bundled from rrtmg_lw.nc)
        # for the faithful RCE, instead of legoESM's built-in skewed-Gaussian.
        rad_kwargs = {}
        if rce_cos is not None:
            from legoesm.atmosphere.physics.radiation.config import (
                OzoneProfileConfig,
            )
            rad_kwargs["ozone"] = OzoneProfileConfig(source="mls")
        return RadiationConfig(
            scheme="rrtmgp",
            update_interval_steps=update_interval_steps,
            rrtmgp=RRTMGPConfig(**rrtmgp_kwargs),
            cloud_scheme="resolved" if clouds else "none",
            cloud_config=(
                CloudConfig(scheme="resolved", r_eff_liq=14.0e-6)
                if clouds else None
            ),
            rce_fixed_cos_zenith=rce_cos,
            **rad_kwargs,
        )
    # Gray radiation: fixed zenith via the same field; S_0 lives in gray.
    gray_kwargs = {}
    if s0 is not None:
        from legoesm.atmosphere.physics.radiation.config import (
            GrayRadiationConfig,
        )
        gray_kwargs["gray"] = GrayRadiationConfig(S_0=s0)
    return RadiationConfig(
        scheme=scheme, update_interval_steps=update_interval_steps,
        rce_fixed_cos_zenith=rce_cos, **gray_kwargs,
    )


def _build_microphysics_config(
    scheme: str, homogeneous_ice_nucleation: bool = False,
    hard_saturation_adjustment: bool = False,
    hard_sat_adjust_threshold: float | None = None,
    hard_sat_max_heating_K: float | None = None,
) -> MicrophysicsConfig | None:
    if scheme == "none":
        return None
    valid = ("kessler", "morrison", "sundqvist",
             "seifert_beheng", "thompson", "p3", "sdm", "fast_sbm",
             "ml_emulator")
    if scheme not in valid:
        raise ValueError(
            f"Unknown --microphysics: {scheme!r}; "
            f"choose from {valid + ('none',)}."
        )
    if scheme == "morrison":
        # The plane CRM validates against the gSAM oracle, so use the SAM
        # M2005 flavor here (the global/GCM default is "mg" — E3SM
        # Morrison-Gettelman). See ``resolve_morrison_flavor``.
        from legoesm.atmosphere.physics.microphysics.config import (
            MorrisonConfig,
        )
        cfg = MicrophysicsConfig(
            scheme=scheme,
            morrison=MorrisonConfig(
                morrison_flavor="sam",
                homogeneous_ice_nucleation=homogeneous_ice_nucleation,
            ),
        )
    else:
        cfg = MicrophysicsConfig(scheme=scheme)

    # Hard (iterated) saturation-adjustment guard.  This driver is the RCE/CRM
    # lane, i.e. exactly where a super-saturated initial sounding can occur, so
    # the guard needs a CLI route here -- it previously had none for ANY
    # scheme, which is why a ~140 %-RH RCEMIP sounding could not be run with
    # the guard on without editing code.
    #
    # Threading goes through the SHARED
    # ``apply_microphysics_experiment_flags`` (never a private copy of the
    # dispatch): it applies the flags to the ACTIVE sub-config only and raises
    # LOUDLY for a scheme that does not carry the guard (sdm / fast_sbm), so a
    # silently-inert flag is impossible.  Wrapped in a Python ``if`` so the
    # all-defaults path returns the exact object it always did.
    if (hard_saturation_adjustment
            or hard_sat_adjust_threshold is not None
            or hard_sat_max_heating_K is not None):
        # A float override without the boolean gate would be SILENTLY INERT
        # (the schemes branch on a static ``if config.hard_saturation_
        # adjustment``), so refuse it rather than let a user believe they
        # tuned something. Mirrors ExperimentConfig.validate_strict, which
        # this driver does not go through.
        if not hard_saturation_adjustment:
            _set = [n for n, v in
                    (("--hard-sat-adjust-threshold", hard_sat_adjust_threshold),
                     ("--hard-sat-max-heating-k", hard_sat_max_heating_K))
                    if v is not None]
            raise ValueError(
                f"{', '.join(_set)} requires --hard-saturation-adjustment "
                "(the override would be silently inert without it)."
            )
        from legoesm.atmosphere.physics.microphysics.config import (
            apply_microphysics_experiment_flags,
        )
        cfg = cfg._replace(**{scheme: apply_microphysics_experiment_flags(
            getattr(cfg, scheme), scheme,
            hard_saturation_adjustment=hard_saturation_adjustment,
            hard_sat_adjust_threshold=hard_sat_adjust_threshold,
            hard_sat_max_heating_K=hard_sat_max_heating_K,
        )})
    return cfg


def dx_aware_hyperdiff(dx, base=1.0e8, dx_ref=1000.0):
    """Biharmonic hyperdiff coefficient scaled ``K = base·(dx/dx_ref)⁴``.

    The biharmonic CFL (``K·dt/dx⁴``) AND the 2Δx damping rate (``K/dx⁴``) both go
    as ``K/dx⁴``, so a FIXED coefficient over-damps at finer dx (16× too strong
    at dx=500 m vs the dx=1000 m default). SAM uses NO explicit hyperdiffusion
    (monotone advection + SGS); legoESM's compressible dycore needs a weak,
    dx-appropriate 2Δx filter. Single source of truth for the SAM-case drivers
    (codex iter-54 F).
    """
    return base * (dx / dx_ref) ** 4


def cfl_guard(dt, dx, dz_min, n_acoustic_substeps=6, c_s=350.0, label="run"):
    """Warn (fail-fast) when a config likely violates an acoustic CFL — the
    HORIZONTAL substep limit (``c_s·dt/(n_sub·dx)``) or the fine vertical-grid
    limit (``dt/min_dz``). The fine-dx SAM-resolution config trips this (codex
    iter-54 D: the dx=100 m blow-up is the acoustic, not the hyperdiff)."""
    import warnings
    acoustic_cfl = c_s * dt / (n_acoustic_substeps * dx)
    if acoustic_cfl > 0.5:
        warnings.warn(
            f"{label}: horizontal acoustic CFL ≈{acoustic_cfl:.2f} "
            f"(c_s·dt/(n_sub·dx); dx={dx:.0f} m, dt={dt} s, n_sub="
            f"{n_acoustic_substeps}) > 0.5 — raise n_acoustic_substeps or reduce "
            f"dt; the run may go non-finite.", stacklevel=3)
    if dt > 0.041 * dz_min:
        warnings.warn(
            f"{label}: dt={dt} s vs min_dz={dz_min:.0f} m exceeds the vertical "
            f"acoustic CFL (dt≲{0.04 * dz_min:.1f} s) — reduce dt.", stacklevel=3)


def parse_args(argv=None):
    """Parse the driver CLI. ``argv=None`` reads ``sys.argv`` (production);
    passing a list is what the CLI round-trip tests use — same convention as
    ``run_omip.parse_args`` / ``run_amip.parse_args``."""
    p = argparse.ArgumentParser(description="RCEMIP1 plane NH harness.")
    p.add_argument("--nx", type=int, default=16)
    p.add_argument("--ny", type=int, default=16)
    p.add_argument("--nlev", type=int, default=30)
    p.add_argument("--dx", type=float, default=4_000.0,
                   help="Horizontal grid spacing [m]; RCEMIP1 full = 1 km.")
    p.add_argument("--H", type=float, default=33_000.0,
                   help="Model top height [m]; RCEMIP1 = 33 km.")
    p.add_argument("--dt", type=float, default=6.0)
    p.add_argument("--steps", type=int, default=50)
    p.add_argument("--T-sfc", type=float, default=300.0)
    p.add_argument("--land", action="store_true",
                   help="Run RCEMIP-LAND slab RCE: prognostic surface "
                        "temperature and beta-limited evaporation. Default is "
                        "the fixed-SST ocean RCE.")
    p.add_argument("--land-heat-capacity", type=float, default=5.0e5,
                   help="LAND slab heat capacity C_slab [J/m^2/K]. Small "
                        "values (2e5-1e6) give hour-to-day skin-temperature "
                        "adjustment.")
    p.add_argument("--land-albedo", type=float, default=0.20,
                   help="LAND surface albedo used by radiation and the slab "
                        "surface energy balance.")
    p.add_argument("--land-beta", type=float, default=0.70,
                   help="LAND moisture availability beta in [0,1], multiplying "
                        "the potential latent-heat flux.")
    p.add_argument("--land-T-init", type=float, default=None,
                   help="Initial LAND slab surface temperature [K]. Default: "
                        "use --T-sfc.")
    p.add_argument("--hyperdiff", type=float, default=1.0e6)
    p.add_argument("--smag-cs", type=float, default=0.19,
                   help="Smagorinsky Cs; SAM default 0.19 (dosmagor).")
    p.add_argument("--turbulence-closure",
                   choices=["smagorinsky", "molecular", "none"],
                   default="smagorinsky",
                   help="SGS closure MODE (CRM/LES/DNS share the same diffusion "
                        "operators; only K_m differs). 'smagorinsky' = eddy "
                        "viscosity — CRM (--smag-cs 0.19) AND LES (--smag-cs 0.15 "
                        "with fine dx + --sgs-vertical). 'molecular' = DNS, "
                        "constant molecular viscosity (--molecular-viscosity; needs "
                        "--sgs-vertical for the full 3-D nu*grad^2 and dx near the "
                        "Kolmogorov scale). 'none' = inviscid.")
    p.add_argument("--molecular-viscosity", type=float,
                   default=constants.nu_air,
                   help="Constant kinematic viscosity nu [m^2/s] for "
                        "--turbulence-closure molecular (DNS). Default = air "
                        f"({constants.nu_air:.2e}); scale up for a reduced-Reynolds "
                        "DNS at a tractable resolution.")
    p.add_argument("--smag-wall-damping", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="Cap the Smagorinsky mixing length at the von Karman wall "
                        "scaling l_m=min(Cs*Delta, kappa*z) (Mason 1989). DEFAULT "
                        "False = SAM-faithful CRM (dosmagor smix=grd, no cap). Pass "
                        "--smag-wall-damping for the LES near-surface preset.")
    p.add_argument("--smag-stability-length", action=argparse.BooleanOptionalAction,
                   default=False,
                   help="SAM dosmagor STABLE-layer Deardorff mixing-length limit "
                        "(tke_full.f90:285-298): in weakly-stable layers shrink "
                        "smix=min(grd, sqrt(0.76*tk/(Ck*sqrt(N2)))) + vary "
                        "Cee=Ce1+Ce2*smix/grd. DEFAULT False = the pure (Cs*D)^2*|S| "
                        "form (already SAM-faithful in unstable + strongly-stable "
                        "layers); pass --smag-stability-length for the FULL dosmagor "
                        "(adds the minor weakly-stable shrink; use with --no-smag-"
                        "wall-damping).")
    p.add_argument("--smag-delta-max", type=float, default=1000.0,
                   help="SAM-faithful cap [m] on the HORIZONTAL grid spacing in the "
                        "Smagorinsky mixing length Delta=(min(d,dx)*min(d,dy)*dz)^(1/3)"
                        " (SAM sgs.f90:90 delta_max=1000). Prevents SGS over-mixing on"
                        " coarse grids (dx>1 km, e.g. RCE dx=3-4 km). No effect for "
                        "dx,dy<=delta_max; pass a huge value to disable.")
    p.add_argument("--sponge-coeff", type=float, default=0.05)
    p.add_argument("--sponge-width", type=float, default=5_000.0)
    p.add_argument("--semi-implicit", action="store_true",
                   help="Use semi-implicit acoustic substepping (lifts "
                        "vertical CFL). Recommended for long runs at dx>=2km.")
    p.add_argument("--substep-horizontal-acoustic",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Move horizontal pressure gradient + mass continuity "
                        "into the acoustic substep loop (full Skamarock-Klemp "
                        "split). DEFAULT ON (iter-61, DYCORE-ROBUST #82 fix; GATE/"
                        "LBA already do this): REQUIRED for stable runs with a "
                        "perturbed theta' IC — otherwise the horizontal acoustic "
                        "mode is integrated at the outer dt and blows up to NaN "
                        "at finite amplitude. Use --no-substep-horizontal-acoustic "
                        "to opt out (unsafe). Plane SI only.")
    p.add_argument("--sgs-vertical",
                   action=argparse.BooleanOptionalAction, default=True,
                   help="Add the VERTICAL SGS flux ∂_z(K ∂_z φ) for "
                        "u/v/θ'/tracers/w so the Smagorinsky closure is fully 3D "
                        "like SAM (diffuse_scalar/diffuse_mom; SGS-VERT #81). "
                        "DEFAULT ON (faithful) — the plane CRM previously did the "
                        "HORIZONTAL leg only, under-mixing sub-grid vertical "
                        "transport. Small + stabilizing (diffusion number "
                        "K·dt/dz²≪1). Use --no-sgs-vertical for the horizontal-only "
                        "legacy. No-flux interior BC (surface fluxes separate).")
    p.add_argument("--n-acoustic-substeps", type=int, default=6,
                   help="Acoustic substeps per RK3 stage. Default 6 "
                        "matches CompressibleEulerConfig default.")
    p.add_argument("--off-centering", type=float, default=0.1,
                   help="Skamarock-Klemp 2008 off-centering beta for the "
                        "acoustic mode. 0.1 = recommended for moist-convective "
                        "stability; 0.0 = centred (less damping but can grow "
                        "acoustic noise on long runs).")
    p.add_argument("--implicit-buoyancy", action="store_true", default=True,
                   help="Klemp-Wilhelmson 1978 implicit buoyancy in the "
                        "semi-implicit acoustic substep. Closes the w<->theta "
                        "gravity-wave feedback at coarse vertical resolution; "
                        "default ON for RCE (was OFF in iter-78 bench config).")
    p.add_argument("--no-implicit-buoyancy", dest="implicit_buoyancy",
                   action="store_false")
    p.add_argument("--si-w-filter-nu", type=float, default=0.0,
                   help="Vertical Laplacian filter on w inside each SI "
                        "acoustic substep. 0.0 = off (default). 0.3-0.4 "
                        "fully damps the structural exponential mode that "
                        "the SI scheme exhibits with perturbed theta' IC. "
                        "0.5 = explicit-diffusion CFL bound — above NaNs. "
                        "Recommended for RCE runs with theta_noise_amp > 0; "
                        "leave off for clean Wing IC + active moist physics.")
    p.add_argument("--theta-noise-amp", type=float, default=0.0,
                   help="Initial theta' perturbation amplitude [K] at bottom 4 "
                        "levels. 0 = clean Wing IC (stable at dt up to 10 s "
                        "per iter-9/14); 0.1 = Wing 2018 standard symmetry "
                        "breaker (blows up at dx>=2 km without LES — iter-212 "
                        "in run_rce_mpi_long.py).")
    p.add_argument("--seed-kind", choices=["smooth_k1", "band_noise"],
                   default="smooth_k1",
                   help="IC theta' perturbation spectrum (CONV-TRIGGER #83): "
                        "'smooth_k1' (legacy single wave, stable but laminar) or "
                        "'band_noise' (band-limited k=1..seed-kmax, seeds a "
                        "convective-cell population; use with --si-w-filter-nu).")
    p.add_argument("--seed-kmax", type=int, default=6,
                   help="Max wavenumber for --seed-kind band_noise (≪ nx/2).")
    p.add_argument("--vertical-theta-diffusion", type=float, default=0.0,
                   help="Constant-ν vertical Laplacian diffusion of θ' [m²/s] "
                        "(DYCORE-ROBUST #82 probe: tests whether damping the θ' "
                        "vertical mode — not just w via --si-w-filter-nu — also "
                        "stabilises the finite-amplitude blow-up).")
    p.add_argument("--no-physics", action="store_true",
                   help="Skip the physics_fn entirely. Use for dry-dycore "
                        "stability probes.")
    p.add_argument("--no-surface-flux", action="store_true",
                   help="Disable surface bulk fluxes (Cd=Ch=0). For stability "
                        "diagnostics — without surface fluxes RCE cannot reach "
                        "physical equilibrium but the dycore alone can be "
                        "tested.")
    p.add_argument("--advection",
                   choices=["upwind1", "van_leer", "weno5"],
                   default="van_leer",
                   help="Horizontal advection scheme (SCALARS θ'/q_v/tracers; "
                        "also momentum unless --momentum-advection set). van_leer "
                        "(default) = 2nd-order TVD, monotone (≈MPDATA), stencil 4. "
                        "weno5 = 5th-order WENO-Z; upwind1 = 1st-order (smoke).")
    p.add_argument("--momentum-advection",
                   choices=["upwind1", "centered", "van_leer", "weno5"],
                   default="centered",
                   help="ADV-SPLIT #86: SEPARATE horizontal advection for the "
                        "MOMENTUM legs (u/v/w). 'centered' (DEFAULT iter-196, "
                        "validated stable) = SAM-faithful 2nd-order CENTRED "
                        "(= gSAM advect2_mom, NON-diffusive) — van_leer momentum "
                        "over-diffuses + SUPPRESSES updraft cores/w-tails (codex "
                        "iter-68: identical-IC max|w| caps ~2.6 vs ~15). Use "
                        "'centered' for SAM-faithful convective EXTREMES (dispersive "
                        "— relies on hyperdiff+SGS to control 2Δ noise).")
    p.add_argument("--vertical-tracer-advection",
                   choices=["centered", "van_leer"],
                   default="van_leer",
                   help="VERTICAL tracer advection (D5). van_leer (default) "
                        "= monotone TVD, positive-definite (SAM advects "
                        "scalars with a monotone scheme); centered = 2nd-order "
                        "centred (can overshoot into negative tracer at sharp "
                        "convective gradients).")
    p.add_argument("--acoustic-theta-advection",
                   choices=["centered", "van_leer"],
                   default="centered",
                   help="VERTICAL theta' advection INSIDE the acoustic substep "
                        "(iter-200). centered (default, current behaviour) = "
                        "2nd-order centred — overshoots at the sharp tropopause "
                        "theta-gradient ⇒ a dispersive COLD DRIFT of the cold-point "
                        "over long RCE runs. van_leer = monotone TVD (SAM-faithful; "
                        "SAM advects theta with a monotone scheme) — forbids new "
                        "extrema ⇒ no cold drift. Decoupled from the implicit "
                        "w-solve. Recommended van_leer for long RCE; validate then "
                        "promote to default.")
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64",
                   help="fp32 ~5-9x faster than fp64 on consumer GPU "
                        "(fp64 ALU 1:32 ratio); fp32 sufficient for RCE.")
    p.add_argument("--radiation-interval", type=int, default=150,
                   help="Radiation update interval in outer steps. RCEMIP / "
                        "CRM standard: refresh every 5 min sim time. At "
                        "dt=2s, interval=150 = 5 min refresh; interval=900 = "
                        "30 min (less aggressive). 1 = every step (heavy with "
                        "RRTMGP). Gated mode (interval>1) applies cached "
                        "radiation tendency as forward Euler increments "
                        "between recomputes — standard SAM/WRF/CM1 practice.")
    p.add_argument("--radiation", choices=["gray", "rrtmgp", "none"],
                   default="rrtmgp",
                   help="Radiation scheme. DEFAULT rrtmgp = RRTM (the faithful "
                        "RCEMIP/SAM scheme; matches the GATE driver, iter-56 #77). "
                        "CONV-INTENSITY #85 (iter-63): the gray config UNDER-DRIVES "
                        "this RCE relative to rrtmgp (mid-trop w_RMS ~0.07 m/s + "
                        "INVERTED w'² profile vs rrtmgp's ~0.5 m/s mid-trop-peaked "
                        "burst — the RCEMIP-SAM range) — consistent with gray's lack "
                        "of realistic vertical/spatial radiative-heating structure + "
                        "cloud/water-vapor radiative feedbacks, though it could also "
                        "partly reflect gray cooling-rate/profile tuning or spin-up "
                        "(mechanism not yet isolated — needs Qrad(z,t) + equilibrated "
                        "means). Use 'gray' only for fast/low-memory smoke (rrtmgp "
                        "gas-optics OOM at 64²×float64 on 24 GB; ok ≤48²). 'none' "
                        "skips radiation.")
    p.add_argument("--clouds", action="store_true", default=True,
                   help="Enable cloud-radiative coupling for RRTMGP "
                        "(RAD-5: resolved condensate -> LW/SW cloud optics, "
                        "cf~1 in-cloud, r_eff_liq=14um). Default ON; "
                        "gray radiation ignores it (no cloud optics).")
    p.add_argument("--no-clouds", dest="clouds", action="store_false",
                   help="Clear-sky radiation (disable cloud optics).")
    p.add_argument("--insolation", choices=["rcemip", "sam", "off"],
                   default="rcemip",
                   help="Perpetual fixed-zenith RCE insolation (RAD-2). "
                        "'rcemip' (default) = RCEMIP1/prm zenith 42.05deg, "
                        "S_0=551.58 (~410 W/m2); 'sam' = generic doperpetual "
                        "51.7deg, S_0=685 (~425); 'off' = latitude daily-mean "
                        "(legacy, full S_0).")
    p.add_argument("--microphysics",
                   choices=["kessler", "morrison", "sundqvist",
                            "seifert_beheng", "thompson", "p3", "sdm",
                            "fast_sbm", "ml_emulator", "none"],
                   default="kessler",
                   help="Microphysics scheme. 'none' skips the branch.")
    # --- Hard (iterated) saturation-adjustment guard -----------------------
    # This is the RCE/CRM lane, where a super-saturated initial sounding is a
    # real configuration (a ~140 %-RH RCEMIP sounding produced a persistent
    # column-water drift). The guard had NO CLI route in this driver for any
    # scheme, so it could only be enabled by editing code.
    p.add_argument("--hard-saturation-adjustment",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Enable the iterated (bracketed-bisection) saturation "
                        "adjustment: wherever q_v exceeds "
                        "--hard-sat-adjust-threshold * q_sat, condensation is "
                        "blended toward the enthalpy-conserving on-curve "
                        "value instead of the slower smooth-sigmoid rate, "
                        "rate-limited by --hard-sat-max-heating-k. Default off "
                        "=> byte-identical to the historical smooth path. "
                        "Carried by kessler/sundqvist/seifert_beheng/morrison/"
                        "thompson/p3/ml_emulator; sdm and fast_sbm resolve "
                        "super-saturation explicitly and REJECT the flag.")
    p.add_argument("--hard-sat-adjust-threshold", type=float, default=None,
                   help="RH trigger for the hard saturation adjustment "
                        "(default: the scheme's 1.1). Requires "
                        "--hard-saturation-adjustment.")
    p.add_argument("--hard-sat-max-heating-k", type=float, default=None,
                   help="Per-step latent-heating cap [K] for the hard "
                        "saturation adjustment (default: the scheme's 5.0), so "
                        "a large super-saturation pool drains onto the curve "
                        "over MANY steps rather than releasing its full latent "
                        "heat at once. Requires --hard-saturation-adjustment.")
    p.add_argument("--homogeneous-ice-nucleation",
                   action=argparse.BooleanOptionalAction, default=False,
                   help="Morrison M2005 ONLY: enable Koop-2000 homogeneous ice "
                        "nucleation (cirrus). Bursts ice crystals once RH_ice "
                        "exceeds the homogeneous threshold S_hom(T)≈1.5-1.6, "
                        "which deposit the excess vapour and pin RH_ice near "
                        "S_hom — caps the unphysical >1000%% ice-supersaturation "
                        "the SAM-faithful Cooper-only path leaves in violent RCE "
                        "outflow. Default off = byte-identical Cooper-only.")
    p.add_argument("--print-every", type=int, default=10)
    p.add_argument("--snapshot-every", type=int, default=0,
                   help="Emit a surface-snapshot PNG every N steps "
                        "(0 = off). At dt=20s, 4320 steps = 1 sim day.")
    p.add_argument("--stretched-vertical", action="store_true",
                   help="Use create_stretched_height_coordinate (RCEMIP1: "
                        "nlev=74, geometric stretching from dz_sfc=50m near "
                        "surface to ~1500m at model top). Default uses the "
                        "uniform create_height_coordinate.")
    p.add_argument("--dz-sfc", type=float, default=50.0,
                   help="Surface-layer thickness [m] for stretched vertical "
                        "coordinate. RCEMIP1 standard = 50 m.")
    p.add_argument("--sounding", type=str, default=None,
                   help="Initialise from a TABULATED SAM-format sounding "
                        "(gSAM CASES/RCEMIP1/snd_rcemip_300s6.11.2 for RCE300) "
                        "instead of the analytic Wing 2018 profile. Read via "
                        "the shared read_sam_snd + sam_case_setup path used by "
                        "BOMEX/RICO/DYCOMS/GATE. gSAM's own sounding is NOT the "
                        "Wing analytic form (lapse 7.47 vs 6.70 K/km, and a "
                        "WARMING rather than isothermal stratosphere), so this "
                        "is the faithful RCEMIP1 IC. Requires "
                        "--stretched-vertical. Default (unset) = analytic.")
    # NO --sounding-grd. It was drafted to take gSAM's exact grd levels, but
    # CASES/RCEMIP1/grd is a 25-line ONE-column list topping out at 8500 m,
    # while read_sam_grd expects the 3-column "z idx spacing" form that
    # GATE_IDEAL/grd uses — so the flag would have raised ValueError on exactly
    # the file its own help text named, and the CLI round-trip test would not
    # have caught it (argparse strings round-trip fine). Shipping a flag that
    # cannot work is worse than not shipping it; supporting the 1-column form
    # belongs in the shared reader with its own test, not here.
    p.add_argument("--snapshot-days", type=float, default=0.0,
                   help="Save surface + 4-level-field npz every N SIM-DAYS "
                        "(precip, CWV, column-max w, condensate/w/qv/MSE at 4 "
                        "heights) via rce_snapshot. 0 = off.")
    p.add_argument("--snapshot3d-days", type=str, default="",
                   help="Comma-separated sim-days at which to dump the FULL 3D "
                        "condensate + MSE (+w,T) volume for 3D rendering, e.g. "
                        "'15,30,45,60'. Empty = off.")
    p.add_argument("--snapshot-heights", type=str, default="1000,5000,9000,12000",
                   help="Comma-separated heights [m] for the 4-level snapshots.")
    p.add_argument("--checkpoint-every", type=int, default=0,
                   help="Save a full-state checkpoint every N TIME STEPS (resumable). "
                        "0 = off.")
    p.add_argument("--restart", type=str, default="",
                   help="Resume from this checkpoint npz (rce_checkpoint). Empty=fresh "
                        "IC. Use 'latest' to auto-pick the newest in <output>/checkpoints.")
    p.add_argument("--restart-reset-condensate", action="store_true",
                   help="On restart, keep only q_v (slot 0) and zero all "
                        "hydrometeor mass/number. Use when switching FROM a "
                        "single-moment scheme (kessler) TO a double-moment one "
                        "(morrison/thompson/p3/seifert_beheng): copying bare "
                        "condensate mass with zero number is inconsistent and "
                        "blows up. The new scheme re-grows condensate in minutes.")
    p.add_argument("--restart-seed-numbers", action="store_true",
                   help="On restart into a double-moment scheme, KEEP the "
                        "single-moment condensate mass (q_c/q_r) so updrafts "
                        "stay loaded, and seed consistent number "
                        "concentrations (N_r=q_r/x_r). Alternative to "
                        "--restart-reset-condensate; avoids the unloaded-"
                        "updraft spin-up shock that runs Thompson away.")
    p.add_argument("--output", type=Path, default=Path("results/rcemip_plane"))
    # --- f-plane rotation + tropical-cyclone vortex seed (rotating RCE) -------
    p.add_argument("--coriolis-mode", choices=["none", "f_plane", "beta_plane"],
                   default="none",
                   help="Rotation: 'none' (RCEMIP default), 'f_plane' (constant f), "
                        "'beta_plane' (f0+beta*y). f_plane is the tropical-cyclone "
                        "setup (rotating RCE).")
    p.add_argument("--f0", type=float, default=0.0,
                   help="Coriolis parameter f [1/s]. Physical 20°N≈5e-5. For a "
                        "small-domain in-session TC use ENHANCED f≈5e-4 (~10× 20°N) "
                        "— shrinks the Rossby radius so a TC fits + spins up fast. "
                        "0 disables.")
    p.add_argument("--beta", type=float, default=0.0,
                   help="beta-plane df/dy [1/s/m]; only used with beta_plane.")
    p.add_argument("--vortex-seed-vmax", type=float, default=0.0,
                   help="Seed a weak balanced cyclonic vortex of peak tangential "
                        "wind VMAX [m/s] at domain centre (rotating-RCE TC "
                        "intensification). 0 disables. Typical 12.")
    p.add_argument("--vortex-seed-rmax", type=float, default=50.0e3,
                   help="Radius of max wind of the seed vortex [m] (default 50 km; "
                        "kept well inside the half-domain — the wind is radially "
                        "windowed to zero before the periodic seam).")
    p.add_argument("--vortex-seed-ztop", type=float, default=12.0e3,
                   help="Vertical extent of the seed vortex [m]; wind tapers "
                        "cos² from full at surface to 0 at ZTOP (default 12 km).")
    return p.parse_args(argv)


def main():
    args = parse_args()
    if args.land:
        if args.land_heat_capacity <= 0.0:
            raise SystemExit("--land-heat-capacity must be positive.")
        if not (0.0 <= args.land_albedo <= 1.0):
            raise SystemExit("--land-albedo must be in [0, 1].")
        if not (0.0 <= args.land_beta <= 1.0):
            raise SystemExit("--land-beta must be in [0, 1].")
        if args.radiation_interval < 1:
            raise SystemExit("--radiation-interval must be >= 1 for --land.")
    args.output.mkdir(parents=True, exist_ok=True)

    print(f"RCEMIP1 plane: nx={args.nx} ny={args.ny} nlev={args.nlev}")
    print(f"  dx={args.dx} m, Lz={args.H} m, dt={args.dt} s, "
          f"{args.steps} steps -> t_final={args.steps * args.dt:.1f} s")
    print(f"  T_sfc={args.T_sfc} K, hyperdiff={args.hyperdiff:.2e}, "
          f"smag_cs={args.smag_cs}, closure={args.turbulence_closure}"
          + (f" (DNS: nu={args.molecular_viscosity:.2e} m^2/s)"
             if args.turbulence_closure == "molecular" else ""))
    if args.land:
        _land_T0 = args.land_T_init if args.land_T_init is not None else args.T_sfc
        print(f"  LAND RCE: T_s_init={_land_T0} K, "
              f"C_slab={args.land_heat_capacity:.3e} J/m2/K, "
              f"albedo={args.land_albedo:.3f}, beta={args.land_beta:.3f}")
    # codex iter-61: log the stability-critical acoustic config so the iter-61
    # substep_horizontal_acoustic default flip (DYCORE-ROBUST #82) is visible in
    # every run's metadata (it silently changes finite-amplitude stability).
    print(f"  semi_implicit={args.semi_implicit}, substep_horizontal_acoustic="
          f"{args.substep_horizontal_acoustic}, si_w_filter_nu={args.si_w_filter_nu}, "
          f"seed_kind={args.seed_kind}, sgs_vertical={args.sgs_vertical}")
    # codex iter-63 [LOW]: record whether --radiation was DEFAULTED (vs
    # explicit) so logs are self-describing — the gray→rrtmgp default flip
    # (#85) silently changes the experiment when the flag is omitted.
    radiation_defaulted = not any(
        a == "--radiation" or a.startswith("--radiation=") for a in sys.argv[1:]
    )
    print(f"  radiation={args.radiation}"
          f"{' (DEFAULT)' if radiation_defaulted else ''}, "
          f"microphysics={args.microphysics}"
          f"{', homogeneous_ice_nucleation=ON' if (args.microphysics == 'morrison' and args.homogeneous_ice_nucleation) else ''}")
    # codex iter-63 [HIGH]: fail-LOUD preflight for the rrtmgp+float64+large-grid
    # OOM case (rrtmgp gas-optics tables exhaust 24 GB at 64²×float64; ok ≤48²).
    # Do NOT auto-fall-back to gray — a silent scheme switch is worse
    # scientifically than an explicit OOM. Warn (esp. if radiation was implicit
    # so the user did not knowingly pick the heavy scheme).
    if (args.radiation == "rrtmgp" and args.precision == "float64"
            and args.nx * args.ny >= 64 * 64):
        print(
            f"WARNING: rrtmgp + float64 + {args.nx}×{args.ny} grid"
            f"{' (radiation DEFAULTED to rrtmgp)' if radiation_defaulted else ''}"
            " is likely to OOM the RRTMGP gas-optics tables on a 24 GB GPU "
            "(empirically OK ≤48²×float64). If it OOMs: shrink the domain, use "
            "--precision float32 (with LEGOESM_RCEMIP_PLANE_FP32=1), or "
            "--radiation gray (fast/low-memory, but UNDER-drives convection — "
            "see #85). NOT auto-falling-back: a silent scheme switch would "
            "corrupt the experiment.", file=sys.stderr,
        )

    # Per codex iter-... HIGH#3: explicit contract check. fp32 must be
    # requested via the env var BEFORE Python import time; --precision
    # alone is insufficient because jax.config.update("jax_enable_x64",
    # True) runs at module load.
    if args.precision == "float32" and _os.environ.get(
        "LEGOESM_RCEMIP_PLANE_FP32"
    ) != "1":
        raise SystemExit(
            "--precision float32 requires LEGOESM_RCEMIP_PLANE_FP32=1 in "
            "the environment BEFORE python launch (the jax x64 toggle "
            "runs at module import time, before argparse). Example: "
            "LEGOESM_RCEMIP_PLANE_FP32=1 .venv/bin/python "
            "scripts/run_rcemip_plane.py --precision float32 ..."
        )
    dtype = jnp.float32 if args.precision == "float32" else jnp.float64
    grid = create_plane_grid(
        nx=args.nx, ny=args.ny, nlev=args.nlev,
        dx=args.dx, dy=args.dx, dtype=dtype,
        coriolis_mode=args.coriolis_mode, f0=args.f0, beta=args.beta,
    )
    # RCEMIP1 surface pressure (Wing 2018 = 1014.8 hPa). Passing p_sfc
    # switches compute_reference_state to the bottom-up hydrostatic BC
    # (iter-95). Without it, the legacy top-down BC over a 33 km column
    # gives exner_sfc=1.33 (p_sfc~2.7 atm) -> T_lowest=399 K -> surface
    # sensible-heat flux has the WRONG SIGN (cools the air toward a
    # ~236 K cold-biased equilibrium instead of warming toward SST).
    # With p_sfc set: exner_sfc=1.003, T_lowest=301 K, flux correct.
    p_sfc_rcemip = 101480.0
    # Reference θ(z) MUST track the RCEMIP sounding, NOT the constant-θ=300
    # default. A constant potential temperature is ISENTROPIC, and an
    # isentropic atmosphere reaches exner=0 (p=T=0) at z=c_p·θ/g ≈ 30.7 km
    # for θ=300 K — so over the 33 km RCEMIP column the top ~2 levels get
    # exner_ref<0 → rho_ref=NaN → the model NaNs at step 1 (independent of
    # physics/IC). (compute_reference_state now raises on exner≤0 so this
    # misconfig fails loudly at construction rather than at step 1.) The
    # Wing 2018 profile caps θ≈400 K above the 15 km tropopause, which
    # keeps exner_ref ≳0.16 and rho_ref finite over the full 33 km column
    # (the piecewise-profile exner-zero crossing sits ≈39 km, above the
    # top). Using it as the REFERENCE also starts the rest-state IC (θ'=0)
    # at the actual RCEMIP sounding instead of an isentropic 300 K column.
    # Single surface humidity drives BOTH the θ hydrostatic (virtual-T) base
    # and the q_v IC (codex iter-61 [MED] — keep them from desyncing).
    q_sfc_rce = WING_Q_SFC_DEFAULT
    # --sounding: initialise from a TABULATED SAM sounding instead of the
    # analytic Wing form.  gSAM's own RCEMIP1 deck ships one sounding per SST
    # (CASES/RCEMIP1/snd_rcemip_{295,300,305}s6.11.2, `snd` == the 300 K one),
    # and that file is NOT the Wing analytic profile: it has a ~7.47 K/km
    # tropospheric lapse and a stratosphere that WARMS with height, where the
    # analytic form uses 6.70 K/km and an isothermal cap.  No analytic-constant
    # choice can represent the second difference, so the faithful RCEMIP1 IC has
    # to read the table — exactly as BOMEX/RICO/DYCOMS/GATE already do.
    #
    # Reuses the SHARED reader + setup (no second parser, no second IC builder):
    #   read_sam_snd -> extend_sounding_to_top -> build_sam_case_height_coord
    #                                          -> build_sam_case_initial_state
    # so the θ (potential temperature) and q [g/kg -> kg/kg] unit conventions
    # are handled in exactly one place for every SAM-deck case.
    snd = None
    if args.sounding is not None:
        snd, hc = build_sounding_height_coord(args, p_sfc_rcemip, dtype)
    else:
        def theta_ref_fn(z):
            return _rcemip_theta_profile(z, T_sfc=args.T_sfc, q_sfc=q_sfc_rce)
        if args.stretched_vertical:
            hc = create_stretched_height_coordinate(
                args.nlev, H=args.H, dz_sfc=args.dz_sfc, p_sfc=p_sfc_rcemip,
                theta_ref_fn=theta_ref_fn,
            )
        else:
            hc = create_height_coordinate(
                args.nlev, H=args.H, p_sfc=p_sfc_rcemip,
                theta_ref_fn=theta_ref_fn,
            )
    tm = make_flat_plane_terrain_metric(grid, hc)
    cfg = CompressibleEulerConfig(
        sponge_coeff=args.sponge_coeff,
        sponge_width=args.sponge_width,
        sponge_w_only=True,               # SAM damping.f90: damp w only
        sponge_profile_shape="sam_rational",  # SAM zzz/(1+zzz) taper

        hyperdiff_coeff=args.hyperdiff,
        hyperdiff_rho_coeff=args.hyperdiff,
        hyperdiff_w_coeff=args.hyperdiff,
        semi_implicit_acoustic=args.semi_implicit,
        n_acoustic_substeps=args.n_acoustic_substeps,
        acoustic_off_centering=args.off_centering,
        implicit_buoyancy=args.implicit_buoyancy,
        si_w_vertical_filter_nu=args.si_w_filter_nu,
        substep_horizontal_acoustic=args.substep_horizontal_acoustic,
        horizontal_advection_scheme=args.advection,
        horizontal_momentum_advection_scheme=args.momentum_advection,
        vertical_tracer_advection=args.vertical_tracer_advection,
        acoustic_theta_advection=args.acoustic_theta_advection,
        use_coriolis=(args.coriolis_mode != "none"),
        fix_mass=True, anchor_mass_to_initial=True,
        smagorinsky_cs=args.smag_cs, smagorinsky_prandtl=1.0,
        smagorinsky_wall_damping=args.smag_wall_damping,  # False=SAM CRM; True=LES
        smagorinsky_delta_max=args.smag_delta_max,  # SAM 1000 m horiz length cap
        smagorinsky_stability_length=args.smag_stability_length,  # SAM dosmagor
        # weakly-stable Deardorff smix limit (default off = pure Cs form)
        vertical_theta_diffusion=args.vertical_theta_diffusion,
        sgs_vertical_diffusion=args.sgs_vertical,
        turbulence_closure=args.turbulence_closure,
        molecular_viscosity=args.molecular_viscosity,
    )
    # DYCORE-ROBUST (iter-61, ROOT CAUSE): with substep_horizontal_acoustic OFF the
    # HORIZONTAL acoustic mode is integrated at the OUTER dt and a perturbed θ' IC
    # blows up to NaN at finite amplitude. The full Skamarock-Klemp split
    # (--substep-horizontal-acoustic, now DEFAULT ON; GATE/LBA already use it) is
    # the real fix. Warn only if a user explicitly opts out under SI.
    if args.semi_implicit and not args.substep_horizontal_acoustic:
        import warnings
        warnings.warn(
            "RCE semi-implicit with --no-substep-horizontal-acoustic: the "
            "horizontal acoustic mode runs at the outer dt and BLOWS UP to NaN at "
            "finite amplitude (DYCORE-ROBUST #82). Drop the flag (it is ON by "
            "default) unless you know why you need it.", stacklevel=2)
    model = PlaneCompressibleEulerModel(grid, hc, tm, cfg)

    radiation_config = _build_radiation_config(
        args.radiation, update_interval_steps=args.radiation_interval,
        clouds=args.clouds, insolation=args.insolation, t_sfc=args.T_sfc,
    )
    if args.land:
        radiation_config = _with_land_radiation_surface(
            radiation_config, args.land_albedo,
        )
    microphysics_config = _build_microphysics_config(
        args.microphysics, args.homogeneous_ice_nucleation,
        hard_saturation_adjustment=args.hard_saturation_adjustment,
        hard_sat_adjust_threshold=args.hard_sat_adjust_threshold,
        hard_sat_max_heating_K=args.hard_sat_max_heating_k)
    if args.no_physics:
        physics_fn = None
        rad_physics_fn = None
    elif args.land:
        # LAND mode carries radiation and surface fluxes as host-side
        # forward-Euler slow increments with dynamic T_s.  Keep the dycore
        # physics closure free of T_s so it does not re-JIT every slab step.
        physics_fn, _ = split_rad_from_other_physics(
            grid, hc, tm,
            radiation_config=None,
            microphysics_config=microphysics_config,
            dt=args.dt, T_sfc=args.T_sfc, p_sfc=p_sfc_rcemip,
            surface_flux=False,
        )
        rad_physics_fn = None
        if radiation_config is not None:
            sim_refresh_s = args.radiation_interval * args.dt
            print(f"  LAND RADIATION GATED: refresh every "
                  f"{args.radiation_interval} steps = {sim_refresh_s:.0f} s "
                  "sim time; cached surface SW/LW force the slab between "
                  "refreshes.")
    elif args.radiation_interval > 1 and radiation_config is not None:
        # Gated radiation: split heavy radiation from light per-step physics.
        # rad_tendency cached for radiation_interval outer steps, applied as
        # forward Euler each step. Standard CRM treatment (SAM, WRF, CM1).
        physics_fn, rad_physics_fn = split_rad_from_other_physics(
            grid, hc, tm,
            radiation_config=radiation_config,
            microphysics_config=microphysics_config,
            dt=args.dt, T_sfc=args.T_sfc, p_sfc=p_sfc_rcemip,
            surface_flux=not args.no_surface_flux,
        )
        sim_refresh_s = args.radiation_interval * args.dt
        print(f"  RADIATION GATED: refresh every {args.radiation_interval} "
              f"steps = {sim_refresh_s:.0f} s sim time "
              f"(RCEMIP typical 300-1800 s).")
        # Per codex iter-... MEDIUM#7: warn if interval is well outside
        # the operational CRM range (5-30 min sim).
        if sim_refresh_s > 1800.0:
            print(f"  WARN: radiation refresh interval {sim_refresh_s:.0f} s "
                  f"> 1800 s (30 min). Slow-process error grows linearly "
                  f"with interval; SAM/WRF/CM1 typical max = 30 min.")
        elif sim_refresh_s < 60.0:
            print(f"  WARN: radiation refresh interval {sim_refresh_s:.0f} s "
                  f"< 60 s. RRTMGP cost dominates the run; consider "
                  f"--radiation-interval >= {int(300 / args.dt)}.")
    else:
        physics_fn = make_rcemip_physics(
            grid, hc, tm,
            radiation_config=radiation_config,
            microphysics_config=microphysics_config,
            dt=args.dt,
            T_sfc=args.T_sfc, p_sfc=p_sfc_rcemip,
            surface_flux=not args.no_surface_flux,
        )
        rad_physics_fn = None

    # Tracer slot count per scheme (codex iter-... MEDIUM#5):
    #   morrison / seifert_beheng / p3 / thompson: 9 slots (q_v, q_c, q_r,
    #     q_i, q_s, q_g, N_c, N_r, N_i). The slot LAYOUT is FIXED in the
    #     microphysics integration adapter (N_i always lives at slot 8),
    #     so thompson — even though it only PREDICTS N_i (single-moment
    #     for the warm species) — still needs the full length-9 array to
    #     receive its N_i tendency. Allocating 7 dropped slot 8 and the
    #     adapter raised ValueError.
    #   kessler / sundqvist / ml_emulator / none: 3 slots (q_v, q_c, q_r)
    # The microphysics integration validates the slot count at JIT time
    # and raises ValueError if too few — but we allocate generously
    # here to surface schema errors at parse time, not deep in JIT.
    if args.microphysics == "morrison":
        # slot [9] = prognostic snow number, [10] = prognostic graupel number
        # ⇒ fully double-moment M2005 (snow + graupel).
        n_tracers = 11
    elif args.microphysics in ("seifert_beheng", "p3", "thompson", "fast_sbm"):
        # fast_sbm (FSBM-2): q_v,q_c,q_r + N_c,N_r live, ice slots zero ⇒ 9.
        n_tracers = 9
    else:
        n_tracers = 3
    if snd is not None:
        # Shared SAM-deck IC builder: u/v/q_v interpolated from the sounding,
        # θ' = θ_snd − θ_ref (≈0, θ_ref IS the sounding) + the same bottom-level
        # symmetry-breaking seed, and the same DRY hydrostatic ρ' closure the
        # analytic path uses.
        state = build_sounding_initial_state(
            snd, grid, hc, args, n_tracers, dtype=dtype)
    else:
        state = _build_rcemip_initial_state(
            grid, hc, dtype=dtype, theta_noise_amp=args.theta_noise_amp,
            n_tracers=n_tracers, seed_kind=args.seed_kind,
            seed_kmax=args.seed_kmax, q_sfc=q_sfc_rce,
        )
    # Optional restart from a checkpoint (resume long runs after a crash/fix).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rce_checkpoint  # noqa: E402
    _start_step = 0
    if args.restart:
        _rpath = (rce_checkpoint.latest(args.output / "checkpoints")
                  if args.restart == "latest" else Path(args.restart))
        if _rpath is None:
            print(f"  --restart {args.restart}: no checkpoint found, fresh IC.")
        else:
            state, _start_step, _t0 = rce_checkpoint.load(
                _rpath, state, reset_condensate=args.restart_reset_condensate,
                seed_numbers=args.restart_seed_numbers)
            print(f"  RESTART from {_rpath} at step {_start_step} "
                  f"(t={_t0:.0f}s, day {_t0/86400:.2f})")
    if args.vortex_seed_vmax > 0.0:
        state = _add_vortex_seed(
            state, grid, hc, vmax=args.vortex_seed_vmax,
            rmax=args.vortex_seed_rmax, ztop=args.vortex_seed_ztop, dtype=dtype)
        print(f"  VORTEX SEED: cyclonic Vmax={args.vortex_seed_vmax} m/s @ "
              f"r={args.vortex_seed_rmax/1e3:.0f} km, ztop={args.vortex_seed_ztop/1e3:.0f} km "
              f"(f0={args.f0:.2e})")
    land_T_sfc = None
    land_emissivity = _radiation_surface_emissivity(radiation_config)
    if args.land:
        _land_T0 = args.land_T_init if args.land_T_init is not None else args.T_sfc
        land_T_sfc = jnp.full(
            (grid.ny, grid.nx), _land_T0, dtype=dtype,
        )
        if args.restart and args.land_T_init is None:
            print("  LAND restart note: slab T_s is not stored in the "
                  "atmospheric checkpoint; initializing it from --T-sfc. "
                  "Use --land-T-init to prescribe restart skin temperature.")
    _ckpt_every = args.checkpoint_every if args.checkpoint_every > 0 else 0
    mass0 = float(compute_dry_mass_plane(state, grid, hc, tm))

    snap_dir = args.output / "snapshots"
    if args.snapshot_every > 0:
        snap_dir.mkdir(parents=True, exist_ok=True)

    # Additive 3D/surface snapshot saver (separate module; see rce_snapshot.py).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    import rce_snapshot  # noqa: E402
    _snap_heights = tuple(float(x) for x in args.snapshot_heights.split(","))
    _surf_every = (int(round(args.snapshot_days * 86400.0 / args.dt))
                   if args.snapshot_days > 0 else 0)
    _snap3d_steps = set()
    if args.snapshot3d_days.strip():
        for _d in args.snapshot3d_days.split(","):
            _snap3d_steps.add(int(round(float(_d) * 86400.0 / args.dt)))
    if _surf_every or _snap3d_steps:
        print(f"  RCE snapshots: surface/levels every {_surf_every} steps "
              f"({args.snapshot_days} d); 3D dumps at steps "
              f"{sorted(_snap3d_steps)} (days {args.snapshot3d_days})")

    if args.land:
        print("\nstep    t [s]    max|w|     min(theta')   max(theta')   "
              "max(q_v)   d(mass)    T_s[min/mean/max] Qs SH LH")
    else:
        print("\nstep    t [s]    max|w|     min(theta')   max(theta')   "
              "max(q_v)   d(mass)")

    # Gated-radiation runtime state. rad_physics_fn is None when
    # radiation is either off or runs every step inside physics_fn.
    # Bind grid/hc/tm via closure (Python statics) — they're pytrees of
    # arrays; passing as JIT args would require static_argnums=hashable
    # which they aren't. Closure capture is safe: the wrapper recompiles
    # iff the state's shape/dtype changes, not on every call.
    if rad_physics_fn is not None:
        def _rad_wrapper(s):
            return rad_physics_fn(s, grid, hc, tm)
        rad_jit = jax.jit(_rad_wrapper)
    else:
        rad_jit = None
    cached_rad_tend = None
    cached_land_rad_sfc = None
    land_surface_diag = None
    land_energy_diag = None
    land_rad_refresh_jit = None
    land_sfc_flux_jit = None
    if args.land:
        _zero_sfc = jnp.zeros_like(land_T_sfc)
        cached_land_rad_sfc = {
            "sw_down": _zero_sfc,
            "lw_down": _zero_sfc,
            "sw_up": _zero_sfc,
            "lw_up": _zero_sfc,
        }
        land_surface_diag = {
            "shflx": _zero_sfc,
            "lhflx": _zero_sfc,
            "lhflx_potential": _zero_sfc,
        }
        land_energy_diag = {
            "sw_net": _zero_sfc,
            "lw_net": _zero_sfc,
            "lw_up": _zero_sfc,
            "R_net": _zero_sfc,
            "Q_slab": _zero_sfc,
        }
        if not args.no_physics:
            _land_rad_refresh_fn = _make_land_radiation_refresh_fn(
                radiation_config, args.land_albedo, land_emissivity,
            )
            if _land_rad_refresh_fn is not None:
                def _land_rad_refresh_wrapper(s, T_s):
                    return _land_rad_refresh_fn(s, grid, hc, tm, T_s)
                land_rad_refresh_jit = jax.jit(_land_rad_refresh_wrapper)
            if not args.no_surface_flux:
                def _land_sfc_flux_wrapper(s, T_s):
                    return land_surface_flux_tendencies(
                        s, grid, hc, tm, T_s,
                        p_sfc=p_sfc_rcemip, beta=args.land_beta,
                    )
                land_sfc_flux_jit = jax.jit(_land_sfc_flux_wrapper)

    for i in range(_start_step, args.steps):
        if args.land:
            if land_rad_refresh_jit is not None and (
                i % args.radiation_interval == 0 or cached_rad_tend is None
            ):
                cached_rad_tend, cached_land_rad_sfc = land_rad_refresh_jit(
                    state, land_T_sfc,
                )
            if cached_rad_tend is not None:
                state = apply_radiation_forward_euler(
                    state, cached_rad_tend, args.dt,
                )
            if land_sfc_flux_jit is not None:
                surface_tend, land_surface_diag = land_sfc_flux_jit(
                    state, land_T_sfc,
                )
                state = apply_plane_tendency_forward_euler(
                    state, surface_tend, args.dt,
                )
            land_T_sfc, land_energy_diag = update_land_slab_temperature(
                land_T_sfc,
                cached_land_rad_sfc["sw_down"],
                cached_land_rad_sfc["lw_down"],
                land_surface_diag["shflx"],
                land_surface_diag["lhflx"],
                args.dt,
                args.land_heat_capacity,
                args.land_albedo,
                land_emissivity,
            )
            state = model.step(state, dt=args.dt, physics_fn=physics_fn)
        else:
            if rad_jit is not None and (
                i % args.radiation_interval == 0 or cached_rad_tend is None
            ):
                cached_rad_tend = rad_jit(state)
            if cached_rad_tend is not None:
                state = apply_radiation_forward_euler(
                    state, cached_rad_tend, args.dt,
                )
            state = model.step(state, dt=args.dt, physics_fn=physics_fn)
        # POSITIVITY GUARD (CRM-dycore campaign, codex-reviewed): keep the water
        # tracers >=0 in the STATE after each step, matching run_rce_mpi_long.
        # The advective-form tracer advection under strong div(u) — e.g. a
        # weno5-θ' convective overshoot driving noisy w — can leave q<0 in the
        # stored state; the microphysics-read clip (PR #966) alone does not
        # repair it.
        #
        # mode="clip" (local max(q,0)) is used over mode="compensated" ON PURPOSE:
        # codex recommended compensated for exact water-mass conservation, but the
        # GLOBAL redistribution overreacts in the noisy weno5-θ' regime — when
        # many cells go slightly negative the compensated scale collapses toward
        # 0 and zeros positive water, amplifying grid-scale noise (empirically
        # max|w| 7.1 vs 0.35 m/s for weno5 f64 at 1 day). clip is local + robust;
        # its water-creation bias is negligible: the negatives are ~1e-4 and only
        # fire under pathological noisy convection, and the reference van_leer
        # config leaves machine-zero negatives so clip is a no-op there.
        # ponytail: local clip, revisit a mass-conserving *local* limiter (per-
        # column borrow) if a long equilibrium run shows CWV drift.
        state = apply_positive_filter_state(state, mode="clip")
        if (i + 1) % args.print_every == 0 or i == 0:
            t = (i + 1) * args.dt
            max_w = float(jnp.max(jnp.abs(state.w.data)))
            min_th = float(jnp.min(state.theta_prime.data))
            max_th = float(jnp.max(state.theta_prime.data))
            max_qv = float(jnp.max(state.tracers.data[..., 0]))
            max_rhop = float(jnp.max(jnp.abs(state.rho_prime.data)))
            min_tr = float(jnp.min(state.tracers.data))
            mass = float(compute_dry_mass_plane(state, grid, hc, tm))
            rel = abs(mass - mass0) / abs(mass0)
            line = (f"{i+1:5d}  {t:7.2f}  {max_w:9.3e}  "
                    f"{min_th:12.4e}  {max_th:12.4e}  {max_qv:9.3e}  "
                    f"{rel:8.2e}  rho'={max_rhop:.2e} minTr={min_tr:.2e}")
            if args.land:
                Ts_min = float(jnp.min(land_T_sfc))
                Ts_mean = float(jnp.mean(land_T_sfc))
                Ts_max = float(jnp.max(land_T_sfc))
                q_slab = float(jnp.mean(land_energy_diag["Q_slab"]))
                sh_mean = float(jnp.mean(land_surface_diag["shflx"]))
                lh_mean = float(jnp.mean(land_surface_diag["lhflx"]))
                line += (f"  T_s={Ts_min:.3f}/{Ts_mean:.3f}/{Ts_max:.3f}K"
                         f" Qs={q_slab:.2f} SH={sh_mean:.2f} LH={lh_mean:.2f}")
            print(line, flush=True)
            # Per-field NaN trace: report WHICH field fails first (the NaN may
            # originate in a tracer/rho' and only reach w a step later).
            _fld = {"u": state.u.data, "v": state.v.data, "w": state.w.data,
                    "theta'": state.theta_prime.data, "rho'": state.rho_prime.data}
            bad = [k for k, v in _fld.items() if not bool(jnp.all(jnp.isfinite(v)))]
            _tr = state.tracers.data
            badtr = [s for s in range(_tr.shape[-1])
                     if not bool(jnp.all(jnp.isfinite(_tr[..., s])))]
            if bad or badtr:
                print(f"\nNON-FINITE in fields={bad} tracer_slots={badtr} "
                      f"— aborting.")
                break
        if args.snapshot_every > 0 and (i + 1) % args.snapshot_every == 0:
            _emit_surface_snapshot_png(
                snap_dir, i + 1, (i + 1) * args.dt, state, grid, hc,
            )
            _emit_profile_npz(
                snap_dir, i + 1, (i + 1) * args.dt, state, hc,
            )
        # Surface + 4-level field snapshots (rce_snapshot) and full-3D viz dumps.
        if _surf_every and (i + 1) % _surf_every == 0:
            _surface_fields = (
                {"T_s": land_T_sfc} if args.land else None
            )
            rce_snapshot.save_surface_levels(
                args.output, i + 1, (i + 1) * args.dt, state, grid, hc,
                heights_m=_snap_heights, surface_fields=_surface_fields)
        if (i + 1) in _snap3d_steps:
            rce_snapshot.save_3d(
                args.output, i + 1, (i + 1) * args.dt, state, grid, hc)
            print(f"  [3D snapshot dumped @ day {(i+1)*args.dt/86400:.1f}]",
                  flush=True)
        if _ckpt_every and (i + 1) % _ckpt_every == 0:
            rce_checkpoint.save(args.output / "checkpoints", i + 1,
                                (i + 1) * args.dt, state)
            print(f"  [checkpoint @ step {i+1}, day {(i+1)*args.dt/86400:.2f}]",
                  flush=True)

    if args.snapshot_every > 0:
        _render_profile_evolution_png(
            snap_dir, args.output / "profile_evolution.png",
        )

    print(f"\nOutput: {args.output}")


def _emit_profile_npz(snap_dir: Path, step: int, t_s: float,
                       state, hc) -> None:
    """Save horizontal-mean vertical profiles per snapshot day.

    Profiles dumped: T(z), theta'(z), q_v(z), q_c(z), w_RMS(z), CWV(z).
    Read back by render_profile_evolution_png at end of run.
    """
    import numpy as np
    import jax.numpy as _jnp
    nlev = state.theta_prime.data.shape[-1]
    theta_p = np.asarray(state.theta_prime.data)
    rho_p = np.asarray(state.rho_prime.data)
    # w lives at half levels (nlev+1); average to full levels (nlev)
    # so the profile axis aligns with theta/qv/etc.
    w_half = np.asarray(state.w.data)
    w = 0.5 * (w_half[..., :-1] + w_half[..., 1:])
    theta_0 = np.asarray(hc.theta_ref)
    rho_0 = np.asarray(hc.rho_ref)
    pi_0 = np.asarray(hc.exner_ref)
    theta_total = theta_0 + theta_p
    # Hydrostatic Exner -> Temperature at full levels (cheap diagnostic).
    T = theta_total * pi_0
    q_v = np.asarray(state.tracers.data[..., 0])
    n_tr = state.tracers.data.shape[-1]
    q_c = np.asarray(state.tracers.data[..., 1]) if n_tr > 1 else None
    # Convective-intensity signatures for the SAM comparison (w'² variance,
    # updraft mass flux, condensate breakdown, cloud fraction) from the LIVE
    # tested diagnostics — NOT re-derived here (avoids numeric duplication).
    from legoesm.atmosphere.dynamics.crm.rce_diagnostics import (
        condensate_profile_plane, cloud_fraction_profile_plane,
        updraft_mass_flux_plane, vertical_velocity_variance_plane,
    )
    cond = condensate_profile_plane(state, hc)
    # Horizontal means over (ny, nx)
    np.savez(snap_dir / f"profile_step_{step:08d}.npz",
             step=step, t_s=t_s, z=np.asarray(hc.z_full),
             z_half=np.asarray(hc.z_half),
             T_mean=T.mean(axis=(0, 1)),
             theta_mean=theta_total.mean(axis=(0, 1)),
             theta_p_mean=theta_p.mean(axis=(0, 1)),
             theta_p_std=theta_p.std(axis=(0, 1)),
             qv_mean=q_v.mean(axis=(0, 1)),
             qv_std=q_v.std(axis=(0, 1)),
             qc_mean=(q_c.mean(axis=(0, 1)) if q_c is not None
                      else np.zeros(nlev)),
             w_RMS=np.sqrt((w ** 2).mean(axis=(0, 1))),
             w_var=np.asarray(vertical_velocity_variance_plane(state, hc)),
             updraft_mass_flux=np.asarray(
                 updraft_mass_flux_plane(state, hc)),
             q_cloud_mean=np.asarray(cond.q_cloud),
             q_precip_mean=np.asarray(cond.q_precip),
             cloud_fraction=np.asarray(
                 cloud_fraction_profile_plane(state, hc)),
             rho_mean=(rho_0 + rho_p.mean(axis=(0, 1))))


def _render_profile_evolution_png(snap_dir: Path, out: Path) -> None:
    """Compose a 5-panel profile-vs-day PNG from saved profile_*.npz."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np
    npz_files = sorted(snap_dir.glob("profile_step_*.npz"))
    if not npz_files:
        print(f"  no profile npz found in {snap_dir}; skip evolution PNG")
        return
    data = [np.load(f) for f in npz_files]
    days = np.array([d["t_s"] for d in data]) / 86400.0
    z_km = data[0]["z"] / 1000.0
    fig, axes = plt.subplots(1, 5, figsize=(18, 7), sharey=True)
    panels = [
        ("theta_mean", "θ(z) [K]", "viridis"),
        ("qv_mean", "q_v(z) [kg/kg]", "plasma"),
        ("qc_mean", "q_c(z) [kg/kg]", "Blues"),
        ("w_RMS", "w_RMS(z) [m/s]", "magma"),
        ("theta_p_std", "θ' std(z) [K]", "inferno"),
    ]
    cmap = plt.get_cmap("viridis", len(data))
    for ax, (key, label, _cm) in zip(axes, panels):
        for i, d in enumerate(data):
            ax.plot(d[key], z_km, color=cmap(i / max(1, len(data) - 1)),
                    linewidth=0.7, alpha=0.7)
        ax.set_xlabel(label)
        ax.grid(alpha=0.3)
    axes[0].set_ylabel("height [km]")
    sm = plt.cm.ScalarMappable(cmap=cmap,
                                norm=plt.Normalize(vmin=days[0],
                                                   vmax=days[-1]))
    sm.set_array([])
    cbar = fig.colorbar(sm, ax=axes, fraction=0.02, pad=0.04,
                        label="day")
    fig.suptitle("RCE horizontal-mean profile evolution", fontsize=13)
    fig.savefig(out, dpi=130, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {out}")


def _emit_surface_snapshot_png(snap_dir: Path, step: int, t_s: float,
                                state, grid, hc) -> None:
    """Write a 2x2 panel PNG of surface fields at this timestep.

    Panels: (q_v surface), (theta' surface), (max(w) column max),
    (column-integrated water vapor). Top-down: k_sfc = nlev-1.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    k_sfc = state.theta_prime.data.shape[-1] - 1
    qv_sfc = np.asarray(state.tracers.data[..., k_sfc, 0])
    th_sfc = np.asarray(state.theta_prime.data[..., k_sfc])
    w_col_max = np.asarray(jnp.max(jnp.abs(state.w.data), axis=-1))
    # CWV = sum(rho_v dz) = sum(q_v * rho_dry * dz)
    q_v = np.asarray(state.tracers.data[..., 0])  # (ny, nx, nlev)
    rho_total = np.asarray(hc.rho_ref + state.rho_prime.data)  # (ny, nx, nlev)
    dz = np.asarray(hc.dz)  # (nlev,)
    cwv = np.sum(q_v * rho_total * dz, axis=-1)  # kg/m^2

    day = t_s / 86400.0
    fig, axes = plt.subplots(2, 2, figsize=(11, 9))
    panels = [
        (qv_sfc, "q_v surface [kg/kg]", "BrBG"),
        (th_sfc, "theta' surface [K]", "RdBu_r"),
        (w_col_max, "max|w| over column [m/s]", "viridis"),
        (cwv, "column water vapor [kg/m^2]", "Blues"),
    ]
    for ax, (data, label, cmap) in zip(axes.flat, panels):
        im = ax.imshow(data, origin="lower", cmap=cmap, aspect="auto")
        ax.set_title(label)
        plt.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.suptitle(f"RCE plane CRM — day {day:.2f} (step {step})",
                 fontsize=12)
    fig.tight_layout()
    out = snap_dir / f"day_{day:07.2f}_step_{step:08d}.png"
    fig.savefig(out, dpi=110, bbox_inches="tight")
    plt.close(fig)


if __name__ == "__main__":
    main()
