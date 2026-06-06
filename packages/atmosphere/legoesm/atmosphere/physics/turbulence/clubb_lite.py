"""CLUBB-lite — simplified higher-order turbulence closure.

Carries five prognostic second-order moments and diagnoses cloud fraction
from a Gaussian PDF of the saturation deficit, following the approach of
Golaz et al. (2002) and Larson & Golaz (2005).

Prognostic moments
------------------
1. wp2   = w'^2            vertical velocity variance (TKE-like)
2. thlp2 = theta_l'^2      liquid water potential temperature variance
3. rtp2  = r_t'^2          total water mixing ratio variance
4. wpthlp = w'theta_l'     vertical heat flux
5. wprtp  = w'r_t'         vertical moisture flux

Budgets (all at full levels, semi-implicit dissipation):

  d(wp2)/dt   = 2*Km*S^2 + 2*(g/theta_v)*w'theta_v' - C1*wp2/tau + diff(wp2)
  d(thlp2)/dt = -2*wpthlp*(d_theta_l/dz) - C5*thlp2/tau + diff(thlp2)
  d(rtp2)/dt  = -2*wprtp*(d_r_t/dz)      - C5*rtp2/tau  + diff(rtp2)
  d(wpthlp)/dt = -wp2*(d_theta_l/dz) + (g/theta_v)*thlp2 - C4*wpthlp/tau + diff
  d(wprtp)/dt  = -wp2*(d_r_t/dz)     + (g/theta_v)*rtp2  - C4*wprtp/tau  + diff

where tau = l / sqrt(wp2) is the turbulence time scale.

Cloud fraction diagnosis:
  s = r_t - r_sat(T, p)       saturation deficit
  sigma_s = sqrt(rtp2)        width of total water PDF
  cf = 0.5 * erfc(-s / (sqrt(2) * sigma_s))

Eddy diffusivities:
  Km = C_K * l * sqrt(wp2)
  Kh = Km / Pr_t

References
----------
- Golaz, J.-C., Larson, V. E., & Cotton, W. R. (2002). A PDF-based
  model for boundary layer clouds. Part I: Method and model description.
  J. Atmos. Sci., 59, 3540-3551.
- Larson, V. E., & Golaz, J.-C. (2005). Using assumed probability
  distribution functions for HoC. J. Atmos. Sci., 62, 3620-3649.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import (
    saturation_mixing_ratio as _q_sat,
    saturation_mixing_ratio_dT as _dqsat_dT,
)
from legoesm.atmosphere.physics._shared import mixing_length, virtual_temperature
from legoesm.atmosphere.physics.turbulence.config import CLUBBLiteConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def clubb_lite_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    tke: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: CLUBBLiteConfig,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Compute turbulence tendencies using CLUBB-lite higher-order closure.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    tke : jax.Array
        Vertical velocity variance w'^2 [m^2/s^2], shape (ncol, nlev).
        (Re-uses the TKE array slot for wp2.)
    p_full : jax.Array
        Pressure at full levels [Pa], shape (ncol, nlev).
    p_half : jax.Array
        Pressure at half levels [Pa], shape (ncol, nlev+1).
    z_full : jax.Array
        Height at full levels [m], shape (ncol, nlev).
    z_half : jax.Array
        Height at half levels [m], shape (ncol, nlev+1).
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : CLUBBLiteConfig

    Returns
    -------
    TurbulenceOutput
        Turbulence tendencies and diagnostics.
    tke_new : jax.Array
        Updated w'^2 [m^2/s^2], shape (ncol, nlev).
    """
    ncol, nlev = T.shape

    # ===== Geometry =====
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])  # (ncol, nlev)
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    # ===== Initialize prognostic moments =====
    # wp2 is carried via the tke slot
    wp2 = jnp.maximum(tke, config.tke_min)

    # Higher moments initialized from wp2 (cold start: proportional to wp2)
    # In a full implementation these would be carried as separate state arrays.
    # For "lite" version: diagnose thlp2, rtp2, wpthlp, wprtp from gradients.
    sqrt_wp2 = jnp.sqrt(wp2)

    # ===== Mixing length =====
    l_mix = mixing_length(z_full, config.l_mix_max)  # (ncol, nlev)
    l_mix_safe = jnp.clip(l_mix, 1.0, None)

    # iter-172 F841: removed unused ``tau_turb`` (consumed
    # only by the dead higher-moment / cloud-fraction block
    # — see comment block below).

    # ===== Eddy diffusivities =====
    Km_full = config.C_K * l_mix * sqrt_wp2  # (ncol, nlev)
    Kh_full = Km_full / config.Pr_t

    Km_half = 0.5 * (Km_full[:, :-1] + Km_full[:, 1:])  # (ncol, nlev-1)
    Kh_half = 0.5 * (Kh_full[:, :-1] + Kh_full[:, 1:])

    # ===== Vertical gradients at half-levels =====
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2  # (ncol, nlev-1)

    # Virtual potential temperature for buoyancy
    exner_pref = (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2_half = (constants.g / jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz

    # iter-172 F841: removed ``exner`` / ``theta`` /
    # ``dtheta_dz`` / ``drt_dz`` — only consumed by the dead
    # higher-moment + cloud-fraction block (see comment after
    # the eddy-diffusivity section).

    # Interpolate to full levels
    def _half_to_full(field_half):
        """Interpolate (ncol, nlev-1) half-level field to (ncol, nlev)."""
        mid = 0.5 * (field_half[:, :-1] + field_half[:, 1:])
        return jnp.concatenate([
            field_half[:, :1], mid, field_half[:, -1:]
        ], axis=1)

    S2 = _half_to_full(S2_half)
    N2 = _half_to_full(N2_half)
    # iter-172 F841: removed unused ``dtheta_dz_full`` /
    # ``drt_dz_full`` (consumed only by the removed dead
    # higher-moment + cloud-fraction block).

    # ===== Moment budgets (semi-implicit) =====

    # --- w'^2 budget ---
    # Production: shear + buoyancy
    shear_prod = Km_full * S2
    buoy_prod = -Kh_full * N2  # buoyancy production (>0 when unstable)

    # Dissipation coefficient: C1/tau (semi-implicit)
    diss_wp2 = config.C_eps * sqrt_wp2 / l_mix_safe

    # Diffuse wp2.  Pin the surface_flux dtype to the input dtype so the
    # tridiagonal solve does not silently promote the column path to f64.
    wp2_diffused = implicit_vertical_diffusion(
        wp2, Km_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=wp2.dtype),
    )

    # Semi-implicit update
    wp2_new = (wp2_diffused + dt * (shear_prod + buoy_prod)) / (
        1.0 + dt * diss_wp2
    )
    wp2_new = jnp.maximum(wp2_new, config.tke_min)

    # iter-172 (F841 audit): removed dead higher-moment +
    # cloud-fraction diagnostics that were computed every
    # timestep but never returned through ``TurbulenceOutput``.
    # The dead blocks were:
    #   * ``wpthlp = -Kh_full * dtheta_dz_full`` (line ~224)
    #   * ``wprtp  = -Kh_full * drt_dz_full``    (line ~225)
    #   * ``thlp2`` / ``rtp2`` flux-gradient variances (used
    #     only inside the discarded cloud-fraction block)
    #   * ``sigma_s = sqrt(rtp2)`` (overridden by sigma_s_eff)
    #   * Full Gaussian-PDF cloud fraction with ``erfc`` /
    #     ``q_sat_val`` / ``s_mean`` / ``dqs_dT`` /
    #     ``sigma_s_eff`` (lines ~249-268).
    #
    # These are real physics that someone intended to wire up
    # to a unified CLUBB-style cloud scheme, but the
    # ``TurbulenceOutput`` NamedTuple has never carried
    # ``cloud_fraction`` and downstream cloud-scheme dispatch
    # uses Sundqvist / Xu-Randall instead.  Removing the dead
    # computations saves ~6 jnp ops + 1 ``erfc`` per timestep
    # per AMIP-active column with no behavioural change.
    #
    # If a future CLUBB unified scheme is wired in, restore
    # by:
    #   1. Adding ``cloud_fraction: jax.Array`` to
    #      ``TurbulenceOutput`` (output.py).
    #   2. Re-introducing the PDF block here.
    #   3. Wiring the cloud scheme dispatch to consume
    #      ``turbulence_output.cloud_fraction`` when
    #      ``cloud_scheme == "clubb"``.
    # The block was: ``s_mean = q_v - _q_sat(T, p_full);
    # sigma_s_eff = sqrt(rtp2 + (dqs_dT * exner)**2 * thlp2);
    # cloud_fraction = 0.5 * erfc(-s_mean / (sqrt(2) *
    # sigma_s_eff))``.

    # ===== Surface fluxes =====
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # ===== Apply implicit vertical diffusion =====
    # Heat in θ-space (dry-adiabat neutral); momentum and moisture raw.
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    # ===== PBL height =====
    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    output = TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )

    return output, wp2_new
