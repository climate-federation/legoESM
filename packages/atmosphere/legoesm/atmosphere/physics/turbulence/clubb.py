"""CLUBB higher-order turbulence closure (fuller port; ``scheme="clubb"``).

This is the operational entry point for the fuller CLUBB port tracked in
``PORT_CLUBB.md`` — substantially richer than :mod:`clubb_lite`. The supporting
machinery (all golden-locked or parity-tested against CLUBB-JAX) is built up in
the ``clubb_*.py`` modules: the staggered grid (:mod:`clubb_grid`), CAM-default
config (:mod:`clubb_config`), Flatau saturation (:mod:`clubb_saturation`),
``sigma_sqd_w``/Brunt-Vaisala (:mod:`clubb_helpers`), skewness diagnostics
(:mod:`clubb_skewness`), the parcel buoyant-sorting mixing length
(:mod:`clubb_mixing_length`), the ADG1 assumed-PDF cloud/buoyancy closure
(:mod:`clubb_pdf`, :mod:`clubb_pdf_moments`), the implicit solvers
(:mod:`clubb_solve`), and the moment advances (:mod:`clubb_moments`).

Phasing (the scheme is wired in and runnable now; fidelity deepens per phase):

  * **Phase 1 (this entry):** uses CLUBB's exact **parcel buoyant-sorting
    length scale** ``Lscale`` (``compute_mixing_length``, golden-locked vs
    CLUBB-JAX) to set the eddy diffusivity ``Km = c_K · Lscale · sqrt(em)`` —
    the distinctive CLUBB feature, replacing clubb_lite's Blackadar length — and
    diagnoses the cloud fraction from the ADG1 PDF. Mean fields (u, v, T, q_v)
    are advanced by the implicit eddy diffusion shared with the other legoESM
    schemes; ``wp2`` is carried (via the ``tke`` slot) with a production /
    dissipation / diffusion budget whose dissipation time scale is
    ``tau = Lscale / sqrt(em)``.
  * **Phase 2+ (in progress):** replace the diagnostic moment budget with the
    full prognostic higher-order moment transport — ``advance_wp2_wp3``,
    ``advance_xp2_xpyp``, ``advance_xm_wpxp`` — coupled through the ADG1 PDF
    buoyancy flux ``wpthvp`` and the implicit tridiag/penta solves.

The ``TurbulenceOutput`` contract carries no ``cloud_fraction`` field (see the
clubb_lite docstring), so the PDF cloud fraction is computed and exposed only
through diagnostics for now; the eddy-diffusion tendencies are the live output.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
from legoesm.atmosphere.physics._shared import virtual_temperature
from legoesm.atmosphere.physics.turbulence.clubb_config import CLUBBConfig, derive_lmin
from legoesm.atmosphere.physics.turbulence.clubb_diagnostic import (
    diagnose_cloud_and_buoyancy,
)
from legoesm.atmosphere.physics.turbulence.clubb_grid import (
    flip_vertical,
    make_clubb_grid_from_levels,
    zt2zm,
)
from legoesm.atmosphere.physics.turbulence.clubb_mixing_length import (
    compute_mixing_length,
    set_Lscale_max,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import compute_surface_fluxes
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)

from legoesm import constants

# Eddy-diffusivity and dissipation coefficients are read from CLUBBParams
# (c_K, beta, ...) — no hardcoded tunables here.
_PR_T = 1.0   # phase-1 turbulent Prandtl number (Kh = Km/_PR_T); refined in P2


def clubb_turbulence(
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
    config: CLUBBConfig,
) -> tuple[TurbulenceOutput, jax.Array]:
    """CLUBB turbulence tendencies (phase 1: CLUBB ``Lscale`` eddy diffusion).

    Parameters mirror :func:`clubb_lite.clubb_lite_turbulence` (the ``tke`` slot
    carries ``wp2`` [m^2/s^2]); all column fields are TOP-DOWN ``(ncol, nlev)``,
    half-level fields ``(ncol, nlev+1)``.

    Returns
    -------
    TurbulenceOutput
        Tendencies + diffusivities + surface diagnostics.
    wp2_new : jax.Array
        Updated ``w'^2`` [m^2/s^2], ``(ncol, nlev)``, carried to the next step.
    """
    ncol, nlev = T.shape
    params = config.params

    wp2 = jnp.maximum(tke, config.tke_min)
    sqrt_wp2 = jnp.sqrt(wp2)

    # ---- Thermodynamics (top-down) ----
    exner = (p_full / constants.p_ref) ** constants.kappa     # (ncol, nlev)
    theta = T / exner
    theta_v = virtual_temperature(T, q_v) / exner             # = theta * (1 + 0.61 q_v)
    # Dry phase-1 mapping to CLUBB variables (rcm = 0): thl ~ theta, rt ~ q_v.
    thlm_td = theta
    rtm_td = q_v
    thvm_td = theta_v
    thv_ds_td = theta_v

    # ---- CLUBB parcel buoyant-sorting mixing length (ascending grid) ----
    gr = make_clubb_grid_from_levels(z_full, z_half)
    thvm = flip_vertical(thvm_td)
    thlm = flip_vertical(thlm_td)
    rtm = flip_vertical(rtm_td)
    exner_a = flip_vertical(exner)
    p_a = flip_vertical(p_full)
    thv_ds = flip_vertical(thv_ds_td)
    # TKE on momentum (zm) levels from the carried wp2 (ascending zt -> zm).
    em_zm = jnp.maximum(zt2zm(flip_vertical(wp2), gr), config.tke_min)

    mu = jnp.full((ncol,), params.mu)
    lmin = derive_lmin(params.lmin_coef)
    Lscale_max = set_Lscale_max(False, None, None, ncol)
    Lscale_a, _, _ = compute_mixing_length(
        thvm, thlm, rtm, em_zm, Lscale_max, p_a, exner_a, thv_ds,
        mu, lmin, False, gr,
    )
    Lscale = flip_vertical(Lscale_a)                          # back to top-down (ncol, nlev)
    Lscale = jnp.clip(Lscale, 1.0, None)

    # ---- Eddy diffusivities from the CLUBB length scale ----
    Km_full = params.c_K * Lscale * sqrt_wp2                  # (ncol, nlev)
    Kh_full = Km_full / _PR_T

    # ---- ADG1 double-Gaussian PDF: cloud fraction + moist buoyancy flux ----
    # (the distinctive CLUBB closure; ascending grid). The buoyancy production of
    # wp2 uses the PDF flux ``w'thv'`` (with its cloud-water latent-heat term),
    # not a plain down-gradient ``-Kh N2``.
    wp2_a = jnp.maximum(flip_vertical(wp2), config.tke_min)   # ascending zt
    Kh_a = (params.c_K / _PR_T) * Lscale_a * jnp.sqrt(wp2_a)
    cloud_frac_a, rcm_a, wpthvp_a = diagnose_cloud_and_buoyancy(
        thlm, rtm, wp2_a, exner_a, p_a, thv_ds, Kh_a, Lscale_a, gr, config)
    buoy_prod = flip_vertical((constants.g / jnp.clip(thvm, 1.0, None)) * wpthvp_a)

    # ---- Geometry + shear (top-down) ----
    dz_half = jnp.clip(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0, None)
    dz_layer = jnp.clip(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)
    Km_half = 0.5 * (Km_full[:, :-1] + Km_full[:, 1:])
    Kh_half = 0.5 * (Kh_full[:, :-1] + Kh_full[:, 1:])

    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2

    def _half_to_full(field_half):
        mid = 0.5 * (field_half[:, :-1] + field_half[:, 1:])
        return jnp.concatenate([field_half[:, :1], mid, field_half[:, -1:]], axis=1)

    S2 = _half_to_full(S2_half)

    # ---- wp2 budget (production - dissipation + diffusion); tau = Lscale/sqrt(wp2) ----
    shear_prod = Km_full * S2
    diss_wp2 = sqrt_wp2 / Lscale                              # 1/tau
    wp2_diffused = implicit_vertical_diffusion(
        wp2, Km_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=wp2.dtype),
    )
    wp2_new = (wp2_diffused + dt * (shear_prod + buoy_prod)) / (1.0 + dt * diss_wp2)
    wp2_new = jnp.clip(wp2_new, config.tke_min, config.wp2_max)

    # ---- Surface fluxes ----
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )
    sflx_u, sflx_v = tau_x, tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    # ---- Implicit vertical diffusion of the mean state ----
    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T)
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

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
