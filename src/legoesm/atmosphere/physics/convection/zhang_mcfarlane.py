"""Zhang & McFarlane (1995) deep-convection scheme.

A single-plume mass-flux scheme with a quasi-equilibrium CAPE-relaxation
closure: the cloud-base mass flux ``M_b`` is diagnosed from the column
CAPE excess over a threshold and relaxed (implicit Euler) toward the
diagnosed equilibrium value over a tunable timescale.  The plume itself
is integrated using the shared
:func:`legoesm.atmosphere.physics.convection._plume.entraining_detraining_plume`
helper; the environmental tendencies (compensating subsidence +
detrainment) are computed by the existing
:func:`legoesm.atmosphere.physics.convection.mass_flux._apply_mass_flux_kernel`
so this scheme reuses every piece of column physics rather than
re-implementing it.

The scheme is **smooth-everywhere** in the AD sense:

* ``CAPE > threshold`` is replaced by ``cape_trigger`` (sigmoid).
* ``(CAPE - threshold)+`` uses ``smooth_positive_part``.
* Cloud-base / LFC / LNB localization uses the smooth-fractional
  level diagnostics from ``_plume``.
* The plume integrator's mass-flux profile is gated by a sigmoid on
  buoyancy, not a hard cut.

Convective momentum transport (CMT) is enabled by default via the
Gregory et al. 1997 closure
(:func:`legoesm.atmosphere.physics.convection._plume.cmt_gregory_1997`).

References
----------
- Zhang, G. J., & McFarlane, N. A. (1995). Sensitivity of climate
  simulations to the parameterization of cumulus convection in the
  Canadian Climate Centre general circulation model.  *Atmos.-Ocean*,
  33(3), 407–446.
- Gregory, D., Kershaw, R., & Inness, P. M. (1997). Parametrization of
  momentum transport by convection. II.  *Quart. J. Roy. Meteor. Soc.*,
  123, 1153–1183.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.atmosphere.physics.thermodynamics import (
    compute_cape,
    compute_moist_adiabat,
)

from legoesm.atmosphere.physics.convection.config import ZhangMcFarlaneConfig
from legoesm.atmosphere.physics.convection.output import ConvectionOutput
from legoesm.atmosphere.physics.convection.mass_flux import (
    _apply_mass_flux_kernel,
    _compute_column_geometry,
)
from legoesm.atmosphere.physics.convection._triggers import (
    cape_trigger,
    smooth_positive_part,
)
from legoesm.atmosphere.physics.convection._plume import (
    cmt_gregory_1997,
    compute_lcl,
    entraining_detraining_plume,
)


__all__ = ("zhang_mcfarlane_convection",)


def zhang_mcfarlane_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    u: jax.Array,
    v: jax.Array,
    conv_prog_profile: jax.Array,
    dt: float,
    config: ZhangMcFarlaneConfig = ZhangMcFarlaneConfig(),
) -> tuple[ConvectionOutput, jax.Array]:
    """Zhang-McFarlane deep convection (smooth, differentiable).

    Parameters
    ----------
    T : jax.Array, shape (ncol, nlev)
        Environmental temperature [K].  Surface at ``[:, -1]``.
    q_v : jax.Array, shape (ncol, nlev)
        Water-vapor specific humidity [kg/kg].
    p_full, p_half : jax.Array
        Full / half-level pressures [Pa].  Shapes ``(ncol, nlev)`` and
        ``(ncol, nlev+1)`` respectively.
    u, v : jax.Array, shape (ncol, nlev)
        Environmental wind components [m/s].  Used by the Gregory et
        al. 1997 CMT closure when ``config.enable_cmt = True``.
    conv_prog_profile : jax.Array, shape (ncol, nlev)
        Convection prognostic carry (PR-0 schema).  ZM uses only the
        surface-adjacent slot ``[:, -1]`` to remember the previous
        cloud-base mass flux ``M_b`` for implicit-Euler relaxation
        toward the diagnosed equilibrium value; aloft slots are
        unused (zeros in / zeros out).
    dt : float
        Time step [s].
    config : ZhangMcFarlaneConfig
        Scheme tunables — see :class:`ZhangMcFarlaneConfig`.

    Returns
    -------
    out : ConvectionOutput
        Tendencies on environment T, q_v, q_c plus the CAPE diagnostic
        and (when CMT is enabled) du/dt, dv/dt.
    conv_prog_profile_new : jax.Array, shape (ncol, nlev)
        Updated carry with the relaxed ``M_b_new`` packed at
        ``[:, -1]``; aloft entries are zero.
    """
    ncol, nlev = T.shape

    # -- Column geometry, moist adiabat, CAPE --------------------------------
    dz, rho, z = _compute_column_geometry(T, p_full, p_half)
    T_base = T[:, -1]
    q_base = q_v[:, -1]
    p_base = p_full[:, -1]

    T_moist = compute_moist_adiabat(T_base, p_full)
    cape = compute_cape(T, T_moist, p_full, p_half)

    # -- Smooth CAPE trigger and cloud-base mass-flux closure ---------------
    cape_weight = cape_trigger(
        cape, config.cape_threshold, config.cape_sharpness,
    )
    M_b_eq = (
        cape_weight
        * smooth_positive_part(
            cape - config.cape_threshold, config.cape_sharpness,
        )
        / config.tau_cape
    )
    # Implicit-Euler relaxation toward equilibrium — stable for any
    # ``dt / tau_cape`` ratio:
    #     M_b_new = (M_b_old + (dt/tau) * M_b_eq) / (1 + dt/tau).
    M_b_old = conv_prog_profile[:, -1]
    dt_over_tau = dt / jnp.maximum(config.tau_cape, dt)
    M_b = (M_b_old + dt_over_tau * M_b_eq) / (1.0 + dt_over_tau)
    M_b = jnp.maximum(M_b, 0.0)

    # -- Plume launch / cloud-base index ------------------------------------
    # Surface parcel perturbed slightly per Zhang & McFarlane 1995 §3a;
    # this avoids zero-perturbation degeneracies and gives a smooth
    # cloud-base diagnosis.
    T_parcel = T_base + config.parcel_dT
    q_parcel = q_base + config.parcel_dq
    lcl = compute_lcl(T_parcel, q_parcel, p_base, p_full)
    k_base_smooth = lcl.k_lcl_smooth

    # Constant entrainment / detrainment profiles in the surface-last
    # convention (level index nlev-1 = surface, 0 = top).  Tunable per
    # scheme but identical across levels in the standard Zhang-McFarlane
    # bulk plume.
    eps_profile = jnp.full_like(T, config.epsilon_0)
    dlt_profile = jnp.full_like(T, config.delta_0)

    plume = entraining_detraining_plume(
        T, q_v, p_full, p_half, z,
        T_parcel, q_parcel, k_base_smooth,
        eps_profile, dlt_profile, M_b,
    )

    # -- Environmental tendencies via the shared mass-flux kernel ----------
    dT_dt, dq_v_dt, _ = _apply_mass_flux_kernel(
        T, q_v, p_full,
        plume.T_u, plume.q_u, plume.M_u,
        z, rho, config.delta_0,
    )
    # Override the kernel's ``q_c`` source: the kernel assumes ``q_u``
    # carries any super-saturation (Arakawa-Wu convention), but our
    # plume saturates ``q_u`` internally and tracks cloud water in
    # ``plume.q_c_u``.  Use the plume's explicit cloud-water profile.
    rho_safe = jnp.clip(rho, 0.01, None)
    dq_c_conv_dt = config.delta_0 * plume.M_u * plume.q_c_u / rho_safe

    # -- Convective momentum transport --------------------------------------
    if config.enable_cmt:
        du_dt_conv, dv_dt_conv = cmt_gregory_1997(
            u, v, plume.M_u, None,
            p_full, p_half, rho,
            c_u=config.cmt_c_u, c_d=config.cmt_c_d,
        )
    else:
        du_dt_conv = None
        dv_dt_conv = None

    # -- Convective mask (column-mean diagnostic) ---------------------------
    convective_mask = cape_weight  # already a smooth (ncol,) indicator

    out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
        du_dt_conv=du_dt_conv,
        dv_dt_conv=dv_dt_conv,
    )

    # Pack the relaxed M_b back into the surface-adjacent carry slot
    # for downstream visibility (training diagnostics, conservation
    # checks).  Aloft slots are zero-filled — the orchestrator schema
    # is uniform across all schemes.
    conv_prog_profile_new = jnp.zeros_like(conv_prog_profile).at[:, -1].set(M_b)

    return out, conv_prog_profile_new
