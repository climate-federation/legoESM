"""EDMF (Eddy-Diffusivity Mass-Flux) turbulence scheme.

Unified turbulence-convection framework that combines an eddy-diffusivity
(ED) component based on prognostic TKE with a mass-flux (MF) updraft
model. The ED part handles small-scale mixing while the MF part
represents coherent updraft transport in the convective boundary layer.

References
----------
- Siebesma, A. P., Soares, P. M. M., & Teixeira, J. (2007). A combined
  eddy-diffusivity mass-flux approach for the convective boundary layer.
  J. Atmos. Sci., 64, 1230-1248.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import virtual_temperature
from legoesm.atmosphere.physics.turbulence.config import EDMFConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)


def edmf_turbulence(
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
    config: EDMFConfig,
) -> tuple[TurbulenceOutput, jax.Array]:
    """Compute turbulence tendencies using EDMF.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water vapor mixing ratio [kg/kg], shape (ncol, nlev).
    tke : jax.Array
        Turbulent kinetic energy [m^2/s^2], shape (ncol, nlev).
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
    config : EDMFConfig

    Returns
    -------
    TurbulenceOutput
        Turbulence tendencies and diagnostics.
    tke_new : jax.Array
        Updated TKE [m^2/s^2], shape (ncol, nlev).
    """
    ncol, nlev = T.shape
    tke = jnp.maximum(tke, config.tke_min)

    # ===== ED part: TKE-based diffusion (same as tke.py) =====
    dz_half = jnp.abs(z_full[:, :-1] - z_full[:, 1:])  # (ncol, nlev-1)
    dz_half = jnp.clip(dz_half, 1.0, None)

    # Mixing length
    z_abs = jnp.clip(jnp.abs(z_full), 1.0, None)
    l_mix = constants.kappa_vk * z_abs / (
        1.0 + constants.kappa_vk * z_abs / config.l_mix_max
    )

    # Eddy diffusivities from TKE
    sqrt_tke = jnp.sqrt(tke)
    Km_full = config.Ck * l_mix * sqrt_tke
    Kh_full = Km_full / config.Pr_t

    Km_half = 0.5 * (Km_full[:, :-1] + Km_full[:, 1:])
    Kh_half = 0.5 * (Kh_full[:, :-1] + Kh_full[:, 1:])

    # ----- TKE budget -----
    du_dz = (u[:, :-1] - u[:, 1:]) / dz_half
    dv_dz = (v[:, :-1] - v[:, 1:]) / dz_half
    S2_half = du_dz ** 2 + dv_dz ** 2

    # ``exner_pref`` = (p_ref / p)^κ multiplies T to get θ.
    # Distinct from ``exner_inv`` = (p / p_ref)^κ used below to invert θ
    # back to T.  Variable shadowing was a fragility hazard — keep them
    # distinctly named.
    exner_pref = (constants.p_ref / jnp.clip(p_full, 1.0, None)) ** constants.kappa
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    N2_half = (constants.g / jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz

    # Interpolate to full levels: top/bottom take the nearest half-level
    # value, interior is the average of flanking half-levels.  Single
    # concatenate replaces alloc-zeros + 3 scatter ops.
    S2_interior = 0.5 * (S2_half[:, :-1] + S2_half[:, 1:])
    S2 = jnp.concatenate(
        [S2_half[:, :1], S2_interior, S2_half[:, -1:]], axis=1,
    )
    N2_interior = 0.5 * (N2_half[:, :-1] + N2_half[:, 1:])
    N2 = jnp.concatenate(
        [N2_half[:, :1], N2_interior, N2_half[:, -1:]], axis=1,
    )

    shear_prod = Km_full * S2
    buoyancy = -Kh_full * N2

    l_mix_safe = jnp.clip(l_mix, 1.0, None)
    diss_coeff = config.Ce * sqrt_tke / l_mix_safe

    # TKE diffusion
    dz_layer = jnp.abs(z_half[:, :-1] - z_half[:, 1:])
    dz_layer = jnp.clip(dz_layer, 1.0, None)

    tke_diffused = implicit_vertical_diffusion(
        tke, Km_half, rho, dz_layer, dz_half, dt,
        surface_flux=jnp.zeros(ncol, dtype=tke.dtype),
    )

    # ===== MF part: updraft model via jax.lax.scan =====
    # Surface fluxes
    tau_x, tau_y, shflx, lhflx, ustar = compute_surface_fluxes(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface,
    )
    ustar = jnp.clip(ustar, 1e-4, None)

    # Potential temperature for updraft.
    # ``exner_inv`` = (p / p_ref)^κ — divides T to give θ, multiplies
    # dθ to give dT.  Distinct from ``exner_pref`` = (p_ref / p)^κ
    # above.  Keeping the two names separate avoids the fragility
    # of reassigning a single ``exner`` to its reciprocal mid-function.
    exner_inv = (jnp.clip(p_full, 1.0, None) / constants.p_ref) ** constants.kappa
    theta = T / jnp.clip(exner_inv, 1.0e-8, None)

    # Initialize updraft at surface (bottom level = index nlev-1).  Pin
    # the carry dtype to the input field dtype so the scan body cannot
    # promote on x64 mode: ``jnp.full`` defaults to ``float64`` when
    # ``jax_enable_x64`` is True, which would poison ``carry[0]`` to
    # float64 while ``theta_u_init`` (= ``theta + 0.5``) stays at the
    # state precision and the scan rejects the carry-output mismatch.
    _dtype = T.dtype
    w_u_init = jnp.maximum(
        jnp.full(ncol, config.w_updraft_min, dtype=_dtype),
        (2.5 * ustar).astype(_dtype),
    )
    theta_u_init = (theta[:, -1] + config.parcel_dT).astype(_dtype)
    q_u_init = q_v[:, -1].astype(_dtype)  # same moisture

    # Scan from surface upward (reverse level index)
    # Levels are top-down, so we scan from nlev-1 to 0
    z_rev = z_full[:, ::-1]  # (ncol, nlev), surface first
    theta_rev = theta[:, ::-1]
    theta_v_rev = theta_v[:, ::-1]
    q_v_rev = q_v[:, ::-1]
    rho_rev = rho[:, ::-1]

    # Layer spacing from surface upward
    dz_upward = jnp.abs(jnp.diff(z_rev, axis=1))  # (ncol, nlev-1)

    def updraft_step(carry, inputs):
        w_u, theta_u, q_u = carry
        dz_k, theta_env, theta_v_env, q_env = inputs

        # Updraft virtual potential temperature
        theta_v_u = virtual_temperature(theta_u, q_u)

        # Buoyancy [m/s²]
        buoy = constants.g * (theta_v_u - theta_v_env) / jnp.clip(theta_v_env, 1.0, None)

        # Plume vertical velocity equation (Siebesma 2007 / Tan et al. 2018):
        #
        #     w · dw/dz = B − ε · w²       ⇔     d(w²)/dz = 2(B − ε·w²).
        #
        # **Backward-Euler in the linear damping term** so the update is
        # unconditionally stable for any ``ε·dz``:
        #     w²_new = (w²_old + 2·B·dz) / (1 + 2·ε·dz),    clamped ≥ 0.
        # Forward Euler ``w² + 2(B − ε·w²)·dz`` is only stable when
        # ``ε·dz < 0.5``; at T21 with 8 sigma levels ``dz`` can reach
        # ~3-5 km and the default ``ε = 1e-3 /m`` gives ``ε·dz ~ 3-5``,
        # which flips the sign of the w² coefficient and amplifies it
        # each layer — the original sample-46 NaN crash.  Backward
        # Euler matches the forward form to O(ε·dz) and is the
        # canonical choice for stiff linear damping.
        eps = config.entrainment_rate
        w_u_sq_raw = (w_u ** 2 + 2.0 * buoy * dz_k) / (1.0 + 2.0 * eps * dz_k)
        # AD-safe sqrt: ``d/dx sqrt(x) = 1/(2·sqrt(x))`` blows up at 0,
        # so floor the argument before sqrt and zero the result for
        # genuinely-negative w² (dead updraft) via an outer ``where``.
        w_u_sq_safe = jnp.maximum(w_u_sq_raw, 1.0e-20)
        w_u_new = jnp.where(w_u_sq_raw > 0.0, jnp.sqrt(w_u_sq_safe), 0.0)

        # Entrain environment air (mass-conservation form):
        #   d(φ_u)/dz = −ε · (φ_u − φ_env).
        # **Backward-Euler**: ``φ_new = (φ_old + ε·dz·φ_env) / (1 + ε·dz)``.
        # Unconditionally stable convex combination of plume and
        # environment for any ``ε·dz``.  The original forward-Euler form
        # ``φ + dz · [-ε(φ − φ_env)]`` flips the coefficient sign when
        # ``ε·dz > 1`` (which happens at coarse-vertical T21 with the
        # default ``ε = 1e-3 /m``); the resulting θ_u runaway drove the
        # NaN crash in ``combo_turb_edmf`` of the sweep.  Both forms
        # agree to O(ε·dz) so calibration with fine-vertical schemes is
        # preserved.
        eps_dz = eps * dz_k
        theta_u_new = (theta_u + eps_dz * theta_env) / (1.0 + eps_dz)
        q_u_new = (q_u + eps_dz * q_env) / (1.0 + eps_dz)

        # Smooth deactivation where w_u -> 0
        active = jax.nn.sigmoid(
            config.updraft_deactivation_sharpness * w_u_new / config.w_updraft_min
        )
        w_u_new = w_u_new * active
        theta_u_new = theta_u_new * active + theta_env * (1.0 - active)
        q_u_new = q_u_new * active + q_env * (1.0 - active)

        return (w_u_new, theta_u_new, q_u_new), (w_u_new, theta_u_new, q_u_new)

    # Pin scan inputs to the carry dtype (``_dtype = T.dtype``) so a
    # mixed-precision state — e.g. ``T`` from the storage policy
    # (typically f32) but ``q_v``/``q_c`` materialized via ``jnp.ones``
    # under ``JAX_ENABLE_X64=1`` (f64) — does not promote the scan
    # body output to f64 and trip ``scan``'s carry-dtype invariant.
    init_carry = (w_u_init, theta_u_init, q_u_init)
    # Scan over nlev-1 intervals (from surface upward, skipping surface itself)
    scan_inputs = (
        dz_upward.T.astype(_dtype),       # (nlev-1, ncol)
        theta_rev[:, 1:].T.astype(_dtype),
        theta_v_rev[:, 1:].T.astype(_dtype),
        q_v_rev[:, 1:].T.astype(_dtype),
    )

    _, (w_u_scan, theta_u_scan, q_u_scan) = jax.lax.scan(
        updraft_step, init_carry, scan_inputs,
    )
    # w_u_scan: (nlev-1, ncol), surface-to-top order

    # Assemble full updraft profiles (surface first)
    w_u_full = jnp.concatenate([w_u_init[None, :], w_u_scan], axis=0)  # (nlev, ncol)
    theta_u_full = jnp.concatenate([theta_u_init[None, :], theta_u_scan], axis=0)
    q_u_full = jnp.concatenate([q_u_init[None, :], q_u_scan], axis=0)

    # Transpose and reverse back to top-down
    w_u = w_u_full[::-1].T         # (ncol, nlev)
    theta_u = theta_u_full[::-1].T
    q_u = q_u_full[::-1].T

    # Mass flux: M = a_updraft * rho * w_u.
    # Note on column conservation: in this simplified-EDMF formulation
    # the BC is M[surface] = a_updraft·ρ·w_u_init > 0 (with surface
    # mass-source matched to the bulk-formula shflx/lhflx wired through
    # the implicit ED solve), and M smoothly decays via the active
    # gate (line 213) above the PBL top so M[top] ≈ 0 naturally.
    # Strict MF-only column closure is approximate; the small residual
    # is folded into the existing ED + bulk-formula surface-flux
    # accounting (similar to CAM EDMF).  Hard-zeroing M at boundaries
    # would zero the legitimate surface-coupled MF transport — the
    # iter-50 audit attempt to do so broke
    # ``test_mass_flux_active`` and was reverted.
    M = config.a_updraft * rho * w_u  # (ncol, nlev)

    # Explicit-Euler CFL cap on the mass-flux transport.  The MF tendency
    # is ``-(1/ρ) d(M·(φ_u-φ))/dz``; the effective layer Courant number
    # is ``M·dt/(ρ·dz)``.  For deep convection (w_u up to ~10 m/s) M can
    # reach values that drive ``M·dt/(ρ·dz) > 1`` on coarse-vertical
    # boundary layers — the centered-FD update then overshoots and the
    # ED implicit solve cannot recover the integrity of θ/q.  Cap M at
    # the local layer-mass-per-step.
    M_max = 0.5 * rho * dz_layer / jnp.maximum(dt, 1.0e-12)
    M = jnp.minimum(M, M_max)

    # MF tendencies: d(phi)/dt_mf = -(1/rho) * d(M * (phi_u - phi_env)) / dz
    # Compute vertical derivative of mass flux transport
    def _mf_tendency(phi, phi_u):
        flux = M * (phi_u - phi)  # (ncol, nlev)
        # Centered FD interior + one-sided FD at top/bottom; single
        # concatenate replaces alloc-zeros + 3 scatter ops.
        dflux_top = (flux[:, :1] - flux[:, 1:2]) / dz_layer[:, :1]
        dflux_int = (flux[:, :-2] - flux[:, 2:]) / (2.0 * dz_layer[:, 1:-1])
        dflux_bot = (flux[:, -2:-1] - flux[:, -1:]) / dz_layer[:, -1:]
        dflux_dz = jnp.concatenate([dflux_top, dflux_int, dflux_bot], axis=1)
        return -dflux_dz / jnp.clip(rho, 0.01, None)

    dtheta_dt_mf = _mf_tendency(theta, theta_u)
    dT_dt_mf = dtheta_dt_mf * exner_inv
    dq_dt_mf = _mf_tendency(q_v, q_u)

    # ===== ED tendencies via implicit diffusion =====
    sflx_u = tau_x
    sflx_v = tau_y
    sflx_T = shflx / constants.c_pd
    sflx_q = lhflx / constants.L_v

    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = implicit_vertical_diffusion_theta(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    # Combined: ED + MF
    du_dt = (u_new - u) / dt
    dv_dt = (v_new - v) / dt
    dT_dt = (T_new - T) / dt + dT_dt_mf
    dq_v_dt = (q_new - q_v) / dt + dq_dt_mf

    # TKE update: add MF production term
    # MF production ~ M * buoyancy / rho
    theta_v_u = virtual_temperature(theta_u, q_u)
    mf_buoyancy = (
        config.a_updraft * w_u * constants.g
        * (theta_v_u - theta_v) / jnp.clip(theta_v, 1.0, None)
    )

    tke_new = (
        tke_diffused + dt * (shear_prod + buoyancy + jnp.maximum(mf_buoyancy, 0.0))
    ) / (1.0 + dt * diss_coeff)
    tke_new = jnp.maximum(tke_new, config.tke_min)

    h_pbl = diagnose_pbl_height(T, q_v, u, v, p_full, z_full)

    output = TurbulenceOutput(
        du_dt=du_dt,
        dv_dt=dv_dt,
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx,
        ustar=ustar,
        h_pbl=h_pbl,
    )

    return output, tke_new
