"""Holtslag-Boville (1993) nonlocal K-profile boundary-layer turbulence.

Differentiable JAX port of the E3SM/CAM ``hb_diff.F90`` (``eddy_scheme =
'HB'``) plus ``pbl_utils.F90``, validated against those Fortran sources as
an oracle.  Faithful to the oracle block-by-block:

1. Surface friction velocity, Obukhov length, and kinematic surface
   buoyancy flux ``kbfs`` (``pbl_utils.calc_ustar`` / ``calc_obklen``).
2. Bulk-Richardson PBL height with the ``fac*u*^2`` mechanical term, the
   unstable surface-excess (``tlv``) correction, and the ``700*u*``
   minimum-mechanical-mixing floor (``pblintd``).
3. The nonlocal K-profile ``K = fak*u*-or-wm * pblh * vk * zh*(1-zh)^2``
   with the surface-layer / outer-layer split, the convective velocity
   scale ``wm``, the stable ``zl`` forms, and the Prandtl number
   (``austausch_pbl``).
4. The countergradient heat transport ``cgh = khfs*cgs*cpair`` in the
   unstable outer layer (``austausch_pbl``), applied as ``flux =
   -Kh*(dtheta/dz - gamma)``.
5. The free-atmosphere local-Ri mixing-length diffusivity
   ``kvf = ml2*sqrt(s2)*f(Ri)`` (``austausch_atm``), combined with the
   PBL profile via ``K = max(K_pbl, kvf)``.

The oracle's hard ``if`` switches (surface/outer layer, stable
``zl<=1``/``zl>1``, unstable/stable regime, first-crossing scan) are
replaced by smooth sigmoid blends whose sharpnesses live in
:class:`HoltslagBovilleConfig`, so the scheme is ``jax.grad``/``jit``/
``vmap`` safe and smooth everywhere while reproducing the oracle within a
stated tolerance.

Conserved heat variable — deviation from the oracle.  The E3SM ``hb_diff``
oracle diffuses the DRY STATIC ENERGY ``s = c_p T + g z`` in flux form,
which conserves the column enthalpy ``Σ ρ dz c_p T`` exactly.  THIS port
instead diffuses potential temperature θ (see
:func:`diffuse_theta_with_countergradient` and the shared
:func:`legoesm.atmosphere.physics.turbulence.vertical_diffusion.implicit_vertical_diffusion_theta`).
Diffusing θ conserves the mass-weighted column θ, ``Σ ρ dz θ``, but NOT the
column enthalpy, because the Exner function π = (p/p_ref)^κ (T = π θ) varies
with height.  The scheme is therefore faithful to the oracle's K-profile,
PBL-height, and nonlocal-countergradient STRUCTURE, but deviates in the
conserved heat variable (an accepted approximation; switching the diffused
variable to DSE is a validated follow-up).

References
----------
- Holtslag, A. A. M., & Boville, B. A. (1993). Local versus nonlocal
  boundary-layer diffusion in a global climate model. J. Climate, 6,
  1825-1842.
- E3SM ``components/eam/src/physics/cam/hb_diff.F90``, ``pbl_utils.F90``.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import exner_function, virtual_temperature
from legoesm.atmosphere.physics.turbulence.config import HoltslagBovilleConfig
from legoesm.atmosphere.physics.turbulence.pbl_height import (
    first_crossing_height,
)
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    latent_enthalpy_correction,
    surface_moisture_flux,
    compute_surface_fluxes,
    surface_fluxes_at_lowest_level,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    diagnostic_heat_flux_full,
    implicit_vertical_diffusion,
)

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "Holtslag-Boville (1993) nonlocal K-profile PBL turbulence: a "
        "bulk-Richardson PBL height sets a nonlocal eddy-diffusivity profile "
        "with a countergradient heat-transport term, blended with a "
        "free-atmosphere local-Ri diffusivity; differentiable E3SM-HB port."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m",
    },
    "sign_convention": (
        "Down-gradient nonlocal K-profile (Km, Kh >= 0) plus a positive "
        "countergradient gamma that drives an upward theta flux (warming the "
        "layers below the flux convergence). The column budget is OPEN: the "
        "surface flux (shflx > 0 upward, lhflx > 0 upward/moistening) is the "
        "bottom boundary condition, top is zero-flux; z increases upward. "
        "Heat is diffused in theta-space, so column enthalpy is NOT conserved "
        "(mass-weighted theta is) -- an accepted deviation from the DSE oracle."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Holtslag & Boville (1993), J. Climate 6, 1825-1842; "
        "E3SM/CAM hb_diff.F90 + pbl_utils.F90 oracle"
    ),
    "idealized_test": (
        "no surface flux + neutral column -> zero countergradient, "
        "K -> free-atmosphere floor, near-zero interior tendency; unstable "
        "surface buoyancy flux -> deeper h_pbl and positive gamma; oracle "
        "parity vs E3SM hb_diff within a stated tolerance."
    ),
}

_ONET = 1.0 / 3.0  # 1/3 power in the MO gradient expressions (oracle ``onet``)


# Power floor for the gradient only: the cube root / square root have an
# INFINITE derivative at 0, so we floor the argument INSIDE the JVP while the
# forward value stays the exact oracle expression ``max(x, 0)**p``.  This is
# the key to (a) oracle fidelity -- the previous softplus floor shifted the
# value by ~0.69 at the floor, inflating neutral MO factors 4-8% (codex
# iter-1 finding 1) -- and (b) finite gradients in stable columns where the
# cube-root argument is exactly 0 (codex iter-1 finding 3).
_GRAD_FLOOR = 1.0e-6


@jax.custom_jvp
def _pos_cbrt(x: jax.Array) -> jax.Array:
    """``max(x, 0)**(1/3)`` with a finite gradient (floored only in the JVP).

    Forward value is the EXACT oracle expression (so a neutral argument of 1
    returns exactly 1, and ``wstar`` is exactly 0 for ``kbfs<=0``).  The JVP
    floors the argument so the ``(1/3)x**(-2/3)`` slope never blows up.
    """
    return jnp.maximum(x, 0.0) ** _ONET


@_pos_cbrt.defjvp
def _pos_cbrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = jnp.maximum(x, 0.0) ** _ONET
    # Derivative of max(x,0)**(1/3): exactly 0 for x<0 (the forward is flat),
    # and (1/3)x**(-2/3) for x>0 with x floored to _GRAD_FLOOR so the slope
    # near 0 stays finite (codex iter-2 finding 4: don't propagate a positive
    # slope through the x<0 flat region).
    xf = jnp.maximum(x, _GRAD_FLOOR)
    slope = jnp.where(x > 0.0, _ONET * xf ** (_ONET - 1.0), 0.0)
    return y, slope * dx


@jax.custom_jvp
def _pos_sqrt(x: jax.Array) -> jax.Array:
    """``sqrt(max(x, 0))`` with a finite gradient (floored only in the JVP)."""
    return jnp.sqrt(jnp.maximum(x, 0.0))


@_pos_sqrt.defjvp
def _pos_sqrt_jvp(primals, tangents):
    (x,), (dx,) = primals, tangents
    y = jnp.sqrt(jnp.maximum(x, 0.0))
    xf = jnp.maximum(x, _GRAD_FLOOR)
    slope = jnp.where(x > 0.0, 0.5 / jnp.sqrt(xf), 0.0)
    return y, slope * dx


def _safe_cbrt(x: jax.Array) -> jax.Array:
    """Cube root with exact forward value ``max(x,0)**(1/3)``, finite grad."""
    return _pos_cbrt(x)


def _safe_sqrt(x: jax.Array) -> jax.Array:
    """Square root with exact forward value ``sqrt(max(x,0))``, finite grad."""
    return _pos_sqrt(x)


def holtslag_boville_turbulence(
    u: jax.Array,
    v: jax.Array,
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    z_full: jax.Array,
    z_half: jax.Array,
    T_sfc: jax.Array,
    q_sfc: jax.Array,
    rho: jax.Array,
    dt: float,
    config: HoltslagBovilleConfig,
) -> TurbulenceOutput:
    """Holtslag-Boville nonlocal K-profile turbulence (faithful to E3SM HB).

    Index convention: level 0 is the model top, level ``nlev-1`` is the
    surface-adjacent layer (matches CAM ``k=1..pver``).  Diffusivities
    ``K_half[k]`` live on the interface between full levels ``k`` and
    ``k+1`` (``ncol, nlev-1``); ``K_full`` is interpolated back for
    diagnostics.

    Parameters
    ----------
    u, v : jax.Array
        Wind components at full levels [m/s], shape (ncol, nlev).
    T : jax.Array
        Temperature at full levels [K], shape (ncol, nlev).
    q_v : jax.Array
        Water-vapour mixing ratio [kg/kg], shape (ncol, nlev).
    p_full, p_half : jax.Array
        Full/half-level pressure [Pa], shapes (ncol, nlev) / (ncol, nlev+1).
    z_full, z_half : jax.Array
        Full/half-level height above surface [m].
    T_sfc : jax.Array
        Surface temperature [K], shape (ncol,).
    q_sfc : jax.Array
        Surface saturation mixing ratio [kg/kg], shape (ncol,).
    rho : jax.Array
        Air density at full levels [kg/m^3], shape (ncol, nlev).
    dt : float
        Time step [s].
    config : HoltslagBovilleConfig

    Returns
    -------
    TurbulenceOutput
    """
    ncol, nlev = T.shape
    g = constants.g
    vk = constants.kappa_vk
    cpair = constants.c_pd
    # zvir = R_v/R_d - 1 = 1/epsilon - 1 (oracle ``zvir`` in calc_obklen).
    zvir = 1.0 / constants.epsilon - 1.0

    # NOTE: the derived oracle parameters (binm/binh = beta{m,h}*sffrac, the
    # countergradient prefactors ccon=fak*sffrac*vk and fakn, the stable beta
    # ``betas`` and the bulk-Ri critical number ``ricr``) are recomputed from
    # ``config`` inside :func:`hb_diffusivities` (lines ~309-318), where they
    # are actually consumed: the nonlocal K-profile, the countergradient term
    # ``gamma_theta_half`` (applied in :func:`diffuse_theta_with_countergradient`)
    # and the bulk-Richardson PBL height (:func:`_pbl_height` via ``ricr``).
    # They are NOT needed in this driver, which only assembles inputs and runs
    # the implicit diffusion, so they are not duplicated here.

    # ----- Virtual potential temperature theta_v (oracle thv) -----
    # Exner Pi = (p/p_ref)^kappa via the canonical helper (identical 1 Pa
    # pressure floor); potential temperature theta = T/Pi.
    exner = exner_function(p_full)
    theta = T / jnp.clip(exner, 1.0e-6, None)
    theta_v = virtual_temperature(theta, q_v)  # thv on theta (oracle uses thv=virtem(th,q))

    thv_bot = theta_v[:, -1]            # (ncol,)  bottom full level (CAM k=pver)
    th_bot = theta[:, -1]              # potential temp at bottom

    # ----- Half-level shear, N2, Ri for the free-atmosphere K -----
    # Interface k between full levels k and k+1 (downward), k=0..nlev-2.
    dz_half = jnp.clip(jnp.abs(z_full[:, :-1] - z_full[:, 1:]), 1.0, None)
    dvdz2 = (u[:, :-1] - u[:, 1:]) ** 2 + (v[:, :-1] - v[:, 1:]) ** 2
    dvdz2 = jnp.maximum(dvdz2, 1.0e-36)
    s2 = dvdz2 / dz_half ** 2
    # n2 = g*2*(thv_k - thv_{k+1}) / ((thv_k+thv_{k+1})*dz)  (oracle trbintd)
    thv_sum = theta_v[:, :-1] + theta_v[:, 1:]
    n2 = g * 2.0 * (theta_v[:, :-1] - theta_v[:, 1:]) / (
        jnp.clip(thv_sum, 1.0, None) * dz_half
    )
    Ri = n2 / s2  # (ncol, nlev-1)

    # ----- Surface fluxes, ustar, kinematic buoyancy flux, Obukhov -----
    tau_x, tau_y, shflx, lhflx, ustar_raw = surface_fluxes_at_lowest_level(
        u[:, -1], v[:, -1], T[:, -1], q_v[:, -1],
        T_sfc, q_sfc, rho[:, -1], config.surface, z_full[:, -1] - z_half[:, -1])
    ustar = jnp.maximum(ustar_raw, config.ustar_min)

    # rrho = 1/density at the bottom level.  Use the ORACLE dry-air form
    # rrho = R_d*T_bot/p_bot (pbl_utils.calc_ustar) rather than 1/rho_moist,
    # so khfs/kqfs/kbfs/obklen match the Fortran exactly regardless of which
    # density the bridge supplies (codex iter-1 finding 8).
    rrho = constants.R_d * T[:, -1] / jnp.clip(p_full[:, -1], 1.0, None)
    # Kinematic surface fluxes (oracle calc_obklen).
    khfs = shflx * rrho / cpair                    # [m K/s]
    kqfs = surface_moisture_flux(config.surface, lhflx, T_sfc) * rrho   # qflx*rrho
    kbfs = khfs + zvir * th_bot * kqfs             # surface buoyancy flux [m^2/s^3]
    # Obukhov length: L = -thvs*u*^3 / (g*vk*(kbfs + sign(1e-10,kbfs))).
    kbfs_signed = kbfs + jnp.sign(kbfs) * 1.0e-10 + (kbfs == 0) * 1.0e-10
    obklen = -thv_bot * ustar ** 3 / (g * vk * kbfs_signed)

    # ----- HB diffusivities + countergradient (oracle pblintd + austausch_*)
    Km_half, Kh_half, gamma_theta_half, h_pbl, wstar, wm = hb_diffusivities(
        u, v, theta_v, z_full, z_half, p_full, s2, Ri,
        ustar, khfs, kbfs, obklen, config,
    )

    # Interpolate K to full levels for diagnostics.
    Km_full = _half_to_full(Km_half)
    Kh_full = _half_to_full(Kh_half)

    # ----- Apply implicit vertical diffusion -----
    dz_layer = jnp.clip(jnp.abs(z_half[:, :-1] - z_half[:, 1:]), 1.0, None)

    sflx_u = tau_x
    sflx_v = tau_y
    sflx_q = surface_moisture_flux(config.surface, lhflx, T_sfc)   # [kg/m^2/s]
    # Heat BC carries the latent enthalpy correction (water at L(T) vs L_v).
    sflx_T = (shflx + latent_enthalpy_correction(lhflx, sflx_q)) / cpair   # [K kg/m^2/s]

    u_new = implicit_vertical_diffusion(u, Km_half, rho, dz_layer, dz_half, dt, sflx_u)
    v_new = implicit_vertical_diffusion(v, Km_half, rho, dz_layer, dz_half, dt, sflx_v)
    T_new = diffuse_theta_with_countergradient(
        T, Kh_half, rho, dz_layer, dz_half, p_full, dt, sflx_T,
        gamma_theta_half,
    )
    q_new = implicit_vertical_diffusion(q_v, Kh_half, rho, dz_layer, dz_half, dt, sflx_q)

    # Q1 diagnostic (LES-suite): NONLOCAL heat flux F = -Kh·(∂θ/∂z − γ) with the
    # scheme's own countergradient γ (gamma_theta_half), θ [=T/Π] the dry potential
    # temperature.  Pure diagnostic — does NOT feed tendencies.
    wtheta_flux = diagnostic_heat_flux_full(
        theta, dz_half, Kh_half, gamma_theta_half,
    )

    return TurbulenceOutput(
        du_dt=(u_new - u) / dt,
        dv_dt=(v_new - v) / dt,
        dT_dt=(T_new - T) / dt,
        dq_v_dt=(q_new - q_v) / dt,
        Km=Km_full,
        Kh=Kh_full,
        shflx=shflx,
        lhflx=lhflx, evap_sfc=sflx_q,
        ustar=ustar,
        h_pbl=h_pbl,
        wtheta_flux=wtheta_flux,
    )


def _half_to_full(K_half: jax.Array) -> jax.Array:
    """Interpolate interface diffusivities to full levels (diagnostic)."""
    interior = 0.5 * (K_half[:, :-1] + K_half[:, 1:])
    return jnp.concatenate(
        [K_half[:, :1], interior, K_half[:, -1:]], axis=1,
    )


def hb_diffusivities(
    u, v, theta_v, z_full, z_half, p_full, s2, Ri,
    ustar, khfs, kbfs, obklen, config: HoltslagBovilleConfig,
):
    """HB eddy diffusivities + countergradient (pure, prescribed-flux core).

    This is the exact algorithm of the oracle ``pblintd`` + ``austausch_pbl``
    + ``austausch_atm``, driven by PRESCRIBED kinematic surface fluxes so the
    oracle harness can compare against it apples-to-apples (the bulk-flux
    closure that produces ``khfs``/``kbfs``/``ustar``/``obklen`` is the
    caller's responsibility).

    Parameters
    ----------
    u, v : (ncol, nlev) winds [m/s], top-first.
    theta_v : (ncol, nlev) virtual potential temperature [K].
    z_full : (ncol, nlev) full-level height above surface [m].
    s2 : (ncol, nlev-1) interface shear squared [1/s^2].
    Ri : (ncol, nlev-1) interface gradient Richardson number.
    ustar : (ncol,) friction velocity [m/s] (already floored).
    khfs : (ncol,) kinematic surface heat flux [m K/s].
    kbfs : (ncol,) surface kinematic buoyancy flux [m^2/s^3].
    obklen : (ncol,) Obukhov length [m].
    config : HoltslagBovilleConfig

    Returns
    -------
    Km_half, Kh_half : (ncol, nlev-1) interface diffusivities [m^2/s].
    gamma_theta_half : (ncol, nlev-1) countergradient theta gradient [K/m].
    h_pbl : (ncol,) PBL height [m].
    wstar : (ncol,) convective velocity scale [m/s].
    wm : (ncol,) turbulent velocity scale for momentum [m/s].
    """
    g = constants.g
    vk = constants.kappa_vk
    sffrac = config.sffrac
    betam = config.betam
    betah = config.betah
    betas = config.betas
    binm = betam * sffrac
    binh = betah * sffrac
    fak = config.fak
    fakn = config.fakn
    ccon = fak * sffrac * vk
    ricr = config.Ri_crit

    thv_bot = theta_v[:, -1]

    # Smooth unstable indicator (oracle: unstbl = kbfs>0, a hard `>`).  Bias
    # the sigmoid by a tiny positive threshold so EXACTLY neutral kbfs=0 maps
    # to the STABLE branch (unstbl~0), matching the oracle's strict `> 0`
    # rather than the symmetric sigmoid's 0.5 (codex iter-1 finding 2).
    unstbl = jax.nn.sigmoid(
        config.unstable_blend_sharpness * (kbfs - config.unstable_kbfs_threshold)
    )

    # ----- PBL height (oracle pblintd), smooth -----
    h_pbl, wstar, phiminv, phihinv, wm = _pbl_height(
        u, v, theta_v, z_full, z_half, p_full, ustar, obklen, kbfs, unstbl,
        thv_bot, g, vk, binm, binh, fak, ricr, config,
    )

    # ----- Eddy K profile inside the PBL (oracle austausch_pbl) -----
    zmzp = 0.5 * (z_full[:, :-1] + z_full[:, 1:])  # (ncol, nlev-1)
    zh = zmzp / jnp.clip(h_pbl[:, None], 1.0, None)
    zl = zmzp / obklen[:, None]
    zzh = zh * jnp.maximum(0.0, 1.0 - zh) ** 2

    fak1 = (ustar * h_pbl * vk)[:, None]
    fak2 = (wm * h_pbl * vk)[:, None]
    fak3 = (fakn * wstar / jnp.clip(wm, 1.0e-6, None))[:, None]

    # --- unstable surface layer (zh<sffrac) ---
    # MO gradient factors: exact oracle forms (1-betam*zl)^(1/3),
    # sqrt(1-betah*zl). _safe_cbrt/_safe_sqrt give the EXACT forward value
    # max(arg,0)**p (so a neutral arg=1 returns exactly 1) with a finite
    # gradient -- no softplus value shift (codex iter-1 finding 1).
    term = _safe_cbrt(1.0 - betam * zl)
    pblk_sfc_u = fak1 * zzh * term
    pr_denom = _safe_sqrt(1.0 - betah * zl)
    pr_sfc_u = term / jnp.clip(pr_denom, config.arg_floor, None)

    # --- unstable outer layer (zh>=sffrac) ---
    pblk_out_u = fak2 * zzh
    pr_out_u = (phiminv / jnp.clip(phihinv, 1.0e-6, None))[:, None] + (
        ccon * fak3 / fak
    )
    cgs_outer = fak3 / jnp.clip((h_pbl * wm)[:, None], 1.0e-12, None)

    w_outer = jax.nn.sigmoid(config.sfc_blend_sharpness * (zh - sffrac))
    pblk_u = (1.0 - w_outer) * pblk_sfc_u + w_outer * pblk_out_u
    pr_u = (1.0 - w_outer) * pr_sfc_u + w_outer * pr_out_u
    cgs_u = w_outer * cgs_outer

    # --- stable (kbfs<=0) ---
    # The stable branch is evaluated for ALL columns (then blended out by
    # 1-unstbl in unstable ones).  For unstable columns L<0 -> zl<0, so the
    # raw denominators ``1+betas*zl`` / ``betas+zl`` can hit zero or go
    # negative, producing inf that contaminates the VJP via 0*inf -> NaN.
    # Floor the denominators to a positive value so the inactive branch is
    # finite (codex iter-1 finding 7).  Oracle behaviour for the ACTIVE
    # stable branch (zl>=0) is unchanged since the floors only bite at zl<0.
    den_lo = jnp.maximum(1.0 + betas * zl, config.arg_floor)
    den_hi = jnp.maximum(betas + zl, config.arg_floor)
    pblk_st_lo = fak1 * zzh / den_lo
    pblk_st_hi = fak1 * zzh / den_hi
    w_zl = jax.nn.sigmoid(config.stable_blend_sharpness * (zl - 1.0))
    pblk_s = (1.0 - w_zl) * pblk_st_lo + w_zl * pblk_st_hi
    pr_s = jnp.ones_like(pblk_s)

    w_uns = unstbl[:, None]
    pblk = w_uns * pblk_u + (1.0 - w_uns) * pblk_s
    pr = w_uns * pr_u + (1.0 - w_uns) * pr_s
    cgs = w_uns * cgs_u

    # Restrict the PBL profile to interfaces inside the PBL.  Oracle gates
    # pblk by ``z(k) < pblh`` and cgs(i,k) is set whenever ``z(k) < pblh``
    # (the LOWER full level of the interface, closer to the surface), NOT by
    # the interface midpoint zmzp (codex iter-1 finding 4).  For our
    # interface Km_half[j] (between full levels j and j+1, top-first), the
    # lower full level is z_full[:, 1:].
    z_lower = z_full[:, 1:]                       # (ncol, nlev-1)
    zh_lower = z_lower / jnp.clip(h_pbl[:, None], 1.0, None)
    in_pbl = jax.nn.sigmoid(config.sfc_blend_sharpness * (1.0 - zh))
    # Sharper gate for cgs (oracle ``z(k) < pblh`` is a hard step); a
    # dedicated higher sharpness closes the ~2.5% residual at the PBL-top
    # interface (codex iter-2 finding 3) while staying differentiable.
    in_pbl_cg = jax.nn.sigmoid(config.cgs_gate_sharpness * (1.0 - zh_lower))
    pblk = pblk * in_pbl
    cgs = cgs * in_pbl_cg

    # ----- Free-atmosphere local-Ri diffusivity (oracle austausch_atm) -----
    # _safe_sqrt floors the argument so the sqrt gradient stays finite where
    # s2 -> 0 (zero shear) or where (1 - 18*Ri) -> 0 (strongly unstable).
    kvn = (config.ml_free ** 2) * _safe_sqrt(s2)
    f_unstable = _safe_sqrt(
        1.0 - config.free_ri_unstable_coeff * Ri
    )
    f_stable = 1.0 / (
        1.0 + config.free_ri_stable_c1 * Ri
        * (1.0 + config.free_ri_stable_c2 * Ri)
    )
    w_ri = jax.nn.sigmoid(config.stable_blend_sharpness * Ri)
    fofri = (1.0 - w_ri) * f_unstable + w_ri * f_stable
    kvf = jnp.maximum(config.kvf_min, kvn * fofri)

    # ----- Combine: kvm = max(pblk, kvf); kvh = max(pblk/pr, kvf) -----
    Km_half = jnp.maximum(pblk, kvf)
    Kh_half = jnp.maximum(pblk / jnp.clip(pr, 1.0e-6, None), kvf)

    # Countergradient theta gradient [K/m]: gamma = cgh/cpair = khfs*cgs.
    gamma_theta_half = khfs[:, None] * cgs

    return Km_half, Kh_half, gamma_theta_half, h_pbl, wstar, wm


def _pbl_height(
    u, v, theta_v, z_full, z_half, p_full, ustar, obklen, kbfs, unstbl,
    thv_bot, g, vk, binm, binh, fak, ricr, config: HoltslagBovilleConfig,
):
    """Smooth bulk-Richardson PBL height (oracle pblintd).

    Returns ``(h_pbl, wstar, phiminv, phihinv, wm)``.

    Faithful to the oracle's two-pass scan:
      pass 1: rino = g*(thv_k - thv_bot)*(z_k - z_bot)/(thv_bot*vvk),
              vvk = (u_k-u_bot)^2 + (v_k-v_bot)^2 + fac*u*^2.
      unstable surface-excess: tlv = thv_bot + kbfs*fak/(u*·phiminv),
              pass 2 uses tlv in place of thv_bot.
      pblh = max(interp at ricr crossing, 700*u*).
    The hard first-crossing scan + linear interpolation is replaced by a
    smooth lowest-crossing interpolation (sigmoid-weighted), differentiable
    everywhere.  The search is limited to the region below ``pblmaxp`` and,
    when no crossing is found there (well-mixed neutral / fully-mixed
    convective columns), the PBL height defaults to the top of that search
    region (oracle ``pblh = z(pverp-npbl)``).
    """
    ncol, nlev = z_full.shape
    z_bot = z_full[:, -1:]
    u_bot = u[:, -1:]
    v_bot = v[:, -1:]
    vvk = (u - u_bot) ** 2 + (v - v_bot) ** 2 + config.pblh_ustar_fac * ustar[:, None] ** 2
    vvk = jnp.maximum(vvk, 1.0e-36)
    dz_sfc = z_full - z_bot
    thv_bot_c = thv_bot[:, None]

    # Search region: levels with p >= pblmaxp (oracle npbl limit -- an
    # ABSOLUTE pressure threshold, ``pref_mid(k) >= pblmaxp`` with
    # pblmaxp = 4e4 Pa = 400 hPa, NOT relative to the surface).
    # search_ok ~ 1 inside the allowed PBL search region, 0 above it.
    # Top-first ordering: search_ok rises 0 -> 1 going DOWN (top to surface).
    p_thresh = config.pblmaxp
    # Pressure scale for the smooth region edge: a few hPa is sharp enough.
    search_ok = jax.nn.sigmoid((p_full - p_thresh) / 1.0e3)  # (ncol, nlev)
    # Fallback height = height at the TOP of the search region = the highest
    # in-region level (where search_ok transitions 0 -> 1, top-first).  Weight
    # z_full by the downward jump in search_ok, which peaks at that edge.
    d_search = search_ok - jnp.concatenate(
        [jnp.zeros((ncol, 1), search_ok.dtype), search_ok[:, :-1]], axis=1,
    )
    edge_w = jnp.maximum(d_search, 0.0) + 1.0e-20
    z_top_search = jnp.sum(edge_w * z_full, axis=1) / jnp.sum(edge_w, axis=1)

    # Pass 1 bulk Ri (rino) relative to thv_bot.  The oracle first-crossing
    # scan + linear interpolation is the SHARED differentiable
    # ``first_crossing_height`` kernel (pbl_height.py, also used by YSU),
    # with the HB-specific ``search_ok`` region mask and ``z_top_search``
    # fallback (oracle ``pblh = z(pverp-npbl)``).
    rino1 = g * (theta_v - thv_bot_c) * dz_sfc / (jnp.clip(thv_bot_c, 1.0, None) * vvk)

    h1 = first_crossing_height(
        rino1, z_full, ricr, config.pbl_crossing_sharpness,
        search_ok=search_ok, z_top=z_top_search,
    )

    # Unstable surface-excess correction (oracle): only where kbfs>0.
    # phiminv = (1 - binm*pblh/L)^(1/3); arg>1 for unstable (L<0).
    phiminv1 = _safe_cbrt(1.0 - binm * h1 / obklen)
    tlv = thv_bot + kbfs * fak / jnp.clip(ustar * phiminv1, 1.0e-6, None)
    rino2 = g * (theta_v - tlv[:, None]) * dz_sfc / (jnp.clip(thv_bot_c, 1.0, None) * vvk)
    h2 = first_crossing_height(
        rino2, z_full, ricr, config.pbl_crossing_sharpness,
        search_ok=search_ok, z_top=z_top_search,
    )

    # Use pass-2 height in unstable columns, pass-1 otherwise.
    h_pbl = unstbl * h2 + (1.0 - unstbl) * h1

    # Mechanical mixing floor: pblh >= 700*u* (oracle).
    h_pbl = jnp.maximum(h_pbl, config.pblh_mech_coeff * ustar)

    # Lowest-layer ventilation floor (oracle): pblh >= zi(pver) + 50, applied
    # unconditionally (the oracle's cloud test ``cldn(:,pver) >= 0`` is always
    # true).  ``zi(pver)`` is the top interface of the lowest model layer =
    # z_half[:, -2] in our top-first half-level array (z_half[:, -1] = surface).
    h_pbl = jnp.maximum(h_pbl, z_half[:, -2] + config.cloud_pbl_floor_m)

    # Velocity scales (oracle austausch_pbl / pblintd), evaluated at pblh.
    # _safe_cbrt: the cube root of max(0,kbfs) has an infinite derivative at
    # kbfs=0 -> NaN grad in stable columns; floor the argument first.
    wstar = _safe_cbrt(
        jnp.maximum(0.0, kbfs) * g * h_pbl / jnp.clip(thv_bot, 1.0, None)
    )
    phiminv = _safe_cbrt(1.0 - binm * h_pbl / obklen)
    phihinv = _safe_sqrt(1.0 - binh * h_pbl / obklen)
    wm = ustar * phiminv
    return h_pbl, wstar, phiminv, phihinv, wm


def diffuse_theta_with_countergradient(
    T, Kh_half, rho, dz, dz_half, p_full, dt, surface_flux_T, gamma_theta_half,
):
    """Implicit theta diffusion with the HB nonlocal countergradient.

    The countergradient enters as ``flux = -rho*Kh*(dtheta/dz - gamma)``.
    The ``-Kh*gamma`` part is an explicit (known) flux added to the column;
    its divergence is an explicit source applied before the implicit solve,
    exactly as CAM treats ``cgh`` as a known countergradient flux.

    Implementation: convert T->theta, add the explicit countergradient flux
    divergence as a theta source (forward), then run the standard implicit
    theta diffusion of the local gradient with the surface flux, and convert
    back.  This keeps the implicit part identical to the no-cg path while
    adding the nonlocal term consistently with the oracle.

    Conserved-variable caveat.  The E3SM oracle diffuses dry static energy
    s = c_p T + g z (enthalpy-conserving in flux form); this routine diffuses
    θ instead, so it reproduces the oracle's K-profile and countergradient
    structure but conserves mass-weighted θ (``Σ ρ dz θ``) rather than column
    enthalpy (``Σ ρ dz c_p T``), since the Exner π = (p/p_ref)^κ varies with
    height.  An energy-conserving variant would diffuse s; this is an accepted
    approximation and a validated follow-up.

    Surface-Exner proxy.  ``exner_sfc`` below uses the lowest FULL-level Exner
    (exner[:, -1]) as a surface-Exner stand-in.  Since p_low < p_surface this
    proxy is biased low, so the injected surface θ-flux F_θ_sfc =
    F_T_sfc / exner_sfc is biased slightly HIGH; threading a true p_surface
    and using (p_sfc/p_ref)^κ would remove the bias.
    """
    # Exner Pi = (p/p_ref)^kappa via the canonical helper (same 1 Pa floor).
    exner = exner_function(p_full)
    exner_safe = jnp.clip(exner, 1.0e-6, None)
    theta = T / exner_safe
    exner_sfc = exner_safe[:, -1]

    # Explicit countergradient flux on interfaces: F_cg = rho_half*Kh*gamma.
    # Positive gamma (warm, unstable BL) drives an UPWARD theta flux, which
    # warms the layers below the flux convergence -> mixes heat up against
    # the local gradient (the defining HB nonlocal effect).
    rho_half = 0.5 * (rho[:, :-1] + rho[:, 1:])
    F_cg = rho_half * Kh_half * gamma_theta_half  # (ncol, nlev-1), upward +

    # Flux divergence -> theta source: d(theta)/dt = -(1/rho) dF/dz.
    # Interface k between full levels k and k+1. For full level k, the net
    # flux convergence is (F_top - F_bot)/(rho*dz). Interior levels see two
    # interfaces; top/bottom see one (zero-flux top, surface handled by the
    # surface flux BC, so countergradient flux at the surface interface is 0).
    ncol, nlev = T.shape
    # F at interface above level k is F_cg[k-1]; below level k is F_cg[k].
    F_above = jnp.concatenate([jnp.zeros((ncol, 1), F_cg.dtype), F_cg], axis=1)  # (ncol,nlev)
    F_below = jnp.concatenate([F_cg, jnp.zeros((ncol, 1), F_cg.dtype)], axis=1)  # (ncol,nlev)
    # d(theta)/dt = -(F_above - F_below)/(rho*dz)  [upward-positive flux,
    # level 0 = top]. Flux convergence warms the layer where F decreases
    # upward.
    dtheta_cg = -(F_above - F_below) / (jnp.clip(rho, 1.0e-6, None) * dz)
    theta_star = theta + dt * dtheta_cg

    # Implicit local-gradient diffusion with the surface theta flux.
    surface_flux_theta = surface_flux_T / exner_sfc
    theta_new = implicit_vertical_diffusion(
        theta_star, Kh_half, rho, dz, dz_half, dt, surface_flux_theta,
    )
    return theta_new * exner
