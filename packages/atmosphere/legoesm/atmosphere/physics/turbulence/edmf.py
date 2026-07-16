"""EDMF (Eddy-Diffusivity Mass-Flux) turbulence scheme.

Unified turbulence-convection framework that combines an eddy-diffusivity
(ED) component based on prognostic TKE with a mass-flux (MF) updraft
model. The ED part handles small-scale mixing while the MF part
represents coherent updraft transport in the convective boundary layer.

Faithfulness to SST07 (dry-CBL EDMF)
------------------------------------
This is an EDMF-STRUCTURED closure in the spirit of Siebesma, Soares & Teixeira (2007,
"SST07"), NOT a verbatim SST07 or a Tan et al. (2018) plume: it is a STEADY diagnostic
single-plume marching + a prognostic-TKE ED part, with SST07-inspired forms and several
simplified coefficients.
FAITHFUL to the EDMF framework (the DEFINING decomposition):
  * the total turbulent flux is split ``w'φ' = −K ∂φ/∂z + M_kin (φ_u − φ̄)`` (kinematic mass
    flux ``M_kin ≈ a_u·w_u``) into a down-gradient ED part and a nonlocal MF part (SST07 Eq. 5,
    via Eq. 3-4); this code carries a DENSITY-WEIGHTED ``M = a_u·ρ·w_u = ρ·M_kin``, so the
    conservative flux-form MF tendency carries the ``1/ρ``: ``∂φ/∂t = −(1/ρ) ∂z[M (φ_u − φ̄)]``
    (dimensionally ``= −∂z[M_kin (φ_u − φ̄)]``);
  * the updraft entrains environment air, ``d(φ_u)/dz = −ε (φ_u − φ_env)`` (SST07 Eq. 10; here
    ``φ_env`` is the grid mean ``φ̄`` — SST07's small-area (``a_u ≪ 1``) approximation
    ``φ_e ≈ φ̄``, not literal environment air; the ``a_u = 0.1`` default is this code's choice,
    not SST07's stated ~1-5% thermal area).
DEPARTURES from SST07 (physics / structural simplifications):
  * **mass flux** ``M = a_u·ρ·w_u`` is a compressible extension of SST07's Boussinesq
    ``M = a_u(w_u − w̄) ≈ a_u·w_u`` (Eq. 3-5; SST07's dry-CBL closure actually uses
    ``M = c_m·σ_w``, Eq. 21), and is then further CAPPED at ``M_max`` (see NUMERICS) — so
    effectively ``M = min(a_u·ρ·w_u, M_max)``;
  * the **plume vertical velocity** ``d(w²)/dz = 2(B − ε·w²)`` is a SIMPLIFICATION, NOT SST07
    Eq. 15 ``½(1−2β)·d(w²)/dz = B − b·ε·w²`` with ``β = 0.15``, ``b = 0.5`` — this code takes
    ``β = 0`` and ``b = 1`` (buoyancy-vs-entrainment-drag form with unit coefficients);
  * the **ED part is the ``tke`` MY-INSPIRED k-l closure** (``Km = Ck·l·√TKE`` with constant
    ``Ck = 0.1`` / ``Pr_t = 0.33`` and NO stability functions, so the diagnostic ``Km ⊥ N²``),
    NOT SST07's ``z/z*``-dependent Holtslag K-profile (Eq. 18-20);
  * **CONSTANT entrainment** ``ε = 1e-3 /m`` — SST07 Eq. 16 is ``ε ≈ c_ε[1/z + 1/(z*−z)]``
    (height-dependent, singular at both the surface and the PBL top);
  * **tuned surface initialization**: ``w_u(0) = max(w_min, 2.5·u*)`` (a FRICTION-velocity
    proxy, NOT the buoyancy convective scale ``w*``) and a FIXED ``parcel_dT = 0.5 K`` θ-excess;
    SST07 Eq. 17 sets the surface scalar excess from the surface flux divided by ``σ_w``;
  * a **single bulk steady plume** (``n_updrafts`` unused; no detrainment — ``detrainment_rate``
    unused/dead) — faithful to SST07's dry-CBL single-plume scope, but a MAJOR departure from
    Tan (2018) (prognostic plume velocity/area/thermo, updrafts+downdrafts, and prognostic
    plume/subdomain second moments — this code carries only grid-mean TKE);
  * an extra **MF→TKE buoyancy production** term (``max(mf_buoyancy, 0)`` into the TKE budget) —
    SST07 has no prognostic-TKE budget at all, and this is not Tan's energy-consistent partition.
NUMERICS (AD-safety / stability / solver structure, not SST07 physics):
  * **ED and MF are solved SEPARATELY** — ED implicitly, then MF added EXPLICITLY and capped —
    whereas SST07 solves the combined ED+MF advection-diffusion IMPLICITLY together (App. B);
  * **backward-Euler** in the w² and entrainment updates (unconditionally stable for any
    ``ε·dz``; forward Euler NaN'd at T21 where ``ε·dz ~ 3-5``);
  * **AD-safe sqrt** (floor + outer ``where`` so ``d√w²`` stays finite at ``w² ≤ 0``);
  * a **sigmoid smooth deactivation** of the updraft below ``w_updraft_min``;
  * the explicit-CFL cap ``M ≤ 0.5·ρ·dz/dt``;
  * the flux-form **height-weighted interface interpolation** (exact telescoping conservation
    on stretched grids); and the density (``ρ ≥ 0.01``) / θ_v (``≥ 1``) floors.
Non-behavioral pins: ``tests/atmosphere/hydrostatic/unit/test_edmf_faithful.py``.

References
----------
- Siebesma, A. P., Soares, P. M. M., & Teixeira, J. (2007). A combined
  eddy-diffusivity mass-flux approach for the convective boundary layer.
  J. Atmos. Sci., 64, 1230-1248.
- Tan, Z., et al. (2018). An extended eddy-diffusivity mass-flux scheme for
  unified representation of subgrid-scale turbulence and convection.
  J. Adv. Model. Earth Syst., 10, 770-800.  (Cited for CONTRAST only — this
  module is NOT a Tan plume: it has no prognostic plume velocity/area/thermo,
  no downdrafts, and no subdomain second moments.)
"""

from __future__ import annotations

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.atmosphere.physics._shared import (
    buoyancy_coefficient,
    exner_function,
    mixing_length,
    virtual_temperature,
)
from legoesm.atmosphere.physics.turbulence.config import TurbulentEDMFConfig
from legoesm.atmosphere.physics.turbulence.output import TurbulenceOutput
from legoesm.atmosphere.physics.turbulence.pbl_height import diagnose_pbl_height
from legoesm.atmosphere.physics.turbulence.surface_layer import (
    compute_surface_fluxes,
)
from legoesm.atmosphere.physics.turbulence.vertical_diffusion import (
    implicit_vertical_diffusion,
    implicit_vertical_diffusion_theta,
)

# Machine-checked scheme contract (see tests/test_physics_contracts.py).
__physics_contract__ = {
    "summary": (
        "EDMF turbulence: prognostic-TKE eddy diffusivity (ED) for local "
        "down-gradient mixing combined with a buoyant mass-flux (MF) updraft "
        "for nonlocal transport of heat and moisture in the convective PBL."
    ),
    "inputs": {
        "u": "m/s", "v": "m/s", "T": "K", "q_v": "kg/kg", "tke": "m^2/s^2",
        "p_full": "Pa", "p_half": "Pa", "z_full": "m", "z_half": "m",
        "T_sfc": "K", "q_sfc": "kg/kg", "rho": "kg/m^3", "dt": "s",
    },
    "outputs": {
        "du_dt": "m/s^2", "dv_dt": "m/s^2", "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "Km": "m^2/s", "Kh": "m^2/s", "shflx": "W/m^2", "lhflx": "W/m^2",
        "ustar": "m/s", "h_pbl": "m", "tke_new": "m^2/s^2",
    },
    "sign_convention": (
        "ED part is down-gradient (Km >= 0, Kh = Km/Pr_t >= 0); the MF part "
        "adds -(1/rho) d/dz(M*(phi_u - phi)) with density-weighted M = a_u*rho*w_u, "
        "so a warm/moist updraft transports heat "
        "and moisture upward. The column budget is OPEN: the surface flux "
        "(shflx > 0 upward, lhflx > 0 upward/moistening) is injected as the "
        "bottom boundary condition and the top is zero-flux; z increases upward."
    ),
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Siebesma, Soares & Teixeira (2007), J. Atmos. Sci. 64, 1230-1248 "
        "(EDMF framework; this is an SST07-STRUCTURED dry-CBL single-plume "
        "simplification, NOT a verbatim SST07 or a Tan et al. 2018 plume)"
    ),
    "idealized_test": (
        "no surface flux + neutral non-buoyant column -> zero scalar MF "
        "transport (phi_u -> phi_env; the anomaly vanishes, M stays nonzero) "
        "and near-zero interior tendency; Km, Kh >= 0; tke stays >= tke_min."
    ),
}


# Surface updraft velocity-scale coefficient: w_u(0) ~ 2.5*u* (a FRICTION-velocity
# proxy, NOT the buoyancy convective velocity scale w*; SST07 Eq. 17 derives the
# surface updraft properties from the surface buoyancy flux).
_EDMF_WSTAR_COEFF = 2.5


def _mass_flux_tendency(phi, phi_u, M, dz_layer, rho, z_full, z_half):
    """Flux-form vertical divergence of the mass-flux transport of ``phi``.

    ``d(phi)/dt = -(1/rho) d/dz[ M (phi_u - phi) ]`` written in FLUX FORM so the
    column mass-weighted integral telescopes to ``F_top - F_sfc`` exactly.  A
    non-flux-form centred difference at full levels does NOT telescope (its
    mass-weighted column sum leaves a spurious O(interior MF flux) heat/moisture
    source), which is the conservation defect this replaces.

    Index 0 = model top, increasing downward; z increases upward.  The
    interior interface fluxes are interpolated LINEARLY IN HEIGHT from the
    adjacent cell-centred fluxes to the actual interface height ``z_half``
    (weights from ``z_full``/``z_half``).  On a uniform grid the interface is
    midway between the flanking full levels, so this reduces exactly to the
    previous 2-point average; on a STRETCHED grid the unweighted average is
    equivalent to the old centred difference ``(F[k-1]-F[k+1])/(2·dz_layer[k])``
    whose denominator does not match the full-level spacing
    ``z_full[k-1]-z_full[k+1]`` — an O(1) local truncation error the height
    weighting removes (a linear-in-z flux now yields the exact constant
    divergence in every interior row) while PRESERVING the telescoping
    conservation property (the divisor stays the layer thickness ``dz_layer``).

    There is no mass-flux transport through the model top (``F_top = 0``); the
    lowest interface carries the plume's own bottom MF flux ``F_sfc = flux[-1] =
    M·(φ_u − φ)`` — set by the surface updraft initialization (``parcel_dT`` /
    ``w_u``), NOT the physical ``shflx`` / ``lhflx`` (those drive the ED solve).
    Retaining it keeps the surface-driven plume transport; hard-zeroing it broke
    ``test_mass_flux_active`` (iter-50) and is NOT done here.
    """
    flux = M * (phi_u - phi)                       # cell-centred updraft flux, (ncol, nlev)
    ncol = flux.shape[0]
    zero_top = jnp.zeros((ncol, 1), dtype=flux.dtype)
    # Interior interface j (between full levels j-1 and j, top-first) sits at
    # z_half[:, 1:-1]; interpolate the full-level flux linearly to that height.
    # w in (0, 1): fractional distance from the UPPER full level (j-1) down to
    # the interface, w = (z_upper - z_iface)/(z_upper - z_lower).
    z_upper = z_full[:, :-1]
    z_lower = z_full[:, 1:]
    z_iface = z_half[:, 1:-1]
    w = (z_upper - z_iface) / jnp.clip(z_upper - z_lower, 1.0, None)
    flux_int = (1.0 - w) * flux[:, :-1] + w * flux[:, 1:]  # (ncol, nlev-1)
    flux_sfc = flux[:, -1:]                          # surface-coupled MF flux
    flux_iface = jnp.concatenate([zero_top, flux_int, flux_sfc], axis=1)  # (ncol, nlev+1)
    # d(phi)/dt|_k = -(F_upper - F_lower)/(rho_k dz_k); upper iface = k, lower = k+1.
    dflux_dz = (flux_iface[:, :-1] - flux_iface[:, 1:]) / dz_layer
    return -dflux_dz / jnp.clip(rho, 0.01, None)  # coeff-ok: density floor


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
    config: TurbulentEDMFConfig,
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
    config : TurbulentEDMFConfig

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
    l_mix = mixing_length(z_full, config.l_mix_max)

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

    # ``exner_pref`` = (p_ref / p)^κ = 1/Π multiplies T to get θ (canonical
    # helper; same 1 Pa pressure floor).  Distinct from ``exner_inv`` =
    # (p / p_ref)^κ used below to invert θ back to T.  Variable shadowing was
    # a fragility hazard — keep them distinctly named.
    exner_pref = 1.0 / exner_function(p_full)
    theta_v = virtual_temperature(T, q_v) * exner_pref
    theta_v_bar = 0.5 * (theta_v[:, :-1] + theta_v[:, 1:])
    dtheta_v_dz = (theta_v[:, :-1] - theta_v[:, 1:]) / dz_half
    # N² = (g/θ_v)·∂θ_v/∂z via the shared buoyancy coefficient (clip at call site).
    N2_half = buoyancy_coefficient(jnp.clip(theta_v_bar, 1.0, None)) * dtheta_v_dz

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
    ustar = jnp.clip(ustar, 1e-4, None)  # coeff-ok: u* floor [m/s]

    # Potential temperature for updraft.
    # ``exner_inv`` = (p / p_ref)^κ = Π (canonical helper) — divides T to
    # give θ, multiplies dθ to give dT.  Distinct from ``exner_pref`` =
    # (p_ref / p)^κ above.  Keeping the two names separate avoids the
    # fragility of reassigning a single ``exner`` to its reciprocal
    # mid-function.
    exner_inv = exner_function(p_full)
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
        (_EDMF_WSTAR_COEFF * ustar).astype(_dtype),
    )
    theta_u_init = (theta[:, -1] + config.parcel_dT).astype(_dtype)
    q_u_init = q_v[:, -1].astype(_dtype)  # same moisture

    # Scan from surface upward (reverse level index)
    # Levels are top-down, so we scan from nlev-1 to 0
    z_rev = z_full[:, ::-1]  # (ncol, nlev), surface first
    theta_rev = theta[:, ::-1]
    theta_v_rev = theta_v[:, ::-1]
    q_v_rev = q_v[:, ::-1]

    # Layer spacing from surface upward
    dz_upward = jnp.abs(jnp.diff(z_rev, axis=1))  # (ncol, nlev-1)

    def updraft_step(carry, inputs):
        w_u, theta_u, q_u = carry
        dz_k, theta_env, theta_v_env, q_env = inputs

        # Updraft virtual potential temperature
        theta_v_u = virtual_temperature(theta_u, q_u)

        # Buoyancy [m/s²]
        buoy = constants.g * (theta_v_u - theta_v_env) / jnp.clip(theta_v_env, 1.0, None)

        # Plume vertical velocity — SST07-STRUCTURED simplification (NOT SST07
        # Eq. 15's ½(1−2β)·d(w²)/dz = B − b·ε·w² with β=0.15, b=0.5; this code
        # takes β=0, b=1):
        #
        #     w · dw/dz = B − ε · w²       ⇔     d(w²)/dz = 2(B − ε·w²).
        #
        # **Backward-Euler in the linear damping term** so the update is
        # unconditionally stable for any ``ε·dz``:
        #     w²_new = (w²_old + 2·B·dz) / (1 + 2·ε·dz),    clamped ≥ 0.
        # Forward Euler ``w² + 2(B − ε·w²)·dz`` has the amplification
        # factor ``(1 − 2·ε·dz)``: absolutely stable only for ``ε·dz < 1``,
        # and the w² coefficient goes NEGATIVE (sign flip) for ``ε·dz > 0.5``.
        # At T21 with 8 sigma levels ``dz`` can reach ~3-5 km and the default
        # ``ε = 1e-3 /m`` gives ``ε·dz ~ 3-5``, which flips the sign and
        # amplifies it each layer — the original sample-46 NaN crash.  Backward
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
    # The MF surface flux is the plume's OWN bottom flux M·(φ_u − φ), set by the
    # surface updraft initialization (parcel_dT / w_u).  It is not a direct
    # function of the shflx / lhflx OUTPUTS (those drive the ED implicit solve),
    # though the surface ``w_u(0) = max(w_min, 2.5·u*)`` (with a floored u*)
    # shares the same MOST surface forcing when the 2.5·u* branch is active.  M is
    # NOT hard-zeroed at the boundaries: it EVOLVES aloft — decaying as w_u falls
    # through the sigmoid gate where buoyancy is weak, though positive buoyancy
    # can accelerate the plume — and the flux-form divergence (see
    # _mass_flux_tendency) telescopes EXACTLY to the boundary MF fluxes.
    # The iter-50 attempt to hard-zero M at the boundaries broke
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

    # MF tendencies: flux-form vertical divergence of M·(phi_u − phi_env), so
    # the column mass-weighted integral telescopes to the boundary fluxes
    # (conservative — a centred full-level difference leaks a spurious column
    # source).  See module-level ``_mass_flux_tendency``.
    dtheta_dt_mf = _mass_flux_tendency(theta, theta_u, M, dz_layer, rho, z_full, z_half)
    dT_dt_mf = dtheta_dt_mf * exner_inv
    dq_dt_mf = _mass_flux_tendency(q_v, q_u, M, dz_layer, rho, z_full, z_half)

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
