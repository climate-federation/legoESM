"""Mass-flux convection schemes (Arakawa-Wu and simplified EDMF).

This module provides two prognostic mass-flux convection
parameterizations that share most of their numerics:

1. **Prognostic Mass-Flux** (``mass_flux_convection``): Arakawa-Wu type
   scheme with a single column-mean prognostic variable ``M_c``. The
   vertical profile of the mass flux is a fixed sinusoidal shape;
   ``M_c`` relaxes toward an equilibrium value diagnosed from CAPE.

2. **Simplified EDMF** (``edmf_convection``): single-updraft mass-flux
   scheme inspired by Siebesma et al. (2007). The prognostic variable
   is the updraft area fraction ``a_u``; the mass-flux profile
   ``M_u(z) = ρ · a_u · w_u(z)`` is built from a buoyancy-derived
   updraft velocity. This is a reduced-complexity surrogate, not a
   full EDMF (single plume, no downdrafts, no stochastic triggering,
   no PDF closure).

Both schemes share the same downstream kernel: given a vertical
mass-flux profile ``M(z)``, an entraining/diluting updraft ``(T_u,
q_u)``, and grid geometry, they apply the same compensating
subsidence + detrainment tendencies and diagnose surface
precipitation from the column integral of detrained condensate.
The kernel is factored into ``apply_mass_flux_kernel`` so the two
schemes differ only in (a) the prognostic update rule and (b) the
M(z) profile.

All operations use smooth (differentiable) approximations for
compatibility with ``jax.grad``.

References
----------
- Arakawa, A., & Wu, C.-M. (2013). A unified representation of deep
  moist convection in numerical modeling of the atmosphere. Part I.
  J. Atmos. Sci., 70, 1977-1992.
- Siebesma, A. P., et al. (2007). A combined eddy-diffusivity
  mass-flux approach for the convective boundary layer.
  J. Atmos. Sci., 64, 1230-1248.
- Tiedtke, M. (1989). A comprehensive mass flux scheme for cumulus
  parameterization in large-scale models. Mon. Wea. Rev., 117,
  1779-1800.
"""

from __future__ import annotations

from typing import NamedTuple, Tuple

import jax
import jax.numpy as jnp

from legoesm import constants
from legoesm.thermo import saturation_mixing_ratio
from legoesm.timestepping.tridiagonal import thomas_solve_batched
from legoesm.atmosphere.physics._shared import (
    compute_layer_dz,
    compute_rho,
    virtual_temperature,
)
from legoesm.atmosphere.physics.thermodynamics import (
    parcel_profile_and_cape,
)
from legoesm.atmosphere.physics.convection.config import (
    ConvectiveEDMFConfig,
    MassFluxConfig,
)
from legoesm.atmosphere.physics.convection.output import (
    ConvectionOutput,
    split_convective_rain,
)


__physics_contract__ = {
    "summary": (
        "Prognostic bulk mass-flux convection: Arakawa-Wu (M_c x fixed "
        "sinusoidal profile) and a simplified single-updraft EDMF "
        "(M_u = rho*a_u*w_u). Both apply the shared compensating-subsidence + "
        "detrainment kernel; detrained condensate is handed to microphysics "
        "(precip deferred). Smooth (differentiable)."
    ),
    "inputs": {
        "T": "K", "q_v": "kg/kg", "p_full": "Pa", "p_half": "Pa",
        "M_c": "kg/m^2/s (Arakawa-Wu prognostic mass flux) or a_u = updraft area fraction [1] (EDMF)",
        "dt": "s",
    },
    "outputs": {
        "dT_dt": "K/s", "dq_v_dt": "kg/kg/s",
        "dq_c_conv_dt": "kg/kg/s (detrained cloud-water source to microphysics, >=0)",
        "cape": "J/kg", "convective_mask": "1 (0-1 activation)",
        "M_c_new": "kg/m^2/s (Arakawa-Wu) or a_u_new = updraft area fraction [1] (EDMF) — relaxed prognostic",
    },
    "sign_convention": (
        "z up; surface at [:, -1]. M_profile>=0 is the UPWARD updraft mass "
        "flux; the compensating environmental subsidence (-M/rho) warms/dries "
        "where the updraft detrains and stabilizes the column. "
        "dq_c_conv_dt >= 0 is a cloud-water SOURCE to microphysics (condensate "
        "is NOT precipitated here). Compensating subsidence + detrainment "
        "conserve column moist static energy (h=c_p*T+g*z+L_v*q_v) and total "
        "water: exactly (machine precision) in the conservative implicit_flux "
        "solve, to truncation order in the default advective solve."
    ),
    # The DEFAULT public path uses the advective subsidence solve, which
    # conserves MSE + total water only to TRUNCATION ORDER (exact only in the
    # opt-in implicit_flux solve), so no contract-level conservation is
    # guaranteed; the column budget is closed downstream.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Arakawa & Wu (2013), J. Atmos. Sci. 70, 1977-1992; "
        "Siebesma et al. (2007), J. Atmos. Sci. 64, 1230-1248; "
        "Tiedtke (1989), Mon. Wea. Rev. 117, 1779-1800"
    ),
    "idealized_test": (
        "tests/unit/test_physics_convection.py; a dry (low-RH) column produces "
        "no convective cloud water (test_no_cloud_water_in_dry_column); CAPE=0 "
        "-> zero convective_mask; the flux-form transport conserves column MSE "
        "and total water (implicit_flux to machine precision)."
    ),
}


# =============================================================================
# Shared helpers — used by both the Arakawa-Wu and simplified EDMF paths.
# =============================================================================


# --- pspec autoblock
_P_MIN_CONVECTION_PA = 10_000.0
_P_GATE_SHARPNESS_PA = 500.0

def compute_column_geometry(
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    q_v: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute layer thickness, density, and surface-relative height.

    Returns ``(dz, rho, z)``, all of shape ``(ncol, nlev)``.  Delegates
    to the shared atmosphere column helpers (`compute_layer_dz`,
    `compute_rho`) so a single hypsometric/EOS convention is used by
    every parameterization.  When ``q_v`` is supplied the geometry
    uses virtual temperature — moist tropical columns are ~1 % thicker
    and ~1 % less dense than the dry calculation, which biases the
    mass-flux closure when omitted.  Levels are ordered top-down, so
    ``z[:, -1]`` is the surface.
    """
    dz = compute_layer_dz(T, p_half, q_v=q_v)
    rho = compute_rho(T, p_full, q_v=q_v)
    # Full-level (cell-centre) height above the surface.  Cumulative
    # ``cumsum(dz[::-1])[::-1]`` gives the height of the *top* of each
    # layer (interface above the level); subtracting half the local
    # thickness places the height at the layer mid-point, which is
    # where ``T``/``q`` live.  An earlier formulation used the layer-top
    # value, biasing parcel ascent diagnostics by ~½ layer per level
    # (~10–250 m depending on resolution).  Codex iter-3 finding #8.
    z_top = jnp.cumsum(dz[:, ::-1], axis=1)[:, ::-1]
    z = z_top - 0.5 * dz
    return dz, rho, z


def _compute_cape_diagnostics(
    T: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    cape_threshold: float,
    cape_activation_scale: float,
    q_v: jax.Array | None = None,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Compute parcel profile, CAPE, and a smooth convective mask.

    When ``q_v`` is provided the parcel is lifted as **dry adiabat below
    the LCL, moist adiabat above** (using the surface-layer water-vapor
    mixing ratio as the launch humidity) and CAPE is computed with
    **virtual temperature**.  This is the physically correct trigger
    for unsaturated boundary layers; the legacy ``q_v=None`` path keeps
    the saturated-from-base assumption for callers that have not been
    migrated.

    Returns ``(T_moist, cape, convective_mask)``. ``T_moist`` has shape
    ``(ncol, nlev)``; ``cape`` and ``convective_mask`` are ``(ncol,)``.
    The mask is a sigmoid of ``(cape - cape_threshold) /
    cape_activation_scale``.
    """
    # Shared parcel -> CAPE recipe (dry->LCL->moist lift + virtual-T CAPE when
    # q_v is threaded; saturated-from-base legacy when q_v is None).
    T_moist, cape = parcel_profile_and_cape(T, p_full, p_half, q_v=q_v)
    convective_mask = jax.nn.sigmoid(
        (cape - cape_threshold) / cape_activation_scale
    )
    return T_moist, cape, convective_mask


def _compute_subsidence_gradients(
    T: jax.Array,
    q_v: jax.Array,
    z: jax.Array,
) -> Tuple[jax.Array, jax.Array]:
    """Upstream (upwind) vertical gradients for compensating subsidence.

    The compensating-subsidence tendency ``(M/ρ) ∂φ/∂z`` is a vertical
    *advection* by the environmental descent that balances the updraft
    mass flux (environment sinks, ``w_env = −M/ρ < 0``).  Differencing
    that advection with a **centered** stencil under the model's
    forward-Euler step is unconditionally unstable: it neither sees nor
    damps the 2Δz mode, so the moisture field develops a level-to-level
    checkerboard (small, bounded negative ``q_v`` at low mass flux) that
    diverges to NaN once the mass flux is large enough — which is why the
    mass-flux-family schemes had to be held below an artificially tight
    ``M_b_max`` cap, starving their convective heating and leaving the
    free troposphere tens of K too cold and super-adiabatic.

    Subsidence is downward, so the upstream cell is the one *above*
    (lower index, higher ``z``).  The donor-cell gradient at level ``k``
    is ``(φ[k-1] − φ[k]) / (z[k-1] − z[k])``; the model-top level (no
    cell above) gets a zero gradient, consistent with the mass-flux
    profile vanishing there.  This is the upstream differencing of
    Tiedtke (1989, §5): monotone, positivity-preserving, and stable
    under CFL ``(M/ρ)·dt/dz ≤ 1``.  Returns ``(dT_dz, dq_dz)``, both
    shape ``(ncol, nlev)``.
    """
    dz_up = jnp.clip(z[:, :-1] - z[:, 1:], 1.0, None)  # z[k-1]-z[k] > 0
    dT_dz = jnp.pad((T[:, :-1] - T[:, 1:]) / dz_up, ((0, 0), (1, 0)))
    dq_dz = jnp.pad((q_v[:, :-1] - q_v[:, 1:]) / dz_up, ((0, 0), (1, 0)))
    return dT_dz, dq_dz


def stratosphere_mass_flux_gate(
    p_full: jax.Array,
    p_min_convection: float = _P_MIN_CONVECTION_PA,
    p_gate_sharpness: float = _P_GATE_SHARPNESS_PA,
) -> jax.Array:
    """Smooth sigmoid factor in [0, 1] that vanishes above the
    tropopause (low ``p``) and equals one in the troposphere.

    Multiplying any mass-flux profile by the returned factor prevents
    convective tendencies from accumulating in the model top layer,
    where the small mass per unit area (Δp/g) would amplify modest
    heating into unphysical spikes (>400 K observed in 1-year RCE).

    Defaults: cutoff at 100 hPa with a 5-hPa transition width.  This
    gives factor ≈ 4.5e-5 at 50 hPa, 2.5e-3 at 70 hPa, 0.5 at 100 hPa,
    0.998 at 130 hPa, and ≈ 1.0 below 200 hPa — i.e. the gate is
    closed in the stratosphere while leaving the upper troposphere
    unaffected.  ``p_gate_sharpness``
    must be << ``p_min_convection`` for the sigmoid to saturate
    within the integration range; sharpness ≥ p_min only attenuates.

    Differentiable everywhere; ``p_gate_sharpness`` sets the width of
    the transition (Pa).
    """
    return jax.nn.sigmoid(
        (p_full - p_min_convection) / jnp.maximum(p_gate_sharpness, 1.0)
    )


def apply_mass_flux_kernel(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    T_u: jax.Array,
    q_v_u: jax.Array,
    q_c_u: jax.Array,
    M_profile: jax.Array,
    z: jax.Array,
    rho: jax.Array,
    delta_0: float,
    M_u_max: float,
    p_min_convection: float = _P_MIN_CONVECTION_PA,
    p_gate_sharpness: float = _P_GATE_SHARPNESS_PA,
    *,
    subsidence_solve: str = "advective",
    p_half: jax.Array | None = None,
    dt: float | None = None,
    theta_implicit: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    """Mass-flux core kernel: compensating subsidence + detrainment.

    Selectable vertical solve via ``subsidence_solve`` (validated at fn
    entry on the static Python value; CLAUDE.md dispatch-hardening):

    * ``"advective"`` (default) — the legacy donor-cell advective
      compensating-subsidence ``(M/ρ)·∂φ/∂z`` plus local detrainment.
      BYTE-IDENTICAL to the historical kernel; conserves only to
      truncation order because the advective form leaves a
      non-telescoping ``(φ/ρ)·dM/dz`` residual (column total-static-
      energy leak that shrinks with resolution).
    * ``"implicit_flux"`` — an IMPLICIT (backward-Euler, θ-blended)
      CONSERVATIVE flux-form solve of the compensating subsidence +
      detrainment as a tridiagonal system per column (see
      :func:`apply_mass_flux_kernel_implicit_flux`).  CONSERVATION is
      unconditional (the flux divergence telescopes for any M / θ / dt);
      STABILITY is the donor-cell backward-Euler kind (damps the 2Δz
      checkerboard the EXPLICIT flux form NaN'd on; diagonally dominant
      for ``θ·dt·g·M/Δp < 1``, comfortably met in the production regime —
      see the implicit kernel docstring) AND flux-form
      conservative: it transports dry static energy ``s = c_p T + g z``
      and vapor ``q_v`` so that the column integrals ``∫ c_p dT dp/g``
      and ``∫ L_v dq_v dp/g`` telescope to the (vanishing) top/base
      boundary flux — i.e. MSE ``h = s + L_v q_v`` is conserved by the
      transport to machine precision.  Requires ``p_half`` and ``dt``.

    Given the vertical mass-flux profile ``M_profile`` and the
    entraining updraft thermodynamics ``(T_u, q_v_u, q_c_u)``,
    returns ``(dT_dt, dq_v_dt, dq_c_conv_dt)`` — all shape
    ``(ncol, nlev)``.

    The decomposition follows Tiedtke (1989) / Siebesma et al. (2007):
      (a) Compensating subsidence: ``(M/rho) * (dT/dz + g/c_p)`` for T
          and ``(M/rho) * dq/dz`` for moisture (q is conserved so no
          adiabatic correction).
      (b) Detrainment mixing: ``+delta_0 * M * (X_u - X) / rho``.

    The mass flux ``M_profile`` is gated by a smooth sigmoid in
    pressure so that levels above ``p_min_convection`` (default 100
    hPa, the canonical tropical tropopause) receive no convective
    tendency.  See ``stratosphere_mass_flux_gate`` for details.

    The convective source for cloud water is the per-level detrainment
    of the plume's cloud water:
    ``dq_c_conv_dt = delta_0 * M * q_c_u / rho`` [kg/kg/s], non-negative
    by construction.  Splitting the plume into separate vapor (``q_v_u``)
    and cloud (``q_c_u``) pieces — instead of a single ``q_u`` that
    conflates total water with vapor — is what makes the column MSE
    budget close.  The earlier formulation passed ``q_u = q_v_u + q_c_u``
    as if it were vapor and computed condensate as ``max(q_u - q_sat,
    0)``, which is essentially zero for an entraining-diluted plume —
    the leaf then leaked latent energy.  Microphysics processes
    ``dq_c_conv_dt`` through its full chain (autoconversion,
    sedimentation, evaporation) and produces the resulting surface
    precipitation; convection no longer assumes the condensate falls
    instantly.
    Unit check: (1/m) * (kg/m²/s) * (kg/kg) / (kg/m³) = 1/s × kg/kg.
    """
    # Dispatch-hardening (CLAUDE.md): validate the static scheme value at
    # fn entry; a bare ``else`` would silently run different physics on a
    # typo.  ``subsidence_solve`` is a Python str (static), so this raises
    # at trace time, never inside a traced branch.
    if subsidence_solve not in ("advective", "implicit_flux"):
        raise ValueError(
            f"apply_mass_flux_kernel: unknown subsidence_solve "
            f"{subsidence_solve!r}; expected 'advective' or 'implicit_flux'"
        )
    if subsidence_solve == "implicit_flux":
        if p_half is None or dt is None:
            raise ValueError(
                "apply_mass_flux_kernel(subsidence_solve='implicit_flux') "
                "requires p_half and dt (the conservative flux solve is a "
                "backward-Euler step)."
            )
        return apply_mass_flux_kernel_implicit_flux(
            T, q_v, p_full, p_half,
            T_u, q_v_u, q_c_u, M_profile,
            z, rho, delta_0, M_u_max, dt,
            p_min_convection=p_min_convection,
            p_gate_sharpness=p_gate_sharpness,
            theta_implicit=theta_implicit,
        )

    dT_dz, dq_dz = _compute_subsidence_gradients(T, q_v, z)
    rho_safe = jnp.clip(rho, 0.01, None)  # coeff-ok: density floor

    # Per-level mass-flux cap.  The plume integrator can yield ``M_u``
    # that grows with height when ``epsilon > delta`` (entraining
    # plumes) or that responds non-linearly to a high-CAPE column.
    # Per-layer convective heating ``≈ delta_0 · M_u · (T_u−T)/ρ``
    # scales linearly with ``M_u``, so an uncapped ``M_u`` produces
    # column heating well in excess of what surface fluxes can supply
    # and destabilises the integration.  Callers thread the cap from
    # their config NamedTuple (typically ``config.M_b_max ≈ 0.05
    # kg/m²/s``, the literature peak tropical updraft mass flux); the
    # clip bounds per-layer tendencies without distorting the moist
    # adiabat or the q_v / q_c split.
    M_profile = jnp.clip(M_profile, 0.0, M_u_max)

    # Stratospheric pressure gate — see ``stratosphere_mass_flux_gate``.
    M_profile = M_profile * stratosphere_mass_flux_gate(
        p_full, p_min_convection, p_gate_sharpness,
    )

    dT_subsidence = (M_profile / rho_safe) * (dT_dz + constants.g / constants.c_pd)
    dq_subsidence = (M_profile / rho_safe) * dq_dz

    dT_detrain = delta_0 * M_profile * (T_u - T) / rho_safe
    dq_detrain = delta_0 * M_profile * (q_v_u - q_v) / rho_safe

    dT_dt = dT_subsidence + dT_detrain
    dq_v_dt = dq_subsidence + dq_detrain

    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
    return dT_dt, dq_v_dt, dq_c_conv_dt


def apply_mass_flux_kernel_implicit_flux(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    T_u: jax.Array,
    q_v_u: jax.Array,
    q_c_u: jax.Array,
    M_profile: jax.Array,
    z: jax.Array,
    rho: jax.Array,
    delta_0: float,
    M_u_max: float,
    dt: float,
    p_min_convection: float = _P_MIN_CONVECTION_PA,
    p_gate_sharpness: float = _P_GATE_SHARPNESS_PA,
    theta_implicit: float = 1.0,
) -> Tuple[jax.Array, jax.Array, jax.Array]:
    r"""IMPLICIT (backward-Euler / θ-blended) CONSERVATIVE flux-form solve
    of the compensating subsidence + detrainment, per column.

    Coordinate / sign convention (stated AT the term per CLAUDE.md):
    ``z`` increases UPWARD; arrays are surface-LAST (index 0 = model
    top, index ``nlev-1`` = surface), so the cell ABOVE an interface has
    the LOWER index.  ``M_profile ≥ 0`` is the UPDRAFT mass flux
    [kg/m²/s] (upward); the compensating environmental subsidence has
    mass flux ``-M`` (downward, ``w_env = -M/ρ < 0``).

    Continuous flux form (z up).  The conservative form CONSISTENT with the
    legacy advective compensating-subsidence ``+(M/ρ)∂φ/∂z`` is::

        ρ ∂φ/∂t = ∂/∂z [ M (φ − φ_u) ]                              (1)

    NOTE the sign: it is ``M(φ − φ_u)``, NOT ``M(φ_u − φ)``.  The latter
    flips the compensating-subsidence sign (anti-diffusive ⇒ a 2Δz
    checkerboard GROWS — verified numerically) — see the ``_solve``
    docstring for the derivation from ``∂s_u/∂z = −ε(s_u − s)``.  The
    single divergence bundles compensating subsidence AND detrainment; its
    column integral telescopes::

        ∫ ρ ∂φ/∂t dz = [M(φ−φ_u)]_top − [M(φ−φ_u)]_base = 0

    because ``M = 0`` below cloud base and above the level of neutral
    buoyancy.  So ``∫ ∂φ/∂t dp/g`` is conserved by the TRANSPORT — unlike
    the advective ``(M/ρ)∂φ/∂z + δM(φ_u−φ)`` form, which leaves a
    non-telescoping ``(φ/ρ)dM/dz`` residual.

    Finite-volume discretisation (conservative, machine-precision
    telescoping).  Layer ``k`` (mass per area ``Δp_k/g``, ``Δp_k =
    p_half[k+1] − p_half[k] > 0``)::

        (Δp_k/g) dφ_k/dt = G_k − G_{k+1}                            (2)

    where the interface flux ``G_i = M_i (φ_env[i-1] − φ_u[i])`` uses
    UPSTREAM donors — the subsiding environment (downward) donates from the
    cell ABOVE (level ``i-1``, the IMPLICIT term); the updraft (upward)
    donates from the cell BELOW (level ``i``, EXPLICIT).  Interface mass
    flux ``M_i =
    ½(M[i-1]+M[i])`` for interior faces, ``M_0 = M_nlev = 0`` (top and
    surface).  The SAME ``M_i`` appears in ``G_k`` (bottom of layer
    ``k-1``) and ``G_{k+1}`` (top of layer ``k``), so the column sum is
    ``Σ_k (G_k − G_{k+1}) = G_0 − G_nlev = 0`` for ANY M-profile —
    conservation is unconditional (independent of θ, dt, or the M
    shape/clip).

    Backward-Euler (θ-implicit) for the IMPLICIT environment ``φ`` with
    the EXPLICIT updraft property ``φ_u``::

        φ^{n+1}_k − φ^n_k = (dt g/Δp_k)·θ·(G_k − G_{k+1})^{n+1}
                          + (dt g/Δp_k)·(1−θ)·(G_k − G_{k+1})^{n}

    Writing ``r_k = θ·dt·g/Δp_k`` and using the env-donor upstream (cell
    above only ⇒ the implicit coupling is to ``φ^{n+1}[k-1]`` ONLY, a
    LOWER-bidiagonal special tridiagonal with zero super-diagonal)::

        -r_k M_k φ^{n+1}[k-1] + (1 + r_k M_{k+1}) φ^{n+1}[k]
            = φ^n_k + r_k (M_{k+1} φ_u[k+1] − M_k φ_u[k])     (+ (1−θ) expl.)

    Stability.  The per-row amplification is ``r_k M_k / (1 + r_k
    M_{k+1})``.  Strict diagonal dominance ``1 + r_k M_{k+1} > r_k M_k``
    holds wherever ``M`` increases downward (entraining lower branch) and,
    crucially, ALWAYS in the limit ``r → 0``; on the DESCENDING side of a
    bell-shaped ``M`` (``M_k > M_{k+1}``, e.g. the layer just above the
    cloud base where ``M_{k+1} → 0``) the row is dominant only when
    ``r_k M_k < 1`` — i.e. a donor-cell CFL-like bound ``θ·dt·g·M_k/Δp_k
    < 1``.  For the production regime (``M ≤ M_u_max ≈ 0.05 kg/m²/s``,
    ``dt ≈ 1800 s``, ``Δp ≳ 2–3 kPa``) ``r_k M_k ≈ 0.3 ≪ 1``, so the
    backward-Euler solve is unconditionally damping in practice — and far
    more robust than the EXPLICIT flux form (which NaN'd on the same
    M-profile): the implicit operator removes the 2Δz checkerboard rather
    than amplifying it.  The claim is "stable for the production M/dt/Δp
    regime", NOT literally unconditional for arbitrary ``M``.  Solved with
    the shared AD-safe
    :func:`legoesm.timestepping.tridiagonal.thomas_solve_batched`
    (reverse-mode differentiable, vmap-able).  ``θ = 1`` is fully implicit
    (default, maximally damping); ``θ ∈ [0.5, 1]`` allowed.

    Conserved quantity & two-test reconciliation.  We transport DRY
    STATIC ENERGY ``s = c_p T + g z`` (plume ``s_u = c_p T_u + g z``;
    ``z`` is Eulerian so ``dT = ds/c_p``) and VAPOR ``q_v`` (plume
    ``q_v_u``) — both via (2), so ``∫ c_p dT dp/g`` and ``∫ L_v dq_v dp/g``
    telescope to the vanishing boundary flux ⇒ column MSE ``h = s + L_v
    q_v`` is conserved by the TRANSPORT to machine precision.  The
    DETRAINED condensate ``dq_c = δ·M·q_c_u/ρ`` (≥ 0, identical to the
    advective kernel) is the genuine cloud-water SOURCE handed to
    microphysics; it is matched by a vapor sink ``−dq_c`` so the column
    TOTAL water ``∫(dq_v + dq_c) dp/g`` closes and the column MSE budget
    ``c_p∫dT + L_v∫dq_v + L_v∫dq_c = −L_v∫dq_c (cloud latent exported) +
    L_v∫dq_c = 0`` closes to machine precision (the detrained cloud
    carries its latent heat forward, released downstream by
    microphysics).  Returns ``(dT_dt, dq_v_dt, dq_c_conv_dt)``.
    """
    g = constants.g
    c_pd = constants.c_pd
    ncol, nlev = T.shape

    # Pin the whole solve to ONE working dtype = the state precision.  The
    # tridiagonal solve requires a/b/c (from M / Δp) and d (from s = c_p T + g z
    # and q_v) to share a dtype, but z / geometry can arrive at a DIFFERENT
    # precision than the state fields — a float32 orchestrator passes float64 z,
    # which would leave d=float64 while a/b/c stay float32 and trip
    # thomas_solve_batched's mixed-dtype guard.  result_type(T, q_v) keeps a
    # float32 run all-float32 and an x64 run all-float64 (pin-to-state doctrine;
    # jax-scan-carry-dtype-stability).
    work_dtype = jnp.result_type(T, q_v)
    T = T.astype(work_dtype)
    q_v = q_v.astype(work_dtype)
    p_full = p_full.astype(work_dtype)
    p_half = p_half.astype(work_dtype)
    T_u = T_u.astype(work_dtype)
    q_v_u = q_v_u.astype(work_dtype)
    q_c_u = q_c_u.astype(work_dtype)
    M_profile = M_profile.astype(work_dtype)
    z = z.astype(work_dtype)
    rho = rho.astype(work_dtype)

    rho_safe = jnp.clip(rho, 0.01, None)  # coeff-ok: density floor

    # Same per-level cap + stratospheric gate as the advective kernel so
    # the transported mass flux is identical at the source.
    M_profile = jnp.clip(M_profile, 0.0, M_u_max)
    M_profile = M_profile * stratosphere_mass_flux_gate(
        p_full, p_min_convection, p_gate_sharpness,
    )

    dp = p_half[:, 1:] - p_half[:, :-1]              # (ncol, nlev) > 0
    dp = jnp.clip(dp, 1.0, None)                     # coeff-ok: Δp floor (safe denom)

    # Interface (face) mass flux, faces i = 0..nlev (nlev+1 of them).
    # Interior faces i=1..nlev-1 sit between level i-1 (above) and i
    # (below); top face (i=0) and surface face (i=nlev) carry M = 0.
    M_int = jnp.zeros((ncol, nlev + 1), dtype=M_profile.dtype)
    M_int = M_int.at[:, 1:nlev].set(0.5 * (M_profile[:, :-1] + M_profile[:, 1:]))
    Mk = M_int[:, 0:nlev]        # face k = TOP face of layer k
    Mk1 = M_int[:, 1:nlev + 1]   # face k+1 = BOTTOM face of layer k

    # θ-implicit blend factor (clamp to [0.5, 1]; θ ≥ 0.5 removes the
    # explicit-side amplification, default 1.0 is fully implicit and most
    # damping).  coeff-ok: stability bound.
    theta = jnp.clip(theta_implicit, 0.5, 1.0)  # coeff-ok: θ-implicit stable range
    r = theta * dt * g / dp                          # (ncol, nlev)

    def _solve(phi_n: jax.Array, phi_u: jax.Array) -> jax.Array:
        """Backward-Euler conservative flux solve for one tracer.

        The conservative env tendency consistent with the legacy advective
        compensating-subsidence ``+(M/ρ)∂φ/∂z`` is ``ρ ∂φ/∂t =
        ∂_z[M(φ − φ_u)]`` (NOT ``∂_z[M(φ_u − φ)]`` — that flips the
        subsidence sign and is anti-diffusive ⇒ a 2Δz checkerboard GROWS,
        verified numerically).  Finite-volume per layer ``k``:

            (Δp_k/g) dφ_k/dt = G_k − G_{k+1},  G_i = M_i(φ[i-1] − φ_u[i])

        with env donor = cell ABOVE (i-1; downward subsidence is upstream
        from above ⇒ implicit sub-diagonal) and updraft donor = cell BELOW
        (i; upward ⇒ explicit).  Backward-Euler:

            -r_k M_k φ^{n+1}[k-1] + (1 + r_k M_{k+1}) φ^{n+1}[k]
                = φ^n_k + r_k (M_{k+1} φ_u[k+1] − M_k φ_u[k])

        Diagonally dominant (b = 1 + r M_{k+1} > |a| = r M_k) wherever M
        increases downward and, in the limit r -> 0, always; on the
        descending side of the bell (M_k > M_{k+1}, M_{k+1} -> 0 near
        cloud base) dominance needs the donor-cell CFL-like bound
        r M_k < 1 (theta*dt*g*M_k/dp < 1), comfortably met in the
        production regime (r M ~ 0.3 at M<=0.05, dt=1800, dp>=2-3 kPa).
        Where dominant the homogeneous solve damps; the explicit updraft
        source is bounded.  See the header docstring for the full bound.
        """
        # Tridiagonal (lower-bidiagonal): a couples k-1, b diag, c=0.
        a = (-r * Mk)
        a = a.at[:, 0].set(0.0)                       # top face M_0 = 0
        b = 1.0 + r * Mk1
        c = jnp.zeros_like(b)
        # Explicit updraft source: r_k (M_{k+1} φ_u[k+1] − M_k φ_u[k]) =
        # −r_k·S_impl.  φ_u[k+1] beyond surface = 0.  The MINUS sign is the
        # upward-updraft flux divergence ``−∂_z[M φ_u]`` (codex/derivation:
        # a ``+`` here flips the updraft transport and destabilises).
        phi_u_kp1 = jnp.pad(phi_u[:, 1:], ((0, 0), (0, 1)))
        S_impl = Mk * phi_u - Mk1 * phi_u_kp1
        d = phi_n - r * S_impl
        if theta_implicit < 1.0:
            # θ-blend: add the (1−θ) EXPLICIT transport of the OLD field,
            # SAME flux convention as the implicit part: G_i^n =
            # M_i(φ_n[i-1] − φ_u[i]); explicit divergence G_k − G_{k+1}.
            phi_n_above = jnp.pad(phi_n[:, :-1], ((0, 0), (1, 0)))   # φ_n[k-1]
            G_k = Mk * (phi_n_above - phi_u)
            G_kp1 = Mk1 * (phi_n - phi_u_kp1)
            div_expl = G_k - G_kp1
            d = d + (1.0 - theta) * dt * g / dp * div_expl
        return thomas_solve_batched(a, b, c, d)

    # --- Transport dry static energy s = c_p T + g z (conserves MSE) ---
    s = c_pd * T + g * z
    s_u = c_pd * T_u + g * z
    s_new = _solve(s, s_u)
    dT_dt = (s_new - s) / (c_pd * dt)                # z Eulerian ⇒ dT = ds/c_p

    # --- Transport vapor q_v (plume q_v_u) -----------------------------
    q_v_new = _solve(q_v, q_v_u)
    dq_v_transport = (q_v_new - q_v) / dt

    # --- Detrained condensate: genuine cloud source (≥ 0), identical to
    #     the advective kernel.  Matched by a vapor sink so column total
    #     water closes; the cloud carries its latent heat forward. ------
    dq_c_conv_dt = delta_0 * M_profile * jnp.clip(q_c_u, 0.0, None) / rho_safe
    dq_v_dt = dq_v_transport - dq_c_conv_dt
    return dT_dt, dq_v_dt, dq_c_conv_dt


# =============================================================================
# Arakawa-Wu prognostic mass-flux (M_c × fixed sinusoidal profile)
# =============================================================================


class MassFluxClosureDiagnostics(NamedTuple):
    """Intermediate closure state reused by physical and ML mass-flux paths."""

    dz: jax.Array
    rho: jax.Array
    z: jax.Array
    T_moist: jax.Array
    cape: jax.Array
    M_eq: jax.Array
    M_c_new: jax.Array
    convective_mask: jax.Array


def diagnose_mass_flux_closure(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    M_c: jax.Array,
    dt: float,
    config: MassFluxConfig = MassFluxConfig(),
) -> MassFluxClosureDiagnostics:
    """Diagnose closure terms before computing mass-flux tendencies."""
    dz, rho, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_moist, cape, convective_mask = _compute_cape_diagnostics(
        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
        q_v=q_v,
    )

    M_eq = convective_mask * config.M_scale
    M_c_new = jnp.maximum(M_c + dt * (M_eq - M_c) / config.tau_adj, 0.0)

    return MassFluxClosureDiagnostics(
        dz=dz,
        rho=rho,
        z=z,
        T_moist=T_moist,
        cape=cape,
        M_eq=M_eq,
        M_c_new=M_c_new,
        convective_mask=convective_mask,
    )


def mass_flux_convection_from_closure(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    closure: MassFluxClosureDiagnostics,
    config: MassFluxConfig = MassFluxConfig(),
    dt: float | None = None,
) -> ConvectionOutput:
    """Compute mass-flux tendencies from a supplied closure state.

    ``dt`` [s] is required ONLY when ``config.subsidence_solve ==
    "implicit_flux"`` (the conservative flux-form solve is a backward-Euler
    step, so it needs the step size).  It stays optional so the historical
    ``advective`` default -- which never reads it -- keeps its existing
    call signature; the shared kernel raises a clear ValueError if
    ``implicit_flux`` is selected without it, rather than silently
    degrading to the leaky solve.
    """
    rho = closure.rho
    z = closure.z
    T_moist = closure.T_moist
    cape = closure.cape
    M_c_new = closure.M_c_new
    convective_mask = closure.convective_mask

    p_base = p_full[:, -1:]
    p_top = p_full[:, :1]
    p_range = jnp.clip(p_base - p_top, 1.0, None)
    # Sinusoidal vertical profile: 0 at base/top, peak at mid-troposphere.
    m_profile = jnp.sin(jnp.pi * (p_base - p_full) / p_range)
    M_profile = M_c_new[:, None] * m_profile

    # Entraining updraft.  The undiluted plume rises with a fixed
    # total-water reservoir equal to the actual launched-parcel vapor
    # ``q_v_sfc`` (NOT ``q_sat_sfc`` — using the saturation value here
    # would let a 5%-RH desert column produce convective cloud water
    # because the plume would "remember" being saturated at base when
    # it never was).  At each level the undiluted plume vapor saturates
    # at ``min(q_sat_moist, q_v_sfc)`` and condenses the excess; below
    # the actual LCL (``q_sat_moist > q_v_sfc``) condensation is zero.
    # Entrainment dilutes both T and q_v with environmental values; the
    # entrained env air carries no q_c, so q_c_u just scales by
    # ``dilution``.  The fix keeps the saturated-surface case
    # (``q_v_sfc = q_sat_sfc``) identical to the old formula.
    dilution = jnp.exp(-config.epsilon_0 * z)
    T_u = dilution * T_moist + (1.0 - dilution) * T
    q_v_sfc = q_v[:, -1:]
    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
    q_v_u_undiluted = jnp.minimum(q_sat_moist, q_v_sfc)
    q_c_u_undiluted = jnp.clip(q_v_sfc - q_sat_moist, 0.0, None)
    q_v_u = dilution * q_v_u_undiluted + (1.0 - dilution) * q_v
    q_c_u = dilution * q_c_u_undiluted

    dT_dt, dq_v_dt, dq_c_conv_dt = apply_mass_flux_kernel(
        T=T,
        q_v=q_v,
        p_full=p_full,
        T_u=T_u,
        q_v_u=q_v_u,
        q_c_u=q_c_u,
        M_profile=M_profile,
        z=z,
        rho=rho,
        delta_0=config.delta_0,
        M_u_max=config.M_b_max,
        # Selectable vertical solve (config default "advective" = shipped
        # behaviour, byte-identical).  ``p_half``/``dt`` are only consumed by
        # the implicit_flux branch; the kernel raises on an unknown value
        # (dispatch-hardening, static Python str).
        subsidence_solve=config.subsidence_solve,
        p_half=p_half,
        dt=dt,
        theta_implicit=config.theta_implicit,
    )

    # In-updraft precipitation: shared rain-split (same knob + mass proof as
    # Tiedtke/Bechtold). precip_efficiency=0 (default) => no split, byte-identical.
    dq_c_conv_dt, dq_r_conv_dt = split_convective_rain(
        dq_c_conv_dt, config.precip_efficiency)

    return ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
        dq_r_conv_dt=dq_r_conv_dt,
    )


def mass_flux_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    M_c: jax.Array,
    dt: float,
    config: MassFluxConfig = MassFluxConfig(),
) -> Tuple[ConvectionOutput, jax.Array]:
    """Compute Prognostic Mass-Flux convection tendencies."""
    closure = diagnose_mass_flux_closure(
        T=T,
        q_v=q_v,
        p_full=p_full,
        p_half=p_half,
        M_c=M_c,
        dt=dt,
        config=config,
    )
    conv_out = mass_flux_convection_from_closure(
        T=T,
        q_v=q_v,
        p_full=p_full,
        p_half=p_half,
        closure=closure,
        config=config,
        dt=dt,
    )
    return conv_out, closure.M_c_new


# =============================================================================
# Simplified EDMF (M_u(z) = ρ · a_u · w_u(z) from a buoyancy integral)
# =============================================================================


def updraft_velocity_from_buoyancy(
    B: jax.Array, dz: jax.Array, w_u_min: float, w_u_max: float,
) -> jax.Array:
    """Updraft vertical velocity from the SIGNED buoyancy integral (surface up).

    Coordinate convention (stated at the term per CLAUDE.md): ``z`` increases
    UPWARD; arrays are surface-LAST (index 0 = model top, index ``-1`` = surface);
    ``dz > 0`` and ``B > 0`` is an upward-buoyant plume.  The textbook updraft
    kinetic-energy equation ``d(½w²)/dz = B`` integrates to::

        w²(z) = w_min² + 2·∫_sfc^z B dz'

    so the plume ACCELERATES through the positively-buoyant CAPE layer and
    DECELERATES above the level of neutral buoyancy where ``B < 0`` — it does NOT
    freeze at its peak.

    #824: the previous EDMF form used ``clip(B·dz, 0, None)``, dropping the
    negative (above-LNB) contribution so the integral FROZE at its peak and the
    mass flux ``M = ρ·a_u·w_u`` plateaued at its cap from the LNB all the way to
    the 100 hPa gate — a top-heavy, non-detraining profile that put the strongest
    compensating subsidence in the thin-mass upper troposphere and drove the
    day-5 full-physics AMIP blowup.  With the SIGNED integral the plume detrains
    to ~0 near its top like every stable scheme.  Sub-cloud CIN (``B < 0`` below
    the LFC) makes the cumulative integral negative there, so ``w_u`` floors
    through the stable sub-cloud layer and the plume accelerates only once it
    reaches the buoyant layer (an elevated plume launches where it is buoyant).

    The sqrt argument is clipped to ``[1e-12, w_u_max²]``: the tiny positive
    floor keeps ``w_u`` AD-safe even at ``w_u_min = 0`` / a non-buoyant column
    (``sqrt'(0) = ∞`` → NaN reverse-mode grad), and the upper cap bounds
    ``sqrt(2·CAPE)`` at a physical maximum updraft speed (``w_u_max`` ~ 50 m/s).

    Parameters
    ----------
    B, dz : jax.Array, shape ``(ncol, nlev)``
        Buoyancy [m/s²] and layer thickness [m], surface-last.
    w_u_min, w_u_max : float
        Minimum (floor) and maximum (cap) updraft speed [m/s].

    Returns
    -------
    w_u : jax.Array, shape ``(ncol, nlev)`` — updraft vertical velocity [m/s] ≥ 0.
    """
    # Reverse to surface-first, cumulative-sum upward, reverse back: B_integral[k]
    # = ∫ from the surface up to level k.
    B_dz_rev = (B * dz)[:, ::-1]
    B_integral = jnp.cumsum(B_dz_rev, axis=1)[:, ::-1]
    return jnp.sqrt(jnp.clip(2.0 * B_integral + w_u_min ** 2, 1e-12, w_u_max ** 2))


def edmf_convection(
    T: jax.Array,
    q_v: jax.Array,
    p_full: jax.Array,
    p_half: jax.Array,
    a_u: jax.Array,
    dt: float,
    config: ConvectiveEDMFConfig = ConvectiveEDMFConfig(),
) -> Tuple[ConvectionOutput, jax.Array]:
    """Compute simplified EDMF convection tendencies.

    Parameters
    ----------
    T : jax.Array
        Temperature at full levels [K], shape ``(ncol, nlev)``.
    q_v : jax.Array
        Water vapor specific humidity [kg/kg], shape ``(ncol, nlev)``.
    p_full : jax.Array
        Pressure at full levels [Pa], shape ``(ncol, nlev)``.
    p_half : jax.Array
        Pressure at half levels [Pa], shape ``(ncol, nlev+1)``.
    a_u : jax.Array
        Updraft area fraction, shape ``(ncol,)``.
    dt : float
        Model time step [s].
    config : ConvectiveEDMFConfig
        Convection configuration.

    Returns
    -------
    ConvectionOutput
        Convective tendencies and diagnostics.
    a_u_new : jax.Array
        Updated updraft area fraction, shape ``(ncol,)``.
    """
    dz, rho, z = compute_column_geometry(T, p_full, p_half, q_v=q_v)
    T_moist, cape, convective_mask = _compute_cape_diagnostics(
        T, p_full, p_half, config.cape_threshold, config.cape_activation_scale,
        q_v=q_v,
    )

    # Diagnosed equilibrium updraft area fraction; prognostic relaxation
    # then clipping to a physically plausible range.
    a_u_eq = convective_mask * config.a_u_init
    a_u_new = a_u + dt * (a_u_eq - a_u) / config.tau_a
    a_u_new = jnp.clip(a_u_new, 0.0, 0.5)

    # Entraining updraft.  Same correction as in ``mass_flux_convection``:
    # the undiluted plume's water reservoir is the *actual* launched-parcel
    # vapor ``q_v_sfc``, not ``q_sat_sfc``.  Without this the plume would
    # condense in dry columns even when the surface parcel never reached
    # saturation (e.g. 5%-RH desert column gives ~2 mm/day of spurious
    # convective precipitation — see ``test_no_cloud_water_in_dry_column``).
    dilution = jnp.exp(-config.epsilon_0 * z)
    T_u = dilution * T_moist + (1.0 - dilution) * T
    q_v_sfc = q_v[:, -1:]
    q_sat_moist = saturation_mixing_ratio(T_moist, p_full)
    q_v_u_undiluted = jnp.minimum(q_sat_moist, q_v_sfc)
    q_c_u_undiluted = jnp.clip(q_v_sfc - q_sat_moist, 0.0, None)
    q_v_u = dilution * q_v_u_undiluted + (1.0 - dilution) * q_v
    q_c_u = dilution * q_c_u_undiluted

    # Buoyancy: B = g * (T_v_u - T_v_env) / T_v_env.  Vapor contributes
    # ``+(R_v/R_d - 1) q_v ≈ +0.608 q_v`` (water vapour is lighter than
    # dry air) and cloud water contributes ``-q_c`` (the loaded
    # condensate is mass drag, not buoyancy).  An earlier formulation
    # used ``T_v = T*(1 + 0.61*(q_v + q_c))`` which treated q_c with the
    # *wrong sign* — making cloudy plumes spuriously buoyant — so we
    # use the standard form here.  ``virtual_temperature`` returns
    # ``T*(1 + 0.608·q_v)`` from ``constants.epsilon``; the cloud-water
    # loading term ``-T_u·q_c_u`` is added explicitly.
    T_v_env = virtual_temperature(T, q_v)
    T_v_u = virtual_temperature(T_u, q_v_u) - T_u * q_c_u
    B = constants.g * (T_v_u - T_v_env) / jnp.clip(T_v_env, 1.0, None)

    # Updraft velocity from the SIGNED buoyancy integral (surface upward): the
    # plume accelerates through the CAPE layer and DECELERATES above the level of
    # neutral buoyancy so the mass flux detrains near its top (see
    # :func:`updraft_velocity_from_buoyancy` for the full derivation + #824).
    w_u = updraft_velocity_from_buoyancy(
        B, dz, config.w_u_min, config.w_u_max)

    # Mass flux profile: M_u(z) = rho * a_u * w_u(z)  [kg/m²/s, upward ≥ 0].
    M_profile = rho * a_u_new[:, None] * w_u

    del dz  # (kept for interface symmetry — kernel no longer needs it)
    # #824: route through the CONSERVATIVE, backward-Euler ``implicit_flux``
    # subsidence solve (config default), mirroring Bechtold — the explicit
    # ``advective`` default leaked column static energy and NaN'd on the 2Δz
    # checkerboard.  ``edmf_convection`` already receives ``p_half``/``dt``.
    dT_dt, dq_v_dt, dq_c_conv_dt = apply_mass_flux_kernel(
        T=T,
        q_v=q_v,
        p_full=p_full,
        T_u=T_u,
        q_v_u=q_v_u,
        q_c_u=q_c_u,
        M_profile=M_profile,
        z=z,
        rho=rho,
        delta_0=config.delta_0,
        M_u_max=config.M_b_max,
        subsidence_solve=config.subsidence_solve,
        p_half=p_half,
        dt=dt,
        theta_implicit=config.theta_implicit,
    )

    # In-updraft precipitation: shared rain-split (EDMF config knob).
    dq_c_conv_dt, dq_r_conv_dt = split_convective_rain(
        dq_c_conv_dt, config.precip_efficiency)

    conv_out = ConvectionOutput(
        dT_dt=dT_dt,
        dq_v_dt=dq_v_dt,
        dq_c_conv_dt=dq_c_conv_dt,
        cape=cape,
        convective_mask=convective_mask,
        dq_r_conv_dt=dq_r_conv_dt,
    )
    return conv_out, a_u_new
