"""Prognostic eddy-kinetic-energy (EKE) closure — Eden & Greatbatch (2008).

A 2-D (depth-integrated) eddy-energy field ``E`` whose budget is

    dE/dt + advection(E) = iso-diffusion(E) + P - eps

with a Visbeck-style **prognostic** GM coefficient that replaces a constant one:

    kappa_GM = c_k * L * sqrt(E)                                   (>= 0)
    P        = kappa_GM * sigma^2     (GM mean-APE -> EKE conversion; sigma = N|S|)
    eps      = c_eps * E^{3/2} / L    (Eden-Greatbatch dissipation, >= 0)

``E`` is 2-D to match the 2-D ``kappa_GM`` the GM/Redi tendency already accepts
(the Visbeck path). ``sigma`` (depth-averaged Eady growth rate ``<N|S|>_z``) and the
mixing length ``L`` (first-baroclinic Rossby radius, floored at ``l_min``) come from
the SHARED GM/Redi Visbeck machinery (``compute_visbeck_kappa_gm`` internals) — this
module does NOT recompute N^2/slopes/L (no duplicate numerics). The closure functions
here are pure and take ``E``, ``sigma``, ``L`` as inputs; the advection + isopycnal
diffusion of ``E`` reuse the tracer-transport machinery; the state-field threading +
GM/Redi coupling are wired separately (build-spec gates E2, E6).

The mixing length ``L`` is selectable (``EKEConfig.mixing_length_scheme``):
``"rossby"`` (default) uses the Visbeck first-baroclinic length ``max(L_rossby,
l_min)`` — legoESM's pre-``eke_len`` behaviour; ``"rhines"`` reproduces Veros's
Rhines-limited ``eke_len = max(l_min, min(eke_cross·L_rossby, eke_crhin·L_rhines))``
from the deformation radius (``eke_deformation_radius``) and the eddy-energy Rhines
scale (``eke_rhines_length``). See ``docs/ocean_fidelity/eke_len_build_spec.md``.

Defaults match Veros ACC (``eke_c_k=0.4``, ``eke_c_eps=0.5``, ``eke_lmin=100``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


class EKEConfig(NamedTuple):
    """Prognostic EKE closure parameters (Eden & Greatbatch 2008)."""

    c_k: float = 0.4          # kappa_GM = c_k * L * sqrt(E)   (Veros eke_c_k)
    c_eps: float = 0.5        # dissipation eps = c_eps * E^{3/2}/L (Veros eke_c_eps)
    l_min: float = 100.0      # mixing-length floor [m]        (Veros eke_lmin)
    k_iso: float = 1000.0     # isopycnal diffusivity for E [m^2/s]
    advection_scheme: str = "superbee"  # E advection (reuses the tracer dispatch)
    e_min: float = 1.0e-8     # positivity floor on E [m^2/s^2]
    kappa_gm_max: float = 1.0e4  # safety ceiling on the prognostic kappa_GM [m^2/s]
    # Mixing-length scheme for kappa_GM = c_k·L·√E:
    #   "rossby" (default) — L = max(L_rossby, l_min), the Visbeck first-baroclinic
    #     length (N̄·H/|f|). legoESM's pre-eke_len behaviour (eke_cross/eke_crhin unused).
    #   "rhines" — Veros eke_len = max(l_min, min(eke_cross·L_rossby, eke_crhin·L_rhines))
    #     from the deformation radius c₁/|f| (c₁=∫N dz/π, equatorial-limited) and the
    #     eddy-energy Rhines scale √(√E/β).
    mixing_length_scheme: str = "rossby"
    eke_cross: float = 1.0    # deformation-radius weight in eke_len (Veros eke_cross)
    eke_crhin: float = 1.0    # Rhines-scale weight in eke_len      (Veros eke_crhin)
    # When True (and EKE on), the Redi *tracer* isopycnal diffusivity follows the
    # prognostic GM coefficient (K_iso = K_gm) instead of the constant kappa_Redi —
    # Veros's ``enable_eke_isopycnal_diffusion`` (default False; Veros ACC = True).
    isopycnal_diffusion: bool = False
    # When True, the eddy-energy field ``E`` is 3-D (depth-resolved, on the interior
    # interfaces / W-grid) and its budget uses the depth-resolved source/sink, the
    # implicit vertical EKE diffusion, and the per-level horizontal transport (the
    # ``eke_3d_*`` functions) — matching Veros's 3-D ``vs.eke`` on the W-grid. When
    # False (default) the 2-D depth-integrated closure (``eke_local_tendency`` etc.)
    # is used, bit-identically to the pre-3-D path. The model step (a later build
    # stage) flips this on for the ACC recipe; the closure functions are selected by
    # the static Python bool, never traced.
    eke_3d: bool = False
    # Vertical-EKE-diffusion factor: the implicit vertical diffusion of ``E`` uses
    # ``K = alpha_eke · A_v`` where ``A_v`` is the vertical viscosity at the W-grid
    # interfaces. Matches Veros ``settings.alpha_eke`` ("factor vertical friction",
    # default 1.0 in veros/settings.py; the ACC setup leaves it at the 1.0 default).
    # Only used by the 3-D path (``eke_3d_vertical_diffusion``).
    alpha_eke: float = 1.0
    # --- EKE SOURCE augmentation (default off ⇒ existing source bit-identical) ---
    # When True, route the mean-KE removed by the harmonic LATERAL viscosity A_h
    # into the EKE source (Veros ``K_diss_h``; veros/core/eke.py:110, computed from
    # the A_h∇²u momentum tendency in veros/core/friction.py:calc_diss_u/v). The
    # legoESM EKE source omits this term, which is ~56% of Veros's ACC EKE forcing
    # (the dominant deficit). The 3-D model step builds the [m²/s³] source from
    # legoESM's own harmonic-viscosity tendency and adds it to the W-grid source.
    # ACC recipe opts in; default off keeps the 2-D + existing-3-D path identical.
    source_kdiss_h: bool = False
    # K_diss_h discretisation (only used when source_kdiss_h=True):
    #   False (default) — the DYNAMICAL KE-tendency form ``-u·(A_h∇²_vec u)``
    #     (``harmonic_lateral_kediss_eke_source`` fed the ``Ah_visc_u/v`` Laplacian
    #     tendencies), which is NOT positive-definite (~35% of wet cells negative
    #     from the transport divergence) and is CLAMPED ≥ 0. The clamp over-credits
    #     the domain-integrated KE dissipation by ~11–20% (probe-measured on the
    #     ACC spin-up). BIT-IDENTICAL to the pre-flux-form path.
    #   True — the FAITHFUL POSITIVE-DEFINITE flux form (Veros K_diss_h analogue),
    #     PAIRED to ``LatLonCGridOceanConfig.lateral_viscosity_operator`` so the EKE
    #     source is the exact energy the SELECTED viscosity operator removes:
    #       - "vector_laplacian" (default operator): the Helmholtz
    #         ``A_h·(div² + <ζ²>)`` (``vector_laplacian_dissipation_cgrid``), the
    #         KE-removal of legoESM's VECTOR-Laplacian viscosity. Energy-consistent to
    #         0.3% on the ACC spin-up; matches Veros's captured K_diss_h to ~7%.
    #       - "flux_divergence" (Veros harmonic friction; ACC recipe): the
    #         component-wise ``A_h·|∇u|² = 0.5·Σ(Δu·flux)``
    #         (``flux_divergence_viscosity_cgrid``, Veros ``calc_diss_u``/``calc_diss_v``),
    #         built from the SAME face fluxes the operator forms — Veros's EXACT EKE
    #         source for its EXACT friction.
    #     Either way ≥ 0 EVERYWHERE by construction (no clamp), vs the dynamical
    #     clamp's ~11–20% over-credit. ACC recipe opts in.
    kdiss_h_flux_form: bool = False
    # GM mean-APE -> EKE conversion source mode:
    #   "parameterized" (default) — P = kappa_GM·sigma² with sigma = <N|S|>_z(z) from
    #     the DM95-tapered, S_max-clipped, face->center->interface-averaged slope
    #     (the Visbeck-style closure; legoESM's pre-2026 behaviour, bit-identical).
    #   "realized" — the REALIZED GM-skew buoyancy conversion -P_diss_skew =
    #     -(g/ρ₀)∇ρ·F_skew (Veros veros/core/isoneutral/diffusion.py:234-281). Built
    #     from the SAME per-triad W-face slopes/tapers the GM/Redi skew flux uses
    #     (no slope pre-averaging), so it captures the per-triad slope VARIANCE
    #     <S²> ≥ <S>² that the parameterized sigma² (a squared slope AVERAGE)
    #     under-counts. Replaces the parameterized P in the EKE source; the GM
    #     tracer flux itself is unchanged. ACC recipe opts in.
    gm_source_mode: str = "parameterized"
    # Static-stability N² mode for the Eady-growth / deformation-radius chain
    # (governs eke_len via the column buoyancy integral ∫N dz and the Eady
    # production σ = N|S|). Mirrors TKEConfig.n2_mode.
    #   "insitu" (default, BIT-IDENTICAL legacy): N² from the in-situ density
    #     gradient (compute_buoyancy_frequency). Biased ~6x too stable
    #     (compressibility) → ∫N dz ~2.7x too large → eke_len ~23% too long →
    #     EKE dissipation (∝1/L) too weak (a runaway-EKE lever; the third
    #     in-situ-vs-locally-referenced bug instance after convection-N² and
    #     neutral slopes).
    #   "adiabatic": N² by adiabatic parcel displacement to the upper cell's
    #     pressure (Veros eke.py:50-54 via thermodynamics.py:99-103) — the true
    #     static stability the Veros EKE chain uses. Requires the EKE step to
    #     supply T, S, an EOS and the cell-centre pressure to displace parcels
    #     through the EOS; raises otherwise. (For "rhines" eke_len this directly
    #     corrects the deformation radius c₁ = ∫N dz / π.)
    n2_mode: str = "insitu"


def eke_mixing_length(L_rossby: jnp.ndarray, cfg: EKEConfig) -> jnp.ndarray:
    """Mixing length L = max(L_rossby, l_min) — the Rossby-radius length from the
    Visbeck machinery, floored so kappa_GM/dissipation stay well-defined. This is
    the ``mixing_length_scheme="rossby"`` length (legoESM's pre-eke_len default)."""
    return jnp.maximum(L_rossby, cfg.l_min)


# Denominator safety floor for the |f| and β reciprocals in the Rhines-limited
# mixing length — a pure numerical floor (matches Veros eke_len's ``max(·, 1e-16)``),
# exempt from the named-constant rule like the other eps floors in this module.
_DENOM_FLOOR = 1.0e-16


def eke_rhines_length(
    E: jnp.ndarray, beta: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """Eddy-energy **Rhines scale** ``L_rhines = sqrt(sqrt(E) / beta)`` [m] (Veros
    ``L_rhines``, ``veros/core/eke.py:62``).

    ``sqrt(E)`` is the eddy velocity scale [m/s] (``E`` is the specific eddy energy
    [m^2/s^2]); ``beta = df/dy`` [1/(m·s)]; ``sqrt(E)/beta`` has units m^2, so the
    result is a length. ``E`` is floored at 0 and regularised by ``+1e-30`` before
    the sqrt (finite gradient at ``E=0``); ``beta`` is floored at ``_DENOM_FLOOR``
    (matching Veros, which assumes ``beta > 0`` — true on the sphere where
    ``f = 2Ω sinφ`` ⇒ ``β = 2Ω cosφ/R ≥ 0``).
    """
    sqrt_E = jnp.sqrt(jnp.maximum(E, 0.0) + 1.0e-30)
    beta_safe = jnp.maximum(beta, _DENOM_FLOOR)
    return jnp.sqrt(sqrt_E / beta_safe)


def eke_deformation_radius(
    int_N_dz: jnp.ndarray, f_coriolis: jnp.ndarray, beta: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """First-baroclinic **Rossby deformation radius**, equatorially limited (Veros
    ``L_rossby``, ``veros/core/eke.py:54``):

        c1       = int_N_dz / pi                      # 1st-baroclinic phase speed [m/s]
        L_rossby = min( c1/|f|,  sqrt(c1/(2·beta)) )  # [m]

    ``int_N_dz = ∫N dz`` [m/s] is the column buoyancy-frequency integral (Veros's
    ``Σ √(max(0,N²))·dzw·maskW``); the ``1/pi`` is the WKB first-baroclinic factor.
    The midlatitude branch ``c1/|f|`` is the deformation radius; the equatorial
    branch ``sqrt(c1/2β)`` caps it as ``|f|→0``. ``|f|`` and ``β`` denominators are
    floored at ``_DENOM_FLOOR``; the equatorial sqrt is regularised by ``+1e-30`` for
    a finite gradient at ``c1=0`` (where the midlatitude branch is exactly 0 and wins
    the min, so the forward value is unaffected).
    """
    c1 = jnp.maximum(int_N_dz, 0.0) / jnp.pi
    f_safe = jnp.maximum(jnp.abs(f_coriolis), _DENOM_FLOOR)
    beta_safe = jnp.maximum(beta, _DENOM_FLOOR)
    L_mid = c1 / f_safe
    L_eq = jnp.sqrt(c1 / (2.0 * beta_safe) + 1.0e-30)
    return jnp.minimum(L_mid, L_eq)


def eke_len_composite(
    L_def: jnp.ndarray, L_rhines: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """Veros ``eke_len`` composite mixing length [m] (``veros/core/eke.py:63``):

        eke_len = max( l_min, min(eke_cross·L_def, eke_crhin·L_rhines) )

    The ``min`` lets the eddy-energy Rhines scale ``L_rhines`` limit the (larger)
    deformation radius ``L_def`` where eddies are weak/small; ``l_min`` floors the
    result. All inputs are lengths [m] ≥ 0.
    """
    inner = jnp.minimum(cfg.eke_cross * L_def, cfg.eke_crhin * L_rhines)
    return jnp.maximum(cfg.l_min, inner)


def eke_kappa_gm(E: jnp.ndarray, L: jnp.ndarray, cfg: EKEConfig) -> jnp.ndarray:
    """Prognostic GM coefficient ``kappa_GM = c_k * L * sqrt(E)`` (clamped >= 0 and
    <= kappa_gm_max). E is floored at 0 before the sqrt (sqrt of a tiny positive is
    used at E=0 so the gradient stays finite)."""
    E_pos = jnp.maximum(E, 0.0)
    kappa = cfg.c_k * L * jnp.sqrt(E_pos + 1.0e-30)
    return jnp.clip(kappa, 0.0, cfg.kappa_gm_max)


def eke_local_tendency(
    E: jnp.ndarray, sigma: jnp.ndarray, L: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """Local EKE source minus sink: ``P - eps`` [m^2/s^3].

    ``P = kappa_GM * sigma^2`` is the rate the GM flux converts mean available
    potential energy into eddy energy (``sigma = N|S|`` is the Eady growth rate, so
    ``kappa_GM * sigma^2 = kappa_GM * M^4/N^2``). ``eps = c_eps * E^{3/2}/L`` is the
    Eden-Greatbatch dissipation. Advection + isopycnal diffusion of E are applied
    separately (tracer machinery), so this returns ONLY the local source/sink.

    Parameters
    ----------
    E : array — eddy kinetic energy [m^2/s^2], 2-D (n_lat, n_lon) or any shape.
    sigma : array — depth-averaged Eady growth rate <N|S|>_z [1/s], same shape.
    L : array — mixing length [m] (already floored, see ``eke_mixing_length``).
    cfg : EKEConfig.
    """
    E_pos = jnp.maximum(E, 0.0)
    kappa = eke_kappa_gm(E_pos, L, cfg)
    production = kappa * sigma ** 2
    dissipation = cfg.c_eps * E_pos ** 1.5 / jnp.maximum(L, cfg.l_min)
    return production - dissipation


def eke_3d_local_tendency(
    E: jnp.ndarray, sigma: jnp.ndarray, L: jnp.ndarray, cfg: EKEConfig,
) -> jnp.ndarray:
    """Depth-resolved EKE source minus sink ``P(z) - eps(z)`` [m^2/s^3] at the
    interior interfaces (the 3-D ``eke_3d=True`` path; Veros W-grid).

    Identical functional form to the 2-D :func:`eke_local_tendency` — and delegated
    to it, since that function is shape-agnostic — but the inputs are 3-D fields on
    the interior interfaces:

    - ``P(z) = kappa_GM(z)·sigma(z)^2`` with ``kappa_GM(z) = c_k·L(z)·√E(z)`` (the
      GM mean-APE -> EKE conversion at each depth, using the LOCAL Eady growth
      ``sigma(z) = N(z)|S(z)|`` rather than its depth average);
    - ``eps(z) = c_eps·E(z)^{3/2}/L(z)`` (Eden-Greatbatch dissipation).

    All from the Stage-1 ``compute_eke_kappa_gm(depth_resolved=True)`` outputs
    ``(kappa_GM(z), sigma(z), L(z))``. The positivity regularisation (``E`` floored
    at 0 before the sqrt; ``L`` floored at ``l_min`` in the dissipation denominator)
    is exactly that of the 2-D closure. Vertical diffusion, horizontal advection and
    lateral diffusion of ``E`` are applied separately
    (:func:`eke_3d_vertical_diffusion`, :func:`eke_3d_horizontal_transport`), so this
    returns ONLY the local source/sink.

    Parameters
    ----------
    E : array (n_lat, n_lon, nlev-1) — eddy kinetic energy at interior interfaces.
    sigma : array (n_lat, n_lon, nlev-1) — LOCAL Eady growth rate N(z)|S(z)| [1/s].
    L : array (n_lat, n_lon, nlev-1) — depth-resolved mixing length [m].
    cfg : EKEConfig.
    """
    return eke_local_tendency(E, sigma, L, cfg)


def eke_apply_local_source(
    E: jnp.ndarray, sigma: jnp.ndarray, L: jnp.ndarray, cfg: EKEConfig, dt: float,
    *,
    production_override: jnp.ndarray | None = None,
    extra_source: jnp.ndarray | None = None,
    return_dissipation: bool = False,
):
    """One step of the local EKE source/sink with **semi-implicit dissipation** —
    unconditionally positivity-preserving (``E_{n+1} >= 0``) with NO clipping/mask.

    Production is explicit (``P = kappa_GM(E_n)·sigma^2 >= 0``); dissipation is
    linearised implicitly (``eps = c_eps·√E_n·E_{n+1}/L``), giving

        E_{n+1} = (E_n + dt·P) / (1 + dt·c_eps·√E_n / L)

    whose numerator is >= 0 (E_n >= 0, P >= 0) and denominator >= 1, so the result
    is >= 0 by construction (not by a floor). Advection + isopycnal diffusion of E
    are applied separately by the step (also positivity-preserving). This is the
    standard stable treatment of the quadratic-in-magnitude EKE dissipation
    (Eden-Greatbatch / Veros).

    Optional EKE-source augmentation (both default ``None`` ⇒ bit-identical):

    - ``production_override`` — replaces the parameterized GM conversion
      ``kappa_GM·sigma²`` with a supplied source [m²/s³] (the ``gm_source_mode=
      "realized"`` skew-flux conversion). ``sigma``/``L`` are then used only for the
      dissipation rate (which depends on ``L``, not ``sigma``).
    - ``extra_source`` — an additional non-negative explicit source [m²/s³] added to
      the production (the ``source_kdiss_h`` lateral-friction term, Veros
      ``K_diss_h``). Must be ≥ 0 to keep the positivity-by-construction guarantee.

    Both enter the EXPLICIT numerator, so the result stays ≥ 0 by construction
    (numerator ≥ 0, denominator ≥ 1) exactly as the base scheme.

    When ``return_dissipation`` is True, returns ``(E_new, eke_diss_iw)`` where
    ``eke_diss_iw = c_eps·√E_n·E_{n+1}/L = diss_rate·E_new`` [m²/s³] ≥ 0 is the
    EKE dissipation rate (Veros ``eke_diss_iw = c_int·eke``, ``c_int =
    eke_c_eps·√E/eke_len``) — routed to the prognostic-TKE source. The
    backward-Euler dissipation uses ``E_{n+1}`` (the implicit factor), matching
    Veros's ``c_int·eke[taup1]``. Default ⇒ returns ``E_new`` only ⇒
    bit-identical.
    """
    E_pos = jnp.maximum(E, 0.0)
    if production_override is None:
        production = eke_kappa_gm(E_pos, L, cfg) * sigma ** 2
    else:
        production = jnp.maximum(production_override, 0.0)
    if extra_source is not None:
        production = production + jnp.maximum(extra_source, 0.0)
    diss_rate = cfg.c_eps * jnp.sqrt(E_pos + 1.0e-30) / jnp.maximum(L, cfg.l_min)
    E_new = (E_pos + dt * production) / (1.0 + dt * diss_rate)
    if return_dissipation:
        # Veros eke_diss_iw = c_int·eke[taup1] = diss_rate·E_new (≥ 0).
        return E_new, diss_rate * E_new
    return E_new


def validate_eke_config(cfg: EKEConfig) -> None:
    """Fail-fast validation of EKE parameters (dispatch discipline). Raises
    ``ValueError`` on non-physical values."""
    if cfg.c_k <= 0.0:
        raise ValueError(f"EKEConfig.c_k must be > 0, got {cfg.c_k!r}")
    if cfg.c_eps <= 0.0:
        raise ValueError(f"EKEConfig.c_eps must be > 0, got {cfg.c_eps!r}")
    if cfg.l_min <= 0.0:
        raise ValueError(f"EKEConfig.l_min must be > 0, got {cfg.l_min!r}")
    if cfg.k_iso < 0.0:
        raise ValueError(f"EKEConfig.k_iso must be >= 0, got {cfg.k_iso!r}")
    if cfg.e_min < 0.0:
        raise ValueError(f"EKEConfig.e_min must be >= 0, got {cfg.e_min!r}")
    if cfg.kappa_gm_max <= 0.0:
        raise ValueError(
            f"EKEConfig.kappa_gm_max must be > 0, got {cfg.kappa_gm_max!r}"
        )
    if cfg.mixing_length_scheme not in ("rossby", "rhines"):
        raise ValueError(
            "EKEConfig.mixing_length_scheme must be 'rossby' or 'rhines', got "
            f"{cfg.mixing_length_scheme!r}"
        )
    if cfg.eke_cross <= 0.0:
        raise ValueError(f"EKEConfig.eke_cross must be > 0, got {cfg.eke_cross!r}")
    if cfg.eke_crhin <= 0.0:
        raise ValueError(f"EKEConfig.eke_crhin must be > 0, got {cfg.eke_crhin!r}")
    if cfg.alpha_eke < 0.0:
        raise ValueError(f"EKEConfig.alpha_eke must be >= 0, got {cfg.alpha_eke!r}")
    if cfg.gm_source_mode not in ("parameterized", "realized"):
        raise ValueError(
            "EKEConfig.gm_source_mode must be 'parameterized' or 'realized', got "
            f"{cfg.gm_source_mode!r}"
        )
    if cfg.kdiss_h_flux_form and not cfg.source_kdiss_h:
        # kdiss_h_flux_form selects the discretisation of the K_diss_h source; it
        # is a no-op unless the source itself is enabled. Reject the silent-ignore
        # combination rather than building an unused dissipation field.
        raise ValueError(
            "EKEConfig.kdiss_h_flux_form=True requires source_kdiss_h=True (it "
            "selects the positive-definite flux-form discretisation of the "
            "K_diss_h EKE source; with source_kdiss_h=False there is no K_diss_h "
            "source to discretise)."
        )


__all__ = [
    "EKEConfig",
    "eke_mixing_length",
    "eke_rhines_length",
    "eke_deformation_radius",
    "eke_len_composite",
    "eke_kappa_gm",
    "eke_local_tendency",
    "eke_3d_local_tendency",
    "eke_apply_local_source",
    "validate_eke_config",
]
