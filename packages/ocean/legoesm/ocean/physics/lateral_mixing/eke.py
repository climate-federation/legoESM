"""Prognostic eddy-kinetic-energy (EKE) closures.

Two selectable closures (``EKEConfig.closure``):

1. ``"eden_greatbatch"`` (default) — Eden & Greatbatch (2008), the original
   legoESM prognostic EKE (2-D specific energy or 3-D W-grid field).
2. ``"geometric"`` — the Torres et al. (2025, JAMES, 10.1029/2025MS005394)
   energetically-constrained GEOMETRIC extension (their Eqs. 1-7; the
   GEOMETRIC ``kappa_gm = alpha·∫EKE dz / ∫(M²/N) dz`` of Mak, Marshall et
   al. 2022 / D. P. Marshall et al. 2012, with EKE in place of total eddy
   energy).  Depth-INTEGRATED 2-D budget — see :class:`GeometricConfig`.

Eden & Greatbatch (2008): a 2-D (depth-integrated) eddy-energy field ``E``
whose budget is

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
scale (``eke_rhines_length``). See ``docs/ocean/fidelity/eke_len_build_spec.md``.

Defaults match Veros ACC (``eke_c_k=0.4``, ``eke_c_eps=0.5``, ``eke_lmin=100``).
"""

from __future__ import annotations

from typing import NamedTuple

import jax.numpy as jnp


__physics_contract__ = {
    "summary": (
        "Prognostic eddy-kinetic-energy closures (Eden & Greatbatch 2008 and "
        "the Mak/Marshall GEOMETRIC extension): evolve an eddy-energy field E "
        "under production (GM mean-APE -> EKE, P = kappa_GM*sigma^2), "
        "dissipation (eps = c_eps*E^{3/2}/L) and transport, and set a "
        "prognostic GM coefficient kappa_GM = c_k*L*sqrt(E)."
    ),
    "inputs": {
        "E": "m^2/s^2 (eddy kinetic energy)", "sigma": "1/s (Eady growth rate)",
        "L": "m (mixing length)", "cfg.c_k": "1", "cfg.c_eps": "1",
    },
    "outputs": {
        "kappa_GM": "m^2/s", "dE_dt_local": "m^2/s^3",
    },
    "sign_convention": (
        "E >= 0, kappa_GM >= 0 (clamped <= kappa_gm_max); production "
        "P = kappa_GM*sigma^2 >= 0 (a source; sigma = N|S| is the Eady growth "
        "rate), dissipation eps = c_eps*E^{3/2}/L >= 0 (a sink); advection + "
        "isopycnal diffusion of E are applied separately. The EKE budget has "
        "genuine sources/sinks -> nothing is conserved here; kappa_GM feeds the "
        "GM/Redi tracer redistribution."
    ),
    # Prognostic eddy-energy reservoir + coefficient producer; source/sink
    # budget, so nothing conserved by the closure itself.
    "conserves": ["none"],
    "differentiable": True,
    "reference": (
        "Eden, C. & Greatbatch, R. J. (2008), Ocean Modelling 20, 223-239; "
        "Marshall et al. (2012); Mak et al. (2022) GEOMETRIC; Torres et al. "
        "(2025) JAMES, doi:10.1029/2025MS005394"
    ),
    "idealized_test": (
        "tests/ocean/unit/test_eke.py + tests/ocean/unit/test_eke_source.py — "
        "kappa_GM grows with sqrt(E); when production balances dissipation E "
        "reaches a steady state; E floored at 0 keeps the sqrt gradient finite."
    ),
}


__param_spec__ = {
    "GeometricConfig": {
        "scheme_key": "ocean.eke.geometric",
        "excluded": {
            "e0_per_depth": "numerics: floor/cap",
            "mn_floor": "numerics: floor/cap",
        },
        "params": {
            "alpha": {"units": "1", "bounds": (0.0132, 0.12), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "c_eps_geometric": {"units": "1", "bounds": (0.00726, 0.066), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "gamma_n": {"units": "1", "bounds": (0.1155, 1.05), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "kappa_e": {"units": "m^2/s", "bounds": (165.0, 1500.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "kappa_gm_max": {"units": "m^2/s", "bounds": (4950.0, 45000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "kappa_gm_min": {"units": "m^2/s", "bounds": (3.3, 30.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "kappa_n_max": {"units": "m^2/s", "bounds": (4950.0, 45000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "kappa_n_min": {"units": "m^2/s", "bounds": (3.3, 30.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "kappa_u": {"units": "m^2/s", "bounds": (495.0, 4500.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "l_mix_max": {"units": "m", "bounds": (13200.0, 120000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "r_d_max": {"units": "m", "bounds": (13200.0, 120000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "r_d_min": {"units": "m", "bounds": (660.0, 6000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
            "rossby_factor": {"units": "1", "bounds": (0.132, 1.2), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "GEOMETRIC eddy energy (Marshall et al. 2012)", "shape": None},
        },
    },
    "EKEConfig": {
        "scheme_key": "ocean.eke.eke",
        "excluded": {
            "e_min": "numerics: floor/cap",
            "l_min": "numerics: floor/cap",
        },
        "params": {
            "alpha_eke": {"units": "1", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "prognostic EKE", "shape": None},
            "c_eps": {"units": "1", "bounds": (0.165, 1.5), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "prognostic EKE", "shape": None},
            "c_k": {"units": "1", "bounds": (0.132, 1.2), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "prognostic EKE", "shape": None},
            "eke_crhin": {"units": "1", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "prognostic EKE", "shape": None},
            "eke_cross": {"units": "1", "bounds": (0.33, 3.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "prognostic EKE", "shape": None},
            "k_iso": {"units": "m^2/s", "bounds": (330.0, 3000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "prognostic EKE", "shape": None},
            "kappa_gm_max": {"units": "m^2/s", "bounds": (3300.0, 30000.0), "tunable_tier": 2, "transform": "sigmoid", "category": "lateral_mixing", "reference": "prognostic EKE", "shape": None},
        },
    },
}


class GeometricConfig(NamedTuple):
    """Torres et al. (2025, JAMES, doi:10.1029/2025MS005394) GEOMETRIC
    mesoscale-EKE closure parameters — the authors' hand-calibrated values
    (their OMIP2 EKE-GM / EKE-GM+N experiments with structure function
    ``phi(z) = 1``).  All defaults carry page/equation provenance from the
    paper.  These are TUNABLE closure parameters (the explicit targets of a
    future differentiable calibration), so they live here in the scheme
    config — NOT in ``legoesm.constants`` — and every formula below accepts
    them as traced JAX values (no Python control flow on their magnitudes).

    The prognostic variable of this closure is the DEPTH-INTEGRATED eddy
    kinetic energy ``∫EKE dz`` [m³/s²] (paper Eq. 1, p. 4):

        ∂t∫E dz + ∇h·(u_h ∫E dz) = B_C + B_T − D_e + T_e

    with (phi(z) = 1 throughout; the paper's optional 3-D structure-function
    variant EKE-GM+N3D reads a static netCDF mode map and is NOT implemented):

      B_C = kappa_gm·∫M⁴/N² dz       (Eq. 2, p. 4; = kappa_gm·∫(N|S|)² dz)
      B_T = kappa_u·∫|∇h u_h|² dz    (Eq. 3, p. 4)
      D_e = (C_eps/R_d)·∫EKE^{3/2} dz  (Eq. 4, p. 5; with phi=1 this is
                                        C_eps·(∫E dz)^{3/2}/(R_d·√H))
      T_e = kappa_e·∇²h ∫E dz        (Eq. 5, p. 5)
      kappa_gm = alpha·∫E dz / max(∫M²/N dz, mn_floor)   (Eq. 6, p. 6)
      kappa_n  = gamma_n·min(R_d, l_mix_max)·√(2·∫E dz/H)  (Eq. 7, p. 6)
      R_d = rossby_factor·∫N dz/|f|, clipped to [r_d_min, r_d_max]
                                      (Appendix D, p. 32)
    """

    # Eddy efficiency alpha (Eq. 6, p. 6): kappa_gm = alpha·∫EKE dz/∫(M²/N)dz.
    # Calibrated 0.04 (Sect. 2.1.2 / Appendix E; NOT bounded by 1, since the
    # paper's alpha uses EKE only — distinct from Marshall et al. 2012's
    # alpha_geom ≤ 1 for the TOTAL eddy energy, Appendix D p. 31-32).
    alpha: float = 0.04
    # Dissipation coefficient C_eps (Eq. 4, p. 5): calibrated 0.022 with
    # phi(z)=1 (0.013 for the N3D structure-function run, Sect. 2.3.2 p. 8;
    # plausible literature range 0.001-0.1, Table E1).  Named distinctly from
    # the Eden-Greatbatch ``EKEConfig.c_eps`` (different closure, different
    # dimensional role).
    c_eps_geometric: float = 0.022
    # Eddy momentum diffusivity kappa_u [m²/s] for the barotropic production
    # B_T (Eq. 3, p. 4): calibrated 1500 to match the equatorial
    # domain-integrated EKE (Appendix E p. 38; range 500-5000 sampled).
    kappa_u: float = 1500.0
    # EKE lateral-diffusion coefficient kappa_E [m²/s] (Eq. 5, p. 5): 500,
    # kept from GEOMETRIC/Mak et al. (2018) (low sensitivity, Fig. E2-E3).
    kappa_e: float = 500.0
    # Rossby-radius prefactor: R_d = rossby_factor·∫N dz/|f| (Appendix D,
    # p. 32).  0.4 in the paper's NEMO v3.6 (replaced by 0.5 in NEMO v4);
    # NOTE this is the paper's chosen consistency factor, not the WKB 1/pi.
    rossby_factor: float = 0.4
    # Dissipation/mixing length-scale bounds on R_d [m]: "the dissipation
    # length scale is bounded between 2 and 40 km" (Appendix D, p. 32).
    r_d_min: float = 2.0e3
    r_d_max: float = 4.0e4
    # Lower bound on the kappa_gm denominator ∫M²/N dz [m/s] (Eq. 6, p. 6):
    # "∫M²/N dz is lower-bounded to 10⁻¹⁰" (avoids division by zero where
    # isopycnals are flat).
    mn_floor: float = 1.0e-10
    # kappa_gm bounds [m²/s].  The paper's experiments floor at 10 (Table 1,
    # p. 8: min-max 10-12,716) and RELAX the DEFAULT run's 1000 upper cap
    # (Sect. 2.3.2, p. 7) — no upper cap is STATED.  legoESM defaults the cap
    # to 1.5e4: ABOVE the paper's realized maximum (12,716, Table 1), so it
    # would not have bound in their eORCA1 experiments, but finite because
    # Eq. 6 is otherwise unbounded where ∫M²/N dz collapses to its 1e-10
    # floor (flat isopycnals — e.g. an idealized cold start), where it
    # produced kappa ~ alpha·∫E/1e-10 ~ 1e6 m²/s > the explicit GM CFL limit
    # (probe: ACC blowup at day ~200; .physics-validator/geometric_build/).
    # A legoESM forward-stability safety, calibration-tunable.
    kappa_gm_min: float = 10.0
    kappa_gm_max: float = 1.5e4
    # --- kappa_n (neutral/Redi diffusivity) coupling, Eq. 7 (p. 6) ---
    # When True, the Redi tracer isopycnal diffusivity follows
    # kappa_n = gamma_n·L_mix·√(2·EKE_0) (the paper's EKE-GM+N experiment);
    # when False, the Redi diffusivity is left at the GM/Redi config value
    # (the paper's EKE-GM experiment).
    kappa_n_coupling: bool = False
    # Mixing efficiency Gamma = 0.35 (Eq. 7, p. 6; Groeskamp et al. 2020;
    # literature range 0.1-0.5, Table E1).
    gamma_n: float = 0.35
    # L_mix cap [m]: "L_mix is defined as the local Rossby radius R_d but
    # capped at 40 km to avoid singularity and large values near the
    # equator" (Sect. 2.1.2, p. 6).
    l_mix_max: float = 4.0e4
    # kappa_n bounds [m²/s]: floor 10 per Table 1 (p. 8: min-max 10-5,291 in
    # EKE-GM+N, up to 10,604 in N3D); no paper cap stated — legoESM defaults
    # 1.5e4 (above the realized range; the same forward-stability rationale
    # as kappa_gm_max, since the explicit Redi triads share the GM CFL
    # bound).  Calibration-tunable.
    kappa_n_min: float = 10.0
    kappa_n_max: float = 1.5e4
    # Cold-start initial condition: ∫EKE dz |_{t=0} = e0_per_depth·H [m³/s²]
    # ("setting the small value of 10⁻⁶·h (in m³/s²) to each cell, where h
    # denotes the depth of the water column", Appendix E, p. 35).
    e0_per_depth: float = 1.0e-6


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
    #   "realized_signed" — the LITERAL SIGNED Veros conversion -P_diss_skew =
    #     -(g/ρ₀)·∇(int_drhodX)·F_skew summed over X∈{T,S} (the dynamic-enthalpy
    #     dissipation of the GM SKEW flux; veros/core/isoneutral/diffusion.py:234-263,
    #     compute_dissipation + the vertical flux_top term). Built from the SKEW-only
    #     isopycnal flux (kappa_Redi=0) the GM tracer tendency assembles, contracted
    #     with the Veros int_drhodT/S dynamic-enthalpy integrands — NOT the positive-
    #     definite κ_GM·N²·⟨S²⟩ parameterization of "realized". On the ACC equilibrium
    #     it is ≥ 0 in every wet cell (matching Veros), but it is NOT clamped: any
    #     locally-negative value is folded SEMI-IMPLICITLY by eke_apply_local_source
    #     as a local sink (like dissipation), keeping E ≥ e_min by construction. This
    #     removes the ~22% positive-definite over-count of "realized" (probe: signed
    #     5.94e10 W vs parameterized 7.29e10 W vs Veros 5.97e10 W on the drop snapshot).
    #     Requires eke_3d=True (3-D W-grid source). ACC recipe opts in.
    gm_source_mode: str = "parameterized"
    # When True (default off ⇒ bit-identical), SUBTRACT the realized SIGNED Redi
    # (isopycnal-diffusive) APE dissipation -P_diss_iso from the EKE source — Veros's
    # P_diss_iso sink (veros/core/eke.py:117; -P_diss_iso term of the EKE forc).
    # Built analogously to the signed skew (the ISO-only flux, kappa_GM=0, plus the
    # implicit K_33 vertical-diagonal dissipation) contracted with the dynamic-
    # enthalpy gradient. Requires eke_3d=True AND gm_source_mode="realized_signed"
    # (the signed Redi sink only makes sense paired with the signed skew source —
    # both are the literal Veros conversions; mixing the parameterized skew with the
    # signed iso sink would be inconsistent). ACC recipe opts into both.
    source_p_diss_iso: bool = False
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
    # Veros dzw slot for the ADIABATIC N² divisor (the deferred EKE-side
    # twin of ``TKEConfig.veros_dz_slots``; only consulted with
    # ``n2_mode="adiabatic"``): divide the adiabatic density contrast by the
    # ACTUAL centre spacing ``dz_half_ref·J`` (Veros ``dzw``,
    # thermodynamics.py:99) instead of the midpoint reconstruction
    # ``0.5·(dzt_k+dzt_{k+1})·J``. On a Veros u_centered coordinate the two
    # alternate by up to ±50% per level. Default False ⇒ BIT-IDENTICAL.
    # --- Closure dispatch (appended LAST: positional construction stable) ---
    # "eden_greatbatch" (default, bit-identical legacy): everything above.
    # "geometric": the Torres et al. (2025) GEOMETRIC depth-integrated EKE
    #   budget (requires ``geometric`` below; 2-D only — eke_3d must be False;
    #   the EG-specific knobs c_k/c_eps/mixing_length/source-augmentation are
    #   unused).  The prognostic ``state.eke`` field then carries ∫EKE dz
    #   [m³/s²] (depth-integrated), NOT the EG specific energy [m²/s²].
    closure: str = "eden_greatbatch"
    # GEOMETRIC closure parameters (Torres et al. 2025); must be a
    # :class:`GeometricConfig` when ``closure="geometric"`` and None otherwise
    # (a set-but-unused GeometricConfig is rejected — no silent ignoring).
    geometric: GeometricConfig | None = None
    # Veros dzw slot for the ADIABATIC N² divisor (the deferred EKE-side
    # twin of ``TKEConfig.veros_dz_slots``; only consulted with
    # ``n2_mode="adiabatic"``; appended last — positional construction
    # stable): divide the adiabatic density contrast by the ACTUAL centre
    # spacing ``dz_half_ref·J`` (Veros ``dzw``, thermodynamics.py:99)
    # instead of the midpoint reconstruction ``0.5·(dzt_k+dzt_{k+1})·J``.
    # On a Veros u_centered coordinate the two alternate by up to ±50% per
    # level. Default False ⇒ BIT-IDENTICAL.
    n2_over_dzw: bool = False


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
    signed_source: jnp.ndarray | None = None,
    clamp_production: bool = True,
    return_dissipation: bool = False,
    production_scale: jnp.ndarray | None = None,
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

    Optional EKE-source augmentation (all default ``None``/``True`` ⇒ bit-identical):

    - ``production_override`` — replaces the parameterized GM conversion
      ``kappa_GM·sigma²`` with a supplied source [m²/s³] (the ``gm_source_mode=
      "realized"`` skew-flux conversion). ``sigma``/``L`` are then used only for the
      dissipation rate (which depends on ``L``, not ``sigma``).
    - ``extra_source`` — an additional non-negative explicit source [m²/s³] added to
      the production (the ``source_kdiss_h`` lateral-friction term, Veros
      ``K_diss_h``). Must be ≥ 0 to keep the positivity-by-construction guarantee.
    - ``signed_source`` — a SIGN-INDEFINITE source/sink [m²/s³] (the realized SIGNED
      ``-P_diss_skew`` source and/or the ``-P_diss_iso`` Redi sink, summed by the
      caller).  Positivity is preserved by SPLITTING it: the non-negative part enters
      the explicit numerator, and the magnitude of the NEGATIVE part is folded into
      the implicit denominator as an extra linear sink rate ``max(0,-signed)/E_n``
      (backward-Euler on the negative-source magnitude), so ``E_{n+1} ≥ 0`` by
      construction even where the conversion is locally a sink.  This is the
      documented treatment vs Veros's plain EXPLICIT ``forc`` (Veros relies on the
      ``E ≥ 0`` re-floor; legoESM folds the local sink semi-implicitly so no floor
      is needed and the step is unconditionally stable).
    - ``clamp_production`` (default True) — when False, a ``production_override`` is
      NOT clamped to ``≥ 0`` but routed through the same negative-part-implicit split
      as ``signed_source`` (the ``gm_source_mode="realized_signed"`` path: the signed
      skew conversion can be locally negative).
    - ``production_scale`` — optional non-negative multiplier applied to the
      internally PARAMETERIZED GM production ``kappa_GM·sigma²`` ONLY (the
      Hallberg resolution-function budget coupling, codex MED-3 r2: the tracer
      path applies ``kappa_eff = f_res·kappa_GM``, so the eddy-energy budget
      must receive the SAME ``f_res``-scaled APE→EKE conversion the applied
      coefficient performs).  Must broadcast against the production; it is
      clamped ``≥ 0`` internally (like ``extra_source``) so positivity-by-
      construction holds unconditionally (``f_res ∈ (0, 1]`` passes through
      untouched).  Ignored when ``production_override`` is supplied — callers
      building an override fold any scaling into the override themselves
      (GEOMETRIC scales B_C only, not B_T; realized modes scale via the kappa
      they hand to the conversion builders).  ``None`` (default) ⇒
      bit-identical.

    When ``return_dissipation`` is True, returns ``(E_new, eke_diss_iw)`` where
    ``eke_diss_iw = c_eps·√E_n·E_{n+1}/L = diss_rate·E_new`` [m²/s³] ≥ 0 is the
    Eden-Greatbatch dissipation rate (Veros ``eke_diss_iw = c_int·eke``, ``c_int =
    eke_c_eps·√E/eke_len``) — routed to the prognostic-TKE source.  It is the
    Eden-Greatbatch interior dissipation ONLY (NOT the realized-source negative part,
    which is folded into the same implicit factor but is the GM/Redi APE conversion,
    not the IW-breaking dissipation Veros routes to TKE).  Default ⇒ returns
    ``E_new`` only ⇒ bit-identical.
    """
    E_pos = jnp.maximum(E, 0.0)
    # Accumulate the EXPLICIT positive production and the IMPLICIT extra sink rate
    # (from sign-indefinite sources' negative part) so the result stays ≥ 0.
    # ``None`` sentinel (not a 0.0 scalar): the default-off path traces the exact
    # pre-existing ``1 + dt·diss_rate`` denominator with no extra add and no
    # scalar-dtype interaction — bit-identical by construction.
    extra_sink_rate = None
    if production_override is None:
        production = eke_kappa_gm(E_pos, L, cfg) * sigma ** 2
        if production_scale is not None:
            # Resolution-function budget coupling: the SAME f_res ∈ (0, 1]
            # the GM/Redi tracer flux applies to kappa_GM scales the
            # parameterized APE→EKE conversion.  Clamped ≥ 0 (the same
            # hardening as ``extra_source``) so the headline positivity-by-
            # construction guarantee holds UNCONDITIONALLY even for a rogue
            # negative scale (codex MED-3 r2 NIT); f_res ∈ (0, 1] is
            # untouched by the clamp.
            production = production * jnp.maximum(production_scale, 0.0)
    elif clamp_production:
        production = jnp.maximum(production_override, 0.0)
    else:
        # Signed override: positive part explicit, negative part implicit.
        production = jnp.maximum(production_override, 0.0)
        neg = jnp.maximum(-production_override, 0.0)        # ≥ 0 sink magnitude
        extra_sink_rate = neg / jnp.maximum(E_pos, 1.0e-30)
    if extra_source is not None:
        production = production + jnp.maximum(extra_source, 0.0)
    if signed_source is not None:
        production = production + jnp.maximum(signed_source, 0.0)
        neg = jnp.maximum(-signed_source, 0.0)
        sink = neg / jnp.maximum(E_pos, 1.0e-30)
        extra_sink_rate = sink if extra_sink_rate is None else extra_sink_rate + sink
    diss_rate = cfg.c_eps * jnp.sqrt(E_pos + 1.0e-30) / jnp.maximum(L, cfg.l_min)
    # Both rates are ≥ 0; the implicit denominator ≥ 1 ⇒ E_new ≥ 0 by construction.
    # ``diss_rate`` itself stays the pure Eden-Greatbatch rate — the
    # ``return_dissipation`` product below must exclude the extra sink folding.
    total_rate = (diss_rate if extra_sink_rate is None
                  else diss_rate + extra_sink_rate)
    E_new = (E_pos + dt * production) / (1.0 + dt * total_rate)
    if return_dissipation:
        # Veros eke_diss_iw = c_int·eke[taup1] = diss_rate·E_new (≥ 0) — the IW
        # dissipation ONLY (excludes the realized-source negative-part folding).
        return E_new, diss_rate * E_new
    return E_new


# ---------------------------------------------------------------------------
# GEOMETRIC closure (Torres et al. 2025, JAMES, doi:10.1029/2025MS005394) —
# pure, shape-agnostic formulas.  The prognostic variable is the
# depth-integrated ∫EKE dz [m³/s²]; column integrals (∫M⁴/N² dz, ∫M²/N dz,
# ∫N dz, H) come from the SHARED GM/Redi Eady machinery
# (``_gm_redi_common.compute_geometric_column_integrals``) — this module never
# recomputes N²/slopes.  All GeometricConfig parameters are usable as traced
# JAX values (calibration-ready: pure jnp arithmetic, no Python branching on
# parameter magnitudes).
# ---------------------------------------------------------------------------


def geometric_rossby_radius(
    int_N_dz: jnp.ndarray, f_coriolis: jnp.ndarray, geom: GeometricConfig,
) -> jnp.ndarray:
    """Local Rossby deformation radius of the GEOMETRIC closure (Torres et al.
    2025, Appendix D, p. 32):

        R_d = rossby_factor · ∫N dz / |f|,   clipped to [r_d_min, r_d_max]

    with ``rossby_factor = 0.4`` (the paper's NEMO v3.6 consistency value;
    0.5 in NEMO v4) and the bounds 2-40 km ("the dissipation length scale is
    bounded between 2 and 40 km").  Used as BOTH the dissipation length of
    D_e (Eq. 4) and — capped again at ``l_mix_max`` — the kappa_n mixing
    length (Eq. 7).  ``|f|`` is floored like the other reciprocals in this
    module; the clip makes the equatorial limit benign anyway.
    """
    f_safe = jnp.maximum(jnp.abs(f_coriolis), _DENOM_FLOOR)
    r_d = geom.rossby_factor * jnp.maximum(int_N_dz, 0.0) / f_safe
    return jnp.clip(r_d, geom.r_d_min, geom.r_d_max)


def geometric_kappa_gm(
    int_E: jnp.ndarray, int_M2_over_N_dz: jnp.ndarray, geom: GeometricConfig,
) -> jnp.ndarray:
    """GEOMETRIC GM coefficient (Torres et al. 2025, Eq. 6, p. 6):

        kappa_gm = alpha · ∫EKE dz / max(∫M²/N dz, mn_floor)

    clipped to ``[kappa_gm_min, kappa_gm_max]`` (paper: floor 10 m²/s per
    Table 1; no upper cap — the DEFAULT run's 1000 m²/s cap is relaxed).
    ``kappa_gm`` is depth-CONSTANT (2-D): the paper tested a phi(z)² vertical
    structure and rejected it (Appendix D, p. 33, Fig. D4).

    ``int_E = ∫EKE dz`` [m³/s²] (floored at 0 — the budget keeps it ≥ 0 by
    construction, the floor only guards round-off); ``int_M2_over_N_dz =
    ∫M²/N dz = ∫N|S| dz`` [m/s] from the shared slope/N machinery.
    """
    denom = jnp.maximum(int_M2_over_N_dz, geom.mn_floor)
    kappa = geom.alpha * jnp.maximum(int_E, 0.0) / denom
    return jnp.clip(kappa, geom.kappa_gm_min, geom.kappa_gm_max)


def geometric_kappa_n(
    int_E: jnp.ndarray, H_col: jnp.ndarray, r_d: jnp.ndarray,
    geom: GeometricConfig,
) -> jnp.ndarray:
    """GEOMETRIC neutral (Redi) diffusivity (Torres et al. 2025, Eq. 7, p. 6),
    with structure function ``phi(z) = 1`` (depth-constant kappa_n):

        kappa_n = Gamma · L_mix · sqrt(2·EKE_0),
        L_mix   = min(R_d, l_mix_max)            (40 km cap, Sect. 2.1.2)
        EKE_0   = ∫EKE dz / ∫phi² dz = ∫EKE dz / H   (phi = 1)

    clipped to ``[kappa_n_min, kappa_n_max]`` (floor 10 m²/s per Table 1).
    ``H_col`` is the wet column depth [m]; dry columns (H=0) are guarded by
    the eps floor and must be masked by the caller (the wet-column mask from
    the shared machinery), exactly like the EG ``kappa_GM``.
    """
    l_mix = jnp.minimum(r_d, geom.l_mix_max)
    eke0 = jnp.maximum(int_E, 0.0) / jnp.maximum(H_col, _DENOM_FLOOR)
    # +1e-30 inside the sqrt: finite gradient at EKE_0 = 0 (the same
    # regularisation as eke_kappa_gm / eke_rhines_length).
    kappa = geom.gamma_n * l_mix * jnp.sqrt(2.0 * eke0 + 1.0e-30)
    return jnp.clip(kappa, geom.kappa_n_min, geom.kappa_n_max)


def geometric_dissipation_length(
    r_d: jnp.ndarray, H_col: jnp.ndarray,
) -> jnp.ndarray:
    """Effective dissipation length ``L_eff = R_d·√H`` [m·√m] that maps the
    GEOMETRIC dissipation (Torres et al. 2025, Eq. 4, p. 5) onto the shared
    semi-implicit sink fold of :func:`eke_apply_local_source`.

    With ``phi(z) = 1`` the depth-integrated dissipation of Eq. 4 is

        D_e = (C_eps/R_d)·∫EKE^{3/2} dz = (C_eps/R_d)·H·(I/H)^{3/2}
            = C_eps·√(I/H)/R_d · I        with  I = ∫EKE dz,

    i.e. a linear-in-I sink with rate ``r = C_eps·√(I/H)/R_d`` [1/s].  The
    shared fold computes ``rate = c_eps·√I/max(L, l_min)``, so passing
    ``L = R_d·√H`` (and ``c_eps = c_eps_geometric``) reproduces ``r``
    EXACTLY — the established unconditionally-positive backward-Euler
    treatment (E_{n+1} = (E_n + dt·P)/(1 + dt·r) ≥ 0 by construction).  The
    paper instead zeroes D_e wherever EKE < 0 (p. 5); the implicit fold is
    strictly stronger (E never goes negative in the first place) and is the
    module's standard treatment — a documented numerical-treatment deviation,
    not a physics one.

    NOTE the ``l_min`` floor of the fold (default 100 m) never binds here:
    ``L_eff ≥ r_d_min·√H ≥ 2000·√H`` for any wet column.
    """
    return r_d * jnp.sqrt(jnp.maximum(H_col, 0.0) + 1.0e-30)


def validate_geometric_config(geom: GeometricConfig) -> None:
    """Fail-fast validation of the GEOMETRIC parameters (dispatch
    discipline).  Raises ``ValueError`` on non-physical values."""
    if geom.alpha <= 0.0:
        raise ValueError(f"GeometricConfig.alpha must be > 0, got {geom.alpha!r}")
    if geom.c_eps_geometric <= 0.0:
        raise ValueError(
            f"GeometricConfig.c_eps_geometric must be > 0, got "
            f"{geom.c_eps_geometric!r}")
    if geom.kappa_u < 0.0:
        raise ValueError(
            f"GeometricConfig.kappa_u must be >= 0, got {geom.kappa_u!r}")
    if geom.kappa_e < 0.0:
        raise ValueError(
            f"GeometricConfig.kappa_e must be >= 0, got {geom.kappa_e!r}")
    if geom.rossby_factor <= 0.0:
        raise ValueError(
            f"GeometricConfig.rossby_factor must be > 0, got "
            f"{geom.rossby_factor!r}")
    if not (0.0 < geom.r_d_min <= geom.r_d_max):
        raise ValueError(
            "GeometricConfig requires 0 < r_d_min <= r_d_max, got "
            f"r_d_min={geom.r_d_min!r}, r_d_max={geom.r_d_max!r}")
    if geom.mn_floor <= 0.0:
        raise ValueError(
            f"GeometricConfig.mn_floor must be > 0, got {geom.mn_floor!r}")
    if not (0.0 <= geom.kappa_gm_min <= geom.kappa_gm_max):
        raise ValueError(
            "GeometricConfig requires 0 <= kappa_gm_min <= kappa_gm_max, got "
            f"kappa_gm_min={geom.kappa_gm_min!r}, "
            f"kappa_gm_max={geom.kappa_gm_max!r}")
    if geom.gamma_n <= 0.0:
        raise ValueError(
            f"GeometricConfig.gamma_n must be > 0, got {geom.gamma_n!r}")
    if geom.l_mix_max <= 0.0:
        raise ValueError(
            f"GeometricConfig.l_mix_max must be > 0, got {geom.l_mix_max!r}")
    if not (0.0 <= geom.kappa_n_min <= geom.kappa_n_max):
        raise ValueError(
            "GeometricConfig requires 0 <= kappa_n_min <= kappa_n_max, got "
            f"kappa_n_min={geom.kappa_n_min!r}, "
            f"kappa_n_max={geom.kappa_n_max!r}")
    if geom.e0_per_depth < 0.0:
        raise ValueError(
            f"GeometricConfig.e0_per_depth must be >= 0, got "
            f"{geom.e0_per_depth!r}")


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
    if cfg.gm_source_mode not in ("parameterized", "realized", "realized_signed"):
        raise ValueError(
            "EKEConfig.gm_source_mode must be 'parameterized', 'realized', or "
            f"'realized_signed', got {cfg.gm_source_mode!r}"
        )
    if cfg.source_p_diss_iso and cfg.gm_source_mode != "realized_signed":
        # The signed Redi (-P_diss_iso) sink is only consistent paired with the
        # signed skew source (both are the literal Veros conversions). Reject the
        # mismatched combination rather than silently mixing parameterized skew
        # with a signed iso sink.
        raise ValueError(
            "EKEConfig.source_p_diss_iso=True requires gm_source_mode="
            "'realized_signed' (the signed -P_diss_iso Redi sink is only "
            "consistent with the signed -P_diss_skew source; got gm_source_mode="
            f"{cfg.gm_source_mode!r})."
        )
    if cfg.closure not in ("eden_greatbatch", "geometric"):
        raise ValueError(
            "EKEConfig.closure must be 'eden_greatbatch' or 'geometric', got "
            f"{cfg.closure!r}"
        )
    if cfg.closure == "geometric":
        if cfg.geometric is None:
            raise ValueError(
                "EKEConfig.closure='geometric' requires EKEConfig.geometric "
                "(a GeometricConfig with the Torres et al. 2025 parameters); "
                "got None."
            )
        validate_geometric_config(cfg.geometric)
        if cfg.eke_3d:
            raise ValueError(
                "EKEConfig.closure='geometric' is the depth-INTEGRATED 2-D "
                "budget (Torres et al. 2025 Eq. 1) and requires eke_3d=False; "
                "the 3-D W-grid path is Eden-Greatbatch only."
            )
        if cfg.isopycnal_diffusion:
            raise ValueError(
                "EKEConfig.closure='geometric' with isopycnal_diffusion=True "
                "is ambiguous: the GEOMETRIC Redi coupling is kappa_n (Eq. 7, "
                "Torres et al. 2025), selected by "
                "GeometricConfig.kappa_n_coupling=True — not the Veros "
                "K_iso=K_gm flag. Set isopycnal_diffusion=False."
            )
        if (cfg.source_kdiss_h or cfg.kdiss_h_flux_form
                or cfg.gm_source_mode != "parameterized"
                or cfg.source_p_diss_iso):
            raise ValueError(
                "EKEConfig.closure='geometric' has its own source terms "
                "(B_C Eq. 2 + B_T Eq. 3, Torres et al. 2025); the "
                "Eden-Greatbatch source-augmentation flags (source_kdiss_h="
                f"{cfg.source_kdiss_h!r}, kdiss_h_flux_form="
                f"{cfg.kdiss_h_flux_form!r}, gm_source_mode="
                f"{cfg.gm_source_mode!r}, source_p_diss_iso="
                f"{cfg.source_p_diss_iso!r}) must stay at their defaults."
            )
    elif cfg.geometric is not None:
        raise ValueError(
            "EKEConfig.geometric is set but closure="
            f"{cfg.closure!r} — it would be silently ignored. Set "
            "closure='geometric' or drop the GeometricConfig."
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
    "GeometricConfig",
    "geometric_rossby_radius",
    "geometric_kappa_gm",
    "geometric_kappa_n",
    "geometric_dissipation_length",
    "validate_geometric_config",
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
