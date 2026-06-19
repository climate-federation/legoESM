"""Reusable MITgcm ocean-numerics recipe card + the MITgcm<->legoESM wiring diagram.

Like ``nemo_recipe.py`` / the ``veros_*`` cards, this is a PURE CONFIG layer: it
selects the canonical legoESM blocks that reproduce MITgcm's hydrostatic z-coordinate
ocean numerics.  It contains NO MITgcm-specific solver, grid, bridge, forcing reader,
or state translation — those mimicry-only pieces live in the fidelity harness
(``mitgcm_io``/``mitgcm_runner``/``mitgcm_state_bridge``), never in the model
(oracle-recipe doctrine, docs/ocean_fidelity/oracle_recipe_strategy.md).

The mapping below was established by TERM-BY-TERM validation against MITgcm's own
momentum diagnostics at a bit-imported state (``scripts/tmp/_bgyre_operator_match.py``)
plus a direct read/audit of the MITgcm Fortran:
  - advection (``Um_Advec``) and Coriolis (``Um_Cori``): corr 1.0000 — bit-faithful.
  - the free surface: MITgcm ``implicitFreeSurface`` is UNSPLIT (audited: no barotropic
    sub-cycle, one CG2D solve for eta, no barotropic-velocity prognostic, uniform
    surface-pressure correction) -> ``barotropic_solver="implicit_unsplit"``
    (docs/ocean_fidelity/mitgcm_unsplit_freesurface_fix.md).
The 5 tutorial recipes (``mitgcm_{barotropic_gyre,baroclinic_gyre,front_relax,
advection_gyre,reentrant_channel}_recipe.py``) are SETUPS that pair this card's block
choices with a specific MITgcm-tutorial domain + forcing.
"""

from __future__ import annotations

from legoesm.ocean.constants_config import ConstantsConfig
from legoesm.ocean.state import LatLonCGridOceanConfig

# MITgcm tutorial defaults: gravity=9.81, rhoNil=999.8 (Boussinesq reference density).
MITGCM_CONSTANTS_CONFIG = ConstantsConfig(g=9.81, rho_0=999.8)


# --- The MITgcm <-> legoESM WIRING DIAGRAM ---------------------------------------
# (MITgcm option / Fortran term, legoESM config field, chosen block + validation note).
# This is the auditable map a user reads to see exactly which legoESM block each MITgcm
# numeric corresponds to, and how faithful it is.
MITGCM_BLOCK_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("constants (rhoNil, gravity)", "ConstantsConfig",
     "MITGCM_CONSTANTS_CONFIG: rho_0=999.8, g=9.81"),
    ("EOS (eosType='LINEAR', tAlpha, sBeta)", "eos + eos_linear",
     "eos='linear', LinearEOSConfig(alpha_T, beta_S) — tAlpha/sBeta direct"),
    ("momentum advection (mom_fluxform, default)", "momentum_advection",
     "flux_form — matches Um_Advec corr 1.0000 (incl. metric terms)"),
    ("momentum flux reconstruction (advScheme=2 centered)", "momentum_flux_scheme",
     "centered"),
    ("KE gradient", "ke_gradient_scheme",
     "n/a in flux_form — the flux-form advection IS -div(uu); the separate KE "
     "gradient is correctly dropped (ocean_pe_latlon_cgrid.py:3180)"),
    ("Coriolis (mom_u/v_coriolis, 4-pt face-f average)", "coriolis_scheme",
     "explicit_ab2 (face-f) — matches Um_Cori corr 1.0000; planetary f x u in du_dt"),
    ("hydrostatic PGF (calc_phi_hyd, integr_GeoPot=2, FS load excluded)",
     "baroclinic PGF (iterate_eos_and_pressure_anomaly)",
     "energy-conserving FD center-staggered integral; baroclinic part explicit/AB2"),
    ("free surface (implicitFreeSurface, implicSurfPress=1, UNSPLIT)",
     "barotropic_solver",
     "implicit_unsplit (theta=1) — one 3D predictor + one elliptic eta solve + "
     "uniform correction, NO barotropic/baroclinic mode split (AUDITED)"),
    ("tracer advection (tempAdvScheme=2 centered)", "tracer_advection", "centered"),
    ("lateral viscosity (viscAh, useStrainTensionVisc=.FALSE.)",
     "A_h + lateral_viscosity_operator",
     "flux_divergence (component del2, matches MITgcm's per-component harmonic friction)"),
    ("lateral tracer diffusion (diffKhT)", "K_h", "Laplacian K_h * laplacian_cgrid"),
    ("vertical viscosity/diffusion (viscAr, diffKrT, implicitDiffusion)",
     "A_v, K_v + implicit_vertical_mixing",
     "implicit backward-Euler vertical mixing"),
    ("convective adjustment (ivdc_kappa)", "physics.convection",
     "enhanced_diffusion with K_conv=ivdc_kappa"),
    ("GM/Redi eddy parameterisation (pkg/gmredi)", "gm_redi",
     "GMRediConfig (kappa_GM/Redi, dm95/gkw91 taper, S_max) — threaded through the "
     "unsplit step (implicit_unsplit supports GM/Redi; prognostic-EKE GM is a TODO)"),
    ("surface restoring (tauThetaClimRelax, RBCS)",
     "physics.surface_forcing / sponge",
     "RestoringConfig (surface) / SpongeForcing (RBCS interior)"),
    ("wind stress (zonalWindFile)", "surface_forcing (OceanSurfaceForcing.tau_x)",
     "external wind stress (sign-flipped: legoESM tau is the atmospheric stress)"),
    ("time integration (Adams-Bashforth, abEps)", "outer_integrator + ab2_epsilon",
     "ab2 with ab2_epsilon=abEps; momDissip_In_AB=T == ab2_scope='total'"),
    ("lateral BC (no_slip_sides=.TRUE.)", "lateral_side_bc", "no_slip"),
)


def mitgcm_canonical_ocean_config(
    *,
    A_h: float,    # noqa: N803 (matches the config field name)
    K_h: float,    # noqa: N803
    A_v: float,    # noqa: N803
    K_v: float,    # noqa: N803
    eos_linear,
    physics,
    **overrides,
) -> LatLonCGridOceanConfig:
    """Return the SHARED MITgcm-faithful ``LatLonCGridOceanConfig`` block choices
    (the numerics common to every hydrostatic-ocean MITgcm tutorial), per
    :data:`MITGCM_BLOCK_MAPPING`.  The per-setup domain/forcing/dimensional knobs
    (``A_h``, ``K_h``, ``A_v``, ``K_v``, the linear-EOS coefficients, the physics
    block) are arguments; ``overrides`` patches any remaining field.

    The 5 tutorial recipes select these same blocks (currently inline); this helper
    is the one auditable place the MITgcm card's choices live.
    """
    cfg = LatLonCGridOceanConfig(
        constants=MITGCM_CONSTANTS_CONFIG,
        eos="linear",
        eos_linear=eos_linear,
        # momentum: MITgcm flux-form, centered (mom_fluxform default).
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        tracer_advection="centered",
        # Coriolis: explicit_ab2 face-f (matches Um_Cori corr 1.0).
        coriolis_scheme="explicit_ab2",
        coriolis_energy_conserving=False,
        outer_integrator="ab2",
        ab2_scope="total",
        # UNSPLIT implicit free surface (MITgcm implicitFreeSurface, theta=1).
        barotropic_solver="implicit_unsplit",
        barotropic_implicit_theta_eta=1.0,
        barotropic_implicit_theta_pgf=1.0,
        # lateral viscosity = component del2 (Veros/MITgcm harmonic friction).
        A_h=A_h,
        lateral_viscosity_operator="flux_divergence",
        lateral_side_bc="no_slip",
        K_h=K_h,
        # vertical mixing implicit (viscAr/diffKrT, implicitDiffusion).
        A_v=A_v,
        K_v=K_v,
        implicit_vertical_mixing=True,
        physics=physics,
        **overrides,
    )
    return cfg
