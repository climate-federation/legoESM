"""Reusable MITgcm ocean-numerics recipe card + the MITgcm<->legoESM wiring diagram.

Like ``nemo_recipe.py`` / the ``veros_*`` cards, this is a PURE CONFIG layer: it
selects the canonical legoESM blocks that reproduce MITgcm's hydrostatic z-coordinate
ocean numerics.  It contains NO MITgcm-specific solver, grid, bridge, forcing reader,
or state translation — those mimicry-only pieces live in the fidelity harness
(``mitgcm_io``/``mitgcm_runner``/``mitgcm_state_bridge``), never in the model
(oracle-recipe doctrine, docs/ocean/fidelity/oracle_recipe_strategy.md).

The mapping below was established by TERM-BY-TERM validation against MITgcm's own
momentum diagnostics at a bit-imported state (``scripts/tmp/_bgyre_operator_match.py``)
plus a direct read/audit of the MITgcm Fortran:
  - advection (``Um_Advec``) and Coriolis (``Um_Cori``): corr 1.0000 — bit-faithful.
  - the free surface: MITgcm ``implicitFreeSurface`` is UNSPLIT (audited: no barotropic
    sub-cycle, one CG2D solve for eta, no barotropic-velocity prognostic, uniform
    surface-pressure correction) -> ``barotropic_solver="implicit_unsplit"``
    (docs/ocean/fidelity/mitgcm_unsplit_freesurface_fix.md).

ALL 5 tutorial recipes (``mitgcm_{barotropic_gyre,baroclinic_gyre,front_relax,
advection_gyre,reentrant_channel}_recipe.py``) select this card's block choices: they
are SETUPS pairing the shared MITgcm numerics with a specific tutorial domain +
forcing.  :func:`mitgcm_canonical_ocean_config` HARD-PINS the numerics common to every
hydrostatic-ocean MITgcm tutorial and exposes the genuinely per-deck axes (solver,
side BC, tracer scheme, abEps, vertical mixing / physics, Laplacian-vs-biharmonic
viscosity) as arguments with MITgcm-faithful defaults.
"""

from __future__ import annotations

from legoesm.ocean.state import LatLonCGridOceanConfig

# MITgcm tutorial defaults (Boussinesq): gravity=9.81, rhoNil~999.8-1035 (per setup).
MITGCM_DEFAULT_G = 9.81
MITGCM_DEFAULT_RHO_0 = 999.8


# --- The MITgcm <-> legoESM WIRING DIAGRAM ---------------------------------------
# (MITgcm option / Fortran term, legoESM config field, chosen block + validation note).
# This is the auditable map a user reads to see exactly which legoESM block each MITgcm
# numeric corresponds to, and how faithful it is.  "[pinned]" = identical in every
# MITgcm tutorial deck (hard-pinned by the card); "[per-deck]" = a card argument with a
# MITgcm-faithful default that an individual setup overrides.
MITGCM_BLOCK_MAPPING: tuple[tuple[str, str, str], ...] = (
    ("gravity, rhoNil (Boussinesq reference)", "g + rho_0",
     "[per-deck] top-level g=9.81, rho_0=rhoNil (per-setup; constants left default)"),
    ("EOS (eosType='LINEAR', tAlpha, sBeta)", "eos + eos_linear",
     "[pinned eos='linear'] LinearEOSConfig(alpha_T, beta_S) — tAlpha/sBeta direct"),
    ("momentum advection (mom_fluxform, default)", "momentum_advection",
     "[pinned] flux_form — matches Um_Advec corr 1.0000 (incl. metric terms)"),
    ("momentum flux reconstruction (advScheme=2 centered)", "momentum_flux_scheme",
     "[pinned] centered"),
    ("KE gradient", "ke_gradient_scheme",
     "n/a in flux_form — the flux-form advection IS -div(uu); the separate KE "
     "gradient is correctly dropped (ocean_pe_latlon_cgrid.py:3180)"),
    ("Coriolis (mom_u/v_coriolis, 4-pt face-f average)", "coriolis_scheme",
     "[pinned] explicit_ab2 (face-f) — matches Um_Cori corr 1.0000; planetary f x u "
     "in du_dt"),
    ("hydrostatic PGF (calc_phi_hyd, integr_GeoPot=2, FS load excluded)",
     "baroclinic PGF (iterate_eos_and_pressure_anomaly)",
     "energy-conserving FD center-staggered integral; baroclinic part explicit/AB2"),
    ("free surface (implicitFreeSurface, implicSurfPress=1, UNSPLIT)",
     "barotropic_solver",
     "[per-deck; default implicit_unsplit] one 3D predictor + one elliptic eta solve "
     "+ uniform correction, NO barotropic/baroclinic mode split (AUDITED). theta=1 "
     "pinned. Single-layer / 1-D decks (barotropic/advection gyre, front_relax) use "
     "implicit_cn (the per-tendency oracle tier; no 2dx checkerboard there)."),
    ("tracer advection (tempAdvScheme; 2=centered, 80=DST3)", "tracer_advection",
     "[per-deck; default centered] centered (tempAdvScheme=2) / dst3_multidim "
     "(advScheme=80) / 'tvd' for homogeneous decks (tracer dynamically inert)"),
    ("lateral viscosity (viscAh/viscA4, useStrainTensionVisc=.FALSE.)",
     "A_h / B_h + lateral_viscosity_operator",
     "[pinned operator flux_divergence] component del2 (viscAh) or component del4 "
     "(viscA4, B_h via overrides) — MITgcm per-component harmonic/biharmonic friction"),
    ("lateral tracer diffusion (diffKhT)", "K_h",
     "[per-deck] Laplacian K_h * laplacian_cgrid"),
    ("vertical viscosity/diffusion (viscAr, diffKrT, implicitDiffusion)",
     "A_v, K_v + implicit_vertical_mixing",
     "[per-deck; implicit_vertical_mixing default True] implicit backward-Euler "
     "vertical mixing"),
    ("convective adjustment (ivdc_kappa)", "physics.convection",
     "[per-deck] enhanced_diffusion with K_conv=ivdc_kappa"),
    ("GM/Redi eddy parameterisation (pkg/gmredi)", "gm_redi",
     "[per-deck; via overrides] GMRediConfig (kappa_GM/Redi, dm95/gkw91 taper, S_max) "
     "— threaded through the unsplit step (prognostic-EKE GM is a TODO)"),
    ("surface restoring (tauThetaClimRelax, RBCS)",
     "physics.surface_forcing / sponge",
     "RestoringConfig (surface) / SpongeForcing (RBCS interior)"),
    ("wind stress (zonalWindFile)", "surface_forcing (OceanSurfaceForcing.tau_x)",
     "external wind stress (sign-flipped: legoESM tau is the atmospheric stress)"),
    ("time integration (Adams-Bashforth, abEps)", "outer_integrator + ab2_epsilon",
     "[pinned ab2 / ab2_scope total; per-deck abEps] ab2 with ab2_epsilon=abEps; "
     "momDissip_In_AB=T == ab2_scope='total'"),
    ("lateral BC (no_slip_sides)", "lateral_side_bc",
     "[per-deck; default no_slip] no_slip (no_slip_sides=.TRUE.) / free_slip "
     "(no_slip_sides=.FALSE., e.g. front_relax)"),
)


def mitgcm_canonical_ocean_config(
    *,
    eos_linear,
    g: float = MITGCM_DEFAULT_G,
    rho_0: float = MITGCM_DEFAULT_RHO_0,
    A_h: float = 0.0,      # noqa: N803 (matches the config field name)
    K_h: float = 0.0,      # noqa: N803
    A_v: float | None = None,    # noqa: N803  (None -> config default; single-layer)
    K_v: float | None = None,    # noqa: N803
    physics=None,
    tracer_advection: str = "centered",      # MITgcm tempAdvScheme=2 (per-deck)
    lateral_side_bc: str = "no_slip",        # MITgcm no_slip_sides=.TRUE. (per-deck)
    barotropic_solver: str = "implicit_unsplit",   # unsplit implicitFreeSurface
    ab2_epsilon: float = 0.1,                # MITgcm abEps (per-deck)
    implicit_vertical_mixing: bool = True,   # MITgcm implicitDiffusion=.TRUE.
    **overrides,
) -> LatLonCGridOceanConfig:
    """Return the SHARED MITgcm-faithful ``LatLonCGridOceanConfig`` block choices.

    HARD-PINS the numerics common to EVERY hydrostatic-ocean MITgcm tutorial
    (per the ``[pinned]`` rows of :data:`MITGCM_BLOCK_MAPPING`): linear EOS,
    flux-form centered momentum, explicit_ab2 face-f Coriolis, AB2 (``ab2_scope=
    'total'``), component ``flux_divergence`` lateral friction (no cos-lat
    scaling), the fully-backward-Euler (``theta=1``) free surface, and the
    differentiable / no-conservation-fixer runtime flags.

    Exposes the genuinely PER-DECK axes as arguments with MITgcm-faithful
    defaults: the per-setup dimensional knobs (``g``, ``rho_0``, the linear-EOS
    coefficients, ``A_h``/``K_h``/``A_v``/``K_v``, the physics block), and the
    axes that differ between tutorial decks (``barotropic_solver``,
    ``lateral_side_bc``, ``tracer_advection``, ``ab2_epsilon``,
    ``implicit_vertical_mixing``).  ``A_v``/``K_v`` left ``None`` fall back to the
    config default (single-layer homogeneous decks).  ``overrides`` patches any
    remaining field (e.g. ``gm_redi``, biharmonic ``B_h``/``B_h_lat_scaling``).

    This is the one auditable place the MITgcm card's choices live; all 5 tutorial
    recipes call it.
    """
    vmix: dict[str, float] = {}
    if A_v is not None:
        vmix["A_v"] = A_v
    if K_v is not None:
        vmix["K_v"] = K_v
    return LatLonCGridOceanConfig(
        # --- per-deck dimensional knobs ---
        g=g,
        rho_0=rho_0,
        eos="linear",
        eos_linear=eos_linear,
        A_h=A_h,
        K_h=K_h,
        physics=physics,
        # --- pinned: MITgcm flux-form centered momentum ---
        momentum_advection="flux_form",
        momentum_flux_scheme="centered",
        # --- per-deck tracer advection (tempAdvScheme) ---
        tracer_advection=tracer_advection,
        # --- pinned: explicit_ab2 face-f Coriolis + AB2(total) ---
        coriolis_scheme="explicit_ab2",
        coriolis_energy_conserving=False,
        outer_integrator="ab2",
        ab2_scope="total",
        ab2_epsilon=ab2_epsilon,
        # --- free surface: per-deck solver, pinned fully-backward-Euler theta ---
        barotropic_solver=barotropic_solver,
        barotropic_implicit_theta_eta=1.0,
        barotropic_implicit_theta_pgf=1.0,
        # --- pinned: component del2 lateral friction, no cos-lat scaling ---
        lateral_viscosity_operator="flux_divergence",
        A_h_lat_scaling=False,
        C_smag=0.0,
        lateral_side_bc=lateral_side_bc,
        # --- per-deck vertical mixing (implicit backward-Euler) ---
        implicit_vertical_mixing=implicit_vertical_mixing,
        bottom_drag_r=0.0,
        # --- pinned: differentiable barotropic solve, no conservation fixer ---
        differentiable_barotropic=True,
        use_conservation_fixer=False,
        enable_runtime_checks=False,
        **vmix,
        **overrides,
    )
