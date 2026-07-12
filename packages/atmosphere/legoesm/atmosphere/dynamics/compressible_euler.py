"""Shared utilities for the non-hydrostatic compressible Euler equations.

This module provides shared infrastructure used by all non-hydrostatic
compressible Euler solvers (C-D grid cubed-sphere, lat-lon FV, MPAS,
spectral):

- ``CompressibleEulerConfig`` — base configuration NamedTuple
- ``compute_exner_perturbation`` — Exner function perturbation from EOS
- ``sponge_profile`` — Rayleigh damping profile
- ``acoustic_substeps`` — forward-backward acoustic substeps
- ``acoustic_substeps_semi_implicit`` — tridiagonal implicit acoustic substeps

The A-grid cubed-sphere slow-tendency solver that previously lived here
has been removed.  Use ``cdgrid_compressible_euler_slow_tendencies`` from
``compressible_euler_cdgrid.py`` instead.

Equations solved
----------------
Compressible-Euler dry-air system in perturbation form. The prognostic
state ``NonHydrostaticState`` carries

    u, v          horizontal velocity components
    w             vertical velocity at half-levels
    theta_prime   potential-temperature perturbation about ref(z)
    rho_prime     density perturbation about ref(z)
    phis          surface geopotential (diagnostic)
    tracers       passive + reactive tracer mixing ratios

with the full state recovered as

    theta(x, y, z, t) = theta_ref(z) + theta_prime(x, y, z, t)
    rho(x, y, z, t)   = rho_ref(z)   + rho_prime(x, y, z, t).

The reference profile ``(theta_ref, rho_ref)`` is supplied by
``HeightCoordinate`` and is hydrostatically balanced. Subtracting it
keeps the acoustic-substep pressure-gradient terms well-conditioned
because the dominant background ``g * rho_ref`` cancels analytically.

Acoustic substepping uses the Skamarock-Klemp split-explicit scheme:
slow horizontal advection + tracer flux divergence + diffusion are
frozen for ``n_acoustic_substeps`` short substeps that resolve the
fast acoustic modes. Vertical acoustic terms can be advanced either
forward-backward (default) or with a tridiagonal implicit solve when
``semi_implicit_acoustic=True`` — see ``acoustic_substeps`` and
``acoustic_substeps_semi_implicit`` for the exact update formulas.

Grid callback contract (existing consumers)
-------------------------------------------
The acoustic substep routines in this module are grid-agnostic with
respect to the horizontal stencil: each existing consumer
(``compressible_euler_cdgrid.py``, ``compressible_euler_mpas.py``,
``spectral_nh.py``) assembles the horizontal pressure-gradient and
flux-divergence contributions in its own slow-tendency routine and
then calls the shared substep kernel for the vertically coupled
acoustic update.

For that pattern to work, every dycore caller currently must provide:

- ``HeightCoordinate`` exposing ``rho_ref``, ``theta_ref``, ``dz`` and
  the half-level / full-level arrangement used by the chosen Lorenz
  staggering.
- ``TerrainMetric`` exposing the column-local Jacobian ``J`` and
  half-level scale-factor used inside the vertical implicit solve.
- A ``physics_fn`` callable that returns physics tendencies on the
  same state pytree as the dycore, applied between split-explicit
  outer stages.

This contract is **descriptive, not prescriptive**: it documents the
shape of what cubed-sphere C-D and MPAS Voronoi do today.
Plane-specific or lat-lon-specific extensions land in their own
modules in follow-up PRs of the CRM rollout and may add new optional
callbacks (vertical-tridiagonal coefficient assembly, periodic-halo
operator) without changing the existing signatures.

Conservation invariant
----------------------
Discrete dry-air mass on each existing grid is ``sum_{cells} rho * J
* area_cell * dz``. When ``CompressibleEulerConfig.fix_mass=True`` the
dycore applies a uniform additive correction to ``rho_prime`` so that
this sum equals a stored target (``anchor_mass_to_initial=True``
anchors the target to ``t=0`` and prevents drift). The correction is
constant per outer step and so preserves all spatial gradients used by
the slow-tendency routine.

References
----------
- Skamarock & Klemp (2008): A Time-Split Nonhydrostatic Atmospheric Model.
- Klemp et al. (2007): Terrain-Following Coordinate.
"""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp

from legoesm.core.state import NonHydrostaticState, NonHydrostaticTendencies
from legoesm.grids.vertical import HeightCoordinate, TerrainMetric
from legoesm.timestepping.split_explicit import SplitExplicitConfig
from legoesm.timestepping.tridiagonal import thomas_solve_batched
from legoesm.atmosphere.physics.thermodynamics import sanitize_theta_rho
from legoesm import constants


class CompressibleEulerConfig(NamedTuple):
    """Configuration for the non-hydrostatic compressible Euler model."""
    g: float = constants.g
    hyperdiff_coeff: float = 0.0
    hyperdiff_rho_coeff: float = 0.0
    hyperdiff_w_coeff: float = 0.0
    sponge_width: float = 10000.0   # Sponge layer width from model top [m]
    sponge_coeff: float = 0.05      # Maximum Rayleigh damping rate [1/s]
    sponge_w_only: bool = False     # If True, the top sponge damps ONLY w
                                    # (SAM damping.f90: w/(1+taudamp); u/v/θ/ρ
                                    # untouched ⇒ anvil wind+thermo preserved).
                                    # False = damp all 5 fields (MPAS-style).
                                    # Plane dycore only; SAM-faithful CRM = True.
    sponge_profile_shape: str = "sin2"   # Sponge taper: "sin2" (gentle) or
                                    # "sam_rational" (SAM damping.f90
                                    # zzz/(1+zzz), ramps fast after the base).
                                    # SAM-faithful CRM = "sam_rational".
    n_acoustic_substeps: int = 6
    small_earth_factor: float = 1.0
    use_coriolis: bool = True       # Set False for f=0 tests (e.g. DCMIP TC3)
    semi_implicit_acoustic: bool = False  # Use tridiagonal solve for acoustic substeps
    outer_integrator: str = "ssp_rk3"  # "ssp_rk3" | "ssp_rk34"/"ssp34" | "ssp_rk54"/"ssp45"
    fix_mass: bool = False            # Apply NH mass conservation fixer
    anchor_mass_to_initial: bool = False  # Anchor to initial mass (prevents drift)
    acoustic_off_centering: float = 0.0   # Off-centering parameter beta for acoustic steps
                                          # 0.0 = centered (neutral), 0.1 = slightly damped
                                          # Damps vertically-propagating acoustic modes
                                          # without horizontal CFL constraint (Skamarock 2008)
    si_w_vertical_filter_nu: float = 0.0  # Vertical Laplacian filter for w inside
                                          # each SI acoustic substep:
                                          #   w[k] += nu * (w[k-1] - 2*w[k] + w[k+1])
                                          # Default 0.0 = off (preserves the iter-78
                                          # contract of the SI scheme).
                                          # Empirical: the SI dycore has a structural
                                          # exponential mode (growth rate ~1.25/step)
                                          # triggered by ANY perturbed theta' IC
                                          # (warm-bubble tests xfailed at iter-65 had
                                          # the same root). The mode saturates at
                                          # max|w|~15 m/s with active moist physics
                                          # (gray+Kessler at N=128 dx=2km nlev=30 ran
                                          # 100 days stably) but blows up to NaN with
                                          # cleaner physics (RRTMGP+Morrison) or a
                                          # dry dycore. Setting nu>=0.4 fully damps
                                          # the unstable mode at every dt tested
                                          # (0.5-20 s). 0.5 = explicit-diffusion CFL
                                          # bound; above that the filter itself NaNs.
                                          # Recommended: 0.3-0.4 for runs with
                                          # perturbed theta' IC; 0.0 for clean Wing
                                          # IC + active moist physics.
    implicit_buoyancy: bool = False       # Klemp-Wilhelmson 1978 implicit-buoyancy
                                          # in the SI acoustic substep. Substitutes
                                          # theta_p_new = theta_p_c
                                          #   - dt_s*w_full/J * dtheta_ref/dz
                                          # into the buoyancy term g*theta_p/theta_0
                                          # of the w-equation. Adds three nearest-
                                          # neighbour bands to the existing implicit
                                          # PG tridiagonal: kappa = 0.25*dt_s^2*g/
                                          # (theta_0_half*J). Stabilizes the
                                          # w-theta gravity-wave feedback that
                                          # destabilizes the plane NH dycore at
                                          # coarse vertical resolution
                                          # (dz~1000 m) with stratified ICs.
                                          # Only active when
                                          # semi_implicit_acoustic=True.
    substep_horizontal_acoustic: bool = False
                                          # Plane SI dycore only. When True the
                                          # horizontal pressure gradient AND the
                                          # mass-continuity divergence are moved
                                          # OUT of the slow tendency and INTO the
                                          # acoustic substep loop (full Skamarock-
                                          # Klemp split-explicit). The slow
                                          # tendency then carries only advection +
                                          # diffusion + Coriolis + sponge.
                                          # WHY: with the horizontal PG in the slow
                                          # tendency (applied once per RK3 stage at
                                          # the OUTER dt) the horizontal acoustic
                                          # mode is integrated at dt, not dt/nsub.
                                          # At fine dx (<=1 km) that mode grows
                                          # unboundedly from any perturbed-theta'
                                          # IC (u -> O(1e4) m/s -> NaN) because the
                                          # C-grid PG/divergence adjoint pairing is
                                          # only energy-neutral for constant
                                          # theta_0/rho_0. Substepping the
                                          # horizontal acoustic terms lowers their
                                          # effective CFL to c_s*dt/(nsub*dx) << 1.
                                          # Default False preserves the iter-183
                                          # coarse-grid contract bit-for-bit.
    # The following two knobs are consumed ONLY by the doubly-periodic
    # plane non-hydrostatic dycore
    # (:mod:`legoesm.atmosphere.dynamics.compressible_euler_plane`).
    # Cubed-sphere, MPAS, and spectral NH dycores ignore them entirely
    # — each of those has its own Smagorinsky knob in its own
    # ``*CompressibleEulerConfig`` (e.g. ``CDGridCompressibleEulerConfig.
    # smagorinsky_cs``). Defaults of ``0.0`` and ``1.0`` make this a
    # no-op on every dycore.
    # The plane closure is a FULL-3D Smagorinsky-Lilly: the strain
    # magnitude |S|² = 2 S_ij S_ij includes all six components (∂u/∂x,
    # ∂v/∂y, ∂w/∂z and the mixed S12/S13/S23), the SAME convention as
    # SAM ``SGS_TKE/shear_prod3D.f90``.  It also subtracts SAM's sub-grid
    # buoyancy frequency ``Pr·N²`` inside the strain sqrt (the dosmagor
    # stratification correction) so mixing shuts off in stable layers —
    # see ``compressible_euler_plane._compute_smagorinsky_K_m_plane``.
    # SAM defaults: Cs=0.19 (CRM 0.22, LES 0.15; sgs.f90:76), Pr=1.0
    # (tke_full.f90).  The Prandtl number controls the thermal (K_h) leg
    # AND the −Pr·N² stratification subtraction.
    smagorinsky_cs: float = 0.0           # Smagorinsky-Lilly LES coefficient.
                                          # K_m=(C_s·Δ)^2·sqrt(max(0,|S|²−Pr·N²)).
                                          # SAM Cs=0.19 (CRM 0.22). 0.0 disables.
    smagorinsky_prandtl: float = 1.0      # Turbulent Prandtl number K_h = K_m / Pr.
                                          # SAM dosmagor uses Pr=1.0 (stratification
                                          # dependence is via the −Pr·N² term, not a
                                          # stratification-dependent Pr).
                                          # Must be > 0 when smagorinsky_cs > 0
                                          # (validate_plane_config enforces).
    smagorinsky_wall_damping: bool = True
                                          # Cap the Smagorinsky mixing length at
                                          # the von Kármán wall scaling κz
                                          # (l_m=min(C_s·Δ, κz); Mason 1989). SAM
                                          # dosmagor does NOT (smix=grd) — set
                                          # False for SAM-faithful CRM runs so
                                          # l_m=C_s·Δ at every level (the cap only
                                          # bites the lowest cell). Plane only.
    smagorinsky_delta_max: float = 1000.0
                                          # SAM-faithful cap on the HORIZONTAL grid
                                          # spacing entering the Smagorinsky mixing
                                          # length: Δ=(min(δmax,dx)·min(δmax,dy)·dz)^⅓
                                          # (SAM SGS_TKE/tke_full.f90:42
                                          # coef=min(delta_max,dx·mu)·min(delta_max,
                                          # dy·ady); sgs.f90:90 delta_max=1000 m).
                                          # WITHOUT it the isotropic Δ over-grows on
                                          # coarse grids (dx>1 km, e.g. RCE dx=3-4 km)
                                          # ⇒ K_m too large ⇒ convective variance
                                          # over-mixed. No effect for dx,dy≤δmax
                                          # (GATE/LBA dx=1 km). Plane only.
    smagorinsky_stability_length: bool = False
                                          # SAM dosmagor STABLE-layer Deardorff
                                          # mixing-length limit (tke_full.f90:
                                          # 285-298): where N²>0, shrink smix=
                                          # min(grd, √(0.76·tk/(Ck·√N²))) + vary
                                          # Cee=Ce1+Ce2·smix/grd ⇒ tk=√(Ck³/Cee·
                                          # (|S|²−Pr·N²))·smix² (Ck=0.1). Default
                                          # False keeps the pure (Cs·Δ)²·|S| form
                                          # bit-for-bit (the |S|²−Pr·N² cutoff
                                          # ALREADY matches SAM in unstable +
                                          # strongly-stable layers; this adds the
                                          # WEAKLY-stable shrink SAM also applies,
                                          # a minor stratosphere/inversion effect).
                                          # Pair with wall_damping=False (SAM
                                          # dosmagor caps neither). Plane only.
    smagorinsky_dynamic: bool = False
                                          # DYNAMIC Smagorinsky (Germano 1991 /
                                          # Lilly 1992): replace the FIXED
                                          # smagorinsky_cs with a coefficient
                                          # C_s(z) computed each step from the
                                          # resolved field via a horizontal test
                                          # filter + the Germano identity, plane-
                                          # averaged over the homogeneous (x,y)
                                          # directions and clipped to
                                          # [0, smagorinsky_dynamic_cs_max]. For
                                          # canonical homogeneous-horizontal LES
                                          # (GABLS1, Wangara). smagorinsky_cs is
                                          # then only the FALLBACK / initial value.
                                          # Plane only; single-rank plane average
                                          # (per-rank under MPI — see driver note).
    smagorinsky_dynamic_cs_max: float = 0.4
                                          # Stability clip on the dynamic C_s
                                          # (typical LES C_s∈[0,0.25]; 0.4 is a
                                          # generous backstop against the Germano
                                          # ratio spiking where ⟨M_ijM_ij⟩→0).
    smagorinsky_scale_dependent: bool = False
                                          # SCALE-DEPENDENT dynamic Smagorinsky
                                          # (Bou-Zeid, Meneveau & Parlange 2005,
                                          # Phys. Fluids 17:025105 — the "LASD"
                                          # closure used by the jax-alfa LES
                                          # oracle). Extends smagorinsky_dynamic
                                          # with a SECOND test filter at 4Δ and
                                          # solves the 5th-order Germano-identity
                                          # polynomial for β = C_s²(2Δ)/C_s²(Δ),
                                          # relaxing the scale-INVARIANCE
                                          # assumption of the standard Germano
                                          # procedure (which over-dissipates near
                                          # the wall, where Δ is no longer ≪ the
                                          # integral scale). Requires
                                          # smagorinsky_dynamic=True. Returns a
                                          # LOCALLY-averaged 3D C_s field (oracle
                                          # Imfilter 3×3) — single-rank/GPU LES;
                                          # MPI uses the static closure.
    sgs_vertical_diffusion: bool = False
                                          # SGS-VERT (#81): add the VERTICAL SGS
                                          # flux ∂_z(K ∂_z φ) for u/v/θ'/tracers
                                          # (and w) on top of the horizontal one,
                                          # so the Smagorinsky closure is fully 3D
                                          # like SAM (diffuse_scalar/diffuse_mom add
                                          # the vertical leg). Default False (the
                                          # horizontal-only legacy) to preserve
                                          # existing runs/tests; CRM drivers set
                                          # True for the SAM-faithful 3D SGS.
                                          # No-flux interior BC (surface fluxes are
                                          # applied separately by the surface
                                          # scheme). Plane only.
    # ----- Turbulence-closure mode (CRM / LES / DNS) ----------------------
    # Selects HOW the viscosity K_m driving the diffusion operators above is
    # built. The operators (horizontal + vertical ∂_z(K ∂φ)) are IDENTICAL
    # across modes — only the K field differs, so LES and DNS are a small
    # extension of the existing CRM closure:
    #   "smagorinsky" : K_m = Smagorinsky-Lilly EDDY viscosity (gated by
    #                   smagorinsky_cs>0). CRM (Cs~0.19-0.22, dx~1-4 km) AND
    #                   LES (Cs~0.15, dx~10-100 m, wall_damping=True) — only
    #                   the resolved scale (dx) changes between them.
    #   "molecular"   : K_m = molecular_viscosity (CONSTANT ν), K_h = ν/Pr.
    #                   DNS — NO eddy model, molecular momentum/heat/scalar
    #                   diffusion only. Needs sgs_vertical_diffusion=True for
    #                   the full 3-D ν∇²; dx must resolve ~the Kolmogorov scale.
    #   "none"        : no SGS/molecular diffusion (inviscid; hyperdiff only).
    # Default "smagorinsky" keeps every existing config bit-for-bit (the
    # smagorinsky_cs>0 gate is unchanged). Plane dycore only.
    turbulence_closure: str = "smagorinsky"
    molecular_viscosity: float = 0.0      # Constant kinematic ν [m²/s] for the
                                          # "molecular" (DNS) closure. 0.0 = off;
                                          # set to constants.nu_air (1.5e-5) for
                                          # air, or a scaled value for a reduced-
                                          # Reynolds DNS. Ignored unless
                                          # turbulence_closure="molecular".
    molecular_prandtl: float = constants.prandtl_air  # ν/κ for the DNS heat +
                                          # scalar legs (K_h = ν/Pr). Must be >0
                                          # when the molecular closure is active.
    vreman_c: float = 0.07                # Vreman (2004) model constant (≈2.5·C_s²)
                                          # for turbulence_closure="vreman" — an
                                          # OPTIONAL eddy closure that vanishes in
                                          # resolved laminar/2-D shear and handles
                                          # anisotropic Δx≠Δz grids. Ignored unless
                                          # turbulence_closure="vreman". NOTE: this
                                          # is NOT SAM-faithful (SAM uses
                                          # Smagorinsky) — for experimentation, not
                                          # the gSAM-match runs. K_h = K_m/smagorinsky_prandtl.
    amd_c: float = 0.3                    # Anisotropic Minimum-Dissipation constant
                                          # (modified Poincaré const ≈0.3; Rozema 2015
                                          # / Abkar-Bae-Moin 2016) for
                                          # turbulence_closure="amd" — an OPTIONAL
                                          # eddy closure that gives ZERO SGS viscosity
                                          # where the resolved flow needs none
                                          # (minimum-dissipation) and handles Δx≠Δz.
                                          # Not SAM-faithful. K_h = K_m/smagorinsky_prandtl.
    horizontal_advection_scheme: str = "upwind1"
                                          # Horizontal advection of theta_prime, u, v, w
                                          # (and tracers) on the plane dycore. Three
                                          # choices, defined by
                                          # HORIZONTAL_ADVECTION_HALO_REQUIREMENT in
                                          # compressible_euler_plane.py:
                                          # "upwind1" — first-order upwind (cheap, very
                                          # dispersive at coarse dx).
                                          # "van_leer" — 2nd-order TVD, stencil 4
                                          # (iter-183 production: 3x wall-time speedup
                                          # vs upwind1 at dt=20 thanks to lower
                                          # numerical diffusion + monotonicity).
                                          # "weno5" — 5th-order WENO-Z, stencil 6
                                          # (least grid-scale dispersion but ~3x
                                          # per-step cost; opt-in for sharp-front
                                          # problems).
                                          # Class default stays "upwind1" for back-
                                          # compat with iter-7 fixtures; the
                                          # production driver overrides to van_leer
                                          # at parse_args time (iter-183).
                                          # Consumed by
                                          # ``compressible_euler_plane.py`` only;
                                          # cubed-sphere / MPAS ignore it.
    horizontal_momentum_advection_scheme: str | None = None
                                          # ADV-SPLIT (#86): separate horizontal
                                          # advection scheme for the MOMENTUM legs
                                          # (u, v, w) only. None ⇒ momentum uses
                                          # `horizontal_advection_scheme` (legacy,
                                          # both legs same). Set "centered" for the
                                          # SAM-faithful split: NON-diffusive
                                          # 2nd-order centred momentum (= gSAM
                                          # `advect2_mom`) + monotone van_leer
                                          # SCALARS (≈ MPDATA). van_leer momentum
                                          # over-diffuses ⇒ suppresses updraft
                                          # cores / w-variance tails (codex iter-68).
                                          # Plane only; halo = max(scalar, momentum).
    vertical_tracer_advection: str = "centered"
                                          # VERTICAL advection scheme for TRACERS
                                          # on the plane dycore (D5). Two choices:
                                          # "centered" — 2nd-order centred (class
                                          # default; can overshoot into negative
                                          # tracer at sharp convective gradients).
                                          # "van_leer" — monotone van-Leer TVD
                                          # (positive-definite; SAM advects scalars
                                          # with a monotone scheme). u/v momentum
                                          # always stay centred (SAM convention).
                                          # Consumed by
                                          # ``compressible_euler_plane.py`` only.
    acoustic_theta_advection: str = "centered"
                                          # VERTICAL θ' advection scheme INSIDE the
                                          # acoustic substep (the w·∂θ/∂z update of
                                          # the column kernels). "centered" (class
                                          # default = current behaviour, bit-for-bit)
                                          # or "van_leer" — monotone van-Leer TVD,
                                          # SAM-faithful (SAM advects θ with a
                                          # monotone scheme; momentum stays centred).
                                          # The centred scheme overshoots at the
                                          # SHARP tropopause θ-gradient ⇒ a dispersive
                                          # COLD DRIFT of the cold-point over long RCE
                                          # runs (iter-200 diagnosis). Decoupled from
                                          # the implicit w-solve (the θ' update uses
                                          # the post-solve w explicitly) ⇒ the limiter
                                          # touches ONLY θ' transport, not the acoustic
                                          # solve. Plane SI dycore.
    vertical_theta_diffusion: float = 0.0
                                          # Explicit vertical Laplacian diffusivity
                                          # on theta_prime [m^2/s], applied per outer
                                          # RK stage. Damps the buoyancy-driven
                                          # gravity-wave amplification that
                                          # destabilises the plane NH dycore at
                                          # dx ~ 2 km / dt > 0.5 s with a coarse
                                          # vertical grid (dz ~ 1000 m). Rigid (zero)
                                          # boundary condition at top + bottom.
                                          # Typical effective value: nu_v ~ 1e3-5e3
                                          # so dt * nu_v / dz^2 stays below ~0.1
                                          # (explicit-Euler CFL bound). 0.0 disables.
                                          # Consumed by plane dycore only.
    moist_buoyancy: bool = True           # Add the MOIST part of the SAM
                                          # buoyancy (buoyancy.f90) to the plane
                                          # slow-tendency dw/dt:
                                          #   B_moist = g*(eps_v*(qv - qv_mean)
                                          #               - (qcond - qcond_mean))
                                          # eps_v = 1/eps-1 ~= 0.61; bar = HORIZONTAL
                                          # MEAN profile (= SAM qv0/qn0/qp0 base
                                          # state) so the term is a pure perturbation
                                          # (zero horizontal mean -> no spurious mean
                                          # updraft against the dry reference). The
                                          # dry theta' buoyancy stays in the acoustic
                                          # substep; this adds the vapour +
                                          # condensate-loading part the dry term omits
                                          # (#1 missing convective driver vs SAM).
                                          # qcond = all condensate mass (slots 1..5).
                                          # True = CRM-faithful; False = bit-exact
                                          # dry-buoyancy dycore. Plane only; no-op when
                                          # no tracers / moisture identically zero.
    acoustic_moist_buoyancy: bool = True  # Apply the moist buoyancy INSIDE the
                                          # acoustic substep loop (every substep,
                                          # frozen moisture) instead of once per RK
                                          # stage in the slow tendency. This makes the
                                          # condensate-loading DRAG act at the same
                                          # frequency as the dry theta' buoyancy in the
                                          # substeps; without it (False = old path)
                                          # latent-heated updrafts feel the dry warming
                                          # n_substeps x per step but the moist drag
                                          # only 1x -> resolved convection runs away
                                          # (max|w|->40 m/s, RCE blows up ~day 2.5).
                                          # Requires moist_buoyancy=True. Plane only.
    acoustic_moist_global_mean: bool = False  # MPI ONLY. The SAM moist-buoyancy
                                          # perturbation subtracts the horizontal
                                          # mean of qv/qcond/theta'. Default
                                          # (False) uses a rank-LOCAL mean under a
                                          # plane pencil decomposition (zero comm,
                                          # the scalable production choice) — which
                                          # diverges ~6e-4 from the single-rank
                                          # reference. Set True to use the exact
                                          # GLOBAL mean (one allreduce per field)
                                          # for serial-parity / oracle validation
                                          # runs, at ~5-16% per-step comm cost.
                                          # No effect single-rank or n_ranks==1.


# ==============================================================================
# Equation of state
# ==============================================================================

def compute_exner_perturbation(
    rho_prime: jax.Array,
    theta_prime: jax.Array,
    height_coord: HeightCoordinate,
) -> jax.Array:
    """Compute Exner function perturbation from density and theta perturbations.

    The full (dimensionless) Exner function is:
        pi = (R_d · rho · theta / p_0)^(R_d/c_v)

    The perturbation is pi' = pi_total - pi_0.

    Parameters
    ----------
    rho_prime : jax.Array
        Density perturbation [kg/m^3], shape (6, n, n, nlev).
    theta_prime : jax.Array
        Potential temperature perturbation [K], shape (6, n, n, nlev).
    height_coord : HeightCoordinate
        Vertical coordinate with reference state.

    Returns
    -------
    jax.Array
        Exner perturbation [-], shape (6, n, n, nlev).
    """
    R_d = constants.R_d
    c_v = constants.c_vd

    rho_0 = height_coord.rho_ref  # (nlev,)
    theta_0 = height_coord.theta_ref  # (nlev,)
    pi_0 = height_coord.exner_ref  # (nlev,)

    # Ratio form to avoid catastrophic cancellation in pi_total - pi_0.
    # Since pi = (R_d*rho*theta/p_0)^(R_d/c_v), we have:
    #   pi_total/pi_0 = ((rho_0+rho')*(theta_0+theta') / (rho_0*theta_0))^(R_d/c_v)
    #                 = ((1 + rho'/rho_0)*(1 + theta'/theta_0))^(R_d/c_v)
    #   pi' = pi_0 * (ratio^exponent - 1)
    # This is exact: zero when rho'=theta'=0, no large-value subtraction.
    exponent = R_d / c_v
    rho_rel = 1.0 + rho_prime / jnp.clip(rho_0, 1.0e-9, None)
    theta_rel = 1.0 + theta_prime / jnp.clip(theta_0, 50.0, None)
    ratio = jnp.clip(rho_rel * theta_rel, 1.0e-12, 1.0e12)
    return pi_0 * jnp.expm1(exponent * jnp.log(ratio))


# ==============================================================================
# Sponge layer
# ==============================================================================

def sponge_profile(
    z_full: jax.Array,
    H: float,
    sponge_width: float,
    sponge_coeff: float,
    shape: str = "sin2",
) -> jax.Array:
    """Compute Rayleigh damping coefficient profile.

    Increases from 0 to ``sponge_coeff`` over the top ``sponge_width`` metres.
    ``shape`` (static config string) selects the taper:

    * ``"sin2"`` (default) — ``sin²(0.5π·frac)``, a gentle ramp.
    * ``"sam_rational"`` — SAM ``damping.f90``: ``zzz/(1+zzz)`` with
      ``zzz=100·frac²`` (``frac=(ν−nub)/(1−nub)`` exactly, since the sponge
      base ``H−sponge_width`` ⇔ SAM ``nub``). Ramps FAST after the base (≈half
      strength by ``frac=0.1``) — concentrates the damping in the lower sponge,
      unlike sin²'s upward-shifted profile. SAM-faithful CRM runs use this.

    Returns shape (nlev,).
    """
    z_sponge_bottom = H - sponge_width
    # Fraction into sponge layer: 0 below, 1 at top
    frac = jnp.clip((z_full - z_sponge_bottom) / sponge_width, 0.0, 1.0)
    if shape == "sam_rational":
        zzz = 100.0 * frac ** 2
        return sponge_coeff * zzz / (1.0 + zzz)
    if shape == "sin2":
        return sponge_coeff * jnp.sin(0.5 * jnp.pi * frac) ** 2
    raise ValueError(
        f"Unknown sponge profile shape: {shape!r}. Expected "
        f"'sin2' or 'sam_rational'."
    )


# ==============================================================================
# Acoustic substeps (forward-backward)
# ==============================================================================


def _theta_vert_advection_van_leer_kernel(
    theta_total: jax.Array,
    w_half: jax.Array,
    height_coord: HeightCoordinate,
    J: jax.Array,
) -> jax.Array:
    """Monotone (van-Leer TVD) vertical advection tendency ``-(w/J) ∂θ/∂z``.

    Layout-agnostic ([..., nlev]) counterpart of the plane's
    ``vertical_advection_van_leer_plane`` (``J[..., None]`` broadcast so the
    shared acoustic column kernels can call it). Returns the tendency to ADD
    (same sign + smooth-limit as the centred update it replaces — van-Leer
    reduces to the centred scheme for a smooth field with uniform ``w``).

    ``w_half`` is on the native ``w`` interface grid ([..., nlev+1]); the flux
    ``F[j]=w[j]·θ_face[j]`` reconstructs the slope-limited 2nd-order upwind face
    value chosen by ``sign(w[j])`` (no half→full averaging). Rigid lids
    (``w[...,0]=w[...,-1]=0``) ⇒ zero flux through top/surface. The van-Leer
    limiter forbids NEW extrema ⇒ no dispersive over/undershoot at the sharp
    tropopause θ-gradient (the centred scheme's cold-drift mechanism, iter-200).
    """
    from legoesm.core.flux_limiters import van_leer_face_values
    nlev = theta_total.shape[-1]
    f = theta_total
    w = w_half
    dz = height_coord.dz                                # (nlev,)
    pad_axes = ((0, 0),) * (f.ndim - 1)
    fp = jnp.pad(f, (*pad_axes, (2, 2)), mode="edge")   # (..., nlev+4)
    f_jm2 = fp[..., 0:nlev + 1]
    f_jm1 = fp[..., 1:nlev + 2]
    f_j = fp[..., 2:nlev + 3]
    f_jp1 = fp[..., 3:nlev + 4]
    phi_pos, phi_neg = van_leer_face_values(f_jm2, f_jm1, f_j, f_jp1)
    q_face = jnp.where(w <= 0.0, phi_pos, phi_neg)       # (..., nlev+1)
    flux = w * q_face
    flux = flux.at[..., 0].set(0.0).at[..., -1].set(0.0)
    flux_div = (flux[..., 1:] - flux[..., :-1]) / dz
    dwdz = f * (w[..., :-1] - w[..., 1:]) / dz
    return (flux_div + dwdz) / J[..., None]


def acoustic_column_kernel(
    w_c: jax.Array,
    theta_p_c: jax.Array,
    rho_p_c: jax.Array,
    height_coord: HeightCoordinate,
    J: jax.Array,
    dt_s: float,
    beta: float,
    g: float,
    theta_vert_van_leer: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single forward-backward acoustic substep (column-local algebra).

    This kernel encapsulates the vertical-only update of ``(w, theta',
    rho')`` for one acoustic substep. It is grid-agnostic in the
    horizontal: any number of leading axes broadcast through the
    ``[..., k]`` slicing.

    Contract (vertical = last axis)
    -------------------------------
    - ``w_c`` : ``(..., nlev+1)`` — vertical velocity at Lorenz half
      (interface) levels. Rigid boundaries ``w_c[..., 0] = w_c[..., -1]
      = 0`` must already hold on input; the kernel updates only the
      interior ``[..., 1:-1]`` slice and leaves the boundary values
      untouched.
    - ``theta_p_c`` : ``(..., nlev)`` — potential-temperature
      perturbation at full (cell-centre) levels.
    - ``rho_p_c`` : ``(..., nlev)`` — density perturbation at full
      levels.
    - ``height_coord`` : ``HeightCoordinate`` providing
      ``rho_ref(nlev,)``, ``theta_ref(nlev,)``, ``exner_ref(nlev,)``,
      ``dz(nlev,)`` (full-level spacing), ``dz_half(nlev-1,)``
      (half-level spacing between adjacent cell centres).
    - ``J`` : Jacobian, shape broadcastable to the horizontal leading
      axes of ``w_c``, ``theta_p_c``, ``rho_p_c``. The kernel uses
      ``J[..., None]`` to broadcast over the vertical axis.
    - ``dt_s`` : substep size in seconds (Python scalar).
    - ``beta`` : Skamarock-Klemp off-centering parameter in ``[0, 1)``.
      ``0`` is centred, larger values damp vertically-propagating
      acoustic modes.
    - ``g`` : gravitational acceleration [m/s^2].

    Returns
    -------
    (w_new, theta_p_new, rho_p_new)
        Updated arrays with the same shapes as the inputs. Rigid w
        boundaries preserved.
    """
    c_p = constants.c_pd
    dz = height_coord.dz
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p_c,
        rho_0 + rho_p_c,
    )

    # --- Forward: update w ---
    pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)

    dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / (
        0.5 * (dz[:-1] + dz[1:])
    )

    theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])

    theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
    theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
    buoyancy = g * theta_p_half / theta_0_half

    dw_dt_inner = (
        -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
        + buoyancy
    )

    w_new = w_c.at[..., 1:-1].set(
        w_c[..., 1:-1] + dt_s * dw_dt_inner
    )

    # --- Backward: update rho' using continuity ---
    rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
    pad_axes_w = ((0, 0),) * (w_new.ndim - 1)
    rho_w = jnp.pad(rho_half * w_new[..., 1:-1], (*pad_axes_w, (1, 1)))

    vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
    vert_div = vert_div / J[..., None]

    rho_p_new = rho_p_c - dt_s * vert_div
    if beta != 0.0:
        rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

    # --- Backward: update theta' using vertical w advection ---
    nlev = theta_total.shape[-1]
    if theta_vert_van_leer and nlev > 2:
        # Monotone (van-Leer TVD) θ vertical advection — forbids new extrema at
        # the sharp tropopause θ-gradient (iter-200 cold-drift fix). w_new on its
        # native interface grid; decoupled from the implicit w-solve above.
        theta_adv = _theta_vert_advection_van_leer_kernel(
            theta_total, w_new, height_coord, J)
        theta_p_new = theta_p_c + dt_s * theta_adv
    else:
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        if nlev > 2:
            dz_half_val = height_coord.dz_half  # (nlev-1,)
            dz_centered = dz_half_val[:-1] + dz_half_val[1:]  # (nlev-2,)
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            top_grad = (theta_total[..., 0:1] - theta_total[..., 1:2]) / dz_half_val[0]
            bottom_grad = (
                theta_total[..., -2:-1] - theta_total[..., -1:]
            ) / dz_half_val[-1]
            dtheta_dz = jnp.concatenate([top_grad, inner_grad, bottom_grad], axis=-1)
        else:
            dtheta_dz = jnp.zeros_like(theta_total)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

    return (w_new, theta_p_new, rho_p_new)


def acoustic_substeps(
    state: NonHydrostaticState,
    slow_tend: NonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
) -> NonHydrostaticState:
    """Run N acoustic substeps using forward-backward scheme.

    Each substep:
    1. Forward: update w using vertical Exner gradient + buoyancy
    2. Backward: update rho' using 3D divergence
    3. Backward: update theta' using vertical advection by w

    The per-substep vertical algebra is factored into
    :func:`acoustic_column_kernel` so it can be reused by other dycores
    (currently the future plane dycore in the CRM rollout, PR2b).

    Parameters
    ----------
    state : NonHydrostaticState
        State after slow tendency update.
    slow_tend : NonHydrostaticTendencies
        Slow tendencies (held constant during substeps).
    dt_s : float
        Acoustic substep size [seconds].
    n_substeps : int
        Number of substeps.
    config : SplitExplicitConfig
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    euler_config : CompressibleEulerConfig

    Returns
    -------
    NonHydrostaticState
        State after all acoustic substeps.
    """
    g = euler_config.g
    J = terrain_metric.jacobian

    w = state.w.data
    theta_p = state.theta_prime.data
    rho_p = state.rho_prime.data

    beta = euler_config.acoustic_off_centering

    def substep_body(carry):
        w_c, theta_p_c, rho_p_c = carry
        return acoustic_column_kernel(
            w_c, theta_p_c, rho_p_c,
            height_coord, J, dt_s, beta, g,
        )

    # Python-loop unroll (n_substeps is compile-time static via
    # SplitExplicitConfig). See semi-implicit variant for full rationale.
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _i in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = substep_body(
            (w_final, theta_p_final, rho_p_final),
        )

    return NonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )


# ==============================================================================
# Semi-implicit acoustic substeps (tridiagonal)
# ==============================================================================

def precompute_si_tridiag_bands(
    height_coord: HeightCoordinate,
    J: jax.Array,
    dt_s: float,
    g: float,
    implicit_buoyancy: bool = False,
    nlev: int | None = None,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Loop-invariant pieces of the semi-implicit acoustic tridiag system.

    ``alpha``, ``a_tri``, ``b_tri``, ``c_tri`` and the implicit-buoyancy
    band additions only depend on ``dt_s``, ``height_coord``, ``J``, and
    ``g`` — they do NOT depend on the substep carry ``(w, theta_p,
    rho_p)``. Callers running a multi-substep ``jax.lax.fori_loop``
    can call this once outside the loop and pass the result into
    :func:`semi_implicit_acoustic_column_kernel` via
    ``precomputed_tridiag`` to skip recomputing them every iteration.

    Contract
    --------
    The returned ``(a_tri, b_tri, c_tri)`` are tied to the EXACT
    ``(dt_s, height_coord, J, g, implicit_buoyancy)`` passed here.
    When forwarded to :func:`semi_implicit_acoustic_column_kernel`,
    the same ``dt_s``, ``height_coord``, ``J``, ``g`` MUST be passed
    to the kernel (used for the RHS / backward updates). When
    ``precomputed_tridiag is not None`` the kernel IGNORES its own
    ``implicit_buoyancy`` flag because the bands already encode it —
    so callers should not mix bands built with ``implicit_buoyancy=A``
    and a kernel call with ``implicit_buoyancy=B≠A`` expecting B to
    take effect. The mismatch is silent at runtime.

    Returns
    -------
    (a_tri, b_tri, c_tri) — each broadcastable to ``(*spatial, nlev-1)``.
    """
    c_p = constants.c_pd
    R_d = constants.R_d
    c_v = constants.c_vd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    gamma = c_p / c_v
    T_ref = theta_0 * height_coord.exner_ref
    cs2 = gamma * R_d * T_ref
    cs2_half = 0.5 * (cs2[:-1] + cs2[1:])
    dz_inner = 0.5 * (dz[:-1] + dz[1:])
    theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
    if nlev is None:
        nlev = int(theta_0.shape[-1])

    alpha = dt_s ** 2 * cs2_half / (dz_inner * J[..., None]) ** 2
    pad_axes_a = ((0, 0),) * (alpha.ndim - 1)
    a_tri = jnp.pad(-alpha[..., 1:], (*pad_axes_a, (1, 0)))
    alpha_interior = jnp.pad(alpha[..., 1:-1], (*pad_axes_a, (1, 1)))
    b_tri = 1.0 + alpha + alpha_interior
    c_tri = jnp.pad(-alpha[..., :-1], (*pad_axes_a, (0, 1)))

    if implicit_buoyancy and nlev > 2:
        dz_centered = dz_half[:-1] + dz_half[1:]
        inner_grad_ref = (theta_0[:-2] - theta_0[2:]) / dz_centered
        top_grad_ref = (theta_0[0:1] - theta_0[1:2]) / dz_half[0]
        bottom_grad_ref = (theta_0[-2:-1] - theta_0[-1:]) / dz_half[-1]
        dtheta_ref_dz = jnp.concatenate(
            [top_grad_ref, inner_grad_ref, bottom_grad_ref], axis=-1,
        )
        kappa = 0.25 * dt_s ** 2 * g / (theta_0_half * J[..., None])
        d_above = dtheta_ref_dz[:-1]
        d_below = dtheta_ref_dz[1:]
        a_buoy_full = kappa * d_above
        b_buoy_full = kappa * (d_above + d_below)
        c_buoy_full = kappa * d_below
        a_buoy = jnp.pad(a_buoy_full[..., 1:], (*pad_axes_a, (1, 0)))
        c_buoy = jnp.pad(c_buoy_full[..., :-1], (*pad_axes_a, (0, 1)))
        a_tri = a_tri + a_buoy
        b_tri = b_tri + b_buoy_full
        c_tri = c_tri + c_buoy

    return (a_tri, b_tri, c_tri)


def semi_implicit_acoustic_column_kernel(
    w_c: jax.Array,
    theta_p_c: jax.Array,
    rho_p_c: jax.Array,
    height_coord: HeightCoordinate,
    J: jax.Array,
    dt_s: float,
    beta: float,
    g: float,
    implicit_buoyancy: bool = False,
    precomputed_tridiag: tuple[jax.Array, jax.Array, jax.Array] | None = None,
    si_w_vertical_filter_nu: float = 0.0,
    theta_vert_van_leer: bool = False,
) -> tuple[jax.Array, jax.Array, jax.Array]:
    """Single semi-implicit acoustic substep (column-local algebra).

    Layout-agnostic counterpart of :func:`acoustic_column_kernel`.
    Treats the vertical pressure-gradient term in the w equation
    implicitly via a per-column tridiagonal Thomas solve so the
    vertical acoustic CFL constraint is lifted.

    Inputs follow the same contract as the explicit kernel:
    ``[..., nlev+1]`` w (rigid lid/bottom), ``[..., nlev]`` theta'+rho',
    ``J`` broadcastable to the horizontal leading axes.

    Boundary contract (post iter-69)
    --------------------------------
    Unlike :func:`acoustic_column_kernel` (explicit) which preserves
    the input boundary values ``w_c[..., 0]`` and ``w_c[..., -1]``,
    this kernel **overwrites** them with 0 (rigid lid/bottom BC) on
    output. Callers MUST already obey the rigid BC on entry; this
    enforcement is a fusion optimization, not new physics. If a future
    layout exposes a nonzero-w boundary (e.g., moving bottom), revert
    to the explicit kernel's ``at[..., 1:-1].set(...)`` pattern.

    When ``implicit_buoyancy=True`` the buoyancy contribution
    ``g * theta_p_half / theta_0_half`` in the w-equation is treated
    implicitly by substituting the backward theta'-update into the
    buoyancy term (Klemp-Wilhelmson 1978). This augments the
    tridiagonal system with three nearest-neighbour bands proportional
    to the mean-state stratification ``dtheta_ref/dz`` and closes the
    w<->theta gravity-wave feedback that otherwise grows at coarse
    vertical resolution.
    """
    c_p = constants.c_pd
    R_d = constants.R_d
    c_v = constants.c_vd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    gamma = c_p / c_v
    T_ref = theta_0 * height_coord.exner_ref           # (nlev,)
    cs2 = gamma * R_d * T_ref                          # (nlev,)
    cs2_half = 0.5 * (cs2[:-1] + cs2[1:])              # (nlev-1,)
    dz_inner = 0.5 * (dz[:-1] + dz[1:])                # (nlev-1,)
    nlev = theta_p_c.shape[-1]

    theta_total, rho_total = sanitize_theta_rho(
        theta_0 + theta_p_c, rho_0 + rho_p_c,
    )

    # --- Explicit RHS for w (same as forward step) ---
    pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)
    dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / dz_inner
    theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])
    theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
    theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
    buoyancy = g * theta_p_half / theta_0_half
    dw_dt_inner = (
        -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
        + buoyancy
    )
    rhs = w_c[..., 1:-1] + dt_s * dw_dt_inner

    # --- Tridiagonal coefficients for implicit w solve ---
    # Hoisted fast path: callers running a multi-substep loop can pass
    # the precomputed (a_tri, b_tri, c_tri) — they are loop-invariant
    # (depend only on dt_s, height_coord, J, g) — to skip the rebuild
    # every iteration. See :func:`precompute_si_tridiag_bands`.
    if precomputed_tridiag is not None:
        a_tri, b_tri, c_tri = precomputed_tridiag
    else:
        alpha = dt_s**2 * cs2_half / (dz_inner * J[..., None])**2
        pad_axes_a = ((0, 0),) * (alpha.ndim - 1)
        a_tri = jnp.pad(-alpha[..., 1:], (*pad_axes_a, (1, 0)))
        alpha_interior = jnp.pad(alpha[..., 1:-1], (*pad_axes_a, (1, 1)))
        b_tri = 1.0 + alpha + alpha_interior
        c_tri = jnp.pad(-alpha[..., :-1], (*pad_axes_a, (0, 1)))

        if implicit_buoyancy and nlev > 2:
            # Mean-state d(theta_ref)/dz at full levels (sign convention
            # matches the backward theta'-update used downstream).
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad_ref = (theta_0[:-2] - theta_0[2:]) / dz_centered
            top_grad_ref = (theta_0[0:1] - theta_0[1:2]) / dz_half[0]
            bottom_grad_ref = (theta_0[-2:-1] - theta_0[-1:]) / dz_half[-1]
            dtheta_ref_dz = jnp.concatenate(
                [top_grad_ref, inner_grad_ref, bottom_grad_ref], axis=-1,
            )  # shape (nlev,)
            # kappa at interior half-levels k_int=0..nlev-2, (..., nlev-1).
            kappa = 0.25 * dt_s ** 2 * g / (theta_0_half * J[..., None])
            d_above = dtheta_ref_dz[:-1]
            d_below = dtheta_ref_dz[1:]
            a_buoy_full = kappa * d_above
            b_buoy_full = kappa * (d_above + d_below)
            c_buoy_full = kappa * d_below
            # Boundary handling: drop sub-diag at k_int=0, super-diag at -1.
            a_buoy = jnp.pad(a_buoy_full[..., 1:], (*pad_axes_a, (1, 0)))
            c_buoy = jnp.pad(c_buoy_full[..., :-1], (*pad_axes_a, (0, 1)))
            a_tri = a_tri + a_buoy
            b_tri = b_tri + b_buoy_full
            c_tri = c_tri + c_buoy

    w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)
    # Rigid lid/bottom: w=0 at top and bottom interfaces. Padding with
    # 0 fuses better than `w_c.at[..., 1:-1].set(...)` because pad is a
    # simple HLO op without the dynamic-update-slice fusion barrier on
    # the cuSPARSE custom-call output.
    pad_axes_w = ((0, 0),) * (w_inner_new.ndim - 1)
    w_new = jnp.pad(w_inner_new, (*pad_axes_w, (1, 1)))

    # SI-stability filter: vertical Laplacian damping on w. Defaults
    # off (nu=0.0). Empirical fix for the structural exponential mode
    # that grows in the SI scheme with perturbed theta' IC — see
    # CompressibleEulerConfig.si_w_vertical_filter_nu docstring.
    if si_w_vertical_filter_nu > 0.0:
        w_above = w_new[..., :-2]
        w_below = w_new[..., 2:]
        w_interior = w_new[..., 1:-1]
        w_filt = w_interior + si_w_vertical_filter_nu * (
            w_above - 2.0 * w_interior + w_below
        )
        w_new = w_new.at[..., 1:-1].set(w_filt)

    # --- Backward: update rho' using continuity ---
    # Sign convention: z positive UP, level index TOP→DOWN, so
    # (rho_w[k] − rho_w[k+1]) / dz[k] below is the physical +∂(ρw)/∂z and
    # ρ' ← ρ' − dt·∂(ρw)/∂z is the flux-divergence sink form of continuity.
    # Mass MUST be transported by the SAME post-filter w that θ-advection
    # uses below and that is carried to the next substep — w_new is the
    # single source of truth after the SI filter. At nu=0 this is
    # bit-identical: w_new[..., 1:-1] == w_inner_new (pad-then-slice).
    rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
    rho_w = jnp.pad(rho_half * w_new[..., 1:-1], (*pad_axes_w, (1, 1)))
    vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
    vert_div = vert_div / J[..., None]
    rho_p_new = rho_p_c - dt_s * vert_div
    if beta != 0.0:
        rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

    # --- Backward: update theta' using w-advection of theta_total ---
    if theta_vert_van_leer and nlev > 2:
        # Monotone (van-Leer TVD) θ vertical advection — forbids new extrema at
        # the sharp tropopause θ-gradient (iter-200 cold-drift fix). Decoupled
        # from the implicit w-solve (uses the post-solve w_new explicitly).
        theta_adv = _theta_vert_advection_van_leer_kernel(
            theta_total, w_new, height_coord, J)
        theta_p_new = theta_p_c + dt_s * theta_adv
    else:
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        if nlev > 2:
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            top_grad = (theta_total[..., 0:1] - theta_total[..., 1:2]) / dz_half[0]
            bottom_grad = (theta_total[..., -2:-1] - theta_total[..., -1:]) / dz_half[-1]
            dtheta_dz = jnp.concatenate([top_grad, inner_grad, bottom_grad], axis=-1)
        else:
            dtheta_dz = jnp.zeros_like(theta_total)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz
    return (w_new, theta_p_new, rho_p_new)


def acoustic_substeps_semi_implicit(
    state: NonHydrostaticState,
    slow_tend: NonHydrostaticTendencies,
    dt_s: float,
    n_substeps: int,
    config: SplitExplicitConfig,
    height_coord: HeightCoordinate,
    terrain_metric: TerrainMetric,
    euler_config: CompressibleEulerConfig,
) -> NonHydrostaticState:
    """Semi-implicit acoustic substeps using tridiagonal solve for w.

    Instead of a forward Euler update for w (explicit), the vertical
    pressure gradient term is treated implicitly by solving a tridiagonal
    system for w at each substep. This removes the acoustic CFL
    constraint in the vertical direction, enabling larger time steps
    and longer stable integrations.

    Boundary contract (post iter-69): the substep body overwrites
    ``w[..., 0]`` and ``w[..., -1]`` with 0 (rigid lid/bottom BC) on
    every iteration via ``jnp.pad(w_inner_new, ..., (1, 1))``. Callers
    MUST obey the rigid BC on input. This is a fusion optimization, not
    new physics; revert to ``at[..., 1:-1].set`` if a moving boundary
    is ever introduced.

    Substep loop unroll (post iter-70): ``n_substeps`` MUST be a
    Python ``int`` (it always is when passed through ``SplitExplicitConfig``).
    The function uses a plain Python ``for _ in range(n_substeps)`` to
    fully unroll the substep sequence so XLA can fuse across iterations.
    Passing a traced ``n_substeps`` (e.g., from ``lax.cond``) will fail
    at trace time with ``TracerIntegerConversionError`` — that error is
    the correct guard, do not silence it with ``int(...)``.

    The implicit equation for w at interior half-levels is:

        (1 + dt_s^2 * c_s^2 / dz^2 / J^2) * w_new = w_old + dt_s * RHS_explicit

    where c_s^2 = c_p * R_d * T_ref is the linearized sound speed squared.
    The resulting tridiagonal system is solved per column via the Thomas
    algorithm, with jax.vmap over all columns.

    Parameters
    ----------
    state : NonHydrostaticState
        State after slow tendency update.
    slow_tend : NonHydrostaticTendencies
        Slow tendencies (held constant during substeps).
    dt_s : float
        Acoustic substep size [seconds].
    n_substeps : int
        Number of substeps.
    config : SplitExplicitConfig
    height_coord : HeightCoordinate
    terrain_metric : TerrainMetric
    euler_config : CompressibleEulerConfig

    Returns
    -------
    NonHydrostaticState
        State after all acoustic substeps.
    """
    g = euler_config.g
    c_p = constants.c_pd
    dz = height_coord.dz
    dz_half = height_coord.dz_half
    theta_0 = height_coord.theta_ref
    rho_0 = height_coord.rho_ref
    J = terrain_metric.jacobian  # (*spatial,) — cubed-sphere (6,n,n), latlon (ny,nx), etc.
    beta = euler_config.acoustic_off_centering
    implicit_buoyancy = euler_config.implicit_buoyancy

    # Extract mutable arrays
    w = state.w.data       # (..., nlev+1)
    theta_p = state.theta_prime.data  # (..., nlev)
    rho_p = state.rho_prime.data      # (..., nlev)

    nlev = theta_p.shape[-1]

    # Loop-invariant pieces: dz_inner used in the explicit RHS dpi/dz,
    # and the tridiag bands (alpha + buoyancy) shared across substeps.
    # Hoisted out of the fori_loop via precompute_si_tridiag_bands.
    # NOTE: dz_inner is also recomputed inside precompute_si_tridiag_bands;
    # the recomputation is a 5-character expression and XLA folds it. The
    # helper consumes it for alpha; we use it here for dpi/dz in the RHS.
    dz_inner = 0.5 * (dz[:-1] + dz[1:])  # (nlev-1,)
    a_tri_pre, b_tri_pre, c_tri_pre = precompute_si_tridiag_bands(
        height_coord, J, dt_s, g, implicit_buoyancy, nlev=nlev,
    )

    def substep_body(carry):
        w_c, theta_p_c, rho_p_c = carry

        theta_total, rho_total = sanitize_theta_rho(
            theta_0 + theta_p_c,
            rho_0 + rho_p_c,
        )

        # --- Explicit RHS for w (same as forward step) ---
        pi_p = compute_exner_perturbation(rho_p_c, theta_p_c, height_coord)
        dpi_dz_inner = (pi_p[..., :-1] - pi_p[..., 1:]) / dz_inner

        theta_half_inner = 0.5 * (theta_total[..., :-1] + theta_total[..., 1:])
        theta_p_half = 0.5 * (theta_p_c[..., :-1] + theta_p_c[..., 1:])
        theta_0_half = 0.5 * (theta_0[:-1] + theta_0[1:])
        buoyancy = g * theta_p_half / theta_0_half

        dw_dt_inner = (
            -c_p * theta_half_inner * dpi_dz_inner / J[..., None]
            + buoyancy
        )

        # RHS of tridiagonal system: w_old + dt_s * explicit_tendency
        rhs = w_c[..., 1:-1] + dt_s * dw_dt_inner

        # Tridiag system reused across substeps (loop-invariant).
        a_tri, b_tri, c_tri = a_tri_pre, b_tri_pre, c_tri_pre

        # Solve tridiagonal system
        w_inner_new = thomas_solve_batched(a_tri, b_tri, c_tri, rhs)

        # Construct full w_new via pad-with-0 (rigid lid/bottom BC),
        # avoiding the dynamic-update-slice fusion barrier that the
        # `w_c.at[..., 1:-1].set(...)` pattern emits on top of the
        # cuSPARSE custom-call output.
        pad_axes_w = ((0, 0),) * (w_inner_new.ndim - 1)
        w_new = jnp.pad(w_inner_new, (*pad_axes_w, (1, 1)))

        # --- Backward: update rho' using updated w ---
        # ``rho_w`` has zero at top/bottom interfaces (rigid lid / rigid
        # bottom).  Single Pad HLO op replaces alloc-zeros + scatter.
        rho_half = 0.5 * (rho_total[..., :-1] + rho_total[..., 1:])
        rho_w = jnp.pad(rho_half * w_inner_new, (*pad_axes_w, (1, 1)))
        vert_div = (rho_w[..., :-1] - rho_w[..., 1:]) / dz
        vert_div = vert_div / J[..., None]
        rho_p_new = rho_p_c - dt_s * vert_div

        # Off-centering: damp acoustic mode (Skamarock & Klemp 2008).
        # Python-guard when beta=0 (default config) — skips a kernel
        # in the substep tail that XLA may not fully fold.
        if beta != 0.0:
            rho_p_new = (1.0 + beta) * rho_p_new - beta * rho_p_c

        # --- Backward: update theta' using vertical w advection ---
        # ``dtheta_dz`` is zero at top/bottom (one-sided would require
        # ghost cells); centred difference fills the interior.  Single
        # Pad HLO op replaces alloc-zeros + scatter.
        w_full = 0.5 * (w_new[..., :-1] + w_new[..., 1:])
        if nlev > 2:
            dz_centered = dz_half[:-1] + dz_half[1:]
            inner_grad = (theta_total[..., :-2] - theta_total[..., 2:]) / dz_centered
            top_grad = (theta_total[..., 0:1] - theta_total[..., 1:2]) / dz_half[0]
            bottom_grad = (
                theta_total[..., -2:-1] - theta_total[..., -1:]
            ) / dz_half[-1]
            dtheta_dz = jnp.concatenate(
                [top_grad, inner_grad, bottom_grad], axis=-1,
            )
        else:
            dtheta_dz = jnp.zeros_like(theta_total)
        theta_p_new = theta_p_c - dt_s * w_full / J[..., None] * dtheta_dz

        return (w_new, theta_p_new, rho_p_new)

    # Python-loop unroll: n_substeps is compile-time static so XLA can
    # fuse the post-cuSPARSE tail of one substep with the pre-cuSPARSE
    # head of the next. lax.fori_loop kept the substeps as a while-loop
    # and prevented inter-iteration fusion.
    w_final, theta_p_final, rho_p_final = (w, theta_p, rho_p)
    for _i in range(int(n_substeps)):
        w_final, theta_p_final, rho_p_final = substep_body(
            (w_final, theta_p_final, rho_p_final),
        )

    return NonHydrostaticState(
        u=state.u,
        v=state.v,
        w=state.w.replace(data=w_final),
        theta_prime=state.theta_prime.replace(data=theta_p_final),
        rho_prime=state.rho_prime.replace(data=rho_p_final),
        phis=state.phis,
        tracers=state.tracers,
    )
