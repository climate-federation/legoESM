# Phase G — Veros recipe audit

## Status

**First-pass audit, revised.** Companion artifact to
``docs/ocean_fidelity/phase_g_recipe_fidelity_plan.md``. Maps Veros's
canonical scheme choices onto legoESM's existing dispatch menus and flags
every dimension where the two diverge.

**Framing principle.** "Faithful match" means matching *schemes*, not just
constants. Where Veros uses scheme X and legoESM does not have X, the
correct response is to add X to legoESM — not to substitute a near-cousin
scheme and widen the tier-2 tolerance. The recipe-as-feature-work principle
from the Phase G plan rules out shortcuts at the scheme level.

**Anchor point.** First-pass test targets are drawn from **Veros's built-in
setup gallery** (``veros.setups.*``), not from custom legoESM ports. Veros's
gallery is the authoritative enumeration of what "running Veros" means in
practice; matching against those configurations is what the recipe acceptance
gate actually has to satisfy.

Audit sources:

- Veros recipe surface as exposed via
  ``src/legoesm/ocean/fidelity/veros_configs/{acc_channel,global_overturning,
  lock_exchange,overflow,eady_uniform,dino}.py`` — the first two wrap Veros's
  built-in ``ACCSetup`` and ``GlobalFourDegreeSetup`` directly; the others
  are custom adapters that mirror legoESM experiments and are not anchored
  in Veros's gallery.
- legoESM scheme dispatch literals from
  ``ocean/eos.py``, ``ocean/state.py``, ``ocean/mpas_config.py``, the
  per-physics ``config.py`` under
  ``ocean/physics/{vertical_mixing,lateral_mixing,convection,bottom_drag,
  surface_forcing}/``, and ``timestepping/``.
- Existing bulk-metric harness in
  ``scripts/ocean_fidelity/compare_legoesm_vs_veros.py`` (Veros linear-EOS
  coefficients + partial Eady/DINO recipe mapping; not gallery-anchored).

Veros is not installed on the audit host (the harness assumes a sibling
``../Veros`` checkout per ``veros_runner._require_veros``). The audit
reads Veros's API surface through the adapters that import ``veros.*``
directly. Audit blind spots are listed at the end.

## Veros's canonical gallery as test targets

Veros's built-in setup gallery (``veros.setups.*``) is the canonical
enumeration. Two are already wrapped in legoESM:

| Veros gallery setup | Wrapper in legoESM | Status |
|---|---|---|
| ``veros.setups.acc.acc.ACCSetup`` | ``veros_configs/acc_channel.py`` | wired, 30×42×15 |
| ``veros.setups.global_4deg.global_4deg.GlobalFourDegreeSetup`` | ``veros_configs/global_overturning.py`` | wired, requires ~50 MB forcing-data download |
| ``veros.setups.global_1deg`` | (not wrapped) | gap |
| ``veros.setups.global_flexible`` | (not wrapped) | gap |
| ``veros.setups.north_atlantic`` | (not wrapped) | gap |
| ``veros.setups.wave_propagation`` | (not wrapped) | gap |

Verification of the unwrapped names against a local Veros install is a
work item under *Audit blind spots*. Wrappers are thin — ``acc_channel.py``
is ~30 LOC including docstring.

**Recommended G.0 first target: ``acc``.** Reasons:

1. Already wrapped, no new adapter needed.
2. Smaller than ``global_4deg`` (30×42×15, no forcing-file download).
3. Built-in Veros canonical example, derived from pyOM2.
4. Exercises GM/Redi + TKE + harmonic friction in their canonical Veros
   form, so the recipe acceptance is meaningful (it tests the closure path,
   not just the trivial path).
5. The existing bulletproof tier-3 gate does *not* exercise ACC today, so
   adding ACC into the Phase G workflow also strengthens tier-3 coverage.

Configurations in the gallery escalate naturally: ``wave_propagation`` →
``acc`` → ``global_4deg`` → ``global_1deg`` → ``north_atlantic`` →
``global_flexible``. Smaller / more idealized first, full GCM later.

Custom adapters (``lock_exchange``, ``overflow``, ``eady_uniform``, ``dino``)
remain useful for tier-3 bulk-metric coverage but are **not** the Phase G
recipe-acceptance targets. The legoESM-Veros recipe is defined by Veros's
own canonical configurations.

## Veros recipe surface (per-dimension)

Drawing on the adapter files and the ACC / global_4deg gallery cases that
they wrap:

1. **Equation of state** — ``eq_of_state_type ∈ {1, 2, 3, 4, 5}``.
   Verified by reading Veros source ``veros/core/density/``:
   ``1``: linear (``betaT=1.67e-4``, ``betaS=0.78e-3``, ``rho_0=1024``);
   ``2``: nonlinear variant 1;
   ``3``: nonlinear variant 3 — a **quadratic-in-T** polynomial
   (``rho = -[βT·θ + βTs·θ² − βS·(S−S₀)]·ρ₀`` with ``βS=0`` by
   default, i.e., **no salinity dependence**), used by ``ACCSetup``.
   This is NOT Jackett-McDougall 1995 as the earlier draft of this
   audit claimed.
   ``4``: nonlinear variant 2;
   ``5``: gsw / TEOS-10.
2. **Tracer advection** — ``enable_superbee_advection = True`` (Sweby
   superbee flux limiter, canonical for ACC / global_4deg). Otherwise
   centred second-order.
3. **Momentum advection** — C-grid energy/enstrophy-conserving
   stencil (Sadourny); built into the core, not configurable.
4. **Coriolis** — ``vs.coriolis_t = 2·Ω·sin(yt)`` at velocity points;
   C-grid Sadourny stencil baked in.
5. **Lateral momentum mixing** — ``enable_hor_friction``, ``A_h``,
   ``enable_hor_friction_cos_scaling``, ``hor_friction_cosPower``
   (cos(lat)-scaled harmonic Laplacian). No built-in biharmonic.
6. **Bottom drag** — ``enable_bottom_friction`` + ``r_bot`` (linear only).
7. **Vertical friction** — ``enable_implicit_vert_friction`` (implicit
   tridiagonal ``A_v ∂²u/∂z²``).
8. **Vertical mixing closure** — ``enable_tke`` (Gaspar et al. 1990 /
   Burchard 2002 TKE closure with ``c_k``, ``c_eps``, ``alpha_tke``,
   ``mxl_min``, ``tke_mxl_choice``, ``kappaM_min``, ``kappaH_min``,
   ``enable_kappaH_profile``); ``enable_eke`` (Eden & Greatbatch 2008);
   ``enable_idemix`` (Olbers & Eden 2013 internal-wave-driven mixing).
9. **Lateral tracer mixing** — ``enable_neutral_diffusion = True`` (Redi)
   with ``K_iso_0``, ``K_iso_steep``, ``iso_dslope``, ``iso_slopec``;
   ``enable_skew_diffusion = True`` (GM-skew) with ``K_gm_0``.
10. **Time integrator** — leapfrog + Adams-Bashforth-2 + Robert-Asselin
    time filter; not configurable.
11. **Grid staggering** — Arakawa **C-grid** (confirmed by reading Veros
    ``variables.py`` — ``U_GRID = ("xu", "yt", "zt")``,
    ``V_GRID = ("xt", "yu", "zt")``). **Same staggering as legoESM's
    lat-lon C-grid.** Earlier drafts of this audit and a comment in
    ``scripts/ocean_fidelity/compare_legoesm_vs_veros.py`` (line 158)
    described Veros as B-grid; that is incorrect.
12. **Free surface** — implicit linear free-surface solver; not configurable.
13. **Constants** — ``rho_0 = 1024`` (linear EOS), ``grav = 9.81``,
    ``omega = 7.292115e-5``, ``radius = 6.370e6``, ``betaT = 1.67e-4``,
    ``betaS = 0.78e-3``.

## legoESM dispatch surface (ocean)

| Dimension | legoESM dispatch | Module |
|---|---|---|
| EOS | ``"wright"``, ``"linear"`` (with ``LinearEOSConfig``) | ``ocean/eos.py`` |
| Tracer advection (lat-lon C-grid) | ``"upwind"``, ``"tvd"``, ``"ppm"``, ``"ppm_fct"``, ``"dst3"``, ``"dst3_multidim"``, ``"som"``, ``"weno5"``, ``"weno7"`` | ``ocean/state.py``, ``ocean/dynamics/ocean_model_latlon_cgrid.py`` |
| Tracer advection (MPAS) | ``"upwind"``, ``"tvd"`` | ``ocean/mpas_config.py`` |
| Momentum advection | ``"vector_invariant"``, ``"weno5"``, ``"weno7"`` | ``ocean/state.py`` |
| PGF scheme | ``"adcroft"``, ``"smc03"``, ``"ahh08"`` (requires ``eos="wright"``), ``"centered"``, ``"zero"`` | ``ocean/dynamics/ocean_pe_{latlon_cgrid,mpas}.py`` |
| Lateral mixing | ``"harmonic"``, ``"biharmonic"``, ``"gm_redi"``, ``"none"`` (slope ``"triads"``/``"centered"``) | ``ocean/physics/lateral_mixing/config.py`` |
| Vertical mixing | ``"constant"``, ``"richardson"``, ``"kpp"``, ``"tidal"``, ``"none"`` | ``ocean/physics/vertical_mixing/config.py`` |
| Convection | ``"enhanced_diffusion"``, ``"plume"``, ``"none"`` | ``ocean/physics/convection/config.py`` |
| Bottom drag | ``"linear"``, ``"quadratic"``, ``"none"`` | ``ocean/physics/bottom_drag/config.py`` |
| Surface forcing | ``"prescribed"``, ``"restoring"``, ``"combined"``, ``"bulk_formulas"``, ``"none"`` (bulk: ``"constant"``, ``"coare3"``, ``"large_yeager"``) | ``ocean/physics/surface_forcing/config.py`` |
| Outer time integrator | ``"ssp_rk3"``, ``"ssp_rk34"``, ``"ssp_rk54"`` | ``timestepping/split_explicit.py`` |
| Grid staggering | C-grid only (lat-lon, MPAS Voronoi, cubed-sphere) | — |
| Constants | ``g=9.80616``, ``R_earth=6.371229e6``, ``rho_ocean=1025``, ``T_freeze=273.15`` | ``legoesm/constants.py`` |

## Per-dimension Veros → legoESM mapping

Status legend:

- **clean** — legoESM has Veros's exact choice; recipe pins parameter values.
- **wire** — legoESM has the implementation but it is not currently the
  default or dispatched literal needed; small wiring change.
- **add** — legoESM needs to add the scheme. The match-very-well principle
  rules out substitution; this is feature work.
- **structural** — difference cannot be expressed in legoESM's menu without a
  deeper architectural change (rare in practice — most "structural"
  rows turned out to be misclassifications once Veros source was
  read; see the audit-update commit ``c5acd28-ish``).

| Dimension | Veros canonical | legoESM today | Status | Action |
|---|---|---|---|---|
| EOS (linear) | ``eq_of_state_type=1`` + Veros hard-coded coefs | ``eos="linear"`` + ``LinearEOSConfig`` | **clean** | Pin coefficients via recipe. |
| EOS (nonlinear, ``type=3``) | **Quadratic-in-T polynomial with βS=0** (verified against ``veros/core/density/nonlinear_eq3.py``; NOT JM95 as the original audit draft claimed). Used by ``ACCSetup``. | ``eos="veros_nonlin3"`` — bit-for-bit-equivalent ``veros_nonlin3_eos`` ported into ``ocean/eos.py``. | **clean** | Pin ``VerosNonlin3Config()`` defaults. JM95 / TEOS-10 ports remain as future work for ``eq_of_state_type=4, 5`` setups. |
| Tracer advection (superbee) | ``enable_superbee_advection=True`` (Sweby superbee limiter) | ``"tvd"`` dispatch wires ``_van_leer_limiter`` — Sweby is implemented as ``_sweby_limiter`` but not wired. | **wire** | **Either** (a) add a new ``tracer_advection="superbee"`` literal that calls ``_sweby_limiter``, or (b) add a ``tvd_limiter: Literal["van_leer", "sweby"]`` config knob with ``"van_leer"`` default. Recipe pins ``"superbee"`` / ``tvd_limiter="sweby"``. Small change. |
| Momentum advection (C-grid) | built-in | C-grid ``"vector_invariant"`` (default), ``"weno5"``, ``"weno7"`` | **close** | Both C-grid (Veros confirmed via ``variables.py``). Stencil family differs (Sadourny vs vector-invariant) but staggering is identical, so tier-2 momentum-advection comparison should land at discretization-truncation level. |
| Coriolis (C-grid) | Sadourny energy-conserving | C-grid energy-conserving | **clean** | Same staggering, same family — pin via recipe with no documented stencil delta. |
| Lateral momentum mixing (harmonic) | ``enable_hor_friction=True``, ``A_h`` | ``lateral_mixing="harmonic"``, ``A_h`` | **clean** | Pin ``A_h``. |
| Cos(lat) ``A_h`` scaling | ``enable_hor_friction_cos_scaling=True``, ``hor_friction_cosPower=1`` | not exposed in ``lateral_mixing/harmonic.py`` (verify) | **add** | **Add ``A_h_cos_power: float = 0.0``** field to ``HarmonicMixingConfig`` and apply ``A_h · cos(lat)^N`` inside ``harmonic.py``. Trivial change (~10 LOC + test). Required by global_4deg and DINO. |
| Bottom drag (linear) | ``enable_bottom_friction=True``, ``r_bot`` | ``bottom_drag="linear"``, ``r_bot`` | **clean** | Pin ``r_bot``. (legoESM also has ``"quadratic"`` which Veros lacks — a legoESM superset.) |
| Vertical friction (implicit) | ``enable_implicit_vert_friction=True`` | ``vertical_mixing/implicit_solver.py`` present but verify dispatch wiring | **wire** | Verify ``vertical_mixing="constant"`` with ``implicit=True`` routes through the implicit solver. If not, wire it. (Audit blind spot — re-verify with a local Veros install.) |
| Vertical mixing closure (TKE) | ``enable_tke=True`` (Gaspar 1990 / Burchard 2002 prognostic TKE closure) | only ``"constant"``, ``"richardson"``, ``"kpp"``, ``"tidal"`` available | **add** | **Port Veros's TKE module** to legoESM as ``vertical_mixing scheme="tke"``. Estimated 400–600 LOC: prognostic TKE budget, mixing-length closure (``tke_mxl_choice``), surface flux ``forc_tke_surface``, ``c_k``/``c_eps``/``alpha_tke`` parameters, ``kappaM_min``/``kappaH_min`` floors, ``enable_kappaH_profile`` taper, tridiagonal coupling to the implicit vertical solver. Plus unit tests against analytic Gaspar 1990 limits. **Required by ACC and every non-trivial gallery setup.** |
| EKE closure | ``enable_eke=True`` (Eden & Greatbatch 2008) | not in legoESM | **add** | Port. Estimated 200–300 LOC. Required by some gallery setups; sequence after TKE. |
| IDEMIX (internal-wave mixing) | ``enable_idemix=True`` (Olbers & Eden 2013) | not in legoESM | **add** | Port. Estimated 300–400 LOC. Required by ``global_flexible`` and the high-fidelity ``global_*`` recipes. Sequence after EKE. |
| Redi (neutral diffusion) | ``enable_neutral_diffusion=True``, ``K_iso_0``, ``K_iso_steep``, ``iso_dslope``, ``iso_slopec`` | ``lateral_mixing="gm_redi"`` (Redi component) | **clean** / **wire** | Verify legoESM's taper parameter names map cleanly to Veros's (``iso_dslope``, ``iso_slopec``, ``iso_steep``). If parameter sets differ, expose a Veros-compatible adapter in the config. Modest work. |
| GM-skew (eddy bolus) | ``enable_skew_diffusion=True``, ``K_gm_0`` | ``lateral_mixing="gm_redi"`` (GM component) | **clean** | Pin ``K_gm_0``. |
| Time integrator (leapfrog + AB2 + Robert-Asselin) | hard-coded in core | ``"ssp_rk3"`` / ``"ssp_rk34"`` / ``"ssp_rk54"`` / semi-implicit | **add** | **Port leapfrog + AB2 + Robert-Asselin filter** to legoESM as ``outer_integrator="leapfrog_ab2"``. Engineering questions: (1) the Robert-Asselin time blend is a linear combination of ``τ-1, τ, τ+1`` — autodiff-friendly (linear). (2) leapfrog requires a two-level state carry; integrates cleanly into ``SegmentCarry`` via an additional pytree field but it is a cross-cutting change (every direct ``SegmentCarry(...)`` constructor must learn the new field, per CLAUDE.md SegmentCarry discipline). Estimated 500–800 LOC including state-carry plumbing + tests. **Required for tier-3 match with Veros, and for tier-2 of any integrator-coupled tendency (forced drift over a step is integrator-dependent).** |
| Grid staggering | Arakawa **C-grid** (verified against ``veros/variables.py``: ``U_GRID = ("xu", "yt", "zt")``, ``V_GRID = ("xt", "yu", "zt")``) | Arakawa C-grid (lat-lon, MPAS Voronoi, cubed-sphere) | **clean** | Same staggering — no B-grid parallel core needed. The "structural delta" recorded in earlier audit drafts was based on an incorrect comment in ``compare_legoesm_vs_veros.py`` (which described Veros as B-grid). Phase G.0c verifies this against Veros source. |
| Free surface (implicit) | hard-coded in core | ``barotropic`` dispatch with implicit barotropic available | **wire** | Pin ``barotropic="implicit"``. |
| Cyclic-X | ``enable_cyclic_x=True`` | grid ``periodic_x=True`` | **clean** | Pin in grid factory. |
| Constants | ``g=9.81``, ``omega=7.292115e-5``, ``radius=6.370e6``, ``rho_0=1024`` (linear EOS), ``betaT=1.67e-4``, ``betaS=0.78e-3`` | ``g=9.80616``, ``R_earth=6.371229e6``, ``rho_ocean=1025`` (verify ``omega`` value in ``constants.py``) | **wire** | **Add recipe-level constants override.** Two options: (a) monkey-patch ``legoesm.constants`` at recipe load (simple, global state); (b) plumb a ``ConstantsConfig`` NamedTuple through every leaf that reads ``constants.X`` (clean, invasive — touches dozens of modules). **Recommended: start with (a) for G.0 to validate the harness; plan (b) as a refactor once the recipe workflow is proven.** ~60 LOC + test for option (a). |

## legoESM development sequence (Phase G)

The audit identifies eight scheme/feature additions on the critical path for
a faithful Veros recipe. Estimated work per item plus suggested ordering:

| # | Item | Effort | Critical for | Sequence |
|---|---|---|---|---|
| 1 | Wire ``superbee`` tracer advection (call ``_sweby_limiter``) | S (~20 LOC) | every Veros gallery setup | G.0a |
| 2 | Recipe-level constants override (option (a)) | S (~60 LOC) | every Veros recipe | G.0a |
| 3 | ``A_h_cos_power`` field in ``HarmonicMixingConfig`` | S (~10 LOC) | ``acc``, ``global_4deg`` | G.0a |
| 4 | Verify implicit vertical-friction wiring; fix if broken | S (audit + maybe wire) | every gallery setup | G.0a |
| 5 | Tendency probe harness + per-region metrics | M (~300 LOC) | every Phase G run | G.0b |
| 6 | First G.0 recipe ``configs/recipes/veros_acc.yaml`` + acceptance run on ``acc`` | M | gate G.0 acceptance | G.0c |
| 7 | Port Veros TKE closure as ``vertical_mixing="tke"`` | M-L (~400–600 LOC) | ``acc``, every non-trivial setup | G.1a (before ACC tier-2 mixing match) |
| 8 | Port Veros nonlin3 EOS as ``eos="veros_nonlin3"`` (the actual ACC EOS — quadratic-in-T, NOT JM95). UNESCO 1980 also landed for general-purpose use; JM95 / TEOS-10 deferred to setups that need ``eq_of_state_type=4, 5``. | S (~50 LOC + tests) | every Veros setup with ``eq_of_state_type=3`` | G.1b |
| 9 | Veros taper-parameter mapping in ``gm_redi.py`` | S | ``acc``, every GM/Redi setup | G.1c |
| 10 | Port leapfrog + AB2 + Robert-Asselin as ``outer_integrator="leapfrog_ab2"`` | L (~500–800 LOC + SegmentCarry plumbing) | tier-3 match across all setups; tier-2 of any integrator-coupled tendency | G.2 |
| 11 | Port EKE (Eden & Greatbatch 2008) | M (~200–300 LOC) | gallery setups with ``enable_eke=True`` | G.3 |
| 12 | Port IDEMIX (Olbers & Eden 2013) | M-L (~300–400 LOC) | ``global_flexible`` recipe | G.3 |
| 13 | Wrap remaining gallery setups (``global_1deg``, ``global_flexible``, ``north_atlantic``, ``wave_propagation``) | S each (~30 LOC + test) | gallery coverage | rolling |
| — | ~~B-grid parallel ocean core~~ | — | — | **NOT REQUIRED** — Veros is C-grid (verified against ``veros/variables.py``). |

Items 1–6 unblock G.0 (the first tier-2 acceptance run, on ``acc``).
Item 7 (TKE port) unblocks the mixing-tendency match on ``acc`` and is the
largest scheme-level item on the G.1 critical path. Item 10 (leapfrog) is
required to close the integrator gap to Veros; without it the tier-3
free-run statistics will always show drift, even if every tier-2 tendency
matches.

## Recipe spec for G.0 (``acc``)

Based on the audit, the first ``legoesm-veros`` recipe target is
``configs/recipes/veros_acc.yaml`` — wrapping ``veros.setups.acc.acc.ACCSetup``:

```yaml
# configs/recipes/veros_acc.yaml
# Veros ACC channel (veros.setups.acc.acc.ACCSetup) — 30x42x15 re-entrant
# channel, pyOM2-derived. First-pass Phase G acceptance target.

constants:
  g: 9.81                       # ↔ veros.settings.grav
  R_earth: 6.370e6              # ↔ veros.settings.radius
  rho_ocean: 1024.0             # ↔ veros.settings.rho_0
  omega: 7.292115e-5            # ↔ veros.settings.omega

grid:
  type: latlon_cgrid_regional   # both Veros and legoESM are C-grid; no staggering delta
  nx: 30
  ny: 42
  nz: 15
  periodic_x: true              # ↔ enable_cyclic_x

vertical:
  type: z_star                  # confirm against ACC's vertical grid
  H_max: <from ACCSetup>
  dz_profile: <from ACCSetup>

eos:
  scheme: linear                # ↔ eq_of_state_type=1
  rho_ref: 1024.0
  alpha_T: 1.67e-4
  beta_S: 0.78e-3
  T_ref: <from ACCSetup>
  S_ref: <from ACCSetup>

dynamics:
  tracer_advection: superbee    # NEW LITERAL — pending G.0a item #1
  momentum_advection: vector_invariant   # C-grid analogue; tier-2 documents stencil delta
  pgf_scheme: adcroft           # legoESM canonical; closest functional form to Veros

barotropic:
  scheme: implicit              # ↔ Veros implicit free-surface

physics:
  lateral_mixing:
    scheme: harmonic
    A_h: <from ACCSetup>
    A_h_cos_power: 1.0          # NEW FIELD — pending G.0a item #3
  vertical_mixing:
    scheme: tke                 # NEW SCHEME — pending G.1a item #7
    tke:
      c_k: <from ACCSetup>
      c_eps: <from ACCSetup>
      alpha_tke: <from ACCSetup>
      mxl_min: <from ACCSetup>
      tke_mxl_choice: <from ACCSetup>
      kappaM_min: <from ACCSetup>
      kappaH_min: <from ACCSetup>
      enable_kappaH_profile: <from ACCSetup>
    implicit: true              # ↔ enable_implicit_vert_friction
  lateral_tracer_mixing:
    scheme: gm_redi             # ↔ enable_neutral_diffusion + enable_skew_diffusion
    K_iso_0: <from ACCSetup>
    K_iso_steep: <from ACCSetup>
    iso_dslope: <from ACCSetup>
    iso_slopec: <from ACCSetup>
    K_gm_0: <from ACCSetup>
  bottom_drag:
    scheme: linear
    r_bot: <from ACCSetup>
  convection:
    scheme: none                # ACC turns off ad-hoc convective adjustment
  surface_forcing:
    scheme: prescribed          # ACC uses prescribed wind stress + restoring; verify against ACCSetup
    # ... fields filled in from ACCSetup

time_integrator:
  outer: leapfrog_ab2           # NEW SCHEME — pending G.2 item #10
                                # Until G.2 lands, run with ssp_rk3 and document
                                # tier-3 delta. Tier-2 acceptance is integrator-
                                # independent for un-integrated tendency match.

validation_status:
  tier_1: not_applicable
  tier_2: g_0_target
  tier_3: untested              # ACC not currently in bulletproof gate
  tier_4: not_attempted

gaps_closed_in_this_recipe:
  - superbee_tvd_wiring         # G.0a item #1
  - constants_override          # G.0a item #2
  - cos_lat_a_h                 # G.0a item #3
  - tke_closure                 # G.1a item #7 — required for ACC

gaps_documented_not_closed:
  - leapfrog_ab2_integrator     # G.2 item #10 — tier-3 only at G.0/G.1 acceptance
  - b_grid_staggering           # deferred — tier-2 documents stencil delta
```

Empty ``<from ACCSetup>`` slots get filled in once a local Veros install is
available and we can introspect ``ACCSetup().setup().state.settings``.

## G.0 acceptance plan (revised)

G.0 targets ``acc`` (Veros gallery), single-process tendency comparison.
Revised ordering:

1. **G.0a — scheme prerequisites** (items #1–4 from the development sequence).
   No Phase G acceptance runs until these land.
2. **G.0b — tendency probe harness** (item #5). Sets up the comparison
   infrastructure: snapshot ingest, isolated-tendency callsite, per-region
   metrics, per-process Markdown report.
3. **G.0c — first acceptance run** (item #6). Wire ``veros_acc.yaml`` and
   run the comparison. Tracer advection (now superbee on both sides),
   PGF (linear EOS), Coriolis (with B/C-grid stencil delta documented),
   GM/Redi (with parameter mapping closed), harmonic lateral friction
   (with cos(lat) scaling). Vertical mixing comparison is **expected to
   fail** at G.0c because TKE is not yet ported (item #7); that failure
   is the acceptance criterion for moving to G.1.

Acceptance gate for G.0c (per-process): pattern correlation > 0.99 on
tracer-advection ``dT/dt`` and PGF ``du/dt``; per-region L2 within 2× scheme
truncation error in the interior; sign-match > 0.999 in the interior. The
non-blocking mixing-tendency failure is the trigger for G.1a.

## G.1 acceptance plan

After G.0c lands:

1. **G.1a — TKE port** (item #7). Add ``vertical_mixing="tke"`` to legoESM.
2. **G.1b — DONE**: Veros's actual ``eq_of_state_type=3`` is quadratic-T (NOT JM95). Ported as ``eos="veros_nonlin3"`` with bit-for-bit parity against ``veros/core/density/nonlinear_eq3.py``. UNESCO 1980 also landed as ``eos="unesco80"`` for setups needing a real-world nonlinear EOS.
3. **G.1c — Re-run ``veros_acc.yaml`` with TKE wired**. Mixing tendency now
   passes the per-process gate.
4. **G.1d — Add ``veros_global_4deg.yaml``** (uses JM95 if ``global_4deg``
   turns on nonlinear EOS; verify). Run G.1 acceptance.

## Phase G–level open questions (revised)

1. **B-grid acceptance boundary**. Under "match very well", how much
   discretization-level delta in momentum advection / Coriolis stencils is
   acceptable in tier-2 before we are forced to add a parallel B-grid
   core? Needs a quantitative gate set after G.0c sees the actual delta on
   ``acc``.
2. **Leapfrog AD audit**. The Robert-Asselin filter is linear, and
   leapfrog is fine for ``jax.grad`` in principle, but the two-level state
   carry interacts with checkpointing and donation. Needs a small spike
   *before* committing to the full port (item #10).
3. **TKE differentiability**. The prognostic TKE equation has source terms
   that involve clipping (e.g. ``max(TKE, mxl_min)``) — these break smooth
   gradients. Veros has the same issue and runs fine forward; AD-mode
   correctness for tuning loops needs verification.
4. **Constants override v1 → v2 migration**. Option (a) monkey-patch is
   the v1 plan; option (b) plumbed ``ConstantsConfig`` is the v2 long-term
   plan. When does v2 become required? Likely when novel mixed recipes
   (Phase G.4) compose two different constants sets in the same process —
   monkey-patching can't represent that.

## Audit blind spots

What this audit does NOT cover and should be filled in before G.1
production work:

1. **Veros gallery enumeration**. Names of unwrapped gallery setups
   (``global_1deg``, ``global_flexible``, ``north_atlantic``,
   ``wave_propagation``) are from common Veros knowledge and have NOT been
   verified against a local Veros install. Verify import paths before
   spending work on wrappers.
2. **Implicit vertical-friction dispatch wiring** — whether
   ``vertical_mixing="constant"`` actually routes through
   ``implicit_solver.py`` was not traced in the audit. Item #4 above.
3. **Redi taper functional form** — the exact mapping between Veros's
   ``iso_dslope``/``iso_slopec``/``iso_steep`` and legoESM's tapering in
   ``gm_redi.py`` was not verified line-by-line. Item #9 above.
4. **Surface forcing parameter mapping for ``acc``** — ACC's wind stress
   profile and T/S restoring formulas live in
   ``veros.setups.acc.acc.ACCSetup`` and need to be read off and pinned in
   ``veros_acc.yaml`` once Veros is installed.
5. **Coriolis sign convention** — both Veros (B-grid) and legoESM
   (C-grid) compute ``2·Ω·sin(lat)``, but stencil sign conventions at
   velocity points differ. Tier-2 must explicitly account for the
   staggering when computing the residual.
6. **Veros default settings for knobs NOT touched in any wrapped setup**.
   The audit covers the ``acc`` + ``global_4deg`` surface; setups like
   ``global_flexible`` will turn on additional knobs (EKE, IDEMIX,
   variable bathymetry tapers) that this audit does not address.

## Relation to existing work

- **Predecessor**: ``docs/ocean_fidelity/phase_g_recipe_fidelity_plan.md``.
  This audit is the gap-accounting artifact promised under the
  "Deferred work" section. The plan's G.0 (``Veros 4° global``) is
  retargeted here to ``acc`` for size reasons; both are Veros gallery
  setups.
- **Bulletproof gate**: ``docs/ocean_fidelity/bulletproof_summary.md``
  reports 6 / 6 Eady-uniform + 9 / 9 DINO bulk metrics at 5% tier-3
  tolerance — but Eady and DINO are *custom* legoESM ports, not Veros
  gallery configurations, so the bulletproof scoreboard does not yet cover
  the Phase G recipe-acceptance surface. Phase G strengthens *and*
  broadens it.
- **Existing harness reused**: ``scripts/ocean_fidelity/compare_legoesm_vs_veros.py``
  encodes Veros's linear-EOS coefficients and the ``_VEROS_*`` constants
  that the recipe override mechanism will replace.
