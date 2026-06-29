"""Unified ocean case board — the single inventory of every ocean experiment/case,
what it tests, whether a true oracle exists in another model, and whether legoESM
reproduces it (stable / matches / blocked).

This is the source of truth; ``scripts/validate/ocean_fidelity/build_recipe_case_board.py``
renders it to a table, and ``tests/ocean/fidelity/test_recipe_case_board.py`` is the
RATCHET: every registered experiment (``AVAILABLE_EXPERIMENTS``) AND every oracle
comparison driver (``scripts/validate/ocean_fidelity/compare_*.py``) MUST appear here,
so a new case can't be added without classifying it. Holes (``todo``) are the roadmap;
``blocked`` cells link to the issue explaining the fundamental limitation.

NO distinction between idealized and oracle cases — one board. The ``tests`` column is
deliberately a short physics summary so overlapping/duplicate cases are visible at a
glance (the board doubles as a retire-the-repeats inventory).

Honesty rule: a cell is ``verified``/``partial``/``blocked`` ONLY with a concrete
measured basis (PR/issue/metric); everything not yet assessed is ``todo`` — the board
must not over-claim.
"""

from __future__ import annotations

# --- Status vocabulary -------------------------------------------------------
VERIFIED = "verified"   # reproduces a real oracle (or analytic truth): stable + matches, tested
WORKS = "works"         # runs stably, but no external oracle to match against (idealized)
PARTIAL = "partial"     # runs / early-matches but does not meet the fidelity bar
BLOCKED = "blocked"     # applicable but fundamentally not achievable yet — see ref (issue)
TODO = "todo"           # not yet assessed — a hole to fill
NA = "n/a"              # not applicable

VALID_STATUSES = frozenset({VERIFIED, WORKS, PARTIAL, BLOCKED, TODO, NA})


# --- The board ---------------------------------------------------------------
# Each row: case, aliases (registry/driver names that map to this row), tests
# (1-line physics summary, for spotting duplicates), grids, oracle (external truth
# model or None=idealized), recipe (the config/dycore identity used), status, note.
# ``aliases`` lets one physical case absorb its several registry names (e.g. the
# experiment "baroclinic" == the oracle case "baroclinic_adjustment").
CASES: tuple[dict, ...] = (
    # ---- cases with an external oracle (verified/blocked/partial known this session) ----
    dict(case="baroclinic_adjustment",
         aliases=("baroclinic", "oceananigans_baroclinic_adjustment", "oceananigans_spherical_baroclinic"),
         tests="stratified baroclinic instability; eddy growth + inverse cascade (unforced)",
         grids=("latlon_cgrid (β-plane)",), oracle="Oceananigans WENOVectorInvariant(9)+ImplicitFreeSurface",
         recipe="oceananigans + power_law stack", status=VERIFIED,
         note="EKE 0.86×, cascade k_e 5.11 vs 5.19, stable 40 d, no backstop — PR #672"),
    dict(case="silvestri_jet_forced",
         aliases=("silvestri_baroclinic_jet", "silvestri", "oceananigans_silvestri_s5"),
         tests="forced (τ=50 d restoring) eddy-resolving baroclinic jet; Silvestri 2024 §5",
         grids=("latlon_cgrid (spherical channel)",), oracle="Oceananigans Silvestri 2024 §5",
         recipe="oceananigans + power_law stack", status=BLOCKED,
         note="over-energizes ~7× under sustained restoring (eddy-mean equilibration) — issue #673"),
    dict(case="barotropic_gyre",
         aliases=("oceananigans_barotropic_gyre",),
         tests="wind-driven barotropic gyre spin-up (steady, laminar window)",
         grids=("latlon_cgrid",), oracle="Oceananigans ImplicitFreeSurface gyre",
         recipe="oceananigans (implicit_cn)", status=VERIFIED,
         note="surface-u pattern corr 0.93 days 1–20 (kinematic wind-stress unit fix)"),
    dict(case="bickley_jet",
         aliases=("oceananigans_bickley_jet",),
         tests="single-layer Bickley-jet barotropic instability; vortex roll-up (eddy dissipation)",
         grids=("latlon_cgrid (unit sphere)",), oracle="Oceananigans WENOVectorInvariant",
         recipe="oceananigans (implicit_cn)", status=PARTIAL,
         note="tracks oracle to t6 (corr 0.97) then under-dissipates; enstrophy ~1.2–1.9× late"),
    dict(case="internal_tide",
         aliases=("oceananigans_internal_tide", "oceananigans_internal_tide_w"),
         tests="tidal flow over topography; internal-tide generation (Flat-y, free surface)",
         grids=("latlon_cgrid (meridionally_flat)",), oracle="Oceananigans internal_tide",
         recipe="oceananigans (meridionally_flat + upwind3)", status=VERIFIED,
         note="corr ≥0.6 through ~2 days; dispersion cleared by upwind3 — #576"),
    dict(case="geostrophic_adjustment",
         aliases=("oceananigans_geostrophic_adjustment",),
         tests="geostrophic adjustment from a temperature front; free-surface coupling",
         grids=("latlon_cgrid", "cubed_sphere", "mpas"), oracle="Oceananigans",
         recipe="oceananigans", status=TODO, note="comparison driver exists; not re-assessed post-#501"),
    dict(case="gridmode_decay",
         aliases=("oceananigans_gridmode_decay",),
         tests="2Δx grid-mode decay rate (numerical dissipation probe)",
         grids=("latlon_cgrid",), oracle="Oceananigans", recipe="oceananigans",
         status=TODO, note="diagnostic comparison; not re-assessed"),
    dict(case="acc_channel",
         aliases=("acc", "veros_acc", "veros_acc_basic", "tendencies_acc"),
         tests="ACC-like re-entrant channel with ridge; eddy saturation + transport",
         grids=("latlon_cgrid", "mpas"), oracle="Veros ACC",
         recipe="veros_faithful_v1", status=TODO, note="extensive Veros ACC work exists — re-confirm via scorecard"),
    dict(case="advection_gyre",
         aliases=("mitgcm_advection_gyre",),
         tests="passive-tracer advection in a wind-driven gyre",
         grids=("latlon_cgrid",), oracle="MITgcm tutorial_advection_in_gyre",
         recipe="mitgcm", status=TODO, note="comparison driver exists; not re-assessed"),
    dict(case="global_omip",
         aliases=("veros_global_1deg", "veros_global_4deg", "veros_global_flexible",
                  "legoesm_vs_veros", "omip_nemo_match_tripole", "omip_nemo_match_mpas"),
         tests="global forced (OMIP) ocean climate; SST/MOC vs reference GCM",
         grids=("latlon_cgrid (tripole)", "mpas"), oracle="Veros / NEMO (OMIP)",
         recipe="veros_faithful_v1 / nemo_v1 / omip_nemo_match_*", status=TODO,
         note="OMIP NEMO-match recipes exist (SST RMSE ~0.8–1.2) — re-confirm via scorecard"),

    # ---- idealized cases (no external oracle; analytic/benchmark validation) ----
    dict(case="rest_state",
         aliases=("rest_state_stratified_with_land", "rest_state_uniform_with_land",
                  "rest_state_stratified_no_land", "rest_state_uniform_no_land"),
         tests="rest-state stability / PGF balance (η≈0); conservation control",
         grids=("latlon_cgrid", "cubed_sphere", "mpas"), oracle=None,
         recipe="default_wright_v1", status=TODO, note="analytic truth (η≈0); covered by matrix — confirm status"),
    dict(case="barotropic_wave",
         aliases=(), tests="barotropic gravity-wave propagation (Gaussian SSH)",
         grids=("latlon_cgrid", "cubed_sphere", "mpas", "spectral"), oracle=None,
         recipe="default_wright_v1", status=TODO, note="analytic dispersion; matrix-covered"),
    dict(case="inertia_gravity_wave",
         aliases=(), tests="Poincaré (inertia-gravity) wave; Bishnu et al. 2024",
         grids=("latlon_cgrid", "cubed_sphere", "mpas"), oracle="analytic / Bishnu 2024",
         recipe="default_wright_v1", status=TODO, note="analytic benchmark; matrix-covered"),
    dict(case="lock_exchange",
         aliases=(), tests="density-driven gravity current; RPE mixing (Petersen 2015)",
         grids=("latlon_cgrid",), oracle="Petersen et al. 2015 (RPE benchmark)",
         recipe="default_wright_v1", status=TODO, note="RPE diagnostic benchmark; matrix-covered"),
    dict(case="overflow",
         aliases=(), tests="dense water descending a bathymetric slope",
         grids=("latlon_cgrid", "cubed_sphere"), oracle=None,
         recipe="default_wright_v1", status=TODO, note="matrix-covered"),
    dict(case="phillips_two_layer",
         aliases=(), tests="Phillips two-layer baroclinic instability",
         grids=("latlon_cgrid", "cubed_sphere", "mpas"), oracle=None,
         recipe="default_wright_v1", status=TODO, note="matrix-covered"),
    dict(case="eady_uniform",
         aliases=(), tests="classical Eady instability (uniform N², linear shear), re-entrant channel",
         grids=("latlon_channel", "mpas_channel"), oracle="analytic Eady growth rate",
         recipe="eady_weno5_v1", status=TODO, note="analytic growth-rate benchmark"),
    dict(case="eady_instability",
         aliases=(), tests="Eady instability from a meridional temperature front",
         grids=("latlon_channel",), oracle=None, recipe="eady_weno5_v1",
         status=TODO, note="POSSIBLE DUPLICATE of eady_uniform — review for retire"),
    dict(case="held_larichev",
         aliases=(), tests="Held–Larichev eddying channel; APE→KE cascade saturation (k⁻³)",
         grids=("latlon_channel", "mpas_channel"), oracle="Held & Larichev (spectral law)",
         recipe="default_wright_v1", status=TODO, note="spectral-slope benchmark"),
    dict(case="stommel_gyre_tracer",
         aliases=(), tests="passive tracer in a wind-driven Stommel gyre (Hecht 2000)",
         grids=("latlon_cgrid", "mpas"), oracle=None, recipe="default_wright_v1",
         status=TODO, note="matrix-covered"),
    dict(case="munk_gyre",
         aliases=(), tests="Munk gyre; lateral-viscosity western boundary current",
         grids=("latlon_regional", "mpas_regional"), oracle="analytic Munk BL width",
         recipe="default_wright_v1", status=TODO, note="WBC benchmark"),
    dict(case="baroclinic_gyre",
         aliases=("regional_gyre",), tests="wind-driven regional gyre with surface restoring + thermal wind",
         grids=("latlon_regional", "mpas_regional"), oracle=None,
         recipe="default_wright_v1", status=TODO, note="matrix-covered"),
    dict(case="global_barotropic_wind",
         aliases=(), tests="global 3-belt wind-stress barotropic circulation",
         grids=("latlon_cgrid", "mpas"), oracle=None, recipe="default_wright_v1",
         status=TODO, note="matrix-covered"),
    dict(case="global_overturning",
         aliases=(), tests="global baroclinic overturning (Wolfe & Cessi 2010 idealized THC)",
         grids=("latlon_cgrid", "mpas"), oracle="Wolfe & Cessi 2010 (idealized)",
         recipe="legoesm_linear_v1", status=TODO, note="idealized THC benchmark"),
    dict(case="dino",
         aliases=(), tests="Double-gyre Idealized North Ocean (Kamm et al. 2025); diabatic basin",
         grids=("latlon_cgrid", "mpas_regional"), oracle="Kamm et al. 2025 / NEMO-DINO",
         recipe="nemo_dino_v1", status=TODO, note="DINO reference"),
    dict(case="neverworld2_lite",
         aliases=(), tests="idealized global basin + ACC band (NeverWorld2-lite)",
         grids=("latlon_cgrid",), oracle="NeverWorld2 (idealized)", recipe="legoesm_linear_v1",
         status=TODO, note="idealized reference"),
    dict(case="isomip_plus",
         aliases=(), tests="ISOMIP+ ice-shelf cavity benchmark (Asay-Davis 2016)",
         grids=("latlon_regional",), oracle="ISOMIP+ intercomparison", recipe="legoesm_linear_v1",
         status=TODO, note="ice-shelf-cavity benchmark"),
)
