"""Unified ocean case board — the single inventory of every ocean experiment/case,
what it tests, whether a true oracle exists in another model, and — per run — whether
legoESM reproduces it (stable / matches / blocked).

Data model (tidy / long form): the unit is a RUN = (case, grid, recipe) → status.
Grid and recipe are intricately linked (together = the numerical setup, the repo's
"recipe + setup(grid)" decomposition), so both live in each ``result``; the CASE
carries only physics metadata (what it tests, oracle). One experiment run on several
grids and/or recipes = several results, NOT several cases. Views are derived: a flat
table (one row per run) and a pivot matrix (case×grid rows × recipe columns) both
render from this one source
(``scripts/validate/ocean_fidelity/build_recipe_case_board.py``).

RATCHET (``tests/ocean/fidelity/test_recipe_case_board.py``): every registered
experiment (``AVAILABLE_EXPERIMENTS``) AND every oracle comparison driver
(``compare_*.py``) MUST appear here, so a new case can't be added without
classifying it. Holes (``todo``) are the roadmap; ``blocked`` cells cite the issue.

NO distinction between idealized and oracle cases — one board. The ``tests`` column
is a short physics summary so overlapping/duplicate cases are visible (the board
doubles as a retire-the-repeats inventory).

Honesty rule: a result is ``verified``/``partial``/``blocked`` ONLY with a concrete
measured basis (PR/issue/metric); everything not yet assessed is ``todo``.
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


def _r(grid: str, recipe: str, status: str, note: str) -> dict:
    """One run result: a (grid, recipe) pair and its status for a case."""
    return {"grid": grid, "recipe": recipe, "status": status, "note": note}


# --- The board ---------------------------------------------------------------
# Each case: case, aliases (registry/driver names mapping to this row), tests
# (1-line physics summary, for spotting duplicates), oracle (external truth model
# or None=idealized), results (list of runs {grid, recipe, status, note} — grid and
# recipe are linked: together they are the numerical setup of one run).
# ``aliases`` lets one physical case absorb its several registry names.
# Seed = the run(s) actually assessed; the primary grid only for not-yet-run cases.
# Per-grid runs become their own result as they are assessed.
CASES: tuple[dict, ...] = (
    # ---- cases with an external oracle (verified/blocked/partial known this session) ----
    dict(case="baroclinic_adjustment",
         aliases=("baroclinic", "oceananigans_baroclinic_adjustment", "oceananigans_spherical_baroclinic"),
         tests="stratified baroclinic instability; eddy growth + inverse cascade (unforced)",
         oracle="Oceananigans WENOVectorInvariant(9)+ImplicitFreeSurface",
         results=[_r("latlon_cgrid (β-plane)", "oceananigans + power_law stack", VERIFIED,
                     "EKE 0.86×, cascade k_e 5.11 vs 5.19, stable 40 d, no backstop — PR #672")]),
    dict(case="silvestri_jet_forced",
         aliases=("silvestri_baroclinic_jet", "silvestri", "oceananigans_silvestri_s5"),
         tests="forced (τ=50 d restoring) eddy-resolving baroclinic jet; Silvestri 2024 §5",
         oracle="Oceananigans Silvestri 2024 §5",
         results=[_r("latlon_cgrid (spherical channel)", "oceananigans + power_law stack", BLOCKED,
                     "over-energizes ~7× under sustained restoring (eddy-mean equilibration) — issue #673")]),
    dict(case="barotropic_gyre",
         aliases=("oceananigans_barotropic_gyre",),
         tests="wind-driven barotropic gyre spin-up (steady, laminar window)",
         oracle="Oceananigans ImplicitFreeSurface gyre",
         results=[_r("latlon_cgrid", "oceananigans (implicit_cn)", VERIFIED,
                     "surface-u pattern corr 0.93 days 1–20 (kinematic wind-stress unit fix)")]),
    dict(case="bickley_jet",
         aliases=("oceananigans_bickley_jet",),
         tests="single-layer Bickley-jet barotropic instability; vortex roll-up (eddy dissipation)",
         oracle="Oceananigans WENOVectorInvariant",
         results=[_r("latlon_cgrid (unit sphere)", "oceananigans (implicit_cn)", PARTIAL,
                     "tracks oracle to t6 (corr 0.97) then under-dissipates; enstrophy ~1.2–1.9× late")]),
    dict(case="internal_tide",
         aliases=("oceananigans_internal_tide", "oceananigans_internal_tide_w"),
         tests="tidal flow over topography; internal-tide generation (Flat-y, free surface)",
         oracle="Oceananigans internal_tide",
         results=[_r("latlon_cgrid (meridionally_flat)", "oceananigans (meridionally_flat + upwind3)", VERIFIED,
                     "corr ≥0.6 through ~2 days; dispersion cleared by upwind3 — #576")]),
    dict(case="geostrophic_adjustment",
         aliases=("oceananigans_geostrophic_adjustment",),
         tests="geostrophic adjustment from a temperature front; free-surface coupling",
         oracle="Oceananigans",
         results=[_r("latlon_cgrid", "oceananigans", TODO,
                     "comparison driver exists; not re-assessed post-#501 (also intended: cubed_sphere, mpas)")]),
    dict(case="gridmode_decay",
         aliases=("oceananigans_gridmode_decay",),
         tests="2Δx grid-mode decay rate (numerical dissipation probe)",
         oracle="Oceananigans",
         results=[_r("latlon_cgrid", "oceananigans", TODO, "diagnostic comparison; not re-assessed")]),
    dict(case="acc_channel",
         aliases=("acc", "veros_acc", "veros_acc_basic", "tendencies_acc"),
         tests="ACC-like re-entrant channel with ridge; eddy saturation + transport",
         oracle="Veros ACC",
         results=[_r("latlon_cgrid", "veros_faithful_v1", TODO,
                     "extensive Veros ACC work exists — re-confirm via scorecard (also intended: mpas)")]),
    dict(case="advection_gyre",
         aliases=("mitgcm_advection_gyre",),
         tests="passive-tracer advection in a wind-driven gyre",
         oracle="MITgcm tutorial_advection_in_gyre",
         results=[_r("latlon_cgrid", "mitgcm", TODO, "comparison driver exists; not re-assessed")]),
    # global_omip = the genuine MULTI-RUN exemplar: the same forced-global case run
    # under several (grid, recipe) pairs.
    dict(case="global_omip",
         aliases=("veros_global_1deg", "veros_global_4deg", "veros_global_flexible",
                  "legoesm_vs_veros", "omip_nemo_match_tripole", "omip_nemo_match_mpas"),
         tests="global forced (OMIP) ocean climate; SST/MOC vs reference GCM",
         oracle="Veros / NEMO (OMIP)",
         results=[
             _r("latlon_cgrid", "veros_faithful_v1", TODO, "Veros global transfer (1°/4°/flexible) — re-confirm via scorecard"),
             _r("latlon_cgrid", "nemo_v1", TODO, "NEMO-faithful dycore — re-confirm via scorecard"),
             _r("latlon_cgrid (tripole)", "omip_nemo_match_tripole_v1", TODO, "tripole eORCA025 NEMO-climate-match (SST RMSE ~1.15)"),
             _r("mpas (ico6)", "omip_nemo_match_mpas_v1", TODO, "MPAS ico6 NEMO-climate-match (SST RMSE ~0.84, best grid)"),
         ]),

    # ---- idealized cases (no external oracle; analytic/benchmark validation) ----
    dict(case="rest_state",
         aliases=("rest_state_stratified_with_land", "rest_state_uniform_with_land",
                  "rest_state_stratified_no_land", "rest_state_uniform_no_land"),
         tests="rest-state stability / PGF balance (η≈0); conservation control",
         oracle=None,
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "analytic truth (η≈0); matrix-covered (also: cubed_sphere, mpas) — confirm status")]),
    dict(case="barotropic_wave",
         aliases=(), tests="barotropic gravity-wave propagation (Gaussian SSH)", oracle=None,
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "analytic dispersion; matrix-covered (also: cubed_sphere, mpas, spectral)")]),
    dict(case="inertia_gravity_wave",
         aliases=(), tests="Poincaré (inertia-gravity) wave; Bishnu et al. 2024",
         oracle="analytic / Bishnu 2024",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "analytic benchmark; matrix-covered (also: cubed_sphere, mpas)")]),
    dict(case="lock_exchange",
         aliases=(), tests="density-driven gravity current; RPE mixing (Petersen 2015)",
         oracle="Petersen et al. 2015 (RPE benchmark)",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "RPE diagnostic benchmark; matrix-covered")]),
    dict(case="overflow",
         aliases=(), tests="dense water descending a bathymetric slope", oracle=None,
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "matrix-covered (also: cubed_sphere)")]),
    dict(case="phillips_two_layer",
         aliases=(), tests="Phillips two-layer baroclinic instability", oracle=None,
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "matrix-covered (also: cubed_sphere, mpas)")]),
    dict(case="eady_uniform",
         aliases=(), tests="classical Eady instability (uniform N², linear shear), re-entrant channel",
         oracle="analytic Eady growth rate",
         results=[_r("latlon_channel", "eady_weno5_v1", TODO,
                     "analytic growth-rate benchmark (also: mpas_channel)")]),
    dict(case="eady_instability",
         aliases=(), tests="Eady instability from a meridional temperature front", oracle=None,
         results=[_r("latlon_channel", "eady_weno5_v1", TODO,
                     "POSSIBLE DUPLICATE of eady_uniform — review for retire")]),
    dict(case="held_larichev",
         aliases=(), tests="Held–Larichev eddying channel; APE→KE cascade saturation (k⁻³)",
         oracle="Held & Larichev (spectral law)",
         results=[_r("latlon_channel", "default_wright_v1", TODO, "spectral-slope benchmark (also: mpas_channel)")]),
    dict(case="stommel_gyre_tracer",
         aliases=(), tests="passive tracer in a wind-driven Stommel gyre (Hecht 2000)", oracle=None,
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "matrix-covered (also: mpas)")]),
    dict(case="munk_gyre",
         aliases=(), tests="Munk gyre; lateral-viscosity western boundary current",
         oracle="analytic Munk BL width",
         results=[_r("latlon_regional", "default_wright_v1", TODO, "WBC benchmark (also: mpas_regional)")]),
    dict(case="baroclinic_gyre",
         aliases=("regional_gyre",), tests="wind-driven regional gyre with surface restoring + thermal wind",
         oracle=None,
         results=[_r("latlon_regional", "default_wright_v1", TODO, "matrix-covered (also: mpas_regional)")]),
    dict(case="global_barotropic_wind",
         aliases=(), tests="global 3-belt wind-stress barotropic circulation", oracle=None,
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "matrix-covered (also: mpas)")]),
    dict(case="global_overturning",
         aliases=(), tests="global baroclinic overturning (Wolfe & Cessi 2010 idealized THC)",
         oracle="Wolfe & Cessi 2010 (idealized)",
         results=[_r("latlon_cgrid", "legoesm_linear_v1", TODO, "idealized THC benchmark (also: mpas)")]),
    dict(case="dino",
         aliases=(), tests="Double-gyre Idealized North Ocean (Kamm et al. 2025); diabatic basin",
         oracle="Kamm et al. 2025 / NEMO-DINO",
         results=[_r("latlon_cgrid", "nemo_dino_v1", TODO, "DINO reference (also: mpas_regional)")]),
    dict(case="neverworld2_lite",
         aliases=(), tests="idealized global basin + ACC band (NeverWorld2-lite)",
         oracle="NeverWorld2 (idealized)",
         results=[_r("latlon_cgrid", "legoesm_linear_v1", TODO, "idealized reference")]),
    dict(case="isomip_plus",
         aliases=(), tests="ISOMIP+ ice-shelf cavity benchmark (Asay-Davis 2016)",
         oracle="ISOMIP+ intercomparison",
         results=[_r("latlon_regional", "legoesm_linear_v1", TODO, "ice-shelf-cavity benchmark")]),
)


def all_recipes() -> list[str]:
    """Every recipe appearing in any run (the pivot-matrix column axis)."""
    seen: dict[str, None] = {}
    for c in CASES:
        for res in c["results"]:
            seen.setdefault(res["recipe"], None)
    return list(seen)
