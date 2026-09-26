"""Unified ocean case board — the single inventory of every ocean experiment/case,
what it tests, what external truth (if any) it can be validated against, and — per
run — whether legoESM reproduces it (stable / matches / blocked).

Data model (tidy / long form): the unit is a RUN = (case, grid, recipe) → status.
Grid and recipe are intricately linked (together = the numerical setup, the repo's
"recipe + setup(grid)" decomposition), so both live in each ``result``; the CASE
carries physics metadata (what it tests) + the external REFERENCE (kind + name).
One experiment run on several grids and/or recipes = several results, NOT several
cases. Views are derived: a flat table (one row per run) and a pivot matrix
(case×grid rows × recipe columns), both rendered from this one source
(``scripts/validate/ocean_fidelity/build_recipe_case_board.py``).

REFERENCE KIND (truth strength — distinct from "did legoESM pass"):
  oracle    — a model we can RUN to generate reference data (tendency/field match):
              Oceananigans, Veros, MITgcm, NEMO. The strongest.
  analytic  — a closed-form / exact solution we check against (Eady growth, Munk
              width, IGW dispersion, rest-state η≈0).
  published — a paper / intercomparison (figures/tables) with NO runnable data —
              weaker than an oracle (eyeball / scaling-law only).
  none      — idealized; no external truth (only norms / conservation).

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

# --- Status vocabulary (did legoESM reproduce the run?) ----------------------
VERIFIED = "verified"   # reproduces its reference: stable + matches, tested
WORKS = "works"         # runs stably, but no external reference to match against
PARTIAL = "partial"     # runs / early-matches but does not meet the fidelity bar
BLOCKED = "blocked"     # applicable but fundamentally not achievable yet — see ref (issue)
TODO = "todo"           # not yet assessed — a hole to fill
NA = "n/a"              # not applicable

VALID_STATUSES = frozenset({VERIFIED, WORKS, PARTIAL, BLOCKED, TODO, NA})

# --- Reference kind (what external truth exists — truth strength) ------------
ORACLE = "oracle"        # runnable model + data: Oceananigans, Veros, MITgcm, NEMO
ANALYTIC = "analytic"    # closed-form / exact solution we check against
PUBLISHED = "published"  # paper / intercomparison (figures/tables) — NO runnable data
NO_REF = "none"          # idealized — no external truth (norms / conservation only)

VALID_REF_KINDS = frozenset({ORACLE, ANALYTIC, PUBLISHED, NO_REF})


def _r(grid: str, recipe: str, status: str, note: str) -> dict:
    """One run result: a (grid, recipe) pair and its status for a case."""
    return {"grid": grid, "recipe": recipe, "status": status, "note": note}


# --- The board ---------------------------------------------------------------
# Each case: case, aliases, tests (1-line physics summary, for spotting duplicates),
# ref_kind (ORACLE/ANALYTIC/PUBLISHED/NO_REF), ref (the truth's name), results (runs
# {grid, recipe, status, note} — grid+recipe linked = one numerical setup).
CASES: tuple[dict, ...] = (
    # ---- ORACLE cases (runnable model + data) ----
    dict(case="baroclinic_adjustment",
         aliases=("baroclinic", "oceananigans_baroclinic_adjustment", "oceananigans_spherical_baroclinic"),
         tests="stratified baroclinic instability; eddy growth + inverse cascade (unforced)",
         ref_kind=ORACLE, ref="Oceananigans WENOVectorInvariant(9)+ImplicitFreeSurface",
         results=[_r("latlon_cgrid (β-plane)", "oceananigans_v1 (+power_law)", VERIFIED,
                     "EKE 0.86×, cascade k_e 5.11 vs 5.19, stable 40 d, no backstop — PR #672")]),
    dict(case="silvestri_jet_forced",
         aliases=("silvestri_baroclinic_jet", "silvestri", "oceananigans_silvestri_s5"),
         tests="forced (τ=50 d restoring) eddy-resolving baroclinic jet; Silvestri 2024 §5",
         ref_kind=ORACLE, ref="Oceananigans Silvestri 2024 §5",
         results=[_r("latlon_cgrid (spherical channel)", "oceananigans_v1 (+power_law)", BLOCKED,
                     "over-energizes ~7× under sustained restoring (eddy-mean equilibration) — issue #673")]),
    dict(case="barotropic_gyre",
         aliases=("oceananigans_barotropic_gyre",),
         tests="wind-driven barotropic gyre spin-up (steady, laminar window)",
         ref_kind=ORACLE, ref="Oceananigans ImplicitFreeSurface gyre",
         results=[_r("latlon_cgrid", "oceananigans_v1", VERIFIED,
                     "surface-u pattern corr 0.93 days 1–20 (kinematic wind-stress unit fix)")]),
    dict(case="bickley_jet",
         aliases=("oceananigans_bickley_jet",),
         tests="single-layer Bickley-jet barotropic instability; vortex roll-up (eddy dissipation)",
         ref_kind=ORACLE, ref="Oceananigans WENOVectorInvariant",
         results=[_r("latlon_cgrid (unit sphere)", "oceananigans_v1", PARTIAL,
                     "tracks oracle to t6 (corr 0.97) then under-dissipates; enstrophy ~1.2–1.9× late")]),
    dict(case="internal_tide",
         aliases=("oceananigans_internal_tide", "oceananigans_internal_tide_w"),
         tests="tidal flow over topography; internal-tide generation (Flat-y, free surface)",
         ref_kind=ORACLE, ref="Oceananigans internal_tide",
         results=[_r("latlon_cgrid (meridionally_flat)", "oceananigans_v1 (meridionally_flat+oceananigans_up3)", VERIFIED,
                     "corr ≥0.6 through ~2 days; dispersion cleared by the UP3 arm — #576")]),
    dict(case="geostrophic_adjustment",
         aliases=("oceananigans_geostrophic_adjustment",),
         tests="geostrophic adjustment from a temperature front; free-surface coupling",
         ref_kind=ORACLE, ref="Oceananigans",
         results=[_r("latlon_cgrid", "oceananigans_v1", TODO,
                     "comparison driver exists; not re-assessed post-#501 (also intended: cubed_sphere, mpas)")]),
    dict(case="gridmode_decay",
         aliases=("oceananigans_gridmode_decay",),
         tests="2Δx grid-mode decay rate (numerical dissipation probe)",
         ref_kind=ORACLE, ref="Oceananigans",
         results=[_r("latlon_cgrid", "oceananigans_v1", TODO, "diagnostic comparison; not re-assessed")]),
    dict(case="acc_channel",
         aliases=("acc", "veros_acc", "veros_acc_basic", "tendencies_acc"),
         tests="ACC-like re-entrant channel with ridge; eddy saturation + transport",
         ref_kind=ORACLE, ref="Veros ACC",
         results=[_r("latlon_cgrid", "veros_faithful_v1", TODO,
                     "extensive Veros ACC work exists — re-confirm via scorecard (also intended: mpas)")]),
    dict(case="advection_gyre",
         aliases=("mitgcm_advection_gyre",),
         tests="passive-tracer advection in a wind-driven gyre",
         ref_kind=ORACLE, ref="MITgcm tutorial_advection_in_gyre",
         results=[_r("latlon_cgrid", "mitgcm_v1", TODO, "comparison driver exists; not re-assessed")]),
    dict(case="global_omip",
         aliases=("veros_global_1deg", "veros_global_4deg", "veros_global_flexible",
                  "legoesm_vs_veros", "omip_nemo_match_tripole", "omip_nemo_match_mpas"),
         tests="global forced (OMIP) ocean climate; SST/MOC vs reference GCM",
         ref_kind=ORACLE, ref="Veros / NEMO (OMIP)",
         results=[
             _r("latlon_cgrid", "veros_faithful_v1", TODO, "Veros global transfer (1°/4°/flexible) — re-confirm via scorecard"),
             _r("latlon_cgrid", "legoesm_nemo_like_v1", TODO, "NEMO-style dycore (barotropic solver is legoESM's own generic arm, not NEMO's) — re-confirm via scorecard"),
             _r("latlon_cgrid (tripole)", "omip_nemo_match_tripole_v1", TODO, "tripole eORCA025 NEMO-climate-match (SST RMSE ~1.15)"),
             _r("mpas (ico6)", "omip_nemo_match_mpas_v1", TODO, "MPAS ico6 NEMO-climate-match (SST RMSE ~0.84, best grid)"),
         ]),

    # ---- ANALYTIC cases (closed-form / exact solution) ----
    dict(case="rest_state",
         aliases=("rest_state_stratified_with_land", "rest_state_uniform_with_land",
                  "rest_state_stratified_no_land", "rest_state_uniform_no_land"),
         tests="rest-state stability / PGF balance (η≈0); conservation control",
         ref_kind=ANALYTIC, ref="exact rest state (η≈0)",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "matrix-covered (also: cubed_sphere, mpas) — confirm status")]),
    dict(case="barotropic_wave",
         aliases=(), tests="barotropic gravity-wave propagation (Gaussian SSH)",
         ref_kind=ANALYTIC, ref="gravity-wave dispersion relation",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "matrix-covered (also: cubed_sphere, mpas, spectral)")]),
    dict(case="inertia_gravity_wave",
         aliases=(), tests="Poincaré (inertia-gravity) wave",
         ref_kind=ANALYTIC, ref="Poincaré dispersion (Bishnu 2024 cross-check)",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "analytic benchmark; matrix-covered (also: cubed_sphere, mpas)")]),
    dict(case="phillips_two_layer",
         aliases=(), tests="Phillips two-layer baroclinic instability",
         ref_kind=ANALYTIC, ref="Phillips two-layer instability criterion",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO,
                     "matrix-covered (also: cubed_sphere, mpas)")]),
    dict(case="eady_uniform",
         aliases=(), tests="classical Eady instability (uniform N², linear shear), re-entrant channel",
         ref_kind=ANALYTIC, ref="Eady growth rate",
         results=[_r("latlon_channel", "eady_weno5_v1", TODO,
                     "analytic growth-rate benchmark (also: mpas_channel)")]),
    dict(case="eady_instability",
         aliases=(), tests="Eady instability from a meridional temperature front",
         ref_kind=ANALYTIC, ref="Eady growth rate",
         results=[_r("latlon_channel", "eady_weno5_v1", TODO,
                     "POSSIBLE DUPLICATE of eady_uniform — review for retire")]),
    dict(case="munk_gyre",
         aliases=(), tests="Munk gyre; lateral-viscosity western boundary current",
         ref_kind=ANALYTIC, ref="Munk boundary-layer width",
         results=[_r("latlon_regional", "default_wright_v1", TODO, "WBC benchmark (also: mpas_regional)")]),

    # ---- PUBLISHED cases (paper / intercomparison, NO runnable data) ----
    dict(case="lock_exchange",
         aliases=(), tests="density-driven gravity current; RPE mixing",
         ref_kind=PUBLISHED, ref="Petersen et al. 2015 (RPE intercomparison)",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "RPE diagnostic benchmark; matrix-covered")]),
    dict(case="overflow",
         aliases=(), tests="dense water descending a bathymetric slope",
         ref_kind=PUBLISHED, ref="DOME / Petersen 2015 overflow benchmark",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "matrix-covered (also: cubed_sphere)")]),
    dict(case="held_larichev",
         aliases=(), tests="Held–Larichev eddying channel; APE→KE cascade saturation (k⁻³)",
         ref_kind=PUBLISHED, ref="Held & Larichev 1996 (spectral scaling law)",
         results=[_r("latlon_channel", "default_wright_v1", TODO, "spectral-slope benchmark (also: mpas_channel)")]),
    dict(case="stommel_gyre_tracer",
         aliases=(), tests="passive tracer in a wind-driven Stommel gyre",
         ref_kind=PUBLISHED, ref="Hecht et al. 2000",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "matrix-covered (also: mpas)")]),
    dict(case="global_overturning",
         aliases=(), tests="global baroclinic overturning (idealized THC)",
         ref_kind=PUBLISHED, ref="Wolfe & Cessi 2010 (idealized THC)",
         results=[_r("latlon_cgrid", "legoesm_linear_v1", TODO, "idealized THC benchmark (also: mpas)")]),
    dict(case="dino",
         aliases=(), tests="Double-gyre Idealized North Ocean; diabatic basin",
         ref_kind=PUBLISHED, ref="Kamm et al. 2025 / NEMO-DINO (no runnable data yet)",
         results=[_r("latlon_cgrid", "nemo_dino_v1", TODO, "DINO reference (also: mpas_regional)")]),
    dict(case="neverworld2_lite",
         aliases=(), tests="idealized global basin + ACC band (NeverWorld2-lite)",
         ref_kind=PUBLISHED, ref="NeverWorld2 intercomparison",
         results=[_r("latlon_cgrid", "legoesm_linear_v1", TODO, "idealized reference")]),
    dict(case="isomip_plus",
         aliases=(), tests="ISOMIP+ ice-shelf cavity benchmark",
         ref_kind=PUBLISHED, ref="ISOMIP+ (Asay-Davis 2016) intercomparison",
         results=[_r("latlon_regional", "legoesm_linear_v1", TODO, "ice-shelf-cavity benchmark")]),

    # ---- NO external reference (idealized; norms / conservation only) ----
    dict(case="baroclinic_gyre",
         aliases=("regional_gyre",), tests="wind-driven regional gyre with surface restoring + thermal wind",
         ref_kind=NO_REF, ref="",
         results=[_r("latlon_regional", "default_wright_v1", TODO, "matrix-covered (also: mpas_regional)")]),
    dict(case="global_barotropic_wind",
         aliases=(), tests="global 3-belt wind-stress barotropic circulation",
         ref_kind=NO_REF, ref="",
         results=[_r("latlon_cgrid", "default_wright_v1", TODO, "matrix-covered (also: mpas)")]),
)


def all_recipes() -> list[str]:
    """Every recipe appearing in any run (the pivot-matrix column axis)."""
    seen: dict[str, None] = {}
    for c in CASES:
        for res in c["results"]:
            seen.setdefault(res["recipe"], None)
    return list(seen)
