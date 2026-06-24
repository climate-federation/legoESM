# MPAS realistic-bathymetry validation scripts

P6 of `docs/ocean/experiments/realistic_geometry_mpas_plan.md`.
Standalone scripts that exercise the MPAS partial-cell stack
(ETOPO ingestion, Adcroft PGF, hybrid vertex thickness, donor-cell
continuity, min-rule edge thickness, bottom drag at
maxLevelEdgeBot) under increasingly realistic configurations.

| Script | What it tests | Approx. wall time (default) |
|---|---|---|
| `run_mpas_seamount_rest.py` | PGF over isolated topography (canonical "no spurious flow over a seamount" gate) — compares `pgf_scheme="centered"` vs `"adcroft"` | seconds (subdiv 2), minutes (subdiv 4) |
| `run_mpas_etopo_spinup.py` | Wind-driven GO spinup on ico4 with ETOPO bathymetry — the headline production-scale validation | hours (subdiv 4, 30 days), days+ (subdiv 7, 5 years) |

## Setup

ETOPO bathymetry must be supplied via `--etopo-path PATH/TO/etopo.nc`
to `run_mpas_etopo_spinup.py`. Without it the script falls back to
idealized bathymetry (lat-threshold land, flat ocean) and only
exercises the integration plumbing, not the realistic-bathy gates.

ETOPO data is local-only per `feedback_docs_references_gitignored.md`;
download a recent ETOPO release (ETOPO 2022 1-arcmin or 60-arcsec works)
and pass the path explicitly.

## Acceptance gates (production runs)

For the 5-yr ico4 ETOPO spinup
(`--subdivision 7 --years 5`):

1. Integration completes 5 sim-years without NaN.
2. `max|u|` stays bounded (< 2 m/s) once spun up.
3. Time-mean V_baro grid-noise σ < 3× the implicit-CN flat-bottom
   baseline (1.92e-2 m/s per `project_mpas_barotropic_noise.md`).
4. AMOC magnitude in observed range (15-25 Sv).
5. No spurious deep flow under western boundary topography
   (visual inspection of saved snapshots).

For the seamount rest-state:

1. Both `pgf_scheme="centered"` and `pgf_scheme="adcroft"` integrate
   for the requested duration without NaN.
2. AC residual `max|u|` stays bounded (< 1e-3 m/s at moderate
   resolution; lower at higher).
3. Spurious flow concentrated near the seamount edge, not basin-wide.

## Out of scope (deferred to follow-ups)

- **DOME overflow** (Legg et al. 2006) — partial-cell community
  benchmark for dense-water plume preservation. Channel + sloping
  bottom + dense-water inflow + tracer; setup is involved enough
  to deserve its own dedicated script (see plan doc).
- **COMODO seamount** (Auclair et al. 2018) — internal-tide
  generation diagnostic; same.
- **Ilıcak et al. 2012 triplet** — overflow / internal-wave /
  baroclinic-eddy joint test.
- **Regional spherical Voronoi mesh** with North Atlantic
  bathymetry — leverages the regional-mesh work in
  `project_regional_mpas.md`.

When these land, add them here as additional `run_mpas_*.py`
scripts following the same pattern.
