# Ocean experiments — DINO replication

Quick orientation for running the **DINO** (Diabatic Neverworld Ocean,
Kamm, Deshayes & Madec 2025, GMD) configuration in legoESM.

## In one minute

```bash
pip install -e ".[dev]"

# Lat-lon Mercator (recommended starting point), 1° R1, 10 simulated days:
JAX_ENABLE_X64=1 python scripts/run_dino.py --days 10

# MPAS regional Voronoi at matched ~equivalent cell area:
JAX_ENABLE_X64=1 python scripts/run_dino.py --grid mpas --days 10

# Visualize what came out:
JAX_ENABLE_X64=1 python scripts/plot_dino.py results/dino
```

Outputs land in `results/dino/snapshots/snapshot_*.npz` plus
`run_metadata.json` with the full DINOConfig. Both directories are
gitignored.

`--help` documents every flag. `--days` is capped at 365 by local-machine
policy; multi-decade spin-ups are GPU-machine territory (the output
directory is portable — copy it across).

## What the experiment does

DINO is an idealized pole-to-pole Atlantic-sector ocean basin (50° wide,
±70° lat) with a re-entrant channel between 45° and 65°S, designed by
Kamm et al. as a benchmark for testing eddy parameterizations. legoESM
replicates the **R1 (1°) eddy-parameterizing reference run** with full
surface forcing (wind, T/S restoring with Q_sr split, Jerlov SW
penetration) and physics (KPP boundary mixing, GM/Redi with Visbeck-1997
adaptive coefficient, enhanced-diffusion convection).

Both the **lat-lon Mercator path** (`create_mercator_grid` from PR #262)
and the **MPAS regional Voronoi path** (`create_regional_voronoi_mesh`
with `periodic_x=True` + seam-wall land mask) are end-to-end functional.

## Where things live

| What | Where |
|---|---|
| Configuration & physics | `src/legoesm/ocean/experiments/dino.py` |
| Production script | `scripts/run_dino.py` |
| Visualization script | `scripts/plot_dino.py` |
| Unit tests (188 across 11 files) | `tests/ocean/unit/test_dino_*.py`, `test_levy_stretched_z_star.py`, `test_partial_periodic_seam_wall.py`, `test_ke_gradient_hollingsworth.py` |
| Full plan, decisions log, lessons | `docs/ocean_experiments/dino_replication_plan.md` |
| Documentation plots (committed) | `docs/ocean_experiments/dino_plots/` |
| Run output from local simulations | `results/dino/` (gitignored) |

Documentation plots in `dino_plots/` show what DINO looks like:
`bathymetry.png` (basin shape + Drake sill ring), `initial_conditions.png`
(T(lat,z) and S(lat,z)), `surface_forcing.png` (wind, Q_sr, T*, S*
profiles vs latitude), `mpas_mesh.png` (regional Voronoi mesh + seam
wall + bathymetry). Useful first-look before running anything.

The DINO experiment is registered in
`legoesm.ocean.experiments.AVAILABLE_EXPERIMENTS["dino"]` with
metadata, scientific purpose, paper reference, and the standard
`create_initial_conditions(grid_type, ...)` /
`create_forcings(grid_type, ...)` / `validate(...)` interface.

## Reading order for a new agent

1. **Run it once** (the one-minute commands above) and inspect the
   plots — this gives you the simulated state and a feel for what
   the experiment looks like.
2. **Skim** `dino_replication_plan.md` "Implementation Status" table
   at the top to see what's done; "Decisions Log" for paper-vs-our
   choices; "Investigation Findings" appendix for the stability
   story.
3. **Read** `src/legoesm/ocean/experiments/dino.py` from the
   `DINOConfig` dataclass downward — every function has a docstring
   linking back to the paper equation it implements.
4. **Use** the unit tests as living documentation when you need to
   understand a specific piece (e.g., `test_dino_bathymetry.py`
   shows what shape of input each helper expects and what it
   returns).

## What's NOT replicated (out of scope)

- Multi-decade spin-up (paper does 3000 yr R1 + 400 yr production)
  — local cap is 1 yr; production runs are GPU territory
- R4 (1/4°) and R16 (1/16°) eddy-permitting / eddy-resolving variants
- Seasonal-cycle forcing (annual mean only)
- Coarse-graining / subgrid-flux diagnostics for ML eddy training
- TKE vertical mixing closure (we use KPP)
- Roquet simplified EOS (we use Wright)
- Tréguier (1997) GM coefficient (we use Visbeck 1997)

See the Decisions Log in the plan for the rationale on each.

## Known stability story

Two stability fixes are baked into the DINO defaults; both were
discovered during 30-day testing:

1. **Hollingsworth correction** (legoESM #263 / PR #264): NEMO's
   `nn_dynkeg=1` default that legoESM was missing. DINO's default
   `ke_gradient_scheme="hollingsworth"` enables it.
2. **A_h floor + equatorial boost**: MOM6 OM4 conventions
   (`A_h_floor=1000`, `A_h_eq_boost=3.0`) that compensate for NEMO
   stability features legoESM doesn't yet match (MSC, Robert-Asselin
   filter). Already enabled by DINO defaults.

Without either, the lat-lon path blew up at day 8-20. With both,
the 30-day unforced + physics run is stable end-to-end. See the
plan appendix "Why these stability fixes are not in the DINO paper
namelist" for the algorithmic-vs-pragmatic discussion.

## Quick troubleshooting

- **`FutureWarning: scatter inputs have incompatible types`** — cosmetic,
  comes from internal JAX precision-promotion warning when running with
  x32. Always run with `JAX_ENABLE_X64=1`.
- **`ValueError: Some generators fall outside the periodic zonal extent`** —
  mesh generator quantization gap; some MPAS resolutions don't seed
  cleanly. Default 97 km works; if you change `--mpas-resolution-km`
  and hit this, try ±5 km off your target.
- **Run blows up after some days** — first sanity check that
  `ke_gradient_scheme="hollingsworth"` and the `A_h_*` knobs are still
  set in DINOConfig. If you've overridden them or built the config
  manually, re-enable.

## Reference

Kamm, D., Deshayes, J., & Madec, G. (2025). DINO: a diabatic model of
pole-to-pole ocean dynamics to assess subgrid parameterizations across
horizontal scales. *Geosci. Model Dev.*, 18, 8091-8107.
[doi:10.5194/gmd-18-8091-2025](https://doi.org/10.5194/gmd-18-8091-2025).
NEMO source code: [vopikamm/DINO@v0.2.0](https://github.com/vopikamm/DINO/tree/v0.2.0)
([Zenodo deposit](https://doi.org/10.5281/zenodo.15016824)).
