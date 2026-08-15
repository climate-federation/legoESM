# Global land-carbon initial condition — spin-up recipe and published product

The DifferLand 8-pool carbon cycle needs centuries of spin-up before soil carbon
stops drifting. This recipe replaces that spin-up with a **finidat**: a per-cell
equilibrium `CarbonState` (+ per-cell permafrost `phi`) that a coupled run
ingests at `t = 0`, so it starts *at* equilibrium instead of walking towards it.

**The product of this recipe is already built and published — download it before
you spend the 7 h rebuilding it.**

## 1. Use the published IC

Release [`land-carbon-ic-v1`](https://github.com/climate-federation/legoESM/releases/tag/land-carbon-ic-v1)
(private to the org, 3.8 MB total):

```bash
mkdir -p data/land_carbon_ic
gh release download land-carbon-ic-v1 -R climate-federation/legoESM \
    -D data/land_carbon_ic

python scripts/run/run_coupled.py \
    --carbon-ic data/land_carbon_ic/global_carbon_ic.npz  ...
```

| asset | what |
|---|---|
| `global_carbon_ic.npz` | the finidat — per-cell 8-pool `CarbonState` [gC/m2] + `soil_frozen_fraction` phi + lat/lon/land_mask/pft_weights |
| `archetypes.npz` | the 187 spun-up (PFT x climate x soil) archetypes the map was assembled from, with per-archetype QC |
| `validation_report.md` / `.json` | the drift QC below |
| `carbon_ic_maps.png` | SOC / biomass / permafrost-phi maps + zonal SOC vs observations |
| `SHA256SUMS` | checksums of the two `.npz` |

`--carbon-ic` is a **coupled-path-only** feature: `run_amip` prescribes carbon
and discards evolved pools, so it is intentionally not wired there.

### Grid contract (strict, fail-loud)

96 x 144 (1.9 deg x 2.5 deg CLM5 `surfdata` native grid), 13824 cells, **4914
land**, row-major flatten, 10 soil layers, 3.0 m depth, dt = 3600 s.

`load_finidat_carbon_ic` enforces the run's land-column count *and* per-column
lat/lon element-for-element, and raises on any mismatch — the pools are per-area
stocks pinned to these cells, so a silent reshape would corrupt the IC. There is
no cross-grid regridding: a different lat-lon resolution, a cubed-sphere or a
Voronoi run must build its own IC with the recipe below.

> The `resolution_deg = 1.0` scalar inside the file is a **stale metadata label**
> — the grid is 1.9 x 2.5. Nothing reads it (the loader matches on ncol + coords),
> so it is inert, but do not cite it.

## 2. Rebuild it

Three SLURM jobs, all submitted **from the repo root of the checkout you want to
run**. `WORKDIR` defaults to `SLURM_SUBMIT_DIR`, so the chain works in any
worktree; `WORKDIR`, `PYJ`, `SURF`, `CLIM`, `OUTDIR` are environment overrides.

```bash
mkdir -p scripts/tmp                       # SLURM opens --output before the job starts
J1=$(sbatch --parsable scripts/cluster/land_carbon/build_land_forcing_climatology.sbatch)
J2=$(sbatch --parsable --dependency=afterok:$J1 scripts/cluster/land_carbon/build_global_carbon_ic.sbatch)
sbatch --dependency=afterok:$J2 scripts/cluster/land_carbon/validate_global_carbon_ic.sbatch
```

**Stage 1 — ERA5 land-forcing climatology** (~32 min, `short`, needs outbound
internet so compute node only). Streams ARCO-ERA5 via anonymous `gcsfs`, years
2015-2020 at stride 8, and writes monthly `tas`/`pr`/`rsds`/`netrad` to
`results/land_forcing/era5_land_forcing_clim_2deg.nc`. This is what makes the IC
science-grade: the alternative `--climate-from-latitude` is a zonal
*demonstration* that collapses desert and rainforest onto one latitude.

**Stage 2 — the finidat** (~7 h 16 min, CPU, x64). Reads CLM5 cover/soil on its
native grid plus the Stage-1 climatology, samples the (PFT x climate x soil)
space into archetypes, spins each to a verified semi-analytic soil-C equilibrium
(Xia 2012), and maps the cover-weighted mix onto the grid.

- **CPU on purpose.** GPU x64 is compile-bound (>50 min, no result); the CPU lane
  also avoids the cross-node SIGILL / silently-wrong-x64 risk that made us
  disable the compile cache there. Keep CPU for scientific trust.
- Cost is ~19 min of cold compile per `(is_woody, is_evergreen,
  is_cold_deciduous, soil_class)` group. The persistent XLA compile cache
  (`.xla_cache_carbon_ic`) and the deterministic equilibrium *result* cache
  (`.carbon_equilibrium_cache`) make a resume cheap — if the 12 h `short` limit
  bites, just resubmit.
- The opt-in arctic gate flags are hashed into the equilibrium cache key, so a
  flags-on build never reads a flags-off cached result.

**Stage 3 — drift QC** (~13 min). Re-integrates every archetype for 3 yr from the
mapped equilibrium *and* from a cold start and compares total-carbon drift. This
is archetype-level and therefore *indicative*: `map_to_grid` is linear but a
mixed cell re-integrates nonlinearly, so it is necessary, not per-cell
sufficient. Point `ARCH` at a downloaded `archetypes.npz` to re-verify the
published IC on your own tree.

## 3. What the published IC actually scores

- mapped-IC median total-C drift **0.025 %/yr** vs cold-start **5.341 %/yr**
  (pass threshold 5) → **PASS**; worst archetype |drift| 3.14e-3 /yr.
- area-weighted (cos-lat, land only) **SOC 3.47 kgC/m2**, live biomass
  **7.88 kgC/m2**. By band, SOC = 0.61 (66-90N) / 6.66 (50-66N) / 2.52 (23S-23N).

### Known limits

- **Global SOC 3.47 vs observed ~9.5 kgC/m2**, and the deficit is
  Arctic-localized: temperate and tropical biomes sit in published ranges, 66-90N
  is collapsed.
- **`needleleaf_deciduous_boreal` (larch) is 0/0 at all 12 archetypes.** Audited
  in `docs/land/arctic_carbon_residual_audit.md`: a *healthy* deciduous PFT placed
  on larch's own climates also dies there, so those cells are genuinely
  non-viable under the model's cold-climate leaf-out — an input/structural limit,
  not a PFT parameter bug. The opt-in arctic gates were tested against it and do
  not fix it.
- Built **flags-off**. The arctic productivity mechanisms (NSC-gated `R_maint`,
  cold-deciduous dormancy, leaf bootstrap, leaf-C resorption) are all opt-in and
  default-off, so this IC is the equilibrium of the *production* model. A gate-ON
  IC is only an equilibrium of the gate-ON model, and the gate configuration is
  not stored in the finidat — a run that seeds a gate-ON IC with gates off will
  drift straight back. Rebuild flags-off unless you also thread the flags.

## 4. Provenance of the published build

| | |
|---|---|
| build job | SLURM **8973690**, 7 h 16 min CPU, exit 0, 2026-07-14 |
| validation job | SLURM **8974824** (`scripts/validate/global_carbon_ic_map.py`) |
| code at build | `d4c5f5115` (branch `land/carbon-global-init`, merged to main via #1015) |
| cover / soil | `surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc` (CLM5, native grid) |
| climate | ARCO-ERA5 monthly climatology 2015-2020, stride 8 |

Build command as executed:

```
python scripts/data/build_global_carbon_ic.py \
  --surfdata-preset clm5_surfdata --surf-path <surfdata_1.9x2.5_16pfts_CMIP6_simyr2000.nc> \
  --climatology <era5_land_forcing_clim_2deg.nc> \
  --output <results/global_carbon_ic> \
  --compilation-cache-dir <.xla_cache_carbon_ic> \
  --equilibrium-cache-dir <.carbon_equilibrium_cache>
```

## Related

- `docs/land/arctic_carbon_residual_audit.md` — why the high latitudes are low.
- `docs/land/carbon_equilibrium_audit.md` — the equilibrium method.
- `scripts/plot/plot_global_carbon_ic.py` — regenerates `carbon_ic_maps.png`.
