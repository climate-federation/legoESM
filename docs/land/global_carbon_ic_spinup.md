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

# coupled ESM — whole-grid seed
python scripts/run/run_coupled.py \
    --carbon-ic data/land_carbon_ic/global_carbon_ic.npz  ...

# single-point LMIP — nearest-land-cell seed, or just use the shipped config
python scripts/run/run_lmip.py --lat 40.0 --lon -105.0 \
    --carbon-scheme differland \
    --carbon-ic data/land_carbon_ic/global_carbon_ic.npz
python scripts/run/run_lmip.py --config config/lmip/lmip_carbon_ic.yaml
```

| asset | what |
|---|---|
| `global_carbon_ic.npz` | the finidat — per-cell 8-pool `CarbonState` [gC/m2] + `soil_frozen_fraction` phi + lat/lon/land_mask/pft_weights |
| `archetypes.npz` | the 187 spun-up (PFT x climate x soil) archetypes the map was assembled from, with per-archetype QC |
| `validation_report.md` / `.json` | the drift QC below |
| `carbon_ic_maps.png` | SOC / biomass / permafrost-phi maps + zonal SOC vs observations |
| `SHA256SUMS` | checksums of the two `.npz` |

### Which drivers can consume it

| driver | seeded? | how |
|---|---|---|
| `scripts/run/run_coupled.py` (coupled ESM) | yes | `--carbon-ic <finidat>` → `CoupledConfig.carbon_ic_path`; STRICT whole-grid match, needs multilayer land |
| `scripts/run/run_lmip.py` (single-point LMIP) | yes | `--carbon-ic <finidat>` with `--carbon-scheme differland`, or the shipped `config/lmip/lmip_carbon_ic.yaml`; seeds the one column from the **nearest land cell** |
| `scripts/run/run_amip.py` (AMIP) | no, by design | AMIP prescribes carbon and discards evolved pools |
| `scripts/run/run_lmip_biophys.py` (global LMIP biophysics) | not yet | that driver pins `carbon="none"` — energy/water/snow/soil only, with GPP taped as a diagnostic. There are no pools to seed until its carbon cycle lands |

For the single-point LMIP the whole-grid contract cannot be met (one column at an
arbitrary site), but the column can still be *initialised* from the nearest cell:
the pools are per-area densities [gC/m2], so `load_finidat_carbon_ic_at_point`
simply reads the column that the spin-up equilibrated at the nearest place on
Earth — no regridding, no area weighting. It mirrors `surface_params_at_point`,
which already picks the nearest surfdata gridcell for the same driver, so cover,
soil and carbon come from the same neighbourhood. Candidates are restricted to
land cells and ranked by great-circle angle (a lat/lon Euclidean metric mis-ranks
them at high latitude); exact ties fall to the lowest flat index, i.e. file
order; and a match farther than 3° is refused rather than silently seeding a
column from a different climate. `run_lmip`'s soil defaults (10 layers / 3.0 m)
already match the column the IC was spun up on.

> **It is a nearest-cell initialisation, not an equilibrium of the site.** The
> pools equilibrated under the finidat *cell's* grid-mean climate and its
> cover-weighted **PFT mixture**, whereas a point run picks one `veg_type` and
> one `soil_texture` at coordinates that need not resemble any of that. The
> driver prints the matched cell's dominant cover and its cover fraction, and
> warns when that cover or the soil column disagrees with the run. Expect the
> pools to drift toward the point configuration's own equilibrium — far less
> than from a cold start, but not to zero. Do not quote a seeded point run as
> "at equilibrium" without showing its drift.

The permafrost `phi` that scaled the seed is loaded with it and threaded into the
run's SOM protection, so seeded high-latitude carbon is held by the same
protection that produced it. Without that, the seeded Arctic SOC decays back
toward the unprotected equilibrium and the seed is cosmetic.

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
run**: the working directory defaults to `SLURM_SUBMIT_DIR`, so the chain works
in any worktree, and each job stamps `WORKDIR @ <sha> dirty=<n>` in its log so
the product is attributable to a commit rather than to a path. Every stage exits
with the status of the work it ran, so the `afterok` chain is a real gate.

Environment overrides (all namespaced — a bare `WORKDIR`/`OUT`/`ARCH` inherited
from your shell would otherwise outrank the submission directory, since `sbatch`
exports the caller's whole environment):

| variable | stage | meaning |
|---|---|---|
| `LEGOESM_WORKDIR` | all | the checkout to run (default: submit dir) |
| `LEGOESM_PYTHON` | all | interpreter |
| `CIC_CLIM` | 1 writes, 2 reads | the ERA5 climatology NetCDF |
| `CIC_SURF` | 2 | CLM5 surfdata NetCDF |
| `CIC_OUTDIR` | 2 writes, 3 reads | the build-product directory |
| `CIC_CCACHE` / `CIC_ECACHE` | 2, 3 | XLA compile cache / equilibrium result cache |
| `CIC_ARCH` | 3 | `archetypes.npz` to validate (defaults inside `CIC_OUTDIR`) |
| `CIC_VALDIR` | 3 | where the validation report is written |

Submit plainly — `sbatch --chdir=…` is **not** supported, because the
`--output=scripts/tmp/%x_%j.out` directive is resolved by SLURM against the job
working directory while the script separately `cd`s to `LEGOESM_WORKDIR`, and the
directive cannot follow it. These are also not array jobs: every task of an
`--array` would share one output directory and one cache.

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
