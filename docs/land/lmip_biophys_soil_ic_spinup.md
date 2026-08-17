# LMIP biophysics soil-state initial condition — spin-up recipe and published product

<!-- DRAFT: every [TBD-n] below must be filled from the actual Derecho run /
     Zenodo upload before this document is published.  The list of what each
     needs is at the bottom (“Before publishing”). -->

A cold-started global land run spends its first years drifting while soil
moisture and soil temperature equilibrate to the forcing. This recipe replaces
that spin-up with a published **land restart**: the end state of a 10-year
CRU-JRA spin-up (1975–1984) of the calibrated 2° biophysics LMIP — spun-up soil
moisture and temperature profiles (10 layers / 3.0 m), snow and canopy state for
every land column — that a run ingests at `t = 0` via `restart.from`.

**The product is already built and published — download it before you spend an
A100 job (≤ 4 h walltime) rebuilding it.**

## 1. Use the published IC

Zenodo record [TBD-2: DOI + link]:

```bash
mkdir -p data/lmip_soil_ic
curl -fL -o data/lmip_soil_ic/[TBD-3: filename] \
    "https://zenodo.org/records/[TBD-2]/files/[TBD-3]"
# verify
echo "[TBD-4: sha256]  data/lmip_soil_ic/[TBD-3]" | shasum -a 256 -c -
```

(Once the IC is staged by `scripts/data/download_lmip_data.sh`, use that
instead — it is idempotent and checks the file magic.)

> ⚠️ **The ice-sheet snowpack in this IC is not physical.** The single-bulk
> snow scheme has no glacier sink (no ablation, no calving, no ice conversion),
> so Greenland and Antarctica accumulate every year's snowfall forever —
> ~3100 Gt/yr in the spin-up, reaching 15.4 m water-equivalent at the SE
> Greenland margin by year 10. The IC carries that inflated snowpack, and any
> continuation inherits it and keeps growing it (earlier 30-yr runs reached
> ~45 m w.e.). Warm-starting from this IC is fine for non-glacier land; do not
> use the ice-sheet cells' snow, albedo, or water storage for science, and do
> not expect the global snow-mass budget to be meaningful until a glacier sink
> exists.

**Consume it** — any global biophysics LMIP run on the same grid:

```bash
# direct driver call
python scripts/run/run_lmip_biophys.py --config <your config.yaml> \
    --output-dir <dir> --restart-from data/lmip_soil_ic/[TBD-3]

# experiment workflow (production block warm-started at 1985)
python scripts/run/init_experiment.py biophysics/lmip_canopy_10yr \
    --name prod --output-dir $ROOT/prod --machine derecho_gpu \
    -o forcing.year_start=1985 -o forcing.year_end=1994 \
    -o restart.from=data/lmip_soil_ic/[TBD-3]
```

The restart timestamp is the start of model year 1985 (after forcing years
1975–1984, noleap), so 1985 is the natural first production year. Starting a
different year re-uses the same equilibrated state under shifted forcing — fine
for climatological work, but say so.

### Which drivers can consume it

| driver | seeded? | how |
|---|---|---|
| `scripts/run/run_lmip_biophys.py` (global biophysics LMIP) | yes | `--restart-from`, or `restart.from` in the config / `-o` override |
| `scripts/run/run_lmip.py` (single-point LMIP) | no | different state schema; use `config/lmip/lmip_carbon_ic.yaml` for its carbon seed |
| `scripts/run/run_amip.py` / coupled | not yet | the coupled multilayer land initializes its own state; no loader wired to this file |

### What is (and is not) in the file

A land restart round-trips the **prognostic** fields — soil temperature and
moisture profiles, snow (SWE, age), surface/canopy stores, and the freeze/thaw
state — and the driver grafts them onto a canonical cold-start template for the
optional structural fields. It contains **no carbon pools** (the biophysics
driver runs carbon-off; for carbon see `docs/land/global_carbon_ic_spinup.md`)
and **no atmosphere**: forcing comes from CRU-JRA at run time.

### Grid contract (fail-loud, with one gap)

~2° lat-lon (90 × 180), 7955 land columns (tape-derived; confirm against the
restart — TBD-5), 10 soil layers / 3.0 m
(growth factor 2.0 — a 2.93 mm top layer), dt = 3600 s, noleap calendar,
surfdata `c260716`.

`load_land_restart` **raises** on restart-version, `land_mode`, land-column-
count, or soil-layer-count mismatch, so the common mistakes (different
resolution, 8-layer soil) fail loudly. It does **not** compare per-column
coordinates: a different grid that happened to produce the same land-column
count would load silently. Use the IC only on this exact grid + surfdata; any
other grid must rebuild with the recipe below.

## 2. Rebuild it

The spin-up config is committed as `config/lmip/lmip_biophys_soil_ic.yaml` —
the rebuild is one run of it. Cost as-built: one A100 within a 04:00:00
walltime request (Derecho, `derecho_gpu` machine profile).

```bash
# 1. stage forcing (10 years, ~10 GB/yr off-site) + surfdata
for y in $(seq 1975 1984); do ./scripts/data/download_lmip_data.sh --year $y; done

# 2. run (experiment workflow; or call run_lmip_biophys.py --config directly)
python scripts/run/init_experiment.py biophysics/lmip_calibrated_spinup \
    --name soil_ic_rebuild --output-dir $ROOT/soil_ic_rebuild --machine derecho_gpu \
    -o forcing.year_start=1975 -o forcing.year_end=1984
cd $ROOT/soil_ic_rebuild && qsub -A <ACCT> run.sh

# 3. the IC is the end-of-run restart:
ls $ROOT/soil_ic_rebuild/restart_1985_d000h00.npz
```

Notes that keep a rebuild honest:

- **Forcing variant.** The published build used the NCAR glade
  `filled_antarct_and_grnlnd` CRU-JRA archive. The public CESM mirror that
  `download_lmip_data.sh` falls back to off-site serves the **unfilled**
  variant under a different file prefix — an off-NCAR rebuild therefore differs
  over Antarctica and Greenland. On glade, point `--crujra-src` at the filled
  archive (prefix `clmforc.CRUJRAv2.5_filled_antarct_and_grnlnd_0.5x0.5`) to
  reproduce the published forcing.
- **Template availability.** `biophysics/lmip_calibrated_spinup` ships on the
  calibrated-config branch ([TBD-6: PR #]); until that merges, the committed
  YAML above is the self-contained equivalent (same resolved physics).
- **Tapes.** The as-built run predates budget-closure taping in the templates;
  a rebuild from the current template tapes more diagnostics. Tapes do not
  enter the model state, so the restart is unaffected.
- **Not bit-identical.** GPU float math is free to differ across driver/XLA
  versions; reproduce the end-state scorecard (§3), not the bytes.
  Bit-identity comes only from downloading the published file.

## 3. What the published IC actually scores

All numbers from the as-run tapes: **final forcing year (1984)** monthly means,
area-weighted by cos(lat) × surfdata `land_fraction`, land only, float64. The
weighting was validated against a known answer first: total land area
reproduces 146.94 × 10¹² m² exactly as the template header records. Two
populations are reported side by side per the repo's QC rule — **all-land**
(nothing excluded) and **spike-excluded** (the runoff-defect cells below
removed, 77.3 % of land area) — because the defect cells move the means.

| quantity (1984 mean) | all land | spike-excluded |
|---|---|---|
| ET [mm/day] | 1.35 | 1.02 |
| precip [mm/day] | 2.06 | 1.91 |
| runoff [mm/day] | unphysical (see below) | 1.26 |
| Bowen ratio (SH/LH) | 0.74 | 0.94 |
| Rnet / SH / LH [W/m²] | 67.4 / 28.7 / 39.0 | 54.8 / 27.5 / 29.4 |
| GPP [gC/m²/day] | 3.58 | 2.78 |

- **Stability (the flagged 2.93 mm-top-layer risk): good.** The NaN-revert
  guard fired 938 cell-steps total in 87 600 steps; 464 of 7955 land cells
  (5.8 %, 6.3 % of land area) ever fired; 73 cells still fired in the final
  year. `lmip_biophys.reverts.nc` in the run dir carries the per-cell counts.
- **Near-equilibrated (year-10 minus year-9, identical protocol both sides):**
  ΔT_sfc −0.06 K, ΔET −0.003 mm/day, Δθ_top −0.0006 m³/m³ (all-land). These
  pairs differ in forcing year as well as state, so they bound drift rather
  than isolate it.
- **Energy residual** Rnet − SH − LH = −0.3 W/m² (all-land). This residual is
  G + snowmelt energy, which the as-run tapes did not record, so exact closure
  is not checkable from this run's outputs (later template versions tape it).
- **Snow**: seasonal snow behaves; the ice sheets accumulate ~3100 Gt/yr with
  no glacier sink (known limit below), reaching 15.4 m w.e. at the SE Greenland
  margin by year 10.

### ⚠️ Known open issue: episodic runoff blow-ups

A known, open defect of this configuration; the numbers below are what this
run's tapes measure it at.

**CONFIRMED (from the tapes):** ~950 land cells every year (12 % of land
cells; 23 % of land area in the final year at a >20 mm/day annual-mean
threshold, mostly tropical but with mid- and high-latitude members) produce
episodic, single-month runoff spikes up to 10⁸–10⁹ mm/day — finite, so the
NaN-revert guard never fires (only 24 of 1079 flagged cells ever reverted).
The population is stationary from year 1 (not a developing instability), and
spike months sit between months of normal ~0.1 mm/day runoff. The global
water budget consequently does not close on the all-land population, and even
the spike-excluded subset carries a −0.36 mm/day P−ET−R residual.

**What this means for the IC:** the *stored soil state* in affected cells
looks bounded and plausible (θ_top within physical range, normal ET) — the
blow-up is in the runoff flux, not visibly in the state the IC carries.

**PLAUSIBLE, not confirmed:** a near-zero denominator in the
saturation-excess / drainage path of the thin (2.93 mm) top layer. Whether
the spiked mass is really removed from soil storage or fabricated by a clamp
has not been instrumented; treat runoff from this configuration as unusable
until the mechanism is found, and do not use this run for any water-budget
science.

### Known limits

- **Calibration provenance.** The albedo and root-zone calibrations were tuned
  under SimpleSEB (the AMIP land scheme), then adopted by this two-leaf-canopy
  configuration; the pairing has not been independently re-validated at LMIP
  scale.
- **GPP is high** (#730): the interactive canopy conductance is uncalibrated on
  the deep, moist Richards soil, and CO2 is held at a constant 412 ppmv. The
  soil/energy state is the product here; treat taped GPP as a diagnostic.
- **The 2.93 mm top layer** changes the meaning of top-layer diagnostics
  (`theta_soil_top`, `T_soil_top`) relative to any 8-layer baseline — they are
  not comparable across soil grids.
- **Ice sheets**: the single-bulk snow scheme has no glacier sink, so
  Greenland/Antarctica accumulate snow without bound — ~3100 Gt/yr in this
  run, 15.4 m w.e. at the SE Greenland margin by year 10 (earlier 30-yr runs
  reached ~45 m). The IC carries that inflated ice-sheet snowpack; a
  continuation inherits and extends it.
- **Land fraction**: totals computed with the model's own `land_fraction` from
  surfdata `c260716` are within ~1.4 % of the CLM reference. Runs made with the
  superseded `c250617` surfdata over-counted land area by ~26 % — do not mix.

## 4. Provenance of the published build

| | |
|---|---|
| experiment | `spinup_1975_calib`, created 2026-07-28T14:14:45Z, run dir `/glade/derecho/scratch/linnia/lmip/spinup_1975_calib` |
| spin-up job | PBS 6933356 (walltime request 04:00:00), outputs written 2026-07-30 |
| machine | Derecho (`derecho_gpu` profile): 1 × A100, 32 cpus / 64 GB |
| precision | `JAX_ENABLE_X64=1`, `JAX_PLATFORM_NAME=gpu` (explicit — no silent CPU fallback) |
| code at build | `14461553438f18d336985870770c38a4555d3552` (branch `lmip-calibrated-config`) |
| stack | Python 3.12.13, JAX 0.10.2; config hash `sha256:2894ff1fc7df91cb` (from `experiment.tag`) |
| template | `biophysics/lmip_calibrated_spinup` v1.0.0, as at that commit (2026-07-27) |
| config as run | `config/lmip/lmip_biophys_soil_ic.yaml` (paths repo-relativized; as-run they pointed into `/glade/work/linnia/legoESM/data/`) |
| surfdata | `legoesm_surfdata_c260716.nc` (Zenodo record 21401647) |
| forcing | CRU-JRA v2.5 `filled_antarct_and_grnlnd`, glade archive, years 1975–1984 |
| product | `[TBD-3: filename]`, [TBD-12: size], sha256 `[TBD-4]` |

## Before publishing (fill-in checklist)

| tag | needed | where to get it |
|---|---|---|
| TBD-2 | Zenodo record + DOI | after upload |
| TBD-3 | published filename (recommend `legoesm_lmip_soil_ic_1985_<date>.npz`) | your choice at upload |
| TBD-4 | sha256 of the file | `sha256sum restart_1985_d000h00.npz` on Derecho |
| TBD-5 | land-column count (ncol) | the tapes say 7955 cells have land under `c260716`; confirm against the restart: `python -c "import numpy; print(numpy.load('restart_1985_d000h00.npz')['T_soil'].shape)"` |
| TBD-6 | calibrated-config PR number | after opening it |
| TBD-12 | file size | `ls -lh restart_1985_d000h00.npz` |

## Related

- `docs/land/lmip_biophys_runbook.md` — the driver + template workflow.
- `docs/land/global_carbon_ic_spinup.md` — the carbon-pool IC (separate file,
  separate drivers); the publication pattern this document follows.
- `config/lmip/lmip_biophys_soil_ic.yaml` — the committed spin-up config.
