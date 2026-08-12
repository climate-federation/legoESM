# Runbook — the 5-year AMIP production chain

Written 2026-08-11 for a handover. Everything here is what someone who did NOT
launch the run needs in order to keep it alive, read it, or decide to stop it.

## What is running

| | |
|---|---|
| SLURM job | `26881914`, name `amip_mpas_gpu_chain.sbatch` |
| launcher | `scripts/cluster/levante/amip_mpas_gpu_chain.sbatch` |
| output | `/scratch/b/b309178/amip_mpas/amip5y_2026-08-11/` |
| config | `config/amip/amip_production.yaml` |
| target | `TARGET_DAYS=1825` (5 years, noleap), started from day 0 |
| tree | branch `ap/amip-cmip6-integration` at `0b03cc832` |
| cadence | self-chaining; each link runs 11 h then checkpoints and resubmits |

Model: MPAS Voronoi level 5 (~240 km) / L30 sigma / dt 75 s / fp64. Bechtold
convection, Morrison microphysics, Sundqvist cloud, RRTMG radiation, Louis
turbulence, McFarlane+Hines GWD, COARE3 surface, multilayer Richards land.

**Two firsts for a production run here** (both visible in the launch line):
real sub-grid orography (`--subgrid-orography-file`) instead of a fabricated
500 m mountain over every ocean column, and Pierre's ERA5-tuned CLM land
albedo reaching radiation (`Land albedo: ... mean=0.276, max=0.718` — the
0.718 is glacier, i.e. Antarctica is finally white and not dark rock).

## Is it alive?

```bash
squeue -u b309178 -n amip_mpas_gpu_chain.sbatch
D=/scratch/b/b309178/amip_mpas/amip5y_2026-08-11
ls $D/checkpoint_day_*.npz | sed 's/.*day_0*//;s/\.npz//' | sort -n | tail -1   # latest sim day
tail -20 $D/slurm-*.out | grep -E "^  Day |COMPLETED|BLOWUP|chain"
```

Healthy progress looks like `Day  NNN.0: T=[...]K mean=~251K p_s=~985hPa
|u|_max=40-60m/s CWV=~24kg/m2` and, at each link end,
`MPAS run COMPLETED in ...s` followed by `[chain] reached N/1825 d`.

**Exit code is NOT evidence.** OOM kills and wallclock exits both surface as
success. Always read the model's own completion line.

## If a link fails

The chain resubmits itself on a clean link end. It does NOT resubmit after a
blowup or a node failure. To restart from the last good checkpoint:

```bash
cd /work/bd1083/b309178/diffESM/legoesm_ap
sbatch --account=bb1596 \
  --export=ALL,NAME=amip5y_2026-08-11,TARGET_DAYS=1825,\
REPO=/work/bd1083/b309178/diffESM/legoesm_ap,\
LEGOESM_JIT_CACHE_DIR=/scratch/b/b309178/legoesm_jit_cache \
  scripts/cluster/levante/amip_mpas_gpu_chain.sbatch
```

It picks up the newest `checkpoint_day_*.npz` in the output directory by
itself. **Always pass `--account=bb1596`.**

`LEGOESM_JIT_CACHE_DIR` is set deliberately: the launcher hard-codes another
user's scratch path (`amip_mpas_gpu_chain.sbatch:100`,
`/scratch/b/b381103/...`), so without the override every compilation-cache
write is denied and nothing is ever cached. Harmless to correctness, wasteful
in compile time.

## What a blowup looks like

```
Day  NNN.0: T=[nan,nan]K mean=nanK ...
BLOWUP at day NNN
Blow-up state written for autopsy: .../blowup_state_day_NNNN.npz
[chain] run_amip exit=1
```

Historically this lane died by a **stochastic single-column moist
detonation**: one tropical column goes from normal to NaN in under ~23
timesteps with no precursor in any global diagnostic. Root cause was an
orphaned rain-number ratchet, **fixed by #1476 (merged 2026-08-05)**. Three
pre-registered arms from a previously-lethal checkpoint survived 3/3 on the
fixed code where the old tree died 2/3. **This run carries the fix**, so a
detonation here would be a NEW failure and worth reporting on #1515.

If it does blow up: keep `blowup_state_day_*.npz`, note the day, and restart
from the last clean checkpoint. A restart is a fresh stochastic draw and may
well survive the same window.

## Reading the output

CMOR files land in `$D/cmor/{Amon,day,fx}/` and are flushed **only at a clean
link boundary**, not continuously — a mid-link kill loses that link's CMOR but
not its checkpoints.

```bash
python scripts/plot/plot_amip_pattern_eval.py $D \
  --months 1-12 --years 1979 --label amip5y_y1 \
  --out /scratch/b/b309178/amip5y_y1 --json /scratch/b/b309178/amip5y_y1.json
```

**Month/year matching is mandatory.** `pattern_eval` matches observations to
whatever months the run wrote; a partial year silently compares against the
wrong climatology. Always pass `--months` and `--years` explicitly.

Reference data: `/scratch/b/b309178/climateeval_data/` (CERES-EBAF for TOA
radiation, GPCP for precipitation, ERA5 for tas/prw). CERES `rsut` global mean
is **98.78 W/m²**, area-weighted, read from file — the number to compare
against. Note the epoch mismatch: model 1979-80, CERES 2000-2025.

## When stopping is the right call

- **A defect is found in the physics this run uses.** The run is only worth its
  GPU-hours if the configuration is one we would publish.
- **The aerosol-CCN coupling is adopted.** If `--aerosol-ccn` is switched on
  (see the handover issue), this trajectory becomes the pre-coupling baseline
  and a new one should start. Killing early costs a few GPU-hours; discovering
  it at year 3 costs the campaign.
- It is NOT worth stopping for: a single blowup (restart), a slow link, or the
  JIT-cache warning.

Stop with `scancel 26881914` (and any queued successor).

## Known limitations of this configuration

Each is a deliberate open decision, not an oversight — see the handover issue.

- **`--hines-launch-p` unset.** Non-orographic gravity waves launch at the
  surface and brake the Southern Ocean jet directly. The arm that tested a
  600 hPa launch gave +2.08 m/s against a pre-registered confirm floor of
  +2.5, i.e. UNRESOLVED. The deficit is still in this run.
- **`--aerosol-ccn` unset.** Droplet number is a global constant
  (100 cm⁻³), so clouds do not respond to the aerosol field the run loads.
- **`--mpas-vert-advection-scheme` left at `upwind`**, whose own help text
  documents +0.822 K/day of tropical UTLS warming. `van_leer` exists.
- **`--cloud-saturation-scheme` left at `liquid`** — correctly. The
  mixed-phase alternative was measured and makes both TOA components far
  worse while flattering the net.
