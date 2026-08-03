# AMIP configuration outcomes: archive walk (2026-08-03)

Source archive: `legoesm_pg/amip_runs` (a colleague's AMIP run archive, 205 run
directories as of this walk). Compared against this repo's production
configuration, `config/amip/amip_production.yaml`, and two in-flight 1-year
arms at `/scratch/b/b309178/amip_ccab/{cc_on,cc_off}`.

This document reports what the archive contains and what that scope implies.
It does not assign responsibility for any run's outcome.

## 1. Method and its limits

- 205 directories under `amip_runs/` are model-run outputs (3 entries —
  `_figs`, `monitor_figs`, `slurm_logs` — are shared plotting/log-collection
  folders, not runs, and are excluded).
- 161 of 205 have a `results.txt` (grid/resolution/levels/dt/target-days,
  `COMPLETED` or `BLOWUP at day N`). 192 of 205 have `experiment_config.json`
  (the full resolved physics config). 44 directories have no `results.txt`
  (31 of those do have a config; 13 have neither — no evidence beyond a
  SLURM log, if that).
- **A directory can hold more than one SLURM job.** Long runs are
  checkpointed and, on hitting the wall-clock limit, resubmitted with
  `--restart-from <checkpoint>` to continue the *same* experiment. When that
  happens, `results.txt` is overwritten by the *last* job and reports only
  that job's own requested day count — not the run's total simulated
  duration. This was confirmed directly: `climeval_bech`'s `results.txt`
  reads "65 days, COMPLETED" (the second of two jobs), but its own SLURM logs
  show job 1 requested `--days 365`, job 2 restarted from
  `checkpoint_day_0300.npz`, and the run's internal chain-tracking
  (`[chain] reached 365/365 d — chain complete.`) confirms the full 365 days
  were simulated continuously. A read of `results.txt` alone would have
  under-counted this run by 300 days.
  - Because this changes the answer to "did any run reach 365 days,"
    **every directory was cross-checked**: all SLURM logs (not just the
    current `results.txt`) were searched for `--days 365` (a segment that
    explicitly requested a full year) and for
    `[chain] reached N/365 d — chain complete.` (a verified, chain-aware
    completion). This is the basis for the counts in §3.
  - **`BLOWUP at day N`** was separately confirmed to already report the
    model's cumulative day count, not a per-segment count (checked on
    `century3_rc2`, which restarted from its own `checkpoint_day_0130.npz`
    and reported `BLOWUP at day 673.0`, consistent with 130 + 543 more days
    before failing) — so blow-up days in the table below did not need this
    correction.
  - Some directories hold checkpoints that **do not start near day 0**
    (e.g. `gwdA_base`'s earliest file is `checkpoint_day_0080.npz`): these
    are short physics-variant branches seeded from state copied in from
    elsewhere, whose *own* config governs only the days after the seed
    point. The config for the seed portion is not recoverable from the
    directory. These are marked "branch-seeded" below and excluded from
    "longest continuous run" claims for that reason (they are still counted
    in completion-rate denominators, since COMPLETED/BLOWUP status itself is
    unaffected by this ambiguity).
- **Code version is not controlled across the archive.** Of 191 runs whose
  `run_manifest.json` records a commit, there are **53 distinct commit
  SHAs**, and **135 of 191 (71%) were run with `git_dirty=True`** (uncommitted
  local changes on top of the recorded commit). Two different repository
  checkouts were used (`.../legoesm_pg/legoESM`, 140 runs, and
  `.../legoesm_pg/legoESM_campaign`, 16 runs — the other manifests are
  unparseable or missing). The four runs central to §3 below each ran on a
  **different** commit SHA. This is a hard limit on any cross-run comparison
  in this archive; see §6.
- Could not determine: cause of non-completion for two directories
  (`pilot1y_sigL40`, `pilot2_landfix_cap10`) that have neither `results.txt`
  nor any checkpoint file — the model appears to have started (SST-forcing
  and setup lines are present in the SLURM log) but left no further trace,
  and no error is visible in the captured log tail. Also could not determine:
  the config for `llH_mlland` (a bechtold + multilayer-land + latlon,
  365-day-target attempt) since it failed during
  `ExperimentConfig.validate_strict()` before `experiment_config.json` was
  written; its intended config is reconstructed here from its SLURM log's
  recorded command line only, and is labeled as such.

## 2. Outcomes by physics configuration

Grouped by the three knobs that vary in the archive: convection scheme, land
model, and grid. Counts and COMPLETED/BLOWUP status are read directly from
each directory's own `results.txt`; "longest completed" is chain-corrected
per §1 (self-originated chains only — see footnotes for branch-seeded caveats).

| Convection | Land | Grid | Runs (n) | Completed | Completion rate | Longest completed (days) |
|---|---|---|---:|---:|---:|---:|
| bechtold | not multilayer (bucket/slab) | mpas | 84 | 62 | 74% | 365 [a] |
| sbm | not multilayer (bucket/slab) | mpas | 54 | 17 | 31% | 365 [b] |
| bechtold | not multilayer (bucket/slab) | latlon | 13 | 3 | 23% | 120 [c] |
| sbm | not multilayer (bucket/slab) | latlon | 6 | 1 | 17% | 300 |
| **bechtold** | **multilayer (Richards soil)** | **latlon** | **2** | **0** | **0%** | **— (168 max, blew up)** |
| tiedtke | not multilayer | mpas | 1 | 1 | 100% | 40 |
| zhang_mcfarlane | not multilayer | mpas | 1 | 1 | 100% | 40 |
| bechtold | multilayer (Richards soil) | **mpas** | **0** | **—** | **—** | **no archived attempt** |

[a] Verified via chain log on `climeval_bech`/`climeval_bech_sm` (365/365,
2 SLURM jobs each, self-originated — checkpoints present from near day 0).
A naive `results.txt`-only read would report 200 days here (`llN_cland`),
but that run is branch-seeded from a day-150 checkpoint of unknown
provenance, so 200 is not a verified single-config span; 365 is.
Branch-seeded chains in this group reach as far as 580 days (`clamp_on`),
but the pre-seed portion's config cannot be confirmed, so it is not used as
the headline figure.

[b] `qcapv2_pilot1y` (1 job) and `climeval_pilot2` (2 jobs, chain-verified
365/365) — no correction needed relative to the naive reading for
`qcapv2_pilot1y`; `climeval_pilot2`'s naive reading was 35 days.

[c] Naive reading of this cell would say 200 days (`llN_cland`), but that run
is branch-seeded (checkpoints start at day 150); the longest *self-originated*
bechtold+latlon completion is 120 days.

The bottom two rows are the ones directly relevant to the production
configuration (§4).

## 3. Runs that targeted a continuous 365-day integration

A directory was counted here if **any** of its SLURM logs shows a job
requesting `--days 365` in one segment (i.e., a stated attempt to reach a
full simulated year, whether or not it was later split across multiple
wall-clock-limited jobs). **17** such directories were found — 6 more than
a naive `grep` of current `results.txt` files would find, because 6 of the
17 had their `results.txt` overwritten by a later, shorter restart segment.

| Outcome | Count | Directories |
|---|---:|---|
| **Completed** (365/365, chain-verified) | 4 | `qcapv2_pilot1y` (sbm), `climeval_pilot2` (sbm), `climeval_bech` (bechtold), `climeval_bech_sm` (bechtold) |
| Blew up | 10 | days 3–168 (see table below) |
| Failed before integration started | 1 | `llH_mlland` — `ExperimentConfig.validate_strict()` rejected the config (`surface_tiled=True` requires `turbulence` in `('louis','clubb_lite','clubb')`; this run requested `turbulence=holtslag_boville`) |
| No recorded outcome | 2 | `pilot1y_sigL40`, `pilot2_landfix_cap10` — no `results.txt`, no checkpoints |

All 4 completions: MPAS grid, level 5, L30, dt 75 s, fp64, **use_multilayer_land = False**.
Two used `convection=sbm`, two used `convection=bechtold`. Convection scheme
did not distinguish completion in this set; land model (see below) did.

Land-model split of the 17 attempts:

| Land model | Attempts | Completed | Notes |
|---|---:|---:|---|
| Not multilayer (bucket/slab) | 14 | 4 | 8 blew up (day 10–160), 2 no recorded outcome |
| Multilayer (Richards soil) | 3 | 0 | 2 blew up (day 168, day 3), 1 failed before integration (config-validation error) |

## 4. The completed 365-day configurations, stated exactly

There is **not** a single completed 365-day configuration in this archive —
there are two distinct physics recipes, each completed twice:

**Recipe A** (`qcapv2_pilot1y`, single SLURM job; `climeval_pilot2`, 2 jobs,
chain-verified 365/365): MPAS level 5, L30, dt 75 s, fp64;
`convection=sbm`, `convective_precip_efficiency=0.0`,
`gravity_wave_drag=mcfarlane`, `use_multilayer_land=False`,
`microphysics=morrison`, `cloud_scheme=sundqvist`.

**Recipe B** (`climeval_bech`, `climeval_bech_sm`, each 2 jobs, chain-verified
365/365): MPAS level 5, L30, dt 75 s, fp64; `convection=bechtold`,
`convective_precip_efficiency=0.0`, `gravity_wave_drag=mcfarlane+hines`,
`use_multilayer_land=False`, `microphysics=morrison`, `cloud_scheme=sundqvist`.

Both recipes: MPAS grid, non-multilayer (bucket/slab) land. Each of the four
runs was on a different git commit (`ad7c1b1a…`, `87a6801d…`, `649d8480…`,
`621781aa…`; two of the four with uncommitted local changes at run time —
see §1), so this is not a controlled comparison between the two recipes.

## 5. Configurations that have never completed, with run counts

- **`convection=bechtold`, `use_multilayer_land=True`, `grid=latlon`: 3
  attempts, 0 completions.** `llH2_ml_hb` blew up at day 168; `llI_fullphys`
  blew up at day 3; `llH_mlland` failed `validate_strict()` before any
  integration step (config incompatibility, not a numerical blow-up).
- **`convection=bechtold`, `use_multilayer_land=True`, `grid=mpas`: 0
  archived attempts at any length.** This is the exact grid/convection/land
  combination in `config/amip/amip_production.yaml`. No run at this
  combination — completed, blown up, or otherwise — exists in this archive.
  The two in-flight arms at `/scratch/b/b309178/amip_ccab/{cc_on,cc_off}`
  (started 2026-08-03, both `use_multilayer_land=True`, `convection=bechtold`,
  MPAS level 5/L30/dt 75, differing only in `convective_cloud`) are, as far
  as this archive shows, the first attempts at this combination. As of this
  walk they are in progress (day 60 of 365, no `results.txt` yet).

## 6. Observations

- The one archived configuration that reached a full simulated year via a
  single SLURM job (`qcapv2_pilot1y`) uses `convection=sbm` — a relaxation
  closure (per `amip_production.yaml`'s own comment: "SBM relaxation") — and
  bucket/slab land. The production configuration uses `convection=bechtold`
  — a mass-flux closure — and the multilayer Richards-soil land model. Two
  archived runs (`climeval_bech`, `climeval_bech_sm`) show that
  `bechtold` + bucket/slab land can also complete a full year on this grid;
  none show `bechtold` + multilayer land completing at any length. In this
  archive, convection scheme did not separate completions from blow-ups;
  land model did.
- The multilayer Richards-soil land model and the sbm/bucket "year 1"
  configuration are different pieces of work of different maturity in this
  archive: the bucket-land configuration has 4 verified full-year
  completions (2 sbm, 2 bechtold, §3); the multilayer-land configuration has
  0 completions at any length and 0 archived attempts at all on the
  production grid (mpas). This is a description of what has and has not been
  exercised in the archive, not an assessment of either model's correctness.

## 7. Confounds

This archive is **not** a controlled experiment and cannot be read as one:

- **Grid, convection scheme, land model, timestep, vertical levels, and code
  version all vary simultaneously** across the 161 runs with results. A
  completion-rate difference between two rows of the §2 table cannot be
  attributed to any single knob from this data alone — e.g. the two
  archived bechtold+multilayer-land attempts are also the only two
  bechtold+multilayer-land runs on the **latlon** grid at `dt=150s`, versus
  the production `mpas`/`dt=75s` combination; grid and timestep, not just
  land model, differ from production.
- **Code version was not held fixed.** 53 distinct commit SHAs and a 71%
  "dirty working tree" rate among manifests (§1) mean two runs with the same
  reported physics knobs may not have run the same code.
- **Restart chaining conflates job-count with physics behavior only if not
  corrected for** (§1); this walk corrected for it for the 365-day question
  specifically (§3) and for the "longest completed" column of §2, but did
  **not** re-verify chain provenance for every one of the 161 results — only
  for the entries flagged in the footnotes of §2 and the 17 runs in §3.
  Completion/blow-up status itself (not duration) is unaffected by this,
  since it is read fresh from the terminal segment of each chain.
- A completion rate computed across this heterogeneous set is descriptive of
  what has been tried and how it went, not a causal statement that any one
  knob (grid, convection, land model, or timestep) is responsible for
  instability. A controlled one-variable A/B on the land model, holding
  everything else — grid, convection, timestep, code version — fixed, is
  needed to make that claim, and is being run separately from this archive
  walk.
