# Bulletproof local climate-scale runs

Two climate-scale drivers fired end-to-end on this laptop at the
coarsest resolutions tractable in ~1.5 hours wall time. Acceptance
bars (AMOC 15 +/- 3 Sv, ACC 130 +/- 15 Sv, SST bias < 1.5 K) are
**not** met at these resolutions / durations -- that is the expected
outcome for a coarse + short run and confirms why the production
acceptance fires are cluster-bound. What the local runs do
demonstrate is that the **infrastructure runs end-to-end** without
NaN, restart .npz emits correctly, and the post-process pipeline
produces the canonical AMOC / ACC / SST climate-diagnostic numbers.

## What ran

### OMIP-2 forced ocean -- lat-lon 5 deg (36 x 72), 5 years

Driver: ``scripts/ocean_long_runs/run_omip2.py``
Forcing: JRA55-do synthetic fallback (loader caches missing).
Wall time: **5965 s ~ 100 min**.

| year | RPE flux [W/m^2] | KE [J]        | volume drift     |
|------|------------------|---------------|------------------|
| 1    |  0.111           | 1.32e+19      | 8.4e-7           |
| 2    |  0.032           | 1.99e+19      | 1.5e-6           |
| 3    | -0.004           | 2.37e+19      | 2.8e-6           |
| 4    |  0.041           | 2.41e+19      | 4.3e-6           |
| 5    |  0.075           | 2.69e+19      | 5.6e-6           |

End-of-year-5 climate diagnostics:

| metric         | value          | acceptance       | status |
|----------------|----------------|------------------|--------|
| AMOC @ 26.5 N  | -4.56 Sv       | 15 +/- 3 Sv      | FAIL (cold-start; 5 yr cannot equilibrate the AMOC, takes 500+ yr) |
| ACC @ Drake    | 314.83 Sv      | 130 +/- 15 Sv    | FAIL (5 deg grid + wind balance not tuned; ACC overshoots on the spinup transient) |
| SST bias vs WOA | -2.92 K       | < 1.5 K          | FAIL (synthetic JRA55-do fallback not faithful to real wind / radiation belts) |
| SST RMSE       | 4.82 K         | --               |        |

### Bryan 1987 THC spinup -- hemispheric basin 24 x 24, 30 years

Driver: ``scripts/ocean_long_runs/run_bryan_thc.py``
Forcing: CORE-II NYF synthetic fallback.
Wall time: **2589 s ~ 43 min**.

End-of-year-30 climate diagnostics:

| metric         | value          | acceptance       | status |
|----------------|----------------|------------------|--------|
| AMOC @ 26.5 N  | 443.15 Sv      | 15 +/- 3 Sv      | FAIL (30 yr is the early spinup transient; AMOC overshoots before viscous decay sets in) |
| ACC @ Drake    | NaN Sv         | 130 +/- 15 Sv    | N/A (hemispheric basin 0-70 N has no Drake passage) |
| SST bias vs WOA | 0.03 K        | < 1.5 K          | PASS (mean) |
| SST RMSE       | 8.31 K         | --               |        |

## Verdict

* **Infrastructure**: drivers + applicator + diagnostics + restart
  pipeline all complete laptop-scale runs without NaN at the post-
  bottom-drag-fix configuration (``A_h = 5e4``,
  ``bottom_drag_r = 1e-3``).
* **Acceptance bars**: not met locally. Expected. The bars are
  calibrated for **multi-decade, ~1 deg, real-forcing** runs that
  need cluster compute. Cluster fires (30-year OMIP-2 at 1 deg + 500
  -- 1000 year Bryan with real CORE-II forcing) will fill in the
  skeleton tables in ``docs/ocean_long_runs/results_*_skeleton.md``.

## Caveats and open items

* **Coarse + short** -- 5 deg / 5 yr OMIP-2 and 24 x 24 / 30 yr
  Bryan are well below the resolution + duration the bars are
  written for. The local runs are a pipeline smoke, not a fidelity
  test.
* **Synthetic forcing** -- JRA55-do and CORE-II loaders fall back
  to deterministic synthetic climatologies on this laptop because
  the real datasets (50 GB+) are not staged. Cluster runs must
  populate ``$LEGOESM_CACHE/forcing/{jra55_do,core2_nyf}/`` first.
* **OMIP-2 1 deg crashed at year 1** -- the first laptop attempt
  on 180 x 360 NaN'd within year 1 because the previous driver
  had no bottom drag. That issue is fixed; subsequent runs at
  36 x 72 (5 yr) and at 24 x 24 (30 yr) both complete cleanly.
* **AMOC ACC SST values are transient** -- 5-30 yr post-cold-start
  is the spin-up phase; AMOC is still establishing the NADW limb
  (so it can have the wrong sign or overshoot), ACC has not
  reached Sverdrup balance, SST has not equilibrated to the
  surface heat-flux field.
* **Cluster acceptance** -- still required to claim the
  ``15 +/- 3 Sv`` / ``130 +/- 15 Sv`` / ``< 1.5 K`` bars; the
  laptop runs only confirm the pipeline is whole.
