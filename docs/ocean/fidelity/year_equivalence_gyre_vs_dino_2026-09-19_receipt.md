# GYRE versus DINO: one-year equivalence to NEMO

Date: 2026-09-19

**Result.** GYRE is the closer case through day 180, but its temperature gap
accelerates and first loses the normalized comparison to the DINO twin at day
240. At day 360 the 3-D temperature RMS is **MEASURED** as `1.122357391e-2 K`
for GYRE and `3.861851488e-3 K` for the DINO twin. The older DINO standalone
result is **REPRODUCED-FROM-ARTIFACT; not rerun** at `4.222061753e-2 K`.

The predictions below were committed before either operator run was scored in
`88fff93420270e55f8dd47973828162a6c5e7209`. No model, NEMO source,
configuration, or physical constant was changed. The only new code is the
read-only scorer and its planted control.

## Metric and labels

Every cell in the central table is `absolute / normalized`:

- absolute is fp64 wet RMS of legoESM minus NEMO;
- normalized is that RMS divided by NEMO's population spatial standard
  deviation for the same field, wet mask, and day (`ddof=0`);
- T and S have units K and psu, u and v have units m/s, SSH has units m;
  normalized values are dimensionless;
- G = GYRE, DT = DINO twin, DS = DINO standalone;
- every G and DT cell is **MEASURED** from the completed 2026-09-19 operator
  artifacts; every DS cell is **REPRODUCED-FROM-ARTIFACT; not rerun**.

GYRE uses NEMO's full wet `tmask`/`umask`/`vmask`: 18,000 T/S cells,
17,400 u cells, 17,100 v cells, and 600 SSH cells. DINO uses the campaign's
canonical `twin_nemo_ts_maps.py` one-ring `[1:-1,1:-1]` population, extended
unchanged to u/v after dropping legoESM's redundant west/south face: 339,744
T/S cells, 333,952 u cells, 337,951 v cells, and 9,850 SSH cells. This choice
reproduces the preregistered DINO temperature series rather than silently
changing its population. It is not the existing tools' `frac_*` statistic,
which divides by change from rest rather than by NEMO spatial standard
deviation.

## The comparison

| day | G T | G S | G u | G v | G SSH | DT T | DT S | DT u | DT v | DT SSH | DS T | DS S | DS u | DS v | DS SSH |
|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| 30 | `6.890485e-05 / 9.521487e-06` | `1.178125e-05 / 1.578933e-05` | `5.579572e-06 / 4.979972e-04` | `4.620277e-06 / 3.291092e-04` | `6.802771e-06 / 1.346028e-04` | `6.817990e-04 / 1.365422e-04` | `6.112258e-05 / 1.352637e-04` | `1.811432e-04 / 4.371928e-03` | `9.802157e-05 / 2.101555e-03` | `5.249448e-05 / 1.396826e-04` | `5.697944e-02 / 1.141113e-02` | `3.417939e-03 / 7.563866e-03` | `4.327100e-03 / 1.044355e-01` | `3.622603e-03 / 7.766760e-02` | `4.311950e-03 / 1.147367e-02` |
| 60 | `1.932997e-04 / 2.782999e-05` | `2.012327e-05 / 2.739005e-05` | `2.050481e-05 / 1.550434e-03` | `7.642226e-05 / 5.013268e-03` | `1.680256e-05 / 2.154137e-04` | `1.118417e-03 / 2.159407e-04` | `1.157700e-04 / 2.537872e-04` | `2.982056e-04 / 6.815183e-03` | `1.645160e-04 / 3.466512e-03` | `5.789504e-05 / 1.472122e-04` | `4.517142e-02 / 8.721562e-03` | `3.141170e-03 / 6.885966e-03` | `3.381777e-03 / 7.728705e-02` | `4.621456e-03 / 9.737855e-02` | `3.873666e-03 / 9.849737e-03` |
| 90 | `1.864502e-03 / 2.732128e-04` | `2.347862e-04 / 3.239120e-04` | `1.584008e-04 / 1.053580e-02` | `1.710046e-04 / 1.003976e-02` | `3.540726e-05 / 3.644605e-04` | `1.943342e-03 / 3.655668e-04` | `1.761927e-04 / 3.836411e-04` | `4.542611e-04 / 1.234200e-02` | `3.401658e-04 / 7.587488e-03` | `5.187624e-05 / 1.316963e-04` | `3.953920e-02 / 7.437815e-03` | `2.900813e-03 / 6.316216e-03` | `4.476672e-03 / 1.216285e-01` | `2.527962e-03 / 5.638687e-02` | `3.912100e-03 / 9.931504e-03` |
| 120 | `1.050126e-03 / 1.536787e-04` | `2.427471e-04 / 3.365129e-04` | `1.314810e-04 / 7.869706e-03` | `2.634259e-04 / 1.372593e-02` | `4.582671e-05 / 4.182662e-04` | `3.802619e-03 / 7.007933e-04` | `3.911618e-04 / 8.471052e-04` | `5.701019e-04 / 1.598549e-02` | `3.498123e-04 / 8.378006e-03` | `4.665271e-05 / 1.174541e-04` | `4.202017e-02 / 7.743993e-03` | `3.338052e-03 / 7.228930e-03` | `3.490079e-03 / 9.786077e-02` | `2.453109e-03 / 5.875197e-02` | `3.531992e-03 / 8.892233e-03` |
| 180 | `3.580551e-03 / 5.078848e-04` | `9.001074e-04 / 1.252006e-03` | `1.273269e-04 / 6.673313e-03` | `1.699182e-04 / 7.696491e-03` | `7.857227e-05 / 6.429187e-04` | `3.427368e-03 / 6.111878e-04` | `3.329697e-04 / 7.152247e-04` | `5.230687e-04 / 1.421977e-02` | `3.821927e-04 / 9.698740e-03` | `4.944080e-05 / 1.222362e-04` | `4.302836e-02 / 7.673063e-03` | `3.467434e-03 / 7.448108e-03` | `2.881485e-03 / 7.833396e-02` | `1.887574e-03 / 4.790015e-02` | `3.331333e-03 / 8.236307e-03` |
| 240 | `1.644674e-02 / 2.276599e-03` | `1.231720e-03 / 1.716567e-03` | `3.072945e-04 / 1.381165e-02` | `4.360950e-04 / 1.729052e-02` | `1.934733e-04 / 1.466422e-03` | `3.887190e-03 / 6.763185e-04` | `3.485263e-04 / 7.439604e-04` | `9.916118e-04 / 2.865498e-02` | `8.971361e-04 / 2.296919e-02` | `7.014420e-05 / 1.715683e-04` | `4.234638e-02 / 7.367697e-03` | `3.491891e-03 / 7.453752e-03` | `2.745836e-03 / 7.934746e-02` | `1.878931e-03 / 4.810589e-02` | `3.189986e-03 / 7.802505e-03` |
| 300 | `1.359740e-02 / 1.909966e-03` | `1.210091e-03 / 1.689268e-03` | `3.155329e-04 / 1.216572e-02` | `2.959221e-04 / 1.038600e-02` | `1.787469e-04 / 1.215124e-03` | `3.619625e-03 / 6.207442e-04` | `4.037088e-04 / 8.573322e-04` | `4.139945e-04 / 1.209173e-02` | `3.195506e-04 / 8.254176e-03` | `4.714657e-05 / 1.138171e-04` | `4.195759e-02 / 7.195478e-03` | `3.486565e-03 / 7.404210e-03` | `2.531265e-03 / 7.393180e-02` | `1.648571e-03 / 4.258352e-02` | `3.150651e-03 / 7.606024e-03` |
| 360 | `1.122357e-02 / 1.658987e-03` | `1.117196e-03 / 1.574376e-03` | `4.350342e-04 / 1.553001e-02` | `4.259231e-04 / 1.448470e-02` | `1.931004e-04 / 1.149021e-03` | `3.861851e-03 / 6.568438e-04` | `3.888773e-04 / 8.221397e-04` | `6.092594e-04 / 1.814655e-02` | `5.270196e-04 / 1.399115e-02` | `6.372234e-05 / 1.520220e-04` | `4.222062e-02 / 7.181103e-03` | `3.567379e-03 / 7.541926e-03` | `2.534610e-03 / 7.549235e-02` | `1.730079e-03 / 4.592959e-02` | `3.092727e-03 / 7.378301e-03` |

## Preregistered predictions and verdicts

- **P1 — REFUTED.** GYRE day-360 T3D RMS is **MEASURED** as
  `1.122357391e-2 K`, 3.74 times the registered upper bound of `3e-3 K`.
  The requested power-law fit, defined post-hoc as ordinary least squares in
  log space over the eight registered days, gives `a = 2.28425`
  (`R² = 0.9300`). This is not DINO-like saturation: DINO's corresponding
  all-day exponent is `0.72783`, its late-day exponent is `0.13071`, and its
  day-360 value is 99.35% of its day-240 peak. GYRE overshoots to
  `1.644674193e-2 K` at day 240 and retreats to 68.24% of that peak by day
  360. Its late four-point fit is `a = 1.56756` with weak `R² = 0.4678`, so a
  single saturating power law is not a good description.
- **P2 — CONFIRMED.** DINO twin day-360 T3D RMS is **MEASURED** as
  `3.8618514881188677e-3 K`. It differs from the rounded registered target
  `3.862e-3 K` by `0.00385%`, and is bit-for-bit the same scored value as the
  prior artifact. The new day-360 snapshot SHA-256,
  `0b3189a4e2d44dcee0cbf0ad3ea95a33467e5f6bb0c8a05360e0aa8afd01bd60`,
  is also the prior r14 snapshot hash. Nothing changed except output cadence.
- **P3 — REFUTED first at day 240.** GYRE is lower through day 180. At day
  240 its normalized T error is **MEASURED** as `2.276599e-3`, versus
  DINO twin's `6.763185e-4`: GYRE is 3.366 times larger. It remains larger at
  days 300 and 360.
- **P4 — CONFIRMED.** Over the exact daily NEMO record at days 1--30, GYRE
  has three decreases. Its largest consecutive-day ratio is **MEASURED** as
  `10.26891493`, day 22 to day 23; day 23 is
  `5.283867390e-4 K`, reproducing the preregistered bump, and day 24 falls to
  `4.647409732e-5 K`. For DINO twin, the largest available exact consecutive-
  day ratio is **MEASURED** as `3.252472835`, day 1 to day 2. That DINO result
  is limited to days 1--5: the named NEMO artifacts then jump to 10-day
  cadence, so a full-year *daily* ratio is **UNMEASURED**, not interpolated.

## Initial state and noise floors

GYRE has no separate twin/standalone split. Its existing A1 artifact is
**REPRODUCED-FROM-ARTIFACT** from
`year_fromrest_head/alignment_gate.json` (SHA-256
`53f9304cf50e195d0e5ca96417c1d159c3ce4307a10066892da6705535a3fc69`):
the card's initial state and NEMO's step-1 BEFORE level have zero unequal cells
and exactly `0.0` maximum difference on T, S, u, v, and SSH. The fresh run's
first saved day has a **MEASURED** T3D gap of `7.565532850e-7 K`. Therefore
the GYRE result is divergence from a shared analytic initial state, not an
initial-state mismatch.

The GYRE NEMO-vs-NEMO T3D floor exists and is
**REPRODUCED-FROM-ARTIFACT** as the maximum pairwise NEMO-member RMS:
days 30/60/90/120/180/240/300/360 are respectively
`1.443890e-10`, `1.591236e-10`, `2.144227e-10`, `1.283173e-9`,
`2.986308e-10`, `3.304068e-10`, `2.595660e-10`, and `3.394535e-10 K`.
Thus the day-360 GYRE gap is `3.3064e7` NEMO floors from zero. The once-quoted
`6.806545e-10 K` at day 30 is real but was mislabeled: the source record says
it is `sqrt(spread_lego² + spread_nemo²)`, with legoESM spread
`6.651635e-10 K` and NEMO spread `1.443890e-10 K`; it is not a
NEMO-vs-NEMO floor. Source:
`year_fromrest_head/verdict_year.json`, SHA-256
`257bdb20032791164593cbf021071696733abf38e17ad3e01af01e2ffe3e1f64`.

For DINO twin and standalone, a NEMO-vs-NEMO T3D state-RMS floor is
**UNMEASURED in the named artifacts**. Existing DINO ensemble floors are for
different transport/density statistics or for legoESM's own spread; this
receipt does not relabel them. Consequently no honest DINO gap/floor ratio is
reported.

## Reference correction and controls

The operator's GYRE acquisition succeeded (`MEMBER_EXIT 0`, 360 snapshots),
but the requested scorer did not: `year_owners/nemo_seed0` ends at day 30, so
`day_gap_year.log` records `DAYGAP_EXIT 1` at day 31. The user-named
`decompose_day360.json` itself points to `year_fromrest`, not `year_owners`.
This receipt therefore uses `year_owners` for the exact daily days 1--30 and
the archived `year_fromrest` monthly continuation for later registered days.
Their shared day-30 restart is byte-identical, SHA-256
`853b3d41b2aa512e934430cc1fcbf36ea574c2148419d6c4b98a1e16db94cfc6`.
The two NEMO namelists differ in run length/output cadence and the explicit
zero perturbation selector; the byte-identical endpoint proves the seed-0
arithmetic agrees through their common interval.

DINO uses `RUN_TRAJ` through day 180, `RUN_VERDICT360_M0` at days 240 and
300, and `RUN_FROMREST_Y1` at day 360. All T/S/u/v/SSH fields are bit-identical
between `RUN_TRAJ` and the continuation at day 180, and between
`RUN_FROMREST_Y1` and the continuation at day 360. The tiled files themselves
need not hash identically because one set carries additional variables.

Controls all passed:

- a `0.25` plant in one of two wet cells moves RMS by the analytic
  `0.1767766953`, while a dry-cell plant changes nothing;
- all scored candidate arrays are stored fp64 and all arithmetic is fp64;
- the complete DINO acquisition hash manifest passes `sha256sum -c --quiet`;
- the selected GYRE and DINO reference-continuity checks above pass;
- the archived standalone day-360 T3D result is reproduced with absolute
  difference exactly `0.0`;
- the scorer checkout was clean at committed SHA
  `cb6becd39aa503ab1f7c63e0344c60057650a8b0`.

## Provenance

### GYRE, MEASURED

Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/gyre`.
The operator actually ran clean commit
`4d250301588d3ed0ad83fb20d6bf520e175d576e` on branch
`fidelity/nemo-testcases-l2-gyre-codex2`; the manifest SHA-256 is
`aef8a3f026ead8d360f3fa36d61dafb6fc590164d3a2501f19dc238b85e0c3c6`.
This is later than the preregistered GYRE tip
`c8f5d513df453f3f12a3d5eb05b05be8c8d3a2fd`, but its entire
`packages/ocean` tree is identical (`4b4280060ab1aec92100a622ad7cb3b8e2f9d8ca`)
and the year-runner blob is identical
(`24c5cc4e0c4541fd62f4cd04699869bbf9a257cf`). The intervening commits are
validation records/probes, not model changes.

Acquisition command:

```bash
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 \
  /home/dbalwada/legoESM/.venv/bin/python \
  scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_year_fromrest.py \
  --member 0 --days 360 --snap-steps 6 --tag year \
  --root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/gyre
```

The day-360 snapshot SHA-256 is
`47aaa021d6206cd83834a3feddfffedff4ba4f6e3645c51b28c29a2e435f670e`.

### DINO twin, MEASURED

Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/dino/twin_year_daily`.
Producer branch `fix/dino-faithful-grid-true-frame`, clean SHA
`f0a82e87ca545e55df548bfd5e0d55c50d1330f8`, fp64, GPU
`GPU-f425371f-af10-621f-60e0-b3c455a77e9f`; completion stamp
`2026-09-19T16:00:57.071906+00:00`. Generated metadata SHA-256:
`a83a9544d7935739558ca5b51ae28ebb78d82fbd6e5ed3634bc3345ab14795b4`.

Acquisition command:

```bash
CUDA_VISIBLE_DEVICES=GPU-f425371f-af10-621f-60e0-b3c455a77e9f \
JAX_PLATFORMS=cuda JAX_ENABLE_X64=1 FP64=1 \
  /home/dbalwada/legoESM/.venv/bin/python scripts/run/run_dino.py \
  --config scripts/experiment/dino/nemo_faithful_kamm_mlf.yaml \
  --days 360 --snapshot-every-days 1 \
  --output-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/dino/twin_year_daily
```

The named prior receipts are
`scripts/validate/ocean_fidelity/dino_1226/PREREG_surface_restoring_nemo_explicit.md`
and `docs/ocean/fidelity/dino_twin_nemo_ts_maps_day360_result.md`.

### DINO standalone, REPRODUCED-FROM-ARTIFACT; not rerun

Source:
`/data/abyssal/dbalwada/dino_fromrest_y1/lego_trueframe_fp64`, acquired
2026-09-08 20:31--20:42 America/New_York. Run-metadata SHA-256:
`e03a43a5f68a90a68b47467d10b83af56cf6aef0e2d45027f14c8fcd38669322`;
day-360 snapshot SHA-256:
`01040b15421d2018dec9821da9377743f2b98b8481dd3418b506bd6bfdd7b99c`;
prior scored JSON SHA-256:
`1d1ebcb801a47aa44fe598544b32f71665dbaf3a6420228161766abc6e17de0d`.
The metadata predates producer-git stamping, so the producer commit is
honestly unknown; the exact input bytes are pinned. This is the known
pre-#1729 standalone from-rest initialization/first-step-mismatch artifact,
which is why its T gap is already `5.69794e-2 K` at day 30 and remains in the
`0.04 K` class. It cost 636.5 GPU-seconds, but the task explicitly requested
the prior result and the operator did not rerun it.

### Scoring command and outputs

The exact command, input paths, per-snapshot hashes, NEMO restart paths and
hashes, masks, NEMO standard deviations, controls, growth fits, and daily rows
are in
`docs/ocean/fidelity/year_equivalence_gyre_vs_dino_2026-09-19_summary.json`
(SHA-256
`48188b5c991d8b10cd5ae473f26273ab854f661093a269db0e09aedd92c56c7c`).
It was produced by:

```bash
/home/dbalwada/legoESM/.venv/bin/python \
  scripts/validate/ocean_fidelity/testcases/year_equivalence_gyre_vs_dino_score.py \
  --gyre-lego-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/gyre \
  --gyre-early-nemo-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners \
  --gyre-year-nemo-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest \
  --gyre-mesh /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_owners/nemo_seed0/mesh_mask.nc \
  --dino-twin-dir /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/dino/twin_year_daily \
  --dino-standalone-dir /data/abyssal/dbalwada/dino_fromrest_y1/lego_trueframe_fp64 \
  --dino-early-nemo-root /data/abyssal/dbalwada/dino_fromrest_y1/nemo_earlydays \
  --dino-traj-root /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ \
  --dino-continuation-root /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_VERDICT360_M0 \
  --dino-day360-root /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_FROMREST_Y1 \
  --dino-mesh /home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/RUN_TRAJ/mesh_mask.nc \
  --dino-twin-hash-manifest /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_equivalence/dino/twin_year_daily.sha256 \
  --standalone-prior-day360-json /data/abyssal/dbalwada/dino_fromrest_y1/maps_trueframe_fp64/twin_nemo_ts_maps_day360_kt11520.json \
  --gyre-noise-floor-json /data/abyssal/dbalwada/nemo-testcases-l2/phase3/year_fromrest_head/verdict_year.json \
  --output /tmp/year-equiv-2467431738/docs/ocean/fidelity/year_equivalence_gyre_vs_dino_2026-09-19_summary.json
```

Self-check:

```bash
/home/dbalwada/legoESM/.venv/bin/python \
  scripts/validate/ocean_fidelity/testcases/year_equivalence_gyre_vs_dino_score.py \
  --self-test
```

Result: `passed: true`; wet plant recovered and dry plant excluded.
