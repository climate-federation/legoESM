# NEMO testcase fidelity receipt — lane 3b, SI3 Phase 2b

Tracker: `climate-federation/legoESM#1699`

## Verdict

**DEBT.**  The two named near-bar rows are measured float re-association, not
different physics or a time-level error.  A scalar binary64 replay of NEMO's
written operation order reproduces every dumped Thomas forward state,
temperature solution, iteration-2 `qns_ice`, and final kt3 ice enthalpy at
**0 ULP**.  The executing legoESM path uses the mathematically equivalent
normalised shared Thomas solve and first differs by 1 ULP in the Picard
temperature solution.  No physical selector or SI3 equation was changed.

The complete oracle-entry sweep reads all 8 frames at all 8,760 steps: 70,080
thermodynamics frames and 621,960 field rows.  Its first over-bar frame remains
`kt=3 POST_ZDF.e_i`, absolute `7.152557373046875e-7 J m-3`, normalised
`2.559660701514903e-15` against the fixed `1e-15` bar.  Later measurements are
reported as debt, not as trajectory agreement.

The independently reported continuous driven column accumulates large seasonal
debt.  Its end-year normalised errors include `0.1061` in `t_su`, `0.9098` in
`e_i`, and `0.7932` in ice thickness.  This rung therefore makes no coupled or
full-trajectory fidelity claim.

## Coupled-rung debt

| component | status | reason |
|---|---|---|
| NEMO bulk-flux computation (`sbcblk`) | **DEBT / NOT CERTIFIED BY THIS RUNG** | The isolated column consumes NEMO's dumped entry `qns_ice` and `dqns_ice`.  It tests SI3 downstream of that boundary and does not recompute or certify `sbcblk.F90:1273,1480-1491` feeding `icestp.F90:201,206`. |

## Arithmetic replay

The replay follows the active source, not an analytic replacement:

- surface-flux update: `icethd_zdf_bl99.F90:367-379`;
- unnormalised forward elimination/back substitution:
  `icethd_zdf_bl99.F90:516-558`;
- post-convergence enthalpy: `icethd_zdf_bl99.F90:799-804` and
  `icevar.F90:938-946`;
- executing legoESM normalised solve:
  `packages/core/legoesm/timestepping/tridiagonal.py:103-159`;
- executing BL99 call and enthalpy update:
  `packages/ice/legoesm/ice/bitz_lipscomb.py:335-496`.

For both Picard iterations at kt1 and kt3, NEMO-order forward and solution
replays are 0 ULP from the dumps.  The scalar normalised-order probe is 1 ULP
from each solution.  This is the first arithmetic divergence, before either
reviewed output row.

| reviewed row | executing legoESM residual | NEMO-order replay | classification |
|---|---:|---:|---|
| kt1 iteration-2 `qns_ice` | `2.5011104298755527e-12 W m-2`; 176 ULP; normalised `2.0979870951387943e-14` | exact, 0 ULP | **FLOAT RE-ASSOCIATION** |
| kt3 `POST_ZDF.e_i`, layer 2 | `-7.152557373046875e-7 J m-3`; 12 ULP; normalised `2.559660701514903e-15` | exact, 0 ULP | **FLOAT RE-ASSOCIATION** |

The scalar normalised-order `qns_ice` surrogate is 48 ULP from NEMO, while the
actual JAX/XLA execution is 176 ULP away; compiler association within the
normalised solve is therefore disclosed rather than conflated with the exact
scalar surrogate.  The decisive discriminator is that the NEMO-written order
closes both rows to 0 ULP (within the preregistered 2-ULP threshold), with
identical dumped inputs.  Because this is not an operand/time-level difference,
the preregistered private one-variable replacement arm and SI3-identity repair
are inapplicable.

## Full-year protocols

The oracle-entry operator sweep restarts from each NEMO ENTRY frame and uses
the exact same-step ZDF input frame.  It uses eager, batched execution to retain
the arithmetic path independently reproduced by the reviewed Phase-2 gate.
It does not stop at debt: all 70,080 frames are cursor-, shape-, finite-, and
dtype-checked.  There are 116,274 over-bar field rows; the largest normalised
row is `1.1108092231885993e8`.  That maximum is an inventory statistic, not a
claim of a useful relative error where the oracle scale is below one.

The continuous protocol starts once at kt1 ENTRY, then advances legoESM's own
EXIT state under all 8,760 hourly NEMO input frames.  It is JIT executed, as
stamped in the JSON, and is deliberately not used to redefine the eager
operator sweep's first divergence.  Metric below is
`max(abs(legoesm-NEMO))/max(1,max(abs(NEMO)))` at EXIT.

| step | `t_su` | `e_i` | `e_s` | `h_i` | `h_s` | `a_i` | `sv_i` |
|---:|---:|---:|---:|---:|---:|---:|---:|
| 1 | `5.5627e-16` | `2.4890e-15` | `6.6565e-16` | `0` | `5.5511e-17` | `0` | `0` |
| 10 | `4.5223e-16` | `5.3075e-15` | `1.2532e-15` | `2.2171e-16` | `3.8858e-16` | `0` | `1.5636e-16` |
| 100 | `2.9230e-5` | `7.5693e-7` | `6.6610e-4` | `8.7949e-13` | `1.3048e-4` | `0` | `3.5789e-12` |
| 1000 | `8.3686e-4` | `6.9246e-5` | `1.8817e-2` | `2.1059e-6` | `5.2206e-3` | `0` | `9.4599e-6` |
| 8760 | `1.0611e-1` | `9.0984e-1` | `1.7346` | `7.9324e-1` | `4.3955e-1` | `0` | `5.3097e-1` |

The committed JSON contains every per-step value rather than only these five
requested slices.  At the snow-free seasonal state the ENTRY bridge now uses
NEMO's own rule: `ice_thd_1d2d` sets volumetric snow enthalpy to zero when
category snow volume is at or below `epsi20` (`icethd.F90:416-434`).  The
reader fails on non-finite values; it does not accept `0/0`.

## Phenomenology and measurable precision floor

Dates use the preregistered 2018 UTC hourly/daily index.  Onset is the first
daily EXIT beginning seven consecutive changes of the required sign after the
model's maximum or subsequent minimum.  The available floor is legoESM
float32-vs-float64, not a NEMO float32/FCT spread.

| quantity | NEMO | legoESM fp64 | legoESM fp32 | fp64-NEMO distance | fp32-fp64 floor | status |
|---|---:|---:|---:|---:|---:|---|
| minimum thickness | `0.5550952377 m`, 2018-09-09 00Z | `1.7824334139 m`, 2018-09-19 00Z | `1.7815639973 m`, 2018-09-19 00Z | `1.2273381762 m` | `8.6941662e-4 m` | **ABOVE-FLOOR** |
| maximum thickness | `2.4456886226 m`, 2018-05-13 00Z | `2.4457929329 m`, 2018-05-13 00Z | `2.4456565380 m`, 2018-05-13 00Z | `1.0431027e-4 m` | `1.3639490e-4 m` | **AT-FLOOR** |
| melt onset | 2018-05-14 | 2018-05-14 | 2018-05-14 | `0 d` | `0 d` | **AT-FLOOR** |
| growth onset | 2018-09-09 | 2018-09-19 | 2018-09-19 | `10 d` | `0 d` | **ABOVE-FLOOR** |

The matching melt-onset dates and maximum-thickness row are only the measured
phenomenology classifications above.  They are not evidence that the yearly
state trajectory is faithful.

## Inputs, run roots, and hashes

The measurement root is
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_phase2b_replay`.
It is a full 8,760-step CPU/no-MPI copy-run ending `STOP 0`, using copy-only
NEMO sources under
`/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3thd_phase2b_src`.
Its thermodynamics stream, final restarts, and `ocean.output` are byte-identical
to the accepted Phase-2 input run.  No shipped NEMO source or configuration was
modified or deleted.  No stale run root was deleted.

| artifact | SHA-256 |
|---|---|
| official archive (`MD5 9456e6a0a84d40630ad1804fd4061caf`) | `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| ERA5 North Greenland member | `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| replay `nemo.exe` | `4cefe8fd94b14e2e0031faaec85ea0733d10710c686f0f06fda21419b1cbbd57` |
| `namelist_cfg` | `6151c0fdd2431d07c7897d5852846a620edd58c55255f11b7fb3569803b5342c` |
| `namelist_ice_cfg` | `da7b4fc5865edf6a6a912d6316a51f6b845aaf7e8b87e9da121c6f0a46278033` |
| 70,080 thermodynamics frames | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| 8,760 exact ZDF-input frames | `5522eadce595408b00065fa30d8b41fccb3815bee76d6fbf5ba3adbb2656cb27` |
| kt1+kt3 BL99 operand frames | `aad46579fb2d2cc19299d7a25992802525603bb9858adf1e892ff5d85bf40442` |
| kt3 enthalpy operands | `4b832b0c274d6aab032f16958224ebfb6fea603af22e1a1f1d474ff71b1e4589` |
| final ice restart | `b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e` |
| final ocean restart | `84ed40c5e5d46f9830f4c203b3e3a79f6dfffaf142dc4c44347648e2cd9f5265` |
| `ocean.output` | `3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430` |
| copy-only `MY_SRC/icethd.F90` | `80fff2751a240a9e87f8692eec25c9bbcbff667d3df049a9e2a58bd89f79fb1c` |
| copy-only `MY_SRC/icethd_zdf_bl99.F90` | `b94881987515fbc5ecde3f24f331276e56a7ba884e5a96e0c40fe2e324f55bab` |
| year gate at implementation commit `46a5da23ff` | `e23e40354de389d6a746bc2c75e4a73db45251d53a013fbcc5deb9f64e8ef7b3` |
| Phase-2b unit test | `dd64eb96e5ab2e909fdd23c41ee4459ce403b07c9cfb22aa8b2a92e891da8268` |
| full per-step JSON at `46a5da23ff` | `6c21d14f3c85d0fa99be7e31770a4a4dcad548d98f857ea78455bd6344622160` |

The archive provenance remains NEMO `sette_inputs` r5.0.0 and the Phase-1
pinned official URL.  No synthetic forcing was used.  The expanded exchange
stream is not consumed by this gate: it includes unwritten C1D halo storage
whose digest changes across otherwise byte-identical runs.  The deterministic,
selected-category ZDF-input stream above is the actual card boundary and is
hash-pinned.

The requested oracle-root move from
`c1d_omip_l3_sasice_scope_gate2` to
`c1d_omip_l3_sasice_phase2_inputs` is **ASKED**.  The two roots' namelists are
byte-identical: the `namelist_cfg` SHA-256 is `6151c0...342c` and the
`namelist_ice_cfg` SHA-256 is `da7b4f...8033` in both.

## Controls and tests

The ordinary full-year CLI writes the complete JSON and exits 1 because the
scientific verdict is DEBT.  Both independent plants also exit nonzero:

- `--plant-arithmetic`: changes the NEMO solution by 3 ULP and raises
  `NEMO operation-order replay exceeds 2 ULP: 3`;
- `--plant-truncate`: withholds the final expected frame and raises
  `planted truncated thermodynamics stream`.

The direct Phase-2 and Phase-2b suite reports:

```text
============================= 16 passed in 16.46s ==============================
```

The hardcoded-constant audit reports no hit in each touched Python file.  The
collected exact test node for the new test file reports:

```text
============================== 1 passed in 0.43s ===============================
```

The audit's current `/tmp` path filter omits scripts from pytest collection, so
the same `banned_hits` function was invoked directly on both touched gate
scripts and the test file; all three printed `[]`.  This limitation is
disclosed rather than calling the script checks collected tests.

## End-of-task choice register

- ASKED — add the coupled-rung `sbcblk` non-certification as a plain DEBT row.
- ASKED — record the oracle-root move and byte-identical namelists.
- ASKED — classify both near-bar rows using a NEMO-order, at-most-2-ULP replay.
- ASKED — fix only a genuine operand/time-level difference inside the fixed
  SI3 identity; neither row met that discriminator, so no model fix landed.
- ASKED — consume all 8,760 steps and 70,080 frames, retain per-step metrics,
  and report steps 1, 10, 100, 1000, and 8760.
- ASKED — compare annual thickness extrema and onset dates against a measured
  legoESM fp32/fp64 floor, with literal AT-FLOOR/ABOVE-FLOOR labels.
- ASKED — use only the ORCA1-resolved identity, CPU execution, copy-only NEMO
  instrumentation, explicit pathspec commits, local-git bundle, and no push.
- UNASKED — certify NEMO bulk fluxes, construct alternative ice identities,
  infer an unavailable NEMO precision/scheme spread, or modify shipped NEMO.

## FLAGGED FOR FUTURE DELETION

Nothing was deleted.  The earlier exploratory run roots, including
`c1d_omip_l3_sasice_phase2b_oracle`, remain on disk and are not acceptance
artifacts.  They are flagged for a future owner decision only; this dispatch
does not authorise their removal.
