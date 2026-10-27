# GYRE high-resolution round 15: NEMO budget-recorder closure repair

Date: 2026-10-19

## Outcome

The day-30 NEMO refusal is an instrumentation extent defect, not a missing or
double-counted physical tendency.  The no-surface-stress vertical-solve
companion had 122 inner-domain columns and was passed whole to a recorder row
with NEMO's 126-column native extent.  Its shifted intermediate cancels in the
vertical-plus-stress sum through native column 122, but the recorder leaves the
vertical row zero at native column 123 while recording a nonzero stress row.
That eastmost wet V column accounts for every word over the registered closure
bar.

The writer patch now allocates the companion on the full native `jpi` extent,
so its indices align with the production `pvv` array at the recorder boundary.
This changes only a private WRITE-only `MY_SRC` companion; no value flows back
to NEMO, legoESM, or any shared physics path.  The exact short verification and
corrected 180-day acquisition remain operator work.  No owner attribution is
made from the refused NEMO archive.

## Finding, disposition, and evidence

The committed diagnostic at `0e9b32e7a77d3c1ab02e1c826dea730e6181c6a4`
read the already-existing refused record rather than rerunning either model:

```text
input: round14/.../instrumented/oracle_gyre_r14_budget_day0030.bin
input sha256: 7a76204572750eed9336aa287f24cb55b4968d648ef09cb6c4990b341e7e7795
wet words: 9480
words above 1e-12 m/s: 79
failing zero-based output columns: [120]
eastmost wet zero-based output column: 120
maximum residual: 0.06024934686392794 m/s
maximum location: output [j=16, i=120], native Fortran [j=19, i=123]
maximum residual west of the eastmost wet column: 2.2436208706871907e-16 m/s
```

At the maximum, the recorded vertical-mixing row is exactly zero, the
surface-stress row is `-0.06719176099072825 m/s`, and the independently
recorded total is `-0.00011298168891713065 m/s`.  All 79 failures occupy the
same eastmost wet column; the other 9,401 wet words close at roundoff.  The
machine-readable result is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/round14_day30_closure_diagnosis.json`.

The same diagnostic compared the round-14 control and instrumented terminal
restarts: all 2,611,081 numeric words across 28 variables are byte-identical,
with identical SHA-256
`a8e33a3d22e4483bac777839f6cdc9bdf8c2e348c3a1d638adeafa114fff1c12`.
The bad row therefore did not move the NEMO trajectory.  It made the diagnostic
decomposition false at a recorder boundary.

Disposition: retain all eight preregistered rows and the exact
`1.0e-12 m/s` pointwise bar; repair the shape mismatch rather than excluding a
wet boundary word or relabelling a remainder.  The synthetic V mask now matches
the real 120-by-79 wet rectangle, and independent closure plants fire both at
an interior word and at zero-based output column 120.  The source regression
also refuses restoration of the 122-column companion declaration.

## Short verification, then corrected acquisition

The minimal verification is a 30-day run from rest.  The registered window
begins at step 697 and day 30 ends at step 720, so this is the shortest run that
can emit a complete 24-step registered daily record.  `--run-short` builds a
separate single-rank writer configuration, runs one binary with the recorder
off and on, requires a byte-identical terminal restart, requires exactly the
day-30 record, and applies the unchanged pointwise closure bar.  It writes
`momentum_budget_short_verification.json`; this artifact is verification-only
and cannot masquerade as the 151-day budget contract.

From a clean checkout of this round's final commit, run the short proof first:

```bash
EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_verify_day30 RANKS=1 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-r15-SXES52/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/run.sh --run-short
```

Only after that prints `ACQUISITION_DONE` and its short-verification JSON says
`PASS`, run the corrected 180-day acquisition:

```bash
EVIDENCE=/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_180d_corrected RANKS=1 /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-r15-SXES52/scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget/run.sh --run
```

The read-only source/deck/syntax preflight passes and prints both commands.  It
pins the vector-math NEMO engine, the exact `r4_smag` deck, one rank, 3,600-s
steps, daily cadence, the registered term order, and the same-binary passive
control.  It uses distinct round-15 configuration names and evidence roots, so
the failed round-14 evidence remains immutable.

After the corrected full run packages successfully, the joint admission is:

```bash
cd /data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gyrehr-r15-SXES52 && PYTHONPATH=packages/core:packages/ocean:packages/atmosphere:packages/coupler:packages/ice:packages/land:packages/ml:packages/tools:src JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 /home/dbalwada/legoESM/.venv/bin/python scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_hires_budget_preflight.py --legoesm-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/legoesm_runs/r4_smag_gpu_budget_180d_round12 --nemo-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_hires/gyre_r4_5yr_smag --legoesm-budget-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round14/legoesm_gyre_r4_smag_budget_180d_round14 --nemo-budget-root /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/oracle_gyre_r4_smag_budget_180d_corrected/instrumented --output /data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyrehr_rounds/round15/momentum_budget_admission.json
```

## What the legoESM record can establish alone

The completed legoESM record remains usable for recorder validation and
single-model diagnostics.  Its contract covers all 151 days, closes over all
9,480 wet surface-V words with a maximum residual of
`1.5151725360484924e-16 m/s`, and its recorder-on/off proof moves zero of
2,818,060 words.  Its per-term fields may be inspected as legoESM tendencies.

They cannot name the cross-model owner.  Until the corrected NEMO contract
passes, term-by-term model differences, budget closure between providers, and
any closure-learning target remain `UNMEASURED`.  In particular, the refused
NEMO row must not be compared after silently dropping its east boundary.

## Inherited pins / OPEN

Inherited unchanged: the `r4_smag` deck, 31 levels, RK3, split-explicit free
surface, TKE/EVD, Redi tracers, the literal 360-day forcing clock, vector-math
NEMO reference, native-fp64 legoESM GPU policy, registered days 30--180 window,
eight-term registry, and absolute `1.0e-12 m/s` pointwise closure bar.  No
physics, card, namelist value, model driver, or existing evidence artifact
changes in this round.

OPEN, in order:

1. Operator runs the exact short proof above and checks its PASS artifact.
2. Operator runs the corrected 180-day NEMO pair and packages it.
3. Run the joint admission command; only then score the preregistered
   days-30--180 western-boundary term differences and closure.
4. Use that measured owner to decide whether shared-path repair or learned
   closure work is warranted.
5. Post the finding and acquisition result to issue #1455 when network access
   is available.

## Tests

The broader CPU/x64 lane battery reports `240 passed, 5 deselected` in
561.65 seconds; the five exclusions are inherited slow controls.  The focused
writer/admission subset is 13 of those passes, with 6 acquisition tests.  The
source guard explicitly applies its condition to the incoming pre-fix blob and
proves that exact reversion fails.  The clean-tree NEMO preflight ends in
`PREFLIGHT_OK`.  The citation gate passes both mapped citations with no
unmapped or failing entries; shifting the first citation by two lines produces
the expected nonzero exit and `SYMBOL-NOT-AT-LINE` refusal.

## Compiled-source citations

The refused binary declared both its production vertical-solve work arrays and
the no-stress companion on the 122-column `ntsi:ntei` inner interval at
`GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/dynzdf.f90:100-105`.  The recorder
accepted that companion as an assumed-shape whole array, then assigned its
vertical and stress differences into full-native rows at
`GYRE_OMIP_L2_P3_R14BUD/BLD/ppsrc/nemo/gyre_r14_budget.f90:153-163`.  Those
different extents are the measured writer defect.

The corrected tracked `dynzdf_round14.patch` keeps the production work arrays
unchanged but declares the recorder-only companion with the full `jpi` extent.
Its forward and backward solve still uses NEMO's already-formed production
matrix; only the recorder array's horizontal index alignment changes.
