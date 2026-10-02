# ORCA2 round 105 — EEN accumulator recorder repair

Date: 2026-10-02. Base `fe48ccb3197d9bfcd027ede0edbf62a2d66ee5b7`.
Scope is ocean-only instrumentation. No numerical ORCA2 result is admitted in
this round. Every future rung-0 number from this acquisition is labelled
**independent**.

## Verdict

**STOPPED_FOR_RECORD.** The round-104 operator run wrote one complete kt=1
operand file on each rank, then the recorder retried those same filenames at
kt=2 and NEMO aborted. A fresh-target repair opens one absolute, rank-tagged
file during initialization, writes and closes it on the first EEN coefficient
construction, and makes every later construction a no-op. The source patch is
additions-only, compiles against the pinned producer, and all four pre-run
controls fire. The launcher is ready for the operator; it was not run here.

No `packages/` file, model physics, card field, configuration value, threshold,
stabilizer, carried state, sea-ice selector, or ORCA2 `unmeasured_features`
entry changes. GYRE, DINO, tanks, rung 0, and rung 7 are unchanged by
construction.

## Source-cited diagnosis

The split-explicit configuration calls its initialization routine in the
compiled dispatcher
(`ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg.f90:300-303`). During the
time step, the compiled rung-0 branch rebuilds the Coriolis coefficients at the
first step and again whenever the free surface is nonlinear
(`ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:302`). The round-104
dump is inside that repeatedly called builder
(`ORCA2_OMIP_L4_R104EENACC/BLD/ppsrc/nemo/dynspg_ts.f90:1294-1297`).

The failed writer used the constant start step `nit000` in every filename and
opened it with `STATUS='NEW'` (`l4_r104_een_accum.F90:23-27`). The two complete
files on disk and the rank logs show a successful kt=1 dump followed by the
kt=2 `cannot open` refusal. Therefore the abort is the recorder's duplicate
open, not an ocean statement.

## Repair and controls

The new target is `ORCA2_OMIP_L4_R105EENACC`; its output directory is
`round105/acquisition/orca2_rung0_een_accum_repair_10step_np2`. The launcher
creates that directory before MPI, exports its absolute path, and the writer
opens one `STATUS='NEW'` stream per rank from initialization. The first dump
writes the unchanged round-104 self-describing schema; later calls return.

Search-before-build: repository and oracle acquisition sources were searched
for an existing environment-directed, initialization-opened recorder. No ORCA
recorder had that lifecycle. The repair reuses the committed round-104 binary
schema and admission checker rather than creating another parser.

The clean-tree preflight reports:

```text
SYNTAX_PROOF_PASS l4_r105_een_accum.f90 dynspg_ts.f90
ORCA2_ROUND105_EEN_ACCUM_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round105/acquisition/orca2_rung0_een_accum_repair_10step_np2
```

The layout, absolute-path, duplicate-write, and producer-content plants each
exit 69 with `STATUS PLANT-FIRED`. After the operator run, the existing
self-describing checker must additionally fire all eight payload/restart
plants, reconstruct every final coefficient as `scale * accumulator` bitwise,
prove exactly-once global rank coverage, and compare all 20 terminal restart
shards byte-for-byte with the admitted round-98 run.

## Preregistered result status

- R105-P1 through P4 are **UNMEASURED** until NEMO runs. Preflight proves the
  source and launcher shape, not runtime success or observational identity.
- R105-P5, the prediction that accumulation owns the first remaining non-bit
  operand, is **UNMEASURED** until the record admits.
- No coefficient, Coriolis consumer row, ladder row, or month metric is quoted.

## Validation and review

- Fortran module compilation and patched `dynspg_ts` syntax proof: PASS.
- Focused recorder/admission tests: 3 passed.
- An initial citation-test invocation while the citation map was intentionally
  uncommitted produced the expected two worktree-stamp refusals; it is not a
  product failure. The clean committed rerun is recorded below.
- Separate read-only Codex claim and diff reviews were attempted with both the
  ordinary and ephemeral CLI modes. Verdict: **independent review unavailable
  in-sandbox** (`failed to initialize in-process app-server client: Read-only
  file system`). No other independent reviewer is callable in this sandbox.

## OPEN

1. Operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round105_een_accum_acquisition/run.sh`.
2. Admit the two-rank record and all 20 restart comparisons; measure R105-P1
   through P5.
3. Compare every first unequal accumulator/scale bit and land only the first
   compiled NEMO statement that passes the full ORCA2 and shared-card gates.
4. Then resume the independent northern-fold 66/67 magnitude debt and the
   later 68-cell substep-2 U residual.
5. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## UNVERIFIED

- Runtime success, one initialization/dump per rank, and restart identity.
- The exact accumulator or scale statement producing the remaining zero signs.
- The repaired record's contents until operator acquisition and admission.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.

