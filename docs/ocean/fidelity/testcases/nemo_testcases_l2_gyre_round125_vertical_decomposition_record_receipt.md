# NEMO testcase L2 GYRE phase 3 — round 125 vertical-decomposition record receipt

Date: 2026-09-20

Incoming tip: `de1b14749a9d39eda7e8555761956ab244ec283e`

Status: **STOPPED_FOR_RECORD — the admitted magnitude result reproduces
exactly, but every existing `tra_zdf` internal record ends at step 2, so no
day-240 vertical sub-owner can be named without a new passive NEMO record.**

Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round125/`

## Outcome first

Round 125 reproduced the Round-124 magnitude owner without moving it:

| measured row | reproduced value |
|---|---:|
| day-240 wet-T3D RMS | `1.6446741930292448e-2 K` |
| vertical-diffusion signed carry | `+2.4168271578053416e-2 K` |
| strongest vertical birth interval | days 190--200, `+5.6288525578725495e-3 K` |
| endpoint reconstruction residual | `3.552713678800501e-15 K` |
| signed carry sum minus endpoint | `-3.469446951953614e-18 K` |

The complete filesystem inventory then found 66 existing
`oracle_trazdf_matrix` files.  The corrected files are all step 1 or step 2;
the sole older malformed file is also step 1.  None is from steps 1081--1440.
The admitted step-1080 and step-1440 restarts contain endpoint `en`, `avt_k`
and `avm_k`, but no per-step matrix, LU diagonal, right-hand side, forward
sweep or solved-column boundary.  The inventory is preserved in
`round125/existing_record_inventory.log`.  Running the new admission mode
against the absent target exits 1 with:

```text
GATE FAILED: vertical-record set is not exactly steps 1, 2 and 1081..1440
EXPECTED_STOPPED_FOR_RECORD_EXIT=1
```

This confirms the preregistered insufficiency prediction.  An endpoint
profile, early-step matrix or legoESM-only replay cannot attribute a
day-180-to-day-240 independent-trajectory carry.  Accordingly this round
does **not** name a first non-bit vertical statement, does not build a physics
candidate and does not run the Decision-43 ladder/month or Decision-45 year
landing gates.

The round commits one acquisition unit only.  It reuses the established
self-describing writer and reader, extends the writer additively over steps
1081--1440 in a new NEMO target, and admits every interval step against the
already stamped Round-123 process trajectory.  The agent did not run
`makenemo`, `mpirun` or NEMO.

## Frozen preregistration ledger

The preregistration is
`docs/ocean/fidelity/PREREG_nemo_testcases_l2_gyre_round125.md`, committed
before the first Round-125 measurement as full commit
`7f32088d626a324a63f179f14229d3ee4688255b`.

- **P0 CONFIRMED.** Every frozen Round-124 magnitude, birth and closure value
  reproduced exactly in `round125/baseline_process_budget.json`.
- **P1 CONFIRMED.** The full record inventory and both restart headers show
  that no admitted artifact contains the requested internal boundaries for
  all 360 interval steps.  The normal admission mode refuses the absent
  record rather than substituting an early record.
- **P2 PREPARED, NOT MEASURED.** The new target is
  `GYRE_OMIP_L2_P3_SM_R125ZDFMAG`; the requested run root is
  `round125/oracle_vertical_decomposition`.  The additive source patch and
  sole `nn_itend=2160 -> 1440` namelist replacement apply with zero fuzz, and
  the patched source passes `gfortran -fsyntax-only`.  Passivity remains an
  acquisition condition, not a claimed result.
- **P3 PARTLY CONFIRMED / RECORD-BLOCKED.** On the admitted step-1 record, the
  rebuilt mixing profile, three matrix coefficients, LU diagonal, RHS,
  forward sweep and temperature solution are each bit-exact.  The
  consumed-diagonal plant moves exactly one registered coefficient, the
  truncation plant is rejected, the trajectory control moves exactly one
  cell, and the wrong-stamp helper rejects the wrong producer.  The four
  persisted end-to-end plant reports remain blocked on the requested record
  and must run during acquisition.
- **P4 CONFIRMED.** Only diagnostic acquisition/admission code, tests,
  citations, preregistration and this receipt land.  No production physics,
  card default, restart schema or carried state changes.

No preregistered prediction was refuted.

## Compiled-source record

The cited source is the compiled branch of the admitted
`GYRE_OMIP_L2_P3_SM_YRPERT` year binary.  Its active instrument arm enables
the record through `kt = nit000 + 1`, allocates the matrix/recurrence buffers,
and captures the incoming accumulator at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:132-146`.  This is why
all admitted year records stop at steps 1--2.

The compiled stream writer, including its header, branch flags, all scalar
and array payloads, and close, occupies
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:160-335`.  The exact
entry, matrix, solve, diffusivity and free-surface fields used by this round
are written at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:213-333`.

The temperature path assembles `zwt` from the selected diffusivity family at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:414-450`, constructs the
tridiagonal coefficients from the live geometry and free-surface slots at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:462-476`, and forms the
LU diagonal at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:523-527`.  The content
RHS is formed at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:546-560`; the forward
and backward solve completes at
`GYRE_OMIP_L2_P3_SM_YRPERT/BLD/ppsrc/nemo/trazdf.f90:563-578`.

These statements establish the five measurable boundaries.  They do not
establish which boundary owns the independent month-scale error; that claim
remains blocked on the live interval record.

## Acquisition and admission contract

The operator-run card is:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_round125_vertical_decomposition/run.sh`

It creates the new target `GYRE_OMIP_L2_P3_SM_R125ZDFMAG` from
`GYRE_PISCES`, copies the source card file by file, applies only the additive
writer patch and the preregistered end-step replacement, syntax-checks the
exact patched source, builds, and runs the independent NEMO trajectory.  The
script prints a named `REFUSE` line on every unexpected nonzero exit.  It
refuses an existing target/run root and never overwrites a prior acquisition.

The compiled writer has 11 binary64 scalars, 27 full-`jpk` arrays, three
`jpkm1` arrays and three horizontal arrays behind 44 self-describing headers.
For GYRE's `36 x 26 x 31` domain:

```text
80 + 44*32 + 11*8
   + 27*36*26*31*8
   +  3*36*26*30*8
   +  3*36*26*8 = 6,965,416 bytes/frame
```

The exact file set is steps 1, 2 and 1081--1440: 362 files,
`2,521,480,592` bytes total, of which the 360 scored interval files occupy
`2,507,549,760` bytes.  The admission reader additionally requires the exact
compiled field order, finite payloads, fixed branch flags, a committed
producer/manifest stamp and exact EOF.

Passivity is fail-closed twice.  The target's step-1080 and step-1440 restart
SHA-256 values must remain respectively
`6c0c7a950b30b9d59dbf2673833ddf462a5f8ea5650f496f2f772e1e17092976`
and `96529a98da0e0d89b328632a826a9d41593f81f0d1a917350f28f184d49b163a`,
and each must be byte-identical to the admitted year run.  At all 360 steps,
recorded `T_Kbb_in`, `T_Krhs_in`, `sol_T_post_clamp` and all three `r3t`
slots must be bit-identical to the independently stamped Round-123 process
boundaries.  Consecutive solved columns must chain bit-for-bit.

The normal gate reconstructs `zwt_mix`, `zwi`, `zwd`, `zws`, `zwt_lu`,
`rhs_T`, `fwd_T` and `sol_T` with the compiled association and requires zero
unequal bits.  Its wrong-stamp, truncation, consumed-matrix-ULP and
trajectory-ULP plants must each print `STATUS PLANT-FIRED` and exit 1 before
the script prints `ROUND125_VERTICAL_RECORD_READY`.

The source patch's independent compile proof is
`round125/gfortran_syntax_proof.log`:

```text
SYNTAX_PROOF_PASS trazdf.f90
```

## Landing and cross-card scope

There is no physics landing and therefore no moved production row to
register.  The current admitted headlines remain:

- kt2 T/S: `1.4210854715202004e-14 K` /
  `2.1316282072803006e-14 psu`;
- kt2 U/V: `2.7377110452773967e-12` /
  `3.2849219221489645e-12 m s-1`;
- kt3 T/S: `8.600419718618468e-7 K` /
  `6.979441735666114e-8 psu`;
- day-30 T3D RMS: `6.890484901489568e-5 K`;
- day-240 T3D RMS: `1.6446741930292448e-2 K`; and
- day-360 T3D RMS: `1.1223573910167267e-2 K`.

The committed source patch exists only in an unbuilt, new NEMO acquisition
target; no legoESM production statement executes it.  GYRE production,
generic NEMO-GYRE, DINO, LOCK_EXCHANGE and OVERFLOW therefore do not move.
ORCA2 remains **UNMEASURED-WITH-SPEC**: repeat the same native-card interval
record, independent process trajectory, matrix/solve reconstruction,
passivity controls and day-240 endpoint projection before making an ORCA2
vertical sub-owner claim.

No configuration or carried-state choice is exposed.  `DECISION_NEEDED` is
`NONE`.

## Independent adversarial review

The required `codex exec --sandbox read-only` command was run against the
complete incoming-tip-to-citation-map diff.  Its verbatim terminal result was:

```text
WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)
Reading additional input from stdin...
Error: failed to initialize in-process app-server client: Read-only file system (os error 30)
```

The command exited 1.  Independent review unavailable in-sandbox.  It emitted
neither `SHIP` nor `DO NOT SHIP`; no verdict is fabricated.

## Verification

The owner self-check prints `self-check: all checks passed`, including the
frozen record geometry and trajectory-ULP control.  The focused owner test
reported `18 passed, 1 skipped in 5.16s`; the skip is the deliberately absent
end-to-end Round-125 record.  The compiled-source citation test reported
`16 passed in 1.88s`.  `bash -n`, Python byte compilation and `git diff
--check` also pass.

The final clean-tree focused owner, citation and worktree-stamp suite reported
`45 passed in 33.72s`; its JUnit is `round125/focused_tests.xml`.  The receipt
citation gate found eight citations, zero unmapped citations, zero citation
failures and zero map-audit failures in `round125/citation_gate.json`.
Shifting the compiled writer-arm citation by two lines produced
`SYMBOL-NOT-AT-LINE`, printed no PASS verdict and exited 1; the report and log
are `round125/citation_gate_shift_plant.json` and
`round125/citation_gate_shift_plant.log`.

## OPEN — round 126

Do not resume a last-bit walk and do not select a physics candidate before
the requested record is admitted.

1. The operator runs the absolute `ACQUISITION_NEEDED` script reported below.
   A usable run ends with `ROUND125_VERTICAL_RECORD_READY` and `RUN_DONE`.
   If it refuses, diagnose the preserved target/run and request a **new**
   target only when the source card itself must change; never overwrite or
   silently rerun this target.
2. Admit all 362 files, both passive restart hashes, the per-step Round-123
   trajectory alignment, all eight compiled-arithmetic calibration rows and
   all four persisted plants.  A single unequal bit keeps the record refused.
3. Preregister, then add the matching write-only legoESM production-step
   trace for the same 360 steps.  Reproduce each model's recorded matrix and
   solve before projecting any hybrid boundary.
4. Rank, by signed day-240 endpoint carry, the live entry/content,
   free-surface weighting, vertical-diffusivity family, matrix construction
   and solve/association boundaries.  Keep any interaction/order residual as
   its own row; do not distribute it among preferred owners.
5. The mechanically largest admitted vertical sub-owner alone becomes the
   next implementation candidate.  If it requires a configuration or
   carried-state choice, stop with one explicit `DECISION_NEEDED`; otherwise
   its eventual landing still requires the Decision-43 ladder/month and
   Decision-45 year gates.

The vertical bucket remains first by magnitude.  Lateral diffusion and
advection remain cancellation context, not simultaneous candidates.
