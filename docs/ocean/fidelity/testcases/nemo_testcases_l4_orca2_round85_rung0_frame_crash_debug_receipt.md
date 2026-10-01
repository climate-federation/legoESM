# ORCA2 round 85 — rung-0 frame crash debug handoff

Base: `ac0205fe270d87613baf53697fa9bb29376626cc`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round85.md`.  Claim label:
**independent**.  No package file, hierarchy assignment, shipped ORCA2 card,
NEMO source, CPP key, carried state, threshold, stabilizer, or sea-ice selector
changed.

## Verdict

**STOPPED_FOR_RECORD.**  The round-84 optimized frame build wrote one complete
stage-0 frame on each MPI rank and then crashed before stage 1.  Both files
parse to physical EOF under the committed self-describing reader: `kt=1`,
`stage=0`, `level=1`, shape 94 x 152 x 31, 64-bit payloads, five finite fields,
and exactly zero u, v, and ssh.  Their SHA-256 digests and field census are in
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round85/inherited_stage0_parse.json`.

Those two valid files do not admit the record.  Seventy-eight of the required
80 rank/step/stage frames are absent, and there are no terminal restarts with
which to prove that the instrument is observational.  No rung-0 card or model
physics statement lands in this round.

## Crash localization

The exact compiled optimized branch writes the inherited lane-1 entry stream
and then calls the new stage-0 writer
(`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:91-107`).  It next executes
the surface boundary condition and calls the inherited surface-input writer
(`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:146-154`).  That writer
canonicalizes every surface field
(`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:390-441`) through the local
wet-cell copy loop
(`ORCA2_OMIP_L4_R84FRAMES/BLD/ppsrc/nemo/stprk3.f90:443-473`).

Post-hoc triage of the unsymbolized O3 backtrace is deliberately not treated
as the decisive measurement.  Using the `_start` frame to recover the PIE load
base, `addr2line` maps the first three NEMO frames to
`l4_canon_2d`, `l4_dump_ocean_surface_input`, and `stp_RK3`; the O3 binary has
no source line table.  This supports, but does not confirm, the preregistered
prediction that the crash is in inherited accumulated instrumentation rather
than `r84_dump_frame`.  The admitted round-83 binary contains neither those
surface-writer symbols nor that stream, so its successful trajectory does not
certify the inherited writer copied into the round-84 build.

## Debug acquisition issued

The committed launcher reproduces the exact round-84 source, writer module,
additions-only patch, rung-0 deck, two-rank layout, and scalar-math toolchain in
the fresh `ORCA2_OMIP_L4_R85FRAMEDEBUG` target.  It changes only compiler
diagnostics to `-O0 -g -fbacktrace -fcheck=bounds`, requires the resolved flags
and all four frame calls in preprocessed source, rejects vector-math symbols,
and requires a nonzero run with a source-resolved instrument line.  Debug
output is explicitly non-admissible and cannot be used for a scientific
number.  Preflight passes and prints
`ORCA2_ROUND85_RUNG0_FRAME_DEBUG_PREFLIGHT_READY`.

The operator must run:

`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round85_rung0_frame_debug/run.sh --run`

Only that source line chooses the repair.  A subsequent production acquisition
must use a new target, reproduce the admitted optimized scalar-math toolchain,
write all 80 frames, fire the five round-84 corruption plants, and compare all
20 rank/step terminal restarts byte-for-byte with the admitted round-83 record.

## Prediction ledger

| frozen prediction | result |
|---|---|
| the exact round-84 patch on a debug build reproduces the failure | **UNMEASURED — operator acquisition required** |
| the first source-resolved frame is the inherited `l4_canon_2d` assignment reached by `l4_dump_ocean_surface_input` before stage 1 | **UNMEASURED — O3 symbol names are supportive post-hoc triage only** |
| both inherited stage-0 files parse completely and are finite | **CONFIRMED — two ranks, five fields, physical EOF** |
| debug output is never admissible | **CONFIRMED — launcher has no admission path and stamps it as debug** |
| a repaired optimized build preserves all 20 terminal restarts and emits 80 frames | **UNMEASURED — repair is forbidden before the debug line exists** |

## Gates, review, and tests

The debug launcher passes its clean-tree preflight.  The reused round-84 frame
gate passes its focused tests **2/2**, including its corruption controls.  The
campaign citation gate passes the default cumulative receipt and this receipt
with no unmapped or failing citations; its shifted-citation plant fires.

The prescribed `tests/ocean/fidelity -n 12` battery selected 2,209 tests,
reached 99%, and repeated the established no-summary stall.  The three visible
failures were the known missing case-board row, SI3 scalar-math provenance,
and round-51 private-trace registry reds.  No round-85 test failed.  The run was
stopped only after repeated silent waits at 99%; it was not rerun.

The required `codex exec --sandbox read-only` review failed before reading the
diff: `failed to initialize in-process app-server client: Read-only file
system`.  Verdict: **independent review unavailable in-sandbox**.

No `packages/` file changed.  GYRE, DINO, tank cards, the shipped ORCA2 card,
and its `unmeasured_features` sea-ice tuple are unchanged by construction.

## OPEN

1. The operator runs the round-85 debug launcher and returns its source-resolved
   backtrace.
2. Repair only the named instrumentation statement, then issue a fresh
   optimized production target under the 20-restart / 80-frame admission gate.
3. Once the frames admit, name the true rung-0 entry and the first non-bit
   boundary in compiled stage order.
4. The rung-0 card, given-entry and independent ladders, and independent month
   follow only after that statement is discharged.  Rung 1 remains untouched.

Repository search found the existing self-describing round-84 module, patch,
reader, corruption plants, and compiled-build template; all are reused.  The
only new acquisition artifact is the fail-closed debug launcher.

ASKED: diagnose the failed rung-0 frame acquisition with a source-resolved
debug reproduction before repairing it.

UNASKED: guessing a repair from an optimized symbol name, admitting debug
numbers, changing physics or hierarchy configuration, changing the shipped
card or its ice tuple, or relaxing any gate.
