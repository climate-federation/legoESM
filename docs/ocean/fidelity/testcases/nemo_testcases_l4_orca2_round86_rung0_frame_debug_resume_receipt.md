# ORCA2 round 86 — rung-0 frame debug preflight repair

Base: `d67cbfd9ea8371461096ae057a842aeec371b996`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_round86.md`.  Claim label:
**independent**.  No package file, hierarchy assignment, shipped ORCA2 card,
NEMO source, CPP key, carried state, threshold, stabilizer, or sea-ice selector
changed.

## Verdict

**STOPPED_FOR_RECORD.**  The operator's round-85 action created the exact debug
target but stopped before compilation.  Its architecture contains one real
`%FCFLAGS` assignment and a second textual occurrence where `%FFLAGS`
references it.  The old unanchored literal count returned two and falsely
refused the valid generated architecture.  No debug binary, run directory, or
trajectory exists, so round 85's predicted crash location remains
**UNMEASURED**.

The committed continuation now accepts only that exact intermediate, proves
the parser with duplicate-assignment and reference-only plants, and is ready
for the operator.  Its preflight prints:

`ORCA2_ROUND86_RUNG0_FRAME_DEBUG_RESUME_PREFLIGHT_READY ... assignments=1 literals=2`

No rung-0 card, ladder row, model statement, or month score lands.

## Fail-closed continuation

The continuation pins the inherited architecture, build configuration, CPP
keys and histories, original and patched driver, frame module, admitted rung-0
binary and namelist.  It reconstructs the committed round-84 additions-only
patch with zero fuzz and requires byte identity with the inherited driver.  It
also inventories all `MY_SRC` entries and requires every unchanged file to be
byte-identical to the pinned rung-0 source.

The corrected parser anchors on `%FCFLAGS` as the first token.  Its first plant
adds a second assignment and must count two.  Its second removes the assignment
but retains the `%FFLAGS` reference and must count zero.  The real inherited
file must report one assignment and two literal occurrences.  The old broad
count no longer controls a build.

On `--run`, the continuation invokes `fcm build` on the existing target rather
than regenerating it.  It requires the resolved debug flags, four frame calls,
the frame magic, and absence of vector-math symbols before launching two MPI
ranks.  Debug output goes to a new round-86 directory and remains
diagnostic-only.  The compiled order still writes the new stage-0 frame before
surface-boundary work (`stprk3.f90:91-107`) and reaches the inherited surface
writer before stage 1 (`stprk3.f90:147-154`); the source-resolved failure line,
not the optimized symbol trace, will choose any repair.

## Prediction ledger

| frozen prediction | result |
|---|---|
| anchored assignment count is one while literal count is two | **CONFIRMED — preflight reports 1/2** |
| exact inherited intermediate is safely resumable | **CONFIRMED at preflight — all hashes, reconstruction and inventory pass** |
| duplicate and reference-only parser plants fire | **CONFIRMED — preflight refuses unless both report 2 and 0** |
| resumed build has debug flags, four frame calls and scalar math | **UNMEASURED — operator acquisition required** |
| debug run fails before stage 1 in the inherited surface canonicalizer | **UNMEASURED — no binary or run exists** |
| debug output is never admitted as scientific evidence | **CONFIRMED — continuation has no record-admission path** |

## Gates, review, and tests

The continuation passes `bash -n` and its full clean-tree preflight.  The two
parser plants execute inside that preflight.  The focused frame and citation
tests, campaign citation gate, planted citation shift, and prescribed ocean
fidelity battery are recorded in the final round commit.

The required `codex exec --sandbox read-only` review failed before reading the
diff: `failed to initialize in-process app-server client: Read-only file
system`.  Verdict: **independent review unavailable in-sandbox**.

No `packages/` file changed.  GYRE, DINO, tank cards, the shipped ORCA2 card,
and its `unmeasured_features` sea-ice tuple are unchanged by construction.
No unasked configuration choice was made.

## OPEN

1. The operator runs the round-86 continuation and returns its source-resolved
   backtrace.
2. Repair only the named instrumentation statement, then issue a fresh
   optimized target under the 20-restart / 80-frame admission gate.
3. Once the frames admit, name the true rung-0 entry and the first non-bit
   boundary in compiled stage order.
4. The rung-0 card, both ten-step ladders and independent month follow only
   after that statement is discharged.  Rung 1 remains untouched.

Repository search found and reused the round-84 writer, additions-only patch,
self-describing reader, corruption plants, and round-85 debug target.  No
second writer, parser, scientific deck, or model implementation was added.

ASKED: repair the false build preflight and safely resume the exact round-85
debug target.

UNASKED: regenerate a different target, guess a crash repair, admit debug
values, change physics or hierarchy configuration, change the shipped card or
its ice tuple, or relax any gate.
