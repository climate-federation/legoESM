# ORCA2 round 235 preregistration — OMT-4 external-substep discrimination

Date: 2026-10-10. Frozen base: `b4d6c1ea0`. Scope: measurement only. This
round changes no model file, card, deck, carried state, selector, stabiliser,
sea-ice field, or `unmeasured_features` tuple. Decision 114 remains pending and
is not acted on.

## Frozen question

Round 234 proved NEMO's seven-array post-substep association bit-exact locally
but it left the complete round-217 unit's stage-1 `vn_adv` unequal on all 8,589
live rank-0 V faces. Determine whether the first non-rounding internal debt is
fold-local or is already present in a global split-explicit operand.

NEMO copies the stage forcing at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-294`, subtracts
the entry Coriolis trend at `:323-328`, constructs the midpoint transports at
`:512-539`, advances SSH at `:549-561`, accumulates `un_adv/vn_adv` at
`:566-575`, updates vector-form velocity at `:669-682`, and associates the
seven post-step arrays at `:738-756`. That is the frozen source order.

## Existing records and reused instrument

No NEMO acquisition is authorised or expected. The oracle is the admitted
OMT-4 twin under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2/`:
`oracle_bt_substeps_kt00000001.bin`, `oracle_bt_ordered_operands_kt00000001.bin`,
`oracle_bt_drag_operands_kt00000001.bin`,
`oracle_bt_advmean_operands_kt00000001.bin`, and
`oracle_bt_frames_kt00000001.bin`, with its admitted twin used for defined-byte
identity. The candidate-side observer reuses round 146's passive named trace
and round 205's 65-substep source table. The ordinary and traced completed
states must be array-identical before any row is read.

Both labels are measured separately: **independent** and **given NEMO's
entry**. CPU, production JIT, fp64/libm only. The comparison floor is
`2e-10`; bit equality is separately recorded. For each substep and operand,
report unequal count, maximum absolute difference, argmax, and maximum over
the interior rows `j <= jpj-4` separately from the final three-row fold band.

## Frozen predictions and falsifiers

- **R235-P1 — record and observer.** CONFIRM if every required inherited
  stream parses, the two twins have exact defined payloads, and every ordinary
  completed state array is bit-identical to its traced counterpart for both
  labels. REFUTE on any missing/malformed stream, twin payload difference, or
  observer bit movement. A planted observer ULP and a reordered source registry
  must each refuse.
- **R235-P2 — owner class.** Prediction: class **GLOBAL**. At substep 1, a
  source-ordered external-mode operand has an interior-row maximum above
  `2e-10`; the first such operand is expected no later than the completed slow
  forcing (`zu_frc/zv_frc`). CONFIRM GLOBAL if that signature appears under
  both labels. REFUTE it, and classify **FOLD-LOCAL**, if every interior maximum
  remains at or below `2e-10` through the first fold-band debt. Classify
  **RECORD_GAP** if the first candidate boundary has no matching oracle field;
  name the exact missing stream instead of inferring an owner.
- **R235-P3 — source order.** The first numerical debt is the earliest row in
  the frozen order whose absolute difference exceeds `2e-10`; signed-zero-only
  rows remain separately registered and cannot own the numerical walk.
  REFUTE if changing source order changes the selected first row or if the
  chosen row is not present in the self-described oracle registry.
- **R235-P4 — label agreement.** Prediction: independent and given-entry labels
  name the same first substep, operand, and owner class. REFUTE if any of those
  three differ; retain both results and claim no common owner.
- **R235-P5 — landing predicate.** No landing is attempted in this measurement
  round. A later statement arm is eligible only after the table names a cited
  statement and its one-variable replay moves the registered boundary under
  Decision 96. Package diffs at round end must be empty.

## Outcomes

- `HELD`: table is complete and names GLOBAL, FOLD-LOCAL, or a precise
  RECORD_GAP; no package change.
- `STOPPED_FOR_RECORD`: an exact required oracle operand is absent; write a
  fail-closed acquisition only after naming that operand.
- `FAILED`: passivity, twin identity, source order, label coverage, or a plant
  fails.

ASKED choices: Decisions 103, 109, 113, standing Decision 96. Decision 114 is
pending with the user and untouched. UNASKED choices: empty.
