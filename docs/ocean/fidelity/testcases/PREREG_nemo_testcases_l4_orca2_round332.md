# ORCA2 round 332 preregistration — full re-score and first non-bit source-order walk

Frozen 2026-10-24 before running the current-tip ladders or any new replay.
Base: `f2dcf0a8cfff`.  This round changes no configuration value and Codex will
not run NEMO.

## Measurements

- Re-run all 200 `(kt, checkpoint, field)` rows for both rung-10 labels and
  for rung 0, OMT-0, OMT-2, OMT-3, and OMT-4.  Re-run all 160 available rows
  for OMT-1.  Classify a row as bit-exact when every binary64 word is equal;
  otherwise as at-bar when
  `max_abs / max(max(abs(oracle)), 1) <= 1e-15`, and over-bar otherwise.
  Publish every row's maximum, RMS, unequal count, normalization, and class.
- Name the first non-bit row independently for each rung-10 label.  A row
  which differs only in the sign bit of zero remains non-bit and at-bar.
- Replay that first row in the executing NEMO statement order from the
  admitted round-5/326/328/331 records.  A statement is CONFIRMED only when
  the record contains all operands needed to reproduce its output and a
  one-statement replay accounts for every unequal word without moving any
  nonzero value.  Otherwise it remains PLAUSIBLE, or the round stops
  ACQUISITION_NEEDED when a discriminating NEMO operand is absent.
- Replay the seven residual post-TKE `avm` and `avt` cells from the exact
  admitted returned energy.  Distinguish literal NEMO operation order from a
  legoESM transcription difference only when the mixing-length, floors,
  masks, and every multiply operand are record-proven at that boundary.

## Predictions and falsifiers

- **R332-P1 — rung-10 labels.**  The two labels produce identical row metrics.
  Their first non-bit row is kt 1, stage 1, `v`: 2,032 unequal words, all
  signed-zero differences, with maximum and RMS both exactly zero.  Any label
  disagreement, earlier non-bit row, nonzero difference, or unequal-count
  change refutes this prediction.
- **R332-P2 — census.**  The complete campaign table has 1,560 rows: 200 for
  each of seven pair/label trajectories and 160 for OMT-1.  Counts of
  bit-exact, at-bar-not-exact, and over-bar sum to 1,560.  Missing rows or a
  different sum refute this prediction.
- **R332-P3 — first non-bit owner.**  At the from-rest row, NEMO's executing
  EEN vorticity statement adds an exactly positive-zero `zva` to the shared
  `pv_rhs` accumulator.  IEEE `-0 + +0` canonicalizes the result to `+0`;
  legoESM's combined/vectorized accumulation retains `-0` at the same fold
  cells.  Confirmation requires an admitted operand boundary showing the
  accumulator and `zva`, or an equivalent admitted before/after boundary,
  and literal replay healing all 2,032 words.  A missing boundary triggers
  acquisition rather than a physics proposal.
- **R332-P4 — seven TKE cells.**  The seven residual `avm/avt` words are
  expected to be explained by a binary64 boundary in NEMO's literal sequence
  `zsqen = sqrt(en); zav = rn_ediff * zmxlm * zsqen; max(floor, zav); mask`,
  rather than by the already exact returned energy.  This is CONFIRMED only
  if all inputs are admitted and literal replay equals NEMO in all seven
  cells; lacking recorded `zmxlm` makes the explanation merely PLAUSIBLE and
  requires acquisition if it is the first unresolved owner.
- **R332-P5 — eligibility.**  No production statement changes unless one
  source-cited statement owns the first departure and the candidate is
  bit-identical or strictly better under Decision 122 on all seven required
  pairs and both rung-10 labels.  An eligible change additionally needs a
  production-call-site revert-failing test, every caller card measured, and
  finite rung-10 N=1 gradients with reverse equal to forward.  Otherwise the
  round stops with the measured proposal or ACQUISITION_NEEDED.

