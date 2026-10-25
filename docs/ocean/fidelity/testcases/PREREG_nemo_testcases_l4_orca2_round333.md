# ORCA2 round 333 preregistration — first wet-value departure

Frozen 2026-10-25 before running a round-333 ladder or offline source replay.
Base: `072e4aa1b1c8`.  This round changes no configuration value and Codex will
not run NEMO.  Signed-zero-only dry-cell rows are outside this round by user
instruction.

## Measurements

- For each rung-10 label, walk rows in campaign order `(kt, entry/stage1/
  stage2/stage3, T/S/u/v/ssh)` and name (a) the first row with a nonzero
  difference on the field's NEMO wet support and (b) the first `OVER_BAR` row.
  For each named row publish maximum absolute difference, wet-support RMS,
  unequal wet-cell count, and the global zero-based `(i,j,k,rank)` of the
  maximum.  The two row identities are determined independently even if they
  coincide.
- Classify the maximum location with mechanical flags: cyclic seam means
  `i in {0, nx-1}`; north fold means the northern T-pivot/image rows used by
  the field's NEMO grid-point nature; coast means a wet cell has a dry
  horizontal neighbour on its native mask; under ice means the admitted SI3
  `fr_i` at that column is positive at the step; open ocean means none of
  seam/fold/coast/under-ice applies.  Report every true flag rather than
  forcing overlapping categories into one label.
- Replay the earlier of the first-wet and first-over-bar boundaries in the
  executing NEMO source order, offline from the admitted round-5, 326, 328,
  331 and 332 records.  A first owning statement is **CONFIRMED** only if all
  operands at its executed program point are admitted, the replay reproduces
  NEMO, and substituting that one statement accounts for the complete earlier
  boundary without moving any preceding boundary.  Otherwise it is
  **PLAUSIBLE**; a missing discriminating NEMO operand triggers a reviewed
  acquisition extension and `ACQUISITION_NEEDED`.

## Predictions and falsifiers

- **R333-P1 — row identity.**  Round 332's published full ladder puts both
  labels' first nonzero and first over-bar row at kt 2 stage 3 `T`, with
  full-row maximum `7.678080393702658e-12 K`, RMS
  `1.2687664596813964e-14 K`, and 18 unequal words.  Round 333 predicts all
  18 are wet, so the wet metrics are identical.  Either label selecting a
  different row, any dry-only unequal word, or any changed metric refutes P1.
- **R333-P2 — labels.**  The given-entry and independent labels are row-for-row
  identical after Decision 157; their maximum index, rank, wet census and
  location flags are therefore identical.  Any label difference refutes P2.
- **R333-P3 — geography.**  The published first unequal index is on global
  `j=147`, so every maximum is predicted to be on the north fold, on rank 0,
  and off the cyclic seam.  The coast and admitted-ice flags are measured,
  not inferred.  A maximum off the fold, on rank 1, or at a seam column
  refutes P3.
- **R333-P4 — ownership.**  Because kt 2 stage-2 `T` is at the bar while the
  kt 2 stage-3 `T` row is over bar, the first owner is predicted to be in the
  executed stage-3 tracer program after the admitted stage-2 state, not in
  the earlier signed-zero V row.  Confirmation requires a record-proven
  source-order boundary.  If the admitted records skip across the first
  unequal stage-3 statement, the result is `ACQUISITION_NEEDED`, not an
  inferred owner.
- **R333-P5 — landing.**  A production edit is eligible only when one cited
  statement is confirmed and its one-statement fix is bit-identical or
  strictly better under Decision 122/139 on all seven pairs and both rung-10
  labels.  It additionally needs a revert-failing production-call-site test,
  measurements on every other executing card (faithful DINO probe/month and
  grid-ladder lanes), and finite rung-10 `N=1` gradients with reverse equal to
  forward.  Otherwise this round stops with the measured proposal or
  `ACQUISITION_NEEDED`.

## Resume 8 preregistration — post-LDF boundary and stored-divisor reverse pass

Frozen 2026-10-25 at `aa31c5c6bac1` before extending or running either replay.
No configuration value changes and no NEMO execution is authorized.

- **R333-R8-P1 — the 18 temperature words.**  The admitted inner-LDF replay
  says NEMO's pre-add accumulator, scaled tendency, and post-add accumulator
  are exact at all 18 registered words.  The outer replay is therefore
  predicted to remain exact through its live LDF term materialisation and
  ordered add.  A value mismatch in the actual term, masking, ordered add, or
  a fold/halo operation before the recorded post-LDF boundary refutes this.
  If none differs, the first confirmed owner moves to the next admitted outer
  call boundary; no inner statement is inferred across a missing record.
- **R333-R8-P2 — NaN localization.**  The stored thickness is predicted to
  contain zero dry/sub-bottom cells that reach a division transpose, where a
  zero cotangent still evaluates a `0/0`-derived divisor derivative.  The
  first non-finite reverse primitive and every affected cell class will be
  measured at the smallest production-faithful N=1 replay.  A first NaN on a
  strictly positive wet divisor, or before the stored-divisor graph, refutes
  this mechanism.
- **R333-R8-P3 — candidate.**  A branch-safe divisor form may use the stored
  value on positive/live cells and a finite inactive surrogate on zero/dry
  cells, while retaining the previously finite rebuilt tangent.  It qualifies
  only if its forward bits equal the direct stored-divisor candidate, its
  reverse gradients equal forward mode, and Decision 122, both rung-10 labels,
  the faithful DINO probe, and the N=1 gate all pass.  Otherwise Resume 8 is a
  measured held proposal with no package-source landing.
