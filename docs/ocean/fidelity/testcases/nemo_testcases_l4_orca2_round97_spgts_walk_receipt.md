# ORCA2 round 97 — rung-0 split-explicit statement walk

Date: 2026-10-02. Base `ae79a83f7`; measurement commit `538a93078`.
Scope is ocean only. Every number below is labelled **independent**: the rung-0
deck starts from its own climatological initial state, not from NEMO's recorded
entry state.

## Verdict

**HELD.** The existing round-96 NEMO output is sound and now admits under
content-addressed producer provenance. Its two rank-complete substep streams
cover the domain exactly once, and the Decision-83 harmonized run's twenty
terminal restart shards are byte-identical to the admitted round-93 run.

The first non-bit split-explicit statement is `dyn_cor_2D` at substep 1. Its U
output differs at 398 active cells and its V output at 400 active cells, all by
signed zero: maximum absolute error and rms are both exactly zero. At substep
2 the error becomes nonzero. NEMO's four-cell masked `e3f_0vor` divisor is a
confirmed contributor, but a one-variable substitution does not make the
65-substep Coriolis record exact. No model statement lands in this round.

No `packages/` file, public card field, configuration choice, stabilizer,
carried-state convention, threshold, sea-ice field, or `unmeasured_features`
entry changes. The held round-94 slow-depth candidate remains absent.

## Existing record admission and Decision 83

The round-96 launcher had already run NEMO to `STOP 0`; its only refusal was a
producer-commit allowlist written before later round commits. Round 97 replaces
that moving commit-object requirement with exact SHA-256 pins for the deck
patch, writer patch, checker, launcher preregistration, and pinned pre-patch
compiled source. The `--plant-toolchain` control changes one recorded digest
and fires before admission.

Admission of the existing directory reports:

| Item | Independent result |
|---|---:|
| rank streams | 2, exactly-once coverage |
| bytes per stream | 218,151,872 |
| frames per stream | 67 |
| named groups per stream | 2,106 |
| substeps from header | 65 |
| terminal restart comparisons | 20/20 byte-identical |

The record SHA-256 values are `d24710b945b007f8ee6823e0554274ca3d29c9f3d26e548b2694f275f4b7e7cf`
for rank 0 and `fe8a930a74f094a16f8d9bcab37e31b1a0c128fc2fbeb162e5b9cee8363cadb1`
for rank 1. This confirms R97-P1 and R97-P2. It also completes Decision 83's
inertness requirement: all ten steps on both ranks retain their pre-harmonized
restart bytes.

The admission JSON is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round96/acquisition/orca2_rung0_spgts_ranked_10step_np2/round96_spgts_admission.json`.

## Source-ordered walk

The committed walker validates the complete self-describing header before
extracting any payload, assembles owned-only and two-cell-haloed fields by
their own declared shapes, and compares the trace hook against the ordinary
solver at the same barotropic boundary. SSH, 3-D velocity, barotropic velocity,
and both transport averages are bit-identical between observer and ordinary
paths.

NEMO computes continuity, transport accumulation, face SSH, backward SSH and
pressure gradient before calling and recording `dyn_cor_2D` in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:580-666`. With NEMO's
recorded final slow forcing and raw barotropic histories substituted one
variable family at a time, the first-substep read-out is:

| Boundary | U | V | SSH |
|---|---:|---:|---:|
| entry forcing/state | bit | bit | bit |
| midpoint and face depth | bit | bit | bit |
| metric transport | bit | bit | n/a |
| after continuity | n/a | n/a | bit |
| transport sums and face SSH | bit | bit | bit |
| backward SSH and pressure gradient | bit | bit | bit |
| Coriolis | 398 signed-zero cells | 400 signed-zero cells | n/a |

Thus R97-P3 is **CONFIRMED**: the first non-bit boundary is no later than the
first recorded after-SSH boundary, and in fact lies later at the Coriolis call.
The first active non-bit statement is the source-associated four-pair
Coriolis application in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1369-1392`.

The full result is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round97/spg_walk_divisor_final.json`.

## The `e3f_0vor` arm

The compiled EEN coefficient program divides each potential-vorticity term by
`e3f_0vor * (1 + r3f*fe3mask)` in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265`.
For `nn_e3f_typ=0`, NEMO constructs `e3f_0vor` as the four surrounding masked
T-cell reference thicknesses divided by four, exchanges the F-point fold, then
replaces remaining zeros from the mesh F thickness in
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynvor.f90:912-937`.

The card already says `een_e3f_scheme="nemo_avg4"`, but changing that selector
to `min` produces an identical 130-row Coriolis score. This is a useful
reachability control: the literal coefficient builder reads its carried raw
F divisor and bypasses that selector. The active one-variable arm therefore
changes only that raw divisor from mesh `e3f_0` to the source-ordered
`e3f_0vor`; the F-column depth, masks, live stretch, metrics, forcing,
histories, and every other coefficient operand stay fixed.

| 65-substep Coriolis score | Baseline mesh `e3f_0` | Source `e3f_0vor` |
|---|---:|---:|
| face/substep rows | 130 | 130 |
| bit-exact rows | 0 | 1 |
| sum of active differing cells | 2,026,979 | 1,995,569 |
| maximum absolute error | 7.92217515982671e-6 | 5.672230850339018e-6 |
| substep-2 U maximum absolute error | 1.6425537782543814e-7 | 2.9617669311254642e-8 |
| substep-2 U differing active cells | 15,692 | 68 |

R97-P4 is **CONFIRMED as contributor and REFUTED as sole owner**. The arm
strongly improves the first nonzero Coriolis row, but leaves the substep-1
signed-zero statement and substantial later debt. It is therefore not a
single-statement landing for this round. Decision 84's shared-path landing is
already assigned to the GYRE lane; this lane will remeasure after that merge
instead of duplicating the shared change.

## Controls, tests, and review

All three round-97 controls fire:

- rank-owned layout overlap/gap: `STATUS PLANT-FIRED layout`;
- one-ULP record mutation: `STATUS PLANT-FIRED record-bit`;
- one-ULP trace-output mutation: `STATUS PLANT-FIRED trace-bit`.

The focused record/launcher/walker and citation-gate selection passes 29/29 in
3.30 s. The default citation receipt passes with 274 citations, zero failures,
zero unmapped citations, and a clean map audit. This receipt passes with four
citations and the same zero-failure result. A rigid +2 shift of the
`dyncor_2D` citation is rejected by the planted control.

The single `tests/ocean/fidelity -n 12` battery reached 99% before the known
xdist silent tail left no live pytest process and no terminal summary; it is
therefore recorded as incomplete, not PASS. Its six visible failures are the
same established failures reported by the preceding rung-0 round:

- test_nemo_testcase_l2_gyre_round129_spread_floor_gate.py::test_record_backed_gate_passes;
- test_nemo_testcase_l2_gyre_round51_live_operands.py::test_live_trace_and_raw_history_arms_are_private_and_off_by_default;
- test_nemo_testcase_round35_stamp_scope.py::test_every_driver_that_arms_the_escape_scopes_it;
- test_recipe_case_board.py::test_every_oracle_comparison_has_a_row;
- test_nemo_testcase_worktree_stamp.py::test_every_report_emitter_stamps_the_worktree;
- test_nemo_si3_scalarmath_v2_gate.py::test_full_v2_gate_and_plants.

The required separate `codex exec --sandbox read-only` review did not reach
the diff: `failed to initialize in-process app-server client: Read-only file
system`. Verdict: **independent review unavailable in-sandbox**.

The package tree is identical to base, so GYRE's certified ladder and year
trajectory are unchanged by construction. No GYRE trajectory claim is used
to promote the held measurement arm.

## OPEN

1. After the GYRE lane's Decision-84 shared divisor statement reaches this
   branch, rerun this exact 65-substep record and register every moved row.
2. Walk the residual coefficient program from the first substep-2 nonzero
   cell at `(j=103, i=114)` in the baseline and the fold-row residual at
   `(j=147, i=134)` under the source divisor; distinguish coefficient
   association, fold mapping, and signed-zero construction one variable at a
   time.
3. Only after the Coriolis row is closed, score the explicit bottom-drag and
   velocity-update statements that follow it.
4. Keep the rung-0 card's given-entry and independent ten-step ladders and its
   independent month as the next hierarchy deliverables after the statement
   walk closes.

## UNVERIFIED

- The source-exact divisor is not a complete owner of the Coriolis mismatch.
- The residual coefficient/fold/signed-zero owner is not yet named.
- No package change or candidate ten-step ladder is proposed in this round.
- The rung-0 given-entry ladder, independent ladder, and month remain later
  hierarchy work.
