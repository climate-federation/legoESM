# NEMO SWG level-1 round 3b — non-degenerate production replay

Date: 2026-10-11. Base: `640a4ff8c3fe7f34845d96025da9845fa87c181a`.
Evidence directory:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swg_rounds/round3b`.

## Preregistration

This section is frozen before running a round-3b legoESM measurement.  The
operator's independent review reproduced five blockers in round 3.  This fix
round changes only the stage program, its production dispatch, the ladder and
their direct tests and receipts.

Frozen predictions and falsifiers:

1. The admitted `kt=1, stage=1 -> stage=2` boundary will be labelled
   **NON-DISCRIMINATING**, never AT-BAR, because all 13 target fields equal the
   entry fields.  Any fidelity verdict attached to that identity boundary
   **REFUTES** the repaired instrument.
2. The same stage kernel called by normal production `step()` will implement
   all three compiled stages in source order: `Dt/3`, `Dt/2`, `Dt`; C2 then
   one-layer-zero ZAD; total ENS; pressure; continuity from the unchanged
   whole-step before SSH and the current-stage transport; QCO ratios; and the
   before-level masked velocity update.  Stage 3 additionally executes the
   shared NEMO Laplacian, centred wind stress divided by current thickness,
   and `-rn_rfr` times the whole-step before velocity, as compiled at
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:141-367` and
   `SWG_OMIP_L1_P3/BLD/ppsrc/nemo/dynldf_lev.f90:130-150`.  A separate replay
   arithmetic path, an unknown stage/selector fallback, or ordinary `step()`
   rejecting `swe_rk3` **REFUTES** the production repair.
3. Given-entry rows will start at nonzero `kt=2, stage=1` and cover every
   available stage transition through `kt=10, stage=2 -> stage=3`; stage-3
   transitions will be scored through `kt=9 -> kt=10`.  Every output field is
   predicted AT-BAR at `1e-15`; any unequal wet value is DEBT, and the first
   source-order statement producing it will be named.  Missing computable rows
   or relabelling DEBT as AT-BAR **REFUTES** coverage.
4. A chained replay seeded only from admitted `kt=2, stage=1` will run stages
   1, 2 and 3 through the remaining record and score every boundary.  It may
   expose accumulated DEBT, but its first-over-bar boundary may not precede the
   given-entry boundary.  An unreported divergence or a chain that rereads the
   oracle between stages **REFUTES** the chain.
5. Separate zero and one controls will perturb a wet, dynamically consumed
   `u_k` stage-entry input by exactly one `nextafter` step before JIT.  The
   scored `u_k` output must change and the control must exit nonzero.  Touching
   a pass-through field, scoring the input copy, a no-op ULP, or either control
   staying bit-identical **REFUTES** the controls.
6. A nonzero manufactured whole-stage test will exercise C2, ENS, pressure,
   continuity, QCO and U/V masking together and compare against a separately
   written source-statement oracle.  Each named contribution must be nonzero,
   and deleting or changing any one named term must fail the test.  A
   pressure-only case or a test assembled from the production helper
   **REFUTES** the manufactured oracle.
7. The round-3 preregistration's three post-measurement citation edits will be
   reverted byte-for-byte to commit `edcc0c9c1`; their corrected anchors will
   appear only in an ERRATUM below.  Any further mutation of that frozen section
   **REFUTES** provenance.
8. No other card selects `swe_rk3`.  Existing certified rows/hashes must remain
   unchanged, and no first-over-bar boundary may move earlier.  Otherwise Rule
   12 eligibility is **REFUTED**.

No configuration or carried-state choice is introduced.  The three stages,
forcing values, viscosity, drag, gravity, masks and timestep are all already
fixed by the shipped deck and the round-1 resolved card.

## ERRATUM to the frozen round-3 preregistration

Round 3 improperly edited three citation ranges after measurement.  The
historical text is restored to `dynadv.f90:134-139`, `domqco.f90:207-234`, and
`stprk3.f90:170-176`.  The precise compiled statement anchors are respectively
`dynadv.f90:134-138`, `domqco.f90:208-234`, and `stprk3.f90:172-176`; this
erratum records the correction without rewriting the preregistration again.

## Measurement

Pending.

## Review findings and fixes

Pending measurement.  The operator's round-3 verdict will be quoted verbatim.

## First non-bit statement

Pending.

## Rule 12 disposition

Pending.

## Gates

Pending.

## Independent review

Independent review run by the operator.

## OPEN

Pending measurement.  Thirty days and the shipped-run statistics remain out
of scope for this fix round.
