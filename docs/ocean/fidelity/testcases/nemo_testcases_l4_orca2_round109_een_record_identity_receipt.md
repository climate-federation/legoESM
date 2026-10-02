# ORCA2 round 109 — EEN record identity and per-level walk

Date: 2026-10-02. Base `1ab7699e61a8876d6eb076cbf3dd5fdaa311e64e`.
Scope is ocean-only measurement on hierarchy rung 0. Every ocean number below
is **independent** because rung 0 starts from NEMO's own from-rest state.

## Verdict

**HELD.** None of the operator's three proposed producer failures is true.
The base, round-105 accumulator, and round-108 per-level builds have byte-
identical rank-0/rank-1 kt=10 restarts; the round-105 and round-108 decks are
content-identical; and the inherited round-105 streams are byte-identical.
The refusal was a checker-axis defect: it compared a native `(i,j)` owned slab
to the inherited helper's latitude-major `(j,i)` slab without transposing.

After that one association repair, the existing record admits with all twenty
kt=1..10 restarts byte-identical. The source-ordered walk names NEMO's
three-term `zpvo_nw` assignment as the first raw non-bit statement. No model
physics, card field, configuration value, threshold, stabilizer, carried
state, sea-ice selector, or ORCA2 `unmeasured_features` entry changes.

## Three-build discrimination

The committed three-build gate reports `PASS_R109_THREE_BUILD_IDENTITY` and
both of its controls fire. The two kt=10 restart SHA-256 values are:

| rank | base = round 105 = round 108 SHA-256 |
|---:|---|
| 0 | `1ad3955cc76ea57d897ef30202cffdf6ce5a675025ede56b1df29a61d0eeab6a` |
| 1 | `7251f2f300ae0b81e2d6bf02f97a035fc92ac903eae15219f8f25b031d3392ab` |

Round 105 and round 108 also share the exact namelist, deck manifest, and
input manifest. The round-83 base has the earlier Decision-83 deck text (the
already gated inert `nn_chldta`, `ln_sssr_bnd`, `ln_spc_dyn`, and explicit
vertical-scheme spelling), while its input manifest and terminal restart bytes
are identical. Thus R109-P1 is **REFUTED literally** by known inert deck text,
but candidate (b), a different numerical deck/build, is refuted by both the
recorder-build content identity and the three-way restart identity.

- R109-P2: **CONFIRMED**; round 105 is passive relative to the base.
- R109-P3: **REFUTED**; round 108 is also passive.
- R109-P4: **REFUTED AS WRITTEN**; no reduced writer is needed. The compiled
  diff contains the registered additions only, and the checker, not the
  computation, moved.
- R109-P5: **CONFIRMED** until admission; no operand was quoted from the
  refused association.

The fixed checker explicitly transposes the inherited `(j,i)` block to native
`(i,j)` before bit comparison. A new association plant suppresses that
transpose and reproduces the exact old refusal. The unplanted gate admits two
13,320-cell terminal accumulators at zero unequal bits, verifies the recorded
recurrence at zero unequal bits, and compares all twenty restarts bytewise.

## First non-bit statement

The compiled U loop is bounded by `mbku`, then evaluates `zpvo_nw` as three
ordered `ff_f / e3f` fractions
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1241-1245`). It then
records the live thicknesses, neighbor mask, product, accumulator-before,
source recurrence, and accumulator-after in that order
(`ORCA2_OMIP_L4_R107EENSTEP/BLD/ppsrc/nemo/dynspg_ts.f90:1256-1263`).

The corrected rank-complete, executed-level-only comparison is:

| source-order boundary | bit unequal | magnitude unequal | signed-zero only |
|---|---:|---:|---:|
| `mbku` | 0 | 0 | 0 |
| `zpvo_nw` | 180 | 180 | 0 |
| live U thickness | 0 | 0 | 0 |
| live V thickness | 0 | 0 | 0 |
| neighbor V mask | 0 | 0 | 0 |
| stored product | 2 | 0 | 2 |
| accumulator before | 1,618 | 0 | 1,618 |
| accumulator after | 5,195 | 0 | 5,195 |

The 180 `zpvo_nw` magnitude differences are all on global row `j=0`, level
`k=0`; the first is `(j,i,k)=(0,0,0)`. Therefore R109-P6 is **REFUTED AS
WRITTEN**: it correctly predicted the first named boundary but incorrectly
predicted signed-zero-only differences. R109-P7 and R109-P8 are **CONFIRMED**:
both thicknesses and the mask are bit-exact, and NEMO's own recorded
`acc_after = acc_before + term` recurrence has zero unequal bits.

The two downstream product sign differences and 5,195 accumulator-after sign
differences are reported, not independently attributed. The first-non-bit
rule stops the walk at the earlier `zpvo_nw` statement. Its three individual
fractions were not recorded, so deciding whether the owner is the southern
halo association, one quotient, or the ordered sum is the next measurement.

## Retraction

An initial round-109 probe compared all 30 legoESM levels against NEMO arrays
that are initialized to zero and written only inside `DO jk=1,mbku`. Its large
375,383-cell thickness counts are **RETRACTED**. The corrected probe first
gates `mbku`, proves NEMO's dummy level 31 is untouched, zeroes both arms
outside the executed loop, and produces the table above. The invalid counts
are preserved in the preregistration correction and are not used by any claim.

## Mechanical evidence and validation

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round109/`:
`three_build_identity.json`, `een_step_admission.json`, and
`een_per_level_walk.json`. The three-build restart/deck plants, the admission
header/field/dimension/truncation/missing-field/rank/bottom/recurrence/
association/restart plants, and the per-level oracle/model bit plants all
fire. The measurement runs production JIT on CPU in fp64/x64 with libm.

Focused recorder, operand-walk, and three-build tests pass. This round changes
no `packages/` file, so the GYRE, DINO, tank, rung-0 trajectory, and rung-7
trajectory implementations do not move.

## OPEN

1. Record the three individual `zpvo_nw` fractions on both ranks, including
   the southern halo values, and compare them in compiled source order.
2. Only after `zpvo_nw` is bit-exact, return to the two product signed zeros
   and the accumulator-addition signed-zero semantics.
3. Then resume the separate 68-cell south-U debt, northern V cancelling pair,
   and later 68-cell substep-2 U residual.
4. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy deliverables.

## UNVERIFIED

- Which of the three `zpvo_nw` fractions or their association owns row `j=0`.
- The owner of the downstream signed-zero addition differences after the
  earlier `zpvo_nw` statement is closed.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
