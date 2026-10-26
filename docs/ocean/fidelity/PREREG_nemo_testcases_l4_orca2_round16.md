# NEMO testcase Lane 4 — ORCA2 card round 16 preregistration

Date: 2026-09-25

Parent: `956fd11fc257dc88bfc18c39399fe350f2e3301a`

Status: **PREREGISTERED BEFORE ANY ROUND-16 MEASUREMENT.**

Round 15 substituted the recorded drag and six cold histories.  The histories
moved the end-of-step sea surface by exactly zero and advanced the first active
mismatch to substep-1 `slow_u`: max `5.370080135032166e-12`, followed by
`slow_v` at `2.2928581685638914e-11`.  The same trace reported `u_exit` and
`v_exit` differences of `8.922594685670249e-10` and
`3.809672033947605e-09`.  Round 16 substitutes only the two recorded slow
forcings, measures their causal movement, and walks the compiled vector update
one arithmetic statement at a time.

Every number is labelled **given NEMO's entry**.  Decision 52's recorded entry
sea surface is retained.  The six sea-ice selectors and the card's
`unmeasured_features` tuple are frozen.  Decisions 54, 57 and 58 remain pending
and nothing in this round acts on them.

## Record and compiled statement

The admitted record is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition/orca1ice_surface_entry_every_step_a_np2`.
The executing compiled branch is
`ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY/BLD/ppsrc/nemo/dynspg_ts.f90:666-679`.
It evaluates, for each face,
`(entry + rDt_e * (pgf + trend + slow)) * mask`.  The ordered record supplies
all five operands and the exit for the first two substeps on rank 0.

The gate will reuse the round-15 record reader, native-face slicing, masks,
entry/drag/history substitutions and production-JIT trace.  It will reuse the
existing private `barotropic_slow_forcing_override` hook; no model API or
configuration selector is added.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R16-P1 | The inherited round-15 arm reproduces before the new substitution. | cold histories move end-of-step SSH by exactly `0.0 m`; first active rows are `slow_u=5.370080135032166e-12` and `slow_v=2.2928581685638914e-11`. | any different value; stop for instrument drift. |
| R16-P2 | The private slow-forcing hook is a one-variable channel. | feeding legoESM's own slow forcing leaves the traced exits and end-of-step SSH bit-exact; feeding the record lands `slow_u/v` bit-exact on rank 0 while all earlier rows remain unchanged. | either no-op moves a field, replacement misses its boundary, or an earlier row moves. |
| R16-P3 | The slow-forcing mismatch is not a material owner of the `0.2448430937728719 m` SSH row. | recorded slow forcing moves end-of-step SSH by `< 1.0e-06 m` and `< 2.4e-02 m`. | movement `>= 1.0e-06 m`; it has more leverage than the direct one-substep scale predicts. Movement `>= 2.4e-02 m` makes it an owner and stops the walk. |
| R16-P4 | The pre-substitution exit mismatch begins at the addition of slow forcing, not at pressure-gradient plus trend. | `pgf + trend` is bit-exact; adding `slow` is first non-bit for each non-bit face, and the following increment/exit rows inherit it. | an earlier arithmetic row is non-bit, or the slow-addition row is bit-exact. |
| R16-P5 | Substituting recorded `slow_u/v` closes the complete substep-1 vector update. | arithmetic replay reproduces NEMO's recorded `u_exit/v_exit` bit-exactly and the production-JIT substituted trace is bit-exact at both exits. | either exit remains non-bit; the first remaining arithmetic row becomes the named statement and no fix lands without a direct source-level proof. |
| R16-P6 | The existing kt=1..10 ORCA2 ladder is unchanged. | `LADDER_MEASURED`, same first statement, kt=10 entry temperature max `3.9430791763114783` on `430552` cells. | refusal, earlier first statement, or unregistered moved row. |
| R16-P7 | No production statement lands. | no `packages/` diff; validation-only instrumentation is private and off by default. | any ungated model diff. |

Failed predictions remain in the receipt as **REFUTED**.

## Controls and stop rules

- The replay must first reproduce NEMO's recorded `u_exit/v_exit` from NEMO's
  own recorded operands.  If it does not, its intermediate rows are not
  interpretable and the gate stops.
- The no-op arm must be array-equal to the inherited arm.  A one-ULP plant in
  one active recorded slow-forcing cell must reach the traced exit, or the
  substitution channel is deaf.
- Comparisons cover rank 0's owned 90 longitude columns with the admitted
  staggered masks.  No rank-1 or full-domain claim is made.
- No stabiliser, configuration selection, carried-state change, sea-ice change
  or production fix is allowed in this measurement round.
- The citation gate must pass, and its planted rigid shift must fail on a
  compiled citation rendered in the receipt.

## Choices

ASKED: retain the cold-history substitution, substitute recorded `slow_u/v`,
measure causal movement, then name the first arithmetic boundary in the
`u_exit/v_exit` update.

UNASKED: none.
