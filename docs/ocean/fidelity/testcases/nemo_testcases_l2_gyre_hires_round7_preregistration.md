# NEMO testcase L2 GYRE high-resolution — round 7 preregistration

Date: 2026-10-13

Incoming tip: `d279942e5e666bbbcf880cfaeffa1ad5eefd4770`

This registration precedes every Round-7 legoESM trajectory score.  The two
operator-provided scalar-math roots are admitted as the controlling oracle
flavour because their NEMO executable was built with the campaign's
`-fno-tree-vectorize` scalar-libm policy.  The Round-1 and Round-2 roots remain
admitted as the labelled vector-math flavour.  No root may be substituted by
path alone: the executable, namelist, and ten step-entry payload hashes must
match the committed admission.

## Ten-step ladders

For each of `r4` and `r4_smag`, score T, S, U, V, and SSH at step entries
`kt=1..10` against both math flavours under legoESM CPU/fp64/libm.  The exact
bar remains zero unequal active float64 words.  Each flavour must retain the
five one-ULP plants at `kt=1`, and every plant must fire.

The scalar flavour is controlling.  Before seeing the score, the alternatives
are registered as follows:

1. If `kt=2` is exact in all fields, the vector-flavour `kt=2` debt is assigned
   to the compiled vector-math forcing program only; continue to the first
   scalar-flavour debt.
2. If `kt=2` is not exact, locate the first differing field, active index,
   level, absolute error, and ULP count before considering the step-2 owner.
   Every SIN, COS, and EXP on the executed `usrdef_sbc` path must be disposed
   from the compiled source before a physical operator is blamed.
3. For the first scalar-flavour debt at `kt=3`, name the largest temperature
   location and classify it as surface, interior, wall, or mixed-layer-base.
   A first RK stage or operand may be named only from a scalar-math record that
   serializes that boundary.  If the present record lacks it, ship a write-only
   scalar-math writer and request acquisition; do not infer the owner from the
   final state.
4. One-variable tests substitute one NEMO-recorded operand only.  Confirmation
   requires the downstream residue to fall to the scalar-oracle arithmetic
   floor.  A reconstruction using legoESM's own operator is not evidence.

The candidate order is fixed before the score: step-2 TKE/EVD coefficients and
implicit vertical-solve operands; step-2 solar/non-solar surface heat operands;
isoneutral slopes; then FCT limiter operands.  The scalar writer will record the
earliest unresolved boundary only, using the existing campaign record formats.

## Review repairs

The acquired `ahmt/ahmf` and momentum-LDF discriminator must score both `kt=2`
and `kt=3`.  The `kt=3` record may remain unable to separate inherited input
debt from operator arithmetic; the report must say so rather than promote it.

The GYRE forcing clock will reproduce the executed NEMO program literally:
`REAL(kt)*rn_Dt/(60*60) - (nyear-1)*24*nyear_len(1)` from compiled
`usrdef_sbc.f90:92,105-106` and `:176-177`.  It must be bit-equal over all
43,800 quarter-degree five-year steps.  The shipped one-degree ten-step card
must remain byte-identical because no completed model year occurs there.  The
redundant completed T-coefficient mask is removed only if the compiled dynamic
mode-32 branch confirms its absence.

## Thirty-day report

Score the acquired vector-math `r4_smag` day-boundary entries at
`kt=1,25,...,721` with the existing exact ladder and all one-ULP plants.  This
is a report, not a scalar-policy certification.  The flavour confounds any bit
verdict after the first forcing call.  The Round-6 branch prediction remains
frozen: by day 30 the lower-limit fraction is below its `kt=10` value, the
deformation fraction is nonzero, and the upper limit is active on at least one
coefficient point.  The census must use NEMO's own day-30 before-state.

The required follow-up acquisition is a scalar-math, single-rank copy of the
30-day daily writer.  It uses the exact `r4_smag` deck, `-m conda-scalarmath`,
`-d "OCE"`, the existing `kt=1,25,...,721` write schedule, and no change to
NEMO's engine sources.

## Controls and stop rules

- Hash, flavour, deck, cadence, or record-sequence mutations must refuse.
- Every newly added numerical comparison must have a planted failure.
- The shipped card digest stays `99b2dfce324b0f77`, and its registered
  ten-step trajectory may not move.
- If a first scalar RK-stage or candidate operand is absent, Round 7 stops at
  `ACQUISITION_NEEDED` with exact build/run commands.
