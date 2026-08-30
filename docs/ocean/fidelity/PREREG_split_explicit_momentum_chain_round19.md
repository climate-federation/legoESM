# Preregistration amendment: production literal seed and ordered replay, round 19

Date: 2026-08-29. Frozen before production measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

Round 18 owned row 1.2 exactly: restart BEFORE `sshb` live QCO U/V thickness,
surface-to-bottom left accumulation, then the separately associated live
reciprocal reproduces NEMO on 9,758/9,758 U faces and 9,868/9,868 V faces.
The source is `istate.F90:149-155`; the seed is copied into the split-explicit
carry at `dynspg_ts.F90:571-579`.

## Frozen implementation and controls

Add `barotropic_seed_evaluation`, with `generic` preserving the previous
stacked reduction and `nemo_literal` preserving the cited NEMO recurrence.
Only `nemo_dino_kamm` and `nemo_dino_kamm_mlf` select `nemo_literal` by
default; every other DINO recipe remains byte-pinned to `generic`. Literal
evaluation requires `barotropic_seed_face_depth="nemo_ssh_avg"` and unknown or
illegal selector combinations raise. Tests must prove:

1. the generic leaf and full solver are byte-identical with the selector
   omitted versus explicitly set;
2. the literal primitive matches an independent source-ordered NumPy loop,
   while a reversed-order planted control differs;
3. JIT output is identical and gradients remain finite and nonzero; and
4. both fidelity cards route the literal selector while all other cards route
   generic.

No bar, population, time level, seed input, or scorer logic changes in this
round.

## Frozen production replay bars and ordered release

The first authoritative replay is the existing hardened day-180 recurrence.
Row 1.2 CONFIRMS the production fix only if both U and V have normalized RMS
error `0`, maximum error/NEMO RMS `0`, and mismatch count `0/N_wet`. Any
nonzero value is `IMPLEMENTATION_DEBT` and stops the walk at row 1.2.

If row 1.2 is exact, row 1.3 is re-evaluated in executed source order at
`dynspg_ts.F90:766-850`: back-interpolated SSH/pressure gradient, live EEN
Coriolis, explicit bottom stress, then the final vector update association.
Each operand uses the campaign POINTWISE bar: normalized RMS error and maximum
error/NEMO RMS both at most `1e-15`; a first operand above either bound is the
ordered stop. The final substep-1 U/V outputs are not promoted ahead of an
unclosed operand.

Only if every row-1.3 operand is AT-BAR may row 1.4 be adjudicated against its
unchanged final `puu_b/pvv_b/pssh/un_adv/vn_adv` dumps. Only an AT-BAR row 1.4
releases registry rows 2--6, then the free-surface filter, momentum-RHS, and
tracer-tail chains in NEMO execution order. Existing dumps are inventoried and
used first. A missing operand or a production-only contrast stops measurement
and emits a SLOT-protocol held block; post-hoc substitution is not an ownership
verdict.

The round-16 exact continuity rows 9.1--9.7 remain closed and are regression
checks, not reordered ahead of rows 1.2--1.4.
