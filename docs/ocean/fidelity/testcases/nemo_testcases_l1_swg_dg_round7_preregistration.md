# SWG double-gyre round 7 preregistration

Date: 2026-10-12

This round discriminates the remaining SWG_DG stage-record debt with NEMO's
own per-step restarts.  It changes no physics, configuration, production
arithmetic, acquisition artifact, or fidelity bar.  The exact IEEE-word bar
remains binding, including the inherited forcing signed-zero debt pending the
user's ruling.

## Fixed evidence and coverage

The admitted round-1 instrumented directory contains stage-entry records for
`kt=1..10`, stages 1..3, and instantaneous restarts after steps 1..10.  The
passivity admission proves those restarts and all ordinary NEMO outputs are
word-identical to the uninstrumented reference run.  All inputs remain
content-pinned by the production ladder.

The committed round-7 instrument will compare:

1. the record at `kt=N+1`, stage 1 with restart `N`, for `N=1..9`;
2. legoESM's from-rest production state after `N` steps with restart `N`, for
   `N=1..10`;
3. the ladder's three-stage chained replay after `N` steps with restart `N`,
   for `N=1..10`; and
4. legoESM's production stage-1 output at each `kt=1..10` with that same
   step's recorded stage-2 entry.

Comparison 1 cannot cover `N=10`: the sanctioned writer condition is
`kstp <= nit000 + 9`, so an acquisition starting at `nit000=1` intentionally
contains no `kt=11` entry.  This is recorded as UNMEASURED, not inferred.

The common restart/entry fields are SSH, U, V, whole-step-before U/V, and the
before wind stresses.  Production-state comparison additionally covers every
admitted restart field (`sshn`, `un`, `vn`, `uu_n`, `vv_n`, `utau_b`,
`vtau_b`, `emp_b`, `qns_b`, `sfx_b`).  Stage-output comparison covers all 13
stage fields.  Counts are reported separately for wet interior, wall-adjacent
wet cells, and dry/storage cells; the fidelity verdict uses the field's wet
T/U/V/F mask.

## Oracle capture-point statement

The instrumented compiled source writes stage 2's `ssh(:,:,Nnn)`,
`uu/vv(:,:,Nnn)`, and `r3*(:,:,Nnn)` after stage 1 has applied both U/V and
ratio halo exchanges and swapped `Naa` into `Nnn`
(`SWG_DG_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3.f90:204-235`).  The next statements
zero only `uu/vv(:,:,:,Nrhs)` before stage 2 reads `Nnn` in `dyn_adv`,
`dyn_vor`, pressure, and `ssh_nxt` (`stprk3.f90:238-255`).  Therefore the
existing stage-2 record already captures the array and time level consumed by
stage 2; a second dump is not needed unless the comparisons contradict this
source-level identity.

## Frozen outcomes

- A record stage-1 entry unequal to the preceding restart identifies a
  boundary representation/capture debt.
- A production state or chained endpoint unequal to the same restart
  identifies the first step and field where end-to-end identity fails.
- If both endpoints are exact while a stage-1 output differs from stage 2's
  record, the internal intermediate is not word-exact even though later RK
  stages cancel or round away the difference.  That result does not authorize
  changing arithmetic or weakening the bar.
- A one-ULP change to a nonzero wet admitted U input must change the stage-1
  SSH output, or the instrument is non-discriminating.

A writer owner yields a proposed additive writer correction as
`DECISION_NEEDED`.  A ladder-composition owner may be fixed only after the
shipped rows are re-proven unchanged.  An arithmetic-only internal residual is
reported as a decision, never patched in this round.

## Preserved gates and inherited items

The shipped no-argument card digest must remain `f9d46bfd08f23702`; its rows
must remain given-entry 338/338, chained 338/338, independent 50/50, carried
29/29, and month 690/690.  Both decks' stage/month acquisition preflights must
remain passing.  The unrelated inherited GYRE/LOCK_EXCHANGE/OVERFLOW digit
pins remain stale: expected `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
`7f1ec89b3e6578f5`; current `f02d2f42a570f413` /
`d5dfdf83b3a1e0ab` / `83de0010e1170032`.
