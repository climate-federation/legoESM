# Preregistration — round 197 / VORTEX round 13

Frozen BEFORE any measurement of this round.  Lane tip `3b4120055`
(round 196's HELD landing).  Case `VORTEX_VEC-zco`.  Evidence root
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round197/`.
Decision 82 (user, 2026-10-01 20:00): round 194's two-solve candidate
stays HELD and the two-ULP ratchet is unchanged.

## What round 196 left

Round 196 admitted a per-substep record of NEMO's `dyn_spg_ts` (ten
records, `kt=1..10`, 50 frames each) and walked the barotropic
sub-time-step loop against legoESM's production-jitted step.  Both
loop-entry operands were substituted with NEMO's own, one at a time,
and neither moved the end of the window: the owner is INSIDE the loop.
Every boundary stays within 6.2x of the `1e-15` bar through substep 25
and then grows together at 2.17x per substep, reaching
`1.2458906277138998e-08` in the updated velocity at substep 48 — the
certified downstream residual to fourteen significant digits.  The
velocity update's budget on field maxima closes to 4-5% from substep
28 with the 2-D Coriolis trend (`dyn_cor_2D`, `dynspg_ts.f90:503`)
carrying 92% of the per-substep injection.  Round 196 did NOT show
that the Coriolis statement differs: per cell its response to the
operand it is handed is a flat 2-3 times the Coriolis parameter across
the whole window, while the ratio of field maxima at substep 48 is
about 80 times the frozen operator's `||L||_inf ~ f` norm bound.  The
two readings disagree, and only a substitution decides which of them
decides the window.

## What this round builds

ONE new instrument: a private per-substep override of the barotropic
trend inside the compiled scan.  It is a static closure capture keyed
by the scan's own `substep_index`, in the style of the existing
stage-2/3 and loop-entry hooks: default `None`, no constructible model
configuration can select it, and when it is `None` the Python `if` is
resolved at trace time, so the jaxpr is unchanged.

## Predictions (falsifiers named)

* **P1 — the hook is zero-change when unset.** With the hook present
  and unset, (a) the production barotropic walk at this round's commit
  reproduces round 196's `spgts_walk_kt1.json` for every one of its
  1059 rows and 336 scalars, bit for bit, and (b) the GYRE certified
  kt=1..10 trajectory ladder and the round-191 VORTEX-vector 50-row
  registry are byte-identical, and the certified GYRE days (day 30
  `2.3432510206121264e-06`, day 240 `6.581707093530567e-05`, day 360
  `5.407735418221895e-05` K) are unmoved.  FALSIFIER: any row, scalar
  or certified number that differs.
* **P2 — the substitution binds, and says so itself.** With NEMO's
  recorded `cor_u`/`cor_v` substituted at every substep, the walk's
  `cor.u`/`cor.v` rows are bit-exact (zero unequal faces) at all 48
  substeps.  FALSIFIER: any non-zero count — the override did not land
  at the boundary claimed and the arm reports NULL, not a result.
* **P3 — the question.** The end-of-window updated velocity
  (`new.u` at substep 48, normalized max abs; production
  `1.2458906277138998e-08`).  CONFIRMS "the barotropic Coriolis
  operator is the statement" if it is AT-BAR, `<= 1e-15`.  REFUTES it
  if it stays within one order of magnitude of production,
  `>= 1.2e-09`.  Anything between is PARTIAL: the Coriolis trend owns
  part of the window and the remainder is reported with the next
  operand named, not called closed.
* **P4 — the next operand, measured in the same round if P3 does not
  confirm.** The surface-pressure gradient (`zu_spg`/`zv_spg`,
  `dynspg_ts.f90:498`) substituted the same way, one variable.
  Round 196's budget gives it 8% of the per-substep injection against
  Coriolis's 92%, so the prediction is that it does NOT close the
  window either (`>= 1.2e-09`).  FALSIFIER: it closes, which would
  mean the maxima budget mis-ranked the two terms.
* **P5 — nothing lands unless a cited compiled statement closes every
  gate.**  A substitution that closes the window is not a fix: it
  names the statement, and the fix is then walked to the compiled line
  and landed under Decisions 43/45/55/59, the certified 50-row
  trajectory gate, GYRE byte-identical, the generic NEMO-GYRE recipe
  and the DINO month gate.  Otherwise the round is HELD.

## One variable

Each arm changes exactly one operand of the compiled loop and nothing
else: same card, same seed from NEMO's own recorded kt=1 step entry,
same production-jitted `model.step`, same precision policy
(fp64/libm), same CPU backend, production JIT on.
