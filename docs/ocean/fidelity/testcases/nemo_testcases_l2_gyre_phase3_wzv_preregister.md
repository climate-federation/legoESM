# NEMO testcase lane 2 GYRE — stage-3 WZV boundary preregistration

Date: 2026-09-03  
Baseline: `fidelity/nemo-testcases-l2-gyre-codex2` after the stage-2 literal
HPG recurrence made stage-2 momentum Kaa AT-BAR.  CPU/fp64 only.

## Fixed measurement

The first remaining ordered transport boundary is the stage-3 tracer `zFw`:
the existing post-`tra_adv_trp` record measures absolute `1.0086864676850382e-10`
and normalized `2.4980249777374853e-13`.  NEMO's live vector-invariant branch
sets `ll_Fw`, calls `wzv(...,np_transport)`, applies `wAimp` at stage 3, then
writes `pFw=e1e2t*ww` (`src/OCE/TRA/traadv.F90:138-141,220-226`).

A config-local, WRITE-only `MY_SRC/traadv.F90` extension will dump, for
`kstp=nit000,kstg=3`, the arrays at these immutable boundaries:

1. `ww` immediately after `wzv(...,np_transport)`;
2. `ww` and `wi` immediately after `wAimp`;
3. final `pFw` after the `e1e2t` multiplication.

The pre-existing stage-3 state, kt=2 entry, restart, and post-`tra_adv_trp`
transport hashes must remain bit-identical.  A planted wet-cell `ww` violation
must classify DEBT by at least `0.9` absolute.  Each oracle/candidate array is
float64 and is scored pointwise with the immutable normalized `1e-15` bar.

## Outcomes fixed before the oracle run

- If pre-`wAimp` `ww` is first over bar, label only `wzv np_transport` as the
  first-divergence boundary; do not assign an owner until a one-variable arm
  moves at the residual scale.
- If pre-`wAimp` clears but post-`wAimp` first exceeds the bar, label the
  adaptive partition as the first-divergence boundary and compare its movement
  with the faithful residual before any owner label.
- If both clear but final `pFw` does not, label the metric multiplication or
  layout mapping as the first-divergence boundary.
- If all clear, advance to stage-3 tracer advection and then ZDF in source
  order.  No outcome changes the bar or permits a whole-step certification
  claim unless kt=2 itself clears.

The oracle source tree is not edited.  Instrumentation is confined to
`cfgs/GYRE_OMIP_L2_P3/MY_SRC`; no shipped NEMO file is modified.
