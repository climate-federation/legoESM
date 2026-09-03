# NEMO testcase lane 2 — GYRE stage-2 HPG geometry receipt

Date: 2026-09-01

Session: `ea650f83-28b8-4c68-b0cc-809a9fd417de`

Preregistration commit: `a117ccb8de227edb3188de96ddec0c26f29e35be`

Parent term split: `e0079d6acdcd7c7917f32970e96e14d0e58a7ff3`

## Verdict

**DEBT, with the full stage-1 Kmm T/S/SSH bundle confirmed as the dominant
causal contributor to stage-2 HPG but not the sole owner.**  Stage-1 SSH itself
is AT-BAR at `1.951563910473908e-18`.  SSH-only removes 94.51% of the corrected
stage-2 residual but leaves direct HPG and Kaa DEBT.  The full oracle stage-1
thermodynamic bundle makes direct HPG AT-BAR and moves
`0.9999992257753059` of the faithful residual, reducing corrected stage-2 U/V
from `1.0381851039053744e-7`/`2.4701793455947666e-7` to
`4.461405813467774e-13`/`4.915858979647261e-13`.  Because those rows remain
over the exact bar, the honest label is
**CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER**.

## Source and direct operands

The HYB program creates stage-1 N+1/3 SSH and QCO geometry at
`stprk3_stg.F90:130-175`, swaps it to stage-2 Kmm at
`stprk3.F90:215-225`, and calls EOS/HPG on Kmm at
`stprk3_stg.F90:338-344`.  The selected `hpg_sco` recurrence reads Kmm `e3w`
and `gdept_z0` at the surface (`dynhpg.F90:343-360`) and every interior level
(`:367-388`).  The existing oracle stage dump's SSH payload is now consumed
rather than discarded.

The candidate stage-1 SSH is the literal HYB interpolation from the accepted
post-`stp_2D` state.  It agrees to `1.951563910473908e-18` absolute on 600 wet
cells.  That proves the geometry state itself is available exactly; the debt
comes from failing to feed it to the later momentum-stage EOS/HPG.

## Scaling before labels

| arm / row | U absolute | V absolute | movement / faithful residual | disposition |
|---|---:|---:|---:|---|
| stage-1 SSH | — | `1.951563910473908e-18` | — | AT-BAR |
| SSH-only direct HPG | `3.029437947896299e-12` | `3.0294371790285413e-12` | — | DEBT |
| SSH-only corrected Kaa | `2.1811929415822785e-8` | `2.181192368912112e-8` | `0.9451076290264119` | CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER |
| full bundle direct HPG | `2.4902438534337878e-17` | `2.6792038271212327e-17` | — | AT-BAR |
| full bundle corrected Kaa | `4.461405813467774e-13` | `4.915858979647261e-13` | `0.9999992257753059` | CAUSAL_CONTRIBUTOR_NOT_SOLE_OWNER |

For the full bundle, direct HPG's propagated `dt/2` error is approximately
40% of the remaining corrected-stage residual.  That residual is real debt;
the direct HPG row's absolute `<=2.68e-17` clears its tendency-space bar but is
not silently promoted to a whole-stage match after multiplication by 7200 s.

## Controls, review, and next boundary

The stage reader now fails closed on a missing/truncated SSH payload.  The
private full-bundle arm changes only the stage-2 EOS/HPG input and substitutes
no returned state.  All prior oracle artifacts remain bit-identical.  The
planted `+1 m` stage-1 SSH control fired at exactly `1.0` (`VERIFIED`), and the
selected geometry/tracer tests passed `26/26` after correcting the fixture's
Fortran `CHARACTER(16)` space padding; the corrected reader fixture then passed
its targeted rerun.

This result is **UNREVIEWED** by independent Claude/GLM reviewers.  The next
ordered boundary is the production WS tracer stage program: stage-1 T/S are
already measured DEBT by `0.3241418107674683` and
`1.8542789845810148e-6`.  The canonical landing must interleave live Kmm
tracer/geometry state with momentum stages as one collapsed WS-RK3 identity;
no oracle injection or private micro-selector may ship.  No kt=2 trajectory
match is claimed.

Run root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gyre_kt1_10_stage2_terms`.
