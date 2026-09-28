# GYRE round 62 pre-code self-review

Date: 2026-09-12. Tip read: `b6027cebae17`; preregistration commit in the
writable stamp clone: `4189e2c68`.

## Claim boundary

No production change is justified yet. The current clean-stamped equal-input
run leaves kt=3 T at max `1.627511417652e-4 K`; replacing only the recorded
content RHS leaves `1.002651828230e-8 K`, replacing only effective K leaves
`1.627503129029e-4 K`, and replacing every recorded ZDF solve operand leaves
`3.310773877274e-11 K`. These are measurements, not an attribution.

## Source alignment to test

| order | compiled NEMO statement | existing legoESM boundary | next arm |
|---|---|---|---|
| 1 | `trazdf.f90:546-560` forms the surface and interior thickness-content RHS from `e3t(Kbb)`, `e3t(Kmm)`, state, and tendency | captured `content_T/S` at the production dispatch | recorded content only |
| 2 | `trazdf.f90:450,461-477` zeros the surface coefficient, then forms lower/upper from effective K and `e3w(Kmm)`, and diagonal from `e3t(Kaa)` | captured `K`, `dz_after`, and `e3w_now` | each operand alone, then source-order cumulative arms |
| 3 | `zdfevd.f90:107-110` replaces tracer K where `MIN(rn2,rn2b)<=-1e-12` | captured post-EVD K | EVD and stable-interface K partitions |
| 4 | `trazdf.f90:523-528,545-578` performs the ordered factor, forward sweep, bottom divide, and backward sweep | already bit-exact given NEMO operands | surface/interior/bottom row substitutions retain boundary location |

## Instrument checks before trusting a result

- Extend the existing round-54 probe; repository search found no second live
  equal-input ZDF operand ladder.
- Every arm changes one captured dispatch operand, except explicitly named
  cumulative and row-location arms. The model entry state, forcing, card,
  masks, solver, and NEMO record remain identical.
- Report exact unequal cells and normalized max, not RMS alone. A production
  fix is eligible only when the changed statement is bit-exact given NEMO
  inputs and the full kt3 T row is at its `1e-15` normalized bar.
- Add a nonzero one-ulp operand plant and require nonzero exit. Keep the stamp
  fail-closed. Do not touch the year harness or reconciliation gate.
- No configuration choice, carried-state change, NEMO source edit, GPU run, or
  new numerical implementation is authorized by this instrument extension.

