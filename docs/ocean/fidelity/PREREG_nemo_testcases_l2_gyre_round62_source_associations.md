# GYRE round 62 source-association candidate preregistration

Date frozen: 2026-09-12. Parent tip: `84cff05d8208`. CPU/fp64/libm.

This is frozen after the operand ladder, but before changing production code.
It is a candidate test, not a post-hoc promotion of the month predictions.

1. Materializing the complete tracer face coefficient after NEMO's written
   `-p2dt*zwt/e3w` assignment will make the production matrix and sweep
   bit-exact with all recorded operands. REFUTE if any consumed `zwi`, `zwd`,
   `zws`, T, or S bit differs.
2. Replacing the split stage-3 content update by NEMO's written
   `e3t(Kbb)*T(Kbb) + p2dt*e3t(Kmm)*Krhs` association will remove the
   `1.679392692e-3` content-T maximum and take the independently advanced kt=3
   T row to its `1e-15` normalized bar. REFUTE if content is non-bit or kt=3 T
   remains above bar.
3. No card/configuration selector is added. A shared-code change is ineligible
   to ship unless the focused tests, planted violation, GYRE ladder/day gates,
   and Rule-12 static card census all pass without moving first-over-bar
   earlier.
