# Preregistration: split-explicit chain round 10 — V-face flux metric

Date: 2026-08-29. Status: **FROZEN BEFORE MEASUREMENT**.

## Question and fixed input

Round 9 stopped formally at subrow 9.3. Its frozen nested-arm rule did not
award the flux owner because the Q-depth arm was `UNRESOLVED`, not `REFUTES`:
its normalized prediction error and explained fraction met the refutation
amplitude clauses, but its correlations (0.832738 whole-domain, 0.998268 at
`j=1`, and 0.850052 at `j=197`) exceeded the registered 0.20 ceiling. That
verdict will not be changed post hoc.

The same receipt nevertheless supplies a separate, narrower question. The
F-flux arm predicts the pre-fix continuity residual with normalized errors
2.33174e-11 whole-domain, 9.71415e-12 at `j=1`, and 1.79595e-11 at `j=197`.
The Q-depth arm explains only 1.56505e-4 of residual RMS whole-domain. Is the
dominant flux-product residual the known V-face zonal-width construction?

This is an offline production counterfactual on the *same* six NEMO operand
dumps, bracket receipt, restart bridge, and CPU/fp64 scorer used for round 9.
No NEMO or legoESM integration is run. The only production difference is the
reviewed V-face metric fix carried from commits `c1cbef72ae4` and
`6917c20714c` on `fidelity/dino-register-items` (cherry-picked here as
`4ff604100fe` and `9445d8228c8`).

Post-measurement citation corrigendum (no bar, operand, or statistic changed):
the active uninstrumented oracle source is `zhU` at
`dynspg_ts.F90:699-701`, `zhV` at `:702-704`, and `zhdiv/ssha_e` at
`:722-724`. The `:703-708` references below were taken from the instrumented
writer source, whose inserted declarations/comments shift these statements;
they are retained here only as the frozen preregistration text. Result claims
use the uninstrumented active-source line numbers.

## Source hypothesis

NEMO's active DINO path is:

- `cfgs/DINO/MY_SRC/dynspg_ts.F90:703-705`: `zhU = e2u * ua_e * zhup2_e`;
- `cfgs/DINO/MY_SRC/dynspg_ts.F90:706-708`: `zhV = e1v * va_e * zhvp2_e`;
- `cfgs/DINO/MY_SRC/usrdef_hgr.F90:98,108,113,118`: `e1v` is evaluated at
  the half-integer Mercator V latitude, and equals `e2v`.

legoeSM forms the corresponding products at
`packages/ocean/legoesm/ocean/dynamics/barotropic_latlon_cgrid.py:851-852`
and consumes the stored V-face zonal width in divergence at
`packages/core/legoesm/grids/operators_latlon_cgrid.py:959-965`. Before the
carried fix, `packages/core/legoesm/grids/latlon.py` built that width at the
arithmetic mean of adjacent T latitudes. The carried implementation instead
uses the true V-face latitude and assembles `dx_v` at `latlon.py:1739-1806` for the
`nemo_isotropic`, variable-meridional-spacing contract.

## Registered outputs and bars

The committed adjudicator
`scripts/validate/ocean_fidelity/dino_1226/split_explicit_momentum_chain_round10.py`
will compare the bound round-9 artifact with a fresh invocation of the
unchanged round-9 scorer at the fixed production commit. It produces:

1. reduction in subrow 9.5 (`zhV`) normalized RMS error;
2. reduction in subrow 9.6 (`zhdiv`) normalized RMS error;
3. reduction in RMS of the pointwise NEMO-minus-legoESM divergence residual at
   each registered wall (`j=1`, `j=197`), read from the exact B arm;
4. the fixed-tree ordered subrow table and first DEBT;
5. the unchanged round-9 Q/F/B verdicts, explicitly retained as historical
   frozen decisions.

The V-face width is **CONFIRMED as the dominant pre-fix flux-composition
owner** iff all four registered, correlated reductions are at least 0.99: subrow 9.5,
subrow 9.6, wall `j=1`, and wall `j=197`.

It is **REFUTED** iff all four reductions are at most 0.10. Every mixed result
is **OPEN_UNRESOLVED**. These are ownership bars on the *pre-fix dominant flux
residual*, not permission to rewrite the frozen nested-arm disposition and not
an AT-BAR criterion for the post-fix rows.

The post-fix ordered table keeps the existing exact arithmetic gates:
pointwise rows require correlation >= `1-1e-9`, mean-absolute ratio within
`1e-6`, and max error/NEMO RMS <= `1e-15`; accumulating rows use `1e-12` for
the last clause. Any remaining first DEBT is the next ordered stop. In
particular, this counterfactual does not assume that the already-measured
`zhvp2_e` depth residual or multiplication-association residuals vanish.

## Controls and provenance

The adjudicator refuses a changed round-9 SHA, changed six-dump binding,
changed NEMO binary/source/bracket binding, dirty or failed scorer controls,
or a post artifact whose package commit is not supplied explicitly. It also
requires equal nonempty session IDs; applies the shared classifier to a
data-derived pre-as-post plant that must REFUTE; and applies it to a one-axis
unfixed plant that must remain OPEN, proving the four-axis conjunction. Both
artifacts, the adjudicated artifact, package commit, backend, fp64 state, and
session ID are SHA-bound.

No GPU, `mpirun`, push, or external write is authorized.
