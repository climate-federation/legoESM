# Preregistration: row-1.2 vertical seed association, round 17

Date: 2026-08-29. Frozen before measurement. Session
`01a04e34-d1fb-73e0-b25a-177641f0a246`.

## Question and source order

The ordered chain is stopped at the centred barotropic seed: SSH is exact, but
U/V fail the frozen POINTWISE `1e-15` maximum-error bar at only roundoff scale.
The 3-D BEFORE velocities, wet-level counts, and total face thicknesses are
already exact. NEMO constructs the carried seed by multiplying live QCO
`e3u/e3v` by the BEFORE velocity level-by-level, left-accumulating from surface
to bottom, then multiplying by `r1_hu_0/(1+r3u_f)` or its V analogue
(`cfgs/DINO/MY_SRC/dynatf_qco.F90:255-267`). `dynspg_ts.F90:571-579` copies
that carried field without arithmetic.

This round uses only the existing day-180 restart, mesh, r3/seed dumps, and the
bound round-16 recurrence. It reconstructs the exact source expression and
three association controls: live-QCO source-ordered left accumulation, static
weights with the same left accumulation, a vector reduction, and reversed
vertical accumulation. Every result is scored over the registered 9,758 U and
9,868 V wet faces. Input hashes, checkout-local script hash, CPU/fp64 state,
session, populations, and the prior recurrence hash are mandatory.

## Frozen bars and decision

The authoritative seed bar remains POINTWISE `1e-15`: normalized RMS and
maximum absolute error divided by NEMO RMS must both be at or below the bar.

- `OWNED_BY_LEFT_ACCUMULATION` requires the live-QCO source-ordered arm at bar
  for both U and V, while at least one association control is above bar in each
  component. The static-left arm may also pass because the QCO stretch is a
  column scalar; if so, ownership is the vertical multiplication/accumulation
  topology, not the physical stretch.
- `INPUT_OR_TIME_LEVEL_OPEN` if the source-ordered arm is above bar. No
  production change is then permitted; the first differing live weight/time
  level must be dumped under SLOT protocol.
- `ASSOCIATION_NOT_DISCRIMINATING` if all controls also pass. Row 1.2 remains
  open because the proposed operator is not identifiable at the frozen bar.

Two red controls must fire: a one-ULP perturbation at the largest wet seed and
reversed vertical order must be distinguishable from the literal expression
on at least one registered component. No row after 1.2 is promoted by this
probe. If ownership is established, production may add a named NEMO-literal
seed-reduction selector/default with autodiff/JIT tests; rows 1.2--1.4 must
then be replayed before row 2 is opened.

