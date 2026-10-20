# PREREG — ORCA2 round 319: TKE bottom-BC drag casts, no inert trainables (frozen before measuring)

Base `dd74a18bf08b` (round 318). One model change, gradient-only and
operator-approved: drop the five `float()` casts on the drag coefficients in
the TKE bottom boundary condition (`_tke_bottom_dirichlet`), as round 316 did
in the momentum drag. Everything else is measurement, tests and docs.

Read before freezing (not predictions): rung 10's card sets
`bottom_drag_scheme = nemo_linear` (rate = cd0 * uc0, no velocity),
`kappa_Redi = 1000`, `tke.kappaH_min = 1e-10`, `tke.tke_background = 1e-10`.
Not read before freezing: round 317's detached OMT-3/4 and no-remat result
files; any rung-10 perturbation or N >= 2 isoneutral row.

Memory rule: one process > 100 GB at a time; a guard stops my own process
if available memory < 150 GB.

- P1 forward bits: the step jaxpr is identical base vs fix on all 13 cards
  (rung 0, OMT-1..4, rung 10, GYRE, VORTEX x4, OVERFLOW, LOCK_EXCHANGE;
  the config fields are Python floats, so `float(x)` is the identity), hence
  0 leaf-steps moved. Falsifier: any jaxpr change or moved leaf.
- P2 trace test: `jit(grad)` of a rung-10 one-step loss w.r.t. a traced cd0
  traces with the fix and raises `ConcretizationTypeError` with the casts
  restored. Falsifier: it raises with the fix (another cast blocks cd0), or
  traces without it.
- P3 inert scan, rung 10, one step, each scalar alone x (1 + 1e-3), traced:
  state moves for cd0, uc0, kappa_Redi (isoneutral, top-level copy),
  lateral viscosity scale, TKE minimum (tke_background); NO state moves for
  cd_max, z0, ke0 (not read by nemo_linear) and kappaH_min (round 317: AD
  exactly 0). Falsifier: any scalar on the other side.
- P4 rung-10 isoneutral gradient, T and SSH loss, N = 1, 2, 5, 10, `--remat`:
  finite at every N; reverse = forward mode <= 1e-10 (round 316's scalar
  rows 1e-12); SSH nonzero for N >= 2; FD best over the six-eps sweep
  <= 1e-6 at N = 1, 2; at N = 5, 10 it may miss (PLAUSIBLE: convective
  switches). Falsifier: NaN, a reverse/forward gap > 1e-10.
