# PREREGISTRATION — end-wall split-explicit source

Frozen before any new barotropic measurement on this branch.

## Candidate and existing limit

Candidate: a difference inside legoESM's split-explicit barotropic wall-row
update versus DINO's active `dynspg_ts` path injects the sustained end-wall 2dt
mode. Existing five-state matched-state deposit maps do not assign ownership:
their two-step projection annihilates a state-constant field, and about 95% of
the wall residual is state-constant. They measure injection from a fresh
bridge, not free-run amplification.

## Exact ownership number and frozen bars

The final ownership measurement, once one statement-level mechanism is
selected, is a five-day per-step free-run A/B from the same bridged day-180
state. The primary number is the last-half, per-cell-first zonal-wall 2dt
amplitude ratio `A_lego/A_nemo`; the companion is wall variance share.

- **Control validity:** shipped-card ratio in `[2.60, 3.18]` (registered 2.89
  ±10%) and wall share in `[0.75, 0.95]` (registered 0.85 ±0.10).
- **CONFIRMS ownership:** candidate ratio `<=1.25` and share `<=0.08`.
- **REFUTES ownership:** candidate ratio `>=2.30` and share `>=0.68`.
- Otherwise: **UNRESOLVED**.

The candidate may replace only one measured statement-level DIFF with literal
`dynspg_ts` ordering/arithmetic. It must not substitute oracle state or tune a
coefficient. Its source diff must show one changed mechanism and its day-0
bridge must be bit-identical.

## Registered next discriminator — STOP before a GPU arm

The source audit leaves several live implementation differences: NEMO's
per-substep halo/LBC commit versus legoESM's masked serial array; the temporary
pre-`dyn_zdf` transport-mean versus velocity-mean installation; and legoESM's
post-solver uniform `fix_eta_drift` projection. Source reading does not show
which, if any, first changes a consuming wall-row stencil.

Extend the existing `spg_substep_chain.py`/`substep_traj_compare.py` dump
convention with wind ON to print, for every substep, the first nonzero
NEMO-lego difference at the *first subsequent stencil that consumes committed
values* (face-depth/flux, pressure gradient, or Coriolis), separately on j=1
and j=197. Comparing the physical row immediately before/after a halo write is
not sufficient: the row may be unchanged while the next stencil consumes a
changed halo.

- If the first consuming-stencil divergence is uniquely attributable to one
  alignment DIFF, amend this preregistration with the exact selector and
  invocation **before** a five-day run, then use the frozen ownership bars.
- If no divergence appears, refute that statement-level mechanism and run no
  GPU arm.
- If multiple differences appear together, report `UNRESOLVED`; do not
  combine them.

No executable barotropic candidate selector exists yet, so this registration
deliberately contains no nominal GPU command. No GPU arm runs in this lane.
