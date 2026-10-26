# Preregistration: bn2 native-e3w fix and ordered ZDF continuation

Date: 2026-08-28.  Status at commit: **no round-2 code or measurement run**.

This round accepts the row-2 localization in
`docs/ocean/fidelity/dino_zdf_chain_sweep_result.md`, implements its registered
fix, re-verifies row 2, and then resumes the call table in
`PREREG_zdf_chain_sweep.md` without skipping a live operation.

## Fix contract

There is one static vertical-geometry selector on the coordinate, not separate
TKE/EVD/GM flags and not an environment variable:

- `mesh_reference` (default): preserve NEMO `mesh_mask.nc` raw `gdept_0` and
  `e3w_0`; form the bn2 divisor from the independent reference spacing as
  `e3w_0*(1+r3t)`.  Missing, nonpositive, or shape-incompatible mesh operands
  fail closed.
- `depth_difference` (explicit legacy): retain `diff(live_gdept)` exactly for
  generic/synthetic consumers that deliberately request historical behavior.

The same canonical live `e3w` array must be passed to every `nemo_bn2`
consumer and to the immediately paired `zdf_mxl` multiplication.  The fix is
rejected if only the divisor changes and the multiply still reconstructs a
different spacing.

## Red-capable tests

Before implementation, committed tests must fail for the missing production
surface.  After implementation they must prove:

1. `mesh_reference` is the constructor/kernel default and uses a provided raw
   reference spacing; `depth_difference` reproduces the old bits;
2. selector typos and missing/bad-shape/nonpositive native operands fail;
3. `NemoGrid` reads `e3w_0` with correct halo/axis order;
4. raw `gdept_0`/`e3w_0` and the selector propagate through both partial-cell
   and full-step coordinate wrappers without changing pytree structure;
5. a planted last-bit disagreement between `e3w_0` and `diff(gdept_0)` makes
   the two arms numerically distinguishable;
6. the shared bn2-divide / zdf_mxl-multiply pair cancels the identical supplied
   `e3w` exactly;
7. `jax.jit` succeeds and `jax.grad` with respect to T, S, and eta is finite.

The test record will state the pre-fix failing tests and post-fix passing tests.

## Row-2 acceptance

Re-run `zdf_chain_sweep.py` on `DINO_1226_LANE=d180`, CPU/fp64, with the
production default arm (no probe-side divisor substitution).  Confirm only if:

- `bn2(Nbb)` has 0/9,920 failed wet columns and maximum column error `<=1e-15`;
- all four registered southern focus columns pass the same bar;
- the explicit legacy arm reproduces the accepted baseline
  (`4,630/9,920` failed, max `6.366584806460317e-15`) within literal fp64
  reproducibility;
- perturbation, horizontal-roll, and nonfinite controls fire;
- the exact same canonical live `e3w` is used by the subsequent MLD integral.

Any miss rejects the fix; the sweep does not advance.

## Continuation and stop rule

After row 2 is verified, continue in the already-committed execution order:

1. row 3 `eos_rab/bn2(Nnn)`;
2. row 4 `zdf_sh2`;
3. row 5 bottom drag operands;
4. rows 6-7 `zdf_mxl` index and depth;
5. rows 8 onward: every active `zdf_tke` term, closure `avm/avt`, EVD,
   turbocline, `ldf_slp`, then momentum/tracer implicit solves.

Existing bars remain unchanged: POINTWISE `1e-15`, vertically accumulating or
solve `1e-12`, exact masks/indices exact, aggregate correlation/ratio diagnostic
only.  Every row reports the four focus columns and all wet columns.  Stop at
the first `DIVERGED` row, localize within that row in NEMO evaluation order,
and leave every later live row `UNMEASURED`.

No pattern statistic can promote or demote a row.  No downstream result may be
used to waive an upstream failure.
