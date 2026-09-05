# SI3 lane 3b round 19 preregistration — integration convergence

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `8b224b0b27605e5435089640c364ac745a35b6e5`

Integration source:
`refs/remotes/origin/fidelity/nemo-testcases-l2-gyre-reconciled` at
`03c6e8d96ff7f69207abdee43ec28e223d083f4e`; merge base
`45db99b404ffee65e34598277274238a4bf77239`.

State: **PREREGISTERED; NO ROUND-19 MERGE OR NUMERICAL MEASUREMENT YET.**

## A. Merge identity and conflict rule

The merge keeps the canonical integration implementation for shared ocean and
core numerics, then reapplies only lane-3b-owned C1D/SI3 exchange, coupled slab,
NCAR, gates, and receipts.  The already byte-identical `precision.py`,
`transcendentals.py`, and `source_rounding.py` are retained unchanged.  Every
conflicted file and hunk will be enumerated in the receipt with the chosen
side and discarded behavior; no bulk `ours`/`theirs` resolution is evidence.

The lane-3b Kmm barotropic reconstruction introduced in round 17 is predicted
to be absent after resolution.  NEMO carries prognostic `uu_b/vv_b` from
`dynspg_ts.F90:862,890`; `restart.F90:181,313` stores/loads them and
`restart.F90:316` reconstructs them only as a missing-field fallback.  A
re-derivation from the 3-D state cannot be the bit identity.  Its old coverage
row will become **SUPERSEDED_PENDING_PROGNOSTIC_STATE**, owned by GYRE.

The complete momentum-advection compositions remain:

- LOCK/OVERFLOW: flux form + UP3 + `nemo_up3`;
- C1D coupled slab: the complete OFF program.

This follows `dynadv.F90:35-46,78-90,128-134`; the pairing validator remains
only if the merged cards satisfy those complete programs.

## B. Revert of the non-NEMO dry-face stabilizer

Round-18 commit `a200b4bd4a6` replaced NEMO's stage mask multiplication with
`jnp.where`.  NEMO uses multiplication by `umask`/`vmask` at
`stprk3_stg.F90:367,375,382`; the `where` changes dry signed-zero bits and
masks a bad producer.  The preregistered change is a literal revert to
multiplication after the integration merge.

LOCK prediction: the lane-1 UP3 transport-sign fix in `60d0c542065a` moves
stage-2 normalized error from `9.76564494e-11` to approximately
`2.3e-17`, AT_BAR.  Confirmation requires the stage-2 row to be at `1e-15`
and the planted row to exit nonzero; a merely smaller debt refutes closure.

If OVERFLOW remains non-finite, the ordered stop is the first non-finite
operand at the stage update.  A producer fix is eligible here only if it lies
inside slab/ice/exchange.  A shared WS-RK3 owner is registered to GYRE without
another mask or stabilizer.  The stage-transport `rDt` clock discrepancy
(`stprk3_stg.F90:123-124`; `sshwzv.F90:334-335`) is explicitly GYRE-owned and
will not be changed here.

## C. Frozen cross-card register

After the merge and focused tests, the following production-JIT CPU/fp64
comparisons are frozen:

1. LOCK and OVERFLOW stage sweeps and kt=1..10 trajectories;
2. GYRE Oracle-V2 kt=1..10 production-JIT register;
3. C1D coupled-slab full-year gate;
4. C1D SI3 thermodynamics, exchange, bulk-ice and stream-schema gates;
5. ORCA2 O1 NCAR bulk gate.

Each moving row is scored cellwise against its oracle and compared with its
last certified value using row-scale ULP
`spacing(max(max(abs(oracle_row)),1))`.  The register reports AT_BAR separately
from bit identity and preserves each first-over-bar boundary.  Any shared
movement is owned by GYRE unless a lane-3b-only boundary is demonstrated.

The NCAR prediction remains `0 / 158,292` non-bit source-owned rows.  Its
coverage count is **20 registered frame-1 fields: 18 VERIFIED computations +
2 WAIVED source-unowned writer slots**.  All row plants must bind.  Native-XLA
LOG/LOG10/POW remains a machine-stack hazard rather than a portable bit claim.

## D. One bulk implementation

The backward-compatibility facade
`packages/ocean/legoesm/ocean/bulk_flux_omip.py` will be removed.  Its three
`omip2_applicator.py` calls, validation scripts, and tests will import the sole
shared implementation from `legoesm.core.bulk_flux`.  No second NCAR formula
or scalar-libm LOG/LOG10/POW implementation will be introduced.

## ASKED / UNASKED

| action | state | disposition |
|---|---|---|
| merge canonical lane-1/GYRE integration at `03c6e8d96ff7` | ASKED | preregistered with per-conflict ledger |
| discard lane-3b's reconstructed Kmm seed | ASKED | preregistered as SUPERSEDED |
| revert `where` masks to NEMO multiply | ASKED | preregistered |
| run full fidelity tests and six gate families | ASKED | frozen above |
| delete NCAR facade and update call sites | ASKED | preregistered |
| extend libm to LOG/LOG10/POW | UNASKED, decision pending | forbidden this round |
| mask a remaining OVERFLOW NaN | UNASKED and source-refuted | forbidden |
| repair shared GYRE stage clock | UNASKED, other-lane owner | record only if reached |
| delete retained data roots or diagnostic trees | UNASKED | flag only |
| edit shipped NEMO, use GPU, push, commit large runtime artifacts | UNASKED | forbidden |

## CONFIRMED / PLAUSIBLE

**CONFIRMED before measurement:** NEMO selector semantics; the stale-base
provenance of the round-18 LOCK row; the presence of lane-1 `60d0c542065a` and
GYRE `c83f73c23ff8` on the integration line; NEMO's prognostic barotropic-state
contract; and the O1 schema count of 20.  **PLAUSIBLE / UNMEASURED:** every
post-merge numerical prediction, including the expected LOCK movement and
NCAR retention.  No round-19 numerical result is claimed here.
