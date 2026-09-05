# Lane 3b round 21 preregistration — scalar-libm LOG/LOG10/POW

Date: 2026-09-05

Tracker: `climate-federation/legoESM#1699`

Parent: `17693b92997f4c70a508803876251286aa8800d7`

State: **PREREGISTERED; NO ROUND-21 NUMERICAL MEASUREMENT HAS RUN.**

## A. Shared precision-policy extension

The one existing implementation is
`packages/core/legoesm/core/transcendentals.py`: scalar `libm.so.6` callbacks
through `jax.pure_callback`, custom JVPs, CPU-only admission, and policy
dispatch through `PrecisionPolicy.transcendentals`.  This round extends that
module with `log`, `log10`, and binary `pow`; it does not add a second callback
module or a new policy name.  `precision.py` is expected to remain unchanged.

Frozen contracts:

1. `transcendentals="native"` still calls `jnp.log`, `jnp.log10`, and
   `jnp.power`; default policy and all non-selecting cards are unchanged.
2. `transcendentals="libm"` calls the scalar `log`, `log10`, and `pow` symbols
   from `libm.so.6`, once per element, preserving broadcast shape and input
   floating dtype.  Non-CPU execution raises before a claim.
3. JVPs are the analytic real derivatives: `1/x`, `1/(x*ln(10))`, and both
   `y*x**(y-1)` / `x**y*log(x)` partials for pow.  Tangent terms follow the
   scalar libm finiteness/domain result rather than clipping inputs.
4. Tests compare exact uint64 bytes with ctypes scalar-libm values and run
   `jax.test_util.check_grads(order=2)` at positive, finite, smooth inputs.

CONFIRM: eager and JIT libm rows are bit-identical to ctypes for every pinned
input; native remains separately selectable; all three order-2 gradient gates
pass; a poisoned callback changes a scored NCAR row.  Any dtype/shape drift,
non-CPU silent fallback, or missing derivative is REFUTE.

## B. Certified NCAR source sites

The resolved ORCA2 O1 path executes these statement families:

| NEMO identity | source | legoESM site to route |
|---|---|---|
| Goff saturation `LOG10` and base-10 exponentiation | `sbc_phy.F90:645-651` | `_nemo_goff_water_saturation_pressure` |
| pressure iteration exponentiation | `sbc_phy.F90:256,272-277` | `nemo_ncar_pressure_at_height` |
| Exner exponentiation | `sbc_phy.F90:321-337` | `nemo_ncar_theta_exner` |
| unstable momentum stability logs | `sbcblk_algo_ncar.F90:312-326` | `_nemo_ncar_psi_m` |
| unstable scalar stability log | `sbcblk_algo_ncar.F90:350-363` | `_nemo_ncar_psi_h` |
| roughness/neutral-wind log | `sbcblk_algo.F90` `z0_from_Cd` / `UN10_from_CD`, called by `sbcblk_algo_ncar.F90:176-200` | `_nemo_ncar_un10` |
| 10 m height-correction logs | `sbcblk_algo_ncar.F90:169-205` | `nemo_ncar_transfer_coefficients` |

Each call remains inside its existing `nemo_source_round` source-statement
boundary.  Prediction: the scalar-math ORCA2 oracle was built against scalar
glibc, so the NCAR gate remains `0 / 158,292` non-bit rows after routing.  All
18 VERIFIED-row plants still bind.  The unchanged SI3 ice bulk remains
`271,560 / 271,560` bit-identical.

## C. Rule 8/12 cross-card prediction

Every comparison is CPU, binary64, production JIT where the existing gate uses
it, and explicitly selects `transcendentals="libm"`.

- LOCK, OVERFLOW, GYRE, and C1D slab: LOG/LOG10/POW are predicted not to execute
  on their already certified changed boundary; their candidate outputs and
  oracle-relative rows should move by exactly `0` row-scale ulp.
- C1D/ORCA2 RGB shortwave uses policy EXP only, so this extension should move
  no RGB row.
- ORCA2 may execute LOG/EXP/POW through NCAR surface bulk, `zdftke.F90`, or
  `zdfiwm.F90`; the resolved namelist and source call graph are audited before
  attribution.  The first moved row, if any, is named with its first live
  source statement.  A moved downstream row without a live call is a REFUTE.
- Prognostic `uu_b/vv_b` remains GYRE-owned work and is not implemented here.

## D. Remaining-site inventory

After the certified NCAR calls are routed, repository `jnp.log`, `jnp.log10`,
`jnp.power`, and `**` sites under NEMO-identity paths are enumerated.  Sites
not executed by a certified card this round remain named debt; they are not
converted opportunistically.

## E. ASKED / UNASKED

| choice or action | status | preregistered disposition |
|---|---|---|
| extend shared scalar-libm policy to LOG/LOG10/POW | ASKED, User Decision 9 | one canonical module, native default unchanged |
| route the certified NCAR path | ASKED | all live NEMO statement families above |
| BBL defaults `0 / 0.0` | ASKED, User Decision 10 | unchanged; no silent activation |
| add/carry prognostic `uu_b/vv_b` | GYRE-owned User Decision 8 | not implemented here |
| implement ORCA2 multi-category SI3 | decision pending | not started |
| alter non-certified native transcendental sites | UNASKED | inventory only |
| modify shipped NEMO, delete evidence, use GPU/`mpirun`, or push | forbidden | not done |

All measurements are Codex-internal unless a review artifact records reviewer
identity, reviewed commit, and verdict.
