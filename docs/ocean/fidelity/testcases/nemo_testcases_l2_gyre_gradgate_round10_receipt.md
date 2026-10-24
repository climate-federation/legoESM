# Certified GYRE reverse-mode gradient gate — round 10 receipt

Date: 2026-10-24

Status: **HELD pending GPU acquisition**.  The Round 9 GPU-only primal failure
is localized and repaired in the explicit local AD composition.  The certified
CPU one-step row is byte-primal-identical and passes the unchanged derivative
bars; the doubled-pullback plant flips it; and the ten-step production
trajectory is byte-identical to Round 9.  The corrected GPU 1 one-, six-, 60-,
and 360-step agreement rows and all nine memory/time rows remain UNMEASURED.
No physics, card configuration, stabilizer, guard, clipping, stop-gradient,
loss, perturbation, finite-difference step, or bar changed.

## Frozen protocol and provenance

Round 10 was preregistered at clean commit `52e7e18ee`.  The implementation
producer is clean commit `91a5fe9440bc26a599944dd59a4fd1f835af7d27`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round10/`.
CPU uses the certified fp64/libm policy.  GPU acquisition selects fp64/native
and keeps the label **GPU, XLA transcendentals — not the certified bit-exact
policy**.  The reverse/forward bar remains `1e-9`; reverse/central-FD remains
`0.05`.

The only card is the `GYRE-zco` dispatch at
`nemo_testcase_recipe.py:3351`.  Every forward step enters production
`_step_jitted` at `ocean_model_latlon_cgrid.py:13219`.

## Round 9 GPU failure localization

The operator ran clean commit `2c62c8e57` on physical GPU 1.  Ordinary
production, base and both FD perturbation rollouts, and the JVP boundary were
byte-identical.  The one-step VJP boundary differed on exactly 15 leaves:

| leaf | maximum absolute difference |
|---|---:|
| `u` | `6.938893903907228e-18` |
| `v` | `6.938893903907228e-18` |
| `T` | `7.105427357601002e-15` |
| `S` | `7.105427357601002e-15` |
| `eta` | `4.336808689942018e-18` |
| `w` | `5.300573365239801e-21` |
| `uu_b` | `1.6263032587282567e-19` |
| `vv_b` | `1.6263032587282567e-19` |
| `bt_hist[0..5]` | `1.3552527156068805e-19` to `1.6432439176733427e-19` |
| `eta_rk3_after` | `8.673617379884035e-18` |

The first changed pytree leaf was `u`; the largest changes were `T` and `S`.
All leaves remained float64 and both computations used the same native-XLA
transcendental policy.  Dtype conversion and a different transcendental route
are therefore REFUTED owners.  The raw VJP replay differed on 18 leaves and
the VJP boundary shared 15 of those differences; the JVP replay differed on
seven leaves while its boundary remained exact.

The exact defective operation was not an ocean statement.  The compiled
per-step VJP returned the primal reconstructed by `jax.vjp` of the large step.
That is a separately lowered program, not the already materialized production
buffer.  The optimization barriers stopped cross-boundary motion but could not
require GPU reverse lowering to reproduce the standalone production kernel's
last-bit evaluation order.  This explains the backend-specific result without
claiming an unmeasured individual HLO fusion as owner.

RETRACTED: Round 9's claim that the barriers prevent the reverse-lowered
production arm from changing is valid for its CPU measurement only.  The GPU
artifacts refute it as a cross-backend guarantee.

## Materialized production boundary

The explicit transforms at
`nemo_testcase_l2_gyre_gradient_gate.py:456` now take the separately dispatched,
blocked production after-state as an argument.  The unchanged step replay
supplies only its pullback or tangent.  The production after-state is returned
and propagated as the local primal; the raw replay remains recorded for
diagnosis.  The explicit reverse/forward composition at
`nemo_testcase_l2_gyre_gradient_gate.py:562` passes the stored production state
into every per-step transform.

A hostile direct control makes the replay differ by one ULP.  Eager/JIT VJP
and JVP boundaries preserve the supplied production bits, while selecting the
replay as a planted primal changes those bits.  The plant therefore proves the
boundary check can fail; it does not compare a state with itself without a
competing arm.

## Certified CPU one-step result

| row | reverse | forward | central FD | rev/fwd relative | rev/FD relative | boundary bits | status |
|---|---:|---:|---:|---:|---:|---|---|
| one step | `-3.884979632703884e-06` | `-3.884979632703880e-06` | `-3.884977393501259e-06` | `1.090138208349858e-15` | `5.763743536804898e-07` | VJP/JVP exact | **CONFIRMED** |
| doubled-VJP plant | `-7.769959265407769e-06` | `-3.884979632703880e-06` | `-3.884977393501259e-06` | `0.5000000000000006` | `0.5000002881871768` | VJP/JVP exact | **REFUTED** |

The final row's `sum(abs(g*d))` is `1.8130019301617077e-04`,
`norm(g)*norm(d)` is `4.348526797535058e-04`, and its scale-aware discrepancy
is `2.3359957128637767e-17`.  The final JSON SHA-256 is
`17d93af39b7ed287d30be0052e8a1ea3ff3526ab034fd14bf67156c3705fe45e`;
the plant JSON SHA-256 is
`05d3da63df0cf3e7f18a8ab2b28986284bd641c4ee616efaac5eb6a7300aae6c`.
P1 and P2 are CONFIRMED.

## Forward trajectory identity

The ten-step comparison against the Round 9 post-boundary trajectory passed
all 70 rows with zero changed cells and zero oracle-residual worsening ULPs.
The first-over-bar record remains step 3 for `T`, `S`, `u`, `v`, and `ssh`.
The new and Round 9 residual NPZs both have SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
The inherited NEMO-relative trajectory status remains DEBT; the controlled
before/after result is PASS.  P3 is CONFIRMED.

## GPU acquisition and remaining training rows

The corrected entrypoint is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_round10_run.sh`.
It now creates its evidence directory before acquisition, pins physical GPU 1,
selects fp64/native, disables preallocation, refuses the platform allocator,
and runs agreement at 1/6/60/360 steps followed by `none`, `uniform`, and
`binomial` resource rows at 6/60/360 steps.  It retains the existing
classification, provenance, memory-ordering, time-scaling, and planted-failure
checks and exits nonzero unless every required outcome is present.

Run from the clean committed clone root:

```bash
scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_round10_run.sh
```

The operator must write to
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round10/operator_rows/`.
Until that run completes, P4/P5, six/60/360-step gradients, all GPU peak-memory
and warm-time values, and the 60-day training cost remain **UNMEASURED**.  No
Round 9 refused diagnostic and no projection is promoted in their place.

## Tests and gates

- Direct instrument controls: **52 passed in 8.41 s**.  They include the
  hostile one-ULP materialized-primal control, replay-primal plant, doubled-VJP
  plant, directory creation ordering, runner continuation, allocator refusal,
  and memory/time classifiers.
- Certified CPU one-step: CONFIRMED; clean producer stamp rechecked after the
  measurement.
- Doubled-VJP certified plant: REFUTED with expected process exit 1; clean
  producer stamp rechecked.
- Ten-step trajectory comparison: PASS, 70 rows, zero ULP worsening.
- Combined direct controls and `tests/ocean/unit/test_nemo_recipe.py`:
  **77 passed in 356.65 s**.
- Receipt citation gate: PASS with four citations and no unmapped or failing
  anchors.  Shifting `nemo_testcase_l2_gyre_gradient_gate.py:456` by two lines
  produces SYMBOL-NOT-AT-LINE and expected exit 1.

## OPEN and choice audit

OPEN:

1. Operator GPU 1 acquisition of one-, six-, 60-, and 360-step agreement.
2. Operator GPU 1 acquisition of nine checkpoint-schedule memory/time rows.
3. Admission of multi-day gradients to online closure training only after the
   unchanged byte and agreement gates pass.

DECISION_NEEDED: NONE.  The user explicitly approved a primal-preserving local
AD boundary.  No unasked scientific or configuration choice was made.
