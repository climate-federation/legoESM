# Certified GYRE reverse-mode gradient gate — round 3 receipt

Date: 2026-10-11

Status: **LANDED (measurement debt retained)**. The CPU push gate is green and
now records the certified card's one-step reverse derivative as an expected,
named DEBT instead of aborting the suite. A production-path VJP bisect assigns
the first non-finite cotangent to the CEN2 tracer tendency's `e3t(Kmm)`
division on the dry perimeter. Six-step CPU and all GPU resource rows remain
unmeasured.

## Provenance and preregistration

Round 3 was preregistered at clean commit `3e3c49970` before the bisect. The
committed preregistration is
`docs/ocean/fidelity/testcases/nemo_testcases_l2_gyre_gradgate_round3_prereg.md`;
the external copy is `gradgate_rounds/round3/preregistration.md` (SHA-256
`04ff897bfa3c3e90fbe8445df2e7977ab0a3d4e8c422b8dfad55022ccf0602a3`).
It predicted that the one-step reverse row would remain non-finite, required a
finite boundary before a non-finite operand cotangent to assign an owner, and
forbade a guard, configuration change, clipping, or `stop_gradient` in this
round.

All evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round3/`.
The decisive `vjp_statement_bisect.json` has SHA-256
`58d6bb85e3e6382bb1bbadbdba9fafb3b6cacf6d572972eff0b8ed4d8f3eeb99`
and stamps clean commit `bd85e2183b0f82cc90e9e1c758affceb580c66ce`, CPU,
fp64/libm, one step, and production entry
`LatLonCGridOceanModel._step_jitted`. The registered scalar is the T-only
normalized after-state component of the frozen four-field loss; the smaller
closure still reproduces the production reverse NaN.

The first attempt, `vjp_boundary_bisect.json` (SHA-256
`7a9eda2b005b5cc44a049e7a825af3b82b2363b1ca7d37cf488d1c1349714a8d`),
installed taps after the production function had already been traced and
therefore recorded no events. It is retained as **INVALID FOR LOCALIZATION**.
The corrected broad trace is `vjp_boundary_bisect_traced.json` (SHA-256
`6d96a10177bd859e1a4925ef91649bb7ba132ec60b7d3c549c95414059a7a72d`);
only the final statement trace supports the attribution below.

## CPU agreement rows

The unchanged bars are reverse/forward relative error at most `1e-9` and
reverse/central-FD relative error at most `0.05`, with central-FD step `1e-3`.

| steps | model time | reverse | forward | central FD | fwd/FD relative | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | 14,400 s | NaN | `-3.884979632703889e-06` | `-3.884977393501259e-06` | `5.763743549886549e-7` | **DEBT / REFUTED** |
| 6 | 86,400 s | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **CPU LLVM ABORT; CI SKIP** |

The one-step pytest row asserts this exact debt, including the owner named
below. A future finite reverse derivative fails until the expected state is
changed visibly to CONFIRMED. The six-step parametrization skips before
compilation with the recorded LLVM reason, so pytest cannot abort after the
other rows finish.

The legacy `adv_ppm_fct` row also remains strict: reverse mode is required to
be finite and flowing, while the documented finite-difference comparison is
not attempted because its selected tridiagonal solve supplies a custom VJP but
no JVP. No numerical tolerance was relaxed.

## Statement-level VJP owner

The reverse event sequence first observes a finite cotangent at
`_nemo_cen2_tracer_rhs.output` (zero non-finite values), immediately followed
by **3,120 non-finite values** in `_nemo_cen2_tracer_rhs.input[4]`, the
`e3t_kmm` denominator. They occupy all 30 cell levels on exactly 104 horizontal
cells: latitude rows 0 and 21 in all 32 columns, plus longitude columns 0 and
31 in rows 1 through 20. This is precisely the `22 x 32` perimeter; the first
flat index is 0. Later poisoned velocity and tracer cotangents are consequences,
not competing first owners.

legoESM transcribes the horizontal and vertical divisions at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:1779` and
`:1792`, inside `_nemo_cen2_tracer_rhs` at `:1755-1792`. The certified FCT
card executes that CEN2 helper in RK3 stages 1 and 2. The compiled GYRE oracle
proves the active branch:

- `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:295-298` disables FCT before RK3
  stage 3, and `:374-380` dispatches those stages to `tra_adv_cen`.
- `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90:137-149` divides the horizontal
  divergence by `e3t(ji,jj,jk,Kmm)`; `:203-216` performs the corresponding
  vertical divisions.
- `cfgs/GYRE_OMIP_L2_P3_SM/EXP00/namelist_cfg:133-135` selects FCT with
  second-order horizontal and vertical operands.

NEMO evaluates these statements over its owned positive-thickness domain and
halo representation. The cropped legoESM card stores zero `e3t_kmm` on its dry
outer perimeter and masks the tendency after the literal division. The primal
after-state is finite, but the transpose encounters the inactive `0/0`
division. This is the smallest production sub-step closure that reproduces the
reverse NaN.

This result confirms the preregistered prediction that the Round 2 NaN would
remain, but **retracts TKE dissipation as the remaining owner**. The approved
September TKE guard is necessary and primal-bit-identical; it is not sufficient
for the current production card. No new guard was landed here.

## Proposed NEMO-faithful remedy — not implemented

The recommended next experiment is a primal-identity active-cell denominator:
use the real `e3t_kmm` where the tracer cell is active and a benign nonzero
denominator where it is inactive, while explicitly retaining zero inactive
contribution, for both literal divisions. A double-`where` construction should
be used so the inactive transpose never sees the zero divisor. This follows
the compiled NEMO loop domain rather than changing wet-cell physics. It must be
authorized separately and must pass the unchanged one-step/six-step gate plus
the ten-step certified trajectory byte-identity control before acceptance.

## LLVM abort diagnosis

The Round 2 six-step process exited 134 while compiling the Python-unrolled
production graph. The exact decisive log messages are:

```text
LLVM compilation error: Cannot allocate memory
contiguous_section_memory_manager.cc:101] allocateMappedMemory failed with error: Cannot allocate memory
Failed to satisfy suballocation request for 3791
LLVM ERROR: Unable to allocate section memory!
```

The same log first warns of a very slow `jit_production_step` compilation.
This is the known XLA-CPU large-function section-memory failure, not a model
forward or reverse verdict. The failing six-step path already applies
per-step `jax.checkpoint`, so that schedule does **not** avoid the abort.
Checkpointing each RK stage is UNMEASURED and cannot be claimed as an
avoidance. GPU avoidance is also UNMEASURED until the operator script completes.

## Memory, wall time, and GPU rows

| platform/policy | schedule | 1-step reverse peak/time | 6-step reverse peak/time | 360-step extrapolation |
|---|---|---|---|---|
| CPU fp64/libm | none | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | uniform | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | binomial | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | none | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | uniform | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | binomial | UNMEASURED | UNMEASURED | UNMEASURED |

No failed-compilation allocation, process RSS, or estimate is substituted for
a measured reverse-pass peak. Consequently no defensible 60-day/360-step
memory extrapolation exists yet.

The prior operator attempt refused at its first row with the exact message
`RuntimeError: PrecisionPolicy.transcendentals='libm' is CPU-only; got 'gpu'`,
raised at `packages/core/legoesm/core/transcendentals.py:49`. The GPU runner now
passes `PrecisionPolicy.fp64(transcendentals="native")` explicitly; the policy
definition at `packages/core/legoesm/core/precision.py:95-103` identifies this
as JAX/XLA intrinsics rather than the CPU-only libm callbacks. Every resulting
row must therefore be labeled **“GPU, XLA transcendentals — not the certified
bit-exact policy”**. CPU certification remains fp64/libm.

The committed operator entry point is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.
It serially runs one- and six-step agreement and all three checkpoint resource
schedules on GPU 1, refuses overwrite/dirty/wrong-device conditions, and stamps
the selected policy in each JSON. This sandbox still reports
`CUDA_ERROR_NO_DEVICE`; the runner then emits its named
`REFUSE gyre-gradgate-gpu: command failed at line 22 (exit 1)` line.

## Trajectory identity and tests

Round 3 changes only measurement instruments, tests, and the gate expectation;
it changes no model or configuration source. The Round 2 guard control remains
the applicable physics identity proof: before and after the guard, all 73
ten-step trajectory rows had identical `status` and `absolute_max`, and the
residual NPZ files were byte-identical.

- Full CPU gradient matrix: **10 passed, 1 skipped in 1105.12 s**. All nine
  legacy variants pass; the one-step GYRE debt assertion passes; the six-step
  GYRE compiler row is the sole named skip.
- Direct tests for the GYRE gate and new VJP instrument: **10 passed in 3.37 s**.
  The instrument's unsafe-sqrt plant exits nonzero unless a finite output
  cotangent is shown to poison its input, proving that the detector can fail.
- `tests/ocean/unit/test_nemo_recipe.py`: **25 passed in 316.09 s**.
- Citation gate: **PASS**, 274 citations; all seven plants and both controls
  fired as expected.
- GPU runner syntax passed `bash -n`; sandbox execution produced the required
  named REFUSE line before its nonzero exit.

## Review status

The campaign's required independent reviewer pair is **UNREVIEWED** in this
environment. No independent-review claim is made; review remains open before
these numbers are promoted beyond this measurement receipt.

## OPEN

1. **DECISION_NEEDED:** authorize a new measurement round to apply the
   active-cell, double-`where` denominator treatment at both CEN2 divisions,
   then prove certified primal byte identity and rerun the unchanged gate.
   **Pick: yes.**
2. The operator must run
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`
   on GPU 1. GPU agreement, peak memory, wall time, and whether GPU compilation
   avoids the CPU LLVM abort remain unmeasured.
3. CPU resource rows and every 60-day/360-step memory extrapolation remain
   unmeasured. A stage-level checkpoint/compiler discriminator is also open.
4. Add the SWG certified card's own gradient gate when that card exists.
5. Complete the independent adversarial reviewer pair.

## Choice audit

- ASKED: production statement-level VJP bisect, named fail-closed CPU debt,
  six-step compiler skip, explicit GPU-native transcendental policy, operator
  runner, and receipt.
- UNASKED: none. No physics, card configuration, stabilizer, clipping,
  `stop_gradient`, threshold, loss, random direction, or timestep changed.
