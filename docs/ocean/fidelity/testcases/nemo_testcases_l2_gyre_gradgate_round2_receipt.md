# Certified GYRE reverse-mode gradient gate — round 2 receipt

Date: 2026-10-11

Status: **HELD**. The approved TKE dissipation guard is primal-bit-identical on
the certified ten-step trajectory, but the production one-step reverse
derivative remains NaN. The six-step process aborts during LLVM compilation
before writing a stamped row. GPU rows remain operator work through the
committed fail-closed runner named below.

## Provenance and preregistration

This round retains the frozen Round 1 numerical protocol and bars: CPU
fp64/libm, production `LatLonCGridOceanModel._step_jitted`, a per-step Python
composition, reverse/forward relative error at most `1e-9`, reverse/central-FD
relative error at most `0.05`, and a central-FD step of `1e-3`. A NaN, a zero
derivative, a production/rematerialized primal mismatch, or a broken-VJP plant
that stays green refutes the row.

Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round2/`.
The final one-step row was measured from clean commit
`31fa0a7e9604f2b06980143bf5c7541efd8f46b8`; its JSON is
`cpu_agreement_1step_final.json` (SHA-256
`71645d61b6f38496a25479decc6468e31686b18f3b9f78560188229f1ce8d2b0`).
The six-step attempt ran from clean commit
`68a9d13dfc98adcfb4192b7eabce966b058fba56` and wrote no JSON before abort.

## Pre-existing matrix repair

The untouched tip reproduced nine eager failures in `_variants()`:
`LatLonCGridOceanConfig._replace` rejected the removed top-level
`barotropic_solver` field. The implicit-solver config grouping moved that
field to `config.barotropic.barotropic_solver`; commit `f96b8c9ea` updates the
rigid-lid variant through the nested config. All nine legacy rows then build.

The legacy-only run produced **8 passed, 1 failed**. `adv_ppm_fct` now reaches
the custom-VJP Thomas solve, cannot execute `jax.jvp`, and falls into the old
FD fallback; its measured AD/FD relative error is `6.689326443338824e-2`,
while the same file documents an approximately `1e-1` FCT finite-difference
floor. No threshold was changed in this round. Consequently the requirement
that the entire matrix file be green is not met.

## Approved guard and primal identity

Commit `d0d71f430` ports the user-approved guard from
`e7f3797de`. Its unchanged one-step gate still returned a reverse NaN. This
refuted the claim that guarding only the square-root input was sufficient on
the current lane. The earlier twin branch already contains the discriminator:
masked cells can pair zero TKE with zero mixing length, leaving `0/0` in the
inactive divisor transpose. Commit `31fa0a7e9` therefore reuses the completed
guard and direct zero/zero VJP control from `901cfe0d8`, guarding both inactive
operands at
`packages/ocean/legoesm/ocean/physics/vertical_mixing/tke.py:3512-3523`.

The oracle statement remains `zsqen = SQRT(en)` followed by
`dissl = zsqen / zmxld` in
`/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/MY_SRC/zdftke.F90:832-837`.
The guard changes no finite primal selected by that statement.

The required certified trajectory control confirms that result. Before and
after the final guard, all **73** named rows have identical `status` and
`absolute_max`; the two residual NPZ files are byte-identical with SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
The before JSON is `phase3_gate_before_guard.json`; the final after JSON is
`phase3_gate_after_guard_final.json`. Both fidelity gates exit nonzero because
their existing NEMO-comparison DEBT rows are unchanged; that exit is not a
trajectory-identity failure.

## CPU agreement rows

| steps | model time | reverse | forward | central FD | rev/fwd | rev/FD | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | 14,400 s | NaN | `-3.884979632703889e-06` | `-3.884977393501259e-06` | undefined | undefined | **REFUTED** |
| 6 | 86,400 s | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **ABORTED BEFORE ROW** |

For one step, forward mode and central FD agree to
`5.763743549886549e-7` relative. The reverse-only NaN is therefore confirmed,
but it is no longer attributable to the guarded dissipation statement: the
same numbers survive after both inactive cotangents are directly proven zero.
This retracts Round 1's implication that the cited TKE statement was the sole
poison source.

`JAX_DEBUG_NANS=1` localizes the remaining NaN only to the backward graph of
the executed production JIT at
`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:13210-13218`.
The nested JIT prevents a statement-level stack. A diagnostic
`JAX_DISABLE_JIT=1` run changes the forward arithmetic graph and produces a
forward NaN in the tracer content division; it is therefore not evidence for
the production backward cause and is not used as an attribution.

The six-step process terminates with exit 134 and the decisive messages:
`LLVM compilation error: Cannot allocate memory` and
`LLVM ERROR: Unable to allocate section memory!`. This is the reason behind
the operator-observed `Fatal Python error: Aborted`; it is not reported as a
six-step gradient verdict.

## Memory, wall time, and GPU rows

No successful six-step reverse pass exists, so CPU peak memory, wall time, and
all 360-step (60-day) extrapolations remain **UNMEASURED**. No compile-process
RSS or failed-allocation size is substituted for a reverse-pass peak.

The sandbox cannot open GPU 1. Running the committed operator script reproduces
`CUDA_ERROR_NO_DEVICE` and then prints
`REFUSE gyre-gradgate-gpu: command failed at line 22 (exit 1)`.
The operator entry point is:

`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`

It runs, serially on `CUDA_VISIBLE_DEVICES=1`, the one-step and six-step GPU
agreement rows and the one-step/six-step resource rows for `none`, `uniform`,
and `binomial`. It stamps the clean commit, refuses existing output files, and
writes JSON under the supplied evidence directory.

## Tests and controls

- Guard instrument plus NEMO TKE identity tests: **74 passed in 42.35 s**.
- `tests/ocean/unit/test_nemo_recipe.py`: **25 passed in 331.60 s**.
- Citation gate: **PASS**, 274 citations, no failures; all seven violation
  plants and both unplanted controls fired as expected.
- GPU runner syntax: `bash -n` passed; sandbox preflight emitted the required
  named `REFUSE` line on `CUDA_ERROR_NO_DEVICE`.
- The Round 1 broken-VJP plant remains direct-tested in the 74-test focused
  suite: the unplanted synthetic row passes and the doubled pullback refutes.

## Review status

The required Claude-plus-GLM adversarial review pair is **UNREVIEWED**. No GLM
reviewer is exposed in this environment. Two read-only Claude CLI attempts
produced no review output and were terminated after waiting; neither attempt
is counted as a review. This receipt therefore makes no independent-review
claim, and review remains open before any numerical result is cited beyond
this held measurement record.

## OPEN

1. **DECISION_NEEDED:** authorize a Round 3 statement-level VJP bisect inside
   the production `_step_jitted` graph before any further guard. **Pick: yes.**
2. The operator must run
   `nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`; GPU agreement, device peak
   memory, and GPU wall time are unmeasured here.
3. Obtain a successful six-step CPU reverse pass before computing any 60-day
   memory extrapolation; the current LLVM section-allocation abort is binding.
4. Repair the legacy `adv_ppm_fct` matrix row without weakening its gradient
   bar, then rerun the entire matrix file.
5. Add the SWG certified card's gradient row when that card exists.
6. Complete the required independent Claude and GLM adversarial reviews.

## Choice audit

- ASKED: nested barotropic config repair, approved TKE guard, unchanged
  agreement protocol, GPU-1 runner, and HELD-on-CPU-failure status.
- UNASKED: none. No physics/configuration value, stabilizer, clipping,
  `stop_gradient`, threshold, objective, direction, or timestep was changed.
