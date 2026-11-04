# Certified GYRE reverse-mode gradient gate — round 6 receipt

Date: 2026-10-12

Status: **HELD**. The approved cold-step/warm-loop segmentation is implemented,
the certified one-step CPU gradient and live plant are valid, and the exact CPU
push battery is green. It does not produce an admissible six-step scheduled
gradient row: every checkpoint schedule changes 19 production-state leaves
before differentiation. The largest difference is
`1.2526868431450566e-10` in salinity. The gate refuses those rows before
reverse, forward, finite-difference, memory, or wall-time measurement. No
physics, card configuration, objective, tolerance, stabilizer, clipping,
guard, or `stop_gradient` changed in this round.

## Provenance and preregistration

The incoming lane tip was `8b59d6657337e30ae2d218819ab5fe01e127e740`.
Round 6 was preregistered at clean commit `4b1d2b00d`; the external copy is
`gradgate_rounds/round6/preregistration.md`, SHA-256
`ee52d9f52207ba1ae960082308c497d59537f06be96ba13ef924a2b0cd86d52a`.

The interrupted work was recovered exactly as directed from
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-gradgate-122942952`.
Its six commits are `4b1d2b00d`, `e619f1cd6`, `2b577372b`, `f6b0b4fc2`,
`d981341d5`, and `0bb00dcd0`. The final measurement-instrument commits are
`041231601` and `1dc5e2f67`. Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round6/`.

The requested fetch of `origin/feat/gyre-gradient-twin-2` failed because the
fresh clone's local origin no longer advertises that ref. The immutable commits
remain reachable. The harness reuses the four-field normalized loss and
Decision-49 per-step boundary from `901cfe0d8` and the landed TKE guard from
`e7f3797de`; neither was reconstructed from memory.

## Fixed program and source identity

The gate constructs only `build_nemo_testcase_card("GYRE-zco")`; the fail-closed
card dispatch begins at `nemo_testcase_recipe.py:3351`. Every modeled step
enters production `LatLonCGridOceanModel._step_jitted` at
`ocean_model_latlon_cgrid.py:13219`. The harness wiring and explicit fp64 checks
begin at `nemo_testcase_l2_gyre_gradient_gate.py:143`.

The compiled GYRE oracle disables FCT during RK3 stages 1 and 2 at
`cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:295-298`, then dispatches those
stages to `tra_adv_cen` at `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:370-380`.
The centered horizontal and vertical divisions are the active NEMO statements
at `cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90:137-149,203-216`.
The TKE dissipation statement reused from the twin finding is
`cfgs/DINO/MY_SRC/zdftke.F90:832-837`.

## Segmentation result

`checkpoint_after_cold_step` runs the structure-changing first production step
outside the shared checkpoint loop, then applies `none`, `uniform`, or
`binomial` to the fixed warm carry. Both the cold segment and every warm
iteration call the rematerialized wrapper around production `_step_jitted`;
the three composition entry points are
`nemo_testcase_l2_gyre_gradient_gate.py:194,205,214`. A synthetic direct test
proves the former unsegmented structure-changing loop is rejected and all three
segmented schedules differentiate a known scalar problem.

Preregistered prediction R6-P1 is **REFUTED**. At six steps all three schedules
produce the same 19 mismatching state leaves against the campaign-order Python
production loop. The largest differences include `S=1.2526868431450566e-10`,
`T=7.754508146717853e-11`, `tke_avm=8.274637919303274e-12`,
`eta_rk3_after=2.4753587851353287e-12`, `v=1.8940651998200497e-12`, and
`u=1.133454441415438e-12`. The gate does not accept these values as “close.”
This reproduces the earlier twin branch's scan-versus-per-step last-bit gap in
the warm fixed-pytree loop.

| CPU fp64/libm schedule | six-step production-byte check | reverse | forward | central FD | verdict |
|---|---:|---:|---:|---:|---|
| `none` | 19 leaves differ; max `1.253e-10` | UNMEASURED | UNMEASURED | UNMEASURED | **REFUTED** |
| `uniform` | 19 leaves differ; max `1.253e-10` | UNMEASURED | UNMEASURED | UNMEASURED | **REFUTED** |
| `binomial` | 19 leaves differ; max `1.253e-10` | UNMEASURED | UNMEASURED | UNMEASURED | **REFUTED** |

The strict final JSON SHA-256 values are respectively
`a0766617fdb5787992bbba9f185d553a88e97837cc1e443603cc49e4441ee2fd`,
`f94b38463b245f49c4bf48a1384ff35fbeceb9f433a25d15b6c91d970607b787`,
and `b40db42a27b96528577bbd50a743d31fd36f894d5524e0af09f020cee7cc1d64`.
Every JSON stamps a clean commit, CPU fp64/libm, float64 state and geometry,
schedule, production entry, and exact mismatching leaf paths.

## CPU agreement and cancellation scale

The isolated one-step `uniform` row is CONFIRMED and byte-identical to
production at zero, the registered base perturbation, and both FD endpoints.
The cancellation diagnostics are diagnostic only; the registered relative bars
remain unchanged.

| steps | reverse | forward | central FD | rev/fwd relative | rev/FD relative | `sum(abs(g*d))` | `abs(rev-fwd)/sum(abs(g*d))` | verdict |
|---:|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `-3.8849796327038826e-06` | `-3.884979632703889e-06` | `-3.884977393501259e-06` | `1.7442211333597704e-15` | `5.763743532444348e-07` | `1.813001930161708e-04` | `3.7375931405820424e-17` | **CONFIRMED** |
| 6 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **primal REFUTED** |

The one-step gradient-direction norm product is
`0.0004348526797535058`. The row SHA-256 is
`2a4c2fa303d06f1ca838d198204e1cfbfab16a163eb9ba3daa7ea4c8ed33c395`.

The live doubled-VJP plant wrote a strict REFUTED row: reverse
`-7.769959265407765e-06`, forward `-3.884979632703889e-06`, and rev/forward
relative error `0.4999999999999991`. Its JSON SHA-256 is
`d30766102880261ffbcf15df5c4982f72dd6c2ceb2fe5b696bc004653a5f8e4f`.

The requested largest compiling certified CPU window remains **one step**. A
two-step `uniform` run passed the production-byte check and entered derivative
compilation, then exited 134 with:

```text
LLVM compilation error: Cannot allocate memory
contiguous_section_memory_manager.cc:101] allocateMappedMemory failed with error: Cannot allocate memory
contiguous_section_memory_manager.cc:139] Failed to satisfy suballocation request for 399
LLVM ERROR: Unable to allocate section memory!
```

Its retained log SHA-256 is
`c630b7ed710c7f7fe54c66a3e5fc31f65a57c72bbcbd866fd70ea44471f94d21`.
This is the same XLA:CPU LLVM JIT section-memory failure as rounds 2 and 5,
with a smaller request (`399` versus `2396`/`3791`), not a numerical gradient
verdict. Round 5 established that host RAM, ordinary process limits, and
overcommit pressure do not own this failure.

## CPU memory and wall time

Peak host RSS is sampled only during the second, warmed, blocked reverse pass.
The cold compile-plus-reverse is outside the sampler. All three one-step rows
have finite, nonzero gradients and at least 26 in-pass samples.

| CPU fp64/libm schedule | warmed reverse wall time | sampled baseline RSS | peak RSS | increment |
|---|---:|---:|---:|---:|
| `none` | `1.349639606 s` | `23289.809 MiB` | `23532.531 MiB` | `242.723 MiB` |
| `uniform` | `1.378157441 s` | `23415.246 MiB` | `23654.977 MiB` | `239.730 MiB` |
| `binomial` | `1.241333757 s` | `23510.391 MiB` | `23695.855 MiB` | `185.465 MiB` |

The resource JSON SHA-256 values are
`d0cc1b227187c60695f3700c41be02f0efdd21642aea9a004df8881da7825477`,
`e02069ee8a96026bbf9bf5af035347b9971c1395430afe3a4af77dfd8ed9add5`,
and `922cedaeb649b424fc4c882d777967540a4d0097bd8d90a4ce9b248c3e5928b7`.

No valid six-step CPU resource row exists because the production-byte gate
refuses first. CPU six-step peak and wall time are therefore UNMEASURED.

## GPU operator result

The Round 5 operator JSON appeared during this round. It is from clean commit
`8b59d6657`, physical GPU 1 exposed as `cuda:0`, fp64, and explicitly uses
`PrecisionPolicy.fp64(transcendentals="native")`. These rows are labeled
**GPU, XLA transcendentals — not the certified bit-exact policy**.

| GPU fp64/native steps | reverse | forward | central FD | rev/fwd relative | absolute rev/fwd | rev/FD relative | verdict |
|---:|---:|---:|---:|---:|---:|---:|---|
| 1 | `-3.884979632702670e-06` | `-3.884979632702686e-06` | `-3.884977393500650e-06` | `4.142525191730739e-15` | `1.609e-20` | `5.763741977909062e-07` | **CONFIRMED** |
| 6 | `-8.185163817998039e-08` | `-8.185163024909660e-08` | `-8.134481958857320e-08` | `9.689340321095861e-08` | `7.931e-15` | `0.006191917506803739` | **REFUTED** (`>1e-9`) |

The JSON SHA-256 values are
`e1ad50421cc6066f1dc6bdf3d90244cea29b0583d1222431a2d3485a4544e1e6`
and `fdfa79b61ebaba3a248c0354dc6d654074adc83fa15c189023f0ab54fb3d9276`.
The six-step row is reported as OFF-BAR BY THE CURRENT BAR. Its cancellation
scale is not present in the Round 5 schema, so no scale-aware conclusion is
claimed.

The committed operator runner now uses the reused Decision-49 per-step Python
composition for GPU agreement, adds `sum(abs(g*d))`, continues after an
off-bar row, disables allocator preallocation for resource sampling, and then
runs all three schedule resource rows. In this sandbox it refuses before the
first row with `CUDA_ERROR_NO_DEVICE`; the retained log SHA-256 is
`aa025d1ca453326175f3a1aab80a6e87f52c55caba6e798e26b5a2ea250e51f5`.

## Projections

The preregistered 60-step and 360-step formulas remain encoded and directly
tested. They require a valid six-step sampled baseline and peak for the same
schedule. No such CPU or GPU resource JSON exists, so no numeric projection is
fabricated.

| platform/policy | schedule | 6-step peak/time | 60-step projection | 360-step projection |
|---|---|---|---|---|
| CPU fp64/libm | `none` | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | `uniform` | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | `binomial` | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | `none` | OPERATOR RERUN REQUIRED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | `uniform` | OPERATOR RERUN REQUIRED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | `binomial` | OPERATOR RERUN REQUIRED | UNMEASURED | UNMEASURED |

## Tests and gates

- Direct gradient-gate controls: **31 passed in 3.94 s**. These include the
  structure-changing-loop plant, all three segmented scalar gradients,
  full-gradient directional/cancellation identities, doubled-VJP refusal,
  byte-level signed-zero detection, projection formulas, policy dispatch,
  sampler failure propagation, and evidence overwrite/provenance refusals.
- Exact push battery: **52 passed, 1 skipped in 1501.38 s**. The skip is the
  named six-step CPU compiler row; the citation tests, recipe tests, nine
  legacy gradient variants, and isolated one-step GYRE row pass. Log SHA-256:
  `918c8556b4a95b46e456d95a7a0ff10e65a4708a7387ca6169873384595aa3e0`.
- Standalone default citation gate: **PASS**, 274 citations and zero failures
  at clean commit `1dc5e2f67`. The receipt-specific rerun also passes: 8
  citations, zero unmapped citations, zero failures, and every planted
  self-test fired at clean commit `0e079ad81`; JSON SHA-256
  `a7244d9094c9eb79c066038e4a216f4e98a432b0e20e9b2e50d48271748f8657`.

## Retractions, OPEN, and decision

R6-P1 (“all segmented schedules remain byte-identical”) and R6-P2 (“all
six-step CPU rows compile and agree”) are retained as **REFUTED** predictions.
The smaller LLVM allocation request does not establish that segmentation cures
the compiler failure. No six-step scheduled CPU derivative or resource number
exists.

OPEN:

1. Operator runs
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`
   on physical GPU 1. The script records the native-transcendental policy,
   continues after the six-step off-bar row, and retains every valid JSON.
2. Use the operator rerun's six-step `sum(abs(g*d))` before deciding whether the
   GPU-only relative-bar failure is cancellation/reordering. No bar change is
   proposed from the present evidence.
3. Replace the warm `lax.scan` adjoint with explicit per-production-step VJP
   composition plus checkpoint-managed state replay. **Pick: yes.** This keeps
   the mandated Python per-step primal; accepting the `1.25e-10` drift or
   relaxing the byte gate is not recommended.
4. `VORTEX-zco` and `VORTEX_VEC-zco` are the two dispatched RK3 cards that
   execute the CEN2 stage statement. Their controlled pre/post trajectory SHA
   pairs require an allowed pre-fix producer checkout; construction hashes are
   not relabeled as trajectory evidence. No gradient row for either card was
   added or run.
5. Add the SWG card's own gradient gate when that certified card exists.

## Choice audit

- ASKED: cold-step/warm-loop segmentation, schedule-specific CPU attempts,
  largest compiling window, cancellation-scale diagnostic, Round 5 GPU
  evidence ingestion, operator GPU runner, projections only from valid
  resource rows, exact push battery, receipt, and commits.
- UNASKED: none. No physics, card, precision, tolerance, schedule bar, or
  production-byte requirement changed. Missing resource rows and projections
  remain explicitly UNMEASURED.
