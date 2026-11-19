# Certified GYRE reverse-mode gradient gate — round 5 receipt

Date: 2026-10-12

Status: **LANDED**. The exact three-file push battery now exits normally after
the certified GYRE one-step row is compiled and measured in a fresh child
process. The unchanged one-step CPU fp64/libm agreement row is CONFIRMED. The
isolated six-step CPU program still aborts inside LLVM section allocation, so
its derivatives and every six-step resource number remain UNMEASURED. No
physics, card configuration, timestep, objective, direction, tolerance,
stabilizer, clipping, guard, or `stop_gradient` changed.

## Provenance and preregistration

The incoming tip was `a6018ed90ff8ce3beeb144a10edda589255054cc`.
Round 5 was preregistered at clean commit `389144e9b`; the external copy is
`gradgate_rounds/round5/preregistration.md`, SHA-256
`a779c9cd78f8369cf2237d1bd6ea7ca525697bc07b104ce0a4dddeaa12f4797a`.

The isolation measurement and exact battery ran at clean commit
`f59610e7a41f5e71188c4568c5ce9c7634f2e074`. The GPU policy-label repair is
clean commit `6053d444b1668fcf3b81fd9335fb850ccaee244e`. All evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round5/`.

The missing `origin/feat/gyre-gradient-twin-2` remote ref was recorded before
work: this clone's local origin no longer advertises it. The immutable reused
commits remain present (`901cfe0d8`, per-step twin and four-field loss;
`e7f3797de`, September TKE square-root guard). Nothing was reconstructed from
memory.

## CPU agreement rows

The program remains the certified `GYRE-zco` card from day 0, production
`LatLonCGridOceanModel._step_jitted`, CPU fp64/libm, four-field normalized
after-state loss, fixed PRNG-0 perturbation and PRNG-42 direction. Bars remain
reverse/forward relative error at most `1e-9` and reverse/central-FD relative
error at most `0.05`, with FD step `1e-3`.

| steps | reverse | forward | central FD | rev/fwd relative | rev/FD relative | verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 | `-3.8849796327038826e-06` | `-3.884979632703889e-06` | `-3.884977393501259e-06` | `1.7442211333597704e-15` | `5.763743532444348e-07` | **CONFIRMED** |
| 6 | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **LLVM abort, exit 134** |

The one-step JSON is
`cpu_agreement_1step_isolated_battery.json`, SHA-256
`64c5088fbfb78efd58acfdadf5afa386711438a51b09109ed30dbb8c3b4a70c2`.
It stamps clean commit, CPU, fp64/libm, float64 state and geometry, the
production entry, and per-step Python composition. This reproduces Round 4's
accepted row; its tiny last-bit movement in the printed reverse/relative error
does not change either registered bar.

The deliberately doubled-VJP plant remains pinned by the direct gate tests and
was not weakened. Round 4's live plant row remains the applicable production
control: forward and FD were unchanged while reverse doubled and exited 1.

## Push-gate process isolation

The pre-change operator battery passed its nine small variants, then died at
the tenth parametrization while compiling the GYRE forward JVP in the same
Python process. The repair at
`tests/ocean/validation/test_step_gradient_matrix.py:65-95,221-240` launches
the existing committed gate in a child process and asserts its return code,
strict JSON, clean commit, status, card, step count, backend, fp64/libm policy,
and production entry. The parent never imports the GYRE gate module.

The isolated row alone passed normally in `1333.67 s`. The exact operator
battery then produced:

```text
52 passed, 1 skipped in 2545.46s (0:42:25)
```

All 17 citation-gate tests, all 25 recipe tests, all nine legacy gradient
variants, and the isolated GYRE one-step row passed. The sole skip is the named
six-step LLVM row before tracing. The log is
`exact_three_file_battery.log`, SHA-256
`579b9755753ce0172aa4cdb1fe55b6a7b2d26edf0f630eb2b8592ee477047157`.
This confirms the working hypothesis for the one-step battery failure:
retaining the nine earlier executables in the same process prevented LLVM from
placing the later large GYRE program. A fresh child avoids that accumulation.

## Six-step CPU compiler result

The unchanged six-step agreement program was also run alone. It completed one
slow `jit_production_step` compilation in `3m7.294513582s`, then aborted during
a later derivative compilation with:

```text
LLVM compilation error: Cannot allocate memory
contiguous_section_memory_manager.cc:101] allocateMappedMemory failed with error: Cannot allocate memory
Failed to satisfy suballocation request for 2396
LLVM ERROR: Unable to allocate section memory!
```

The process exited 134 and wrote no strict JSON. The retained log is
`cpu_agreement_6step.log`, SHA-256
`a73724195d189f0ac94c7f57ff880eca1966619e45075d98c1ca3b3b952ed5e2`.
The request differs from Round 2's `3791`, but the failing allocator and
terminal signature are the same.

This is not host RAM pressure or a process ulimit. Immediately before launch,
the host had 1006 GB total and 961 GB available; data, resident, and virtual
memory limits were unlimited; stack was 8 MB; overcommit mode was `0`
(heuristic); `vm.max_map_count` was `65530`; and a fresh Python process used 54
maps. The environment artifact SHA-256 is
`52d7a02a8e7fe4e07f627aca539ace20a2b09058b1f8ab098314bf240393da3c`.
The failure is therefore classified as the XLA:CPU LLVM JIT section-memory
failure for this very large compiled derivative function, not as a numerical
gradient result.

The instrument already uses the smallest requested directional closure:
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradient_gate.py:383-396`
differentiates one scalar `alpha`, rather than materializing the full input
gradient. It still fails at six steps. Per-step rematerialization also already
wraps every production step. Per-RK-stage rematerialization was not silently
added: the three stages are nested inside the monolithic production
`_step_impl`, with no separate production stage API preserving the whole-step
primal. Creating one would be a physics-source refactor outside this
measurement lane. HLO/code-section size and the thunk-runtime true/false arms
remain UNMEASURED; the CPU campaign reached 95 minutes before resource work,
so repeating two thirty-minute compiler failures would exceed the requested
two-hour cap.

## Memory and wall time

| platform/policy | schedule | 1-step reverse peak/time | 6-step reverse peak/time | 360-step extrapolation |
|---|---|---|---|---|
| CPU fp64/libm | none | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | uniform | UNMEASURED | UNMEASURED | UNMEASURED |
| CPU fp64/libm | binomial | UNMEASURED | UNMEASURED | UNMEASURED |
| GPU 1 fp64/native | none | OPERATOR RUN REQUIRED | OPERATOR RUN REQUIRED | UNMEASURED |
| GPU 1 fp64/native | uniform | OPERATOR RUN REQUIRED | OPERATOR RUN REQUIRED | UNMEASURED |
| GPU 1 fp64/native | binomial | OPERATOR RUN REQUIRED | OPERATOR RUN REQUIRED | UNMEASURED |

The attempted one-step CPU uniform resource row refused before any reverse
pass: `checkpointed_loop` uses `lax.scan`, but the certified cold first step
changes the state pytree from cold to warm (three `None` leaves become fields
or a tuple). JAX requires identical input/output carry structure. No RSS or
wall time is therefore accepted, and no 360-step estimate is fabricated.
The initial incorrect commit-stamp attempt also refused before compilation and
is retained as `cpu_resources_uniform_1step_stamp_refusal.log`.

The resource harness needs an explicit next-round decision: make the cold first
production step its own adjoint segment, then apply `none`, `uniform`, or
`binomial` to the fixed-structure warm steps. **Pick: yes.** This preserves the
campaign's per-step boundary, but it changes the declared schedule partition
and was not silently chosen here.

## GPU operator runner

The committed operator entry point is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.
It pins physical GPU 1, exposes one logical device, selects fp64 with native
XLA transcendental functions, runs rows serially, refuses overwrite or dirty
state, and verifies every JSON carries the exact label
**“GPU, XLA transcendentals — not the certified bit-exact policy.”** Unsupported
GPU/libm and CPU/native pairs now raise at
`nemo_testcase_l2_gyre_gradient_gate.py:84-93` and have direct tests.

The sandbox preflight produced `CUDA_ERROR_NO_DEVICE`, then the required
`REFUSE gyre-gradgate-gpu: command failed at line 22 (exit 1)`. Its log SHA-256
is `aa025d1ca453326175f3a1aab80a6e87f52c55caba6e798e26b5a2ea250e51f5`.
No GPU number is claimed; the operator must run the script from this clean tip.

## Card scope and primal evidence

No VORTEX or TSUNAMI gradient selector was added or run. The GYRE six-step row
is not green, so the user's prerequisite for smaller qualifying component rows
is unmet. Instantiating the current cards confirms that `VORTEX-zco` and
`VORTEX_VEC-zco` both run RK3-WS plus FCT2 tracer stages (and therefore the
CEN2 stage-1/2 tracer statement), while only `VORTEX_VEC-zco` shares GYRE's
vector-invariant momentum family; the withdrawn broad gradient-card request
remains withdrawn.

Round 4 already proved GYRE's ten-step pre/post trajectory byte identity. The
reviewer's requested pre/post short trajectories for both VORTEX cards were
not available from the current post-fix tip without checking out an older
commit, which this lane explicitly forbids. They remain OPEN rather than being
replaced by the existing construction hashes.

## Tests and review

- Exact three-file battery at the isolation commit: **52 passed, 1 skipped**.
- Gradient-gate direct tests after the GPU policy-label change: **10 passed**.
- GPU runner shell syntax: PASS; sandbox refusal: expected named REFUSE.
- Citation gate is included in the exact battery and passed all live controls.

The independent Claude CLI review could not start because this fresh workspace
was not trusted and was terminated without a verdict. GLM is unavailable.
Round 5 is therefore **UNREVIEWED**; the earlier operator gg_r4 review confirms
the underlying science change, not this process-isolation diff.

## OPEN

1. Run the committed GPU script on GPU 1. Agreement, peak device memory, and
   warm reverse wall time for one and six steps under all schedules remain
   unmeasured.
2. Decide whether the cold first production step becomes a separate adjoint
   segment before the warm fixed-pytree checkpoint schedule. Then measure all
   CPU schedules and only afterward evaluate the frozen 360-step formulas.
3. Measure lowered HLO/code-section size against the small variants and test
   XLA thunk-runtime true/false without exceeding a declared CPU budget.
4. Run controlled pre/post short trajectories for `VORTEX-zco` and
   `VORTEX_VEC-zco` when an allowed pre-fix worktree is supplied.
5. Add the SWG card's own gradient gate when that certified card exists.
6. Complete the independent reviewer pair.

## Choice audit

- ASKED: subprocess isolation, exact three-file battery, isolated six-step
  compiler attempt, host-limit record, explicit GPU-native policy and runner,
  GYRE-only gradient scope, receipt, and commits.
- UNASKED: none. No model physics or configuration changed. No compiler flag,
  checkpoint partition, cross-card gradient program, memory estimate, or
  trajectory claim was silently selected.
