# Certified GYRE reverse-mode gradient gate — round 8 receipt

Date: 2026-10-12

Status: **HELD** on the production-step transform byte gate and pending the
operator's GPU multi-step and resource acquisition; the CPU and GPU acquisition
instruments are landed fail-closed.
An independent review found that the one-step transformed VJP and JVP replay
primals were not byte-identical even though their derivatives agreed. The byte
gate is now executable: the CPU one-step row is **UNMEASURED**, its top-level
derivative fields are null, and the agreeing triplet is retained only as
refused diagnostic evidence. The certified ten-step production trajectory is
scientifically byte-identical to Round 7. CPU multi-step reverse passes remain
UNMEASURED because the operator prohibited another attempt after the isolated
six-step XLA:CPU section-memory abort. GPU six/60/360-step and resource rows
remain OPEN for the committed GPU-1 operator runner. No physics, card setting,
timestep, objective, tolerance, stabilizer, clipping, guard, or
`stop_gradient` changed.

## Provenance and frozen protocol

The incoming lane tip was
`f04f9f1d8b3a19968113ab66746866d46a4acdd9`. Round 8 was preregistered at clean
commit `307100d112892d244ea8300e24c21fb214a92760`; its external copy has SHA-256
`f94962bf44740d97c7cd256aef87a487e4fc287a69270d1b5b2babce329a9855`.
Implementation commit `575f4d276be3d5583daa101508c85306cd0b8f1a` produced both
pre-review CPU diagnostic artifacts from a clean tree and rechecked that commit
after each measurement. Review-repair commit
`730909660e96e0a5f8f690dbf13c254f05d64428` produced the binding UNMEASURED
one-step JSON from a clean tree with `verified_after_measurement=true`.
Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round8/`.

The only card remains the fail-closed GYRE dispatch at
`nemo_testcase_recipe.py:3351`; every modeled step enters the production JIT at
`ocean_model_latlon_cgrid.py:13219`. CPU uses the certified fp64/libm policy.
GPU rows must use fp64/native and retain the label **GPU, XLA transcendentals —
not the certified bit-exact policy**. The reverse/forward bar remains `1e-9`,
the reverse/central-FD bar remains `0.05`, and the four-field normalized loss,
PRNG keys, perturbation, direction, and FD step remain unchanged.

## Executable classification repair

The Round 6 review blocker is closed in executable code. A row refused before
derivatives exist now starts from an explicit UNMEASURED record at
`nemo_testcase_l2_gyre_gradient_gate.py:719`: reverse, forward, FD, and their
errors are null. A derivative that was actually evaluated as NaN still goes
through the ordinary agreement classifier and remains REFUTED. Direct controls
exercise both cases, and the CLI exits nonzero for either non-CONFIRMED status.

Review then exposed a second fail-closed gap. The transformed production-step
primal checks were recorded but did not affect the verdict. The gate at
`nemo_testcase_l2_gyre_gradient_gate.py:514` now requires every VJP and JVP
replay leaf to be byte-identical. A planted mismatch flips a nominally
CONFIRMED row to UNMEASURED, nulls every top-level derivative field, preserves
the values under `refused_derivative_evidence`, and produces a nonzero CLI
exit. The live CPU row exercises that refusal.

R8-P1 is **CONFIRMED**. The GPU runner's self-test also proves that CONFIRMED,
REFUTED, UNMEASURED, VACUOUS, and RETRACTED rows are all recognized and that a
classified non-CONFIRMED row does not prevent later acquisition rows.

## Cold-step segmentation and checkpoint schedules

The operator's failed Round 7 resource rows showed that the first production
step changes the carry pytree by materializing TKE, barotropic-history, and RK3
leaves. The resource rollout now executes model step 1 before the shared
checkpoint loop, seeds that loop with `kt=1`, and schedules only steps 2..N at
`nemo_testcase_l2_gyre_gradient_gate.py:277,298,306`. One step returns the cold
production result directly; zero steps return the entry state.

A direct GYRE-card control builds the real certified context, verifies its TKE
leaf is initially absent, plants a cold structure change, and makes the mocked
checkpoint loop reject a cold carry. The repaired rollout calls model steps
`1,2,3,4,5,6` exactly and hands the warm state to the loop. This control fails
if step 1 is moved back inside the schedule. The live six-step schedule primals
and resource rows are not claimed here: they require the operator GPU run.

R8-P2 is **CONFIRMED for structure and ordering; GPU numerical rows OPEN**.

## What the CPU compiler was actually compiling

The incoming agreement path did not wrap the entire six-step rollout and its
transpose in one outer JIT. It was already a Python loop around production
steps, but it constructed a fresh VJP and JVP closure for every static `kt`.
That meant six distinct transform closures and repeated warm-state
specializations.

Round 8 constructs one dynamic-time step VJP and one dynamic-time step JVP at
`nemo_testcase_l2_gyre_gradient_gate.py:456`, then reuses them in the explicit
production-boundary loop at
`nemo_testcase_l2_gyre_gradient_gate.py:562`. The one-step JSON records one
cold-state specialization, one VJP invocation, one JVP invocation, and
`whole_window_transform_jitted=false`. A direct synthetic trace control over
three steps observes four traces total: cold plus warm for the VJP and cold
plus warm for the JVP. Thus the structural compile footprint is:

| window | distinct state-pytree programs per transform | VJP/JVP invocations | whole-window transform |
|---:|---:|---:|---|
| 1 step | 1 cold | 1 / 1 | no |
| 6 steps | 2 (cold + warm) | 6 / 6 | no |
| 60 steps | 2 (cold + warm) | 60 / 60 | no |
| 360 steps | 2 (cold + warm) | 360 / 360 | no |

This table is a measured trace/cache structure, not LLVM object-code bytes.
The standalone CPU control log names the compiled module
`jit_step_vjp` and reports a successful `7m50.938565134s` compile; the live
plant reports `7m9.586543677s`. The prior operator six-step attempt at
`f04f9f1d8` ran for 3837 seconds and then emitted:

```text
execution_engine.cc:54] LLVM compilation error: Cannot allocate memory
contiguous_section_memory_manager.cc:101] allocateMappedMemory failed with error: Cannot allocate memory
LLVM ERROR: Unable to allocate section memory!
```

The retained operator log is
`/home/dbalwada/.claude/jobs/b3690d5d/tmp/gyre_sixstep_cpu.log`. It does not
name which per-step closure was compiling at the abort, so ownership by the
warm specialization remains PLAUSIBLE, not CONFIRMED. The current log does
confirm that even the successful certified one-step row compiles a very large
one-step VJP; it is not evidence of a whole-window transpose.

The limit audit in `cpu_memory_limits.log` records 1006 GiB total and 965 GiB
available, unlimited virtual/data/RSS limits, an 8 MiB stack, and Linux
overcommit mode 0. Therefore ordinary RAM pressure and process ulimits are
REFUTED as owners. The confirmed boundary is XLA:CPU's LLVM section-memory
manager for the large compiled transform. No XLA flag, stage split, or smaller
objective was selected in this measurement lane.

## Certified CPU byte-gate refusal and diagnostic agreement

| CPU fp64/libm row | reverse | forward | central FD | rev/fwd relative | rev/FD relative | sum(abs(g*d)) | scale-aware | verdict |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| 1 step, binding row | null | null | null | null | null | null | null | **UNMEASURED** |
| refused derivative diagnostic | `-3.884979632703884e-06` | `-3.884979632703880e-06` | `-3.884977393501259e-06` | `1.090138208349858e-15` | `5.763743536804898e-07` | `1.8130019301617077e-04` | `2.3359957128637767e-17` | not a verdict |
| pre-review doubled-VJP diagnostic | `-7.769959265407769e-06` | `-3.884979632703880e-06` | `-3.884977393501259e-06` | `0.5000000000000006` | `0.5000002881871768` | `3.6260038603234154e-04` | `0.010714218137532193` | **REFUTED before byte gate** |
| 6 steps | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | UNMEASURED | **CPU COMPILER LIMIT** |

The binding clean-producer JSON has SHA-256
`a13e1639fa9d24fce37a5a4e77ad1ae2d38f124f42e94495b31bfd8ea8205dbf`.
The VJP replay differs on 18 leaves and the JVP replay on 7; both have maximum
absolute difference `1.4210854715202004e-14` in salinity. The direct
base/FD production primals remain byte-identical, but that does not satisfy the
registered transformed-primal byte gate. The earlier one-step and plant JSONs
remain evidence of derivative agreement and plant sensitivity before the gate
was made binding; they are not current verdict artifacts. R8-P4's trajectory
prediction is CONFIRMED, but its one-step agreement prediction is
**REFUTED** by the byte gate.

## Production trajectory identity

The current ten-step trajectory gate and the retained Round 7 control differ
only in `artifacts`, `oracle_relative_residual_artifact`, and `worktree` path or
provenance fields. Every scientific top-level key is equal. Both residual NPZs
have SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
The current JSON has SHA-256
`0bb6626c956ca8f130d3e3af4b6726fb923cf8e8cd71b7a763418747a83b68c1`.
The gate's nonzero exit is its inherited NEMO fidelity-debt verdict; the
before/after scientific comparison is exactly equal. R8-P4's trajectory
prediction is **CONFIRMED**.

## GPU carry-forward, cancellation discriminator, and runner

No GPU is available in this sandbox. The latest measured GPU evidence remains
the operator's Round 5 fp64/native rows:

| GPU 1 fp64/native row | reverse | forward | central FD | rev/fwd relative | rev/FD relative | current-bar verdict |
|---:|---:|---:|---:|---:|---:|---|
| 1 step | `-3.884979632702670e-06` | `-3.884979632702686e-06` | `-3.884977393500650e-06` | `4.142525191730739e-15` | `5.763741977909062e-07` | **CONFIRMED** |
| 6 steps | `-8.185163817998039e-08` | `-8.185163024909660e-08` | `-8.134481958857320e-08` | `9.689340321095861e-08` | `0.006191917506803739` | **REFUTED** |

The six-step absolute reverse/forward difference is about `7.93e-15`, but the
old artifact predates `sum(abs(g*d))`; its scale-aware discrepancy remains
UNMEASURED. The unchanged `1e-9` relative bar therefore still fails. The
preregistered cancellation prediction is neither confirmed nor refuted.

The committed operator entry point is
`scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`.
It selects physical GPU 1, native XLA transcendentals, disables preallocation,
refuses the platform allocator, and validates policy, provenance, device,
classification, compile footprint, and cold-step segmentation in every JSON.
It runs agreement at 1, 6, and 60 steps; attempts 360 steps only if all shorter
windows produced classified JSON; then runs all three six-step resource
schedules and the ordering plant. Classified off-bar agreement rows do not
stop later memory rows. Any missing JSON or non-CONFIRMED measurement produces
a named REFUSE line and a final nonzero exit.

## Memory, wall time, and projections

The Round 8 GPU resource rows have not yet been run. The Round 7 CPU resource
attempts failed before measurement, and CPU multi-step reverse compilation is
now prohibited. Therefore no current peak-memory or warm-wall-time row exists,
and no 60-step or 360-step projection is admissible:

| schedule | measured 6-step peak/time | 60-step projection | 360-step projection |
|---|---|---|---|
| `none` | UNMEASURED | UNMEASURED | UNMEASURED |
| `uniform` | UNMEASURED | UNMEASURED | UNMEASURED |
| `binomial` | UNMEASURED | UNMEASURED | UNMEASURED |

The resource producer and fail-closed classifier remain at
`nemo_testcase_l2_gyre_gradient_gate.py:1168` and
`nemo_testcase_l2_gyre_gradient_gate.py:1334`. Projections will be written only
from the operator's valid measured six-step rows; this receipt fabricates none.

## Tests and gates

- Direct instrument controls: **43 passed in 7.22 s** after the review repair.
  They include the planted transformed-primal byte mismatch, computed NaN
  versus absent-derivative classification, the real-card cold-step segment
  plant, reusable cold/warm transform tracing, doubled-VJP behavior, policy and
  allocator refusal, timer classification, memory projections, sampler
  failures, and GPU-runner classified-outcome continuation.
- The pre-review exact three-file battery was **52 passed, 1 skipped in
  1837.09 s** with log SHA-256
  `24ee9653dc6b1fc8f141a616914a5aac308f13c7fa11090eb5879d7186aca3dd`.
  The post-review battery is **52 passed, 1 skipped in 1985.01 s**. It asserts
  the named one-step UNMEASURED debt rather than accepting the diagnostic
  agreement triplet; the parent process exited normally after the isolated
  child returned its expected nonzero classified refusal. The only skip is the
  named six-step CPU compiler limit.
- GPU-runner self-test: all five classified outcomes were recognized and the
  script completed.
- Receipt-specific citation gate and shifted-citation plant are run from the
  clean committed receipt producer; their JSON artifacts live beside the
  other Round 8 evidence.

## Independent adversarial review

The executable review returned HOLD because the prior clean JSON recorded
`step_vjp_replays_bitwise=false` and `step_jvp_replays_bitwise=false` while the
top-level row remained CONFIRMED. That finding is dispositioned by the binding
byte gate and planted mismatch control above. The evidence review returned
SHIP: it independently reproduced every CPU/GPU number, trajectory comparison,
SHA, provenance stamp, and citation control, and agreed that HELD is the only
admissible campaign status while resource rows remain absent. It noted one
non-blocking hygiene gap—direct-control and runner-self-test logs were not
retained—and independently reproduced 42 pre-repair controls and the runner
self-test. After the repair, the executable reviewer returned SHIP: it verified
the live refusal artifact, the planted mismatch, the matrix's expected exit-1
assertion, and the explicit receipt retraction. Both reviewers agree that SHIP
applies to the fail-closed repair while the measurement campaign remains HELD.

## OPEN, retractions, and decision

OPEN:

1. From an operator shell, run
   `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l2_gyre_gradgate_gpu_rows.sh`
   with the Round 8 evidence directory. Ingest its 6/60/360-step agreement,
   cancellation-scale, resource, and plant rows.
2. If all three six-step resource rows are valid, compute the preregistered
   60-step and 360-step memory projections and report their measured spread;
   do not describe the projection as a certified linear law.
3. Keep CPU multi-step GYRE agreement OPEN at the XLA:CPU LLVM section-memory
   boundary. Do not retry it without a new operator protocol decision.
4. Resolve the production-step transformed-primal byte mismatch before using
   any GYRE gradient for training. The binding CPU row is UNMEASURED even
   though its refused reverse/forward/FD diagnostic is numerically on-bar.

RETRACTED: the Round 7 receipt's wording that the six-step reverse had compiled
and only the forward transform timed out is superseded by the operator's
isolated 3837-second reproduction: the agreement process itself aborted in
LLVM compilation and wrote no JSON. The executable matrix skip now quotes that
failure. This round also retracts any implication that `per_step` meant one
reused compiled transform; before Round 8 it meant one newly-created transform
closure per model step. Finally, the first Round 8 receipt commit's one-step
CONFIRMED verdict is retracted: its own replay diagnostics showed the VJP and
JVP transformed primals were not byte-identical, so the executable now reports
that row as UNMEASURED.

DECISION_NEEDED: **Should the next round add a primal-preserving local AD
boundary that returns the separately executed production `_step_jitted` primal
while deriving its pullback/JVP from the unchanged step program? Pick: YES,
with the current byte gate retained and a direct equivalence plant; do not
relax the byte gate or any agreement bar.**

## Choice audit

- ASKED: classify absent derivatives UNMEASURED; segment the cold first step;
  reuse production-step VJP/JVP programs; stop CPU multi-step attempts; retain
  the CPU compiler debt; extend the GPU-1 runner through 6/60/360-step
  agreement and three schedules; preserve bars and byte gates; write this
  receipt and run the push gates.
- UNASKED: none. No physics, card, precision policy, numerical bar, forcing,
  timestep, loss, checkpoint schedule, or production state changed. No
  VORTEX, VORTEX_VEC, or TSUNAMI gradient row was added.
