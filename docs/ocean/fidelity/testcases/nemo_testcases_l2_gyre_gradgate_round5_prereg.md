# Preregistration — certified GYRE reverse-mode gradient gate, round 5

Date: 2026-10-11

Incoming lane tip: `a6018ed90ff8ce3beeb144a10edda589255054cc`.

This document is frozen before any Round 5 GYRE gradient, timing, memory, or
compiler measurement. Evidence belongs under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/gradgate_rounds/round5/`.
No NEMO executable will be run. No physics, card configuration, timestep,
objective, random direction, tolerance, stabilizer, clipping, guard, or
`stop_gradient` change is permitted.

## Fixed card and oracle statements

The program remains `build_nemo_testcase_card("GYRE-zco")` from the recorded
day-0 state, CPU fp64/libm, with every modeled step entering production
`LatLonCGridOceanModel._step_jitted`. The objective, perturbation, direction,
finite-difference step, agreement bars, per-step composition, and broken-VJP
plant are unchanged from rounds 1--4. The one-step accepted values at the
incoming tip are reverse `-3.884979632703882e-06`, forward
`-3.884979632703889e-06`, and central FD `-3.884977393501259e-06`, with
reverse/forward relative error `1.9622487750297417e-15` and reverse/FD relative
error `5.763743530264072e-07`.

The compiled NEMO GYRE program disables FCT in RK3 stages 1 and 2 at
`cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv.F90:295-298`, dispatches those stages to
`tra_adv_cen` at `:374-380`, and divides the horizontal and vertical tracer
flux divergences by live `e3t(Kmm)` inside owned-cell loops at
`cfgs/GYRE_OMIP_L2_P3_SM/WORK/traadv_cen.F90:137-149,203-216`. Round 4's
active-cell denominator treatment transcribes that execution domain and is not
changed here. The earlier TKE source statement remains
`cfgs/DINO/MY_SRC/zdftke.F90:832-837`; its September reverse guard also remains
unchanged.

## R5-P1: process-isolated CI row

Prediction: the GYRE one-step parametrization passes in a fresh subprocess with
the five accepted numerical values above, while the exact operator battery

```text
tests/ocean/fidelity/test_nemo_testcase_receipt_citation_gate.py
tests/ocean/unit/test_nemo_recipe.py
tests/ocean/validation/test_step_gradient_matrix.py
```

reaches a normal green pytest summary rather than exit 134. The parent test
must assert a commit-stamped strict JSON written by the existing gate script;
it must not import or compile the GYRE program in the parent process. A child
nonzero exit, absent/malformed JSON, wrong provenance commit, wrong policy,
wrong production entry, wrong step count, or any numerical value outside the
unchanged bars REFUTES P1. A synthetic child result that is not CONFIRMED must
make the parent assertion fail, proving the isolation test is not exit-code
only.

The operator's pre-change control reached 51 passing tests and then aborted at
`test_step_gradient_health[gyre_nemo_card_1step]` while compiling the forward
JVP after nine other variants. A separately launched one-step gate passed.
Therefore P1 attributes the battery abort to retained compiled executables in
one process only if the unchanged child measurement passes and the combined
parent battery exits normally.

## R5-P2: six-step CPU compiler discriminator

Prediction: the unchanged six-step agreement row still aborts in a fresh CPU
process with the recorded XLA/LLVM section-allocation signature:

```text
LLVM compilation error: Cannot allocate memory
contiguous_section_memory_manager.cc:101] allocateMappedMemory failed with error: Cannot allocate memory
Failed to satisfy suballocation request for 3791
LLVM ERROR: Unable to allocate section memory!
```

A strict six-step JSON containing finite reverse, forward, and FD values
falsifies the prediction and must be scored against the unchanged `1e-9` and
`0.05` bars. A different failure does not confirm the prediction. Before and
during the attempt, record `ulimit -a`, `free -g`,
`/proc/sys/vm/overcommit_memory`, `/proc/sys/vm/max_map_count`, and the current
map count. The operator's control (1006 GB total, 973 GB available; virtual,
resident, and data limits unlimited; 8 MB stack; heuristic overcommit) predicts
that ordinary RAM pressure and a process ulimit are absent.

If the section allocation repeats, compare lowered HLO text size and operation
count for the GYRE directional-derivative closures against the nine small
variants without compiling the machine code. Test each requested discriminator
in a separate process: thunk runtime true/false, the smallest directional
scalar closure, and an explicitly rematerialized per-RK-stage measurement arm
if the production stage API permits it without changing the primal. A
discriminator avoids the failure only if it writes a strict six-step row with
the production primal bitwise check intact; a mere successful lowering or
forward pass is not a gradient result. No compiler workaround becomes a model
default in this lane.

## R5-P3: resource and GPU rows

One-step CPU resource rows for `none`, `uniform`, and `binomial` remain eligible
and use separate clean processes. Six-step CPU resources and the frozen
360-step extrapolations remain UNMEASURED if P2 aborts. Peak host RSS is sampled
during the blocked warm reverse pass; compile RSS is not relabeled as adjoint
memory.

The committed GPU runner must select
`PrecisionPolicy.fp64(transcendentals="native")`, set
`CUDA_VISIBLE_DEVICES=1 JAX_PLATFORMS=cuda JAX_ENABLE_X64=1`, run one process at
a time, and preserve the physical-GPU-1 sampler. Every GPU JSON must stamp that
policy and is labeled **GPU, XLA transcendentals — not the certified bit-exact
policy**. The sandbox is expected to REFUSE before measurement because it has
no GPU; the operator shell owns the numerical run. A missing named REFUSE line,
use of libm, more than one visible device, dirty tree, overwrite, or unstamped
row invalidates the runner.

## Scope and required verdict

The withdrawn broad cross-card selector is not restored. `VORTEX-zco` is
excluded. The permitted `VORTEX_VEC-zco` and `TSUNAMI-zco` component rows may
only start after both GYRE one-step and six-step rows are green; if P2 is
confirmed they are not run. SWG remains OPEN until its certified card exists.

Round 5 is LANDED only if the exact three-file battery exits normally with the
isolated one-step GYRE row CONFIRMED. Six-step agreement, resource rows, and GPU
rows stay explicitly UNMEASURED wherever no strict JSON exists. Failed
predictions and compiler logs are retained in the receipt.
