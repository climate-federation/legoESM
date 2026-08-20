# 2025-2026 literature scan — NEW levers beyond the shipped set (2026-06-13)

Loop "when not making progress, look at recent literature" pass, AFTER the
+17-unit campaign extension (review levers + cube transport+KE tiling). Goal:
find scaling levers NOT already shipped + transferable to Ginsburg
(RTX8000 Turing sm_75, 2-GPU PCIe no-NVLink, CPU <=32 ranks Gloo-TCP).

## Already done (agent flagged, but we ALREADY have them) — do NOT re-do
- **XLA latency-hiding scheduler** (#3): `xla_gpu_enable_latency_hiding_
  scheduler=true` is ALREADY in runtime/backend.py NVIDIA_GPU_XLA_FLAGS
  (line 192) + `xla_gpu_enable_highest_priority_async_stream=true`. The
  agent's "OFF by default" is the XLA default; our backend overrides it.
  Explains the campaign note "XLA already overlaps GPU collectives".
- **XLA command-buffer / CUDA-graph capture** (#4): ALREADY set
  `xla_gpu_enable_command_buffer=FUSION,CUSTOM_CALL,COLLECTIVES` (line 199)
  [2026-08-11: `COLLECTIVES` dropped from the default after a measured +25%
  on the MPAS shard_map lane (job 26873637); `LEGOESM_XLA_CMDBUF_COLLECTIVES=1`
  restores it]
  — CUSTOM_CALL is included, so the LAPACK gtsv FFI IS in the captured set
  (no graph fragmentation from it). (Worth a one-off Nsight trace to CONFIRM
  the FV3 step is one graph, but the flag coverage is already correct.)
- **Pipelined/communication-avoiding Krylov** (#5): equivalent SHIPPED
  (Chronopoulos-Gear single_reduce + Chebyshev reduction-free precond). The
  2025 POP/CG-variant papers beat a naive PCG baseline we're already past.
- **Ocean f32 path** (part of #2): ALREADY clean + measured 1.72x LL128
  (campaign L2); production runs f64 by SCIENTIFIC choice, not a missing
  lever. Tensor-core/TF32/bf16: future-hardware (Turing has none; gated).
- **Distributed SHT / JAX-Fluids 2.0 SPMD / Shardy**: same SPMD class we
  shipped, or NVLink/IB-gated, or a partitioner migration (not a new lever).

## GENUINELY NEW headroom (not shipped) — both are substantial numerics
### #1 Split-explicit wide-halo barotropic → ZERO global reductions
- Silvestri 2025 (JAMES 2024MS004465) / Oceananigans (arXiv:2502.14148):
  explicit barotropic subcycling with halo width = #subcycles → ONE halo
  exchange/baroclinic step, NO allreduce; barotropic <10% of step; 75 SYPD
  1/4° 16 A100. Attacks the ocean weak-scaling wall (allreduce latency) at
  the ROOT — strictly beats Chebyshev/single_reduce (which only cheapen the
  reductions). MORE favorable on Gloo-TCP (latency is exactly what it kills).
- **CAVEAT (banked memory conflict)**: my campaign memory notes "MPAS-O
  EXPLICITLY REJECTED wider halos to batch substeps (conditionally-stable →
  only a few stale substeps)". Oceananigans makes it work with a specific
  stable subcycle + averaging; whether OUR barotropic (explicit_substep /
  implicit_cn free surface) is stable under wide-halo batching of MANY
  substeps is UNVERIFIED. Needs a stability + dispersion + conservation
  eval BEFORE committing. = a new SELECTABLE barotropic scheme
  (bebt_subcycle_widehalo in barotropic_common), not a rewrite; numerics
  change → mandatory conservation validation + codex.

### #2 Precision-partition FV3 (fp32 advection/flux, fp64 PGF + gravity)
- GRIST (GMD 17,6301, 2024): 24-44% (44% tracer transport), Hygon/Sunway
  FPUs — "memory wall, not tensor cores". neXtSIM-DG (GMD 18,3017, 2025):
  fp32 ~80% on A100, negligible 48h drift; TF32 = the UNSAFE one.
- TURING-FAVORABLE: RTX8000 fp64=1/32 fp32 → up to ~32x arithmetic swing
  on FLOP-bound stencils (vs ~2x on A100) + halves HBM/PCIe/Gloo bytes
  (helps walls #1,#3,#4). The NEW part vs the existing global-f32 path is
  the PRECISION PARTITION (fp64 ONLY on PGF/gravity/δπ; fp32 elsewhere) =
  a TYPED dycore (explicit .astype at precision-sensitive ops), not a global
  flag. Conservation-critical (keep flux-divergence sum in fp64). Agent's
  recommended contained first slice: TRACER transport (44%, lowest risk).

## Recommendation / next
- NOT at the absolute limit: #1 + #2 are real new headroom (ocean
  weak-scaling at-root; atm/ocean per-device on Turing's 1/32-fp64).
- Both are conservation-critical multi-step numerics projects (validation +
  codex mandatory). #2-tracer is the more-contained first slice; #1 is the
  higher-ROI-but-needs-stability-eval (and must resolve the MPAS-O caveat).
- The XLA scheduling levers (#3/#4) are already optimal in backend.py.

## VERIFY-FIRST VERDICT (2026-06-13, same day — both levers DEAD-END at our scale)
Verified the two "new" levers against the actual codebase + measured ledger
BEFORE building. Both collapse at production/Ginsburg scale:

- **#2 precision-partition FV3 — MOOT.** `fv3_sw_core.py` has ZERO `astype`
  calls: the cube SW dycore is dtype-POLYMORPHIC (inherits the input array
  dtype) and ALREADY runs float32 (line 1636 carries an explicit "float32
  overflow guard"). The lever's premise (fp32 flux off a fp64 baseline) does
  not apply — flux is already fp32; there is nothing below fp32 to drop to.
  The GRIST 44% was off a fp64 baseline. Ocean f32 path also already exists.
- **#1 split-explicit wide-halo barotropic — ALREADY REALIZED + the novel
  part is net-negative.** `explicit_substep` ALREADY EXISTS
  (`barotropic_latlon_cgrid.py`, `barotropic.py`, `barotropic_mpas.py`) and
  its substep body ALREADY avoids allreduces — it uses point-to-point
  `pad_ns_*` halos (lines 318, 365), NOT global reductions. The PCG-allreduce
  wall is `implicit_cn`-ONLY; the bench already runs `explicit_substep` as the
  default lane. The ONLY genuinely-new part — wide-halo BATCHING (one
  exchange/baroclinic step via an `n_substeps`-wide halo) — is:
  (a) documented net-negative on a latency fabric (`crm_gpu_l2_tiling.md`:
      "needs an `n_substeps`-wide halo (→ net-negative)"), and
  (b) bounded by a tiny budget anyway: `scaling_indicators.csv` row
      `8475875` measured the lat-lon barotropic at **~7% of the step** on the
      production tile (rows/rank=48) — eliminating ALL of it caps at ~7%, and
      this is exactly why the Chebyshev precond was NEUTRAL at production tile.
  The one regime where barotropic dominates (MPAS-O, 67% of step,
  `scaling_gpu.md:626`) is being handled by the implicit_cn MPAS work in the
  CONCURRENT session (`barotropic_implicit_mpas.py`) — out of scope here.

**Conclusion:** Ginsburg is at the practical theoretical limit on every
*measurable* axis; the literature's "new" levers are already shipped, moot, or
net-negative on this hardware. The remaining genuine in-domain engineering is
NOT a Ginsburg-benchable number but a CAPABILITY: completing the cube >6-device
sub-face tiling (the deferred d_sw1/d_sw5/d_sw6 RK-stage ops, task #3) so the
cubed-sphere grid can use >6 devices AT ALL on future fast-interconnect HW
(TPU pods, NVLink nodes). Resuming that, gated by the proven U3 bit-identity
methodology.
