# Fresh scaling review — ranked levers (2026-06-13, 4-agent code+lit pass)

A from-scratch review (4 parallel agents: atm dycores, ocean dycores, latest
JAX/XLA, latest dycore numerics) over the whole repo, going BEYOND the
established walls (`distance_to_limit_2026-06-13.md`). Key reframing (atm
agent): the "AT roofline" verdicts are *fp64-on-Ginsburg-fabric* statements
— the remaining headroom is changing the **precision / algorithm / iteration
count** so the roofline MOVES, not chasing a fixed roofline.

## Implementation order (ROI-per-risk, all AD-safe unless noted)

### Tier 1 — implement now (high confidence, bounded scope, AD-clean)
1. **Chebyshev barotropic preconditioner** (ocean, MPI weak-scaling). The open
   iteration-count lever, via the CHEAP mechanism (not multigrid). `M⁻¹r =
   p_k(A)·r` = k applications of the existing `_helmholtz_apply` with scalar
   Chebyshev coefficients on `[λ_min, λ_max]`. λ_min=1 (analytic: A=I+SPD≥1),
   λ_max=Gershgorin `max(2·diag−1)` via `_helmholtz_inv_diag` (one
   stop_gradient'd global_max at setup; NO per-iteration reduction). A
   polynomial in the W-self-adjoint A is W-self-adjoint → composes with the
   shipped `single_reduce` CG. ~80 LOC + W-self-adjoint/convergence test.
   Trades reductions→matvecs = right for the latency fabric. `_select_
   preconditioner` in barotropic_implicit_latlon_cgrid.py (jacobi|zonal_line).
2. **Spectral SH transform → GEMM/`dot_general`** (atm, per-device throughput).
   `grids/gaussian.py` does the Legendre step as broadcast-multiply-`jnp.sum`
   / `segment_sum` (ZERO `dot_general` in the module) → memory-bound reduction
   that never hits the tensor-core GEMM. Reformulate as `einsum`/`dot_general`
   → 3-10× on the one COMPUTE-bound atm grid (cuHPX/dinosaur evidence).
   AD-safe. (TF32 on synthesis = opt-in, validate vs fp64.)
3. **EOS/neutral-slope hoist in GM/Redi** (ocean, per-device). Density/slope
   recomputed 3× from the same T_mid/S_mid (EKE-κ, gm_redi_tracer_tendency,
   K33). Compute once, thread in. Clean redundant-recompute, ~30-50 LOC + parity.
4. **TF32-on-Turing gate** (correctness). `runtime/backend.py:415` sets
   `jax_default_matmul_precision="tensorfloat32"` for ANY nvidia, but RTX 8000
   is Turing sm_75 (no TF32 hw) — silent no-op now, wrong on a mixed fleet.
   Gate on compute-capability ≥ 8.0. ~5 LOC.

### Tier 2 — measure-first / needs a regression fix
5. **bf16 transport storage** (atm tracers + PPM intermediates, fp32 accum).
   Attacks the measured 580 B/cell·lev wall + halves halo VOLUME (helps
   bandwidth + the fabric). GRIST GMD2024: 44% on tracer transport. Risk:
   positivity/conservation in bf16 → mandatory tests; ship opt-in.
6. **MPAS field-coalesced halo + baroclinic distribution.** The MPAS baroclinic
   tendency never calls VoronoiHaloExchange (single-rank-correct only) — a real
   weak-scaling gap; then stack same-stage fields → 1 exchange. Needs the
   `batched_halo_exchange` 18× CPU regression fixed first (pack via scatter →
   gather).
7. **RAS / block-Jacobi-1-halo preconditioner** (ocean) — if Chebyshev's
   iteration count climbs with rank count (it won't much at ≤32 ranks).
8. **Fused K-build into Thomas (5a)** + **CPU tridiagonal → LAPACK `gtsv`**
   (lax.linalg.tridiagonal_solve is now LAPACK-batched + has a JVP; the legacy
   fori-loop is ~2000× slower). Incremental, AD-safe.
9. **XLA flags**: `xla_gpu_collective_permute_decomposer_threshold` +
   all_reduce/all_gather combine thresholds (absent) — overlap interior compute
   with ppermute halos + merge small collectives. A/B-gated (low-impact on
   PCIe, future-proofs NVLink + cuts Gloo message count).

### Tier 3 — big / deliberate
10. **IMEX semi-implicit extended to the ocean baroclinic** + default-on the
    spectral SI (machinery exists in semi_implicit.py) — larger dt, fewer
    barotropic subcycles. Multi-week, validate.
11. **HEVI-IMEX/ETD for the cube FV3 path** — step-count cut, fabric-independent.
    Multi-week, real numerics risk.
12. **NeuralGCM-style hybrid** (coarser dynamics + learned closure) — biggest
    raw speedup but a modeling/training project, not a numerics swap.

## NOT worth it (and why)
- **Pallas custom kernels**: Turing sm_75 is BELOW both Pallas backends'
  floors (Mosaic-GPU=Hopper, Triton=Ampere); the FV3 stencils are
  bandwidth-bound (XLA already 100% HBM) + Pallas is opaque to XLA's async
  collective scheduler. Defer to Hopper/Blackwell.
- **Full geometric multigrid as the FIRST barotropic move**: iteration ratio
  real but practical comm-event cut small (halos/V-cycle), barotropic ~7-10% of
  step, coarse-grid-under-MPI hard. Do Chebyshev (#1) first; MG only at
  >1000 ranks / NVLink-IB.
- **s-step / communication-avoiding CG**: its sync-per-iteration cut is ALREADY
  shipped (single_reduce). The open lever is iteration COUNT (#1), not
  sync-per-iter.
- **Exponential/ETD for the barotropic mode**: φ-functions = more global dots,
  less AD-friendly than the current implicit elliptic solve.
- **Parallel-in-time (Parareal/MGRIT/ParaDiag)**: wrong regime (chaotic
  hyperbolic, no idle ranks), fights lax.scan/AD.
- **2-D lat-lon decomposition / multinode-GPU SPMD / Shardy migration / NeuralGCM
  sharded transform**: hardware-blocked (no NVLink/IB) or measured
  fabric-blocked; not code levers on this hardware.
- **`uv.lock` jax 0.10 vs mpi4jax `<0.10`**: latent breakage to resolve
  (production runs the 0.9.1 cluster module, so not urgent).

## Bottom line
The two highest-value, AD-clean, implementable-now levers are the **Chebyshev
barotropic preconditioner** (ocean MPI weak-scaling, comm-light) and the
**spectral SH transform → GEMM** (atm per-device throughput, the one
compute-bound grid). Both MOVE the roofline (iteration count / FLOP
efficiency) rather than chase a fixed one. Tier-1 #3/#4 are cheap clean wins
to bundle. The cube-tiling grind (task #3, U1-U3 shipped) remains valid but is
a deep np>6 capability, not a near-term SYPD win.
