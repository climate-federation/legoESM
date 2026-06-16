# Barotropic multi-node preconditioner verdict — 2026-06-15 (measure-first, honest)

The user directive "use multi nodes for tests" + the phase-split (job 8488677,
np32) pinpointing the **barotropic reduction-latency wall as the #1 multi-node
bottleneck** (24% of step; 120-allreduce floor = 21%) drove a thorough A/B of
every barotropic preconditioner/variant at LL192×384, np8/np16/np32. Several of
my own intermediate claims were OVERTURNED by residual-matched + wall-time
measurement — recorded here so they are not repeated.

## The numbers (LL192×384, np32, residual sweep job 8489419; wall-time 8488773/8488551)

| method | M=20 residual | M=60 residual | solve wall-time (np32) | converges? |
|---|---|---|---|---|
| jacobi (default)   | 2.45e-2 | 6.55e-3 | 12.0 ms (M60) | NO (crawls) |
| chebyshev deg-4    | 0.973   | 0.254   | 9.8 ms (M20)  | **NO — DIVERGES** |
| banded multigrid   | 1.95e-8 | 1.39e-16| 75 ms (M12)   | **YES (only one)** |
| single_reduce(jac) | ≈ jacobi| ≈ jacobi| 11.5 ms (M60) | NO (= jacobi) |

## Corrected findings (mea culpa)

1. **CHEBYSHEV DIVERGES at LL192 — NOT a win.** My earlier "chebyshev 1.23× at
   np16/np32" (job 8488773) timed chebyshev WITHOUT checking its residual — it
   was running FAST but NOT SOLVING (residual ~1.0 at M=20, 0.25 at M=60). A
   degree-4 Chebyshev polynomial cannot span the high-resolution polar-anisotropic
   Helmholtz condition number (lmax huge from the polar dx→0 rows). Higher degree
   would help convergence but add matvec-halos (already net-negative at M=40,
   0.93× full-step). **Lesson: ALWAYS check the residual before claiming a solver
   speedup — a fast non-solve is not a win.** (Also fixed a real bug en route:
   the chebyshev preconditioner upcast the f32 PCG carry to f64 — committed.)

2. **The banded MG is the ONLY true converger** (M20→2e-8, M40→machine
   precision) — re-frames its earlier "wall-time-NEGATIVE" verdict. The MG is
   6-9× slower than jacobi ONLY because the comparison was MG-converged vs
   jacobi-UNCONVERGED: jacobi is fast purely because it does NOT converge
   (M60→6.6e-3). The MG is an ACCURACY/CONVERGENCE lever (the only way to a
   truly-solved barotropic), not a wall-time lever at the loose production
   accuracy.

3. **No cheap wall-time fix for the reduction-latency wall on Ginsburg.** At the
   production accuracy bar (jacobi-M60 = 6.6e-3): jacobi is near-optimal (cheap
   per-iter; reductions dominate but the converging alternative, MG, costs more
   compute than the reductions it saves). single_reduce (Chronopoulos–Gear, 1
   reduction/iter, SAME jacobi convergence) is the only clean win — ~1.05× on
   the solve at np32 ⇒ ~2% full step. Shipped opt-in (`--pcg-variant
   single_reduce`); the only no-downside multi-node barotropic knob.

## Recommendation
- Keep **jacobi** the default (near-optimal at production accuracy).
- Use **single_reduce** for multi-node (marginal but free).
- Use the **banded MG** only when a CONVERGED barotropic is required (accuracy),
  accepting the wall-time cost — it is the only method that solves at LL192.
- Do NOT use chebyshev degree-4 at high resolution (diverges).

## Process lesson (repeated this session)
Filtering bench stderr with `grep ms/step` HID the real errors twice (the
chebyshev dtype crash, then the full-step failure). Capture full output, or grep
for BOTH the metric AND error patterns (`Error|Traceback|TypeError`).
