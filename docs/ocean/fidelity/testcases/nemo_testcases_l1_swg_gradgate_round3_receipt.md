# Certified SWG reverse-mode gradient gate — round 3 receipt

Date: 2026-10-12

Status: **LANDED; CPU EXECUTION TIMER REPAIRED; GPU RESOURCE OPERATOR ROWS
OPEN**. The GPU allocator and sampler-lifecycle blockers are fixed, all fifteen
registered CPU resource/timing rows pass, and the true sixty-day uniform CPU
gradient cost is measured directly. No physics, card choice, loss, agreement
bar, checkpoint schedule, stabilizer, clipping, guard, configuration, or
`stop_gradient` changed.

## Provenance and review disposition

The incoming tip was `9ae720dc7`. The frozen round-3 protocol and repaired
instrument are commit `9259e21f386a8c8b02bb4dbd91f9f9945b28bb35`. All CPU
JSON files are under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/swggg_rounds/round3/` and stamp
that clean commit, CPU fp64/libm, float64 state and geometry, the admitted kt=2
stage-1 boundary, and the production `_step_jitted` entry.

Both confirmed review blockers are closed. The GPU runner now disables
preallocation while leaving the default JAX allocator active; it refuses the
platform allocator because that allocator returned no memory statistics on
the operator GPU. Cold and warm passes use distinct sampler objects, and a
CPU-backed direct test executes that two-pass GPU control flow. The round-1
GPU resource retraction remains unchanged.

## GPU result carried into round 3

The operator's round-2 native-policy agreement rows are valid and complete:

| steps | reverse/forward relative | reverse/FD relative | verdict |
|---:|---:|---:|---|
| 1 | `2.27064171926424e-16` | `3.57798841572499e-07` | **CONFIRMED** |
| 24 | `9.30692734647596e-16` | `2.1980290555407674e-06` | **CONFIRMED** |
| 720 | `2.3070259307585205e-16` | `9.041792746430404e-07` | **CONFIRMED** |
| 1440 | `3.03e-15` | `6.9e-06` | **CONFIRMED** |

Every round-2 GPU resource row was **REFUSED/UNMEASURED** because the forced
platform allocator exposed no `memory_stats()` dictionary. No magnitude or
ordering is inferred from those failures. The repaired operator script runs
each resource row in a fresh process, records JAX allocator statistics and
secondary `nvidia-smi` samples, and keeps the existing positive
`none > binomial > uniform` dual-sampler ordering gate and its binding plant.

## Wall-time retraction and repaired span

The round-1/2 wall-time columns are **RETRACTED AS EXECUTION-COST
MEASUREMENTS** in their own receipts. Their second call invoked an un-jitted
`value_and_grad` transform and therefore retraced and compiled. The timer did
not include the sampler join, but it did include that step-independent JAX
work. This explains 24-step values of 10.6–25.0 s and the indistinguishable
720/1440 values.

Round 3 times `jax.jit(jax.value_and_grad(scalar_line))`. One cold call compiles
and executes; the reported warm span covers only a second blocked execution of
that already-compiled reverse, after sampler start and before sampler
stop/join. The direct trace-count control requires the two calls to trace once.

## Five-window CPU cost model and footprint

Every row runs in its own process. All losses and gradients are finite and
nonzero, all schedule primals satisfy the frozen value contract, and all
resource classifiers return CONFIRMED. Absolute peak RSS is the honest total
process footprint during the measured warm pass. Baseline-subtracted RSS is
not interpreted as adjoint memory because the cold pass leaves allocations
resident.

| steps | none time (s) | none peak (MiB) | uniform time (s) | uniform peak (MiB) | binomial time (s) | binomial peak (MiB) |
|---:|---:|---:|---:|---:|---:|---:|
| 24 | `0.287141` | `2767.172` | `0.189970` | `1705.145` | `0.531643` | `1916.305` |
| 96 | `1.067294` | `3936.520` | `0.719704` | `1729.801` | `2.820389` | `1923.418` |
| 360 | `4.006937` | `8208.395` | `2.692934` | `1817.805` | `11.680373` | `1929.375` |
| 720 | `7.894610` | `14009.004` | `5.445533` | `1940.852` | `22.298275` | `1940.723` |
| 1440 | `15.830778` | `25652.617` | `10.883964` | `2185.824` | `45.503263` | `1956.293` |

Post-hoc ordinary least-squares fits of `time = intercept + slope * steps`
give slopes of `0.0109718`, `0.00756102`, and `0.0316504` s/step for none,
uniform, and binomial, with respective `R^2` values `0.999987`, `0.999989`,
and `0.999759`. All slopes are positive and every 1440-step time exceeds its
720-step time, confirming T1. Uniform's direct ratio is
`10.883964 / 5.445533 = 1.998696`, inside the preregistered `[1.5, 2.5]`,
confirming T2. The true cost of a sixty-day CPU gradient with uniform
checkpointing is therefore **10.883964 s warm execution** and **2185.824 MiB
total peak process RSS** on this host.

At 1440 steps, none uses `25652.617 MiB` total peak RSS, uniform uses
`2185.824 MiB`, and binomial uses `1956.293 MiB`. These are process footprints,
not isolated tape allocations. The fifteen content-pinned JSONs remain outside
Git in the evidence directory; every number above comes from the committed
gate at the clean producer commit.

Their SHA-256 values, grouped as `none/uniform/binomial` at each step count,
are: 24 steps `53640fc409461cb21dad4dcb0b9b07bc2f885be22897ca74baabc51368d79d36` /
`b640d9437320e26d94c8051d63b7e0a2fdba74fb7d4e0e3bc4837057e9715b04` /
`6a1593432aaeef48c4c59d6dd8c6c726cc28b3c1effe6c536aa8e5013c92057a`;
96 steps `69d8fbaee366dc2f2aaad1f507f7cef4dba414648ac24694968d103a93b794a1` /
`5fb1e5980ffcd4ed12f5a04458b90b0be08858ba48c07413d171bcf44ce15938` /
`e866841cf4ab4ddd134842862d85e86eda2fe483ecd74467e3a501ef3f6d468d`;
360 steps `ab5318826bd8203eba2f5abe1ecffe81009040a58be5498a538ec60ad9df0d8c` /
`17d0010553c8534b0eb07b1bfc5a88b3f9fd1bf9ca855b26a534307de6e7f616` /
`9926ec493881e389422972e55554b7c075169479cc1efc55d6c937b359193bc2`;
720 steps `2fd171ed78ede618142705d6f4a4a3232611cd4d40958676cce4404d260ac5a9` /
`250a563a7c065319d3d3076f13cc0c7ad1a266ea7ad5accef75bc36f4b54825e` /
`e528d22155cdb9a7a30c10abe94a5f4f2e8bb2cf59bb08745f7c273f73ba2e09`;
and 1440 steps `0d6d7fed4bbc313edcb5cd1e7fa58cabb0a2a42cc0ead5ca6573103349b24ac3` /
`7c105562f9cbd7b48d8959be2b67d9db97986dd4a7dbb95403ba98e90d05caaa` /
`d9308f6f59d53ea6bebbd066cccd2c98d66748a35e3f3fae44c642de0cbfb8d2`.

## Directional agreement and training composition

No agreement calculation changed in round 3. CPU reverse/forward/central-FD
directional derivatives remain, respectively: one step
`-3.055917560677806e-02` / `-3.0559175606777962e-02` /
`-3.0559186540819565e-02`; 24 steps `-2.1842644706797177e-05` /
`-2.1842644706797268e-05` / `-2.1842692717673664e-05`; and 720 steps
`-3.5854846776002614e-09` / `-3.5854846769147900e-09` /
`-3.5854814359568625e-09`. The 1440-step row remains CONFIRMED, and the
broken-VJP plant remains binding.

Training will use the measured per-step Python loop: every modeled step crosses
the production jitted `swe_rk3` boundary and is rematerialized as one step. It
will not use the uniform scan composition. That scan's 24-step primal differs
from production by `8.881784197001252e-16` in SSH,
`1.3877787807814457e-17` in U, and `6.071532165918825e-18` in V. The bitwise
prediction remains **REFUTED**; this last-bit gap is retained rather than
waived.

## Tests, controls, and OPEN

- The shared gate direct tests, including allocator refusal, one-sampler-per-
  pass control flow, and one-trace compiled timing, pass.
- The GPU shell script passes syntax checking and covers agreement at
  1/24/720/1440 plus resources and ordering plants at 24/720/1440.
- The citation gate is run explicitly on this receipt; its shifted-citation
  plant must exit nonzero.
- The inherited stale GYRE/LOCK/OVERFLOW digest pins remain GYRE-lane debt.
  Expected values are `5850bbcfcd351c29` / `4ec82201f0fae7bf` /
  `7f1ec89b3e6578f5`; actual round-6 values are `f02d2f42a570f413` /
  `d5dfdf83b3a1e0ab` / `83de0010e1170032`. They are not re-pinned here.
- The operator must rerun the committed GPU-1 resource script at the final
  clean tip. CUDA acquisition is unavailable in this sandbox.

## Source citations

The certified compiled SWG dispatcher is the three-stage program at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:122-321`; its executed stage-3 momentum
block contains vector advection, ENS vorticity, constant Laplacian viscosity,
centered wind, and linear drag at
`SWG_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:260-273`. The total-ENS operator and its
live F-thickness divisions are in
`SWG_OMIP_L1/BLD/ppsrc/nemo/dynvor.f90:639-675`.
