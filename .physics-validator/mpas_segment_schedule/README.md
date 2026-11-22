# Receipt: checkpoint-scheduled differentiable MPAS segment

`legoesm.driver.mpas_diff_step.mpas_segment(..., schedule=...)` runs the production
MPAS step as a fixed-length loop (one step per iteration, `lax.switch` on the step
counter selects the cadence's physics variant) under
`legoesm.training.checkpoint_schedule.checkpointed_loop`, so reverse-mode memory is
O(K) state copies for any window length.  The Python-unrolled form (`schedule=None`)
is unchanged.

**Status in one paragraph.** Memory and loop form are certified here (peak
memory flat in window length under the schedule; scheduled == unrolled to
roundoff).  The multi-step GRADIENT VALUES were non-finite or 1e79-wrong at
>= 32 steps on the production deck; that is a Morrison/MG2 adjoint defect,
not a checkpointing one, traced below and fixed (gradient-only) in PR #2488
(`fix/mg2-absent-gradient-gate`).  With that fix the 32-step GPU gradient
matches FD to 1.1e-3 (section "Gradient-only fixes").  Per-step logs are in
`runs/`.

## Unit evidence (tests/coupler/unit/test_mpas_diff_step.py; res2 L8, fp64, 10 steps)

| check | measured | asserted |
|---|---|---|
| forward, 1 step, scheduled vs unrolled | bitwise | bitwise |
| forward, 10 steps, scheduled vs unrolled (rel. to max field) | 4.9e-13 | 1e-11 |
| forward, 10 steps, unrolled under one jit vs eager per-step jit | 4.9e-13 | (control: fusion roundoff) |
| gradient wrt T0, uniform / binomial(K=2) vs unrolled per-step remat | 9.8e-16 / 7.8e-16 | 1e-12 |
| mutant: wrong variant table in the scheduled loop | 3.8e-1 | caught |

## Production receipt (this directory)

Launcher: `scripts/cluster/levante/mpas_segment_schedule_receipt.sbatch`
(`scripts/validate/production_ad_map.py --stage step`, production deck at
`--resolution 3`, 128 steps, one A100 80 GB per arm, one process per arm; the
smoke below ran on a CPU node before the switch to GPU).

Deviation from the pinned inputs (user 2026-11-04): IC =
`/work/bb1596/b309178/diffESM/legoesm_ap_main/data/era5_ic_1979-01-01.zarr`
(37-level 1 deg ERA5, the SOTA deck's IC) instead of the unreadable
`era5_ic_1979-01-01_gproxy_L21.zarr`; surface data from the sibling checkout; the
deck's res-6 land IC regridded once to res 3 with the deck's own tool
(`scripts/data/regrid_land_ic.py`, nearest land neighbour, source soil column
attested 10,3.0,2.0 as its metadata records; 642 columns from 17301 source land
columns), because the strict loader refuses a column-count mismatch.

Pre-registered before the run: FD/AD-1 per arm |.| < 1e-4 (as the 16-step probe,
PR #1938) and the three arms' AD agree to roundoff; PEAK_GIB binomial < uniform <
none (refuted if binomial >= none).

Smoke (job 28723677, binomial K=2, res3, 32 steps, fp32 RRTMGP as production,
82 min wall, 1 CPU node): AD finite; FD/AD-1 = 3.4e-4 at 1 step, 1.5e-3 at 32
steps (fp32 radiation roundoff in the FD, as the probe documents); forward
scheduled-vs-unrolled max|diff| T 2.9e-5 K, u 3.6e-6 m/s, p_s 1.5e-5 Pa, q_v
4.2e-9; RSS high-water 36.7 GiB after the 1-step check, 50.7 GiB after 32 steps.

## Results (job 28729306; 128 steps, fp64 RRTMGP via --rad-fp64, K=12, one A100 per arm)

| arm | 1-step FD/AD | 128-step AD | 128-step FD | peak GiB 1 / 128 steps | forward vs unrolled, max diff T / u / p_s / q_v | wall |
|---|---|---|---|---|---|---|
| none (unrolled, per-step remat) | 1.000345 | never produced: XLA compile of the 128-step unrolled graph ran 4 h at 0 % GPU, cancelled | - | 10.98 / - | - | >4 h, cancelled |
| uniform | 1.000345 | **NaN** (all leaves non-finite) | -3.507e8 | 11.02 / 13.72 | 5.0e-6 K / 7.8e-7 m/s / 1.4e-5 Pa / 1.0e-9 | 75 min |
| binomial K=12 | 1.000345 | **NaN** | -3.505e8 | 10.98 / 13.18 | 4.7e-3 K / 1.1e-4 m/s / 1.4e-2 Pa / 4.0e-7 | 87 min |

Pre-registration outcome: PEAK binomial < uniform CONFIRMED (13.18 < 13.72 GiB;
the none arm never reached 128 steps, so binomial < none is not measured --
the unrolled form did not compile, which is the stronger statement about it).
FD/AD-1 < 1e-4 at 128 steps REFUTED for both scheduled arms: the forward loss
is finite (FD evaluates) but every entry of the reverse-mode gradient is NaN.
The 1-step gradient is finite and identical across arms to 1e-15 (same setup).
The 32-step smoke (above) was finite under binomial K=2 with fp32 RRTMGP.
Not yet attributed (schedule defect vs the model's own reverse pass beyond
32 steps); bisection below.  The binomial forward's 1e-3 K drift from the
unrolled run (uniform: 5e-6 K) is unexplained: PLAUSIBLE a threshold switch
(convection trigger) flipping in one column from roundoff; not checked.

## Bisection (job 28735055; same deck, fp64 RRTMGP, one A100 per arm)

Arms: uniform at 32, 48, 64, 96 steps; unrolled (none) at 48 steps.
Pre-registered, written before the job ran:
- each arm prints GRADCHECK finite/ratio and, when non-finite, a NONFINITE
  census (leaf, count, columns);
- uniform 48 NaN with uniform 32 finite -> the second radiation call (step 32)
  on an evolved state is the source (CONFIRMED if the NONFINITE columns are
  cloudy columns at that step; otherwise only the step is located);
- uniform 48 NaN and none 48 finite -> schedule defect (the two differ only in
  loop form + lax.switch); uniform 48 NaN and none 48 NaN -> model-level;
- finite through 96 and NaN at 128 -> onset in (96, 128] (fourth radiation
  call or state drift), next bisection 112.
Results (GPU, fp64 RRTMGP, uniform unless stated; the 48-step arms were refused
by the driver -- a window must be a multiple of the 32-step radiation cadence):

| window | AD finite | max abs gradient | AD dot v | FD | forward vs unrolled, max diff T |
|---|---|---|---|---|---|
| 1 | yes | 2.3e13 | -6.47e6 | -6.47e6 | - |
| 32 | yes | **5.2e79** | **-9.5e72** | -7.99e6 | 0 (bitwise) |
| 64 | yes | **5.2e199** | **-1.2e194** | -2.95e7 | 8e-13 |
| 96 | no (every entry of T and q_v) | inf/NaN | NaN | -5.74e7 | 5e-8 |
| 128 | no | NaN | NaN | -3.51e8 | 5e-6 |
| 32, smoke: CPU, fp32 RRTMGP, binomial K=2 | yes | 2.1e13 | -7.98e6 | -8.00e6 | 2.9e-5 (fp32 radiation) |

Reading (CONFIRMED from the logs above): the NaN at 96 and 128 steps is
overflow, not a kink -- the reverse-mode gradient grows by ~1e66 over steps
1-32 and by a further ~1e120 over steps 32-64 (about 1e2-1e4 per step) while
the forward is stable (the directional FD stays O(1e7), and roundoff
differences between loop forms grow only ~1.4x per step).  A gradient that
grows 1e4 per step while forward perturbations grow 1.4x per step is not the
Jacobian of the forward map that ran: a wrong or unstable reverse rule, or a
backend-specific code path, is the only consistent class.  On a CPU node
(smoke, fp32 RRTMGP, binomial K=2) the same 32-step gradient is sane, so the
amplifier is in one of three co-varying differences: GPU backend, fp64
RRTMGP, uniform (scan + per-step remat) vs binomial loop; the 128-step
binomial arm on GPU also overflowed, which makes the loop form the least
likely of the three.

Checked offline (CPU, scratch script, not committed): the GPU-only PCR
tridiagonal solver (`thomas_solve_batched` dispatches to `pcr_solve_batched`
on CUDA; CLUBB's implicit vertical diffusion reaches it) has a correct and
stable adjoint on diffusion systems with K up to 1e4: gradient through 32
composed solves agrees with the CPU Thomas solver to 2e-12 and with FD.  So
PCR is not an obvious amplifier on well-posed systems; not excluded for the
production matrices.  Other backend-dependent code in the step: RRTMGP's
recurrent transport uses lax.scan on GPU and a Python loop on CPU
(radiation runs only at step 0 of a 32-step window, so it is a one-off
factor, not a per-step one).

Next discriminator (not launched): three one-variable GPU arms at 32 steps:
(a) the smoke's exact setting on GPU (fp32 RRTMGP, binomial K=2) -- insane
means the backend is the variable; (b) unrolled loop (schedule none), fp64 --
sane with uniform insane means the scheduled loop's reverse pass on GPU;
(c) uniform fp64 with LEGOESM_TRIDIAG=legacy -- sane means the PCR adjoint on
the production matrices.

## Discriminating arms (jobs 28747193/4/5, 2026-11-05, GPU A100, res3, 32 steps)

One variable per arm against a run already in hand.  Logs in `runs/mpas_seg_disc_*.txt`;
the CPU smoke they are compared with is `runs/mpas_seg_smoke-2-28723677.txt`.

| arm | backend | RRTMGP | loop | tridiag | 32-step |g|max | AD / FD | node |
|---|---|---|---|---|---|---|---|
| smoke 28723677 | CPU | fp32 | binomial K=2 | legacy (CPU default) | 2.1e13 | 1.0015 | l10689 |
| a 28747193 | GPU | fp32 | binomial K=2 | PCR (GPU default) | NaN, every entry of T and q_v | NaN | l50039 |
| bisect 28735055 | GPU | fp64 | uniform | PCR | 5.17e79 | 1.2e66 | l50003 |
| b 28747194 | GPU | fp64 | none (unrolled) | PCR | 4.41e87 | 1.2e78 | l50136 |
| c 28747195 | GPU | fp64 | uniform | legacy (forced) | 4.41e87 | 1.2e78 | l50136 |

1-step gradients in all five: |g|max 2.329e13, AD/FD 1.00034, agreeing across nodes
and solvers to 1e-9 relative (solver swap b vs c moves the 1-step AD slope in the
14th digit).  Forward: c's scheduled loop matches its unrolled loop bitwise; a's
Treeverse loop differs from unrolled by 4.4e-5 K (fp32 radiation roundoff, as the
CPU smoke's 2.9e-5 K).

Pre-registration outcome:
- (a) INSANE -> **backend CONFIRMED as the variable**: identical configuration
  (res3, same IC, same land IC, 32 steps, fp32 RRTMGP, binomial K=2) is sane on CPU
  and non-finite on GPU.
- (b) INSANE, same value as (c) -> **loop form cleared**: the unrolled Python loop
  blows up identically to the scheduled loop; the schedule machinery is not the
  amplifier.
- (c) INSANE, same value as (b) -> **tridiagonal solver cleared**: forcing the CPU
  Thomas kernel on GPU changes nothing beyond the 11th digit.

Reading.  The blow-up is GPU-specific and sits in the composition of steps, not in
any single step from the IC (the 1-step VJP agrees across backends to 1e-9).  The
32-step value is node-dependent (5.2e79 on l50003 vs 4.4e87 on l50136, from 1e-11
relative differences at step 1), i.e. the amplification is roundoff-sensitive.  30
of the 32 steps are dynamics-only (physics increment applied every 16 steps, zero in
between; radiation only at step 0), and the MPAS dycore uses no tridiagonal solver,
so the remaining GPU-specific suspects are XLA GPU codegen of the dycore/held step
reverse pass and the RRTMGP GPU paths (`rte_utils.py:150` scan, `optics_utils.py:79`),
the latter a one-off factor at step 0 only.  Mechanism: UNKNOWN; two classes remain,
(i) a GPU reverse-pass defect in a per-step kernel, (ii) a steep smoothed switch whose
argument the GPU trajectory lands on and the CPU trajectory misses (huge local
derivative with a bounded FD slope).

Next discriminator (not launched, ~1 GPU node-hour): on one GPU node, which also has
a CPU JAX device, replay the GPU 32-step trajectory and at each stored state compute
the ONE-STEP vector-Jacobian product with a fixed cotangent on both devices.  Per
step: forward max|diff| GPU vs CPU, VJP norm on each device, max|diff| of the VJPs.
Per-step VJPs agreeing at every state while norms grow -> class (ii) (the adjoint is
genuinely that large at those states; find the switch via the leaf census).  A step
where the GPU VJP departs from the CPU VJP -> class (i), with the step index and
physics variant naming the kernel.  Needs a new stage in `production_ad_map.py`
(no existing per-step cross-device VJP probe: grep `devices("cpu")|default_device`
in scripts/validate and the driver found nothing).

## Discriminating arms (jobs 28747193/4/5, 2026-11-05, GPU A100, res3, 32 steps)

Code: `f5bc44b5df` -- the branch was rebased onto main between the bisection
(`dc8cce9256`, 2026-11-04) and these arms (534 files changed on main, among
them CLUBB's band solvers #2337 and the voronoi fp64 mesh cache #2381/#2382).
So these three arms are one-variable comparisons AMONG THEMSELVES, not against
the bisection or the CPU smoke; the 1-step gradient agrees with the bisection
to 1e-9 relative, the 32-step gradient does not (5.2e79 then, 4.4e87 now).

| arm | setting | 32-step |g|max | AD | FD | forward vs unrolled |
|---|---|---|---|---|---|
| a | fp32 RRTMGP, binomial K=2 (the CPU smoke's setting) | NaN (every T, q_v entry) | NaN | -7.992e6 | T 4.4e-5 K |
| b | fp64 RRTMGP, unrolled (schedule none) | 4.413435329728989e87 | -9.947237727083807e84 | -7.989e6 | n/a |
| c | fp64 RRTMGP, uniform, `LEGOESM_TRIDIAG=legacy` | 4.413435329739839e87 | -9.94723772711071e84 | -7.989e6 | bitwise 0 |

Reading:
- b == c to 12 digits although they differ in BOTH loop form and tridiagonal
  solver: neither the scheduled loop's reverse pass nor the GPU PCR adjoint is
  the amplifier (CLUBB's `implicit_vertical_diffusion` does reach
  `thomas_solve_batched`, so arm c exercised the switch). Two independent jobs
  agreeing to 12 digits is also a run-to-run determinism receipt. CONFIRMED on
  this code.
- a: the CPU smoke's exact setting is NaN on GPU. The smoke ran on the older
  code, so backend and code version co-vary; a GPU arm on the pre-rebase tip
  (`962dfcff3b`) at the smoke's setting closes that (not launched).
- GPU startup sets `jax_default_matmul_precision=tensorfloat32`
  (`runtime/backend.py`); this touches float32 matmuls only and the policy is
  fp64, so PLAUSIBLE-low as a cause, unverified.
- Where in the 32 steps the growth happens (the 30 held dynamics-only steps vs
  the two physics steps) is still unmeasured; the driver refuses windows that
  are not multiples of the radiation cadence except n=1.

## Backend isolation (jobs 28751325 / 28751633 / 28751634, 2026-11-05, res3)

Three arms resolving the rebase confound above (same probe, same deck, same
IC; `mpas_seg_disc_a` setting = fp32 radiation, binomial K=2, 32 steps):

| job | backend | code | setting | 32-step gradient |
|---|---|---|---|---|
| 28751325 (`disc_cpu`) | CPU, compute node | f5bc44b5df (today) | disc_a | finite, max 2.0992e13, AD/FD = 1.00154 |
| 28723677 (`smoke-2`, yesterday) | CPU | dc8cce9256 (pre-rebase) | disc_a | finite, max 2.0992e13, AD/FD = 1.00154 (agrees with 28751325 to 12 digits) |
| 28751633 (`disc_prerebase`) | GPU A100 | 962dfcff3b (pre-rebase, worktree + PYTHONPATH on the same venv) | disc_a | NaN, all 20544 T and q_v entries |
| 28747193 (`disc_a`, above) | GPU A100 | f5bc44b5df (today) | disc_a | NaN, all entries |
| 28751634_3 (`ladder`) | GPU A100 | f5bc44b5df (today) | uniform, fp64 rad, 16 steps, `--checkpoint-days 0` | finite, max 2.2265e13, AD/FD = 1.00033 |

Reading:
- CONFIRMED: the backend is the discriminator. Code held fixed (pre-rebase
  tip), CPU is sane and GPU is NaN; backend held fixed (CPU), both code
  versions agree to 12 digits. The 1-step gradient agrees CPU vs GPU to 7e-7
  relative (roundoff), so the divergence is multi-step.
- CONFIRMED (GPU, today's code, fp64 radiation): the 16-step gradient is
  healthy (same order as 1 step, AD/FD 3e-4). The explosion to 4.4e87 at 32
  steps therefore originates in steps 17-32, i.e. the step-16 physics step
  (the no-radiation variant) and/or the 15 held steps after it. The 15 held
  steps after the step-0 full physics step were fine. Which of the two is
  UNMEASURED: a 17-step window would discriminate, but the driver refuses run
  lengths that are not multiples of `physics_update_steps`=16 even with
  checkpoint writing off (ladder arms 0-2, n=2/4/8, exit 1 with that
  ValueError; `--checkpoint-days 0` only disarms the radiation-cadence rule).
  The probe would need a segment length separate from the driver length (the
  1-step check already works that way).
- GPU-only differences known so far: startup enables TF32 for float32 matmuls
  (`runtime/backend.py:435-438`; the run policy is fp64, so PLAUSIBLE-low),
  PCR tridiagonal on CUDA (cleared by arm c, which forced the Thomas solver
  and matched arm b to 12 digits), XLA:GPU fusion/ordering of the adjoint.
- Launcher gained `PROBE_EXTRA` (extra run_amip flags) and runs another
  commit via `REPO=<worktree> PYTHONPATH=<worktree>/packages/*:<worktree>/src`
  on the same venv (the editable .pth entries point at the main checkout;
  PYTHONPATH precedes them, and `legoesm` is a namespace package so the
  worktree copy wins for every subpackage; verified by printing
  `mpas_diff_step`, `tridiagonal`, `checkpoint_schedule`, `clubb` `__file__`).

## Per-step adjoint (job 28780080, 2026-11-06, GPU A100, res3, 32 steps, fp64 rad, unrolled loop)

`production_ad_map.py --stage step --schedule none --stash-grads`: an identity
`custom_vjp` tap on the state entering each step writes max|cotangent| per
leaf into a `jax.ref` row (`make_grad_stash`; the JAX docs' stash_grads
pattern).  Row k = d loss / d(state entering step k); the step-k map lies
between rows k+1 and k.  Cadence: physics at steps 0 (with radiation) and 16
(without), dynamics-only otherwise.

| rows | max leaf | max |cotangent| |
|---|---|---|
| 31 -> 17 (dynamics-held steps after the step-16 physics) | u, then T | 5.0e2 -> 5.3e3, smooth growth x1.1/step |
| 16 (entering the no-radiation physics step) | tracers['q_i'] | 7.7e89 |
| 15 -> 1 (dynamics-held steps after the step-0 physics) | tracers['q_i'] | 7.7e89 -> 7.4e89, flat |
| 0 (entering the full physics step) | phys_state.clubb_moments | 4.3e93 |

`ADJOINT_JUMP step=16 ratio=1.455e+86 (entering 7.714e+89, leaving 5.301e+03)`.

Reading (CONFIRMED on this run): the whole 1e86 amplification happens inside
ONE call, the step-16 physics step without radiation, and lands on the
cloud-ice tracer's cotangent; the 15 dynamics steps before and after it change
the adjoint by at most x1.1 per step, and the step-0 full physics step then
maps the q_i cotangent onto the CLUBB moment carry (x5.6e3).  The dynamics and
the held steps are cleared; the suspect is a q_i consumer in the no-radiation
physics variant (microphysics or CLUBB) evaluated on the GPU step-16 state.
UNMEASURED: which module, and which column/level (the stash records the max,
not its location); the CPU per-step profile at the same setting.

## Inside the step-16 physics call (jobs 28782048/2123/2124/2505/2509/2812/2813, 2026-11-06, res3, fp64 rad)

Instruments: `--stash-grads` now also records the argmax (cell, level) and the
primal there; `--at-step N` advances the exported state N production steps
before the stage; `--x-tracers q_v,q_i` differentiates cloud ice too;
`--physics-variant norad` for the components stage; `--stash-modules`
(`make_call_stash`) taps the input (state, phys_state) and the output of EVERY
module call inside one physics call (MODTAP lines, forward call order).  The
production deck sub-cycles CLUBB + Morrison `cld_macmic_num_steps: 3` times
per physics call, so one call is conv, (turb, micro) x3, gwd [+ rad first].

Single physics call on the step-16 GPU state (x = T, q_v, q_i):

| check | grad max | AD vs FD |
|---|---|---|
| physics_full / physics_norad | 1.2e75 / 1.3e75 | AD 1e70 vs FD 2e3: adjoint wrong |
| per module, first call only (GPU) | micro 5.8e16, turb 2.3e6, conv 1.7e4, gwd 7e-2 | agree except CLUBB (AD/FD 6.1) |
| per module, first call only (CPU) | identical to GPU to 1e-4 | identical |
| physics_full on the step-0 state (control) | 1.4e17 | AD/FD 0.998 |

MODTAP, step-16 norad, read backwards (the reverse pass): gwd out 8e1 ->
micro#3 out 2.8e1 -> **micro#3 in 6.9e32** (leaf N_s, cell 640 level 21,
primal N_s = 0.0) -> turb#3 out 4e30 -> turb#3 in 3e12 -> micro#2 out 4e35 ->
**micro#2 in 8.6e122** (N_s, cell 111 level 25, N_s = 0.0) -> turb#2 out 6e93
-> turb#2 in 1e75 -> micro#1 out 5e125 -> micro#1 in 4e122 -> ... -> conv in
3e44.  The two microphysics sub-steps 2 and 3 multiply the cotangent by 2e31
and 2e87, both landing on the SNOW NUMBER tracer N_s at cells where N_s is
exactly zero.  CLUBB shrinks the cotangent at each of its calls.  The step-0
control shows bounded microphysics factors (<= 1e4 per call) on q_i at tiny
q_i, no N_s blow-up.

CPU, same setting, 32-step stash: `ADJOINT_JUMP step=16 ratio=1.820e+12
(entering 9.651e+15, leaving 5.301e+03)` -- the CPU step-16 physics ALSO
amplifies, by 1.8e12 instead of 1.5e86, and the CPU 32-step gradient stays
FD-consistent (AD/FD 1.0008) because 1e16 does not yet swamp T/q_v.

Reading:
- CONFIRMED: the defect is the reverse pass of Morrison microphysics in its
  2nd/3rd macmic sub-step, through the snow number N_s at N_s = 0.  Same class
  on both backends; the backend changes only the size (1e12 vs 1e86).
- CONFIRMED: dynamics, radiation, convection, GWD and CLUBB are not the
  amplifier (CLUBB has its own AD/FD x6 mismatch -- separate defect).
- PLAUSIBLE: a power or ratio of N_s with an unbounded derivative at zero,
  e.g. the snow slope LAMS = (pi rho_sn N_s / q_s)^(1/3) (morrison.py ~1991;
  d/dN ~ N^(-2/3)) reached through a clamp/where whose zero-cotangent branch
  still multiplies a huge-but-finite local derivative.  Exact op UNMEASURED.
- UNMEASURED: why sub-steps 2-3 and not sub-step 1 (sub-step 1's input has
  the same N_s zeros); the CPU MODTAP profile.

## Same-state cross-backend 2x2 and the per-equation microphysics trace (2026-11-10)

2x2 (jobs 28875158/63/64/65; physics stage, step-16 state saved to npz and
re-loaded, input digest printed and matched; FD = 1.78e3 / 1.95e3 in all four):

| state \ backend | GPU | CPU |
|---|---|---|
| advanced on GPU (digest c112e53439b759bf) | 1.3e75 | 1.2e58 |
| advanced on CPU (digest 5b8b45f7a626e1c5) | 2.0e77 | 2.5e58 |

CONFIRMED: on bit-identical input BOTH backends give an adjoint off by > 50
orders of magnitude; the GPU is systematically ~1e17 larger. Model defect;
backend changes the size only.  CPU jobs ran on GPU nodes with FORCE_CPU=1.

Per-equation trace (job 28877034, `--stage eqns`, `eqn_trace`: forward and
reverse pass one jaxpr equation at a time, jitted sub-jaxprs recursed, the
summed per-equation gradient matches jax.grad to roundoff on every leaf), the
three Morrison calls of the step-16 GPU physics call (x = T, q_v, q_i, N_s;
3424 equations on the reverse path):

| call | largest single-equation gain | where | primal there |
|---|---|---|---|
| 2 | div, x2.5e13 | `_warm_rain.py:2382` `mg2_qi_conservation_ratio` (q_i sink rescale, available/sink) | 2.1e-291 |
| 3 | div, x1.2e28 | `morrison.py:1863` MG2 snow-number aggregation limiter (N_s available / aggregation sink) | 2.6e-223 |
| 2, 3 | mul, x600 (= 1/dt scale) | `morrison.py:2142` `_ns_post` | 2e-291 / 3e-228 |

Reading: the amplifiers are the MG2 conservation limiters, ratios of the form
(available / dt + sources) / sink.  Their denominator is guarded against zero
(`jnp.where(_nlim, -nsagg, 1)`; the q_i one at sink > 1e-30) but not against
TINY POSITIVE values: at sinks and stocks of 1e-223 .. 1e-291 (far below
MG2_QSMALL = 1e-18) the derivative of the ratio w.r.t. the sink is
num/sink^2.  The single-call AD/FD checks pass only because the FD direction
perturbs N_s relative to N_s and never moves these cells.  The snow-number
limiter has NO small-sink gate at all.
Instrument caveat (fixed after the run): the EQN_GAIN ranking printed rows
with zero output cotangent as gain=inf; the EQN_MAX ranking above is the one
used. Eager (unjitted) replay of the calls; roundoff-level differences from
the compiled run are possible.

## Gradient-only fixes (2026-11-10/11, GPU, res3, fp64 rad; forward bitwise unchanged)

Fix 1 (jobs 28877667/68, trace 28879133): `div_gated_tangent` on the two
limiters the trace named (MG2 q_i conservation ratio; MG2 snow-number
aggregation limiter), tangent 0 where the stock is below MG2_QSMALL.
Fix 2 (jobs 28897363/64/65, suite 28897366): `mg2_absent_gradient_gate` at
the Morrison entry of the hydrostatic microphysics wrapper: every hydrometeor
mass below MG2_QSMALL, and its number, held constant for the gradient.

| check | before | fix 1 | fix 1 + 2 | FD |
|---|---|---|---|---|
| one physics call (full), step-16 GPU state, AD | -1.9e70 | -1404.32 | -1404.32 | -1778.28 |
| one physics call (norad), AD | -2.2e70 | 2155.34 | 2155.34 | 1952.02 |
| 32-step window, AD | -9.9e84 | -4.7e72 | **-7.98028e6** | -7.98921e6 (AD/FD 1.0011) |
| 32-step adjoint jump at step 16 | 1.5e86 | 1.8e80 | 1.8e12 | |
| d/d N_s at microphysics pass 2 / 3 input | 1.2e122 / 1.9e99 | same | 2.1e3 / 3.0e3 | |

- CONFIRMED: the 32-step GPU gradient is finite and FD-consistent (1e-3)
  after fix 2; its step-16 jump (1.820e12, entering 9.651e15) now equals the
  CPU's pre-fix profile to 4 digits.
- Remaining: (a) d/d N_s at microphysics pass 1 = 2.5e92 at a cell with
  N_s = 0 and q_s >= MG2_QSMALL (a mass-based gate cannot see it); (b) the
  q_i cotangent entering step 16 is still 9.7e15; (c) one physics call stays
  10-27 % off FD, unchanged by fix 2 -- PLAUSIBLE: CLUBB, whose own AD/FD is
  6.1 on this state (kink test jobs 28899290-92 running).
- Microphysics suite + hydrostatic microphysics tests: 984 passed, 6 failed;
  the 6 fail identically on the unmodified HEAD (worktree, same node type,
  serial): 5 fixture-input regenerations differ in the last bit, 1 segfault.

## CLUBB kink test (jobs 28899290-92, step-16 GPU state, first CLUBB call alone, FD step sweep)

| FD step (x the probe's direction) | perturb all: FD/AD | T only: FD/AD | q_v only: FD/AD |
|---|---|---|---|
| 1e-1 | 2.02 | 2.00 | 1736 |
| 1e-2 | -0.25 | -0.24 | 1.015 |
| 1e-3 (the default step) | 6.11 | 6.11 | 1.0002 |
| 1e-4 | 1.0000 | 1.0000 | 1.0000 |
| 1e-5 | 1.0000 | 1.0000 | 0.996 |

CONFIRMED: CLUBB's adjoint is correct; the x6 mismatch was the default FD
step straddling kinks (clips/limiters) in the temperature direction -- the FD
wanders at large steps and converges to AD to 5 digits once the step is small.
The earlier "CLUBB AD/FD 6.1" is retracted as a defect.  PLAUSIBLE: the
remaining 10-27 % physics-call mismatch is the same FD artefact (not swept).

## Gradient cost (timing stage, 2026-11-11, A100, production deck, fp32 radiation, after the MG2 gradient gate)

`--stage timing`: each gradient compiled once (`jit(...).lower().compile()`),
then 3 steady-state calls timed (median); forward timed the same way.

| res | schedule | window | grad (steady) | per step | grad/fwd | peak | grad compile |
|---|---|---|---|---|---|---|---|
| 3 | uniform (per-step remat) | 64 | 14.4 s | 0.223 s | 7.4 | 15.0 GiB | 45 min |
| 3 | uniform | 256 | 56.7 s | 0.221 s | 7.4 | 16.0 GiB | 44 min |
| 3 | binomial K=12 | 64 | 19.2 s | 0.300 s | 7.9 | 15.0 GiB | 57 min |
| 3 | binomial K=12 | 256 | 118.4 s | 0.463 s | 12.1 | 15.0 GiB | 53 min |
| 4, 5, 6 | binomial K=12 | 64 | OOM in the FORWARD: one allocation of 59 / 237 / 948 GiB (~23 MB per column) | | | | |

Forward: 30-38 ms/step at 642 columns (GPU mostly idle).  Uniform is linear
in window length (+5.3 MB/step); binomial recompute grows a level per
binomial(K+t, t) threshold.  Extrapolation to a 7-day window (5376 steps),
res 3, PLAUSIBLE: uniform ~20 min/gradient at ~44 GiB peak; binomial K=12
~70 min.  The 23 MB/column forward buffer is UNMEASURED (needs a memory
profile) and blocks res >= 4 on one GPU.

## Review of this branch (fresh-context Claude agent, 2026-11-11; codex and GLM unavailable -> UNREVIEWED by the mandated pair)

| finding | severity | action |
|---|---|---|
| scheduled loop did not advance `model_seconds` (the ecrad_ifs McICA seed clock); unrolled loop does | high | `clock_tables` builds every clock key `_clock` advances, from the same sources; test compares per step |
| `--stash-grads` with `--at-step > 0` wrote rows out of bounds, printer crashed on empty rows | medium | rows indexed `step - step0`; printer tolerates unrecorded rows; test at step0 = 16 |
| logged SHA was the probe script's checkout, not the imported code | medium | `CODE` header line: each tested module's imported path, its checkout's HEAD and dirty-path count |
| docstring overstated scheduled-loop memory under reverse-mode `lax.switch` | medium | docstring corrected (one step's residuals = union of the used variants' residual shapes); measured peak stays flat (table above) |
| `--load-state` did not check step / digest | low | refuses a step mismatch or a digest mismatch |
| `SCHED_VS_UNROLLED` compared a step0 = 0 window while differentiating `--at-step` | low | both use `--at-step` |
| timing stage compiled the gradient a third time | low | `time_compiled` returns the last result; reused |
| flags that silently did nothing | low | `mpas_segment` refuses checkpoints/storage without a schedule and checkpoints with uniform; probe refuses `--stash-modules` / `--stash-grads` outside their stage; module-tap overflow warns |
| launcher exited 0 when Python failed | low | exits with Python's status |
| no scheduled test with step0 != 0 or a strict variant subset; none with `model_seconds` | gap | both added |

Not changed, recorded:
- Clock-table dtype: the unrolled clock is a weak-typed f64 scalar, the table
  entry a strong f64; identical in fp64 (all tests), fp32-lane parity untested.
- `eqn_trace`'s self-check proves the summed gradient, not the attribution:
  work hidden inside one opaque scan/while/cond/custom-derivative equation is
  attributed to that equation.  It uses `jax._src` internals (version-fragile).
- `ADJOINT max` is a max over leaves of mixed units; a jump can reflect a change
  of dominant leaf.  The decisive step-16 jump was cross-checked by the module
  taps and the per-equation trace.
- Binomial forward drift (4.7e-3 K at 128 steps vs 5e-6 K uniform) is
  unexplained (PLAUSIBLE: recompute-order roundoff; not measured).
- `schedule` / `checkpoints` / `storage` are API-level knobs no config selects;
  their defaults (`sqrt_checkpoints`, `"recompute"`) are those of the existing
  `checkpoint_schedule.checkpointed_loop`, reused unchanged.
