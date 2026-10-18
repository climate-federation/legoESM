# ORCA2 round 315 — differentiability of the NEMO-exact ORCA2 step (measurement only)

Base `64f0dd0d37f5`. No model file changed; no forward value of any card
changed. Prereg: `PREREG_nemo_testcases_l4_orca2_round315.md` (committed
before the first measurement). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round315/`
(`runs/<card>.json` per card, `locate/` for the NaN locator; every JSON
carries its worktree stamp, clean, at the probe commit named in it).

## Verdict
- **Rung 0 and OMT-4 (every card running tracer advection): the gradient
  with respect to the initial state is NaN at every wet and dry point
  already at N = 1** (799,200 of 799,200 T entries; SSH loss the same). The
  first NaN-producing statement is the stage-1/2 centred tracer tendency,
  which divides by the live cell thickness, exactly zero on dry levels.
- **OMT-1..3 (no tracer advection): finite and nonzero at every wet cell,
  zero on every dry cell, N = 1..10**; the SSH-loss gradient agrees with
  finite differences to 9.5e-9; the T-loss gradient misses the 1e-6 bar
  (best 1.5e-3 .. 1.1e-2) with an error that shrinks about tenfold per
  decade of epsilon — the signature of value switches crossed by the
  perturbation (PLAUSIBLE; not separable here, see below).
- **The bottom-drag coefficient cannot be passed traced** on rung 0,
  OMT-2..4 and rung 10: the linear-drag face rate converts it with Python
  `float()`.
- **The isoneutral diffusivity is inert on rung 0 and OMT-1..4** (AD 0, FD
  0, rebuilt-model FD 0 at every N): those cards do not run it.
- **Forward mode is unavailable on every card**: the implicit vertical
  solve is a `custom_vjp` Thomas solver (no JVP rule), so the planned
  reverse-vs-forward (kink-immune) check could not run; FD is the only
  cross-check.
- Rung 10: see its section (reverse pass with per-step rematerialization).

## Instrument
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round315_grad_probe.py`
(direct test `tests/ocean/fidelity/test_orca2_round315_grad_probe.py`, 6
tests: chained reverse matches an analytic gradient to 1e-13 with and
without remat; a planted 2x pullback fails the FD gate; a `custom_vjp` step
yields "forward mode unavailable", not a crash). Losses: area-weighted mean
SSH^2 over wet T points; volume-weighted mean T^2 (static reference volume).
Scalars are multiplicative scales s (= 1) on the isoneutral diffusivity
(1000 m2/s), the 3-D lateral momentum viscosity arrays (max 4.0e4 m2/s), the
background vertical diffusivity (constant-mixing cards: 1.2e-5 m2/s; rung 10:
the TKE floor avtb = 1e-10 m2/s, which internal-wave mixing replaces the
namelist background with) and the linear drag coefficient (1e-3). Reverse
mode = one `jax.vjp` per production step body under the probe's own
`jax.jit`, chained by hand. x64, CPU, fp64/libm policy, production JIT.

Controls, all run:
- C1 forward parity: the traced-scale path at s = 1 is bit-identical to
  production `model.step` at every kt, every leaf, on all six cards.
- C2 construction parity: the traced path at s = 1 + 1e-5 is bit-identical
  to a model REBUILT with the scaled configuration, for every scalar whose
  rebuilt run completed (isoneutral and viscosity on OMT-1..4; the rest in
  the table) — no scalar is folded at construction.
- C3 the codegen flag below changed nothing: production losses at N =
  1/2/5/10 bit-equal with and without it on all six cards.

Infrastructure findings (each cost a relaunch; none is a model defect):
- One reverse-mode executable of the full step holds ~22,000 memory maps
  (forward: ~4,000); the vjp of steps 1 and 2..N plus the forward
  executables exceeded the machine's 65,530 per-process limit ("LLVM: cannot
  allocate memory", exit 134, all six cards). The probe now clears compile
  caches between stages (peak ~9,000 maps after) and runs with
  `--xla_cpu_parallel_codegen_split_count=1`.
- Reverse-mode compile: ~8 min per state structure (step 1 has its own, the
  cold-start state) plus ~5.5 min per transpose.
- Rebuilt-model compiles late in a process intermittently fail with XLA's
  "failed to materialize symbols" (OMT-1, OMT-2, OMT-4 lost their last
  scalar rows); the probe now clears caches around each rebuild and records
  the error instead of dying; OMT-2 was re-run for its scalar rows.
- Headless rule 5 deviation: one reverse compile exceeds the 10-minute
  foreground limit, so the cards ran detached (`setsid nohup`) and were
  polled in the foreground; nothing was left running unobserved.

## Rung 0 and OMT-4 — NaN at N = 1

| card | N | dL_T/dT0 | dL_ssh/dT0 | d/d(visc) | d/d(Kv bg), T loss | backward |
|---|---|---|---|---|---|---|
| rung 0 | 1 | all NaN | all NaN | NaN | -4.512e-5 | 7 s |
| rung 0 | 2, 5, 10 | all NaN | all NaN | NaN | NaN | 14 / 43 / 72 s |
| OMT-4 | 1 | all NaN | all NaN | NaN | -4.512e-5 | 6 s |
| OMT-4 | 2, 5, 10 | all NaN | all NaN | NaN | NaN | 15 / 33 / 71 s |

Every float leaf of the state cotangent is non-finite (u, v, T, S, SSH,
bathymetry, masks, barotropic velocities). The background-diffusivity
derivative is still finite at N = 1 (why is not measured); from N = 2 on it
is NaN too.

First statement (NaN locator,
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round315_nan_locate.py`,
`jax_debug_nans` on before the first dispatch, one step, rung 0, d L_T / dT0):
`ocean_model_latlon_cgrid.py:1933`, `rhs = sr(-sr(horizontal_scaled / e3t_kmm))`
inside the stage-1/2 CEN2 tracer accumulator, reached from the stage-1
tracer pair step. Mechanism: `e3t_kmm` is the live T thickness, exactly 0
on dry levels; the forward gives 0/0 = NaN there (later masked, so forward
values are unaffected); in the backward pass the masked lanes' zero
cotangent is divided by the same zero (0/0 = NaN), and the next flux
difference spreads it to every neighbouring wet cell — the whole field in
one step. Class: reciprocal of zero on a dry cell under a where-mask.
CONFIRMED for the first statement only: there may be later ones (e.g. the
stage-3 FCT limiter) that this one hides.

The locator's first run was vacuous (debug flag switched on after the
compiled function was cached, so the cached fast path skipped the check and
it returned "raised: false" on an all-NaN gradient); kept under
`locate_vacuous/`, fixed, re-run.

## OMT-1..3 — finite

| N | \|dL_T/dT0\| (OMT-1 / 2 / 3) | \|dL_ssh/dT0\| (OMT-2) | (OMT-3) | backward |
|---|---|---|---|---|
| 1 | 1.9340e-2 (all three) | 6.859e-4 | 6.859e-4 | 6-8 s |
| 2 | 1.9338e-2 | 1.537e-3 | 1.524e-3 | 11-14 s |
| 5 | 1.9335e-2 | 3.843e-3 | 3.704e-3 | 34-42 s |
| 10 | 1.9337e-2 (OMT-2/3; OMT-1 has 8 steps) | 7.652e-3 | 6.715e-3 | 75-80 s |

Gradient norm vs N: the T-loss gradient is flat (1.934e-2 -> 1.934e-2 over
ten 3-hour steps); the SSH-loss gradient grows about linearly (x11 from N = 1
to 10). Nonzero on 100% of wet cells, exactly zero on every dry cell.

FD along one random unit-RMS direction (epsilon 1e-4 and 1e-3 K; OMT-2
shown; OMT-1 and OMT-3 within a factor 2 at every entry):

| loss | N = 1 | N = 2 | N = 5 | N = 10 |
|---|---|---|---|---|
| SSH | 9.5e-9 PASS | 6.9e-3 | 1.6e-2 | 3.6e-3 |
| T | 1.5e-3 | 1.1e-2 | 6.4e-3 | 6.7e-3 |

(best relative error of the two epsilons). At N = 1 the T-loss error is
1.2e-2 at 1e-3 and 1.5e-3 at 1e-4: about linear in epsilon, which is what
branch switches crossed by a perturbation give (their count grows with
epsilon) and not what a wrong gradient gives (an epsilon-independent floor).
PLAUSIBLE only: two epsilons, and forward mode — the check that would
separate a kink from a wrong gradient — is unavailable.

Scalars (OMT-3 runs lateral momentum diffusion; OMT-1/2 do not):
- viscosity: OMT-3 SSH loss AD vs FD 1.7e-10 (N = 2), 6.0e-9 (5), 2.8e-8
  (10) PASS; T loss 2.5e-5 / 1.1e-5 / 5.4e-7 (the T dependence is 4e-10 of
  the loss, at the FD rounding floor). AD = 0 = FD on OMT-1/2.
- isoneutral: 0 = 0 = rebuilt 0 on every card and N (inert there).
- background vertical diffusivity: AD -4.512e-5 (N = 1) .. -4.504e-4
  (N = 10) for the T loss, identical on OMT-1..3; FD: OMT-2 re-run below.
- drag coefficient: untraceable on OMT-2..4; traceable on OMT-1 (AD 0, its
  rebuilt FD lost to the XLA error).

## Rung 10 (full card)
PENDING — filled from `runs/rung10.json` when the run lands.

## Peak memory and wall time
Reverse pass without rematerialization keeps every step's intermediates:
process peak 73 GB (OMT-1, 5 steps), 125-132 GB (OMT-2/3, 10 steps), 160 GB
(rung 0 / OMT-4, 10 steps). The first rung-10 attempt (10 steps, no remat)
was killed by the out-of-memory killer at its tenth vjp step (exit 137, six
probes then held ~730 GB together). Backward wall time is ~3x the forward
per step (6-8 s against ~2 s), linear in N; compile excluded.

## DECISION_NEEDED (remedies, none applied — gradient-only, forward bit-identical)
1. Dry-cell division (`ocean_model_latlon_cgrid.py:1933`): divide by
   `where(wet, e3t, 1)` on wet lanes and keep the original quotient on dry
   lanes behind `stop_gradient` — forward bits unchanged by IEEE (same
   operands on wet lanes, same NaN on dry). Cost: two selects per stage
   call; the helper is in the WS-RK3 identity GYRE also executes, so it is
   a GYRE landing (Decision 43+45 gate) and the seven-pair D122 gate. Pick:
   yes, one guarded helper for every such division, found by re-running the
   locator after each.
2. Drag coefficient `float()` (`ocean_pe_latlon_cgrid.py:4294`, and its
   siblings): pass the config value through without `float()`. Cost: a
   traced multiplier where XLA saw a constant, so bit-identity must be
   re-proven on the cards (constant folding may differ). Pick: yes.
3. No forward mode through the Thomas solver: add a JVP rule (or switch the
   implicit solve to `custom_linear_solve`) only if forward-mode checks are
   wanted. Cost: a second derivative rule to keep consistent. Pick: defer.

## OPEN
- Rung 10 numbers (pending section above).
- Later NaN statements hidden behind the first (re-run the locator after
  remedy 1 lands).
- A six-epsilon T-direction sweep on a finite card (OMT-2) to turn the
  kink reading from PLAUSIBLE into a measurement.
- The probe chain reports non-finite land-mask and V-mask cotangents on
  OMT-1..3; a single fused `jit(grad)` of one OMT-1 step reports none.
  Unreconciled; masks are not trainable inputs.

codex review: operator
