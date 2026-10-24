# ORCA2 round 315 — differentiability of the NEMO-exact ORCA2 step (measurement only)

Base `64f0dd0d37f5`. No model file changed; no forward value of any card
changed. Prereg: `PREREG_nemo_testcases_l4_orca2_round315.md` (committed
before the first measurement). Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round315/`
(`runs/<card>.json` per card, `locate/` for the NaN locator; every JSON
carries its worktree stamp, clean, at the probe commit named in it).

## Verdict (corrected on resume, after two HOLD reviews)
- **Rung 0 and OMT-4 (every card running tracer advection): the reverse-mode T-loss
  gradient with respect to the initial state is NaN at every point already at
  N = 1 (SSH loss: NaN only when hand-chained; fused it is finite, 6.859e-4).** Cause, located on resume (section "Backward NaN sources"): three
  divisions by a thickness that is exactly zero on dry cells in the stage-1/2
  tracer update. Their forward 0/0 is masked out, but the reverse pass
  multiplies or divides a zero cotangent by it. Zeroing those sources (and
  nothing else) gives a finite gradient equal to forward mode to 3e-14.
- **OMT-1..3: finite.** Reverse equals forward mode to <= 1.6e-13 on every
  T-direction and scalar row, N = 1..10, so the reverse gradient is the exact
  derivative of the branch the run takes (CONFIRMED). FD agrees to <= 1e-6
  only in a few cells of the sweep table below; elsewhere the FD error falls
  roughly in proportion to epsilon until it meets the rounding floor near
  1e-7 (non-smooth branch switches: PLAUSIBLE, not located).
- **Forward mode runs on every card** (the earlier "unavailable everywhere"
  was wrong: it came from the solver's docstring, not a run). It is finite
  on rung 0 and OMT-1..4. It is NaN only on rung 10: the T loss at every N,
  the SSH loss from N = 2.
- **The bottom-drag coefficient cannot be passed traced** on rung 0, OMT-2..4
  and rung 10: the NEMO face-rate path converts it, and its four sibling
  parameters, with Python `float()`.
- **The isoneutral diffusivity is inert on rung 0 and OMT-1..4** (AD 0, FD
  0, rebuilt-model FD 0 at every N): those cards do not run it.
- Rung 10: reverse NaN at every N (section below); its one-step sources are
  the rung-0 ones plus four in internal-wave mixing, TKE and shortwave
  penetration.

## Instrument
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round315_grad_probe.py`
(direct test `tests/ocean/fidelity/test_orca2_round315_grad_probe.py`, 6
tests: chained reverse matches an analytic gradient to 1e-13 with and
without remat; a planted 2x pullback fails the FD gate; a `custom_vjp` step
yields "forward mode unavailable", not a crash — a synthetic case; no card's
step holds one: every card ran forward mode). Losses: area-weighted mean
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

| card | N | dL_T/dT0 | dL_ssh/dT0 (hand-chained probe; fused N=1 is finite) | d/d(visc) | d/d(Kv bg), T loss | backward |
|---|---|---|---|---|---|---|
| rung 0 | 1 | all NaN | all NaN | NaN | -4.512e-5 | 7 s |
| rung 0 | 2, 5, 10 | all NaN | all NaN | NaN | NaN | 14 / 43 / 72 s |
| OMT-4 | 1 | all NaN | all NaN | NaN | -4.512e-5 | 6 s |
| OMT-4 | 2, 5, 10 | all NaN | all NaN | NaN | NaN | 15 / 33 / 71 s |

Every float leaf of the state cotangent is non-finite (u, v, T, S, SSH,
bathymetry, masks, barotropic velocities). The background-diffusivity
derivative is still finite at N = 1: the reverse pass collects it in the
vertical-diffusion solve, which it reaches before the tracer-advection
sources below (PLAUSIBLE, read off the walker's order); from N = 2 on the
cotangent arriving from the later step is already NaN.

The first locator (`jax_debug_nans`) stopped on a `neg` at
`ocean_model_latlon_cgrid.py:1936`. That is the FORWARD 0/0 (later masked,
kept as a residual); a `neg` cannot create a NaN, and that run did not show
where the backward pass first goes non-finite. Superseded by the walker below.

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

FD along T_dir0 (one random unit-RMS direction), central differences at
epsilon 1e-3 .. 1e-8 K, against FORWARD mode (finite on all five cards, so
rung 0 and OMT-4 are covered too). Best relative error, epsilon in
brackets; a tick = <= 1e-6 (`resume/sweep_<card>.json`, commit `9fe71ac67`):

| card | T N=1 | T N=2 | T N=5 | T N=10 | SSH N=1 | SSH N=2 | SSH N=5 | SSH N=10 |
|---|---|---|---|---|---|---|---|---|
| rung 0 | 9.4e-7 (1e-7) ✓ | 3.3e-6 | 1.1e-5 | 4.0e-6 | 9.5e-9 (1e-4) ✓ | 1.3e-5 | 1.8e-6 | 2.2e-6 |
| OMT-1 | 1.4e-5 (1e-7) | 3.0e-6 | 2.4e-6 | — | 9.3e-9 (1e-4) ✓ | 1.2e-5 | 2.2e-6 | — |
| OMT-2 | 7.8e-7 (1e-7) ✓ | 8.3e-6 | 3.0e-6 | 4.1e-6 | 9.5e-9 (1e-4) ✓ | 1.8e-5 | 1.1e-6 | 1.2e-7 (1e-7) ✓ |
| OMT-3 | 7.8e-7 (1e-7) ✓ | 1.1e-5 | 8.7e-6 | 9.9e-6 | 9.5e-9 (1e-4) ✓ | 2.0e-5 | 1.8e-6 | 2.3e-6 |
| OMT-4 | 9.4e-7 (1e-7) ✓ | 3.3e-6 | 1.1e-5 | 4.0e-6 | 9.5e-9 (1e-4) ✓ | 1.3e-5 | 1.8e-6 | 2.2e-6 |

(untagged best epsilons are 1e-7, except OMT-3 SSH N = 10 at 1e-8). The
first round's "9.5e-9 SSH agreement" holds at N = 1 only. The T loss at
N = 1 on OMT-2 falls 1.2e-2, 1.5e-3, 2.0e-4, 3.1e-5, 7.8e-7 from 1e-3 to
1e-7 (about proportional to epsilon, the shape a crossed branch switch
gives), then 6.7e-5 at 1e-8 (rounding). At N >= 2 the two errors meet above
1e-6, so FD cannot certify these rows either way; forward mode (equal to
reverse) is the check that carries them. The review's OMT-2 numbers are
reproduced (N = 1 T 7.8e-7; N = 2 8.3e-6 T, 1.8e-5 SSH).

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
Measured (`runs/rung10.json`, probe `dbf8822a3fb9`, per-step remat): C1
forward parity bit-identical kt 1..10; drag coefficient untraceable (same
statement); reverse NaN in every state leaf for both losses at N = 1, 2, 5,
10; forward mode NaN for the T loss at every N and for SSH from N = 2 (SSH
N = 1: forward 4.3608627e-4, FD 4.3608627e-4 at 1e-4, 8.6e-9 apart). The
scalar rows were still running at this commit (that detached process was
left alone). One-step walker: sources in the section below; with them zeroed
dT0 . T_dir0 = -4.17882e-2 against FD -4.18476e-2 (1e-4) — no exact
reference here because forward mode is NaN, so PLAUSIBLE only.

## Backward NaN sources (resume)
Instrument: `nan_locate.py --backward` (commit `9fe71ac67`;
`resume/walk_<card>.json`). It evaluates one step's reverse pass operation
by operation and records every operation whose cotangent output turns
non-finite while its cotangent inputs are finite: "generated" (all inputs
finite: zero cotangent / zero thickness) or "residual" (a masked forward
0/0 kept from the forward pass meets the cotangent). "Sanitized" zeroes each
recorded output and continues, so later sources are not hidden. "Fused"
passes the cotangents of state fields the loss does not use as symbolic
zeros, which is what one `jit(grad)` does. Controls, all run:
- the raw walk reproduces the compiled pullback: same finite pattern on
  every leaf (rung 0, OMT-1, rung 10); finite entries to 4.8e-13 (OMT-1);
- rung 0 fused SSH, raw: finite, |dT0| = 6.859e-4 — the review's compiled
  one-step `jit(grad)` (6.86e-4); dT0 . T_dir0 equals forward mode to 2.6e-14;
- completeness: rung 0 sanitized T loss dT0 . T_dir0 = -4.2080326157567446e-2
  against forward mode -4.2080326157568834e-2 (3.3e-14): the list below is
  every source of the one-step T-loss NaN, and zeroing them gives the exact
  gradient;
- the walker's direct tests fail when sanitizing or the symbolic-zero rule
  is removed.

Rung 0, one step (OMT-4 runs the same path), in reverse-pass order:

| # | statement | what is zero | lanes | class | reaches |
|---|---|---|---|---|---|
| 1 | `implicit_solver.py:363-382` | pivot of the vertical momentum solve, one row of V faces | 180 | residual | land- and V-mask cotangents only |
| 2 | `ocean_model_latlon_cgrid.py:13455` | top V-face thickness in the surface-stress division, same row | 180 | generated + residual | the same two masks only |
| 3 | `:2568-2570` | `q_after` in the stage-1/2 tracer combine (masked by `:2576`) | 368,648 | residual | every leaf |
| 4 | `:1946` | `e3t_kmm` in the vertical tracer flux divergence | 368,648 | generated + residual | every leaf |
| 5 | `:1933` | `e3t_kmm` in the horizontal tracer flux divergence | 368,648 | generated + residual | every leaf |

368,648 = the dry T cells. Sources 3-5 fire for both tracers in stages 1
and 2. "Reaches" for 1-2 is measured on OMT-1, which has them and nothing
else: its T and SSH gradients stay finite and only the two mask cotangents
(180 entries each) go NaN.
- So guarding `:1933` alone leaves `:1946` and the `:2568-2570` division —
  the review's still-all-NaN result. The first non-finite value in reverse
  order is source 1 (the vertical momentum solve, the step's last forward
  operation); the first that reaches T is source 3.
- Chained vs fused: the hand-chained probe hands the step a cotangent with
  materialized ZEROS on every field the loss does not use. Zero times a
  masked forward NaN is NaN. On OMT-1 that is the mask NaNs (fused: none);
  on rung 0 SSH it is the T path (fused SSH: finite, chained: all NaN).
- Training: a training loop takes one fused gradient and never trains the
  masks, so sources 1-2 are harmless unless the masks sit inside the
  differentiated pytree (e.g. a global-norm clip over the whole state).
  Sources 3-5 are not: the fused rung-0 T-loss gradient is NaN (measured).
  Any multi-step loss lets T feed SSH through density, so a fused multi-step
  SSH gradient should hit them too (PLAUSIBLE, not run).

Rung 10 (one step) has 1-5 plus:

| # | statement | what | lanes | class |
|---|---|---|---|---|
| 6 | `internal_wave_mixing.py:444` | square root of the clipped N^2: infinite slope where N^2 <= 0 | 359,826 (60,973 inf) | residual |
| 7 | `tke.py:1082` | square root in the TKE mixing length | 358,441 | residual |
| 8 | `tke.py:3670` | square root over the dissipation length | 358,441 | residual |
| 9 | `shortwave_penetration.py:561` | level thickness below the bottom (masked) | 21,883 | generated + residual |

6, 7 and 9 also fire in the fused T-loss gradient. With 1-9 zeroed the
rung-10 gradient is finite (dT0 . T_dir0 = -4.17882e-2; FD -4.18476e-2 at
1e-4); no exact reference, because rung 10's forward mode is NaN (that
forward-mode NaN is not located: OPEN).

## Peak memory and wall time
Reverse pass without rematerialization keeps every step's intermediates:
process peak 73 GB (OMT-1, 5 steps), 125-132 GB (OMT-2/3, 10 steps), 160 GB
(rung 0 / OMT-4, 10 steps). The first rung-10 attempt (10 steps, no remat)
was killed by the out-of-memory killer at its tenth vjp step (exit 137, six
probes then held ~730 GB together). Backward wall time is ~3x the forward
per step (6-8 s against ~2 s), linear in N; compile excluded.

## DECISION_NEEDED (gradient-only remedies, none applied)
Each must keep every forward value bit-identical, re-proven by C1 on all six
cards; the tracer helpers are in the WS-RK3 path GYRE also runs, so the GYRE
(Decision 43+45) and D122 seven-pair gates apply too.
1. Sources 3-5 (`ocean_model_latlon_cgrid.py:1936`, `:1946`, `:2568-2570`):
   divide by a denominator equal to 1 on dry lanes and the live value on wet
   lanes (same operands on wet lanes; dry lanes stay masked). Measured: zeroing
   exactly these makes rung 0 finite and equal to forward mode. Pick: yes.
2. Sources 1-2 (`implicit_solver.py:363-382`, `ocean_model_latlon_cgrid.py:13455`):
   same guard on the zero pivot / zero face thickness. Reach only the mask
   cotangents. Pick: yes, after 1.
3. Sources 6-9 (rung 10): a square root whose derivative is zero where its
   argument is clipped to zero, and the same dry-level guard in shortwave
   penetration. Check: rung-10 forward mode becomes finite, then reverse
   matches it. Pick: yes, after 1.
4. Drag casts `ocean_pe_latlon_cgrid.py:4294-4298` (cd0, cd_max, z0, ke0,
   uc0): pass the config values through without `float()`. The same five
   casts repeat in the generic drag path, `:4375-4379`, which these cards are
   not shown to run. Pick: yes (constant folding may change, so bits are
   re-proven).
5. The earlier "add a JVP rule to the Thomas solver" is withdrawn: no card
   uses it.

## Gates
Resume, at `6a1ebef604cc` (x64, CPU): default landing battery (land.sh T4 +
this probe's test, private-import ratchet included): 155 passed; private-import
ratchet alone: 6 passed; citation gate on this receipt: PASS, 12 citations,
0 failures (`resume/gate_default.log`, `resume/citation_gate_receipt.json`).
No model file changed in this round or its resume.

## OPEN
- Rung 10: forward-mode NaN (T loss every N, SSH from N = 2) not located;
  scalar rows (detached run, `runs/rung10.json`) still running.
- Multi-step fused gradient on rung 0 (PLAUSIBLE NaN from sources 3-5).
- Which branch switches make FD miss 1e-6 at N >= 2 (forward mode = reverse,
  so the gradient itself is not in doubt).

codex review: operator
