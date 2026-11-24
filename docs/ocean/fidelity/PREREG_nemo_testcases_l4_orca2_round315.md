# PREREG — ORCA2 round 315: differentiability of the NEMO-exact ORCA2 step

Frozen before any measurement (base `64f0dd0d37f5`). MEASUREMENT ONLY: no model
file changes, no forward value of any card changes.

## Instrument
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round315_grad_probe.py`
(x64, CPU, production JIT, fp64/libm policy as every ORCA2 gate). Cards: rung 0
(dynamics only, record kt=1 entry), OMT-1..4 (their own entries; OMT-1 has 8
steps), rung 10 (full card, own entry, record surface forcing). N = 1, 2, 5, 10.
Losses: area-weighted mean SSH^2 (wet T points); volume-weighted mean T^2
(static reference volume). Inputs: initial T; multiplicative scales (=1, traced)
on the isoneutral diffusivity (`kappa_Redi`), the lateral momentum viscosity
(3-D `ahmt`/`ahmf`), the background vertical diffusivity (rung 10: TKE floor
`kappaH_min`; constant-mixing cards: `K_v`), the linear bottom-drag `cd0`.

Controls (each must pass before a gradient number is quoted):
- C1 forward parity: traced-scale path at s=1 bit-identical to production
  `model.step` at every kt.
- C2 construction parity: traced path at s=1+eps bit-identical to a model
  REBUILT with the scaled config (else the scalar is folded at construction).
- C3 the chained-vjp machinery is checked on a toy step with an analytic
  gradient in the probe's direct test (and fails on a planted wrong pullback).

Acceptance: FD best-over-sweep relative error <= 1e-6 (fp64 central
differences: truncation and rounding both ~1e-8 mid-sweep for a smooth loss);
reverse-vs-forward-mode relative error <= 1e-10.

## Predictions (frozen)
- P1 rung 0 and OMT cards: every gradient finite at every N. Falsifier: any
  NaN/Inf.
- P2 rung 10: at least one non-finite gradient (PLAUSIBLE source: a square root
  or a magnitude at exactly zero — friction velocity under zero stress, TKE
  sqrt, drag |u| — whose derivative is infinite, multiplied by a zero).
  Falsifier: all finite.
- P3 reverse == forward mode within 1e-10 on every card where finite.
  Falsifier: any finite disagreement (that would be a wrong custom rule /
  stop_gradient, not a kink).
- P4 FD agreement within 1e-6 at N=1 for T directions on rung 0; at N=10 on
  rung 10 at least one FD miss while reverse == forward (a limiter / max-min
  kink: FCT, TKE floors). Falsifier: rung-0 N=1 miss, or rung-10 all pass.
- P5 C1 holds on every card. Falsifier: any non-bit leaf.
- P6 C2 holds for kappa_Redi, kv_background, bottom_cd0, visc_scale.
  Falsifier: any scalar where rebuilt != traced (folded at construction).
- P7 nonzero d/ds for all four scalars on rung 10; on rung 0 / OMT the
  isoneutral gradient is zero if the card does not run isoneutral diffusion.
- P8 |dL/dT0| varies by less than 10x between N=1 and N=10 (ten 3-h steps).
- P9 backward wall time per N < 10x the forward time of N steps; peak memory
  grows ~linearly in N.

Failed predictions are recorded REFUTED and kept.

## Resume addendum (2026-10-18, after the two HOLD reviews) — frozen before measuring
Instruments: `nan_locate.py --backward` (walks one step's pullback equation by
equation; records every source of a non-finite cotangent; raw and sanitized;
dense "chained" cotangent and symbolic-zero "fused" cotangent) and
`grad_probe.py --fd-sweep` (T_dir0, eps 1e-3..1e-8 K, against forward mode).
Cards: rung 0 and OMT-1 for the walker (rung 10 if memory allows); rung 0 and
OMT-1..4 for the sweep.
- R1 control: walker raw/chained reproduces the compiled pullback (same
  finite mask on every leaf; finite entries to <= 1e-10 relative).
  Falsifier: any mismatch -> walker numbers not quoted.
- R2 control: rung 0 fused SSH (raw) is finite with |dT0| = 6.86e-4 (the
  review's compiled jit(grad)). Falsifier: NaN or a different norm.
- R3 prediction: rung 0 chained T (raw) first source is in the stage-1/2
  tracer path and is class "generated" (finite 0 cotangent / 0 thickness),
  not the forward `neg`; more than one independent source exists.
- R4 completeness: rung 0 sanitized dT0 . T_dir0 equals forward mode
  (-4.2080326157568834e-2 T loss, 4.4076248944997483e-4 SSH, N = 1) to <= 1e-10.
  Falsifier: a miss -> the source list is incomplete or zeroing is wrong.
- R5: OMT-1 chained non-finite mask cotangents come from sources that the
  fused run does not reach (fused: none).
- R6 sweep: FD error falls ~linearly with eps until the rounding floor; at
  N = 1 every card reaches <= 1e-6 at some eps; at N >= 2 PLAUSIBLE either way.
