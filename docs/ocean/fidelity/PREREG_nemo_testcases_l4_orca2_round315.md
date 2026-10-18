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
