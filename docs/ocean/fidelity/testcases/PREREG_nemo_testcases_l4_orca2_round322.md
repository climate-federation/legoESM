# PREREG — ORCA2 round 322: rn_Ud trainable entry (DECISION 149)

Frozen before the rung-10 gradient runs (Claude round; codex review: operator).
Test: `tests/ocean/fidelity/test_orca2_round322_rn_ud_trainable.py`
(rung-10 record, N = 1, T loss of round 315, CPU, fp64, libm, production step
body with rn_Ud spliced from the collector's raw leaf).

Change: rn_Ud (the isoneutral diffusive velocity) gets a spec entry, bounds
0.0033-0.03 m/s = kappa_Redi's 330-3000 m2/s through aht0 = 1/2 rn_Ud rn_Ld
at rn_Ld = 200 km; it is collected only when a run names it with a start
value. kappa_Redi is listed as frozen for rung-10 trainers. No card's
forward configuration changes.

Predictions:
- P1 no tier ("core", "extended", "aggressive") returns rn_Ud.
- P2 an explicit request with start value 0.018 returns exactly rn_Ud;
  without a start value it raises.
- P3 rung 10, N = 1: the T-loss gradient with respect to rn_Ud's raw leaf
  is finite and nonzero.
- P4 converted to a relative change of rn_Ud it equals the round-320/321
  value -4.5e-5 within 1 %, negative.

Falsifiers: any tier returns rn_Ud; a request without a start value
builds; the gradient is 0 or non-finite; P4 off by more than 1 % or sign
flipped (then the splice path differs from the scale path and is reported,
not landed).
