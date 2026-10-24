# ORCA2X round 2 — training readiness: preregistration (frozen before measuring)

Card: rung 10 as the NEMO comparison runs it, from its own initial state
(round 329/330). Path: the model's `integrate_scan` with per-step inputs,
`nemo_kt0 = 1`. Trainable: rn_Ud through the rung-10 trainer entry
(DECISION 149), start 0.018. Losses: round 315's T^2 and SSH^2 of the final
state. N = 1, 2, 5, 10; per-step `jax.checkpoint` on and off; one N per
process, each stopped at 100 GiB resident or below 150 GiB available.
Probe: `nemo_testcase_l4_orca2x_r2_multistep_grad.py`.

Predictions (priors: round 317 on the pre-L8 card, round 330 at N = 1):
- P1: 0 non-finite cotangents (every initial-state float leaf and rn_Ud),
  both losses, every N, both modes. Falsifier: any non-finite entry.
- P2: reverse vs forward along T_dir0 <= 1e-13 at N = 1; at N >= 2 up to
  ~3e-5 (round 317: 2.6e-5). Falsifier: > 1e-3 at any N.
- P3: rn_Ud reverse vs forward <= 1e-12 at every N; best central FD
  <= 1e-5 at N = 1. N >= 2 FD: no prediction (switches).
- P4: |dL_T/dT0| flat within 0.1 % from N = 1 to 10 (~1.93e-2);
  |dL_SSH/dT0| grows ~5-12x by N = 10.
- P5: remat N = 10 peak < 100 GiB (round 317: 89 GiB); no remat: N = 2
  fits (round 317: 85 GB), N >= 5 does not.
- P6: program A's primal equals production `model.step` losses bit for bit
  at N = 1; N >= 2: last-bit differences possible (round 317).
