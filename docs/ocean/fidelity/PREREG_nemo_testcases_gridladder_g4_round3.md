# Grid ladder G4 round 3 (gridladder_r19) — preregistration

Frozen before the acquisition runs. Scope: build the per-step surface-forcing
record both answers to Decision 123 need; land nothing in the model; do not
wire forcing into the card.

## What is acquired

NEMO's own surface forcing at kt = 1..10, written by the G1 step writer
(new frames only) in a new record build (P5's sources + four additive patches):

- frame `s`, after `CALL sbc` (`stprk3.F90` of P5, compiled `stprk3.f90:142`):
  utau, vtau, utauU, vtauV, taum, wndm, qns, qsr, emp, sfx, rnf, erp, qrp,
  rnf_tsc (T, S), h_rnf, fr_i and the surface means. Nothing writes these
  between `sbc` and their readers (grep of the compiled tree; the only
  candidate, `trasbc.f90:269-270`, runs only when `ln_traqsr = F`).
- frame `c`, inside `dyn_spg_ts` after the copies it reads (`dynspg_ts.f90:274-276`).
- frames `n`, `A`..`E`, `k` on bulk steps (kt odd): turb_ncar entry, every
  NCAR iteration (1/L, clamped zeta, psi_m, floored U_N10, coefficients,
  stability flag) and the ocean bulk after the stress.

## Predictions (falsifier in brackets)

- R3-P1 preflight passes on the committed tree; the compiled sources differ
  from P5's only by writer lines [any other compiled difference].
- R3-P2 smoke and twins a, b, c finish (2 and 10 steps) [any stop].
- R3-P3 passivity: every round-17 step-frame field (kt 1..10), substep
  stream (kt 1, 2) and restart is byte-identical [one differing word].
- R3-P4 the forcing frames exist exactly as planned (`s`, `c` every step;
  `n`, `A`..`E`, `k` on kt 1, 3, 5, 7, 9), finite and twin-equal on the
  consumed support [a missing/extra frame, an inner non-finite or unequal word].
- R3-P5 (instrument identities, checked when the record is read next round):
  `c_ssh_frc` = r1_rho0 (s_emp − s_rnf) bit for bit (`stp2d.f90:242-245`);
  `k_utau` = `s_utau` on the inner domain at bulk steps; `s_` stress at an
  even kt equals the odd kt before it (held, nn_fsbc = 2) [any inequality:
  the program-point claim is wrong].

Expected kt = 11 stop: not reached (nn_itend = 10, the round-17 boundary rule).
