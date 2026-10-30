# PREREGISTRATION — TSUNAMI lane, round 3 (measure the RK3 record)

Date 2026-10-08. Lane tip at start `6b4b3ea33b83`. Frozen before the
geometry gate, the ladder harness and the offline replay exist.

## 0. Disclosure: what was seen before this was written

One throwaway probe (not committed, `/tmp`) ran the card at kt = 1 against
the record before this text. It saw: the card's kt = 1 end-of-step ssh, uu_b,
vv_b, u, v differ from NEMO's (max 1.2e-7 on ssh, 6.0e-8 on velocity); the
card's T stays 20 where NEMO's moves by 4.6e-3; each stage, handed NEMO's
external handoff and entry, is at the 1e-17 level on ssh/u/v and off by
1.5e-3 on T/S; NEMO's stp_2D right-hand side differs from the card's by
7.7e-11 (maximum 8.7e-8) on 346 cells; the first unequal boundary of the
card's own substep loop is the velocity update, whose loop-entry slow forcing
already differs by that amount. Predictions P3-P5 below are therefore about
kt = 2..10 and about attributions, not about kt = 1 alone.

## 1. Frozen predictions and falsifiers

- **P1 geometry identity.** Every mesh array the card carries equals the RK3
  run's `mesh_mask.nc` bit for bit: glamt/u/v/f, gphit/u/v/f, e1/e2 t/u/v/f,
  ff_t, ff_f, tmask, umask, vmask, fmask at the executed level, and
  e3t_1d, e3w_1d, gdept_1d, gdepw_1d at the executed level. Falsifier: one
  unequal element in any row. A plant (one element of one card array moved)
  must turn the gate RED.
- **P2 admission.** `check_records.py --program rk3` returns ADMITTED 110
  and its json is byte-identical to the operator's. Groups: kt = 1..10 carry
  60 step groups and an 8-substep record; kt = 11..100 carry the six `f_`
  groups only.
- **P3 (independent ladder).** Over kt = 1..10 the card's end-of-step state is
  never bit-identical to NEMO's on ssh, uu_b, vv_b, u, v; T and S are over
  the bar (1e-15 normalised) from kt = 1 because the card runs tracer
  advection where NEMO runs none (B7). Falsifier: any bit-identical kt for
  all of ssh, uu_b, vv_b, u, v, or T or S inside the bar at kt = 2.
- **P4 (given-NEMO-entry, whole step).** From NEMO's own entry the card's
  end-of-step error is owned, at every kt = 1..10, by the loop-entry slow
  forcing of the external mode (the stp_2D right-hand side): the card's
  substep loop handed NEMO's recorded entry forcing is bit-identical to
  NEMO's through every substep and boundary at every kt = 1..10. Falsifier:
  any first-unequal boundary in that arm.
- **P5 (stage-local).** With NEMO's external handoff and NEMO's stage entry,
  stages 1..3 are within the bar (1e-15) on ssh and velocity at every
  kt = 1..10, and T, S are over the bar by at least 1e-4 (B7). Falsifier:
  a velocity or ssh row over the bar, or T/S inside it.
- **P6 (offline replay).** A float64 replay of NEMO's `eos` (SEOS) and
  `hpg_sco` statements written from the compiled `ppsrc` source,
  fed NEMO's recorded entry ssh, T, S, reproduces NEMO's recorded stp_2D
  right-hand side at kt = 1 bitwise. If it does not, the replay is the
  instrument that is wrong and NO statement is named from it.
- **P7 (which blocker binds first).** In kt = 1..10 B7 (tracer advection)
  binds first, at kt = 1. B4j (j-seam wall) does NOT bind before kt = 8: the
  signal travels about 3.1 cells per step from a bump of radius about 10
  cells centred 39 cells from the j-seam. B6 (momentum advection) is
  inert while the velocity is below the bar-level product it multiplies.
  Falsifier: an unequal cell within 2 cells of the j-seam before kt = 8 in
  the given-entry whole-step arm.

## 2. Gates for this round

Geometry gate and ladder harness each ship a direct test with a plant that
fails when planted; the citation gate over the receipt stays PASS; focused
tests and one codex read-only review run synchronously.
