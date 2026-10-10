# PREREGISTRATION — TSUNAMI lane, round 4 (build B6, B7; select the density depth)

Date 2026-10-08. Lane tip at start `c7115d9fd461`. Frozen before any
round-4 code or measurement exists. Evidence goes under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/tsunami_rounds/round4/`.

## 0. What was seen before this was written

Nothing new was run. Everything below rests on the round-3 receipt
(`nemo_testcases_l1_tsunami_round3_receipt.md`) and on reading NEMO's
compiled source and the reference run's `ocean.output`:

- `ln_dynadv_OFF` resolves `n_dynadv = np_LIN_dyn` (dynadv.f90:185);
  `dyn_adv`'s SELECT has no `np_LIN_dyn` case, so the stage calls
  (stprk3_stg.f90:316, :334) add nothing; stp_2D's advection SELECT has no
  case for it either and its vertical average takes the "averaged 3D RHS
  only" arm (stp2d.f90:176). `ocean.output` lines 723 and 738: "linear
  dynamics : no momentum advection used", "total vorticity = Coriolis".
- `ln_traadv_OFF` resolves `nadv = np_NO_adv` (traadv.f90:444); `tra_adv`'s
  SELECT has no case for it, so `ts(Krhs)` stays the zero it was set to
  (stprk3_stg.f90:467) and stages 1-2 are the thickness ratio alone
  (:503-505). Stage 3 still calls `tra_ldf` (OFF, `ocean.output` 609) and
  `tra_zdf` with the constant `avt` (`ln_zdfcst`). `ocean.output` 690: "NO
  T-S advection".

## 1. Card selections this round (decided or pure additions)

1. `eos_depth = "geometric"` — DECISION 101 (user).
2. B6: the shared flux-form momentum dispatcher gains `"none"` for the
   horizontal and the vertical scheme; the RK3 identity admits the complete
   program `flux_form / none / none`. Library defaults and every other card
   unchanged.
3. B7: the shared tracer-advection dispatcher gains `"none"`; the RK3 stage
   step with `"none"` uses NEMO's zero `ts(Krhs)` and its `(1 + r3t)` ratio
   update. Library default and every other card unchanged.

## 2. Frozen predictions and falsifiers

- **P1 (D101 confirm).** With the card selecting `geometric`, given-NEMO-
  entry kt = 1: stp_2D right-hand side max abs 1.09e-19 (u and v) and ssh
  7.63e-17, reproducing the round-3 measurement arm exactly. Falsifier:
  any different value.
- **P2 (B6 unit).** With `"none"` the dispatchers return the incoming
  right-hand side bit-unchanged; a plant that flips the selection back to
  `nemo_up3` makes the test fail.
- **P3 (B6 ladder, rhs).** With B6 selected, the given-entry stp_2D
  right-hand side at kt = 2..10 drops from 4.7e-10..8.2e-09 to the fixed
  1.09e-19 floor (the lambda = 0 control of round 3). Falsifier: any kt with
  max abs > 1e-17.
- **P4 (B6 ladder, whole step).** Given-entry whole-step ssh, u at
  kt = 2..8 fall below 1e-15 max abs. The first non-bit statement after B6
  is the dyn_spg_ts substep velocity update at a few ulp (round-3 section 2:
  <= 1.7e-15 normalised given NEMO's forcing). Falsifier: whole-step ssh
  > 1e-14 at any kt = 2..8. (kt = 9, 10 depend on the seam record, round-3
  section 9; not predicted.)
- **P5 (B7 unit).** With `"none"` a stage-1/2 tracer update equals
  `((1+r3t_b) T_b) / (1+r3t_a)` bit for bit; a plant selecting `fct2` fails.
- **P6 (B7 ladder).** Given-entry T, S after one step fall from 4.6e-03,
  6.9e-03 (round 3) to <= 1e-13 at every kt = 1..10. Stages 1 and 2
  bit-identical on T, S; stage 3 PLAUSIBLY not, because the card's stage-3
  content update and NEMO's `tra_zdf` solve may associate differently.
  Falsifier: T or S > 1e-12 at any kt.
- **P7 (independent, kt 1..10).** With all three selected, the independent
  ssh error at kt = 10 is < 1e-13 (round 3: 1.05e-03). Falsifier: >= 1e-13.
- **P8 (100-step record).** Independent ssh, uu_b, vv_b against the f_
  groups at kt = 11..100: the error stays near the floor until NEMO's front
  reaches the j-seam (round 3 extrapolated ~kt 15), then leaves it because
  the card walls the j-seam (B4j). Falsifier: the error leaves 1e-12 before
  kt 13, or never leaves it through kt 100.
- **P9 (B4j).** Within kt = 1..10 the first non-bit statement is not on the
  j-seam, so B4j is named and measured, not built, this round.

## 3. Not done this round

No NEMO run (no ACQUISITION_NEEDED expected). No change to any other card.
