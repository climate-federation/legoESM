# ORCA2 round 326 preregistration — SI3 mean ice thickness for TKE's under-ice mixing length (DECISION 154)

Frozen 2026-10-23, before any measurement (Claude round).

## Finding that sets this round

- ORCA2 runs TKE with nn_mxlice = 2 (record `ocean.output`: `nn_mxlice = 2`,
  `ln_mxl0 = T`, `nn_mxl = 3`). Its surface mixing length reads SI3's mean
  ice thickness: `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:621-622`.
- hm_i is assigned in one statement, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:186`,
  inside ice_var_agg(2), called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:260`
  only on ice steps (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:186`; nn_fsbc = 2,
  so kt 1, 3, 5, 7, 9). sbc (which runs ice_stp) precedes zdf_phy every step
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:151`, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`).
  The recorded build (R5FULLENTRY) has the same statements.
- No rung-10 record carries hm_i: the surface frames hold fr_i, taum and
  rCdU_ice but not hm_i; the only hm_i record (phase 2u TKE walk) is kt 2,
  rank 0, from a different run. So this round is ACQUISITION_NEEDED: it
  commits the acquisition and stops; the comparison harness, D122 and the
  gradient check wait for the record.

## Acquisition predictions (`nemo_testcase_l4_orca2_round326_hmi_acquisition/run.sh`)

- A1 NEMO reports STOP 0; all four ten-step restarts and all 40 surface and
  step-entry frames are byte-identical to round 5's record (write-only hook).
  Falsifier: any byte differs (admission refuses).
- A2 fr_i and taum in all 20 records equal round 5's surface frames bit for
  bit on owned points. Falsifier: any unequal point (reported, not refused;
  it would mean the harness feeds TKE a different fr_i than NEMO reads).
- A3 each even step's hm_i equals the previous odd step's bit for bit, both
  ranks. Falsifier: any unequal point (hm_i assigned somewhere not found).
- A4 hm_i >= 0 everywhere (refused otherwise), nonzero on some ice-covered wet
  points, max below 20 m.

## Next round (frozen later, with the record in hand)

The comparison-harness selection, D122 on seven pairs and the N = 1 gradient
check are preregistered when the record is admitted; no prediction is made
here about their numbers.

## Amendment (2026-10-23, operator review, before any NEMO run)

Codex's review of the acquisition found A2-A4 were reported, not enforced.
Predictions and thresholds are unchanged; admission now REFUSES on each
falsifier: fr_i/taum unequal (A2), even-step hm_i unequal (A3), hm_i zero
on every ice-covered wet point or >= 20 m (A4), each with a plant. The build
must also be round 5's program plus the observer (preprocessed sources and
objects byte-identical except zdftke's insertions and the writer).

## Amendment 2 (2026-10-23, record admitted, frozen before any arm is measured)

The operator's `--run` admitted the record (`PASS_R326_HMI_ADMISSION`); the
gate re-run here reproduces it and all nine plants fire. A1-A4: CONFIRMED
(fr_i/taum 0 unequal points, even-step identity on all 10 rows, hm_i max
2.797 m, 37,784 ice-covered wet points with hm_i > 0).

Comparison harness: the ladder's `comparison_card` selects nn_mxlice = 2
(model's existing V1 statement, zdftke.f90:620-623); `_surface_forcings` passes
the admitted hm_i for kt (written just before tke_avn at kt) and refuses unless
the record's fr_i equals the frame's bit for bit. Shipped card: nn_mxlice = 0.
Arms (both drag on): control = comparison card with nn_mxlice 0 (round 325's),
mxlice = comparison card. Rung 0 / OMT-1..4: both arms the shipped card.

- M1 D122 (c)+(d) PASS on all seven pairs, D139 recomputed; rung 0 / OMT-1..4
  rows identical between arms (0 differing rows). Falsifier: any exact row
  lost, or a first over-bar row moving away.
- M2 rung 10, both labels: kt-1 entry, stage-1 and stage-2 rows bit-identical
  between arms (TKE's coefficients are first used at stage 3); kt-1 stage-3 T
  differs. Falsifier: a kt-1 stage-1/2 row differs, or no kt-1 stage-3 row does.
- M3 rung 10 kt-10 full-domain RMS T error falls in both labels (control
  ~1.102e-3 K). Falsifier: it rises in either label.
- M4 N = 1 gradients on the comparison card (T loss w.r.t. initial T and
  w.r.t. rn_Ud): 0 non-finite; reverse = forward within 1e-12 relative.
- M5 refusal: nn_mxlice = 2 without hm_i refuses on step, step_checked,
  integrate and integrate_scan (with and without forcing); removing the
  refusal makes that test fail.
