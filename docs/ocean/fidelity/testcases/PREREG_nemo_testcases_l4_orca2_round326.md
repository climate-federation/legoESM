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
