# ORCA2 round 326 — no record carries SI3's ice thickness for TKE's under-ice mixing length (DECISION 154): acquisition written

Date: 2026-10-23 (Claude round; codex review: operator).

Status: **STOPPED — ACQUISITION_NEEDED** (acquisition committed, preflight green).

## Verdict

- No rung-10 record carries hm_i, so the comparison cannot run nn_mxlice = 2
  yet. This round commits the acquisition and stops (prompt step 1).
- No model code, card or comparison harness changed; steps 2-5 (harness
  selection, refusal plant, D122, gradients) wait for the record.
- Rung 10's kt-10 numbers are therefore unchanged from round 325 (drag on,
  given entry, full domain: max sea level 0.00706750 m, max T 0.19674038 K,
  RMS T 1.10230e-3 K, max u/v 0.0472/0.0516 m/s; independent: 0.02256209 m,
  0.19673128 K, 1.10251e-3 K, 0.0472/0.0516 m/s). No "after" exists.

## What NEMO does (compiled ORCA2 build; the recorded R5FULLENTRY build has the same statements)

- Under ice, with ln_mxl0 = T and nn_mxlice = 2, the surface mixing length is
  ((1 - fr_i) zraug taum + 2 fr_i hm_i) tmask:
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:621-622`. Waves are off
  (ln_wave = F), so this branch runs.
- hm_i is assigned in one statement, `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icevar.f90:186`
  (ice_var_agg(2)), called at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:260`
  on ice steps only (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/icestp.f90:186`;
  nn_fsbc = 2: kt 1, 3, 5, 7, 9).
- sbc, which runs the ice step, precedes zdf_phy every step
  (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:151`,
  `ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:179`). So TKE at kt reads the
  hm_i of the latest odd step <= kt.
- The record's ocean.output states nn_mxlice = 2, ln_mxl0 = T, nn_mxl = 3,
  nn_fsbc = 2.

## What the records hold

- Rung-10 surface frames (round 5, 20 frames): fr_i, taum, rCdU_ice; no hm_i.
- Phase-2u TKE walk: hm_i at kt 2, rank 0 only, from a different run
  (`parallel/orca2/oracle_phase2v_tke_boundary_np2`).
- SI3 frames in the rung-10 run (Prather, zdf inputs, thd): odd steps,
  mid-ice-step or compressed ice columns; none is hm_i after ice_var_agg(2).
  Rebuilding hm_i from them would be a proxy; not done.

## Acquisition (`1abfce93d`, `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round326_hmi_acquisition/`)

- Re-runs round 5's ten-step record with one write-only hook in zdf_tke just
  before tke_avn: hm_i, fr_i, taum, surface mask, inner domain, kt 1..10,
  both ranks (`oracle_tke_hmi_kt*_r*.bin`). Patch is additions only.
- Admission: four restarts and 40 surface / step-entry frames byte-identical
  to round 5; five gate plants fire; the gate then reports fr_i/taum against
  round 5's surface frames and the even-step hm_i identity.
- Preflight (`--preflight-only`, run here, rc 0, `gates/preflight.log`): all
  seven checklist items, each with a plant that fired (continuation insert,
  non-reference config, nn_itend 9, nn_mxlice 0 in a real ocean.output copy,
  disk, existing target); gfortran syntax proof of the patched zdftke and the
  writer; whitespace-tolerant echo checks on the real ocean.output; no
  `grep -q` in pipelines.
- Gate test (writer's byte layout from its Fortran source, five plants,
  missing frame, cross-check on the real round-5 frames incl. a 1-point
  perturbation found): 9 passed.
- Predictions A1-A4 frozen in `PREREG_nemo_testcases_l4_orca2_round326.md`.

## Round-314 audit row

- nn_mxlice stays DIFFERS (b:D135); reason now names DECISION 154's pending
  record. Counts unchanged (180/3 vs record). Audit test 9 passed.

## Choices made

- Stop at ACQUISITION_NEEDED instead of building steps 2-5 ahead of data —
  ASKED (prompt step 1).
- Record hm_i at TKE's call site (just before tke_avn) rather than at
  icevar.f90:186 — UNASKED, instrument choice (it is the value TKE reads);
  offered for revert.
- Also record fr_i, taum and the surface mask as cross-checks — UNASKED,
  instrument choice; offered for revert.
- fr_i/taum mismatch and even-step identity are reported, not refused —
  UNASKED, instrument choice; offered for revert.
- Audit reason text names D154 — text only, no configuration.

## OPEN

- Operator runs `run.sh --run` (about one round-79b-sized NEMO run), then the
  next round does steps 2-5 on the admitted record.

## Gates (x64, CPU; logs `orca2_rounds/round326/gates/`)

- New gate test 9 passed; audit test 9 passed; preflight rc 0.
- Default gate, private-import ratchet and this receipt's citation gate:
  see the final commit's log lines below.

codex review: operator
