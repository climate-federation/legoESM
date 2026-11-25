# ORCA2 round 309 — OMT-0 under round 308's set; rung V3 under Decision 122

Date: 2026-10-17 (Claude round; codex review: paused (quota)).

Status: **LANDED-pending** (operator review + push gate).

## Headline

- Round 308 broke OMT-0: its shipped card refused at trace (the vector Krhs
  order statement on a flux-form card), so OMT-0 had not run since.
- OMT-0 now keeps only the two G6 statements whose compiled lines NEMO's
  linear momentum program runs (stored-ratio wzv, carried after-SSH slot);
  measured against its record they move 0 of 200 rows.  The other six are
  off on OMT-0 and restored on OMT-1, so OMT-1..4 resolve identically.
- V3's EVD trigger on the step-entry N2 lands on rung 10 (the only card that
  can run it): 0 rows move kt 1..10 under both labels; the flag is consumed.
- V3's double diffusion stays off: only rung 10's deck runs it, and the
  imported code refuses there because rung 10 also runs internal-wave mixing.
- Correction to round 308's receipt: its "kt-10 stage-3 sea-level maximum"
  column is the largest |card − NEMO| sea-level error, not the sea level.

## Provenance

- Base `168fc0ae6` (round 308 landed).  V3 import `c7b3a91a9`;
  preregistration + gate `24bc281e4`, addendum `0ff75085e` (each frozen
  before the runs it covers); selection `fc76c38e0`; tests `1c8615b64`.
- Import = package hunks of gridladder-vmix `e1b56d6004..0b9443cef2` (rung
  V3, receipt `nemo_testcases_gridladder_vmix_v3_round13`, addendum 105),
  applied with `git apply --3way`: one conflict, line wrapping only (kept
  this branch's wrapping, added the salt-diffusivity argument).  Not
  imported: the V-ladder card hunks of the recipe (this branch has no
  V-ladder cards).  Both selectors import default-off.
- Gate: `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round309_omt0_v3_d122_gate.py`
  (round-285 ladder, round-276 classifier, round-278 decision; OMT-0 on its
  admitted round-203 kt 1..10 frame record).  CPU, fp64, libm, production
  JIT.  Evidence `orca2_rounds/round309/` (`arms/`, `arms2/`, `final/`).

## Which cards run double diffusion and EVD (deck vs resolved card)

| card | deck ln_zdfddm | card DDM | deck EVD | card EVD trigger N2 |
|---|---|---|---|---|
| rung 0, OMT-0..4 | F | off | T (rn_evd 100, nn_evdm 0) | rebuilt on the solver's sea level (unchanged; the imported code refuses) |
| rung 10 | T | off (labelled unmeasured) | T (rn_evd 100, nn_evdm 0) | step-entry rn2/rn2b (this round) |

NEMO runs zdf_evd's `MIN(rn2, rn2b)` test (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfevd.f90:108-109`)
on the step-entry bn2 on Nbb (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:171`, `:174`)
on every deck; zdf_ddm's divide (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfddm.f90:161`)
runs only on rung 10's.

## OMT-0 (independent label, vs control = round 308's eight off)

What NEMO runs on OMT-0 (ln_dynadv_OFF): vor_een on planetary vorticity only
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:202`), not the np_CRV lines
(`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:762-766`); no KE gradient or
vertical momentum advection (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:160-166`);
the flux-form stage wzv (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:334`),
not the vector one (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:327`).  It
does run the step-entry wzv (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stp2d.f90:156`;
stored ratios `ORCA2_OMIP_L4/BLD/ppsrc/nemo/sshwzv.f90:297-298`) and the
carried slot (`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:237`).

| arm | statements | rows moved | toward/away/equal | exact lost | first over-bar | kt-10 stage-3 max |ssh err| (m) |
|---|---|---:|---|---:|---|---|
| round 308 as shipped | all eight | — | refuses at trace | — | — | — |
| NOADV | eight minus vector Krhs order, ZAD | 181 | 82/48/51 | 0 | kt-1 stage-1 u, equal | 4.345935895244679e-05 -> 4.345935895444519e-05 |
| **LINDYN (landed)** | stored-ratio wzv + carried slot | 0 | 0/0/0 | 0 | kt-1 stage-1 u, equal | unchanged |

NOADV's 181 moved rows come from statements whose NEMO lines OMT-0 never
runs (EEN order, top-F halo, the two vector stage-wzv forms), so they fail
D122 (b) there and are off.  NOADV census early kt 1-3 31/0/10, late
51/48/41; first away kt 5.

## V3 statements, one D122 row per statement on the seven pairs

Control = the shipped card with V3's selectors off.  Census = rows
toward/away/equal.

| statement | rung 0 | OMT-1 | OMT-2 | OMT-3 | OMT-4 | rung 10 indep | rung 10 given | lost | D122 |
|---|---|---|---|---|---|---|---|---:|---|
| EVD step-entry trigger | refuses | refuses | refuses | refuses | refuses | 0/0/0 | 0/0/0 | 0 | PASS on rung 10; off elsewhere |
| double diffusion (literal) | not run by NEMO | not run | not run | not run | not run | refuses | refuses | — | FAIL (b) elsewhere / refuses |
| V3 (both) | — | — | — | — | — | refuses | refuses | — | FAIL |

- EVD refusals: the step-entry trigger needs the step-entry bn2 bundle,
  which a constant-mixing card does not build.
- DDM refusal on rung 10: "IWMConfig.enabled with
  DoubleDiffusionConfig.form='nemo_literal' is not built: zdf_iwm's avs
  update is missing".  V3 has internal-wave mixing off; rung 10 runs it.
- Rung 10 first over-bar unchanged: independent kt-1 entry ssh, given entry
  kt-1 stage-1 T.  kt-10 stage-3 max |ssh err|: independent
  0.27696935149253726 m, given entry 0.26145027148482236 m (unchanged).
- Non-vacuity of the zero: a one-step rung-10 trace calls the legacy EVD
  trigger 2 times in control and 0 times with the flag on
  (`jaxpr/evdse_consumed.log`).  The step-entry and solver-level N2 fire on
  the same interfaces over kt 1..10 on rung 10.

## Shipped cards (selection tree `fc76c38e0`)

- Resolved configs vs `0ff75085e` (568 fields; rung 10 631): OMT-0 differs
  in the six statements only; rung 10 in the EVD trigger only; rung 0 and
  OMT-1..4 identical.  So their ladders are the round-308 ones.
- Shipped OMT-0 reproduces the LINDYN arm and the control row for row (200
  rows); shipped rung 10 reproduces the EVDSE arm and the control row for
  row under both labels (`final/instrument_check.txt`).

## GYRE, DINO, VORTEX (base `168fc0ae6` vs tip)

- Step jaxprs identical: DINO (123,236 lines), VORTEX zco flux/vector,
  VORTEX seamount flux/vector.
- GYRE ten-step residual archive identical (210 arrays, `np.array_equal`;
  sha256 `3b3d865cf5c82d38` both, as rounds 307-308); day-30: 30/30
  snapshots byte-identical (manifest excluded).

## Predictions

| ID | Verdict | Result |
|---|---|---|
| R309-P1 | **REFUTED** | the round-308 set refuses on OMT-0 at trace; there is no run to score. |
| R309-P2 | **UNTESTABLE** | the full set it was to equal refuses; NOADV itself runs and moves 181 rows vs control. |
| R309-P3 | **CONFIRMED** | EVDSE refuses on rung 0 and OMT-1..4. |
| R309-P4 | **CONFIRMED** | EVDSE runs on rung 10, both labels, 0 rows moved, passes (c)+(d). |
| R309-P5 | **CONFIRMED** | DDM and V3 refuse on rung 10 (IWM). |
| R309-P6 | **CONFIRMED** | GYRE ten-step and day-30, DINO and VORTEX identical. |
| R309-P7 | **CONFIRMED** | LINDYN vs control: 0 rows moved. |

## Gates and tests

- New `tests/ocean/fidelity/test_orca2_round309_selection.py` (18 tests):
  reverting the selection fails 3 (OMT-0 set, rung-10 trigger, OMT-0 trace
  refusal reproduced); tree restored, `git status` clean.
- Two tests already red at round 308's tip now state the current cards: the
  OMT-1 moved-field set (also the barotropic flux-form flags) and the
  transfer test's EEN-order assertion.
- Imported unit test `test_ddm_nemo_literal.py` passes; dispatch-hardening
  baseline gains the double-diffusion form guard (its one red, a land-model
  guard in a file this round does not touch, is pre-existing).
- Receipt citation gate on this receipt: PASS, 12 citations, failures 0,
  0 map entries failing (65 entries shifted by the import were re-anchored,
  one range extent 32 -> 33 lines).
- Default gate (land.sh set, JAX_ENABLE_X64=1, run in pieces under the
  10-minute cap): citation gate + private-import ratchet + recipe + ORCA2
  parallel receipt + round-309 selection 68 passed; TKE terms 69;
  freshwater 16; face-mask rank 2; DDM unit 4; barotropic state + ORCA2
  selection/transfer tests 51 — all passed.
- codex review: paused (quota).  The operator runs an independent Claude
  review before landing.

## Configuration choices

- ASKED (D122/D130 + addendum 105, taken): rung 10 reads EVD's step-entry
  trigger; OMT-0 keeps the two statements D122 (b) admits there.
- UNASKED: none.  Leaving DDM off on rung 10 and the step-entry trigger off
  on rung 0/OMT is the imported code's refusal, not a selection.  No runoff
  (D126), TKE switch or fold statement changed.

## OPEN

1. Rung 0 and OMT-0..4 still trigger EVD on N2 rebuilt at the solver's sea
   level; the step-entry form needs a step-entry bn2 bundle on
   constant-mixing cards (no rung builds one).
2. Double diffusion on rung 10 needs zdf_iwm's avs update (a rung with
   internal-wave mixing).
3. Fold statements still wait on a rung for the stage-1 V transport mask;
   D126 and D135 pending.
