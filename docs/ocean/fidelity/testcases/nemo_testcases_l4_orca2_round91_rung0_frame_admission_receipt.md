# ORCA2 round 91 — rung-0 frame admission

**Verdict: HELD.** The repaired independent rung-0 record is admitted: 80/80
self-describing rank/step/stage frames pass, all 11 corruption controls fire,
and all 20 terminal restart shards are byte-identical to the admitted round-83
calibration. The true kt=1 step-entry state is the two-rank stage-0 `Nbb`
frame. No legoESM package, card, hierarchy switch, NEMO physics, threshold,
stabilizer, carried state, or sea-ice selector changed.

Base: `a62f67376`. Preregistration: `PREREG_nemo_testcases_l4_orca2_round91.md`
at commit `8893451b5`. Every number below is labelled **independent**.

## Admission

The committed frame reader admits exactly 80 records: ranks 0 and 1, steps
1 through 10, stages 0 through 3, five fp64 fields per frame, physical EOF,
and producer commit `a62f67376`. Its five plants independently corrupt the
magic, field name, payload length, a finite value, and provenance stamp; every
one refuses. The surface reader admits 25 PRESENT and 10 ABSENT fields under
the resolved `nn_ice=0`, `ln_rnf=false`, and `ln_icebergs=false` deck. Its six
plants all refuse, including ABSENT-as-zero and owner-on controls.

The operator run ended with `STOP 0`. Direct `cmp` over the complete explicit
restart list reports 20 compared and 0 different. The acquisition's own output
manifest also verifies in its record directory. Thus the write-only repair is
observational at every recorded terminal step, not merely at kt=10.

## True entry and stage census

The compiled record branch labels `Nbb` as exact step entry and writes the
rank-complete stage-0 frame immediately afterward, before calendar, surface,
vertical-physics, barotropic, or RK-stage work
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:92-108`). Both ranks carry
level sequence `[1,3,2,3]` at kt=1. At stage 0, u, v, and ssh are exactly
numerical zero while T and S are non-vacuous and finite. R91-P3 is
**CONFIRMED**: these two stage-0 shards are the only admitted rung-0 entry
operand.

The committed census finds the first state movement after the compiled stage-1
call and before its frame write
(`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3.f90:214-217`). Across both raw
rank payloads, stage 0→1 carries bit changes in T 637,687, S 449,743, u 469,570,
v 466,391, and ssh 17,167 values. The maxima are respectively 0.0558675295,
0.00822801187, 0.318832116, 0.313683571, and 0.441293728 in each field's native
units. The stage-1-equal plant moves the first detected transition to stage 2
and is refused; a planted nonzero stage-0 velocity is also refused. R91-P4 is
**CONFIRMED**.

This state movement is **not** called the first non-bit legoESM statement.
There is no explicit rung-0 legoESM card in the repository yet, so no model
stage has been compared with these frames. Naming any statement now would
confuse NEMO's physical evolution with cross-model error.

## Prediction ledger

| Prediction | Verdict |
|---|---|
| R91-P1: existing frame and surface gates admit without modification | **CONFIRMED** — 80 frames, 25 PRESENT/10 ABSENT, all 11 plants fire |
| R91-P2: write-only repair preserves the trajectory | **CONFIRMED** — 20/20 terminal restart shards byte-identical |
| R91-P3: stage 0 is the true two-rank step-entry state | **CONFIRMED** — compiled `Nbb` entry call plus rank-complete admitted frames |
| R91-P4: first state movement is stage 0→1 | **CONFIRMED** — all five fields move; both stage plants fire |

## Gates, tests, and review

The new census reuses the existing frame parser and its validated offsets; the
repository search found no prior rung-0 stage census. Focused tests and the
two real-record plants pass. Citation results, the required read-only Codex
review, and the prescribed fidelity battery are recorded in the final round
commit after they run.

No `packages/` file changed, so GYRE, DINO, tank cards, and the shipped ORCA2
card are unchanged by construction. The shipped card's sea-ice
`unmeasured_features` tuple is untouched. No configuration choice was made.

## OPEN

1. Build the explicit rung-0 legoESM card from the admitted hierarchy deck,
   with one stated line per resolved switch and refusals for every module that
   rung 0 excludes.
2. Initialize it only from the admitted two-rank stage-0 frames, replay stage 1
   in compiled order, and name the first cross-model non-bit statement. When
   the walk reaches stage vertical velocity/ZAD, measure the B26 two-solve plus
   NEMO-literal arm without pre-empting earlier statements.
3. Then score the independent and given-entry ten-step ladders and the existing
   independent 240-step month. Rung 1 remains untouched.

## UNVERIFIED

- No rung-0 legoESM card or cross-model stage replay exists yet.
- Therefore the first non-bit model statement, both ladders, and month score
  remain unmeasured.
