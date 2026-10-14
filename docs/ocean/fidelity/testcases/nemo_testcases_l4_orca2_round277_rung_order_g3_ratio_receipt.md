# ORCA2 round 277 — ascending-sum rung order and the G3 stored ratio

**Status: STOPPED_FOR_DECISION.**  The ascending-level RHS depth sum is
net-toward on every existing card up to rung 0 and first turns away on the
shipped rung-10 card, under both labels.  Rung 0 and rung 10 differ by 19
NEMO physics switches across ten hierarchy modules, so per the round order
no intermediate card was built and the switch to isolate first is a
decision.  G3's stored stage ratio does not qualify under Decision 96 and
stays off.  No production selector changed.

Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round277.md`,
committed at `c0389d115` before any ladder ran; every measurement below is
from that clean tree.  Gate:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round277_rung_census_gate.py`
(round 276's arm selection, ladder and classifier; adds rung 10 under both
labels).  Counts are RMS directions `toward / away / score-equal`; early
kt=1..3, late kt>=4.  Rung 0 and OMT are the independent label; rung 10 is
reported under both labels, never mixed.  Evidence:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round277/`.

## (1) Ascending sum in ladder order

The only legoESM cards between OMT-4 and shipped rung 10 are OMT-4, rung 0
(= OMT-5) and rung 10; hierarchy rungs 1..9 have NEMO records but no card.

| card (label) | all | early | late | first away kt | D96 |
|---|---:|---:|---:|---:|---|
| OMT-4 (independent) | 103 / 80 / 12 | 26 / 19 / 10 | 77 / 61 / 2 | 1 | pass |
| rung 0 (independent) | 91 / 86 / 18 | 20 / 20 / 15 | 71 / 66 / 3 | 1 | pass |
| rung 10 (given NEMO's entry) | 78 / 92 / 10 | 12 / 19 / 9 | 66 / 73 / 1 | 2 | **fail** |
| rung 10 (independent) | 58 / 116 / 6 | 13 / 21 / 6 | 45 / 95 / 0 | 2 | **fail** |

No exact row is lost on any card and the first over-bar row is unchanged
(rung 10 independent: kt=1 entry SSH, the card's own entry).  R277-P1
**CONFIRMED** (round 276's three counts reproduce exactly); R277-P2
**CONFIRMED** (the independent label is more away, so the flip is not the
Decision-52 entry-SSH bridge); R277-P3 **CONFIRMED**: the first away card is
rung 10.

Resolved-deck diff, NEMO's `output.namelist.dyn` (the full 364-row table is
`rung0_to_rung10_resolved_deck_diff.md` in the evidence directory; 339 rows
are namelist blocks only read once their switch is on).  Every row whose key
exists in both decks:

| key | rung 0 | rung 10 |
|---|---|---|
| nambbc.ln_trabbc | F | T |
| nambbl.ln_trabbl | F | T |
| namrun.ln_rst_list | T | F |
| namrun.nn_stock | 1 | 10 |
| namrun.nn_stocklist | 1,2,3,4,5 | 10*0 |
| namsbc.ln_blk | F | T |
| namsbc.ln_flx | T | F |
| namsbc.ln_rnf | F | T |
| namsbc.ln_ssr | F | T |
| namsbc.ln_traqsr | F | T |
| namsbc.nn_fwb | 0 | 2 |
| namsbc.nn_ice | 0 | 2 |
| namsbc_rnf.ln_rnf_mouth | F | T |
| namtra_dmp.ln_tradmp | F | T |
| namtra_eiv.ln_ldfeiv | F | T |
| namtra_mle.ln_mle | F | T |
| namtsd.ln_tsd_dmp | F | T |
| namzdf.ln_zdfcst | T | F |
| namzdf.ln_zdfddm | F | T |
| namzdf.ln_zdfiwm | F | T |
| namzdf.ln_zdftke | F | T |
| namzdf.nn_havtb | 0 | 1 |

The three `namrun` rows are restart protocol; the other 19 are physics.
For reference, OMT-4 -> rung 0 is one physics switch
(`namtra_ldf.ln_traldf_off` T -> F, `ln_traldf_lap` F -> T) plus protocol.
The diff is more than one switch, so no kt=2 statement walk was attempted
and no card was built.

## (2) Addendum 82: G3 stored stage ratio

Imported from the grid-ladder branch (`07a184943`, `07e18b56a`, package
files only) as the default-off `zdf_momentum_stage_r3u_nemo_literal`.  NEMO
reads `r3u(Kmm)` inside the dyn_zdf thickness divisor
(`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/dynzdf.f90:192-198`), stored as the
mean of the two ratios at
`ORCA2G1_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:215-217`.  It has no host
without the G2 divisor statement, and the imported code refuses it alone
(plant fired: "requires zdf_momentum_nemo_face_thickness").  So it was
measured as the pair {divisors, stored ratio} against the control, and as
an increment against the divisors alone.

| card (label) | pair vs control | D96 | increment (vs divisors) | early | D96 |
|---|---:|---|---:|---:|---|
| rung 0 (independent) | 132 / 47 / 0 | pass | 51 / 15 / 40 | 2 / 0 / 5 | pass |
| OMT-1 (independent) | 61 / 78 / 0 | **fail** | 8 / 0 / 37 | 0 / 0 / 0 | pass |
| OMT-2 (independent) | 85 / 94 / 0 | **fail** | 6 / 3 / 76 | 0 / 0 / 0 | pass |
| OMT-3 (independent) | 81 / 98 / 0 | **fail** | 50 / 50 / 11 | 0 / 0 / 0 | **fail (tie)** |
| OMT-4 (independent) | 104 / 75 / 0 | pass | 36 / 45 / 42 | 0 / 0 / 1 | **fail** |
| rung 10 (given NEMO's entry) | 75 / 104 / 0 | **fail** | 85 / 41 / 6 | 1 / 1 / 2 | pass |
| rung 10 (independent) | 72 / 107 / 0 | **fail** | 59 / 39 / 13 | 1 / 0 / 1 | pass |

No exact row is lost and the first over-bar row is unchanged in any arm.
The pair's votes equal the divisors-alone votes on every card, so its red is
the host's (round 276's 61/78 on OMT-1).  The ratio's own increment is
majority-away on OMT-4 and tied on OMT-3, and both of those are late-only
(early rows all score-equal) — by addendum 80's reading, chaotic
amplification rather than an early transcription defect.  R277-P4
**CONFIRMED**: it does not qualify on every rung, so nothing lands.
R277-P5 **CONFIRMED** (no-host plant fired).

## Gates and tests

- GYRE ten-step trajectory: 70 rows, 0 ULP worsening, first over-bar kt=3
  unchanged; residual archive SHA256
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`,
  byte-equal to round 276's.  CPU 30-day member: all 30 daily snapshots
  byte-identical to round 276's (hashed before the rerun overwrote them);
  day-30 SHA256 `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- Plants: exact-row-loss plant fires on the classifier; the stored-ratio
  without-host plant fires on its specific refusal (any other error now
  reports PLANT-NOT-FIRED); an increment census given the wrong host arm
  refuses.  The increment table is produced by the gate's
  `--classify-increment` mode, which checks that pair and host differ by
  exactly the stored ratio.
- Focused tests: 104/104 pass (round-277 import, round-276 imports, G1
  transfer, dyn_zdf literal, G1 ORCA2-grid ladder).  The full
  `tests/ocean/fidelity` battery was not run: no production selector or
  default changed and the shared model path is GYRE-byte-identical.
- Citation gates: the receipt (2 citations) and the default receipt (274)
  PASS after a rigid re-anchor of the pins the import moved (one cited call
  range grew by the two imported arguments, 120 -> 122 lines); the shifted
  stored-ratio citation plant fires.  Citation tests: 192/193 pass; the
  one red is round 228's stale span, a known base red.
- Independent review (`codex exec --sandbox read-only`, one run): "HOLD —
  three confirmed gate defects; selector physics/default-off path and
  citation re-anchors are sound."  The three were: the measurement label
  had a default; the no-host plant would have counted any error as firing;
  the increment controls were relabelled outside the gate.  All three are
  fixed (label required; plant matches its refusal; increment mode checks
  provenance) and the increment table was re-derived through the gate with
  identical counts.  No ladder was re-run: every run had passed its label
  explicitly.

## OPEN

Decision needed: which of rung 0 -> rung 10's ten modules to isolate first
for the ascending sum's kt=2 away votes (requires building the two
hierarchy-rung cards on either side of that switch against their existing
NEMO records).  Decision 122 remains pending and governs whether the sum may
land on the full card meanwhile.  The G3 stored ratio stays off until its
divisor host passes OMT-1..3.
