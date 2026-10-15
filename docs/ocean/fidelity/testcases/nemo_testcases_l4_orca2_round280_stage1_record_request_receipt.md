# ORCA2 round 280 — stage-1 V source split record request

Date: 2026-10-15

Status: **STOPPED_FOR_RECORD**. Round 279's 1,440-cell independent OMT-4
stage-1 V source-row debt cannot be split with the admitted streams. This
round preregistered the split and prepared a fail-closed, rank-complete
acquisition. It changes no package, card, deck, physics, sea-ice selector, or
runoff setting.

Every scientific number below is **independent OMT-4**. No value given NEMO's
recorded entry is mixed into this receipt.

## Provenance

- Branch base: `7b39c70075973f51b9e4544b2f8b3a583a1a2ca0`.
- Frozen preregistration: `PREREG_nemo_testcases_l4_orca2_round280.md`, first
  committed as `721530ffd`; its citation corrections are loud and do not
  alter a prediction or threshold.
- Admitted baseline:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round222/acquisition/orca2_omt4_frames_10step_a_np2`.
- Operator launcher:
  `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round280_stage1_acquisition/run.sh`.
- Requested target: `ORCA2_OMIP_L4_R280STAGE1`, writing to
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round280/acquisition/orca2_omt4_stage1_source_10step_np2`.

## Prediction ledger

| Prediction | Verdict | Evidence |
|---|---|---|
| R280-P1: admitted OMT-4 lacks rank-complete `pre_v` / `cor_v` | **CONFIRMED** | 80 stage frames and the Kmm transport stream exist; no stream brackets the Kaa correction on both ranks |
| R280-P2: raw vector update is exact on all 5,400 row slots | **UNMEASURED_WITH_SPEC** | requires the requested rank-complete `pre_v` |
| R280-P3: correction first creates 1,440/5,400 unequal, max 0.0162073695895 m/s | **UNMEASURED_WITH_SPEC** | requires paired `pre_v` / `cor_v` |
| R280-P4: instrument is additions-only | **UNMEASURED_WITH_SPEC** | admission must reproduce 80 frames and both kt=10 restarts byte-for-byte |

## Compiled source and record gap

The resolved ORCA2 deck executes the vector update at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:467-470`. The subsequent
barotropic diagnosis and correction are at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:513-537`.

The admitted transport writer instead records the old `Kmm` velocity and its
transport operands at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3_stg.f90:286-310`. It does not record the
stage-1 `Kaa` velocity immediately before and after the correction. The
completed stage frames are downstream of both statements. Consequently no
admitted rank-complete stream can decide whether the raw update or the
correction owns round 279's 1,440-cell source-row residual. That is the first
unmeasured boundary, and this round stops there.

## Acquisition and controls

The launcher reuses the already-proven round-175 self-describing writer and
schema without changing their field layout. It records `rhs_v`, `pre_v`, and
`cor_v` once per rank at kt=1 stage 1. The checker parses the magic, header
integers, and every per-field name/rank/dimension tuple before calculating
payload lengths; it then requires exactly-once 148x180 coverage.

The producer is pinned by content hashes of the deck, input manifest, binary,
source, patch, writer, checker, preregistration, and launcher rather than a
moving commit identifier. Admission refuses unless the instrumented run
reproduces all 80 inherited per-stage frames and both terminal kt=10 restart
files byte-for-byte. It retains plants for header, field name, dimensions,
truncation, swapped rank, global coverage, inherited-frame bytes, and restart
bytes. The later source-split measurement must additionally fire raw-bit and
corrected-bit plants before either P2 or P3 may receive a verdict.

The clean-tree final preflight proves that the additions-only patch applies
to the pinned OMT-4 source and that both Fortran units compile. It prints
`ORCA2_ROUND280_STAGE1_PREFLIGHT_READY` for the new target. The corresponding
layout plant exits 69 with `STATUS PLANT-FIRED layout`; both logs live under
the round-280 evidence directory as `final_preflight.log` and
`final_layout_plant.log`.

No NEMO run was attempted in the sandbox. The record-dependent admission
plants remain unexecuted until the operator runs the launcher; they are not
represented as passing controls.

## Gates, tests, and review

The round citation gate passes all 3 mapped citations with no failure or map
audit debt. The cumulative default receipt passes 274/274 citations. Shifting
the correction citation by two lines makes this round's gate fail and exit 1,
so the control is non-vacuous.

Focused citation, round-175 boundary, and OMT-4 acquisition tests pass 31/31
in 5.12 s. The one permitted `tests/ocean/fidelity -n 12` battery reached 99%
with 3,537 passes and 7 skips before xdist lost workers and stopped producing
a terminal summary. It logged 22 failed node IDs. Each exact ID was rerun in
a fresh serial process: all 14 G1/JAX worker-loss nodes pass; eight
pre-existing assertions reproduce:

- the registered GYRE round-129 certified-year harness movement;
- two stale GYRE round-33 mutation anchors;
- the OMT-1 module-edge test's extra `barotropic` group;
- the stale round-228 citation-map expectation;
- the 14-driver dirty-escape scope ratchet;
- the 13-emitter worktree-stamp ratchet;
- the SI3 scalar-math provenance refusal (`A MY_SRC is not verbatim`).

The round changes none of the files named by those eight assertions. No model
file changed, so no ORCA2, GYRE, DINO, or tank trajectory is eligible to move
or land.

Independent review is unavailable in-sandbox. The separate
`codex exec --sandbox read-only` attempt exited before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
No review verdict is claimed; the operator remains the independent reviewer.

## Configuration choices

- **ASKED:** Decision 126 remains pending with the user. The shipped runoff
  configuration is untouched and is not re-raised by this round.
- **UNASKED:** none.
- Sea ice remains exactly as the ORCA2 card carries it, including its
  `unmeasured_features` tuple.

## OPEN

1. The operator runs the absolute launcher reported by this round with
   `--run`. It must build the new target, finish with `STOP 0`, fire all eight
   record plants, and emit `PASS_R280_OMT4_STAGE1_ADMISSION`.
2. The next round admits the paired record and scores `pre_v` then `cor_v` at
   matched pre-`lbc_lnk` program points. Both measurement plants must fire. A
   failed prediction is retained as `REFUTED`.
3. If the correction owns the first non-bit row, walk its recorded operands
   in compiled order. No statement lands without the standing ORCA2/OMT,
   GYRE, DINO, tank, moved-row, citation, test, and operator-review gates.
