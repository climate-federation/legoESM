# ORCA2 round 297 — kt=7 stage-1 writer record request

Date: 2026-10-16

Status: **STOPPED_FOR_RECORD**. Round 296's complete private fold unit first
grows by more than 10x at the finite kt=7 stage-1 boundary under both
**independent OMT-4** and **given NEMO's entry**. The admitted OMT-4 record has
the kt=7 entry and completed stage but not the intervening source-order fields.
This round commits a fail-closed, rank-complete acquisition of that existing
23-field schema. It changes no production package, card, deck, selector,
carried state, sea-ice field, or NEMO physics.

## Provenance and record gap

- Lane base: `6ca3a8892`; frozen preregistration: `07abec778`; acquisition
  implementation: `ecf85c64d`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round297/`.
- Admitted source run:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round280/acquisition/orca2_omt4_stage1_source_10step_np2`.
- Requested target:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round297/acquisition/orca2_omt4_stage7_source_10step_np2`.
- Absolute launcher:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/work/autopilot-orca2-923613361/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round297_stage7_acquisition/run.sh`.

The admitted round-222/280 records contain 80 rank-complete entry/stage
frames for kt=1..10, but the 23-field `rhs/pre/cor/trp/adv/sbc/qco/final`
stream exists only at kt=1. No admitted record brackets the kt=7 stage-1
growth. R297-P1 is therefore **CONFIRMED** and the scientific split remains
UNMEASURED-with-spec until this acquisition admits.

## Compiled source order

The resolved OMT-4 vector path applies the stage-1 velocity update at
`ORCA2_OMIP_L4_R280STAGE1/BLD/ppsrc/nemo/stprk3_stg.f90:477-480`, diagnoses
the barotropic correction at `:521-528`, and adds it at `:547-549`. It then
builds the active-tracer transports at `:565-592`, zeros and fills the tracer
RHS through advection at `:617-662`, applies the surface source at `:664-672`,
and performs the QCO stage update at `:693-720`. The completed U/V/T/S state
is associated at `:812-820`.

The requested record copies round 280's already-admitted writer and call
sites. A committed patch changes only the selected step (`1` to `7`) and the
self-describing magic/file/error labels. The source NEMO stage program is
unchanged. Both labels use the same oracle stage stream; their candidate
replays remain separate in the next round.

## Frozen prediction ledger

| ID | Verdict | Mechanical result |
|---|---|---|
| R297-P1 | **CONFIRMED** | Existing rank-complete intermediates stop at kt=1; kt=7 has entry/completed-stage frames only. |
| R297-P2 | **UNMEASURED_WITH_SPEC** | Admission requires two rank records, 23 fields, exactly-once 148x180 coverage, 80 frame byte identities, and two restart byte identities. |
| R297-P3 | **UNMEASURED_WITH_SPEC** | Frozen first-writer prediction is `adv_t`/`adv_s`; an earlier or later boundary is retained as REFUTED. |
| R297-P4 | **UNMEASURED_WITH_SPEC** | Same-program-point candidate and oracle self-replays require the record. |
| R297-P5 | **PARTIAL** | Writer-layout plant fires now; eight payload/additions-only plants await the record; one-bit replay control belongs to the measurement round. |

## Acquisition and controls

The launcher pins the source stage program, writer, cpp keys, binary, deck
manifest, input manifest, and every committed producer artifact by SHA-256.
It parses the record's own magic, header integers, field names, ranks,
dimensions, and payload sizes. Admission refuses unless both rank slabs cover
148x180 exactly once and all inherited frames plus terminal restarts are
byte-identical.

The clean-tree preflight applies the one-line scientific writer change,
compiles the patched Fortran writer, and prints
`ORCA2_ROUND297_STAGE7_PREFLIGHT_READY`. SHA-256:
`551a8f2740e6909142bc78259721cf2b9e983a45306b79119beb7e2585b1d4cb`.
The writer-layout plant removes the kt=7 selector and exits 69 with
`STATUS PLANT-FIRED layout`. SHA-256:
`378490743b135b49af841552cc0e9fdce317260c2f80f81e7316bb708145d462`.
No `mpirun` was attempted in the sandbox.

## Validation and review

- The generic inherited parser now takes explicit expected step and magic;
  defaults remain the round-175 kt=1 schema. A behavioral test proves kt=7 is
  accepted only when both explicit values are supplied.
- The focused round-295/296/297 plus citation-gate battery passes **25/25**
  in 8.60 s (SHA-256
  `5b385babe7d049495b7b223824c9d220201ee28320e9a63c24cc0f666419d558`).
  Record-dependent plants are deliberately not claimed before the operator
  produces the record.
- The required `tests/ocean/fidelity -n 12` battery reached 99% with **3,589
  passes**, seven deterministic branch-tip reds, and one xdist worker abort
  while compiling G1 case `off1-1` (SHA-256
  `ec4e98085759405819c2a10270ec8658751e891ed43390f4081afb934ac47701`).
  The aborted `off1-1` case passes alone in 142.03 s; the not-run `off2-1`
  case passes alone in 143.77 s (log SHA-256
  `943b7abd86fd719309fb036f3320f193442560a89d89644e82e931d527c3dab1`).
  This is the campaign's documented parallel JAX resource-failure class, not
  a round-297 failure.
- The seven named reds reproduce serially (log SHA-256
  `a20f677d83fe094d8d22fa6faac5d8372ecd2777e971bdb8ea83d8aacbfd4a3d`):
  the stale GYRE round-33 mutation anchor; the GYRE round-129 year-harness
  stamp; OMT-1's added required barotropic card field; round-228's already
  missing citation-map key; the 14 allow-dirty scope ratchets; the 13
  worktree-stamp ratchets; and the SI3 `MY_SRC` provenance mismatch. None is
  in the round-297 change set. In particular, the round-228 citation key is
  absent at base `6ca3a8892`, while this round only adds new map entries.
- Independent review: operator. The required nested `codex exec --sandbox
  read-only` attempt could not initialize its in-process app-server client in
  the read-only sandbox (`Read-only file system`; log SHA-256
  `eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`).
  Independent review is therefore unavailable in-sandbox and remains an
  operator handoff.

## Configuration choices

- **ASKED:** none.
- **UNASKED:** none.
- Decision 125's top-V exchange, the resolved stage mask, raw `hv_0`, raw
  `ssvmask`, and the complete fold unit remain off in production.
- Decision 126 remains pending; shipped runoff is unchanged and is not
  re-raised. Sea ice and `unmeasured_features` remain unchanged.

## OPEN

1. The operator runs the absolute launcher with `--run`. It must finish with
   `STOP 0`, fire all eight record plants, and print
   `ORCA2_ROUND297_STAGE7_ACQUISITION_PASS`.
2. The next round admits the two rank records, replays both labels at matched
   program points, and names the first `rhs/pre/cor/trp/adv/sbc/qco/final`
   boundary that creates the kt=7 stage-1 growth.
3. The complete fold unit remains private and atomic. No constituent lands
   until OMT-4 completes under both labels and the standing gates pass.

## Citation correction

The first committed receipt cited the active-tracer transport as lines
568 through 592.
The branch begins at line 565; `:565-592` is the exact compiled span. The
source order and acquisition schema are unchanged.
