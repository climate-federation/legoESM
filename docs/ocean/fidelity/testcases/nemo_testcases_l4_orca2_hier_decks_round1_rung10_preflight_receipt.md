# ORCA2-DECKS round 1 receipt — hierarchy rung 10 identity preflight

Date: 2026-10-01

Base: `5c47d7340`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round1.md`.

Disposition: **STOPPED_FOR_RECORD**.  The unchanged rung-10 deck, build and
admission pipeline are committed and preflight-clean.  The operator-owned
240-step NEMO record does not exist yet, so no operand or month identity claim
is promoted from UNMEASURED.

Claim label: **independent**.  The planned run starts NEMO from the deck's own
T/S initialisation, not from a recorded legoESM/NEMO bridge state.

## Rung inventory

| rung | NEMO deck | oracle record | side-lane status |
|---:|---|---|---|
| 10 | committed byte-exact copy of the admitted round-69 deck | acquisition preflight only | **STOPPED_FOR_RECORD** |
| 9 | not built; must differ from rung 10 only by removing SI3 | none | **UNMEASURED** |
| 8..1 | not built; top-down order is binding | none | **UNMEASURED** |
| 0 | owned by the main ORCA2 lane | none visible at this branch tip | **OUT OF SCOPE** |

The rung-10 source is
`orca2_rounds/round69/acquisition/orca1ice_surface_only_240step_np2`.
The committed ocean and ice namelists have SHA-256
`036f3cec148b2e89cede910d13189db4cc8e2e74a9b9ecdede7a87a5c14a98ec`
and `6b647863137b518b95ff97f83975d9afcb3944b7f6e8d63e45a05f494f5edc89`.
Each equals the source record byte for byte.

### Namelist delta from the upper neighbour

Rung 10 is the top/source deck, so it has no upper hierarchy neighbour.  Its
diff against the oracle record deck is empty: **zero differing lines, zero
differing assignments, zero differing bytes**.  Rung 9 was not guessed or
created in this round.

### Rung-10 switch census

| hierarchy module | source/rung-10 selection |
|---|---|
| T/S damping | `ln_tsd_dmp=.true.`, `ln_tradmp=.true.` |
| BBL + geothermal | `ln_trabbl=.true.`, `ln_trabbc=.true.` |
| GM + MLE | `ln_ldfeiv=.true.`, `ln_mle=.true.` |
| bulk + restoring + freshwater budget | `ln_blk=.true.`, `ln_ssr=.true.`, `ln_sssr_bnd=.true.`, `nn_fwb=2` |
| RGB shortwave | `ln_traqsr=.true.`, `ln_qsr_rgb=.true.`, `nn_chldta=1` |
| runoff | `ln_rnf=.true.`, `ln_rnf_mouth=.true.` |
| TKE | `ln_zdftke=.true.` |
| internal waves | `ln_zdfiwm=.true.` |
| differential mixing | `ln_zdfddm=.true.`, `ln_tsdiff=.true.` |
| sea ice | `nn_ice=2`, `jpl=1` |
| AGRIF-only special dynamics | deck carries `ln_spc_dyn=.true.`; compiled scope is inert, below |

The acquisition gate parses these values from the committed deck and refuses
if the assignment inventory or value changes.

## Frozen producer

No rebuild occurs.  The rung reuses binary
`450410b5c9b1960c4cc8d05d690decca3308f888e70987b444682315bb846572`.
Its CPP-key file is pinned to
`key_si3 key_qco key_vco_1d3d key_RK3`; it contains no `key_agrif`.
The exact key line is
`cpp_ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE.fcm:1`.

The deck still carries the child-grid-only special-dynamics assignment at
`rung10_namelist_cfg:234`.  The gate searched every file in this build's
compiled `ppsrc/nemo` and found zero occurrences of `ln_spc_dyn`.  Therefore
the assignment remains byte-identical as ordered, but it is inert in this
non-AGRIF binary and NEMO cannot print a resolved value for it.

## Oracle source

The compiled step calls the sea-boundary-condition manager and then the
WRITE-only operand recorder at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3.f90:129-130`.
The compiled recorder enforces fp64, rank-tagged self-describing field headers,
`STATUS='NEW'`, ten named operands, and one file per rank and step at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/l4_r69_surface.f90:65-88`.

The compiled surface manager accepts selection 2 only as SI3 and requires an
active forcing provider at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:247-252`;
the executing step dispatches selection 2 to `ice_stp` at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:499-503`.
The compiled from-rest branch reads the deck's T/S data, sets velocities to
zero, and copies the before level into the now level at
`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/istate.f90:105-140`.

## Acquisition and admission

The launcher stages the committed namelists and manifest, copies all other
deck/input artifacts through the source record's pinned SHA-256 manifests,
copies the admitted binary and compiled sources, and runs exactly 240 steps on
two MPI ranks.  It never invokes `makenemo` and never modifies NEMO source.

Admission requires:

1. exactly 480 self-describing operand frames, all finite, with steps 1..10
   raw-bit equal to the source record for all ten fields on both ranks;
2. all four step-240 ocean/ice restart shards payload-identical to the source
   record except permitted timestamp metadata;
3. fp64 finite `sshn`, `un`, `vn`, `tn`, and `sn` in both ocean restart shards;
4. all eight T/U/V/W month output shards present with finite floating payloads;
5. `STOP 0`, step/restart 240, from-rest, `nn_ice=2`, and `jpl=1` in the
   resolved logs; and
6. a complete SHA256SUMS inventory of regular record files, alongside the
   separately pinned input symlink manifest.

The committed checker parses each frame's magic, header integers, field names,
rank, dimensions and payload length through physical EOF.  It predicts no
whole-file byte count or positional header tuple.

Preflight status is `PREFLIGHT_PASS`; the deck-byte and manifest-field plants
both fire.  The remaining eight plants are run by the acquisition after the
real record exists: bad field name, truncation, missing frame, one-ULP operand,
one-ULP restart, non-finite terminal state, wrong resolved switch, and missing
SHA inventory row.

## Prediction ledger

| preregistered prediction | status | evidence |
|---|---|---|
| HD1-P1 deck identity | **CONFIRMED** | zero byte/assignment/line differences; both namelist hashes pinned |
| HD1-P2 producer identity | **CONFIRMED** | binary, keys, compiled step and writer hashes pinned; no rebuild path |
| HD1-P3 operand identity | **UNMEASURED** | real rung-10 record absent |
| HD1-P4 month identity | **UNMEASURED** | real rung-10 record absent |
| HD1-P5 resolved deck | **PARTLY REFUTED / otherwise UNMEASURED** | preregistration predicted `ln_spc_dyn` would print; the option is not compiled without `key_agrif`; other resolved rows await the run |
| HD1-P6 output inventory | **UNMEASURED** | real rung-10 record absent |
| HD1-P7 stopped-for-record disposition | **CONFIRMED** | preflight directory has no `record/` target |

The failed `ln_spc_dyn` prediction is retained rather than rewritten.  No
scientific switch changed: the source line stays present at every rung unless a
future user decision says otherwise.

## Validation and review

Focused hierarchy plus citation-gate battery: **25 passed** on the clean tree.
The required `tests/ocean/fidelity -n 12` battery ran once and reached 99% of
2,185 collected tests before the existing
`test_prediction_plant_is_fail_closed` stage-sweep control stopped producing
output; it was interrupted after a bounded wait rather than reported as a
pass.  Before the hang it exposed the two declared pre-existing reds
(`test_every_report_emitter_stamps_the_worktree` and
`test_full_v2_gate_and_plants`) plus
`test_live_trace_and_raw_history_arms_are_private_and_off_by_default`.  The
last red reproduces alone: its legacy expected final six trace fields omit the
new `stage_tracer_sources` field already present at the branch base.  This
round changes neither that test nor its model type.  All eight new hierarchy
tests present at that point passed inside the full battery; a ninth regression
test now pins pure-JSON preflight evidence.  Preflight, shell syntax, Python
compilation, the citation gate and both preflight plants pass.

The required separate `codex exec --sandbox read-only` review was attempted
twice and failed before reading the diff with `failed to initialize in-process
app-server client: Read-only file system`.  Verdict: **independent review
unavailable in-sandbox**.  The second attempt is retained as
`orca2_hierarchy/rung10/codex_review.log`.

No file under `packages/` or `src/` changed.  GYRE is therefore byte-identical
by construction; its year gate was not rerun.  Sea-ice physics was not changed:
rung 10 only records today's one-category deck, and rung 9 remains unbuilt.

## Scope and choices

ASKED: exact rung-10 deck, manifest, reusable-build acquisition, gate, tests,
receipt, and operator handoff.

UNASKED choices: none.  No module switch, input, run length, rank count, binary,
CPP key, threshold, scorer, or hierarchy ordering was selected beyond Decision
80 and the side-lane instructions.

## OPEN

1. Operator runs the committed rung-10 acquisition.  Until it passes, rung 10
   has no oracle record and the side lane remains stopped.
2. Next round admits rung 10, then reads the compiled ice-off branch before
   constructing rung 9.  It must cite every line made inert by `nn_ice=0`, prove
   whether `namelist_ice_cfg` is unopened, and print the complete one-module
   namelist diff.
3. Rungs 8 through 1 remain unbuilt.  Main-lane rung 0 remains out of scope.
