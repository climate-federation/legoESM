# ORCA2 round 204 — OMT-0 record admission, card, and ten-step ladders

Date: 2026-10-09. Base: `ba3932304`. Measurement implementation:
`59aa505c5`. Status: **LANDED**.

Every trajectory number below is labelled explicitly. **Independent** starts
from the OMT-0 card's own climatological T/S, zero velocity, and zero SSH.
**Given NEMO's entry** installs the admitted entry frame. The two labels are
never combined in one score.

## Verdict

The existing round-203 NEMO outputs are sound. The earlier refusal was a gate
defect: after checking the instrument binary, the inherited provenance helper
silently rechecked it against the uninstrumented binary hash. The repaired
admission proves 80 self-describing frames per twin, 400 array-equal twin
field comparisons, and four byte-identical terminal restarts. Its ten distinct
plants all refuse at their named predicate.

The Decision-103 OMT-0 card is now executable with exactly five modules OFF:
momentum advection, tracer advection, momentum lateral diffusion, tracer
lateral diffusion, and bottom drag. NEMO's five resolved selections and their
mutual-exclusion checks are mechanically bound to the compiled source:
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynadv.f90:162-190`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv.f90:586-633`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldfdyn.f90:177-228`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/ldftra.f90:214-268`, and
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdfdrg.f90:371-401`.

Both kt=1..10 ladders complete 40 checkpoints / 200 field rows and stay finite.
The independent entry is exact in every active T/S/u/v/SSH cell. Across stored
T, 151,917 inactive cells differ only by the sign of zero; the gate canonicalizes
only that already-certified input representation and uses strict bit comparison
for every executed row. The given-entry bridge is storage-bit exact.

The first non-bit boundary is kt=1 stage 1. In source order the first statement
reached is NEMO's hybrid association of the completed external solution to the
stage-1 SSH, inside the compiled block
`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:137-179`.
The SSH row differs on all 16,433 active columns, RMS
`0.006243153742634906 m`, maximum `0.13136125371061705 m`. The assignment is
the first measured statement, not an internal owner: its `ssha` operand already
comes from the preceding split-explicit solve. The admitted inherited
`oracle_bt_*` streams make that internal walk the OPEN item.

OMT-0 has no month by construction. NEMO's registered boundary remains kt=11
`stp_ctl`: maximum |v| is `10.24 m/s` at `[22,84,27]`, with |u| `3.041 m/s`
and |SSH| `3.853 m` at kt=10. The ten-step legoESM ladders finish before that
boundary; no post-kt10 candidate claim is made.

## Card and namelist census

The NEMO deck changes exactly five live TRUE selectors to FALSE and adds their
five matching OFF selectors:

| module | disabled selector | enabled selector | card selection |
|---|---|---|---|
| bottom drag | `namdrg.ln_lin` | `namdrg.ln_drg_off` | zero legacy drag rate and no matrix/substep drag |
| momentum advection | `namdyn_adv.ln_dynadv_vec` | `namdyn_adv.ln_dynadv_off` | `flux_form/none/none` |
| momentum LDF | `namdyn_ldf.ln_dynldf_lap` | `namdyn_ldf.ln_dynldf_off` | all momentum-viscosity coefficients zero |
| tracer advection | `namtra_adv.ln_traadv_fct` | `namtra_adv.ln_traadv_off` | `tracer_advection="none"` |
| tracer LDF | `namtra_ldf.ln_traldf_lap` | `namtra_ldf.ln_traldf_off` | `K_h=K_bih=0`, no GM/Redi |

The card retains EEN, split-explicit free surface, SCO pressure gradient,
constant vertical mixing, EVD, no ice, and the corrected independent initial
state. There is no new stabiliser or option not present in the deck. The shipped
ORCA2 rung-10 card and its sea-ice `unmeasured_features` tuple are untouched.

`ln_dynadv_OFF` is NEMO's linear `np_LIN_dyn` program, so EEN uses planetary
Coriolis (`np_COR`) rather than the nonlinear flux-form metric term. It still
executes WZV for tracer transports; the legoESM validator therefore preserves
the card's literal WZV operands while allowing the vertical-momentum OFF arm.

## Ladder numbers

The numerical scores below are identical between the independent and
given-entry runs. The only count difference is the classified inactive-T zero
sign: at kt1 stage 1 T, independent has 430,552 unequal active/stored numerical
values while given-entry has 582,469 bit-unequal values including dry signed
zeros.

| label | checkpoint | T RMS / max K | S RMS / max PSU | u RMS / max m/s | v RMS / max m/s | SSH RMS / max m |
|---|---|---:|---:|---:|---:|---:|
| independent | kt1 stage1 | `1.4251786998869236e-05 / 0.0012975435364612764` | `3.28957045667694e-04 / 0.02591648767030108` | `8.562305990585591e-04 / 0.061598565285546344` | `8.94650391772652e-04 / 0.03412538610804825` | `0.006243153742634906 / 0.13136125371061705` |
| given NEMO entry | kt1 stage1 | same | same | same | same | same |
| independent | kt10 stage3 | `3.7889450709893235e-04 / 0.07840362994280436` | `0.0037085398530979827 / 0.5204576806596393` | `0.004914652518929538 / 0.6755649454182693` | `0.0048237653262439 / 0.6292697150937124` | `0.0288100436468264 / 0.5086605459790218` |
| given NEMO entry | kt10 stage3 | same | same | same | same | same |

Artifact: `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round204/omt0_ladders.json`.

## Frozen predictions

| prediction | disposition |
|---|---|
| binary hard-pin caused the round-203 refusal | **CONFIRMED** |
| 80 frames/twin, 400 equal fields, four terminal byte identities | **CONFIRMED** |
| all named record and card plants fire distinctly | **CONFIRMED** |
| card/deck delta is exactly the five OFF modules | **CONFIRMED** |
| both labelled kt=1..10 ladders complete | **CONFIRMED** |
| first debt is external SSH or earlier | **CONFIRMED**: external SSH is the first source-reached boundary |
| candidate remains finite through kt10 and no earlier boundary is claimed | **CONFIRMED** |
| existing cards remain unchanged | **CONFIRMED** by the shared gates below |
| addendum P1: the old month output has a success marker | **REFUTED and retained**: exact MPI abort-123 evidence exists instead |
| exact legacy-stop evidence can be admitted narrowly, with a firing plant | **CONFIRMED** |

## Shared blast radius and controls

- GYRE: all 70 certified ten-step rows are array-identical to round 185;
  maximum worsening is zero ULP and the first-over-bar remains kt=3. A fresh
  CPU day-30 snapshot is byte-identical to the certified baseline, SHA-256
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- LOCK_EXCHANGE: all 50 rows are unchanged; first-over-bar remains kt=8 U.
- OVERFLOW: all 50 rows are unchanged; first-over-bar remains kt=2 T/U.
- DINO: the CPU day-30 wet-3D T RMS is `2.056821682e-03 K`, below the fixed
  `2.244317642e-03 K` bar. Its planted `6.981690958e-03 K` score refuses.
- The card-module and independent-entry plants both refuse. The citation gate's
  rigid shift plant refuses, and its real default receipt has no unmapped
  citations.

## Validation and review

- Round-203 admission: PASS; 80 frames/twin, 400 field comparisons, four
  terminal restart comparisons, ten distinct firing plants.
- Round-204 OMT-0 gate: PASS; two 200-row ladders and two firing plants.
- GYRE: 70-row oracle-relative comparison PASS with zero moved cells; the
  fresh day-30 snapshot is byte-identical to the certified snapshot.
- LOCK_EXCHANGE and OVERFLOW: 50-row comparisons PASS with zero moved cells.
- DINO CPU month and its self-test: PASS at the number above.
- Citation gate: default cumulative receipt and this receipt PASS with no
  unmapped citations; both rigid-shift plants refuse.
- Focused pytest: 41 passed.
- The single `tests/ocean/fidelity -n 12` battery collected 2,969 tests. It was
  interrupted after prolonged tail inactivity at the displayed 97% line; the
  log contains five failure markers and seven skips, matching the registered
  worktree/provenance/root-ratchet class. It emitted no final summary before
  interruption. No second full battery was started.

Independent review unavailable in-sandbox. The required command was invoked as
`codex exec --sandbox read-only`; it exited 1 before reading the diff with
`failed to initialize in-process app-server client: Read-only file system`.
That verbatim failure is retained in `round204/independent_review.log`; it is
not represented as a PASS verdict.

## OPEN

1. Walk OMT-0's kt=1 split-explicit solve using the admitted `oracle_bt_*`
   streams to the first internal non-bit statement; the hybrid stage-1 SSH
   association is only the boundary.
2. Build and acquire OMT-1 (+ linear implicit bottom drag) only after OMT-0's
   first statement or cancelling unit is named. OMT-1 is the first rung with a
   month protocol.

No configuration choice is pending. No acquisition is needed for the OPEN
OMT-0 walk because the kt=1 barotropic streams already exist in the admitted
round-203 run; their separate calibration/admission is the next gate.
