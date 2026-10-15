# ORCA2 round 282 — G1b np_CRV area landing and vector-fold refusal

Date: 2026-10-15

Status: **LANDED**.  NEMO's np_CRV relative-vorticity statement now uses the
ORCA2 F-cell area `e1f*e2f` on the production ORCA2 card.  It passes Decision
96 on rung 0, OMT-1..4, and shipped rung 10 under both available labels: 942
RMS-moving rows toward NEMO, 301 away, zero exact-row losses, and every first
over-bar row unchanged.  The companion vector-form barotropic fold unit is
**HELD**: rung 0, OMT-4, and shipped rung 10 reach the live-W-thickness refusal
at kt=8, so that selector remains false.

Sea ice, runoff, decks, records, and NEMO sources did not change.  Decision
126 remains pending with the user and the shipped runoff value is untouched.

## Provenance and imported statements

- Lane base: `7bbe6b910032187f3acd024fb1128ddc27d40831`.
- Frozen preregistration: `PREREG_nemo_testcases_l4_orca2_round282.md`, commit
  `9cc8d2ba3`.
- Default-off implementation and measurement gate: `107b56c06`; production
  ORCA2 selection and fail-closed census repair: `68455816a`.
- Evidence root:
  `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round282`.
- Imported certification context:
  `nemo_testcases_gridladder_g1b_round2_receipt.md`.  G1b itself was not yet
  certified, so this round applies Decision 96, not Decision 122.
- `npcrv_f_area_decision.json` SHA256:
  `498c98bced51b2cf6ccebc9e27875d9f5cb18aeebc6ba51d3a511167186db470`.

The imported package commits had four conflicts.  In the recipe conflicts the
current lane's G1-only flat-card builder and validator were retained; the
grid-lane-only G1b builder was not imported because that branch infrastructure
does not exist here.  In `state.py`, current lane fields were retained and
only the two new default-false selectors were added; unrelated later grid-lane
wind, stress, and ZDF fields were not imported.  The shared G1 replay support
and its tests were retained.  The generated citation map is the union of both
lines of work.

## Compiled statements and executed branches

The shipped deck resolves vector-form momentum, EEN vorticity, and the
split-explicit solver.  In its compiled source, np_CRV forms the relative
circulation and multiplies by `r1_e1e2f` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynvor.f90:648-651`; that reciprocal is built
from `e1e2f=e1f*e2f` at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/domhgr.f90:155-156`.  This is statement A.

Statement B consists of NEMO's midpoint face-depth construction at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:509-519`, its `zhU/zhV`
transports at `ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:530-535`, and the
post-update exchange at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:721-730`.  These statements sit
outside the vector/flux update choice; the executing vector arm is visible at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/dynspg_ts.f90:666-678`.  The selector therefore
correctly reaches ORCA2's vector-form program, but its present atomic unit is
not landable.

Before measurement both selectors defaulted false.  The final production card
sets only `een_nemo_relative_vorticity=True`; every GYRE, DINO, tank, generic,
and non-ORCA card retains the false path.  The vector-fold selector remains
false everywhere on ORCA2.

The imported validation replay originally referenced the grid branch's
module-global G1/G1b dispatcher, which this lane intentionally did not import.
The full battery exposed that incomplete conflict resolution as a `NameError`.
The replay now resolves planetary versus vector form from the constructed
card, and four field-order assertions were made append-safe without weakening
their selector/default checks.  All ten affected G1 exact IDs pass serially.

## Frozen predictions and controls

| Prediction | Verdict | Measurement |
|---|---|---|
| P1: selector/branch inventory is exact | **CONFIRMED** | ORCA2/OMT execute np_CRV and vector-form dynspg; non-ORCA cards keep both selectors false |
| P2: statement A qualifies under D96 | **CONFIRMED** | all seven labelled ladders have a toward majority, no exact loss, and unchanged first over-bar row |
| P3: statement B may qualify; no direction assumed | **REFUTED** | four required ladders refuse at kt=8 on live W thickness |
| P4: the pair cannot hide a red single | **CONFIRMED** | A+B reproduces B's same kt=8 refusals |
| P5: GYRE/DINO/tanks remain unchanged | **CONFIRMED** | selectors remain false; GYRE ladder/day 30 and focused card gates are green |

Both non-vacuity controls fire.  The extra-selector plant refuses an
unregistered option.  The exact-loss plant reports
`previously exact rows left the bar: [(1, 'entry', 'T')]`.  The latter required
changing a genuinely exact baseline row; the earlier attempt that perturbed an
already-debt row was retained in the evidence log as a failed plant design.

## Statement-A Decision-96 census

Numbers in the table are never mixed across labels.

| Card | Label | toward | away | score-equal | exact lost | first over-bar | kt10/stopping stage-3 SSH max, before -> after (m) |
|---|---|---:|---:|---:|---:|---|---|
| rung 0 | independent | 139 | 46 | 0 | 0 | equal | 0.3835396648865576 -> 0.3830826119572967 |
| OMT-1 | independent | 111 | 30 | 0 | 0 | equal | 0.746717960306863 -> 0.7469693088949136 (kt=8 ladder end) |
| OMT-2 | independent | 142 | 39 | 0 | 0 | equal | 0.3464513662228243 -> 0.34650568911180074 |
| OMT-3 | independent | 128 | 53 | 0 | 0 | equal | 0.3635503799346643 -> 0.36359234522692674 |
| OMT-4 | independent | 122 | 63 | 0 | 0 | equal | 0.39353168928479665 -> 0.3935712593590627 |
| rung 10 | given NEMO's entry | 148 | 37 | 0 | 0 | equal | 0.28674475751010314 -> 0.28669958102920906 |
| rung 10 | independent | 152 | 33 | 0 | 0 | equal | 0.2938484161231321 -> 0.29399731938907275 |

Aggregate: **942 toward / 301 away / 0 score-equal**, with no exact-row loss.
The final-SSH maximum is registered, not used as an extra veto.  Every
labelled census independently passes Decision 96.

## Statement-B and pair refusal

The vector-fold single and A+B pair each complete OMT-1, OMT-2, and OMT-3;
their relaxed RMS censuses are respectively 155/0, 195/0, and 195/0
toward/away.  They nevertheless fail the required full matrix:

| Card | Label | B | A+B |
|---|---|---|---|
| rung 0 | independent | kt=8 live-W-thickness refusal | same refusal |
| OMT-4 | independent | kt=8 live-W-thickness refusal | same refusal |
| rung 10 | given NEMO's entry | live-W-thickness refusal | same refusal |
| rung 10 | independent | live-W-thickness refusal | same refusal |

The refusal is `raw-mesh e3w_int must contain only finite values > 0`.  This
is the already registered fold-unit failure class, not a new stabiliser
request.  B and A+B remain default-off.

## Shared gates and tests

- Current-tip GYRE ten-step residual archive SHA256 is
  `3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`,
  identical to the certified baseline.  All 30 daily snapshots are byte
  identical; day 30 SHA256 is
  `3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
- DINO/tank/generic-card focused gates pass 61/61 in 275.36 s.  This includes
  both LOCK_EXCHANGE and OVERFLOW planted controls and the DINO from-rest
  admission/plant.  The new selector is false on all of those cards.
- The citation/recipe/freshwater push battery passes 128/128 in 377.11 s.
- Round-282 focused tests pass 4/4 on the final production selection.
- The one permitted `tests/ocean/fidelity -n 12` battery selected 3,592
  non-slow tests.  Before its long tail exceeded the outer timeout it logged
  3,554 passes, seven skips, and 18 failures.  Ten failures were the imported
  G1 conflict-resolution defect just described; all ten exact IDs pass after
  the repair (4 + 3 + 3 serial groups).  The hanging stage-sweep plant passes
  alone in 1,151.41 s.  The other eight reds are the branch's registered
  pre-existing SI3 provenance, GYRE round-129/year, two GYRE round-33 anchors,
  OMT-1 module-edge, round-228 citation expectation, dirty-escape scope, and
  worktree-stamp ratchets; their exact IDs were rerun serially and remain
  registered.  The full parallel battery was not rerun.

The known unrelated `tests/test_dispatch_hardening.py` slab-land guard is
outside this battery and outside the changed paths; it was not modified.

Independent review: **operator**.  Per the current lane instruction, no
nested review is attempted in this sandbox.

## Configuration choices

- **ASKED:** none.  Decision 126 remains pending and is not re-raised.
- **UNASKED:** none.
- Sea ice remains exactly as the ORCA2 card carries it, including its
  `unmeasured_features` tuple.

## OPEN

1. Resume round 281's trusted vector-form stage-1 V update split: reproduce
   the candidate completed RHS offline, then test entry/RHS/`rDt`/resolved
   mask/written order at matched program points.
2. Keep the vector-form fold unit off.  Its first upstream partner must be
   named before it is measured for landing again; no stabiliser is permitted.
3. When G1b certifies, reassess whether any newly certified np_CRV statements
   transfer under Decision 122.  The landed F-cell-area statement needs no
   re-landing.
