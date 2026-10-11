# ORCA2 round 242 — generic HPG route repair and round-241 landing

Date: 2026-10-11. Frozen base: `560309036`. Preregistration commit:
`ed5d5c735`. Candidate commits: `ebaba1a78`, `bc78ce6dd`, and
`14287fdd6`. Status: **LANDED**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round242/`.

This round changes no deck, selector, carried state, stabiliser, sea-ice
setting, or `unmeasured_features` entry. Results below explicitly retain the
independent / given-NEMO-entry labels. The round-217 complete transport unit
remains a private diagnostic and does not land.

## Statement and shared-path repair

The executing ORCA2 build writes literal HPG without a mask factor at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`. It applies the
resolved raw face mask later, while depth-integrating the three-dimensional
RHS, at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`. Round 241's
source-exact implementation preserves that order.

Round 241 incorrectly made the new boundary depend unconditionally on raw
NEMO mesh operands. That is valid for ORCA2, but invalid for analytic cards
such as GYRE. Round 242 makes operand selection representation-aware:

* cards carrying the raw NEMO face bundle retain the production
  `nemo_qco_resolved_mesh_operands` route;
* analytic cards retain their existing card-built face thicknesses and masks;
* both paths materialise the V mask at the full vertical shape before the
  literal HPG boundary.

This is a representation repair, not a new configuration choice. The five
GYRE tests that exposed the regression pass 5/5. The exact 128-test push set
passes 128/128 (110 + 18 at file boundaries).

## Frozen predictions

| prediction | result |
|---|---|
| P1: all five GYRE regressions pass; analytic route restored; ORCA2 raw route retained | **CONFIRMED** |
| P2: round-241 V-RHS closure remains 668/1,050 -> 0 under both labels | **CONFIRMED** |
| P2: ORCA2, GYRE, DINO and tank trajectories do not move above the fixed floor | **CONFIRMED** |
| P3: GYRE 210 ladder arrays and daily snapshots stay byte-identical | **CONFIRMED** |
| P4: `vn_adv` falls from 919/8,589 unequal to approximately zero | **REFUTED**: unchanged at 919/8,589 |
| P4: materialised `zFv` becomes exact | **REFUTED**: unchanged at 19,516/226,637 |
| P5: four-part kt=7->8 bisect is reached only after exact `zFv` | **NOT REACHED**, as preregistered |
| P6: citation and plant controls fire | **CONFIRMED** |

The fresh source-exact RHS replay (`r241_rhs_fresh.json`, SHA-256
`9dbeb5238c0cc34ea55e1c6b85221c377f9389af3adfb86c7046fe163b70c82b`)
reports, for both independent and given-NEMO-entry labels: baseline 668
unequal, candidate zero unequal, 668 changed support slots, zero off-support
changes, maximum residual 0.0, float64 execution, and a passive completed
state. This completes the scientific and shared-card disposition that round
241 lacked.

## Trajectory gates

Fresh current-tree OMT-4, rung-0, and shipped rung-10 gates reproduce their
round-241 JSON exactly after removing only the worktree provenance block:

| card / label | result | current evidence SHA-256 |
|---|---|---|
| OMT-4, independent and given NEMO's entry, kt=1..10 | `PASS_R223_OMT4_CARD_AND_LADDERS`; 400/400 rows identical to round 241 | `d486a79cf8e6c5cc9c01f0f99c6386df85e05eaf39e3dbc19bcad16bea8cf167` |
| rung 0, independent, kt=1..10 | `PASS_RUNG0_TEN_STEP_LADDER`; 200/200 rows identical | `cf3ec7d832c9d461e034ffe02ceca59d7a84fabc001b75ceb461428351e20257` |
| shipped rung 10, given NEMO's entry, kt=1..10 | `LADDER_MEASURED`; all scientific content identical | `b22af638c70fadeec93a08b5faf4948a2a061ce26470e9c43348bf960a9fc398` |

The first-over-bar rows, exact rows, and every registered last-bit movement
therefore remain exactly those in round 241; no certified row is lost. The
rung-10 gate used the admitted round-5 rank-complete entry/surface record and
the compiled `R5FULLENTRY` source. A first attempt against the older phase-2x
record correctly refused for missing rank-complete streams and is retained
here as an instrument-use failure; no number from it is cited.

GYRE's fresh ten-step ladder has 0/210 differing residual arrays and retains
NPZ SHA-256
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
All thirty fresh daily snapshot NPZ files are byte-identical to the pre-round
files; day 30 remains
`3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.
Only `manifest.json` changes, because its producer provenance records the new
commit. The focused ORCA2/DINO/tank/OVERFLOW/card battery passes 71/71.

## Downstream discrimination — held unit

The current production D114 operand route was named explicitly for both
labels. The five-operand table retains exact `e1v`, live `e3v`, `vv`, and
`vmask`, but:

| label | `vn_adv` | materialised `zFv` | fold-band S max | fold-band T max |
|---|---:|---:|---:|---:|
| independent | 919/8,589 unequal; max 24.341577728515905 | 19,516/226,637 unequal; max 339,003.55099822127 | 9.604808130845299e-4 PSU | 1.3611107582029436e-3 K |
| given NEMO's entry | same | same | 9.604808130845299e-4 PSU | same maximum |

Thus the literal HPG boundary is necessary but is not the downstream owner.
The refreshed passive external-substep table agrees under both labels:

* first complete/fold-band over-floor row: substep 1 `slow_v`, 35 cells,
  maximum 1.6557659420864476e-6, zero interior cells;
* first non-rounding interior row: substep 6 `metric_transport_v`, 29 interior
  cells, interior maximum 8.189330110326409e-4 and fold-band maximum
  2.690599228371866.

The old substep-3 filter-alpha owner is gone, as Decision 115 predicts. The
complete round-217 unit remains private and HELD. No package change is made
for the substep-6 row in this round.

## Test and control disposition

The once-only `tests/ocean/fidelity -n 12` battery collected 3,203 tests and
was interrupted at 99% after one worker stopped returning: 3,179 passed, 7
skipped, 4 failed, and the active node had not reported. That active node,
`test_prediction_plant_is_fail_closed`, passes alone, 1/1 in 1,191.63 s.

All four printed failures reproduce in serial isolation and are unrelated to
this round:

1. the known worktree-stamp emitter ratchet (13 historical emitters);
2. the known allow-dirty stamp-scope ratchet (13 historical drivers);
3. the known SI3 scalar-math provenance red (`A MY_SRC is not verbatim`);
4. the GYRE round-129 spread-floor record gate, which refuses because its
   certified year harness moved after the registered members ran. The same
   failure reproduces at frozen base `560309036`, proving it is pre-existing.

The focused 71 tests, exact 128 push tests, five GYRE regression tests, and
the isolated long-running tail node all pass. No test was weakened or edited.

The default receipt citation gate passes with zero unmapped citations after
the required SequenceMatcher re-anchor. Its real rendered citation plant
fires. A first plant aimed only at a generated map entry exited through the
wrong usage path and is retained as an instrument failure; the replacement
plant targets a rendered citation and exits red as required.

## Independent review

Independent review is attempted after the final receipt and citation map are
committed. Its exact verdict is recorded in
`round242/independent_review.log`; if Codex cannot initialize under the
read-only sandbox, the round records “independent review unavailable
in-sandbox” rather than inventing a review.

## OPEN

Continue the OMT-4 external-mode walk from the source-ordered substep-6
`metric_transport_v` row. Split its recorded operands offline, using the
current production D114 mesh route and both initial-state labels. The first
fold-local `slow_v` row remains registered debt but does not explain the
global substep-6 departure. The complete round-217 transport unit remains
private until the new downstream owner makes `zFv` exact or names the next
cancelling partner.

No acquisition or configuration decision is needed.
