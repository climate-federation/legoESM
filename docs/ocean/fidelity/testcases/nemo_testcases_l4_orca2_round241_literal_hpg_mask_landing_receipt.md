# ORCA2 round 241 — literal HPG mask-boundary landing

Date: 2026-10-11. Frozen base: `cb476aa34`. Preregistration commit:
`12e59890b`. Candidate commit: `555cf67ea`. Status: **HELD** because the
mandatory full fidelity battery did not produce a terminal verdict.
Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round241/`.

This round changes no deck, selector, carried state, stabiliser, sea-ice
setting, or `unmeasured_features` entry. All numbers below are labelled either
independent or given NEMO's entry.

## First statement and source

The executing rung-0 build writes the literal V pressure-gradient tendency,
without a mask factor, at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/dynhpg.f90:386-427`. NEMO accumulates
HPG first at `ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175` and
applies the raw V mask only in the later depth integral at
`ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:203-215`.

legoESM instead multiplied the completed V tendency by its generic compact
mask at `ocean_pe_latlon_cgrid.py:5983-5996`. The landed statement preserves
the literal HPG contribution where NEMO's resolved raw V mask is wet and the
compact mask is dry. The raw mask comes from the production D114 resolved
mesh route at `ocean_model_latlon_cgrid.py:6007-6087`. The private hook is a
test-only legacy/force control; no public configuration choice was added.

## Frozen predictions

| prediction | result |
|---|---|
| P1: completed V RHS 668/1,050 unequal -> 0 under both labels | **CONFIRMED** |
| P1: zero changed slots outside the 668-slot support | **CONFIRMED** |
| P2: complete round-217 unit reaches kt=8 and S/T return to the OFF values | **REFUTED** |
| P3: boundary alone may land if all campaign gates pass | **UNMEASURED**; mandatory full-battery verdict unavailable |

The source-exact gate reports, for both independent and given-NEMO-entry
labels: baseline 668 unequal, arm 0 unequal, 668 changed support slots, zero
off-support changed slots, maximum residual 0.0, passive completed-state
trace, and float64 execution (`rhs.json`, SHA-256
`53a306033d39f2bda5d2fa3a3dea74f39da85c0a83b17a2a490573f180e298e6`).

The complete atomic fold/transport unit progresses through kt=7 but reaches
the same kt=8 `raw-mesh e3w_int must contain only finite values > 0` refusal
(`atomic.log`). Closing the completed V RHS is therefore necessary but not
sufficient. The first remaining recorded downstream owner class is the
external-mode V correction: after Decision 115 and 114, `vn_adv` remains
919/8,589 unequal and the materialised `zFv` remains 19,516/226,637 unequal.

## Trajectory gates

The candidate OMT-4, rung-0, and shipped rung-10 gates all pass. Compared to
round 238, no exact row is lost, no unequal count changes, and no row moves by
more than the fixed `2e-10` floor.

| card / label | raw summary rows changed | maximum RMS delta | rows above floor |
|---|---:|---:|---:|
| OMT-4, independent | 119 | 1.155e-15 | 0 |
| OMT-4, given NEMO's entry | 119 | 1.155e-15 | 0 |
| rung 0, independent | sub-floor only | 9.099e-16 | 0 |
| rung 10, given NEMO's entry | 145 | 6.026e-14 | 0 |

These are registered last-bit aggregate changes, not toward/away votes under
Decision 96. The first-over-bar row is unchanged.

## Shared-card and test gates

GYRE's 210 ten-step residual arrays are `np.array_equal` to round 238; the NPZ
SHA-256 remains
`3b3d865cf5c82d38e15b24356492572a85546e51e35f70c2e7c89e5b9cb9df5b`.
The fresh CPU/fp64 day-30 snapshot is byte-identical to the pinned baseline,
SHA-256
`3c0602babb535aac55512f3b82561d0f562b1ec51d542552efd8499a119b443b`.

The focused ORCA2/DINO/tank/OVERFLOW/card battery passes 71/71. The required
once-only `tests/ocean/fidelity -n 12` battery reached 98% (3,182/3,203 tests)
with five reds, then its final worker did not return; interruption therefore
prevented pytest from printing node IDs or a terminal summary. A targeted
isolation reached 7/29 stage-sweep tests before the same non-terminal worker
class. This infrastructure limitation is retained verbatim in
`fidelity_battery.log` and `known_reds_isolated.log`; no round-241 or focused
cross-card test is red.

The default citation gate passes with zero unmapped citations after the
required SequenceMatcher re-anchor. Its planted citation violation fires.
Independent review was attempted with `codex exec --sandbox read-only` and
failed before reading the diff: `failed to initialize in-process app-server
client: Read-only file system (os error 30)`. Independent review unavailable
in-sandbox; exact output is in `independent_review.log`.

## OPEN

The literal-HPG candidate must not merge until a serial or otherwise
terminating full fidelity battery identifies and dispositions the five reds
seen before the 98% worker hang. The exact blocker is: `tests/ocean/fidelity
-n 12` reached 3,182/3,203 with five failure markers, then the final worker
never returned and pytest printed no node IDs or terminal summary.

The complete round-217 unit remains private and HELD. The next walk begins at
the external-mode V correction (`zvb` / time-averaged `vn_adv`) downstream of
the now-exact completed 3-D V RHS, using the production D114 operand route.
The rung-0 month boundary must be re-scored only after that downstream owner
is named; this round makes no month-improvement claim.

No acquisition or configuration decision is needed.
