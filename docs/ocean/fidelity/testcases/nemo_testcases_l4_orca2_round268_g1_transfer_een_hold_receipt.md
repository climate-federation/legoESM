# ORCA2 round 268 — G1 transfer and EEN statement hold

Date: 2026-10-13  
Lane base: `ec14bd3d857`  
Labels: **independent** and **given NEMO's entry** (never mixed below)  
Verdict: **HELD**. No model or card change lands in this round.

## 1. Frozen question and imported statements

The preregistration is `PREREG_nemo_testcases_l4_orca2_round268.md` at
`2ce894bf3`. The geometry-ladder handoff was read at pinned tip
`43f4f070c18d`. Its two selector-gated statements were transferred without
re-derivation for measurement:

1. the EEN trend reads the fold-image V transport row; NEMO constructs the
   live V transport at `dynvor.f90:783-785` and consumes its north/south
   triads at `dynvor.f90:789-803`;
2. the split-explicit face-depth statement reads NEMO's wet fold-image V row
   and T-point north neighbour, as cited and gated by geometry-ladder round 3.

The transfer was commit `6ced6cd5a`; the matched-program-point instrument was
temporarily restored at `9fed81f10`. Both, together with the refuted curl arm,
were retired at `5c0352d31`. The final `packages/` tree is byte-identical to
the round base (`git diff ec14bd3d8 -- packages` is empty). The six sea-ice
selectors and the ORCA2 card's `unmeasured_features` tuple never changed.

## 2. Source order and first non-bit statement

The compiled ORCA2 `np_CRV` branch forms absolute vorticity at
`dynvor.f90:762-771`, the live U/V transports at `dynvor.f90:782-786`, the
four EEN triads at `dynvor.f90:788-794`, and adds the U/V trends to the stage
accumulators at `dynvor.f90:796-803`. This is the branch executed by OMT-4.

At the matched stage-2 entry, every carried field is below the campaign's
`2e-10` floor under both labels. The first valued boundary is therefore the
**stage-2 EEN vorticity accumulator**, not carried stage state:

| label | entry ssh max | entry T max | entry S max | entry u/v max | after-VOR max | argmax |
|---|---:|---:|---:|---:|---:|---|
| independent | 0 | 3.55e-15 | 7.11e-15 | 1.39e-17 / 1.39e-17 | 1.2863307536456779e-08 | `[19,33,3]` |
| given NEMO's entry | 0 | 3.55e-15 | 7.11e-15 | 1.39e-17 / 1.39e-17 | 1.2863307536456779e-08 | `[19,33,3]` |

The passive trace reproduced the untraced completed state bit-for-bit before
any intermediate was read. The after-VOR accumulator self-replay residual was
`3.3881317890172014e-21`, also below the floor.

## 3. G1 transfer result

Before transfer (round 267), after-VOR U had maximum
`1.7082229745349855e-06`, RMS `2.1588341793822282e-08`, and argmax
`[147,38,14]` under each label. With both G1 statements selected, both labels
gave maximum `1.2863307536456779e-08`, RMS `1.2191816847794566e-10`, and
argmax `[19,33,3]`. The maximum improves by about 133 times and leaves the
fold, but remains 64.3 floor units: preregistered closure prediction R268-P2
is **REFUTED**.

Because the transferred pair did not close its immediate source boundary,
no trajectory, month, GYRE, DINO, tank, or moved-row landing claim is made.
The selectors were removed rather than leave an incompletely gated physics
change in the lane.

## 4. EEN split attempted

A one-variable diagnostic replaced the EEN relative curl by legoESM's
existing source-ordered circulation helper, corresponding to NEMO's written
metric products at `dynvor.f90:762-770`. It was a null at the headline
maximum (`1.2863307536456779e-08` before and after) and worsened the fold-band
maximum from `2.038267000190315e-09` to `3.313017496944205e-09`. The arm was
therefore **REFUTED and retracted**; it is not a landing.

The remaining leaf lies after an at-bar entry and inside the EEN statement's
live absolute-vorticity/transport/triad arithmetic. Its interior argmax rules
out treating the imported fold-image row as the full owner.

## 5. Mechanical gate and controls

`nemo_testcase_l4_orca2_round268_g1_transfer_gate.py` requires both labels,
an at-bar matched entry, a strictly improved but still over-floor VOR row,
and the relative-curl null. Result: `PASS_R268_HELD_EEN_STATEMENT`.

The planted one-metre matched-entry violation exits 2 with
`REFUSE: independent: matched stage-1 entry left the floor`. Thus the new
entry predicate can fail and is not a file-self-comparison.

Evidence SHA-256 prefixes: classification `46436f994d3bb8d5`, independent
transfer `fb556ed85f0fcc0e`, given-entry transfer `487e359be4e4df8bb`,
independent entry `1245d0e36be21d58`, given entry `cdf166c0cb2709ba`,
relative-curl arm `396b687ea74f2044`, plant log `63c1d87fd8a0dad5`.

## 6. Tests, citations, and review

Focused recipe suite: **25 passed** in 314.22 s.

The required `tests/ocean/fidelity -n 12` run collected 3,466 items, reached
97%, and then stopped making progress with five failure marks and seven skips;
the worker hang prevented xdist from printing node IDs. Following the standing
recovery rule, test files were then run one at a time serially. Two documented
reds reproduced: the report-emitter worktree-stamp ratchet (1 failed, 9
passed) and the SI3 scalar-math provenance gate (1 failed, 4 passed). Three
additional files were isolated:

- GYRE round-129 spread-floor: `test_record_backed_gate_passes` refuses the
  stale registered record because the certified-year harness hash moved;
- GYRE year owners: `test_round179_source_layout_plant_fires` saw the expected
  acquisition clean-tree preflight while this receipt was uncommitted; it is
  re-run on the clean committed tree below;
- generic stage sweep: seven tests passed, including its long end-to-end
  planted control; `test_prediction_plant_is_fail_closed` then exhausted the
  bounded process after cumulative compilation and is re-run alone below.

Clean-tree isolated re-runs, citation audit, and independent read-only review
are recorded in the final paragraph below.

## OPEN

1. On the same passive matched stage-2 entry, replay the EEN statement in
   compiled order and split: `ff_f`; the relative circulation; `r1_e1e2f`;
   live `e3f_0vor*(1+r3f)`; `zwx/zwy`; the four triad sums; final metric and
   accumulator addition. Candidate-side replay must first reproduce the live
   completed VOR row before any leaf is cited.
2. The first operand whose source-exact substitution closes the interior
   `[19,33,3]` residual is re-tested with the two G1 statements as one atomic
   candidate. Only then run the OMT-4, rung-0, rung-10, GYRE, DINO, tank,
   moved-row, and month gates required for a Decision-96 landing.
3. The imported G1 face-depth statement remains measured but unselected on
   this lane until the atomic candidate earns those full gates.
