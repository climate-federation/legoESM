# ORCA2 round 49 preregistration — OVERFLOW cancelling-pair boundary walk

Date frozen: 2026-09-27  
Base: `ab8dd260e8ef198182e9e1d1962c19f3f95d75bf`  
ORCA2 claim label: **given NEMO's entry** (Decision 52)  
GYRE/OVERFLOW/LOCK claim label: **independent**

## Scope and source order

Round 48 proved that the source-ordered stage-1/2 QCO tracer assignment makes
the direct ORCA2 statement bit-exact, but its isolated production candidate
made five later OVERFLOW U rows exceed that card's strict two-row-scale-ULP
non-regression bar.  The candidate was reverted.  This round does not retry it
alone.  It walks the downstream OVERFLOW boundaries to find the first second
NEMO statement that can be tested as a cancelling pair.

The certified OVERFLOW run's `nemo.exe` and
`tests/OVERFLOW_OMIP_L1/BLD/bin/nemo.exe` both have SHA-256
`eb4acf9651b887a3da8834281112d472692caa0bbadcb0d69779e91dee92e6cb`.
Its executing preprocessed source:

1. advances stage-1/2 tracers with the QCO assignment at
   `OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:501-508`;
2. hands that stage tracer to the next stage's `eos` and `dyn_hpg` before any
   other momentum operator at `stprk3_stg.f90:311-338`;
3. dispatches this deck to TEOS-10 and the SCO pressure-gradient recurrence at
   `OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:684-718` and
   `OVERFLOW_OMIP_L1/BLD/ppsrc/nemo/dynhpg.f90:172-176,341-414`.

No configuration, selector, default, carried state, score domain, threshold,
stabiliser, or sea-ice field may change.  The ORCA2 card's six-item
`unmeasured_features` tuple is frozen.

## Existing instruments and record boundary

Repository search found the existing source-ordered replay in
`nemo_testcase_l4_orca2_round45_qco_rk_gate.py`, the existing OVERFLOW
trajectory and stage scorers, and private WRITE-only hooks for tracer stage
outputs, individual momentum operators, and stage-2/3 momentum RHS.  This
round extends those instruments; it does not add a second model ladder.

The admitted OVERFLOW record contains kt=1 stage endpoint frames and kt=1..10
step-entry frames.  It contains no stage-2/3 `rhd` or post-HPG accumulator.
Therefore a stage endpoint can rank where the held change first reaches
momentum, but cannot by itself certify an EOS or HPG statement bit-exact.  If
the walk stops at that boundary, the round writes a fail-closed acquisition
for the minimum missing arrays rather than inferring their values.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes / action |
|---|---|---|---|
| R49-P1 | The round-48 base and held candidate reproduce. | Base OVERFLOW trajectory matches the admitted baseline; the candidate reproduces the same five blocking U rows and maximum 57.8125 row-scale ULP. | Any changed row/count: stop and reconcile artifact or harness drift. |
| R49-P2 | The held statement first changes a tracer-stage boundary, not momentum computed earlier in the same stage. | The first base/candidate unequal internal frame is stage-1/2 T or S; the preceding momentum frame is array-equal. | Momentum differs before the changed tracer assignment: instrument ordering is wrong; do not attribute. |
| R49-P3 | The first downstream momentum movement enters through HPG. | At the first affected next stage, HPG differs between base/candidate while the stage-entry u/v and all momentum operators ordered before HPG are equal (there are none on this deck); later operator differences are inherited. | HPG remains equal but a later operator first differs: name that operator instead and retract the HPG prediction. |
| R49-P4 | The current record is insufficient to call the moved EOS/HPG statement bit-exact. | No admitted stream contains same-stage `rhd` plus post-HPG U/V RHS for the first affected step/stage. | An existing self-describing, binary-bound stream contains both: admit and score it instead of acquiring. |
| R49-P5 | No unmeasured pair lands. | If P4 confirms, disposition is `STOPPED_FOR_RECORD` with a new acquisition path; no model file changes remain. | A record-backed, NEMO-cited one-statement pair is found and passes every landing gate; then it may land. |

Failed predictions remain in the receipt as `REFUTED`; thresholds and scored
rows are not changed after measurement.

## Required measurements and landing rule

1. Reconstruct candidate `1722ef29a3` exactly and prove its diff is the single
   QCO production statement plus its already-reviewed tests/census support.
2. Run the existing OVERFLOW kt=1..10 trajectory gate and comparison to
   reproduce R49-P1.
3. Extend the existing stage scorer only enough to compare, in compiled order,
   tracer stage output, HPG, vorticity, advection, complete stage RHS, and raw
   stage velocity.  The instrument must carry fp64/worktree stamps and a
   planted payload change that refuses.
4. Census every admitted OVERFLOW stream before requesting data.  If `rhd` and
   post-HPG are absent, write `round49/.../acquisition/run.sh` under a NEW
   target name.  It must refuse an existing target/config, dirty source,
   wrong binary precision, wrong stream counts, wrong magic/version/step/stage,
   non-finite payloads, and any failure to change the compiled writer.
5. A model pair may land only if both statements are independently bit-exact
   given NEMO operands on every executing card and the pair passes ORCA2's
   40-checkpoint ladder, GYRE Decision 43/45/55/59, both tank gates, generic
   card, DINO/tank/generic census, citation gates with plant, focused/JIT/AD
   tests, the 170-test card battery, push gate, and one fidelity battery.

## Frozen OPEN

After this boundary is recorded or discharged, return to the first whole-card
ORCA2 non-bit checkpoint (kt=1 stage-1 T).  Round-20 slow forcing and the
Decision-52 independent ORCA2 initial state/year remain open.  Sea ice remains
out of scope at `STOP_SELECTOR_GAP`.

## Choices

ASKED: the user directed a cancelling-pair walk after a faithful shared
statement fails another card's gate.  
UNASKED: none.
