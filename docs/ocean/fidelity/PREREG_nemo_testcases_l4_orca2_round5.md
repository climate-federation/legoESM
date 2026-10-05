# NEMO testcase Lane 4 — ORCA2 card round 5 preregistration

Date: 2026-09-22

Parent: `e96931f39599dbf12751a6e6ee06e221c5db2dfa`

Status: **PREREGISTERED BEFORE ROUND-5 ADMISSION AND TRAJECTORY
MEASUREMENT.**

Scope is the ocean-only `orca2_vector_een_c2` card.  The six-entry sea-ice
registry is frozen and remains outside scope.  Decision 52 authorizes the
step-1 twin to use NEMO's recorded ORCA1ICE sea-surface height as an explicit
ocean entry operand.  Every trajectory result below is labelled **given
NEMO's entry**.  The card's own initial state is labelled **independent** and
is not mixed into the twin trajectory.

## Returned record and current instrument boundary

The operator reports that the round-4 acquisition ended at
`ORCA2_ROUND1_SURFACE_ACQUISITION_PASS`.  Its admitted root is the A twin under
`round4/acquisition/orca1ice_surface_every_step_a_np2`; B is its
reproducibility witness.  The operator report is availability information,
not a round-5 measurement.

The current round-1 ladder gate validates the two rank-local surface frames
for each kt and then returns `READY_FOR_CANDIDATE_TRAJECTORY`.  Its own
`trajectory_claim` remains `RECORDS_AVAILABLE_BUT_CANDIDATE_NOT_RUN`; it does
not assemble the two owned longitude slabs, bridge SSH, or advance legoESM.
Round 5 extends that existing gate rather than creating a second ladder.

## Frozen order

1. Admit the returned A/B roots from their frozen admission artifact: 20
   surface frames per twin, raw A/B equality, compiled rank-aware writer, and
   ordinary-output passivity.
2. Run the unchanged current ladder gate.  It must report 20/20 surface
   frames and `READY_FOR_CANDIDATE_TRAJECTORY` before candidate code runs.
3. Extend the same gate to decode the rank-local surface operands, concatenate
   only the owned slabs in rank order, and load the two NEMO initial-state
   shards.  Prove the assembled shapes against the instantiated 148x180x30
   card before indexing them.
4. Replace only the candidate's initial SSH with NEMO's recorded SSH, as
   authorized by Decision 52.  Prove T/S/u/v/SSH bit-identical at kt=1.
5. Advance the production-jitted WS-RK3 ocean with the recorded per-step
   surface operands through kt=10.  Score the ordered step-entry and available
   RK-stage state statements exactly; retain all rows after the first non-bit
   row so its carried kt=10 magnitude is measured.
6. If the first non-bit statement identifies a single shared-model defect,
   this round remains HELD and preregisters that statement for round 6.  No
   package edit is authorized by this preregistration.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R5-P1 | The returned record is complete and passive. | The frozen admission artifact reports PASS, 20 schema-valid frames, raw A/B equality, 126 streams, and ordinary-output identity. | Any admission field is absent/false, a digest disagrees, or either root is incomplete. |
| R5-P2 | The unchanged readiness gate now clears its record stop. | It reports required=20, present=20, no missing names, status `READY_FOR_CANDIDATE_TRAJECTORY`. | Any missing/schema-invalid frame or another fail-closed condition. |
| R5-P3 | Decision 52's entry bridge is exact in all five fields over the full 148x180 card. | T/S/u/v/SSH each have zero unequal cells at kt=1 after replacing only SSH from NEMO's recorded entry. | Any non-SSH field requires replacement, SSH remains non-bit, or the two NEMO shards cannot be assembled without an unstated convention. |
| R5-P4 | The first scored post-entry state is non-bit at kt=1 stage 1, and the first field in the fixed T/S/u/v/SSH order is T. | kt=1 entry is exact; kt=1 stage 1 T is non-bit. | Stage 1 is exact, another field precedes T, or an earlier scored statement is non-bit. |
| R5-P5 | The first non-bit row remains finite and nonzero at kt=10. | The same field has a finite positive max-absolute residual in the kt=10 entry or final stage. | It returns to bit identity, becomes non-finite, or the trajectory cannot reach kt=10. |

Failed predictions remain in the receipt as **REFUTED**.  A missing surface
channel is a record/API stop, not permission to reconstruct or zero it.

## Controls and stop rules

- Removing either rank's surface frame must restore `STOP_RECORD_GAP`.
- Perturbing one representable kt=1 T value must make entry identity refuse.
- Perturbing one active decoded surface operand must move a candidate row or
  make the gate refuse; a zero-valued plant is forbidden.
- Shifting the final receipt's compiled citation by two lines must make the
  citation gate fail.
- No configuration, carried-state, stabilizer, model package, or sea-ice
  selector change is authorized.
- If an acquired NEMO field has no existing production forcing channel, stop
  with `DECISION_NEEDED` or `ACQUISITION_NEEDED`; do not silently omit it.

## Frozen card registry

```text
staged_gm_eiv
linear_implicit_bottom_drag
internal_wave_mixing
spatial_lateral_viscosity
freshwater_budget_carry
si3_jpl5_layered_prather_state
```
