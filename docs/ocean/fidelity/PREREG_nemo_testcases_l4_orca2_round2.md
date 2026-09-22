# NEMO testcase Lane 4 — ORCA2 card round 2 preregistration

Date: 2026-09-22

Parent: `81dfaa3466f8ef5d5013450ae2f3f7bc760a8b43`

Status: **PREREGISTERED BEFORE ROUND-2 NUMERICAL MEASUREMENT.**

Scope is the ocean-only `orca2_vector_een_c2` card.  The six-entry sea-ice
registry is frozen and remains outside scope.  Decision 52 authorizes the
step-1 twin to use NEMO's recorded ORCA1ICE sea-surface height as an explicit
ocean entry operand; it does not authorize a sea-ice selector change.  Every
twin result below is labelled **given NEMO's entry**.  The card's own initial
state is labelled **independent** and is not mixed into a twin trajectory.

## Frozen records and order

The pinned oracle remains
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_a_10step_np2`.
The operator log
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/autopilot_orca2/iterm_1/acquisition.log`
is admitted only for what its marker states: round-1 surface-acquisition
**preflight** readiness.  It is not treated as proof that NEMO ran.

The ordered walk is:

1. run the round-1 ladder gate unchanged against the pinned root;
2. require and schema-check all ten `oracle_ocean_surface_input_kt*.bin`
   records before executing any candidate step;
3. if they exist, bridge only kt=1 SSH from the pinned entry frame, prove
   T/S/u/v/ssh exact, and run the card through kt=10 with the exact per-step
   operands;
4. stop at the first non-bit statement in NEMO execution order and cite the
   compiled branch that runs;
5. if any surface record is absent, stop `STOPPED_FOR_RECORD`, name every
   missing stream, and hand back the already committed fail-closed acquisition.

No V2-versus-ORCA1ICE differential is a legoESM candidate result.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R2-P1 | The operator action was preflight-only, so the pinned root still contains only the kt=1 surface frame. | The log says `ORCA2_ROUND1_SURFACE_PREFLIGHT_READY`; kt=2 through kt=10 frames are absent. | A completed acquisition admission exists and all ten frames schema-check. |
| R2-P2 | The unchanged round-1 gate therefore returns `STOP_RECORD_GAP`. | Exact missing list is kt=2 through kt=10; exit status is 2. | The gate returns ready or names a different missing set. |
| R2-P3 | Given NEMO's entry, Decision 52 would make kt=1 T/S/u/v/ssh exact once the candidate ladder is executable. | All five rows are bit-identical after replacing only SSH from the pinned kt=1 entry. | Any row is non-bit, or any operand besides SSH is replaced. |
| R2-P4 | The first post-entry non-bit statement and its kt=10 carried magnitude remain UNMEASURED until R2-P1 is refuted by an admitted ten-frame record. | The round stops before candidate execution and makes no statement-level numerical claim. | A candidate step runs without ten admitted frames, or a statement is named from a NEMO-root differential. |

The availability check is exact file inventory plus schema/EOF validation, not
a tolerance.  The existing one-representable-step T plant must still refuse.

## Stop and landing rules

- Missing kt=2..10 surface frames force `STOPPED_FOR_RECORD`; no inferred or
  carried-forward forcing is allowed.
- A preflight marker is not an acquisition marker.
- No `packages/` edit is eligible in this record-stop round, so no GYRE
  trajectory claim is made.
- Failed predictions remain in the receipt as `REFUTED`.
- The independent ORCA2 initial-state transcription required by Decision 52
  remains a separate scheduled round after the step-1 walk names its first
  statement.

## Frozen card registry

```text
staged_gm_eiv
linear_implicit_bottom_drag
internal_wave_mixing
spatial_lateral_viscosity
freshwater_budget_carry
si3_jpl5_layered_prather_state
```
