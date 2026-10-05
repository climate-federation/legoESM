# NEMO testcase Lane 4 — ORCA2 card round 3 preregistration

Date: 2026-09-22

Parent: `ae049c43d58273374273d6304eccf5f9e65d0401`

Status: **PREREGISTERED BEFORE ROUND-3 NUMERICAL MEASUREMENT.**

Scope is the ocean-only `orca2_vector_een_c2` card.  The six-entry sea-ice
registry is frozen and remains outside scope.  Decision 52 authorizes the
step-1 twin to use NEMO's recorded ORCA1ICE sea-surface height as an explicit
ocean entry operand; it does not authorize a sea-ice selector change.  Every
twin result below is labelled **given NEMO's entry**.  The card's own initial
state is labelled **independent** and is not mixed into a twin trajectory.

## Frozen record and ordered walk

The pinned oracle remains
`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_a_10step_np2`.
The returned operator log is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/autopilot_orca2/iterm_1/acquisition.log`.
Its marker is interpreted literally: `ORCA2_ROUND1_SURFACE_PREFLIGHT_READY`
admits preflight readiness only, not a NEMO run or ten-frame acquisition.

The ordered walk is:

1. run the unchanged round-1 ladder gate against the pinned root;
2. require all ten schema-valid `oracle_ocean_surface_input_kt*.bin` records
   before executing any candidate step;
3. if all ten exist, bridge only kt=1 SSH, prove T/S/u/v/ssh bit-identical,
   and execute the card through kt=10 with matching per-step operands;
4. otherwise stop at `STOP_RECORD_GAP`, retain the first post-entry statement
   as unmeasured, and return an operator script whose no-argument mode performs
   the acquisition rather than preflight only.

No V2-versus-ORCA1ICE differential is a legoESM candidate result.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R3-P1 | The returned operator action again ran preflight only; the pinned root still has only the kt=1 surface frame. | The log has only the preflight marker and the root lacks kt=2 through kt=10 frames. | A run-complete/admission marker exists and all ten frames schema-check. |
| R3-P2 | The unchanged ladder gate returns `STOP_RECORD_GAP`. | Exit status 2 and the exact missing list is kt=2 through kt=10. | The gate is ready or names a different missing set. |
| R3-P3 | Given NEMO's entry, the first post-entry non-bit statement and its kt=10 carried magnitude remain UNMEASURED. | No candidate step executes before the ten-frame record gate passes. | A candidate step runs without ten admitted frames, or a statement is inferred from the NEMO-root differential. |
| R3-P4 | The operator handoff failed procedurally because the supplied `run.sh` defaults to `--preflight-only`; making no-argument execution select its existing `--run` path removes that ambiguity without changing scientific configuration. | The only executable change is the mode default/comment; explicit `--preflight-only`, `--run`, and `--finalize` remain available; the script preflight and invalid-mode refusal still fire. | Any model, NEMO source, card selector, record schema, or acquisition physics changes. |

The availability check is exact inventory plus schema/EOF validation, not a
tolerance.  The one-representable-step kt=1 T plant must still refuse.

## Stop and landing rules

- Missing kt=2..10 surface frames force `STOPPED_FOR_RECORD`.
- A preflight marker is never promoted to an acquisition marker.
- No `packages/` edit is eligible in this record-stop round; no GYRE
  trajectory claim is made.
- The acquisition script may change only its operator-facing default mode.
- Failed predictions remain in the receipt as `REFUTED`.
- Decision 52's independent T/S/SSH transcription remains scheduled only
  after the step-1 walk names its first statement.

## Frozen card registry

```text
staged_gm_eiv
linear_implicit_bottom_drag
internal_wave_mixing
spatial_lateral_viscosity
freshwater_budget_carry
si3_jpl5_layered_prather_state
```
