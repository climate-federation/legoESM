# NEMO testcase Lane 4 — ORCA2 card round 6 preregistration

Date: 2026-09-22

Parent: `0dff1f286a32b8dd460b6965cfc965f2ed16b1d2`

Status: **PREREGISTERED BEFORE ROUND-6 RECORD CENSUS, ADMISSION, AND
TRAJECTORY MEASUREMENT.**

Scope is the ocean-only `orca2_vector_een_c2` card.  The six-entry sea-ice
registry is frozen and remains outside scope.  Decision 52 authorizes the
step-1 twin to use NEMO's recorded ORCA1ICE sea-surface height as an explicit
ocean entry operand.  Every trajectory result below is labelled **given
NEMO's entry**.  The card's own initial state is labelled **independent** and
is not mixed into the twin trajectory.

## Returned acquisition and frozen order

The operator reports that the round-5 acquisition ran both twins to STOP 0,
wrote an admission JSON, and was followed by a second attempt that refused
because the target already exists.  This report is availability information,
not an admission or trajectory result.  The acquisition script itself pins
the target configuration `ORCA2_ORCA1ICE_OMIP_L4_R5FULLENTRY` and the A/B
roots under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round5/acquisition`.

The frozen order is:

1. print `ls -la` for that exact target configuration and both exact A/B run
   roots; do not infer a path from older receipts;
2. census every `oracle_*` file in A and B by basename, rank marker, kt, and
   byte size, and reconcile the census against the compiled `stprk3` WRITE
   statements;
3. require the existing acquisition's own admission marker and JSON, then run
   the unchanged round-5 ladder gate against the admitted A root;
4. only if all 20 surface and 20 step-entry slabs are present and schema-valid,
   assemble rank 0 then rank 1 owned slabs, bridge only kt=1 SSH, prove
   T/S/u/v/SSH bit-identical, and execute the production ocean ladder through
   kt=10;
5. name the first non-bit statement in NEMO execution order and report the
   magnitude it carries at kt=10.  If the current ladder has no production
   trajectory executor, extend that existing gate; do not create a parallel
   ladder.

No package edit or scientific/configuration choice is authorized by this
preregistration.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R6-P1 | The actual round-5 target and both exact A/B run roots exist. | `ls -la` resolves all three paths named by the current run script. | Any path is absent, empty, or points at a different target name. |
| R6-P2 | The compiled patch is rank-aware for both per-step surface and step-entry streams. | Compiled `stprk3` has one filename arm per MPI rank for each stream, outside a root-only guard, and all kt=1..10 A/B files exist. | A live `lwp`/`narea == 1` guard still excludes rank 1, a filename aliases ranks, or any expected rank-local file is absent. |
| R6-P3 | The completed record has 20 surface frames and 20 step-entry frames per twin, with ten conventional rank-0 names and ten explicit rank-1 names in each class. | Exact census and schema checks give 20/20 in both classes for A and B. | Any count, rank, kt, size, or schema differs. |
| R6-P4 | The existing admission is valid and the unchanged ladder clears `STOP_ENTRY_RECORD_GAP`. | Admission marker is PASS with 107 inherited streams passive; the ladder reports surface 20/20, entry 20/20, and candidate readiness. | Admission is absent/failing, ordinary output moved, or the ladder stops on a record gap. |
| R6-P5 | Given NEMO's entry, the Decision-52 bridge is bit-exact for all five kt=1 fields after replacing only SSH. | T/S/u/v/SSH each report zero unequal cells over the assembled 148x180 domain. | Any non-SSH field requires replacement, SSH remains non-bit, or slab assembly requires an unstated convention. |
| R6-P6 | The first post-entry non-bit statement is kt=1 RK stage 1 T, and its residual remains finite and nonzero at kt=10. | Entry is exact, stage-1 T is the first non-bit row, and the registered kt=10 magnitude is finite and positive. | An earlier row is non-bit, stage-1 T is exact, another field precedes it, or its registered kt=10 magnitude is zero/non-finite. |

Failed predictions remain in the receipt as **REFUTED**.  If the files exist
under names the admission does not expect, only the admission expectation may
be corrected.  If the compiled patch wrote fewer streams than intended, this
round writes a fail-closed acquisition under a fresh target name and stops for
that record.

## Controls and stop rules

- Removing either rank's surface or entry frame must restore the corresponding
  record stop.
- Perturbing one representable kt=1 T value must make entry identity refuse.
- Perturbing one active decoded surface operand must move a candidate row or
  make the gate refuse; zero-valued plants are forbidden.
- A rigid two-line shift of the final receipt's compiled citation must fail the
  citation gate.
- No configuration, carried-state, stabilizer, model package, or sea-ice
  selector change is authorized.
- The independent ORCA2 initial-state construction required by Decision 52 is
  scheduled only after this walk names the first non-bit statement.

## Frozen card registry

```text
staged_gm_eiv
linear_implicit_bottom_drag
internal_wave_mixing
spatial_lateral_viscosity
freshwater_budget_carry
si3_jpl5_layered_prather_state
```
