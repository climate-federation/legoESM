# NEMO testcase Lane 4 — ORCA2 card round 4 preregistration

Date: 2026-09-22

Parent: `ca36b12db0986da145a4dd654e66d6cf2a3b9247`

Status: **PREREGISTERED BEFORE ROUND-4 INVENTORY MEASUREMENT.**

Scope is the ocean-only `orca2_vector_een_c2` card.  The six-entry sea-ice
registry is frozen and remains outside scope.  Decision 52 authorizes the
step-1 twin to use NEMO's recorded ORCA1ICE sea-surface height as an explicit
ocean entry operand.  Results from that twin are labelled **given NEMO's
entry**.  The card's own initial state is labelled **independent** and is not
mixed into the twin trajectory.

## Returned record and source boundary

The operator ran the round-3 acquisition through both ten-step twins.  Both
runs completed, produced all ten surface frames, and then the acquisition's
own count guard refused: each run has 116 `oracle_*.bin` streams while the
script expects 118.

This is a Note-K reconciliation.  The compiled target calls the surface dump
unconditionally immediately after `sbc` at
`ORCA2_ORCA1ICE_OMIP_L4_R3SURFACE/BLD/ppsrc/nemo/stprk3.f90:151-152`; the dump
constructs a `kstp`-specific `STATUS='NEW'` file and closes it at
`stprk3.f90:398-437`.  These lines establish the intended ten-frame writer but
do not establish the total stream count.  Total-count ownership will be
determined from an exact filename inventory diff and the compiled call sites
for every discrepant stream.

## Frozen order

1. Compare the pinned baseline and both completed targets by exact stream
   filename; partition target files into inherited, newly added, and absent.
2. Prove A/B inventory equality, raw equality of all ten surface frames, ten
   dump notices, kt=10 completion, and schema/EOF validity.
3. For every baseline stream absent from both targets, cite the compiled writer
   and prove whether its guard executes under this deck.  Do not infer a guard
   from a filename or prior receipt.
4. If the two extra absent streams are intentionally conditional and every
   stronger admission check passes, change only the mechanically derived target
   count and run `--finalize` against the existing directories.  If a writer
   that should execute is absent, do not change the count; request a new target
   acquisition.
5. Only after admission, run the Decision-52 entry bridge and kt=1..10 ladder.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R4-P1 | The 116-versus-118 refusal is a stale total-count expectation, not a truncated surface acquisition. | Both targets have ten schema-valid surface frames, ten notices, kt=10 completion, identical filename inventories, and identical surface bytes. | Any target lacks a frame/notice/completion marker, fails schema/EOF, or A/B differs. |
| R4-P2 | Exactly two baseline streams beyond the seven already registered as absent are absent from both targets because their compiled writers are conditional on a branch this deck does not execute. | Exact inventory names two additional common absences and compiled source plus the resolved deck proves both writer guards false. | The shortfall is not exactly two, differs by twin, has no compiled conditional owner, or its owner should execute. |
| R4-P3 | With the target count derived as `baseline 116 - absent 9 + new 9 = 116`, the otherwise unchanged finalizer admits the existing twins. | The only executable edit changes the expected target count to 116; `--finalize` reaches its PASS marker and writes a ten-frame admission artifact. | Any other guard fails or an executable/scientific change is required. |
| R4-P4 | Given NEMO's entry, the admitted ladder can execute kt=1 through kt=10 and mechanically name the first post-entry non-bit statement. | The entry bridge is exact in T/S/u/v/ssh and the ladder produces registered per-step statement rows through the first non-bit row. | The entry bridge is non-bit, the ladder stops on another record/schema gap, or the gate cannot score ordered statements. |

The stream-count change is eligible only if R4-P1 and R4-P2 are confirmed.
Failed predictions remain in the receipt as **REFUTED**.  No missing stream is
manufactured, renamed, or waived.

## Controls and stop rules

- Removing one surface frame must make schema admission refuse.
- Perturbing one representable kt=1 entry value must make identity refuse.
- Shifting the final receipt's compiled citation by two lines must make the
  citation gate fail.
- Any intended writer absence forces `ACQUISITION_NEEDED` under a new target.
- No `packages/` edit is preregistered.  If the ladder identifies a model
  change, this round remains HELD and records the next single-statement plan.
- No configuration, carried-state, stabilizer, or sea-ice selector change is
  authorized.

## Frozen card registry

```text
staged_gm_eiv
linear_implicit_bottom_drag
internal_wave_mixing
spatial_lateral_viscosity
freshwater_budget_carry
si3_jpl5_layered_prather_state
```
