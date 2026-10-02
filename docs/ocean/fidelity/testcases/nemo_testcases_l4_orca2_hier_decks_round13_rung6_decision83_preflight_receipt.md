# ORCA2-DECKS round 13 receipt — Decision 83 rung-6 reacquisition preflight

Date: 2026-10-02

Disposition: **STOPPED_FOR_RECORD**.  The replacement rung-6 deck is sealed as
exactly one namelist-line change, `nn_havtb=1 -> 0`, from the admitted record.
The fail-closed launcher preserves that admitted record and deck under the
required `*_havtb1_superseded` names before creating the replacement run.  NEMO
has not run; the replacement record is UNMEASURED.

Base: `34a6e615f`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round13.md`, commit `9bc30c674`,
before constructing the replacement deck, reading its resolved assignment
diff, or running any gate.  Every run claim is labelled **independent**: NEMO
starts from that rung's own from-rest initialization.

## Hierarchy inventory after Decision 83

| rung | background shape | oracle-record status |
|---:|---|---|
| 10 | shipped `nn_havtb=1` | **ADMITTED**, unchanged |
| 9 | shipped `nn_havtb=1` | **ADMITTED**, unchanged |
| 8 | shipped `nn_havtb=1` | **ADMITTED**, unchanged |
| 7 | shipped `nn_havtb=1`, enters with TKE settings | **ADMITTED**, unchanged |
| 6 | Decision-83 uniform `nn_havtb=0` | replacement deck **PREFLIGHT PASS**; replacement record UNMEASURED; admitted value-1 record awaits launcher migration to `record_havtb1_superseded` |
| 5 | must change from admitted value 1 to 0 | **SUPERSEDED BY DECISION 83**; reacquisition follows rung 6 |
| 4 | must change from admitted value 1 to 0 | **SUPERSEDED BY DECISION 83**; reacquisition follows rung 5 |
| 3 | must change from admitted value 1 to 0 | **SUPERSEDED BY DECISION 83**; reacquisition follows rung 4 |
| 2 | must change from admitted value 1 to 0 | **SUPERSEDED BY DECISION 83**; reacquisition follows rung 3 |
| 1 | must change from admitted value 1 to 0 | **SUPERSEDED BY DECISION 83**; reacquisition follows rung 2 |
| 0 | Decision-83 uniform `nn_havtb=0` | main-lane owned; out of side-lane scope |

Rungs 10 through 7 retain the shipped value.  Rungs 5 through 1 remain on disk
and are not renamed or modified by this round; each is handled only after its
new upper neighbour is admitted.

## Compiled statement and complete deck delta

The compiled vertical-physics initializer reads the selector with the
background coefficients, initializes the horizontal multiplier uniformly,
and enters the equatorial shaping statements only for selector value one
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  Decision 83's zero arm therefore retains `avtb_2d=1` everywhere.
No stabilizer, fallback physics, or CPP-key change is involved.

This is every physical namelist line differing from the admitted rung-6 deck:

```diff
-   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)
+   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)
```

The parsed replacement delta is exactly
`namzdf.nn_havtb: [1, 0]`; its sole physical line is 403.  The complete rung-7
to replacement-rung-6 assignment boundary is exactly:

```text
namzdf.ln_zdfcst  <reference:false> -> true
namzdf.ln_zdftke  true -> false
namzdf.nn_havtb   1 -> 0
```

Thus the background shape enters at rung 7 together with TKE's shipped
settings, as Decision 83 requires.  The exact replacement deck SHA-256 is
`351ef6310b3f98e82dd4907bb370cb50288750db2502295c5c6c72eeef0feef8`;
the execution deck with the inherited unread-TKE sentinel is
`b4967ae2138af43c112d6826bf17d509a82e196da169b6881135bb5b9553089b`.

## Superseded-record preservation and acquisition

Preflight pins the admitted value-1 record by its admission digest
`45d76acb...e1d46`, its complete record-inventory digest
`f2e3f89d...8f488`, both exact namelists, its manifest, and its admitted
480-frame count.  Before running NEMO, the launcher verifies the full old
`SHA256SUMS`, then moves, without deleting:

- `record` to `record_havtb1_superseded`;
- `deck` to `deck_havtb1_superseded`; and
- `rung6_admission.json` to `rung6_admission_havtb1_superseded.json`.

It then installs the already-gated pending deck as `deck`, reuses the unchanged
instrumented binary and inputs, and requests the same two-rank, from-rest,
240-step protocol.  The operator must run the committed launcher; the sandbox
does not run `mpirun`.

## Gates, tests, review, and scope

- The committed launcher prints
  `ORCA2_HIERARCHY_RUNG6_HAVTB0_PREFLIGHT_READY`; all five preflight plants
  print `STATUS PLANT-FIRED`.
- The hierarchy rounds 1 through 13 focused battery passes **113/113**.
- The receipt citation gate passes 2 citations with zero failures and zero
  unmapped citations; its shifted `zdfphy` plant fires.  The cumulative default
  receipt gate passes 274 citations with zero failures or unmapped citations.
- The one allowed `tests/ocean/fidelity -n 12` battery selected 2,290 tests,
  reached 97%, then entered the known late-suite no-output stall and was
  interrupted.  Seven failure markers were visible, but xdist emitted no IDs
  or summary; they are **UNATTRIBUTED**, and no second broad battery ran.
- The exact deck, complete parsed assignment map, physical-line map, compiled
  source pin, superseded evidence pins, build/input/CPP/run pins, and resolved
  output are all fail-closed.  Header-driven frame, terminal restart, month,
  and complete inventory checks remain inherited for admission.
- Separate `codex exec --sandbox read-only` review was attempted and returned
  before reading the diff: `failed to initialize in-process app-server client:
  Read-only file system`.  Verdict: **independent review unavailable
  in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.  No NEMO source or existing
  record was modified during this round.

Search-before-build: the round-5 rung-6 gate, launcher, header parser, frame,
terminal, month, and inventory validators were reused; no second parser or
record format was added.  ASKED choice: `nn_havtb=0` on rungs 0 through 6 and
entry of value one at rung 7 is Decision 83.  UNASKED choices: none.

## Prediction ledger

| prediction | status |
|---|---|
| HD13-P1 superseded evidence | **PARTLY CONFIRMED**; exact old admission/deck/inventory pins pass, but migration waits for the run |
| HD13-P2 one-line Decision-83 delta | **CONFIRMED**; one parsed assignment and one physical line |
| HD13-P3 resolved uniform arm | **UNMEASURED**; frozen checks require the runtime print after acquisition |
| HD13-P4 rung boundary | **CONFIRMED**; only constant-for-TKE plus the authorized shape change |
| HD13-P5 replacement record | **UNMEASURED**; no replacement NEMO run exists |
| HD13-P6 acquisition disposition | **CONFIRMED**; committed preflight-clean launcher, no replacement record |

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round13_acquisition/run.sh --run`.
   Rung 6 remains UNMEASURED until all record plants fire and the clean gate
   admits 480 frames, month outputs, terminal restarts, and the inventory.
2. After rung 6 admits, rebuild and reacquire rungs 5, 4, 3, 2, and 1 in that
   order with the same one-line change, preserving each value-1 record under
   `record_havtb1_superseded`.
3. Re-run the rung-1 versus main-rung-0 semantic diff after the main lane
   harmonizes its cited-inert assignments; the expected remaining physics
   delta is only `ln_tradmp`/`ln_tsd_dmp` plus run-protocol lines.

## UNVERIFIED

- NEMO has not executed the replacement rung-6 deck.
- Runtime `nn_havtb=0`, 480 frames, terminal restarts, month products, and the
  replacement SHA inventory are unmeasured.
- The superseded record has not yet been migrated because migration occurs
  immediately before the operator run.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round13_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung6/record`.
