# ORCA2-DECKS round 14 receipt — rung 6 admitted, rung 5 staged

Date: 2026-10-02

Disposition: **STOPPED_FOR_RECORD**.  Decision 83's independent replacement
rung-6 record is admitted.  Rung 5 is sealed as the same one-line
`nn_havtb=1 -> 0` replacement, and its complete boundary from replacement
rung 6 remains runoff-only.  The committed fail-closed launcher awaits the
operator; rung 5 is UNMEASURED.

Base: `e4a959788`.  Preregistration:
`PREREG_nemo_testcases_l4_orca2_hier_decks_round14.md`, commit `3d5f82437`,
before opening the operator log, reading the replacement record, or measuring
the rung-5 deck.  Acquisition/gate commit: `1365f69a7`.  Every run claim is
labelled **independent**: NEMO starts from that rung's own from-rest
initialization.

## Decision-83 hierarchy inventory

| rung | `nn_havtb` | oracle-record status |
|---:|---:|---|
| 10 | 1 | **ADMITTED**, unchanged |
| 9 | 1 | **ADMITTED**, unchanged |
| 8 | 1 | **ADMITTED**, unchanged |
| 7 | 1 | **ADMITTED**, unchanged; shipped shape enters with TKE settings |
| 6 | 0 | **ADMITTED** this round |
| 5 | 0 | replacement deck **PREFLIGHT PASS**; record UNMEASURED |
| 4 | 0 required | value-1 record superseded by Decision 83; reacquisition follows rung 5 |
| 3 | 0 required | value-1 record superseded by Decision 83; reacquisition follows rung 4 |
| 2 | 0 required | value-1 record superseded by Decision 83; reacquisition follows rung 3 |
| 1 | 0 required | value-1 record superseded by Decision 83; reacquisition follows rung 2 |
| 0 | 0 | main-lane owned; out of side-lane scope |

## Rung-6 admission

The operator's round-13 launcher ran at producer commit `e4a959788` and ended
with `ORCA2_HIERARCHY_RUNG6_HAVTB0_ACQUISITION_PASS`.  The clean gate reports
`PASS_RUNG6_HAVTB0_RECORD`; all eleven record plants independently rerun this
round and printed `STATUS PLANT-FIRED`.  The record contains:

- 480/480 self-describing frames and 4,800 finite field payloads;
- eight finite T/U/V/W month files with 94 floating variables;
- two finite fp64 step-240 ocean restart shards and no ice product;
- resolved constant mixing with `rn_avm0=1.2e-4`, `rn_avt0=1.2e-5`,
  `nn_avb=0`, `nn_havtb=0`, and no TKE initializer; and
- a complete 540-regular-file SHA-256 inventory.

The admitted value-1 record, deck, and admission were moved without deletion
to their required `*_havtb1_superseded` paths.  Its full 528-file inventory
still validates against the pinned `SHA256SUMS`; its admission and inventory
digests remain `45d76acb...e1d46` and `f2e3f89d...8f488`.  HD14-P1 and
HD14-P2 are **CONFIRMED**.

## Compiled statement and exact rung-5 delta

The compiled vertical-physics initializer reads `nn_havtb` with the background
coefficients, initializes the horizontal multiplier to one, applies the
equatorial factor only for selector value one, then writes the result into
every tracer-diffusivity level
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  Decision 83's zero arm therefore retains the constant-mixing
coefficients and makes the background uniform.

This is every physical namelist line differing from the admitted value-1
rung-5 deck:

```diff
-   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)
+   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)
```

The parsed replacement delta is exactly
`namzdf.nn_havtb: [1, 0]`, at physical line 403.  The replacement deck SHA-256
is `9bb44379...e71ead`; the execution deck, retaining the previously gated
unread-TKE sentinel, is `aa1c49f2...e7e70`.

With both adjacent rungs now resolving `nn_havtb=0`, the complete parsed
rung-6 to rung-5 assignment boundary is exactly:

```text
namsbc.ln_rnf  true -> false
```

NEMO initializes the runoff module but calls its active surface update only
when `ln_rnf` is true
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:359-374`,
`:517-521`).  Thus the hierarchy edge remains one module: runoff heat and
water.  HD14-P3 and HD14-P4 are **CONFIRMED**.

## Acquisition and evidence preservation

Search-before-build reused round 7's repaired recorder, version-2 header
parser, and owner-off runoff convention; no second parser, format, binary, or
NEMO build was created.  The launcher pins the repaired binary and compiled
writer, verifies the complete old inventory, then preserves without deletion:

- the admitted value-1 record as `record_havtb1_superseded`;
- its deck as `deck_havtb1_superseded`;
- its admission as `rung5_admission_havtb1_superseded.json`; and
- the earlier SIGSEGV record as `record_pre_absent_repair_failed`.

It installs the preflight-gated deck as `deck`, creates the replacement under
the canonical `record` path, reuses the repaired binary, and requests the same
two-rank from-rest 240-step protocol.  It never invokes `makenemo`.  All six
preflight plants fire, and the clean launcher prints
`ORCA2_HIERARCHY_RUNG5_HAVTB0_PREFLIGHT_READY`.  HD14-P5 is **CONFIRMED**.

## Gates, tests, review, and scope

- The rung-6 admission rerun passes; all eleven record plants fire; the
  superseded record's complete inventory passes.
- The complete hierarchy rounds 1-14 focused battery passes **123/123** (split
  74/74, 27/27, and 22/22).  Initial historical failures exposed that old gates
  still followed the replaced rung-6 path; the checker-only repair now selects
  the explicitly superseded evidence and changes no scientific record.
- The receipt citation gate passes all 4 citations with zero failures or
  unmapped citations; its shifted `zdfphy` plant fires.  The cumulative default
  receipt gate passes 274 citations with zero failures or unmapped citations.
- The single allowed `tests/ocean/fidelity -n 12` battery selected 2,300 tests,
  reached 99% with no visible failure marker, then entered the known late-suite
  no-output stall and was interrupted.  It produced no terminal summary, so it
  is not claimed PASS and was not rerun.
- Separate `codex exec --sandbox read-only` review returned before reading the
  diff: `failed to initialize in-process app-server client: Read-only file
  system`.  Verdict: **independent review unavailable in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.  No NEMO source or admitted record
  was modified by this round.

ASKED choice: Decision 83 assigns `nn_havtb=0` to rungs 0 through 6 and value
one to rung 7.  UNASKED choices: none.  The controlled comparison changes one
namelist assignment and retains the run protocol byte-for-byte.

## Prediction ledger

| prediction | status |
|---|---|
| HD14-P1 rung-6 admission | **CONFIRMED**; complete record and all plants pass |
| HD14-P2 rung-6 preservation | **CONFIRMED**; old admission and 528-file inventory remain pinned |
| HD14-P3 rung-5 one-line delta | **CONFIRMED**; one parsed assignment and one physical line |
| HD14-P4 rung-6 to rung-5 edge | **CONFIRMED**; runoff is the only changed assignment |
| HD14-P5 acquisition disposition | **CONFIRMED**; no value-0 rung-5 record exists and the committed launcher is preflight-ready |

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round14_acquisition/run.sh --run`.
   Rung 5 remains UNMEASURED until every record plant fires and the clean gate
   admits its 480 frames, month products, terminal restarts, and inventory.
2. After rung 5 admits, repeat the one-line Decision-83 replacement for rungs
   4, 3, 2, and 1 in order, preserving each value-1 record.
3. Re-run the rung-1 versus main-rung-0 semantic diff after rung 1 is replaced
   and the main lane harmonizes its cited-inert assignments.

## UNVERIFIED

- NEMO has not executed the replacement rung-5 deck.
- Runtime `nn_havtb=0`, 480 frames, terminal restarts, month products, and the
  replacement rung-5 SHA inventory are unmeasured.
- The broad ocean-fidelity battery produced no terminal summary.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round14_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung5/record`.
