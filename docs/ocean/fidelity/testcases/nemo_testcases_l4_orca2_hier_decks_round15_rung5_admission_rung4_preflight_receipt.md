# ORCA2-DECKS round 15 receipt — plant repair, rung 5 admitted, rung 4 staged

Date: 2026-10-02

Disposition: **STOPPED_FOR_RECORD**.  The shared hierarchy CLI plant contract
is repaired, Decision 83's independent rung-5 replacement is admitted from
the existing operator run without rerunning NEMO, and rung 4 is sealed as the
same one-line `nn_havtb=1 -> 0` replacement.  Its boundary from replacement
rung 5 remains shortwave-only.  Rung 4 is UNMEASURED pending the operator.

Base: `3731c75a0`.  Preregistration commit: `42169850e`, before changing the
gate or admitting the record.  Checker commits: `fcf285245` and `931379981`.
Rung-4 tooling commit: `3be44f0cb`.  Every run claim is labelled
**independent**: NEMO starts from the rung's own from-rest initialization.

## Plant-contract repair

The operator's first rung-5 admission stopped on `terminal-nonfinite`: the
round-2 validator raised its own `GateError`, but the round-14 CLI's explicit
exception inventory ended before that imported class and emitted a traceback
without `STATUS PLANT-FIRED`.  After the raising validator printed its two
terminal plant markers directly, the next inherited `ice-sentinel-read` plant
exposed the same class-inventory defect.  The final repair makes every
hierarchy CLI catch the shared `RuntimeError` base used by every hierarchy
`GateError`; no plant predicate, clean-record predicate, data, or threshold
changed.

A parameterized CLI test injects a foreign hierarchy `GateError` for every
declared non-`none` plant across rounds 1 through 15.  All **247** declared
plant/CLI pairs report `STATUS PLANT-FIRED` and exit nonzero inside the
380-test complete hierarchy battery.  The two real round-2 terminal plants
also print before re-raising.  HD15-P2 and HD15-P3 are **CONFIRMED**.

## Rung-5 independent admission

The round-14 launcher was run with `--admit-existing`; it did not invoke NEMO.
Every real record plant fired, then the clean gate reported
`PASS_RUNG5_HAVTB0_RECORD`.  The admitted record contains:

- 480 self-describing frames: 3,840 PRESENT and 960 ABSENT payloads;
- eight finite T/U/V/W month files with 92 floating variables;
- two finite fp64 step-240 ocean restart shards and no ice product;
- resolved constant mixing with `nn_havtb=0`, runoff off, and TKE absent; and
- a complete 536-regular-file SHA-256 inventory.

The clean admission is stamped at checker commit `931379981`; the NEMO record
producer remains `3731c75a0`.  The earlier reported `sshn` non-finite value was
only the synthetic plant.  The real restart is finite.  HD15-P1 and HD15-P4
are **CONFIRMED**.

## Compiled source and complete rung-4 delta

NEMO reads `nn_havtb` in the closure namelist, initializes the background
shape uniformly, enters the equatorial shaping statements only for value one,
and applies that shape to tracer diffusivity
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  Decision 83 therefore changes exactly this physical line in the
superseded rung-4 deck:

```diff
-   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)
+   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)
```

The parsed replacement delta is exactly
`namzdf.nn_havtb: [1, 0]`, physical line 403.  Exact deck SHA-256 is
`f764a578...bdfa`; execution-deck SHA-256 is `dc810d64...1c34`.

With adjacent rungs both resolving `nn_havtb=0`, the complete rung-5 to rung-4
assignment boundary is exactly:

```text
namsbc.ln_traqsr          true -> false
namtra_qsr.ln_qsr_rgb     true -> false
namtra_qsr.nn_chldta         1 -> 0
```

The compiled deck reads reference then configuration values, refuses any
enabled penetration configuration that does not select exactly one scheme,
and names RGB and two-band as separate arms
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/traqsr.f90:1101-1103`,
`:1123-1137`).  The stage invokes penetration only under `ln_traqsr`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:525`);
with it off, the stage-1 surface statement transfers `qsr` into `qns` and
zeros `qsr`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trasbc.f90:267-270`).
Thus the boundary remains exactly the already source-resolved shortwave
module.  HD15-P5 is **CONFIRMED**.

## Rung-4 acquisition

The fail-closed launcher pins the admitted rung-5 replacement, the complete
value-1 rung-4 admission and inventory, both exact old decks, the repaired
binary/writer/step program, CPP keys, inputs, new manifest, and the compiled
background-shape branch.  Before running, it preserves without deletion:

- `record` as `record_havtb1_superseded`;
- `deck` as `deck_havtb1_superseded`; and
- `rung4_admission.json` as `rung4_admission_havtb1_superseded.json`.

It reuses the unchanged instrumented build and two-rank, from-rest, 240-step
protocol; it never invokes `makenemo`.  All six preflight plants fire and the
clean launcher prints `ORCA2_HIERARCHY_RUNG4_HAVTB0_PREFLIGHT_READY`.
No value-0 rung-4 record existed before handoff.  HD15-P6 is **CONFIRMED**.

## Gates, tests, review, and scope

- The real rung-5 admission passes all record plants and the clean gate.
- The complete hierarchy rounds 1-15 battery passes **380/380**; the focused
  round-2/round-15 battery passes **267/267**.
- The receipt citation gate passes all six citations with zero failures and
  zero unmapped citations; its shifted `zdfphy` plant fires.  The cumulative
  default receipt gate passes 274 citations with zero unmapped citations.
- The one allowed `tests/ocean/fidelity -n 12` run selected 2,557 tests,
  reached 99% with no visible failure, then entered the known late-suite
  no-output tail and was interrupted.  It produced no terminal summary, so it
  is not claimed PASS and was not rerun.
- Separate `codex exec --sandbox read-only` review returned before reading the
  diff: `failed to initialize in-process app-server client: Read-only file
  system`.  Verdict: **independent review unavailable in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.  No NEMO source was modified.

Search-before-build found and reused round 8's rung-4 deck/source checks,
version-2 frame parser, recorder and admission logic, plus round 14's
Decision-83 preservation pattern.  ASKED choice: Decision 83 assigns
`nn_havtb=0` to rungs 0 through 6.  UNASKED choices: none.

## Prediction ledger

| prediction | status |
|---|---|
| HD15-P1 failure ownership | **CONFIRMED**; clean existing record admitted |
| HD15-P2 single repair point | **CONFIRMED**; inherited terminal and resolved plants report correctly |
| HD15-P3 complete plant contract | **CONFIRMED**; every declared hierarchy CLI plant reports and exits nonzero |
| HD15-P4 rung-5 admission | **CONFIRMED**; complete independent record admitted |
| HD15-P5 rung-4 one-line delta | **CONFIRMED**; one replacement line, shortwave-only edge |
| HD15-P6 acquisition disposition | **CONFIRMED**; preflight-ready, value-0 record absent |

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round15_acquisition/run.sh --run`.
   Rung 4 remains UNMEASURED until every real plant fires and the clean gate
   admits 480 frames, month products, terminal restarts, and its inventory.
2. After rung 4 admits, repeat the one-line Decision-83 replacement for rungs
   3, 2, and 1 in order, preserving each value-1 record.
3. Re-run the rung-1 versus main-rung-0 semantic diff after rung 1 is replaced
   and the main lane harmonizes its cited-inert assignments.

## UNVERIFIED

- NEMO has not executed the replacement rung-4 deck.
- Runtime `nn_havtb=0`, its 480 frames, terminal restarts, month products, and
  replacement SHA inventory are unmeasured.
- The broad ocean-fidelity battery produced no terminal summary.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round15_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung4/record`.
