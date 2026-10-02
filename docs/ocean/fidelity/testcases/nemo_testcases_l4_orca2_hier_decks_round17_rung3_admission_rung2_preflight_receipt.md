# ORCA2-DECKS round 17 receipt — rung 3 admitted, rung 2 staged

Date: 2026-10-02

Disposition: **STOPPED_FOR_RECORD**.  The completed Decision-83 rung-3 run is
admitted after separating its own surface predicates from the superseded
inherited background predicate.  Rung 2 is sealed as the same one-line
`nn_havtb=1 -> 0` replacement, and its complete boundary from replacement
rung 3 remains the previously admitted GM/MLE module.  Rung 2 is UNMEASURED
pending the operator.

Base: `1455aa447`.  Preregistration: commit `b2b98c118`, before changing the
checker, admitting the completed record, or measuring the rung-2 replacement
deck.  Checker repair: `6deb9c359`.  Rung-2 tooling: `b7a7b0741`.  Every run
claim is labelled **independent**: NEMO starts from that rung's own from-rest
initialization.

## Rung-3 refusal and admission

The operator run reached `STOP 0`.  The clean gate then called the historical
round-9 rung-3 validator, which descends through the superseded hierarchy and
requires `nn_havtb=1`.  The same round-16 gate also called the replacement
upper hierarchy, which correctly requires `nn_havtb=0`.  Its failure therefore
reported only the historical rung-6 row
`nn_havtb_retained: False`; no record predicate had failed.

The repair extracts the historical rung-3 surface block as a reusable
surface-only validator.  Historical round 9 still calls that block plus its
original value-1 inherited chain.  Round 16 calls the same block plus the
replacement value-0 upper chain.  A regression makes any call from round 16
to the superseded full validator fail.  The existing `surface-consequence`
and `resolved-havtb` plants both fire through their owning paths.  No
predicate, threshold, deck, or record data changed.  HD17-P1 and HD17-P2 are
**CONFIRMED**.

The operator record was admitted in place with `--admit-existing`; NEMO was
not rerun.  Every real plant fired before the clean result
`PASS_RUNG3_HAVTB0_RECORD`:

- 480 self-describing frames: 3,840 PRESENT and 960 ABSENT payloads;
- an exactly zero fp64 five-field surface input on the 148x180 grid;
- eight finite T/U/V/W month files with 86 floating variables;
- two finite fp64 step-240 ocean restart shards and no ice product;
- resolved exact-zero flux forcing, inactive restoring/freshwater budget, and
  uniform `nn_havtb=0`; and
- a complete 542-regular-file SHA-256 inventory.

The admission SHA-256 is
`591bce28aea5626ea009c674271061ae5f0d4c7959df2ef879bd9b36134fb4b0`;
the evidence ledger SHA-256 is
`e14145f86d1224fa1a18c03c0b9b64c13e585cee40db48751995aa78cb5f2f60`.
HD17-P3 is **CONFIRMED**.

## Compiled source and complete rung-2 delta

NEMO reads `nn_havtb` with the vertical-physics settings, starts the
background multiplier uniformly at one, enters the equatorial shaping branch
only for value one, and applies the multiplier to tracer diffusivity
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  Decision 83 therefore changes exactly this physical line in the
superseded rung-2 deck:

```diff
-   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)
+   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)
```

The parsed replacement delta is exactly
`namzdf.nn_havtb: [1, 0]`, physical line 419.  Exact deck SHA-256 is
`b4b8cb87...011e38`; execution-deck SHA-256 is `d691aec3...c0416`.

The compiled EIV initializer reads `ln_ldfeiv`, takes an explicit not-used arm
when false, and allocates its arrays only when true
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/ldftra.f90:572-614`).
Stage 3 invokes EIV and MLE transport under their two independent selectors
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/traadv.f90:253-262`).
The MLE initializer likewise prints its not-used arm when false and allocates
working arrays only when true
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/tramle.f90:643-684`).
With both adjacent rungs resolving `nn_havtb=0`, these remain every parsed
assignment differing from replacement rung 3 to replacement rung 2:

```text
namtra_eiv.ln_ldfeiv    .true. -> .false.
namtra_mle.ln_mle       .true. -> .false.
```

All retained EIV/MLE parameters, the exact-zero input, every other namelist
assignment, run protocol, build, inputs, CPP keys, recorder, and
`ln_spc_dyn=.true.` remain unchanged.  The compiled build has no `key_agrif`,
so that retained assignment remains inert.  HD17-P4 is **CONFIRMED**.

## Rung-2 acquisition

Search-before-build found and reused round 10's rung-2 deck construction,
compiled-source checks, version-2 frame parser, zero-input checker, recorder,
and admission logic, plus round 16's Decision-83 preservation pattern.  The
only extracted helper is the existing GM/MLE resolved-predicate block; no
second parser, record format, binary, or NEMO build was created.

The fail-closed launcher pins the admitted rung-3 replacement, the complete
value-1 rung-2 admission and inventory, both exact old decks, repaired
binary/writer/step program, GM/MLE compiled routines, CPP keys, inputs, new
manifest, and compiled background-shape branch.  Before running, it preserves
without deletion:

- `record` as `record_havtb1_superseded`;
- `deck` as `deck_havtb1_superseded`; and
- `rung2_admission.json` as
  `rung2_admission_havtb1_superseded.json`.

It reuses the unchanged instrumented build and two-rank, from-rest, 240-step
protocol; it never invokes a rebuild.  All six preflight plants fire and the
clean launcher prints `ORCA2_HIERARCHY_RUNG2_HAVTB0_PREFLIGHT_READY`.  No
value-0 rung-2 record existed before handoff.  The preflight JSON SHA-256 is
`a22b837a...a3f4bfd`.  HD17-P5 is **CONFIRMED**.

The Decision-83 replacement had already moved rung 3's old evidence to its
required superseded path.  Historical round 10 still followed the canonical
path and refused after the new admission.  Its source and admission pointers
now follow the preserved value-1 evidence, retaining every historical SHA
pin.  The complete hierarchy battery covers that repair.

## Gates, tests, review, and scope

- The repaired rung-3 admission passes every real plant and the clean gate.
- Rung 2's six preflight plants fire and its clean preflight passes.
- The complete hierarchy rounds 1-17 battery passes **401/401**.
- The receipt citation gate passes all five citations with zero failures and
  zero unmapped citations; its shifted `zdfphy` plant fires.  The cumulative
  default receipt gate passes 274 citations with zero unmapped citations.
- Ruff, Python compilation, shell syntax, and diff whitespace pass.
- The one permitted `tests/ocean/fidelity -n 12` run selected 2,578 tests and
  reached 97%, with six visible failure markers, before entering the known
  late-suite no-output tail and being interrupted.  It produced no terminal
  summary, so it is not claimed PASS and no failure is classified from that
  incomplete run.  It was not rerun.
- Separate `codex exec --sandbox read-only` review returned before reading the
  diff: `failed to initialize in-process app-server client: Read-only file
  system`.  Verdict: **independent review unavailable in-sandbox**.
- No file under `packages/` or `src/` changed.  GYRE is byte-identical by
  construction; its year gate was not rerun.  No NEMO source was modified.

ASKED choice: Decision 83 assigns `nn_havtb=0` to rungs 0 through 6.
UNASKED choices: none.  The controlled replacement changes one namelist
assignment and retains the run protocol byte-for-byte.

## Prediction ledger

| prediction | status |
|---|---|
| HD17-P1 failure ownership | **CONFIRMED**; superseded inherited background predicate only |
| HD17-P2 plant binding | **CONFIRMED**; surface and replacement-background plants fire |
| HD17-P3 rung-3 admission | **CONFIRMED**; complete independent record admitted |
| HD17-P4 rung-2 one-line delta | **CONFIRMED**; one replacement line, GM/MLE-only edge |
| HD17-P5 acquisition disposition | **CONFIRMED**; preflight-ready, value-0 record absent |

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round17_acquisition/run.sh --run`.
   Rung 2 remains UNMEASURED until every real plant fires and the clean gate
   admits its 480 frames, exact-zero input, month products, terminal restarts,
   resolved configuration, and inventory.
2. After rung 2 admits, repeat the one-line Decision-83 replacement for rung 1,
   preserving its value-1 record and its BBL/geothermal-only upper boundary.
3. Re-run the rung-1 versus main-rung-0 semantic diff after rung 1 is replaced
   and the main lane harmonizes its cited-inert assignments.

## UNVERIFIED

- NEMO has not executed the replacement rung-2 deck.
- Runtime `nn_havtb=0`, 480 frames, terminal restarts, month products, and the
  replacement rung-2 SHA inventory are unmeasured.
- The broad ocean-fidelity battery produced no terminal summary.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round17_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung2/record`.
