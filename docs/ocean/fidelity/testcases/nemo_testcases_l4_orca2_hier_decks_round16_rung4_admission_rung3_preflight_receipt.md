# ORCA2-DECKS round 16 receipt — rung 4 admitted, rung 3 staged

Date: 2026-10-02

Disposition: **STOPPED_FOR_RECORD**.  The completed Decision-83 rung-4 run is
admitted after repairing a checker-only inherited-result lookup.  Rung 3 is
sealed as the same one-line `nn_havtb=1 -> 0` replacement, and its complete
boundary from replacement rung 4 remains the previously admitted unforced
surface module.  Rung 3 is UNMEASURED pending the operator.

Base: `5696bf28d`.  Preregistration: commit `5f2103573`, before repairing the
checker, rerunning admission, or measuring the rung-3 deck.  Checker repair:
`0a1f4982f`.  Rung-3 tooling: `b46cabb74`.  Historical evidence-routing
repair: `bf938147f`.  Every run claim is labelled **independent**: NEMO starts
from that rung's own from-rest initialization.

## Rung-4 admission repair and result

The operator run itself reached `STOP 0`.  Its clean admission then refused
even though all three rung-4 shortwave predicates were true: the round-15
checker asked the rung-5 result's top level for `nn_havtb_uniform`, while the
round-14 result stores that predicate in its nested rung-6 result.  The direct
unit fixture had repeated the wrong flat shape and therefore concealed the
defect.

The repair consumes the actual nested result and routes the existing
`resolved-havtb` plant through the same chain.  It changes no record
predicate, threshold, deck, or scientific data.  The operator record was
admitted in place with `--admit-existing`; NEMO was not rerun.  Every real
plant fired before the clean result `PASS_RUNG4_HAVTB0_RECORD`:

- 480 self-describing frames: 3,840 PRESENT and 960 ABSENT payloads;
- eight finite T/U/V/W month files with 92 floating variables;
- two finite fp64 step-240 ocean restart shards and no ice product;
- resolved `ln_traqsr=F`, no shortwave initializer, and uniform
  `nn_havtb=0`; and
- a complete 541-regular-file SHA-256 inventory.

The admission SHA-256 is `b3371422...808089`; the round-15 evidence ledger is
`54a3cf58...bb7b6`.  HD16-P1, HD16-P2, and HD16-P3 are **CONFIRMED**.

## Compiled source and complete rung-3 delta

NEMO reads `nn_havtb` with the vertical-physics settings, starts the
background multiplier uniformly at one, enters the equatorial shaping branch
only for selector value one, then applies that multiplier to tracer
diffusivity
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  Decision 83 therefore changes exactly this physical line in the
superseded rung-3 deck:

```diff
-   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)
+   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)
```

The parsed replacement delta is exactly
`namzdf.nn_havtb: [1, 0]`, physical line 419.  Exact deck SHA-256 is
`0362b8cb...36d74`; execution-deck SHA-256 is `20866bcb...7469`.

NEMO selects exactly one surface-forcing arm, initializes the flux-file arm,
and calls restoring and freshwater-budget logic only under their switches
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/sbcmod.f90:359-374`,
`:517-521`).  With both adjacent rungs resolving `nn_havtb=0`, these are
**every** parsed assignment differing from replacement rung 4 to replacement
rung 3:

```text
namsbc.ln_abl              ABSENT -> false
namsbc.ln_blk                true -> false
namsbc.ln_cpl              ABSENT -> false
namsbc.ln_dm2dc            ABSENT -> false
namsbc.ln_flx              ABSENT -> true
namsbc.ln_mixcpl           ABSENT -> false
namsbc.ln_ssr                true -> false
namsbc.ln_usr              ABSENT -> false
namsbc.nn_fwb                   2 -> 0
namsbc_flx.cn_dir          ABSENT -> './'
namsbc_flx.sn_emp          ABSENT -> rung3_zero_flux:emp
namsbc_flx.sn_qsr          ABSENT -> rung3_zero_flux:qsr
namsbc_flx.sn_qtot         ABSENT -> rung3_zero_flux:qtot
namsbc_flx.sn_utau         ABSENT -> rung3_zero_flux:utau
namsbc_flx.sn_vtau         ABSENT -> rung3_zero_flux:vtau
namsbc_ssr.ln_sssr_bnd       true -> false
```

The gate compares that complete parsed map with the SHA-pinned value-1 rung-3
admission's `deck_delta_from_rung4`; they are identical.  The exact-zero flux
file, all other namelist assignments, run protocol, build, inputs, CPP keys,
recorder, and `ln_spc_dyn` remain unchanged.  HD16-P4 is **CONFIRMED**.

## Rung-3 acquisition

Search-before-build found and reused round 9's exact-zero surface deck,
version-2 frame parser, generated-input checker, recorder, and compiled-source
checks, plus round 15's Decision-83 preservation pattern.  No second record
format, parser, binary, or NEMO build was created.

The fail-closed launcher pins the admitted rung-4 replacement and the complete
value-1 rung-3 admission, inventory, exact and execution decks, repaired
binary/writer/step program, compiled surface routines, new manifest, and
Decision-83 branch.  Before running, it preserves without deletion:

- `record` as `record_havtb1_superseded`;
- `deck` as `deck_havtb1_superseded`; and
- `rung3_admission.json` as `rung3_admission_havtb1_superseded.json`.

It reuses the unchanged instrumented build and two-rank, from-rest, 240-step
protocol; it never invokes `makenemo`.  All six preflight plants fire and the
clean launcher prints `ORCA2_HIERARCHY_RUNG3_HAVTB0_PREFLIGHT_READY`.
No value-0 rung-3 record existed before handoff.  HD16-P5 is **CONFIRMED**.

The Decision-83 replacement also moved rung 4's old evidence to its required
superseded path.  Historical rounds 9 through 12 still followed the canonical
path and their tests refused.  Their shared rung-3 gate now follows the
preserved value-1 rung-4 evidence, as its pinned hashes require.  The first
complete hierarchy battery exposed this; a focused rerun passes 41/41 and the
complete rerun passes 391/391.

## Gates, tests, review, and scope

- The repaired rung-4 admission passes every real plant and the clean gate.
- Rung 3's six preflight plants fire and its clean preflight passes.
- The complete hierarchy rounds 1-16 battery passes **391/391**.
- The receipt citation gate passes all four citations with zero failures and
  zero unmapped citations; its shifted `zdfphy` plant fires.  The cumulative
  default receipt gate passes 274 citations with zero unmapped citations.
- The one allowed `tests/ocean/fidelity -n 12` run selected 2,568 tests,
  reached 99% with no visible failure marker, then entered the known late-suite
  no-output tail and was interrupted.  It produced no terminal summary, so it
  is not claimed PASS and was not rerun.
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
| HD16-P1 failure ownership | **CONFIRMED**; checker-only nested lookup, existing run admitted |
| HD16-P2 plant binding | **CONFIRMED**; clean nested result passes and `resolved-havtb` fires |
| HD16-P3 rung-4 admission | **CONFIRMED**; complete independent record admitted |
| HD16-P4 rung-3 one-line delta | **CONFIRMED**; one replacement line, unchanged surface-module boundary |
| HD16-P5 acquisition disposition | **CONFIRMED**; preflight-ready, value-0 record absent |

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round16_acquisition/run.sh --run`.
   Rung 3 remains UNMEASURED until every record plant fires and the clean gate
   admits its 480 frames, month products, terminal restarts, exact-zero input,
   resolved configuration, and inventory.
2. After rung 3 admits, repeat the one-line Decision-83 replacement for rungs
   2 and 1 in order, preserving each value-1 record.
3. Re-run the rung-1 versus main-rung-0 semantic diff after rung 1 is replaced
   and the main lane harmonizes its cited-inert assignments.

## UNVERIFIED

- NEMO has not executed the replacement rung-3 deck.
- Runtime `nn_havtb=0`, 480 frames, terminal restarts, month products, and the
  replacement rung-3 SHA inventory are unmeasured.
- The broad ocean-fidelity battery produced no terminal summary.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round16_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung3/record`.
