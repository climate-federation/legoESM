# ORCA2-DECKS round 18 receipt — rung 2 admitted, rung 1 staged

Date: 2026-10-02

Disposition: **STOPPED_FOR_RECORD**.  Decision 83's replacement rung-2 run is
admitted.  Rung 1 is sealed as the same one-line `nn_havtb=1 -> 0`
replacement, its upper boundary remains exactly the BBL/geothermal module,
and the main lane's harmonized rung-0 deck leaves only damping plus protocol
differences.  Rung 1 is UNMEASURED pending the operator.

Base: `05e69b3cb`.  Preregistration: commit `91f531b6a`, before rerunning
rung-2 admission, reading its payload results, or measuring the rung-1
replacement.  Tooling: commit `217af36ed`.  Every run claim is labelled
**independent**: NEMO starts from that rung's own from-rest initialization.

## Rung-2 admission

The operator-created record predates this round and was not rerun.  The
committed round-17 launcher ran in admit-only mode: all 21 admission plants
printed `STATUS PLANT-FIRED`, clean admission printed
`PASS_RUNG2_HAVTB0_RECORD`, and the launcher printed
`ORCA2_HIERARCHY_RUNG2_HAVTB0_ACQUISITION_PASS`.

The admitted independent record contains:

- 480 self-describing frames, 3,840 finite PRESENT payloads and 960 ABSENT
  payloads, exactly the owner-off runoff fields;
- one exactly zero fp64 five-field surface input on the 148x180 grid;
- eight finite T/U/V/W month files with 86 floating variables;
- two finite fp64 step-240 ocean restart shards and no ice product; and
- a complete 545-regular-file SHA-256 inventory.

The resolved output proves GM eddy-induced velocity and mixed-layer eddies
are both off, every active print is absent, the background is uniform, and
all inherited rung-3 through rung-10 predicates pass.  The admission SHA-256
is `90ba1895...5dd04f`; the evidence ledger is `85494d88...90a3ce`.
HD18-P1 and HD18-P2 are **CONFIRMED**.

## Compiled source and complete rung-1 delta

NEMO reads `nn_havtb` with vertical-physics settings, initializes the
background multiplier uniformly, enters the equatorial shaping branch only
for selector value one, and applies the multiplier to tracer diffusivity
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/zdfphy.f90:140-149`,
`:205-228`).  Decision 83 therefore changes exactly physical line 419 in the
superseded rung-1 deck:

```diff
-   nn_havtb    =    1         !  horizontal shape for avtb (=1) or not (=0)
+   nn_havtb    =    0         !  horizontal shape for avtb (=1) or not (=0)
```

The parsed replacement delta is exactly
`namzdf.nn_havtb: [1, 0]`.  Exact deck SHA-256 is
`6849907f...07d961`; execution-deck SHA-256 is
`f74f089c...62342b`.  HD18-P3 is **CONFIRMED**.

NEMO always calls the BBL and geothermal initializers from `nemo_init`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/nemogcm.f90:428-434`).
The BBL initializer returns before allocation when `ln_trabbl=false`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trabbl.f90:540-564`).
The geothermal initializer allocates and reads its heat-flow field only when
`ln_trabbc=true`
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/trabbc.f90:200-251`).
The possible BBL coefficient calls and stage-3 BBL/geothermal tracer calls are
guarded by those selectors
(`ORCA2_ORCA1ICE_OMIP_L4_R69SURFACE/BLD/ppsrc/nemo/stprk3_stg.f90:430-458`,
`:523-529`).

With both adjacent rungs resolving `nn_havtb=0`, these are every parsed
assignment differing from replacement rung 2 to replacement rung 1:

```text
nambbc.ln_trabbc    .true. -> .false.
nambbl.ln_trabbl    .true. -> .false.
```

All retained BBL/geothermal parameters, exact-zero input, other namelist
assignments, run protocol, build, inputs, CPP keys, recorder, and
`ln_spc_dyn=.true.` remain unchanged.  HD18-P4 is **CONFIRMED**.

## Main rung-0 boundary

The main lane's Decision-83 harmonized rung-0 deck is pinned at round 96,
SHA-256 `51923558...1d360e8`.  The earlier active background difference and
all nine inert declaration differences are gone.  The complete semantic diff
from replacement rung 1 has exactly 11 rows:

| class | rows | assignments |
|---|---:|---|
| intended damping boundary | 2 | `ln_tradmp`, `ln_tsd_dmp` |
| restart protocol | 4 | `ln_rst_list`, `nn_itend`, `nn_stock`, `nn_stocklist` |
| exact-zero file name | 5 | `sn_emp`, `sn_qsr`, `sn_qtot`, `sn_utau`, `sn_vtau` |

For each file row, replacing only `rung3_zero_flux` by `rung0_zero_flux`
makes the assignments identical.  The gate refuses any additional row, any
changed damping value, or any other difference within a zero-file operand.
The hierarchy's rung-1 to rung-0 physical boundary is therefore exactly the
interior T/S damping module specified by Decision 80.

## Rung-1 acquisition

Search-before-build found and reused round 11's rung-1 construction,
compiled-source checks, version-2 frame parser, exact-zero input checker, and
BBL/geothermal predicate block, plus round 17's Decision-83 preservation and
admission pattern.  No second parser, record format, binary, or NEMO build was
created.

The fail-closed launcher pins the admitted replacement rung 2, complete
value-1 rung-1 admission and inventory, both old decks, repaired binary,
writer and step program, BBL/geothermal and background compiled sources,
harmonized main-rung-0 deck, CPP keys, inputs, and new manifest.  Before
running it preserves without deletion:

- `record` as `record_havtb1_superseded`;
- `deck` as `deck_havtb1_superseded`; and
- `rung1_admission.json` as
  `rung1_admission_havtb1_superseded.json`.

It reuses the unchanged instrumented build and two-rank, from-rest, 240-step
protocol; it never invokes a rebuild.  All seven preflight plants fire and the
clean launcher prints `ORCA2_HIERARCHY_RUNG1_HAVTB0_PREFLIGHT_READY`.  No
value-0 rung-1 record existed at handoff.  HD18-P5 is **CONFIRMED**.

The historical round-11 gate now follows the preserved value-1 rung-2
evidence and exposes its BBL/geothermal checks separately for the replacement
chain.  The central CLI regression covers every named plant in hierarchy
rounds 1 through 18 and requires `STATUS PLANT-FIRED` from each.

## Gates, tests, review, and scope

- Rung 2 passes all 21 real plants and clean admission.
- Rung 1's seven preflight plants fire and clean preflight passes.
- The complete hierarchy rounds 1 through 18 battery passes **492/492**.
- Ruff, Python compilation, shell syntax, and diff whitespace pass.
- The one permitted `tests/ocean/fidelity -n 12` run selected 2,669 tests and
  reached 99%.  Its only visible failure was the declared pre-existing SI3
  scalar-math provenance gate; the known late-suite slow tail produced no
  terminal summary and was interrupted.  No second broad battery ran.
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
| HD18-P1 rung-2 admission | **CONFIRMED**; complete independent record admitted |
| HD18-P2 inherited routing | **CONFIRMED**; replacement upper chain and both owning plants fire |
| HD18-P3 rung-1 one-line replacement | **CONFIRMED**; one physical and parsed assignment |
| HD18-P4 rung-1 module boundary | **CONFIRMED**; BBL/geothermal-only upper edge |
| HD18-P5 acquisition disposition | **CONFIRMED**; preflight-ready, value-0 record absent |

## OPEN

1. The operator runs
   `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round18_acquisition/run.sh --run`.
   Rung 1 remains UNMEASURED until every record plant fires and the clean gate
   admits its 480 frames, exact-zero input, month products, terminal restarts,
   resolved configuration, and inventory.
2. After rung 1 admits, re-run the exact semantic diff against the pinned main
   rung-0 deck.  If it remains damping plus protocol only, the NEMO-side
   hierarchy is complete for rungs 1 through 10.
3. Rung-0 card construction and cross-model scoring remain main-lane work and
   out of this side lane's scope.

## UNVERIFIED

- NEMO has not executed the replacement rung-1 deck.
- Runtime `nn_havtb=0`, 480 frames, terminal restarts, month products, and the
  replacement rung-1 SHA inventory are unmeasured.
- The broad ocean-fidelity battery produced no terminal summary.

Acquisition launcher:
`scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_hier_decks_round18_acquisition/run.sh`.

Target record:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_hierarchy/rung1/record`.
