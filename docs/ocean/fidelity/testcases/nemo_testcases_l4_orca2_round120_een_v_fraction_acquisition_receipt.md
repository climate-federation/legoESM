# ORCA2 round 120 — northern-V EEN fraction acquisition

Date: 2026-10-03. Base `a83abe5510481c8855017b86489d015c38563fd6`.
Every eventual ocean number from this hierarchy-rung-0 record is
**independent** because rung 0 starts from NEMO's own from-rest state. No model
physics, card field, configuration value, carried state, stabilizer, sea-ice
selector, or `unmeasured_features` entry changed.

## Verdict

**STOPPED_FOR_RECORD.** Round 119 names the first northwest and northeast V
boundary at the completed `zpvo`: 1,431 magnitude-unequal cells on the
northern-fold row for each path. The admitted round-118 stream does not carry
the three component fractions, so this round does not infer which fraction or
fold operand owns the difference.

A committed, additions-only, rank-complete acquisition now records both paths'
three fractions, all four denominator operands, partial and final sums, and the
original NEMO sums. The exact-source syntax proof and all five record-independent
launcher controls pass. The new record is absent, so R120-P1 through R120-P4
remain unmeasured and no ocean number is inferred from preflight.

The operator action is:

```text
/tmp/autopilot-orca2-1878126919/scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_round120_een_v_fraction_acquisition/run.sh --run
```

It creates target `ORCA2_OMIP_L4_R120EENVFRAC` and run directory
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round120/acquisition/orca2_rung0_een_v_fraction_ranked_10step_np2`.

## Compiled source boundary

NEMO evaluates northeast then northwest `zpvo`; each is an ordered sum of
three `ff_f / (e3f_0vor * (1 + r3f * fe3mask))` fractions
(`ORCA2_OMIP_L4_R118EENVREC/BLD/ppsrc/nemo/dynspg_ts.f90:1320-1325`).
The additions-only patch places one recorder call after both original
assignments and leaves every original source line untouched. The writer
reconstructs each denominator, quotient, partial sum, and final sum and stores
the original NEMO sums separately, so admission can distinguish an
instrument arithmetic error from a model difference.

## Admission contract and controls

The self-described record carries 43 named fields per rank: six operand or
arithmetic fields for each of three fractions on each of two paths, three
ordered-sum fields per path, and `mbkv`. The checker derives names, ranks,
dimensions, and payload lengths from the header. It requires exactly-once
coverage of the `148 x 180` domain, rejects the dummy level and all writes
outside the literal loop, replays every arithmetic statement bitwise, and
compares the stored NEMO sums to the admitted round-118 `zpvo_ne/nw` fields.

Observational passivity requires all twenty kt=1..10 restart shards and all ten
inherited round-105/107/110/116/118 record shards to be byte-identical to the
admitted round-118 run. The launcher pins source, CPP keys, binary, namelist,
deck/input manifests, patch, writer, checker, launcher, and preregistration by
content rather than a moving commit SHA. It uses a new target, new run
directory, and absolute pre-created per-rank output path.

Header, field-name, field-dimension, truncation, missing-field, duplicate-rank,
bottom-index, denominator, quotient, ordered-sum, inherited-final,
inherited-stream, and restart-byte controls are committed but require the
record. The record-independent layout, absolute-path, six-recorder environment,
producer-content, and rank-log controls each exit 69 with
`STATUS PLANT-FIRED`. Clean committed preflight ends:

```text
SYNTAX_PROOF_PASS l4_r120_een_v_fraction.f90 dynspg_ts.f90
ORCA2_ROUND120_EEN_V_FRACTION_PREFLIGHT_READY /data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round120/acquisition/orca2_rung0_een_v_fraction_ranked_10step_np2
```

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R120-P1: recorder is observationally passive | **UNMEASURED**, operator record absent |
| R120-P2: stream is rank-complete and arithmetically sufficient | **UNMEASURED**, operator record absent |
| R120-P3: NW/NE first differ in one of three fold-row fractions | **UNMEASURED**, no component number inferred |
| R120-P4: no rank seam owns the difference | **UNMEASURED**, no component support inferred |
| R120-P5: no production landing from acquisition alone | **CONFIRMED**, no `packages/` file changed |

## Validation and review

The exact patched source and writer compile under the pinned rung-0 include
set. The focused round-110/118/119/120 plus citation battery passes 33/33.
The default citation gate passes 274 citations and this receipt passes its one
citation with zero unmapped or failed entries; shifting the compiled range by
two lines makes the gate refuse.

The single required `tests/ocean/fidelity -n 12` invocation collected 2,363
tests and reached 98%, then reproduced the registered xdist-controller stall
after the real-pytest process census reached zero. It was interrupted without
a summary and is **incomplete, not PASS**. Re-running the six registered IDs
in isolation reproduces the same six pre-existing reds: SI3 scalar-math source
provenance, the round-129 certified-year stamp, round-51 private-arm scope,
round-35 dirty-escape scoping, worktree stamping, and the
`hires_lane_surface` case-board row. No round-120 failure was exposed.

The separate read-only Codex review returned **independent review unavailable
in-sandbox**: `failed to initialize in-process app-server client: Read-only
file system`.

No model or card file changed, so ORCA2 rung-0/rung-7, GYRE, DINO, tank, and
generic-card trajectories cannot move and are not represented as rerun gates.

## OPEN

1. Operator runs the committed launcher. Admission must prove restart and
   inherited-stream identity, exact two-rank coverage, arithmetic replay,
   admitted-final identity, and every runtime plant before a fraction is
   quoted.
2. Walk NE and NW independently through their three fractions in compiled
   order, then the first unequal fraction's `ff_f`, `e3f_0vor`, `r3f`, and
   `fe3mask` operands. Preserve failed predictions as `REFUTED`.
3. Walk the northern accumulator/final-scale cancelling pair only after the
   fraction association is exact. Resume the later 68-cell substep-2 U
   residual only after coefficient construction closes.
4. The package-exposed rung-0 card and independent 240-step month remain open
   hierarchy work.

## UNVERIFIED

- Runtime observational passivity and arithmetic sufficiency until the
  operator completes acquisition.
- Which NE/NW fraction and operand owns the 1,431 fold-row magnitudes.
- Whether the complete northern accumulator/final-scale pair passes every
  trajectory landing gate.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
