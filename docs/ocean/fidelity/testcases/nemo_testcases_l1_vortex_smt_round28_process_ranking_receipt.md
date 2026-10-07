# Receipt — VORTEX_SMT round 28 (lane round 240): day-100 process ranking

**Status: HELD.** The requested magnitude ranking found no positive-removal
family and therefore names no owner statement. HPG source order is inert at
the run-to-run floor; disabling vertical mixing/EVD, bottom drag, tracer LDF,
or momentum LDF makes the day-100 temperature gap worse; disabling the
barotropic replacement makes the run hit the raw-mesh positive-thickness
guard. Production physics, cards, and accepted trajectories are unchanged.

Base: `ca822f33d` (round 239). Finite-arm measurement commit:
`19a38791ee878118c4952ce4d0cb922fc58130c3`. Final report-emitter commit:
`d42ea6ed7d1f1fd7f8cbf96266e7f2773ae6af52`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/round240/`.
Preregistration:
`docs/ocean/fidelity/PREREG_nemo_testcases_l1_vortex_smt_round28_process_ranking.md`.

## 1. Instrument and compiled execution order

This is an intervention-leverage ranking, not an attribution by itself. Each
arm changes one complete legoESM family while retaining the same admitted
SMT-4 NEMO trajectory. Positive `baseline T_rms - arm T_rms` would have
licensed a statement walk; a negative number means the removed process is
protective in this intervention and cannot be called the source owner.

The compiled SMT-4 stage program calls HPG, VOR, and vector advection in that
order at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:328-344`,
then calls lateral momentum diffusion at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:398-404`
before implicit momentum ZDF at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:415-417`.
The tracer stage calls lateral diffusion before implicit vertical diffusion at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stprk3_stg.f90:539-554`.
The step-entry program forms bottom drag at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:193-200` and then
calls the split-explicit barotropic solve at
`VORTEX_SMT4_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/stp2d.f90:260-267`.
These citations establish which compiled processes execute and their order;
they do not turn an ablation response into source ownership.

The probe reuses the round-210 production-step scorer. It records the exact
resolved diff for each arm and refuses an extra field. The registered changes
are:

| arm | only changed field(s) |
|---|---|
| HPG source order | private hook `nemo_stage_rhs_accumulation_order_arm: False -> True` |
| momentum LDF off | `lateral_viscosity.A_h: 1500 -> 0` |
| tracer LDF off | resolved `gm_redi` subtree -> `None` |
| bottom drag off | `bottom_drag.bottom_drag_cd0: 0.001 -> 0` |
| vertical mixing/EVD off | `A_v: 1.2e-4 -> 0`, `K_v: 1.2e-5 -> 0`, convection scheme `enhanced_diffusion -> none` |
| barotropic replacement off | private hook `stage_barotropic_correction: True -> False` |

The original vertical arm tried `K_conv=K_bg=0` with NEMO replacement
composition. Construction refused because that makes fired-set readback
ambiguous. The preregistration records the refusal and the committed amendment
to use the existing family selector; a regression proves the rejected spelling
still fails closed. No result from the invalid arm is used.

## 2. Calibration and complete ranking

The certified kt=1..10 sanity check reproduces the round-238 SMT-4 registry:
first-over-bar remains kt=2 for T/u/v/ssh, with no mismatch. The fresh baseline
reproduces the exact score-artifact float
`2.5527080520554426e-04 K` at day 100. P1 is **CONFIRMED**.

The ordered day-100 result is:

| rank | arm | day-100 T RMS K | removed T RMS K | fraction removed | verdict |
|---:|---|---:|---:|---:|---|
| 1 | HPG source order | `2.5527080520609173e-04` | `-5.474678869965555e-16` | `-2.1446553065703453e-12` | floor-inert |
| 2 | vertical mixing/EVD off | `3.9879169366947582e-04` | `-1.4352088846393156e-04` | `-0.5622299359629802` | protective |
| 3 | bottom drag off | `2.3568489340129073e-03` | `-2.101578128807363e-03` | `-8.232739843144893` | protective |
| 4 | tracer LDF off | `8.8288189834348080e-03` | `-8.573548178229263e-03` | `-33.586089765830586` | protective |
| 5 | momentum LDF off | `4.7214037086395377e-02` | `-4.6958766281189834e-02` | `-183.95666611142076` | protective |
| 6 | barotropic replacement off | `UNBOUNDED` | n/a | n/a | positive-thickness guard |

The HPG movement is `2.74e-06` of the `2e-10-K` run-to-run floor, confirming
P2. The preregistered tracer-LDF winner prediction P3 is **REFUTED**: removing
that family makes the gap 34.6 times the baseline rather than removing half.
P4 is also **REFUTED** because the barotropic arm does not remain finite. Its
production-JIT error is retained in `process_family_ranking.json`:
`raw-mesh e3w_int must contain only finite values > 0`.

All selected T checkpoints are registered below. Values are RMS K against the
same admitted NEMO trajectory.

| arm | day 1 | day 2 | day 5 | day 10 | day 20 | day 30 | day 60 | day 100 |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| baseline | `1.405694e-07` | `5.100120e-07` | `4.424520e-06` | `2.454181e-05` | `7.226803e-05` | `1.450215e-04` | `2.003483e-04` | `2.552708e-04` |
| HPG source order | `1.405694e-07` | `5.100120e-07` | `4.424520e-06` | `2.454181e-05` | `7.226803e-05` | `1.450215e-04` | `2.003483e-04` | `2.552708e-04` |
| momentum LDF off | `1.009715e-03` | `2.079377e-03` | `4.829621e-03` | `8.851109e-03` | `1.744293e-02` | `2.618260e-02` | `4.626986e-02` | `4.721404e-02` |
| tracer LDF off | `1.913845e-04` | `3.759166e-04` | `9.002928e-04` | `1.683142e-03` | `2.997260e-03` | `4.094815e-03` | `6.633283e-03` | `8.828819e-03` |
| bottom drag off | `4.156512e-05` | `5.537799e-05` | `1.183936e-04` | `3.182103e-04` | `1.033447e-03` | `1.758414e-03` | `2.581657e-03` | `2.356849e-03` |
| vertical mixing/EVD off | `3.292654e-06` | `6.789644e-06` | `1.776394e-05` | `4.234220e-05` | `1.019459e-04` | `1.829012e-04` | `2.900223e-04` | `3.987917e-04` |
| barotropic replacement off | `UNBOUNDED before day 1` | — | — | — | — | — | — | — |

The finite rows were produced together from a clean tree at
`19a38791ee878118c4952ce4d0cb922fc58130c3`, then reused only through an
explicit SHA option after reporting-only corrections. The final JSON records
that SHA and a SHA-256 digest for each arm's eight snapshots. The barotropic
arm was rerun on the final emitter and reproduced the guard. This prevents a
mixed or silently stale arm from entering the ranking.

## 3. Statement verdict, scope, and controls

There is no positive-removal winner. Calling HPG the owner because it sorts
first would invert the measurement: it changes day 100 by only
`5.47e-16 K`, and round 239 already proved that source-order statement locally
exact but trajectory-inert. The four negative arms show that wholesale family
removal destroys compensating/stabilising physics; they do not exonerate every
statement inside those families. The unbounded barotropic arm is likewise not
a statement attribution. Under frozen P5, **no first non-bit statement is
named and no operand acquisition is requested**. In particular, the result
does not license the conditional dynldf acquisition.

Both controls are non-vacuous. Removing `bottom_drag_off` from the registry
prints `STATUS PLANT-FIRED` and exits 1. Adding `1e-8 K` to the HPG endpoint
crosses the floor control, prints `STATUS PLANT-FIRED`, and exits 1. The final
combined round-239, round-240, and citation-gate suite reports `24 passed in
13.68s`.

Only validation scripts, tests, preregistration, citation mappings, and this
receipt change. No file under `packages/` or `src/` changes, no held physics
patch is applied, and all production cards retain the round-239 tree. Thus the
certified GYRE year remains the round-237/223 pin, DINO/tanks/control registries
are out of execution scope, and no unchanged number is re-certified or moved.
The ORCA2 pointer is deliberately negative: SMT-4 shares these compiled process
families, but this round names no statement for ORCA2 to fold in.

## 4. Independent review and citation gates

The required separate Codex review was attempted against the committed diff
from `ca822f33d` through the receipt commit. It exited before inspecting the
diff, so no SHIP verdict is inferred. Its output is quoted verbatim:

> `WARNING: proceeding, even though we could not create PATH aliases: Read-only file system (os error 30)`
> `Reading additional input from stdin...`
> `Error: failed to initialize in-process app-server client: Read-only file system (os error 30)`

**Independent review unavailable in-sandbox.** The HELD disposition follows
from the mechanical no-winner ranking, not from an inferred review verdict.

The first citation run correctly refused two ambiguous/shifted anchors and one
combined unmapped range. The anchors were corrected by occurrence and symbol,
and the momentum-LDF and ZDF spans were split; none was weakened or deleted.
The final round citation gate reports **PASS: 6 citations, 0 failures, 0
unmapped**. The cumulative default receipt reports **PASS: 274 citations, 0
failures, 0 unmapped**. Shifting the drag-loop range by two lines reports
`SYMBOL-NOT-AT-LINE` and exits 1.

The final CPU-only command covered the round-239 control, all six round-240
unit controls, and the complete citation-gate module:

> `24 passed in 13.68s`

No full physics tree or DINO month integration was run because the committed
diff contains no model-code (`packages/` or `src/`) change. The measurement
itself ran the production JIT path on CPU with fp64/libm and the certified
SMT-4 kt=1..10 calibration.

### OPEN — next round

The requested off-family ranking is closed and has no source owner. Do not
infer one from the least-negative row, and do not acquire dynldf operands from
this table. Proceed to the deferred one-rung-per-round 100-day comparison and
movie program with SMT-1 first, registering every checkpoint against its
admitted NEMO run. If a later magnitude walk returns to SMT-4, it must use a
source-faithful one-family substitution rather than another destructive
family-off ablation.

**DECISION_NEEDED: NONE. ACQUISITION_NEEDED: NONE.**
