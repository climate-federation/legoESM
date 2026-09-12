# NEMO-testcases L2 GYRE round 70: FCT/LDF cancelling-pair receipt

Date: 2026-09-12. Final disposition: **HELD / PAIR PREDICTION REFUTED; no
production physics change landed**. Oracle producer
`3b3b045bd9e03b60330204e7590e4c4470b7a0ca`; 132 values admitted (43 of 63
inherited records exact and 20 changed).

## Verdict and first non-bit statement

The preregistered four-arm experiment **CONFIRMED** that the complete FCT
advection association and stage-3 LDF association are a large cancelling pair:
their paired arm reduces the kt3 T maximum from `1.627511417652272e-4 K` to
`5.760972143775689e-8 K`, and the S maximum from
`6.327755180279837e-6` to `6.410431296899333e-9`. It nevertheless
**REFUTED** the landing prediction. Only 83.36% of the 8,263 cells worsened by
LDF-only are removed for T, and only 67.57% of the 5,831 are removed for S;
the frozen requirement was at least 99% for each tracer. The pair leaves 1,375
of those T cells and 1,891 S cells worsened, and itself worsens 3,175 T cells
and 8,265 S cells beyond two row-scale float64 ULPs. The first non-bit boundary
therefore remains the kt3-before T/S result of the kt2 source association. No
trajectory replay or production transcription is justified by this failed
local gate.

This result follows the compiled execution path rather than an inferred
formula. GYRE clears and accumulates stage `Krhs` through advection and surface
forcing before the stage-3 QSR/LDF/ZDF sequence
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:827-950`,
`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/stprk3_stg.f90:917-965`). The
active FCT first-step and second-step branches construct corrected face fluxes
and the second call writes the complete corrected divergence to `pt_rhs`
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/traadv_fct.f90:503-510`,
`:532-540`, `:570-580`, `:602-611`). ZDF subsequently forms the Kbb tracer plus
Kmm-weighted `Krhs` content before its forward sweep
(`GYRE_OMIP_L2_P3_SM_R64KRHS/BLD/ppsrc/nemo/trazdf.f90:547-565`). The frozen
FCT substitution is therefore causal evidence about that compiled writer. It
is not an eligible implementation: its inputs remain unequal, so copying an
oracle output would violate the independent-model goal.

## Frozen causal measurement

The clean report `round70_pair_before.json` is stamped to instrumentation
commit `1758fb9e6bb7b7be6f4fbd8e772b3c3efc9b8b0f`, status **REFUTED**, and has
SHA-256 `c857326b9d4909631d0b4f81be18e37d8bf186d3039762d61d00b6b5ba497ddb`.
All live content, geometry, and admitted record arrays are float64. The ordinary
and default-false executions are bit-identical; the replayed same-step LDF
sources and complete FCT content targets are bit-identical to their admitted
records. Thus the failed pair verdict is not attributable to a moving seed,
host reconstruction, dtype change, or inactive control.

| tracer and arm | content max / RMS versus NEMO | kt3 max / RMS versus NEMO |
|---|---:|---:|
| T untouched | `1.6793926916705004e-3` / `8.709762996938103e-5` | `1.627511417652272e-4` / `4.226559754990046e-6` |
| T LDF-only | `5.954039670541533e-5` / `4.3615109611329324e-6` | `8.916073106490785e-7` / `8.624264889977338e-8` |
| T FCT-only | `1.6784480710612115e-3` / `8.698821369391028e-5` | `1.626533166394495e-4` / `4.22577547257504e-6` |
| T paired | `5.885129041871551e-7` / `2.583257607032639e-8` | `5.760972143775689e-8` / `2.1754691272928742e-9` |
| S untouched | `6.967976673877274e-5` / `6.5443173757255e-6` | `6.327755180279837e-6` / `2.388442668683053e-7` |
| S LDF-only | `7.651457053725608e-6` / `5.281106958882405e-7` | `7.235656340753849e-8` / `6.389237340825272e-9` |
| S FCT-only | `6.966940640040775e-5` / `6.523099428799779e-6` | `6.327236860670382e-6` / `2.3877876342898723e-7` |
| S paired | `2.0635297914850526e-8` / `9.709293941840095e-10` | `6.410431296899333e-9` / `9.926116083353661e-10` |

The pair clears the frozen absolute content ceilings and improves both kt3
tracers over untouched and LDF-only. Those aggregate gains cannot waive the
cellwise safeguard: maximum pair worsening is 2,596,477 row-scale ULPs for T
and 812,832 for S. This is a mechanically rejected cancellation, not a nearly
matched result.

Both preregistered controls fired and exited 1. The null-FCT plant made every
FCT target-exact predicate false and prevented the FCT-only content movement
(`plant_pair_null_fct.json`, SHA-256
`d7de120cd70a1b21a9c5d9f8c220158b92b4e9a260ef4a9f70556442f3e1fe57`).
The one-ULP content plant changed exactly one T cell by
`2.842170943040401e-14 K m` and failed the exact target predicate
(`plant_pair_content_ulp.json`, SHA-256
`a011e7aaf3ac69229c17c1965e51bcb2e69e50dbc864c8eb519ba2cde37dae7b`).
Inherited dry-cell division warnings are diagnostic only; all acceptance rows
are wet-cell masked.

## Rule 12 card

| card | changed statement | Rule-12 disposition |
|---|---|---|
| GYRE | none in production; private four-arm causal instrument only | **REFUTED before trajectory**: the pair fails the frozen cellwise kt3 condition, so kt1--10 and days 1--30 are **UNREACHED** |
| LOCK_EXCHANGE | none | exact before/after **UNREACHED** after the GYRE stop; shared production behavior is unchanged |
| OVERFLOW | none | exact before/after **UNREACHED** after the GYRE stop; shared production behavior is unchanged |
| DINO | none | execution gate **UNREACHED**; no pair is landed. The shared stage helper is callable from DINO, so any eventual association edit carries explicit per-row-cancellation risk and must be gated there rather than inferred from a band aggregate. |
| ORCA2 | none; native record absent | **UNMEASURED WITH SPEC**: resolve its compiled card; record post-SBC, post-QSR, and post-LDF stage-3 `Krhs`, complete FCT inputs/outputs, and pre/post-ZDF T/S for kt1--10; replay each statement independently and paired; require exact statement replay, every moved row registered, no AT-BAR loss, and no earlier first-over-bar boundary. |

The preregistered stopping rule was enforced. GYRE kt1--10, days 1--30, tanks,
DINO, and ORCA2 were not run after the local pair gate failed; none is reported
as passed. No configuration/default, carried state, stabilizer, NEMO
source/build/run, year harness, reconciliation gate, freshwater pair, #1484
guard, or held manifest changed. The only code change is a default-inert private
measurement hook and its focused tests; it cannot select an oracle record in a
normal model construction.

## Review and focused checks

The required separate review was invoked on clean instrumentation commit
`1758fb9e6bb7b7be6f4fbd8e772b3c3efc9b8b0f` with
`codex exec --sandbox read-only`. It exited 1 before reviewing. Its terminal
result, verbatim, was **“Error: failed to initialize in-process app-server
client: Read-only file system (os error 30)”**. There is no verdict to quote;
the review requirement is **UNMET/BLOCKED**, and the absence of a verdict is
not approval. The full log is `codex_round70_review.log`, SHA-256
`eae080369e91b8869ecdd955b8e2a9840b501bc2c8dfb0889bae645cc549d4b5`.
No production diff is being landed, so this blocked review cannot silently
approve one.

The seven focused round-70/helper tests pass (`focused_tests.xml`, SHA-256
`3d5be6ce04ec3e9297cc5e6545afae99cf767740456e39670724f62f5d5168e0`).
Python compilation and `git diff --check` pass.

The citation gate is stamped to the clean receipt-bearing commit
`77d7f1e2a333e6526ce24c23c10b0d4fb428f6c2`. It checks all seven cited
compiled-source ranges, audits the full map with no stale entry, and passes
(`round70_citation_gate.json`, SHA-256
`43a126a4b70462ca0bd3cad716e534f9feab3597ecd394c41cf9eb023835182c`).
Shifting the complete FCT writer citation by two lines produces
SYMBOL-NOT-AT-LINE and exits 1 (`round70_citation_plant.json`, SHA-256
`ccbf428cb61af1528faa5db03a3b458382f5c30d8c2d33035de9c9ab857cf59b`).

## ASKED / UNASKED and OPEN

| state | item | disposition |
|---|---|---|
| ASKED | configuration choice | none encountered |
| UNASKED | configuration, carried state, stabilizer, NEMO, or harness change | none performed |

OPEN for round 71: do not land or retry the complete-FCT/LDF output pair. The
first kt3-before T/S cellwise worsening remains the owner. Preregister a
source-order split of the retained 1,375 T and 1,891 S LDF-worsened cells and
the pair's newly worsened cells, using the admitted round-46 kt2 stage and
round-64 `Krhs` records. Walk the complete FCT writer's unequal inputs in
compiled argument order (Kmm T/S, then transports/metrics), and stop at the
first input statement whose native replay is non-bit. Rank candidates by their
effect on the paired kt3 T maximum first, preserve the exact same-step LDF arm,
and require the same cellwise two-ULP Rule-12 predicate before any trajectory
run. If the existing records do not expose that first unequal operand, write a
new acquisition card; do not infer it or substitute an oracle output.
