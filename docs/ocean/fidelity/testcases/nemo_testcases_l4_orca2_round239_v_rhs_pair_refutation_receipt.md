# ORCA2 round 239 — Ve_rhs pair refuted; HPG owns completed V-RHS debt

Date: 2026-10-11. Frozen base: `1c0b948a3`. Preregistration commit:
`98fe3748c`. Status: **HELD**. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round239/`.

This round changes no production model file, card, deck, carried state,
stabiliser, sea-ice selector, or `unmeasured_features` entry. Every ORCA2
number is kept separately as **independent** or **given NEMO's entry**.

## Frozen statement and causal result

The executing vector-form OMT-4 build forms completed `Ve_rhs` from live
`e3v_3d`, completed three-dimensional V RHS, `vmask`, and `r1_hv_0` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/stp2d.f90:194-199`. It copies
that forcing and subtracts the two-dimensional Coriolis trend through raw
`ssvmask` at
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:289-294` and
`ORCA2_OMIP_L4_R214VECPREV3/BLD/ppsrc/nemo/dynspg_ts.f90:323-328`.

The round-217 Ve_rhs/raw-mask pair is downstream of the completed 3-D RHS.
The round-239 causal arm nevertheless runs the entire passive step with the
pair ON and OFF under both labels. Pair ON and OFF completed RHS arrays are
bit-identical. Against NEMO, both retain exactly `668/1,050` unequal slots on
the 35-face support, maximum `2.3624915336282472e-06 m s-2` at
`[147,55,23]`. Every completed state field is array-identical between traced
and ordinary runs. Thus R239-P2's predicted `668 -> 0` closure is **REFUTED**.

The raw geometry is not reopened by this result: round 238's production-path
gate remains `0/226,637` unequal for live `e3v` and `0/399,600` for `vmask`
under both labels. The older round-236 diagnostic's internal reconstructed
geometry row is not used as current evidence; only its direct record admission
and completed-RHS row are inherited.

## Source-ordered operator table

The admitted RHS record build calls HPG, LDF, VOR, KEG and ZAD in that order
at `ORCA2_OMIP_L4_R92RHS/BLD/ppsrc/nemo/stp2d.f90:141-175`.
The offline pure-function replay is calibrated to the live completed RHS with
zero maximum residual and zero symmetric-difference cells. Its rank-complete
fold-row target is a 68-face superset of the 35 OMT-4 owner faces; HPG differs
on all 68, so the first operator on every possible 35-face subset is HPG.

| accumulator | unequal / support | RMS | maximum | argmax |
|---|---:|---:|---:|---|
| after HPG | 68 / 68 | `1.0748834693208086e-06` | `2.4423127429248864e-06` | `[147,135]` |
| after LDF | 68 / 68 | `1.0748834693208086e-06` | `2.4423127429248864e-06` | `[147,135]` |
| after VOR | 68 / 68 | `1.0748834693208086e-06` | `2.4423127429248864e-06` | `[147,135]` |
| after KEG | 68 / 68 | `1.0748834693208086e-06` | `2.4423127429248864e-06` | `[147,135]` |
| after ZAD | 68 / 68 | `1.0748834693208086e-06` | `2.4423127429248864e-06` | `[147,135]` |

HPG is therefore the first source-ordered owner. LDF, VOR, KEG and ZAD leave
the same fold-row result unchanged. This is an attribution boundary, not a
landing: the HPG surface/interior operands have not yet been split on the
post-Decision-114 geometry.

## Atomic-unit disposition

R239-P2 is a frozen prerequisite for scoring the complete round-217 unit.
Because it is false, R239-P3 through P5 are not activated: no OMT-4/rung-0/
rung-10 candidate ladder, month, GYRE, DINO or tank run is called PASS, and no
production candidate is applied. The current OMT-4 96-step completion, rung-0
step-100 boundary, and all round-238 certified trajectories remain unchanged.

## Instrument failures retained

1. The first combined gate called a retired temporary raw-mask hook and
   refused before measurement.
2. The repaired combined invocation first received the kt=8 round-166 stream
   where the inherited reader requires round 96's kt=1 stream; it refused on
   the missing self-described filename.
3. After both pair labels completed, the combined JAX process hit the known
   `Failed to materialize symbols: Cannot allocate memory` compiler limit.
   Pair and operator graphs were rerun in isolated processes, as prescribed.
4. The isolated operator JSON was complete, but a print-only wrapper expected
   a final `status` key and exited nonzero after writing it. The retained JSON
   parsed and combined cleanly after the wrapper fix; no numerical rerun was
   needed.

## Controls and artifacts

The pair-effect, claim-label coverage, source-order and first-operator plants
all refuse. Binding SHA-256 values are:

- classification: `cf5e546f10abd9735b1d3a90edcdae5183cda4471f70bc309907c3f48691695a`
- pair arm: `f636e2fbc297ffd08beef61697c77d5fd5130822215becc820d371c0700b7ca0`
- operator replay: `b418c387390920610def64a8d962d96fe89090c223c84d0bd487d2825bf8e06a`
- independent/given inherited rows: `92ed5a6215e53e3dedb62b1a754f7c1579c11e5a48396fa52861b35a28ec529d` /
  `9742c200a68cd08801a3eb1e1987ed95bea62c28ae5ceefed8b2b46e7bf3cd6d`

## Preregistered predictions

| prediction | disposition |
|---|---|
| R239-P1 | **CONFIRMED**: both labels admitted and both ON/OFF traces are passive. |
| R239-P2 | **REFUTED**: pair ON equals OFF and leaves `668/1,050` completed-RHS slots unequal. |
| R239-P3 | **NOT ACTIVATED**: P2 is false. |
| R239-P4 | **REFUTED PREREQUISITE**: closure threshold absent. |
| R239-P5 | **NOT ACTIVATED**: no candidate qualified for trajectory gates. |
| R239-P6 | **DOCUMENTARY ONLY**: focused/citation/battery/review results below; no package landing. |

## OPEN

Round 240 splits HPG on the fold row, offline and in compiled source order,
using the admitted component records: surface/interior `zhpj`, `zvap`, then
`pvv = zhpj + zvap`. It reuses round 180/181's records if their corrected
entry and post-Decision-114 geometry provenance closes; otherwise it names
the exact missing operand and requests a fail-closed acquisition. The complete
round-217 unit remains private until that upstream HPG debt closes and is then
re-scored atomically.

ASKED choices: Decisions 96, 103, 109, 113, 114 and 115. UNASKED choices:
empty.
