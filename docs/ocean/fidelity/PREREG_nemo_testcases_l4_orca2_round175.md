# Preregistration — ORCA2 round 175 kt=1 stage-1 column/operator walk

Date: 2026-10-08. Frozen base: `640d7ee6c`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round175/`.

Every scientific number is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity and zero sea surface. No given-NEMO-entry
rung-7 number is mixed into this round. Sea ice, all six sea-ice selectors and
the shipped card's `unmeasured_features` tuple remain unchanged.

## Frozen source order and target

Round 174 selected kt=1 stage 1 as the first greater-than-10x completed-stage
growth boundary. Salinity owns the maximum: `3.2847473521544472 PSU` at the
zero-based global cell `[j=86,i=159,k=3]`.

The compiled rung-0 program first interpolates the external mode and QCO
stretch, then builds the stage transports, applies the already-computed
stage-1 momentum RHS and barotropic correction, constructs tracer transport,
zeros the tracer RHS, calls tracer advection, applies the surface tracer
source, performs QCO tracer time stepping and finally exchanges the stage
fields (`ORCA2_OMIP_L4_R90FRAMES/BLD/ppsrc/nemo/stprk3_stg.f90:127-182`,
`:253-365`, `:462-555`, `:600-697`, `:775-798`). The registered offline table
uses that order:

1. external-mode/QCO stage operands;
2. completed stage-1 momentum RHS;
3. momentum update before the barotropic correction;
4. corrected U/V;
5. metric transports `zFu/zFv/zFw`;
6. tracer RHS after advection;
7. tracer RHS after surface forcing;
8. tracer QCO update;
9. completed stage after boundary association.

No in-executable observer is permitted. Candidate rows come only from pure
operator replays driven by the passive exact entry and already-recorded
operands. Oracle rows must come from self-describing, rank-complete records.

Before the table, the gate prints for `[86,159]`: NEMO `mbkt`, `e3t/e3w` and
T/U/V masks at levels 0--5, the four horizontal neighbours' masks, whether the
column lies on the northern fold or cyclic seam, and the matching legoESM
geometry. Array shapes and dtypes are printed before indexing.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R175-P1 | The target is a shallow partial-cell column, not a fold or cyclic-seam cell. | `mbkt=4`, T levels 0--3 wet and level 4 dry, bottom wet level 3 has partial `e3t/e3w`, at least one horizontal neighbour is land, `j != 147`, and `i` is neither 0 nor 179. | Any listed geometry fact differs. |
| R175-P2 | The existing round-90 tracer-stage operand stream is not rank-complete for the target column. | Its header/shape covers only rank 0's 90 owned longitudes, so global `i=159` is absent; the table marks every downstream target-column tracer boundary UNMEASURED. | A self-describing exactly-once two-rank stream contains the target and every registered tracer boundary. |
| R175-P3 | Where the existing records are complete, the stage-1 table keeps the entry exact and the completed-stage salinity row at the round-174 value; it does not infer an internal owner across a missing row. | Entry is bit-exact, completed-stage S max is `3.2847473521544472 PSU` at `[86,159,3]`, and the first unavailable internal row terminates selection. | Either pinned boundary moves, or the gate selects a statement below an absent boundary. |
| R175-P4 | A rank-complete stage-1 boundary acquisition is required before the salinity owner can be named. | The existing census lacks at least one of corrected velocity, metric transport, after-advection tracer RHS, after-SBC tracer RHS, or pre-association QCO output on rank 1; a fail-closed additions-only launcher is written under a new target name. | Every registered boundary is already rank-complete and admissible. |
| R175-P5 | This round is measurement/acquisition-only. | No `packages/`, card, configuration, carried state, stabiliser, threshold, sea-ice selector or production halo-unit path changes. | Any such change lands. |

Failed predictions remain in the receipt. The cellwise floor is the frozen
`2e-10` quantity floor. Bit claims require `np.array_equal`; a tolerance row
cannot be called exact.

## Controls and terminal rule

The gate must reject planted target indices, neighbour masks, rank placement,
boundary order and first-unmeasured selection. A one-ULP plant must move a
present row. Candidate, oracle and geometry arrays must be float64; backend is
CPU, production JIT is on and the precision policy is fp64/libm.

If an internal oracle row is absent at `[86,159,3]`, stop there and write the
rank-complete self-describing acquisition. The writer records existing arrays
only, uses one file per rank, parses field names/dimensions/payload lengths from
its own header, and proves additions-only by byte-identical kt=10 terminal
restarts against the admitted round-90 producer. No statement below a missing
row is attributed.

ASKED choices: continue the independent rung-0 stage-1 walk at the round-174
column, using offline replay only.  
UNASKED choices: empty.
