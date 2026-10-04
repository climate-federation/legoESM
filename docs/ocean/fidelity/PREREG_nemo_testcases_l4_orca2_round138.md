# Preregistration — ORCA2 round 138 adjacent-beta operand walk

Date: 2026-10-04. Base: `dac955149`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round138`.
All step-36 values are **independent**: the rung-0 card starts from its own
climatological T/S, zero velocity, and zero sea surface. Given-entry ladder
populations remain separate.

Round 137 proved that the first non-finite value at the upstream level-3
target `(j,i,k)=(86,159,3)` is the north incident V-face limiter coefficient
at face `(87,159,3)`. This round extends the admitted private passive FCT trace
with the exact cell-centred operands used by the two adjacent limiter ratios.
The compiled ORCA2 source builds `zup`, `zdo`, `zpos`, `zneg`, and `zbt`, then
the guarded `zbetup`/`zbetdo` ratios at
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/traadv_fct.f90:849-878`; the north V-face
coefficient consumes the adjacent ratios at `:910-913`.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R138-P1 | The extended trace is passive and source ordered. | Every ordinary state leaf and the independent repeat are bit-identical; the registry, passivity, source-order, adjacent-face, and plant controls fire. | Reject every beta number and stop at the observer defect. |
| R138-P2 | The first non-finite adjacent-cell beta operand is an incident sign-split flux sum (`zpos` or `zneg`), before `zbetup`/`zbetdo`. | `zup`, `zdo`, and `zbt` are finite at both adjacent cells, while the first source-ordered non-finite row is `zpos` or `zneg`. | Mark **REFUTED** and retain the actual earliest source row; do not skip it. |
| R138-P3 | The non-finite adjacent operand mechanically owns round 137's north-face coefficient. | The selected adjacent `zbetup`/`zbetdo` is non-finite, the face flux sign selects that ratio pair, and the resulting coefficient reproduces `nan`. | Mark **REFUTED** and walk the first finite-to-non-finite association between the beta rows and face coefficient. |
| R138-P4 | No production statement, configuration, deck, selector, carried state, stabilizer, sea-ice field, or `unmeasured_features` entry changes. | Only the private default-off trace, its gate/tests, receipt, and citation map change; all standing shared-card gates retain their certified rows. | Hold at the first moved shared-gate row. |

## Round bar

The first source-ordered adjacent-cell non-finite row owns the walk. A statement
is named only from the compiled ORCA2 branch and the admitted passive trace.
The distinct global averaged-upstream-flux overflow at `(87,160,5)` remains
separate unless this exact beta stencil mechanically consumes it. No physics
landing, card change, stabilizer, threshold relaxation, or sea-ice change is
authorized.
