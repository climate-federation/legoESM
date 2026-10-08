# Preregistration — ORCA2 round 169 kt=8 substep-1 exit-U walk

Date: 2026-10-07. Frozen base: `e421a39e6`. Evidence root:
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round169/`.

Every ORCA2 number in this round is **independent**: hierarchy rung 0 starts
from its own climatological T/S, zero velocity and zero sea surface. No
given-NEMO-entry rung-7 number is mixed into the result. Sea ice, all six
sea-ice selectors and the shipped card's `unmeasured_features` tuple remain
unchanged.

## Frozen source order and scope

The admitted round-166 rank-complete kt=8 record is the only oracle input.
The compiled rung-0 program evaluates the vector-form substep U update as the
carried U plus the time step times pressure, momentum-trend and slow-forcing
terms at
`ORCA2_OMIP_L4_R166SPG8/BLD/ppsrc/nemo/dynspg_ts.f90:712-723`. It applies
explicit bottom drag before that update at `dynspg_ts.f90:688-706`, refreshes
the exit face depth and reciprocal at `dynspg_ts.f90:760-767`, then associates
U/V, both depths, both reciprocals and SSH in one `lbc_lnk` call at
`dynspg_ts.f90:770-779`. The completed `ua_e` is carried into the next
substep at `dynspg_ts.f90:836-844`.

The walk scores substep 1 in that order: entry U; pressure; Coriolis; drag;
combined trend; slow forcing; ordered RHS; increment; pre-association U; and
post-association U. Each arithmetic replay uses NEMO source rounding and is
tested first with all recorded operands, then with one candidate operand at a
time. A row is named only at the first non-bit boundary. Separately, the
round-168 1/41 face-average residual is localised by native face and by the
pre/post-association timing of its two contributing T cells.

## Frozen predictions and falsifiers

| ID | Prediction | CONFIRM | REFUTE |
|---|---|---|---|
| R169-P1 | Substep 1 enters with finite but already non-bit U, so its exit update is not the first owner. | `i000_un_e` is the first non-bit source-order row and recorded-entry replay closes the substep-1 U update. | Entry U is bit-exact, or recorded-entry replay does not close the update. |
| R169-P2 | Given NEMO's recorded update operands, legoESM's literal vector-form arithmetic reproduces the recorded pre-association U bits. | The recorded-operand update replay is bit-exact before the boundary exchange. | Any active interior bit remains unequal before exchange. |
| R169-P3 | The substep-1 exchange is already exact under the complete private arm. | The pre-to-post association replay reproduces recorded `j001_ua_new` at every active/boundary cell. | Any post-association U bit remains unequal given recorded pre-association U. |
| R169-P4 | The round-168 1/41 face-average residual is a timing/association mismatch, not the literal weighted average. | The residual face is named and the replay becomes exact when the matching pre/post-association after-SSH frame is used. | The matching-time recorded frame still leaves the face non-bit. |
| R169-P5 | This remains a measurement-only hold. | No `packages/`, recipe, selector, forcing, carried-state or sea-ice change lands. | One cited statement closes the upstream unit and passes all Decision-96 landing gates in this round. |

## Controls and terminal rule

The gate must reject independent plants in the source-order registry,
recorded-operand replay, one-variable entry-U replay, exchange replay and
face-average residual registry. It must re-admit both record ranks and the 20
terminal restart comparisons. Failed predictions remain in the receipt. No
stabiliser, clip, configuration choice, bar relaxation or partial atomic-unit
landing is permitted.

ASKED choices: continue round 168's compiled-source independent rung-0 walk.
UNASKED choices: empty.
