# Preregistration — ORCA2 round 100 Decision-84 lane merge

Date: 2026-10-02. Base: `752616c1c12d566a2d5e727434011add95ce2f0a`.
Scope is the merge only. The source fetched from the canonical local GYRE lane
is `413984f3bcab9c2bd287412d06d6a36dff02c868`, the tip named by operator note
B33. No ORCA2 physics, card, deck, sea-ice selector, carried-state convention,
threshold, stabilizer, or `unmeasured_features` entry may be chosen or changed
outside conflict resolution required to combine those two committed trees.

Every ORCA2 number below is labelled **independent**: hierarchy rung 0 starts
from NEMO's own from-rest state. The shipped rung-7 ladder is also scored as
**given NEMO's entry** where its existing gate says so. Those labels may not be
mixed in one results table.

## Cited statement entering through the merge

The source lane transcribes NEMO's frozen vorticity thickness into the shared
literal split-explicit coefficient builder. NEMO selects all curl schemes in
one arm at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynvor.f90:890`, constructs the
`nn_e3f_typ=0` masked four-T-cell mean at line 897, restores fully dry vertices
at line 920, and consumes that thickness in the EEN Coriolis coefficient at
`VORTEX_VEC_R8_OMIP_L1_P3/BLD/ppsrc/nemo/dynspg_ts.f90:960`. ORCA2's executing
EEN construction and application remain the compiled statements at
`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:1213-1265,1369-1392`.

The merge must preserve the union of both branches' citation maps. A conflict
resolution that drops either branch's keys or leaves any unresolved marker is
a hard refusal, not a scientific result.

## Frozen predictions and falsifiers

| ID | Frozen prediction | Confirmation | Falsification / action |
|---|---|---|---|
| R100-P1 | The fetched source is the authorised Decision-84 landing and can be merged without an unasked configuration choice. | The merged ancestry contains both `752616c1c` and `413984f3b`; every conflict is listed with the exact side retained; the resolved card and deck choices equal the ORCA2 parent unless the GYRE side contains a user-authorised shared statement; `CITATION_MAP` is the union. | Any conflict requires selecting between different scientific configurations or cannot be resolved as a mechanical union: **REFUTED**, stop with `DECISION_NEEDED` and the conflict list. |
| R100-P2 | GYRE is byte-identical under the shared `e3f_0vor` statement. | Its certified ten-step ladder has zero moved rows, unchanged statuses and first-over-bar; the certified day-30/day-240/day-360 values remain `2.3432510206121264e-06`, `6.581707093530567e-05`, and `5.407735418221895e-05` K with array-identical residuals/snapshots where the gate exposes them. | Any GYRE trajectory or certified residual byte moves: **REFUTED** and the merge is HELD unless the standing Decision-84 evidence explicitly registers that exact move. |
| R100-P3 | ORCA2's EEN coast/fold coefficients execute the shared statement, so both certified ORCA2 ladders move while their entry rows remain unchanged. | At least one rung-7 row and one rung-0 row move; every moved row is registered; no AT-BAR row leaves the bar and first-over-bar is not earlier; kt=1 entry is bit-identical. | No moved row means the arm did not execute or the comparison is stale. Any unregistered move, earlier first-over-bar, or AT-BAR loss is a gate red and the merge is HELD. |
| R100-P4 | The shared helper supersedes round 99's plain-`e3f_0` production baseline, so round 99's exact coefficient counts and the 68-cell substep-2 U number are stale. | The production builder reports a nonzero old-vs-merged coefficient delta on ORCA2, and the merged coefficient/application probe produces a new complete table for all eight coefficients and all four scored substep rows. | Zero production coefficient movement is **REFUTED** and indicates the shared statement is not on ORCA2's executing path; stop before reusing old numbers. |
| R100-P5 | Round 99's coefficient signed-zero owner survives as the first non-bit boundary, but its counts may change; the northern-fold magnitude debt and the 68-cell substep-2 U residual are independent read-outs, not assumed closed. | The merged probe names the first non-bit boundary and separately reports zero-sign, magnitude/fold, and substep-2 differences. A surviving boundary is recorded with its new counts; an exact boundary is recorded as **REFUTED** and the walk advances only next round. | If the probe cannot distinguish coefficient bits from application bits, or does not cover both rank slabs, these findings remain **UNMEASURED** and no survival claim is made. |
| R100-P6 | Citation and push gates remain binding after conflict resolution. | Default and round receipt citation checks pass with zero unmapped citations and their real-key plants fire; the push-gate battery reaches its own terminal pass line. | Any unmapped citation, non-firing plant, unresolved conflict marker, or new non-listed test failure makes the round HELD. |

## Landing bar

This round lands only the merge. It lands only if the merge is complete, both
parents are ancestors, all conflict resolutions are documented, the citation
map is a union, the required GYRE and ORCA2 measurements satisfy the frozen
bars above, and the citation/push gates pass. Failed predictions stay in the
receipt as **REFUTED**. No follow-on rung-0 walk statement may land in round
100.
