# ORCA2 round 147 — exact T-pivot U association

Date: 2026-10-05. Base `04abf66161`; measurement tip `d28cc069f`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round147.md`
at `c25214d44`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried state, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result

NEMO's one seven-field association is at
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`.
Its executed T-pivot U branch is
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/lbcnfd.f90:639-683`: with
`nn_hls=2`, native pivot-row U columns 0..89 stay in place and columns 90..179
become the sign-minus reverse of columns 89..0. Round 146 had applied only the
compact periodic closure and therefore retained 56 wrong zero sign bits.

The existing private helper now transcribes that half-row assignment inside
the same seven-array call (`barotropic_latlon_cgrid.py:627-668`). The observer
remains passive: T, S, u, v, ssh, `uu_b`, and `vv_b` are array-identical to an
ordinary production step. The association changes 120 compact U storage cells
(the prior 64 periodic cells plus exactly 56 pivot-row sign bits), all within
the registered boundary; zero interior cells move.

All seven post-call arrays now reproduce NEMO bit-for-bit:

| array | unequal cells | maximum absolute difference |
|---|---:|---:|
| U velocity | 0 | 0.0 |
| V velocity | 0 | 0.0 |
| U live depth | 0 | 0.0 |
| V live depth | 0 | 0.0 |
| U reciprocal | 0 | 0.0 |
| V reciprocal | 0 | 0.0 |
| SSH | 0 | 0.0 |

Evidence JSON SHA-256 is
`f02e91d927e460058268f4f1b3f5699bb7718b659bbfc5743acf02185de4d551`.
The first run's science rows were identical, but its stale boundary-scope mask
classified the new pivot half-row as interior. That bookkeeping refusal is
retained at SHA-256
`ca107d751195452f25188ff39c0ff4bb73b73dcc27e30fd4d3ed06577a8d77e3`;
the scope mask was repaired and its planted interior change still refuses.

## First non-bit statement

Given the exact complete call, substep-2 `continuity_du` closes from 64 unequal
cells (maximum 205276.59075050754 transport units) to **0 unequal cells**.
The next source-ordered non-bit statement is the V transport difference:
`zhV(ji,jj) - zhV(ji,jj-1)` in
`ORCA2_OMIP_L4_R144INITIAL/BLD/ppsrc/nemo/dynspg_ts.f90:564-591`.
It differs on 68 active cells, maximum 155776.5627856178 transport units at
`(j,i)=(147,135)`. The following `ssh_after` differs on the same 68 active
cells, maximum 0.003203816535399729 m.

legoESM's literal continuity specialization forms the compact V difference at
`barotropic_latlon_cgrid.py:773-803`. Every earlier registered substep-2 row,
including the exact post-call V, midpoint V, V depth, and metric transport,
is bit-exact. This round does not rewrite that downstream statement and does
not run either ORCA2 ladder: the complete call has not closed SSH, and the
registered ~31 PSU salinity exposure remains a landing veto.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R147-P1 exact U half-row | **CONFIRMED**: U 56 unequal sign bits to zero; all seven rows exact. |
| R147-P2 U divergence closes | **CONFIRMED**: substep-2 `continuity_du` 64 to zero. |
| R147-P3 V difference next | **CONFIRMED**: 68 cells, maximum 155776.5627856178. |
| R147-P4 SSH remains non-bit | **CONFIRMED**: 68 cells, maximum 0.003203816535399729 m. |
| R147-P5 ordinary trajectories unchanged | **CONFIRMED**: observer passivity exact; GYRE ladder/month byte-identical. |

## Shared-path gates

The required GYRE base (`04abf66161`) and tip (`d28cc069f`) trajectory reports
compare PASS over 70 certified rows: zero status changes, zero violations, and
unchanged first-over-bar kt=3. Their residual NPZ files are byte-identical
(SHA-256 `7f34d4d8f42e5a23b2e4c00dcd7d35e0a778a284ed1f54306fb457618dde7af3`).
The independent 30-day members saved 30 daily snapshots each; all 30 file
payloads are byte-identical. Both day-30 files have SHA-256
`b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180`.

The wrong-U-sign full-record plant, observer-bit plant, post-call ULP plant,
and registry-reorder plant each exit 2 with `STATUS PLANT-FIRED`. Focused tests
pass 31/31. The one required `tests/ocean/fidelity -n 12` invocation reached
99% and reproduced the registered xdist tail stall; it is **INCOMPLETE**, not
PASS: 2,563 passed, 7 skipped, and four pre-existing reds (SI3 scalar-math
provenance, round-35 escape scope, worktree-stamp scope, and the GYRE
round-129 spread-record stamp). Every round-147 test passed.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`.

ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Walk the 68-cell substep-2 V transport difference in compiled order. Since
   its recorded V transport operand is already exact, first test the compact
   south-neighbour/fold association used by the difference; do not change the
   transport product or divergence together.
2. Re-test `after_ssh`, then both ORCA2 ten-step ladders only after the V
   difference is exact. The ~31 PSU salinity exposure remains a hard veto.
3. After this barotropic unit closes, re-run the independent rung-0 month and
   then resume the hierarchy/parked-merge program. The EVD-composition and
   bottom-drag-divisor shared statements remain downstream merge items.
