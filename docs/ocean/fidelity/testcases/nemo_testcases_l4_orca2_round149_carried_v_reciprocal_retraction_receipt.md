# ORCA2 round 149 — carried V reciprocal debt retracted

Date: 2026-10-05. Base `0c0ceb3dae`; measurement tip `69b87da1c`.
Preregistration: `docs/ocean/fidelity/PREREG_nemo_testcases_l4_orca2_round149.md`
at `de9f20d4d`. Verdict: **HELD**.

Every ORCA2 number below is **independent**: hierarchy rung 0 starts from its
own climatological T/S, zero velocity, and zero sea surface. No configuration,
forcing, initial state, carried-state form, stabiliser, sea-ice selector, or
`unmeasured_features` entry changed.

## Result and loud correction

Round 148's statement that the substep-2 carried V reciprocal was the first
non-bit operand is withdrawn. The private trace zeroed the reciprocal outside
the active V mask at `barotropic_latlon_cgrid.py:2435-2439`, while NEMO's
associated `hvr_e` retains boundary storage. Preserving the raw private trace
changes the reported `entry_inverse_v` census from 68 unequal cells, maximum
`0.03332976059679253`, to **0 unequal cells and 0.0 maximum**. Observer
passivity, all seven post-association arrays, and all eight literal-EEN
coefficients remain bit-exact.

NEMO computes `hvr_e` from the updated V depth and carries it through the
seven-array association at compiled `dynspg_ts.f90:761-779`. The next substep
uses it in the explicit V bottom-stress product at `dynspg_ts.f90:680-699`.
The corrected trace proves that operand is already bit-exact; the
record-substitution arm therefore has no causal claim to test. R149-P2 is
**REFUTED** and R149-P3 is **UNMEASURED_PREREQUISITE_R149-P2**, not a failed
physics statement.

The final evidence is
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round149/carried_inverse_v_final.json`,
SHA-256 `261d7f38b63a22e4bd94e07d147bb18db655de8799b51c3efd43070df2d1d1d2`.
It is stamped to `69b87da1cdeda80538209df5181d7033c6bc937b`, CPU,
production JIT, fp64/libm, and x64.

## First non-bit statement

With the carried reciprocal removed from the debt register, the first
full-domain non-bit operand is again the substep-2 midpoint V-face depth:
30 northern cells, maximum absolute difference **899 m**. NEMO extrapolates
the midpoint sea surface at `dynspg_ts.f90:515-519`, then forms the V depth
from the local and north T-cell area-weighted values at
`dynspg_ts.f90:535-545`. legoESM's corresponding private, source-rounded
builder is `barotropic_latlon_cgrid.py:588-629`.

The unchanged descendants remain 68 unequal cells in the V transport
(`155776.5627856178` transport units), 68 in the V south difference (same
maximum), and 68 in SSH (`0.003203816535399729` m). Their compiled order is
the V transport at `dynspg_ts.f90:564-570`, the south difference at
`dynspg_ts.f90:580-589`, then the SSH update at `dynspg_ts.f90:591`.
Round 148's row-below north-neighbour arm left all four censuses unchanged, so
that candidate stays **REFUTED**.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R149-P1 prerequisites exact | **CONFIRMED**: observer, seven-array association, and eight EEN coefficients are bit-exact. |
| R149-P2 control has 68 reciprocal differences | **REFUTED**: corrected raw trace has 0 unequal cells and 0.0 maximum. |
| R149-P3 record substitution closes reciprocal | **UNMEASURED_PREREQUISITE_R149-P2**: the ordinary operand is already exact. |
| R149-P4 midpoint/continuity chain stays open | **CONFIRMED**: 30/68/68/68 cells remain at the registered maxima. |
| R149-P5 production unchanged | **CONFIRMED**: both additions are private/default-off; GYRE is byte-identical. |

## Controls and shared path

The original wrong-frame control was vacuous because recorded `i000_hvr_e`
and `j001_hvr_e` are bit-identical. That instrument defect is retained in
`wrong_entry_frame_plant.log` as `plant stayed green`. The corrected control
uses the first distinct recorded frame, `j002_hvr_e`, and fails closed if that
precondition changes. It fires with `wrong-entry-frame plant fired`; the
one-ULP reciprocal plant fires with `entry-inverse-v-bit plant fired`; and the
registry-order plant also fires. Focused tests cover both the distinct-frame
precondition and a deliberately vacuous synthetic record.

The GYRE ten-step comparison passes on 70 certified rows: zero status changes,
zero violations, zero ULP worsening, and unchanged first-over-bar kt=3. The
30-day member has 30/30 byte-identical daily snapshots against round 147;
day-30 SHA-256 remains
`b017623dea468af4f4ba31761148aa52e478d60196828443b56ceeee36534180`.
The differing `manifest.json` only records the producer commit and is not a
state snapshot.

The separate `codex exec --sandbox read-only` review returned **independent
review unavailable in-sandbox**: `failed to initialize in-process app-server
client: Read-only file system`.

No ORCA2 ladder was run: R149-P2 was refuted, the midpoint/continuity chain did
not close, and the preregistration forbids a ladder under that condition. No
production physics or card statement lands in this round.

ASKED choices: none. UNASKED choices: empty.

## OPEN

1. Walk the 30-cell midpoint V-depth calculation in compiled source order:
   the extrapolated `zsshp2_e`, local and north `e1e2t*zsshp2_e` products,
   reference `hv_0`, reciprocal area, and `ssvmask`, one operand at a time.
2. Re-test the 68-cell transport, south-difference, and SSH chain only after
   the V depth is bit-exact. Do not run either ORCA2 ladder before that chain
   closes; the registered ~31 PSU salinity exposure remains a hard veto.
3. After the barotropic unit closes, re-run the independent rung-0 month and
   resume the hierarchy/parked-merge program. The EVD-composition and
   bottom-drag-divisor shared statements remain downstream merge items.
