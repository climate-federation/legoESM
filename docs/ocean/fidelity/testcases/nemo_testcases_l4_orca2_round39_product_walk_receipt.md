# NEMO testcase Lane 4 — ORCA2 card round 39 product-walk receipt

Date: 2026-09-26

Parent: `06d6cf1a68ace46182782069fba5c83f9bfc2ec0`

Status: **HELD — THE THICKNESS/RHS CANCELLATION DOES NOT OCCUR IN THEIR
PER-LEVEL PRODUCT.**  No production model file changed.

All results are **given NEMO's entry**.  The six sea-ice selectors and the
card's `unmeasured_features` tuple remain unchanged.  Evidence is under
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round39/`.

## Compiled statement and result

The executing vertical average multiplies `e3u_3d`, `uu(Krhs)`, and `umask`,
sums the 30 wet levels, then multiplies by `r1_hu_0` at
`ORCA2_ORCA1ICE_OMIP_L4_R20SLOWRANK/BLD/ppsrc/nemo/stp2d.f90:189-204`.

Round 38 reproduces exactly.  R39-P1 is **CONFIRMED**.  On its 64 disputed
rank-1 east-source rows:

| compiled boundary | unequal / scored | maximum |
|---|---:|---:|
| per-level `(e3u_3d*uu(Krhs))*umask` | 1,754 / 1,920 | `5.321462756514503e-08` |
| first partial sum, level 0 | 64 / 64 | `9.242193862245139e-11` |
| completed 30-level sum | 64 / 64 | `2.721729894586411e-07` |
| completed sum times production reciprocal | 64 / 64 | `7.356587026022005e-18` m/s2 |

R39-P2 and R39-P3 are **REFUTED**.  The first non-bit compiled statement is
the per-level product itself; the first partial sum is already non-bit.  The
small final depth-mean boundary is therefore a later compensation involving
the reciprocal depth as well as thickness and RHS, not a bit-exact product
hidden by reduction order.

The one-variable reciprocal arms remain large: candidate sum with NEMO's
reciprocal differs by `9.420110520449829e-11` m/s2, while NEMO's sum with the
candidate reciprocal differs by `9.423577944554755e-11` m/s2.  No isolated
statement reaches zero cells, so R39-P4 is **CONFIRMED** and nothing lands.

## Controls and review

The Round-38 admission, record self-replays, inherited boundary, and both
Round-38 plants reproduce.  A one-ULP mutation of the newly scored product
boundary fires.  The receipt citation gate uses the already pinned compiled
range above.

The required separate `codex exec --sandbox read-only` review remains
unavailable in this sandbox: its app-server initialization fails with a
read-only-filesystem error before reading the diff.

## Choices

ASKED: walk the measured cancelling pair through its compiled product and
reduction.

UNASKED: none.  No model, configuration, carried state, stabilizer, NEMO
source, acquisition, sea-ice selector, or score changed.

## OPEN

1. The slow-forcing compensation spans thickness, completed 3-D RHS, and
   reciprocal reference depth.  Before any fix, split the RHS by its compiled
   momentum operators or prove the geometric identity relating these three.
2. The actual whole-card first non-bit row remains kt=1 stage-1 temperature;
   the barotropic/transport handoff remains its upstream owner candidate.
3. The northern-fold mask/wind debt and Decision-52 independent year remain
   open.
