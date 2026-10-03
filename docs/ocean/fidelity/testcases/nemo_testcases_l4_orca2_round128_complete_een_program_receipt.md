# ORCA2 round 128 — complete literal-EEN source program

Date: 2026-10-03. Base `2baaf09e4`; measurement tip `387003397`.
Rung-0 numbers are **independent**. Shipped rung-7 numbers are **given NEMO's
recorded entry** under Decision 52. No configuration, forcing, carried state,
stabilizer, sea-ice selector, or `unmeasured_features` entry changed.

## Verdict

**LANDED.** The admitted rank-complete records close NEMO's complete eight-
coefficient literal EEN initialization program bit-for-bit. For northern V,
NEMO consumes the already-associated U thickness and U mask in the NW/NE
recurrences
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1339-1348`) and the
associated northern U metric in the final scales
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dynspg_ts.f90:1360-1367`). The
ordinary T-pivot U-grid exchange uses its U-origin permutation and sign +1
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/lbcnfd.f90:639-683`). The shared
builder now transcribes that association (`barotropic_latlon_cgrid.py:953-974`),
keeps NEMO's live face thicknesses unmasked, applies the literal wet-level
loop bounds (`barotropic_latlon_cgrid.py:1079-1094`), and routes all four U
and four V neighbours through the source-order recurrence and scale program
(`barotropic_latlon_cgrid.py:1197-1233`).

The same landing includes two already measured parts of that indivisible
program: host IEEE signed-zero addition and NEMO's constant-zero southern
V-mask boundary. NEMO constructs and exchanges that mask before use
(`ORCA2_OMIP_L4_R120EENVFRAC/BLD/ppsrc/nemo/dommsk.f90:232-258`); production
uses the already-gated zero-fill helper (`barotropic_latlon_cgrid.py:919-921`).

## Frozen prediction disposition and source walk

| prediction | disposition |
|---|---|
| R128-P1: northern U association is row below stored pivot, U-origin permutation, sign +1 | **CONFIRMED** |
| R128-P2: it closes 95 NW / 91 NE thickness and 1,270 NW / 1,283 NE mask cells | **CONFIRMED**; every descendant recurrence item becomes exact |
| R128-P3: associated northern metric closes the two 68-cell scale rows | **CONFIRMED** |
| R128-P4: combined final NW/NE V coefficients become exact | **CONFIRMED**, 0 unequal |
| R128-P5: the complete production builder makes all eight coefficients exact | **CONFIRMED**, every coefficient 0 bit-unequal and 0 magnitude-unequal |
| R128-P6: no certified trajectory row regresses | **CONFIRMED** by the gates below |

The first non-bit round-127 statement was the northern neighbouring U-face
thickness: 95 NW and 91 NE cells. Thickness-only left the masks unequal;
mask-only left the thickness unequal. Their combined arm made every recorded
source row exact. The pre-scale metric then moved 68 cells per northern path
to exact, and the final coefficients were exact. A production check against
the eight admitted final coefficient arrays initially exposed 68 southern-row
signed zeros in each southern U coefficient; applying the previously measured
closed-boundary V-mask association closed those too. This is why the landing
is the complete source program rather than a northern-only patch.

The clean gate reports `MEASURED_R128_EEN_U_FOLD_ASSOCIATION`. Its oracle-bit,
candidate-bit, production-bit, wrong-row, wrong-permutation, and scope-route
plants all fire. The resolved execution census remains ORCA2, both DINO
recipes, VORTEX and VORTEX_VEC true; GYRE and both tanks false.

## Trajectory landing gates

- ORCA2 rung 0 (**independent**): 0/200 rows move versus round 126; no
  bit-identical row leaves the bar and first debt remains kt=1 stage-1 T.
- ORCA2 rung 7 (**given NEMO's recorded entry**): 0/200 rows move; no
  bit-identical row leaves the bar and first debt remains kt=1 stage-1 T.
- GYRE: 0/70 ten-step rows move, maximum worsening 0 ULP, and the residual
  archive remains `377dd4c211d49a8675c9b48eed40d6a694c70c3996ea7aef8698d3f92ab033b7`.
  All 30 daily snapshots are byte-identical; day 30 remains
  `4e36c106403b495e95327213292f0d1655d605fca6b0a75c67cb17833f067cba`.
- DINO: the heartbeat-protected fp64 CPU month completes 960 steps. Day-30
  wet 3-D T RMS is `2.053801168e-03 K`, below the fixed
  `2.244317642e-03 K` bar.
- LOCK_EXCHANGE and OVERFLOW move 0 rows under matching stop-at-first-debt
  comparisons; first debt stays kt=4 U and kt=2 T/U respectively.

Two quiet rung-7 attempts and one quiet DINO attempt were externally
terminated before emitting a result. They are incomplete and are not evidence.
The heartbeat-protected commands above completed and alone supply the verdicts.

## Tests, citations, and review

PENDING_FINAL_VALIDATION.

Separate read-only Codex claim review was attempted and returned
**independent review unavailable in-sandbox**: `failed to initialize in-process
app-server client: Read-only file system`. Final-diff review is repeated after
this receipt is complete.

## OPEN

1. Resume the distinct 68-cell substep-2 U residual now that the frozen EEN
   coefficient program is bit-exact.
2. Continue the independent hierarchy/card month program after the rung-0
   source walk. This receipt makes no independent ORCA2 month claim.
3. The shipped rung-7 first debt remains kt=1 stage-1 T and source-unattributed.

## UNVERIFIED

- The owner of the distinct 68-cell substep-2 U residual.
- The owner of rung-7 kt=1 stage-1 T debt.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
