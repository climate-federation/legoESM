# ORCA2 round 129 — split-explicit fold transport walk

Date: 2026-10-03. Base `1635f0a4a`; held measurement tip `3d066a54d`; final
tip includes the explicit production revert. Rung-0 results are
**independent**. Shipped-card results are **given NEMO's recorded entry** under
Decision 52. No configuration, forcing, initial state, stabilizer, sea-ice
selector, or `unmeasured_features` entry changed.

## Verdict

**HELD.** The registered 68-cell substep-2 U residual is closed locally, but
not by the preregistered reference-depth substitution and not by one compiled
statement. It requires the complete external-mode boundary/transport chain:
associate the U/V carry and live V depth, retain the associated fold transport,
and preserve NEMO's written transport/divergence arithmetic. NEMO updates the
live depths and applies one batched lateral-boundary call after every substep
(`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:761-779`); its executed
T-pivot V branch supplies the sign/permutation association
(`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/lbcnfd.f90:684-721`). NEMO then forms
unmasked metric transports and the SSH update in written order
(`ORCA2_OMIP_L4_R96SPG/BLD/ppsrc/nemo/dynspg_ts.f90:580-666`).

The held chain makes substep-2 SSH, pressure, Coriolis, and both active depth
exits bit-exact; the first active non-bit boundary advances to `trend_u`,
15,789 cells and `2.1780653503519722e-08`. It also exposes a compensating
tracer/correction debt: both ORCA2 ladders move 195/200 rows. No exact row
leaves the bar and neither first-debt position moves, but rung-0 kt=10 stage-3
S max grows `0.4156325968352981 -> 31.361725198021634` and T max grows
`0.8929683387777949 -> 1.5944120037123595`. The shipped card similarly moves
S `0.2880329286832648 -> 30.55712755071444` and T
`1.2418655862627084 -> 1.618276229408231`. The binding independent rung-0
month therefore needs a committed scorer before this multi-statement chain can
land. All production changes were reverted at the final tip; the committed
walk remains.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R129-P1 record passive; eight EEN coefficients exact | **CONFIRMED**: exactly-once rank coverage and 0 unequal coefficients. |
| R129-P2 baseline first active debt is substep-2 U, 68 cells | **CONFIRMED**: maximum `2.9617669311254642e-08`. |
| R129-P3 first unequal input is carried velocity downstream of reference depth | **REFUTED**: the active-mask view first sees substep-2 `after_ssh`; the corrected full-operand view first sees the masked fold-row `entry_inverse_v`. |
| R129-P4 raw `hu_0/hv_0` substitution closes the residual | **REFUTED**: active U depths are exact. V velocity and V-depth associations each close their targeted operand but not the residual; associated transport plus periodic U carry is required. |
| R129-P5 every executing card passes if the statement lands | **NOT A LANDING CLAIM**: the held tip passes all measured shared guards, then is reverted. |

The mask-blind intermediate finding (“first input is after-SSH”) is retracted
by the committed full-operand score. Masked fold faces are prognostically dry
yet feed neighbouring active stencils, so their operands must be scored over
the complete recorded domain.

## Candidate measurements and guards

- Rung 0 (**independent**): gate completes 200 rows; 195 move, 5 remain
  unchanged, no bit-identical loss, first debt stays kt=1 stage-1 T.
- Shipped ORCA2 (**given NEMO's recorded entry**): 195/200 rows move, no
  bit-identical loss, first debt stays kt=1 stage-1 T.
- GYRE: 70 certified rows are array-identical; maximum worsening is 0 ULP.
- DINO: CPU/fp64 month completes 960 steps; day-30 wet 3-D T RMS is
  `2.053801168e-03 K` against the fixed `2.244317642e-03 K` bar. The planted
  `6.981690958e-03 K` regression fails.
- LOCK_EXCHANGE: 20 rows unchanged, 0 ULP worsening, first debt kt=4 U.
- OVERFLOW: 10 rows unchanged, 0 ULP worsening, first debt kt=2 T/U.
- One initial shipped-card invocation was externally quiet/terminated and
  emitted no JSON. The heartbeat retry completed and alone supplies the
  shipped-card result.

Separate read-only Codex review was attempted before the arm and returned
**independent review unavailable in-sandbox**: `failed to initialize in-process
app-server client: Read-only file system`.

## OPEN

1. Build a committed rung-0 independent-month scorer for the admitted
   round-83 240-step terminal restart, including per-field RMS/max, finite-step
   refusal, and a planted terminal ULP/non-finite control.
2. Walk the first exposed tracer/barotropic-correction association in source
   order. The strong SSH/velocity improvement paired with the tracer maximum
   regression is a measured compensating pair, not permission to drop NEMO's
   boundary call.
3. Re-run this complete held chain under that month gate; land it only as a
   finished source unit if the month and both ORCA2 ladders satisfy their
   registered predicates.

## UNVERIFIED

- The exact tracer/correction statement responsible for the kt=1 S/T movement.
- The independent rung-0 240-step month under the held chain.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
