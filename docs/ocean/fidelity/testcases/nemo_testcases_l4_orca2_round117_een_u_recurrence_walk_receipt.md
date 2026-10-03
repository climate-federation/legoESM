# ORCA2 round 117 — remaining U EEN recurrence walk

Date: 2026-10-03. Base `a664ca424d6d931751db5d906d0c5abe859c6957`.
All ocean numbers below are **independent**: hierarchy rung 0 starts from
NEMO's own from-rest state. No model physics, card field, configuration,
carried state, stabilizer, or sea-ice selector changed in this round.

## Verdict

**HELD.** The admitted round-116 record completes the U-coefficient walk but
does not justify a partial production landing. Northeast U confirms round
115's signed-zero recurrence diagnosis: every operand and product is exact,
the first non-bit item is the carried accumulator, and ordinary IEEE zero
addition closes 1,314 before bits and 4,893 after bits without changing a
nonzero value.

Southwest and southeast U refute the frozen same-boundary prediction. Their
first non-bit item is the neighboring V mask, not the recurrence: each has 68
magnitude-unequal cells on the southern row at level 0. Applying NEMO's
constant-zero southern V-mask fill closes those 68 masks and products on each
path; the already registered IEEE-zero addition then closes the remaining
recurrence bits. The combined measurement arm is bit-exact for every recorded
NE/SW/SE field.

The first remaining unmeasured source coverage is the four V recurrences. A
shared arithmetic landing remains held until all eight EEN coefficient paths
are proven and the complete trajectory gates pass.

## Round-116 record admission

The existing target
`orca2_rounds/round116/acquisition/orca2_rung0_een_u_recurrence_ranked_10step_np2`
admits with `STATUS PASS_R116_EEN_U_ADMISSION`:

- both ranks cover exactly 423,817 executed cells for each of NE, SW, and SE;
- recorded product, executed recurrence, and inherited terminal accumulator
  replay are bit-exact on both ranks;
- all twenty kt=1..10 restart shards and all inherited streams are byte-exact;
- all twelve runtime plants fire;
- rank-0 and rank-1 record SHA-256 values are respectively
  `f6216e270d1669c6ab6c50e65041e9109ac4dde3a14beec20d5ed4160330c308`
  and `1318f6348f86a612e42e0052fe41616b1558bf106ff5cbeedc305115041fb242`.

Thus the instrument is observationally passive and its self-describing
two-rank stream is admissible before any recurrence number is interpreted.

## Compiled source statement

The record-producing compiled branch evaluates NE, SW, and SE in source order.
Each statement adds the product of live U thickness, neighboring V thickness,
neighboring V mask, and its `zpvo` coefficient to the carried accumulator
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dynspg_ts.f90:1274-1288`).

The mask owner is earlier. NEMO constructs `vmask` from adjacent T masks and
passes it through the V-grid lateral-boundary exchange
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/dommsk.f90:211-232`). The compiled
exchange sets the default land value to zero
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:1816-1820`), selects
constant fill when the side has neither an MPI receiver nor self-periodicity
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:1864-1872`), and writes
that value into the halo
(`ORCA2_OMIP_L4_R116EENUREC/BLD/ppsrc/nemo/lbclnk.f90:2130-2136`). This is the
first non-bit compiled statement for the two southern U paths.

## Source-ordered measurements

The table separates unequal bit patterns from unequal numeric magnitudes.
Every candidate column is zero for all recorded fields, including `mbku`,
`zpvo`, `e3u`, `e3v`, mask, stored product, accumulator before, and accumulator
after.

| path | baseline first non-bit item | mask bits / magnitude | product bits / magnitude | before bits / magnitude | after bits / magnitude | candidate unequal |
|---|---:|---:|---:|---:|---:|---:|
| NE | before | 0 / 0 | 0 / 0 | 1,314 / 0 | 4,893 / 0 | 0 |
| SW | mask | 68 / 68 | 68 / 68 | 3,171 / 0 | 6,927 / 68 | 0 |
| SE | mask | 68 / 68 | 68 / 68 | 3,350 / 0 | 7,106 / 68 | 0 |

The first SW mask difference is global `(j,i,k)=(0,29,0)`; the first SE mask
difference is `(0,28,0)`. All 136 mask/product magnitude movements are confined
to `j=0, k=0`. After those registered mask associations, every further
movement is an exact-zero sign bit. The gate locks this complete census, so a
future record or implementation drift cannot silently change the claimed
boundary.

The resolved-card census says the southern EEN association executes on ORCA2,
the two DINO recipes, VORTEX, and VORTEX_VEC. GYRE and the two tank cards do
not execute it. This is a scope measurement only, not authorization to land
the shared helper.

## Frozen prediction disposition

| prediction | disposition |
|---|---|
| R117-P1: round-116 instrument passive and complete | **CONFIRMED** by restart/inherited-stream identity, exact replay, exact coverage, and twelve firing plants |
| R117-P2: NE first differs only in the carried recurrence | **CONFIRMED**; operands and product exact, 1,314/4,893 signed-zero-only before/after bits |
| R117-P3: host IEEE-zero arm closes NE | **CONFIRMED**; candidate exact and no magnitude movement |
| R117-P4: SW/SE share NE's first boundary | **REFUTED**; each first differs in 68 southern-row mask magnitudes |
| R117-P5: no U-only production landing | **CONFIRMED**; no `packages/` change |
| R117-P6: NEMO zero-fill closes the 68 SW/SE masks and products | **CONFIRMED**; candidate exact, no off-row movement |
| R117-P7: zero addition then closes the SW/SE recurrences | **CONFIRMED**; every recorded field exact |
| R117-P8: four V recurrences are next missing coverage | **CONFIRMED**; no admitted V recurrence stream exists |

## Mechanical gates, tests, and review

The committed round-117 gate reports
`STATUS MEASURED_R117_EEN_U_RECURRENCES`. Its oracle-bit, candidate-bit, and
scope-route plants each refuse. The citation gate passes both its default
receipt and this receipt with no unmapped citations; shifting the compiled
recurrence citation by two lines makes it refuse. The focused
round-107/109/110/114/115/116/117 chain passes 21/21.

The single required `tests/ocean/fidelity -n 12` invocation collected 2,350
tests and reached 98%, then reproduced the registered xdist-controller stall;
it was interrupted after more than a minute without output and therefore is
**incomplete, not PASS**. Re-running only the six displayed failing IDs
reproduced the same six registered pre-existing reds: SI3 scalar-math source
provenance, round-51 private-arm scope, round-35 escape scoping, worktree
stamping, the round-129 certified-year fixture, and the `hires_lane_surface`
case-board row. No new round-117 test failure was exposed.

The separate read-only Codex review was attempted and returned
**independent review unavailable in-sandbox**: `failed to initialize
in-process app-server client: Read-only file system`.

No model or card file changed, so ORCA2 rung-0/rung-7, GYRE, DINO, tank, and
generic-card trajectories cannot move and are not represented as rerun gates.

## OPEN

1. Preregister and acquire a self-describing, rank-complete record for the
   four V EEN recurrences with the same product, recurrence, terminal, restart,
   and inherited-stream controls.
2. Walk northwest, northeast, southwest, and southeast V in compiled order.
   The first non-bit item owns the next round; do not extrapolate the U result.
3. If all eight paths close, land the southern mask association and IEEE-zero
   recurrence semantics together under the full ORCA2/GYRE/DINO/tank/generic
   gates. Otherwise retain this U result and follow the earlier V owner.
4. Resume the northern-V cancelling pair and the later 68-cell substep-2 U
   residual only after this coefficient path is complete. The package-exposed
   rung-0 card and independent 240-step month remain open hierarchy work.

## UNVERIFIED

- The first non-bit statement and arithmetic sufficiency of all four V
  recurrences.
- Whether the combined shared implementation passes the full landing gates;
  no production implementation was attempted in this round.

## Choices

ASKED: Decisions 52, 80, 83, and 84 remain unchanged. UNASKED: none.
