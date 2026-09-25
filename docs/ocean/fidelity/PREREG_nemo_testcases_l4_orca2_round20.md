# NEMO testcase Lane 4 — ORCA2 card round 20 preregistration

Date: 2026-09-25

Parent: `475d87e8c6f5ec5ba8809a8c13c71e831224a067`

Status: **PREREGISTERED BEFORE ROUND-20 SCIENTIFIC SCORING.**

The operator completed round 19's ranked pre-exchange acquisition at
`/data/abyssal/dbalwada/nemo-testcases-l2/phase3/orca2_rounds/round19/acquisition/orca1ice_u_preexchange_ranked_np2`.
Only its existence, filenames, byte counts, run layout, and compiled source
were inspected before this preregistration; no ocean value from the new stream
has been compared.  Every solver number in this round is labelled **given
NEMO's entry**.  The whole-card ladder remains separately labelled
**independent with Decision-52 SSH**.

The six sea-ice selectors and the card's `unmeasured_features` tuple are
frozen.  Decisions 54, 57, and 58 remain pending and untouched.  No
configuration choice, carried-state change, stabilizer, or sea-ice change is
allowed.

## Compiled branch and exact mapping to test

The executing build resolves `nn_comm = 1`, `jpni = 2`, `jpnj = 1`, and two
94 by 152 rank-local arrays with a two-cell halo.  The solver writes `ua_e`
immediately before its exchange and calls `lbc_lnk(..., ldfull=.TRUE.)` at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/dynspg_ts.f90:766-786`.
The compiled double-precision dispatcher selects the point-to-point arm for
`nn_comm <= 1` at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:623-627`.
That arm defines west/east send and receive offsets at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:1889-1909`, packs
the east/west send buffers without arithmetic at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:1960-1979`, and
assigns received values directly into the halo at
`ORCA2_ORCA1ICE_OMIP_L4_R19PREX/BLD/ppsrc/nemo/lbclnk.f90:2055-2073`.

For the rank layout recorded in `layout.dat`, rank 0's canonical west U face
(Fortran local `i=2`, legoESM global face 0) is therefore rank 1's eastern
send source (Fortran local `i=92`, legoESM duplicate global face 180).  The
round will test that mapping from the direct pre/post records rather than infer
it from periodicity.

## Frozen predictions and falsifiers

| ID | prediction | confirms | refutes |
|---|---|---|---|
| R20-P1 | The operator-completed run admits without repair or rerun. | Both ranked pre-exchange streams are exactly 228,660 bytes, carry distinct rank headers and substeps 1 and 2, have distinct digests, and all completion and split-marker checks pass. | Any wrong byte count, header, rank, substep, completion stamp, marker, or identical ranked payload. Stop for record reconciliation. |
| R20-P2 | The inherited round-19 boundary reproduces before the new stream is used. | The first live non-bit boundary remains substep-1 post-exchange U on 64 rank-0 west-halo faces, maximum `1.2223159767330016e-15`; midpoint/vector replays and history rotation remain bit-exact. | Any changed order, support, magnitude, or replay. Stop for instrument drift. |
| R20-P3 | The compiled MPI exchange is a bit-preserving copy on the disputed face. | For both substeps and every scored row, rank-0 local `i=2` after exchange is bit-identical to rank-1 local `i=92` before exchange; rank-0 local `i=2` before exchange is overwritten wherever it differs. | Any post-exchange value that is not the cited rank-1 source. The first failing mapping statement owns the walk; no production change lands. |
| R20-P4 | The exchange itself is not the arithmetic owner; legoESM already computes NEMO's chosen east-seam source exactly but retains a separately evaluated west duplicate. | On the 64 inherited rows, candidate global east U face 180 is bit-identical to rank-1 local `i=92` before exchange and rank-0 local `i=2` after exchange, while candidate global west face 0 differs from that source by the inherited maximum. | Candidate east differs from the rank-1 source, or candidate west already equals it. Walk the rank-1 source operands in compiled order and name the first earlier non-bit statement instead. |
| R20-P5 | The recorded rank-1 source replays the compiled midpoint and vector update bit-exactly. | On the source face, coefficients, histories, midpoint, `rDt_e`, live `zu_spg`, `zu_trd`, interior `zu_frc`, and `ssumask` are bit-exact through the first mismatch; both recorded and candidate expression replays equal their targets. | The first earlier non-bit operand or replay owns the walk. Later seam predictions are not promoted. |
| R20-P6 | Copying the chosen east duplicate to the west duplicate after both velocity updates is sufficient and minimal. | A one-statement arm `west U := east U`, placed at NEMO's post-update exchange boundary, makes the round-19 U boundary and all 64 inherited substep-2 continuity cells bit-exact without moving any earlier registered row; a west-from-neighbour plant and a one-ULP source plant both fire. | Any residual cell, earlier moved row, V/SSH dependency, or need for a selector/configuration choice. No model statement lands. |
| R20-P7 | If R20-P6 holds, the statement is eligible only under the full landing bar. | The ORCA2 kt=1..10 ladder has no AT-BAR row leave the bar, first-over-bar does not move earlier, every moved row is registered, and base-vs-tip GYRE has 0 differing rows, `np.array_equal` residual ladders, and byte-identical 30-day snapshots. | Any gate regression, unregistered movement, GYRE movement, or shared-card statement requiring a pending choice. Hold or report `DECISION_NEEDED`; do not land. |

Failed predictions remain **REFUTED** in the receipt and are never rewritten
after measurement.  The walk stops at the first non-bit statement in compiled
source order.

## Controls and stop rules

- Admission uses the existing run only.  No rebuild, rerun, or artifact repair
  is allowed.
- Every rank-local/global index map is asserted from the two-rank layout and
  the compiled `ihls=2` offset statements.
- Direct pre/post exchange identity is checked on both substeps and both halo
  layers; the scientific score is the live `i=2` U face used by the canonical
  rank-0 window.
- A swapped-rank header, a wrong neighbor face, and a one-representable-value
  source mutation must each make their checks fail.
- Any model edit must be one cited shared statement, JIT/autodiff-safe, with no
  card-specific physics selector.  It must pass both the ORCA2 landing ladder
  and the required GYRE base/tip trajectory and 30-day identity checks.
- The receipt citation gate must pass, and a rigid two-line shift of every
  rendered compiled citation must fail.
- No action is permitted on Decisions 54, 57, or 58, the northern-fold
  mask/wind debt, or the sea-ice-owned initial SSH.

## Choices

ASKED: admit the completed pre-exchange record, map the neighbor source through
the compiled exchange, and either land one mechanically proven seam statement
or stop at the first earlier non-bit operand.

UNASKED: none.
