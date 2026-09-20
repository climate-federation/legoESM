# NEMO testcase Lane 4 — ORCA2 Phase-2p handoff receipt

Date: 2026-09-06

Starting parent: `df39aea005fb`

Status: **PARTIAL-CELL EEN FIXED; STOP FOR USER-SHELL ZDF TWINS.**  The
Phase-2n ownership error at `(50,2,21)` is corrected and the ORCA2 partial-cell
EEN operand walk closes at the bar.  A WRITE-only, canonical, rank-zero ZDF
entry stream is built and staged in two independent run directories.  No
MPI/NEMO command was run in the sandbox, no shipped NEMO file was changed,
and no artifact was deleted or pushed.

### Phase-2q disposition of the ZDF acquisition prediction

**CONFIRMED / REFUTED PREDICTION (2026-09-06):** both Phase-2p twins ran
successfully and were 95/95 raw-identical, but
`oracle_zdf_sh2_operands_kt00000001.bin` is not schema-valid.  Its header
claims 9,358,640 binary64 payload values while exact EOF contains 9,267,648.
The 90,992-value difference is exactly three full-versus-A2D 3-D allocations
(`sh2`, `avt_k`, `en`) plus one full-versus-A2D 2-D allocation (`taum`):
`3*(94*152-90*148)*31 + (94*152-90*148)`.  Thus the 99-record/schema-admission
prediction in section 3 below is retracted in the validating tool, not merely
in prose.  These two successful runs are preserved as
`REJECTED_MALFORMED_ZDF_HEADER`, not pinned as a V2 extension.  A Phase-2q
WRITE-only replacement derives the header from the same `SIZE(...)`
expressions as the write list.  The four absent EEN streams were also an
unintended build-series omission; their accepted pins remain the Phase-2m
twin-A records.

All numerical claims below are **CONFIRMED** by the committed production-JIT
gate or retained build artifacts.  Source/ownership claims marked
**CONFIRMED** have the named executed NEMO statement and discriminator.
Twin-admission expectations and subsequent TKE/EVD/IWM arithmetic are
**PLAUSIBLE** until the user-shell runs exist.  There are no intentionally
unlabelled claims.

## 1. Ownership correction and partial-cell localization

The review finding is **CONFIRMED**: NEMO's `bottom_level` at one-based
`(i,j)=(50,2)` is 21, making `(50,2,21)` a wet partial-cell bottom on the
southernmost wet row (28 wet columns of 180).  GYRE resolves full steps
(`l_zps=.false.`), so that card cannot reproduce this arm.  The row is now
`ORCA2_OWNER_PARTIAL_CELL_EEN`; “partial-cell bottom” is an explicit Lane-4
operator class beside the north fold and BBL.

The Phase-2n arm reproduces **10,160 / 399,600** unequal live-`e3f_vor` cells.
Their mutually exclusive classes are: 3,117 partial-cell bottoms, 1,285
interior wet cells, 710 north-fold-row cells, and 5,048 coast/land loop cells.
Per-level unequal counts (surface to bottom) are:

`34,34,34,89,110,129,126,125,124,134,141,155,166,173,180,185,207,214,268,329,350,391,456,561,639,748,968,1114,1073,903`.

Per-row unequal counts (global south to north, all 148 rows) are:

`0,4,12,21,47,62,47,34,52,57,50,56,59,51,46,37,40,28,50,62,67,145,116,73,54,52,49,41,35,44,61,68,64,69,62,63,67,75,59,70,52,64,58,87,53,77,93,100,104,127,74,75,72,85,85,132,102,119,111,116,166,88,81,94,82,71,37,39,71,60,63,67,48,57,53,66,78,66,71,69,76,110,92,111,96,116,66,39,59,56,70,111,78,83,64,52,51,50,56,35,40,29,23,46,21,34,35,51,39,46,128,123,134,116,107,113,107,79,27,7,9,0,0,1,6,2,18,18,21,134,91,62,78,54,64,103,121,50,53,55,60,42,57,44,46,63,60,710`.

The machine-readable vectors and first-cell classifications are in
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2p/e3f_partial_cell_gate.json`
(SHA-256 `9b0c243e7398885f5d29de6e6c48a160f561e4b7387cde684b4fe84a2ea0ddb1`).

## 2. First departing statement and repair

The first departure given NEMO's inputs is **CONFIRMED** at the live-thickness
statement `e3f_vor = e3f_0vor * (1 + r3f * fe3mask)` in
`domzgr_substitute.h90:130`.  NEMO forms the four-surrounding-T free-slip
`fmask` in `dommsk.F90:146-174`, applies its F boundary at `:190`, and freezes
that value into `fe3mask` at `:195-198`; only afterward do lateral-slip and
strait edits change the ordinary vorticity `fmask` at `:207-243`.  legoESM
had silently reused the later `fmask` for live QCO thickness.  At `(50,2,21)`
that value is 2 while `fe3mask` is 0, so the wrong stretch starts exactly at
the reviewed partial-cell bottom.

The single shared QCO EEN operand builder now carries `fe3mask` separately,
uses it only at the cited multiplication, and applies NEMO's F-point lateral
boundary to live `r3f` (`domqco.F90:124-135`) after its QCO construction
(`domqco.F90:131,172-177` and `:233-246`).  Reference `e3f_0vor` continues to
follow `dynvor.F90:918-950`, including the `WHERE(...==0)` fallback at
`:947-950`.  The ordinary vorticity `fmask` and non-EEN consumers are
unchanged.  This is one shared implementation with NEMO-named operands, not
an ORCA2 fork.

The corrected production arm scores:

| operand | unequal / n | disposition |
|---|---:|---|
| `e3f_0vor` | **0 / 399,600** | CONFIRMED AT-BAR |
| live `e3f_vor` | **0 / 399,600** | CONFIRMED AT-BAR |
| `q=ff_f/e3f_vor` | **0 / 399,600** | CONFIRMED AT-BAR |
| U NW/NE/SW/SE q triads | **0 / 230,898** each | CONFIRMED AT-BAR |
| V NW/NE q triads | **0 / 230,645** each | CONFIRMED AT-BAR |
| V SW/SE q triads | **0 / 231,278** each | CONFIRMED AT-BAR |

The rank-zero stream does not contain the south halo needed by U triads or
V south-reading triads, nor the north halo needed by V north-reading triads;
those boundary slots are explicitly excluded.  Every recorded complete
stencil is scored.  Thus the ORCA2 partial-cell/fold contribution is landed;
there is no residual shared-association change for this lane to take from the
GYRE Round-21 arm.  The existing eight streams are pre-coefficient EEN q
triads, not an unsupported claim about unrecorded integrated arrays.

Rule 12 is **CONFIRMED**: the unchanged GYRE ENE production path is
**0 / 21,120** old-versus-new; LOCK_EXCHANGE and OVERFLOW select AL81, making
this EEN helper unreachable with zero movement.  The production scorer's
one-bit plant exits 1.  The focused unit rows report 21 passed; the full
50-test affected set previously exposed and then closed the synthetic-mask
fixture error.

## 3. ZDF WRITE-only acquisition frame

NEMO's whole-step RK3 entry calls `zdf_phy` before the three RK stages
(`stprk3.F90:150-165`).  Inside it, `zdf_sh2` executes first and creates the
shear operand (`zdfphy.F90:264-270`; `zdfsh2.F90:80-100`) before the selected
TKE closure.  `dyn_zdf`/`tra_zdf` execute later at RK stage 3; the acquisition
frame is therefore correctly labelled whole-step entry, not “stage-3
`zdf_phy`.”

The config-local override writes exactly one new stream on rank zero:
`oracle_zdf_sh2_operands_kt00000001.bin`.  Magic is `NEMO_L4_ZSH2_1`; its
13 integers are version, `kt`, `Kbb`, `Kmm`, `Krhs`, `jpi`, `jpj`, `jpk`, real
storage bits, 3-D field count, 2-D field count, derived payload count, and
`l_zdfsh2`.  The payload expression is
`21*jpi*jpj*jpk + 4*jpi*jpj`.

The 21 3-D fields, in order, are `sh2`, pre-closure `avm_k`, `avt_k`, `en`,
`rn2`, `rn2b`, U at Kbb/Kmm, V at Kbb/Kmm, `e3uw` at Kbb/Kmm, `e3vw` at
Kbb/Kmm, `umask`, `vmask`, `wumask`, `wvmask`, `gdepw(Kmm)`, `e3t(Kmm)`, and
`e3w(Kmm)`.  The four 2-D fields are `taum`, `fr_i`, `rCdU_bot`, and real-valued
`mbkt`.  Time-level meaning is therefore explicit for every time-dependent
array; `sh2` is the just-produced value and `avm_k/avt_k/en` are pre-closure.

All live substitution geometry is first copied into writer-owned allocated
temporaries initialized to zero.  A writer-local `l4_zdf_canon_3d` admits only
owned W/WU/WV cells; the existing canonical functions handle T/U/V and 2-D
fields.  Halo, land, below-bottom, and inactive slots stay zero.  Model arrays
are only read.  The BBL writer was also hardened to use its own
`l4_bbl_unit/l4_bbl_ios/l4_bbl_filename/l4_bbl_magic` locals instead of the
fragile Lane-1 names; this is visibility-only and WRITE-only.

The scalar-math build uses the accepted CPP key set (`key_si3 key_qco
key_vco_1d3d key_RK3`, no `key_top` or `key_xios`) and
`arch-conda-scalarmath.fcm` with `-fno-tree-vectorize`.  The final binary is
55,589,568 bytes, SHA-256
`5a864322cd696237dd1766cb48fd5f317c9724fa31818832f9279db7bed23861`;
`nm -D` reports **0 `_ZGV*` symbols**.

The committed acquisition gate derives the header, payload and EOF sizes,
validates finiteness and every grid-specific canonical slot, demands 99/99
raw-identical twin records and 98/98 raw-identical inherited records, and has
a binding first-halo-bit plant.  Those are **PLAUSIBLE/PREREGISTERED**, not
results, until both runs finish.  The campaign admission rule remains two
independent executions of the same binary with every record byte-identical.

## 4. Review-fix ledger and ownership register

The Phase-2n receipt is updated in place:

- **CONFIRMED:** runoff is “3/3 with the third at reduced scope (levels 1, 2,
  13, 25 of 101).”  The 101-level OVERFLOW retry remains on the backlog.
- **CONFIRMED:** GYRE Round 22 closed the shared CEN2 precursor and stage
  transports on Lane 4's gates at **0 / 228,641**; neither remains an open
  Lane-4 ownership row.
- **CONFIRMED:** `(50,2,21)` and the live EEN mask operand are Lane 4
  partial-cell/fold ownership.  Shared EEN association stays with GYRE only if
  a future complete-stencil discriminator finds a residual; none does here.
- **PLAUSIBLE:** TKE/EVD/IWM implementation remains unentered pending the new
  frame.  SI3 stays `ORACLE_SUPPLIED` and
  `UNMEASURED_PENDING_ICE_MERGE`.

## 5. ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| re-route and walk partial-cell EEN | ASKED | CONFIRMED fixed, exact on complete stencils |
| modify shared EEN association beyond the isolated operand | forbidden | not done |
| GYRE/LOCK/OVERFLOW Rule-12 rows | ASKED | CONFIRMED 0 movement or selector-unreachable |
| add canonical rank-zero ZDF frame | ASKED | built and preregistered |
| give BBL writer private locals | ASKED if touched | completed WRITE-only |
| user-shell twin execution | ASKED | two directories prepared below |
| sandbox MPI/NEMO execution | forbidden | none |
| enter TKE/EVD/IWM numerics | out of order | not done |
| shipped-tree edit, deletion, push | forbidden | none |

The GYRE probe `/tmp/codex-orca2-r21` remains flagged and was not used as a
source tree.  No state was read from it in this phase.

## Stop receipt

Execute these one at a time with their unchanged `run.sh`:

- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2p_zdf_a_10step_np2/run.sh`
- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2p_zdf_b_10step_np2/run.sh`

Each directory contains byte-copied accepted deck files, 40 absolute read-only
input symlinks, an absolute symlink to the hash-pinned binary, manifests, and
the self-contained two-rank Bash-time launcher.  Resume only for twin/schema/
identity admission and the ZDF entry walk.  This stop contains no legoESM ZDF,
TKE, EVD, IWM, or SI3 numerical change.
