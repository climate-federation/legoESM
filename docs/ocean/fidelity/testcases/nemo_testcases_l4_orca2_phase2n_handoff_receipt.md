# NEMO testcase Lane 4 — ORCA2 Phase-2n receipt and BBL handoff

Date: 2026-09-06

Starting parent: `c862c49f85f5`

Status: **PHASE 2 CONTINUES; STOP FOR USER-SHELL BBL TWINS.**  The Phase-2m
EEN twins are admitted and pinned as a reproducible `VARIANT_ORACLE_V2`
extension.  The ORCA2-owned EEN fold operand is repaired; the next EEN
departure is handed to GYRE.  The Phase-2l review rows are dispositioned.  The
ordered walk reaches ORCA2's selected diffusive BBL arm, for which a canonical
WRITE-only discriminator and two launch directories are prepared.  No
MPI/NEMO command was run in the sandbox, no shared BBL/TKE arithmetic was
changed, and TKE/EVD/IWM was not entered.

All scores below use production JIT on CPU, binary64, the explicit scalar-libm
policy, the oracle-relative cellwise bar, and the rank-zero owned-cell masks.
Every measurement was preregistered in
`nemo_testcases_l4_orca2_phase2n_preregister.md`; BBL acquisition has its own
pre-build preregistration.

Claim labels for this receipt are explicit and exhaustive: record admission,
hashes, schema checks, EEN/metric/runoff/tracer measurements, build facts,
namelist selectors, and prepared-directory facts are **CONFIRMED** by the
cited committed gates or retained artifacts.  Ownership assignments and
source-path diagnoses are **CONFIRMED** where a named source statement and
discriminator close them.  Forward-looking admission expectations and the
TKE/EVD/IWM instrument-needs list are **PLAUSIBLE** until their user-shell
runs exist.  No unlabeled numerical or causal claim is intended.

## 1. Phase-2m record admission and V2 pin

Both user-shell runs report `MPIRUN_RC=0`, `RUN DONE`, and `time.step=10`.
The admission gate schema-walks all four new EEN streams and reports:

| comparison | result | disposition |
|---|---:|---|
| twin A versus twin B record inventory | 97 / 97 raw exact | PASS |
| inherited records versus Phase-2j V2 | 92 raw exact | PASS |
| exceptional inherited transport record | 1 consumed-field exact | PASS |
| owned consumed `zFu` | 0 / 412,920 cells differ | PASS |
| owned consumed `zFv` | 0 / 412,920 cells differ | PASS |
| pre-consumer `zFw` | 1,583 elements / 11,100 owned bytes differ | `UNINFORMATIVE_PRE_CONSUMER_WORKSPACE` |
| restart shards | 4 exact | PASS |
| history data-variable payloads | 8 exact | PASS; registered global `TimeStamp` only excluded |

The exception is confined to `zFw` in
`oracle_transport_kt00000001_s1.bin`.  Vector-form stage 1 writes this stream
before `tra_adv_trp` initializes that workspace; the record readers never
consume the slot.  Header plus all owned `zFu/zFv` cells are exact.  Thus no
consumed byte moved and the WRITE-only admission stands.

The preregistered prediction that all 93 inherited records would be raw exact
is **REFUTED and retracted under Rule 11** by exactly this slot.  The gate now
encodes the GYRE Round-21 consumed-field admission rule rather than the false
raw prediction.  A plant in owned `zFu` traverses that same comparator and
exits nonzero.  Header, defined-payload, and canonical-zero plants also report
`PASS_NONZERO`.

Twin A is pinned as the `VARIANT_ORACLE_V2` extension; twin B is its independent
witness.  Each pin manifest contains the 97 records, four restart shards, and
`ocean.output` (102 entries).  The two manifest files are themselves identical,
SHA-256 `1a4357c55077f0e52f2fc9ddc038160aa0fb06995579e46be20d723155632d08`.
The complete record hashes live in those manifests under the Lane-4 data root;
no large record is placed in Git.

## 2. EEN discriminator and ownership

The gate validates the four stream headers, derived payload counts, EOF sizes,
source-defined cells, canonical zeros, and finiteness before arithmetic.  The
record hashes are:

| primitive stream | SHA-256 |
|---|---|
| `oracle_een_e3f0vor_kt00000001.bin` | `6239bfea69efc878e797229aa04f45b05c2c34622648d8e24d16de63c00056fb` |
| `oracle_een_e3fvor_kt00000001.bin` | `b50e58b3bea3b168a147d57fcaa499d0b0e20c2c05781821cef3c188aa61b1ad` |
| `oracle_een_q_kt00000001.bin` | `446768b7a5f6bdd6aae15d7dde6b87cfeb416290a90389700b5fc663e14d50bf` |
| `oracle_een_zpvo_kt00000001.bin` | `1124e1602ffc2ec39f8d2ec3504bfdc2b1d8885c092f5503c07d2560f1c99da5` |

NEMO constructs `e3f_0vor` in `dynvor.F90:918-950`, including the F-point
fold at `:943` and zero fallback at `:944-950`.  Live `r3f` is the weighted
statement in `domqco.F90:233-246`; `domzgr_substitute.h90:125-130` forms live
`e3f_vor`.  The executing external EEN `q` divisions and eight U/V `zpvo`
triads are in `dynspg_ts.F90:1326-1379`.

The unmodified production arm first fails in reference `e3f_0vor` (36,143 /
399,600).  The GYRE Round-21 live-divisor arm reduces this to 764 / 399,600,
all on the north fold, but remains DEBT.  Lane 4 added the source-literal NEMO
T-pivot F-fold to the one shared QCO live-vorticity-depth helper.  Closed-grid
controls remain byte-identical.  The ORCA fold-fixed arm gives:

| primitive | unequal / n | first NEMO `(i,j,k)` | class |
|---|---:|---|---|
| `e3f_0vor` | **0 / 399,600** | — | AT-BAR |
| live `e3f_vor` | 10,160 / 399,600 | `(50,2,21)` | partial-cell bottom |
| `q` | 10,160 / 399,600 | `(50,2,21)` | partial-cell bottom |
| U `zpvo` NW/NE/SW/SE | 9,891 / 9,938 / 10,016 / 9,836 of 230,988 | `(1,1,1)` | coast/land loop cell |
| V `zpvo` NW/NE | 10,249 / 10,279 of 231,368 | `(50,2,20)` / `(51,2,21)` | partial-cell bottom |
| V `zpvo` SW/SE | 10,279 / 10,221 of 231,368 | `(1,1,1)` | coast/land loop cell |

The Phase-2n first remaining primitive was live `e3f_vor` at `(50,2,21)`.
Review of `bottom_level` proves that point is a wet partial-cell bottom on the
southernmost wet row; GYRE has full steps and cannot reproduce it.  Phase 2p
therefore supersedes the original routing: this is
`ORCA2_OWNER_PARTIAL_CELL_EEN`, not `GYRE_OWNER_SHARED_EXTERNAL_MODE`.  The
Phase-2p discriminator localizes and walks it from `fe3mask` and the live QCO
geometry before deciding whether any residual association belongs to GYRE.
The Round-21 arm was reconstructed only from `git diff` of the explicitly
flagged `/tmp/codex-orca2-r21`; no code, build, or result was taken from that
worktree.  The production-path bit plant exits 1.

## 3. Phase-2l review dispositions

### Tripolar U-face layout: user-facing provenance finding

Commit `4b7eb88ee73` changed the native U-face extension from the wrong
right-appended cell to NEMO's left/prepended periodic cell.  On
`ORCA_R2_zps_domcfg.nc`, the before/after audit is:

| field | legacy shifted cells / n | maximum absolute move | corrected cells / n |
|---|---:|---:|---:|
| `dx_u` | 9,126 / 26,640 | 109,724.3627 m | **0 / 26,640** |
| `dy_u` | 8,935 / 26,640 | 164,474.2087 m | **0 / 26,640** |
| `cos_alpha_u` | 8,833 / 26,640 | 1.9992982 | **0 / 26,640** |
| `sin_alpha_u` | 8,838 / 26,640 | 2.0 | **0 / 26,640** |

The redundant periodic endpoint is exact after the repair; `dx_v`, `dy_v`,
and generic `f_v` remain 0 / 26,820.  The tripolar entry gate is unchanged and
exact.  The pre-change tripolar unit paths report 35 passed / 1 skipped; the
current paths report 36 passed / 1 skipped, the extra test being the EEN fold
row.  The metric plant exits 1.

This is a material provenance finding: every earlier tripolar result produced
through `run_omip.py`, `run_omip_core2.py`, `run_tripole_20yr.py`,
`run_coupled.py`, or the ORCA1 deck carried the shifted U metric and rotation.
Those results must be flagged and re-evaluated before reuse.  They are not
silently grandfathered.

For nontripolar cross-cards, the GYRE candidate artifact is byte-identical to
its pre-change artifact; LOCK compares nine certified rows with zero field
movement and zero oracle-residual worsening.  The full OVERFLOW stage gate was
retried with a larger wall-clock budget but again produced no completed JSON
during JIT compilation.  The registered reduced-level OVERFLOW kt=1 arm uses
the exact oracle transport/mesh and scores both pre/post programs at levels
1, 2, 13, and 25: each is **0 / 200**, including five nonzero operand cells per
level.  This is the reduced-level scoring alternative preregistered for the
resource case; the tripolar constructor is statically unreachable on all
three nontripolar cards.  No 101-level OVERFLOW result is inferred.

### Runoff Rule 12

The reduced OVERFLOW result closes the resource-blocked third row.  With
`runoff_mass_flux=None`, the pre-refactor inline statement and the one shared
current helper are 0 ULP on four representative wet levels.  Its binding plant
exits 1.  Together with zero-movement GYRE and LOCK results, the runoff claim is
**3/3 with the third at reduced scope (levels 1, 2, 13, 25 of 101)**.  The
101-level retry remains on the backlog; no full-column result is inferred.

### Tracer localization and shared routing

The reporting-only extension leaves the Phase-2l score unchanged:

| tracer | unequal / n | bottom unequal | other interior unequal | horizontal-row fraction range |
|---|---:|---:|---:|---:|
| T | 180,882 / 228,641 | 6,778 | 174,104 | 0.526749–0.927613 |
| S | 190,802 / 228,641 | 7,015 | 183,787 | 0.552124–0.968318 |

The JSON gives all 30 per-level counts and every row fraction.  A single-bit
plant yields exactly 1 / 228,641 and exits 1.  These two shared handoffs are
now **CONFIRMED closed by the GYRE lane in Round 22**: its CEN2 precursor and
stage transports score **0 / 228,641** on Lane 4's reproducer gates.  They are
not open Lane-4 ownership rows.

## 4. Ordered continuation: diffusive BBL acquisition

After substituting oracle FCT output, the next selected boundary is BBL.
ORCA2 resolves `ln_trabbl=.true.`, `nn_bbl_ldf=1`, `nn_bbl_adv=0`
(`namelist_cfg:268-271`; `trabbl.F90:118-138`).  legoESM's card currently has
`bbl_adv_option=0`; the existing shared implementation is only the distinct
advective option 2.  The missing selected diffusive arm is
`ORCA2_OWNER_DIFFUSIVE_BBL`.

The new config-copy writer brackets the ordinary stage-3 `tra_bbl` call and
records Kbb T/S, pre- and post-call Krhs T/S, and `ahu_bbl`/`ahv_bbl`.  All
eight arrays pass through the existing zero-first canonical helpers.  It does
not assign a model field.  Its 13-int header derives the payload count as
`6*(jpi*jpj*jpk)+2*(jpi*jpj)`; full schema and time-level semantics are frozen
in the BBL preregistration.

The scalar-math build completed in 17 s with `-fno-tree-vectorize`.  The staged
binary is 55,613,928 bytes, SHA-256
`392a2d59e25a01f2d85923b7726f7676eabfa195fb5f00ed04da87c45471dbdd`;
`nm -D` reports **0 `_ZGV*` symbols**.  Its only source delta from the admitted
Phase-2m instrument is the committed WRITE-only patch.

Prepared twin directories:

- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2n_bbl_a_10step_np2`
- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2n_bbl_b_10step_np2`

They contain byte copies of the accepted icebergs-off ten-step namelists/XML,
40 absolute read-only input symlinks, one absolute binary symlink, manifests,
and self-contained `run.sh`.  Each launcher refuses the wrong directory or
pre-existing output, re-hashes the binary, deck, and every input, sets all
thread counts to one, and runs `mpirun -np 2 --oversubscribe ./nemo` under the
Bash time keyword.  The `(jpni,jpnj)=(2,1)` layout remains the smallest layout
allowed by the oracle's initialization constraints.

Admission after execution requires 98 / 98 raw-identical records between
twins, the frozen BBL schema and canonical slots, exact ordinary outputs and
restart/history payloads, and identity of all 97 inherited records against the
pinned extension except the already admitted pre-consumer `zFw` slot.  Header,
owned-payload, canonical-zero, and consumed-field plants must exit nonzero.
Only then may Lane 4 implement and walk the source-literal diffusive BBL arm,
then enter TKE/EVD/IWM.

## 5. Coverage and time-level update

| boundary/state | disposition | time level |
|---|---|---|
| Phase-2m 97-stream extension | VERIFIED reproducible | inherited registry plus EEN `Kmm=1` |
| `e3f_0vor` | VERIFIED AT-BAR after fold | reference/static |
| live `e3f_vor`, `q`, eight `zpvo` | VERIFIED measured; Phase-2p partial-cell walk is Lane 4 | derived from `Kmm=1` |
| stage-1 FCT T/S output | ORACLE_SUPPLIED; shared FCT debt routed to GYRE | stage-1 Krhs/result |
| SI3 exchange | ORACLE_SUPPLIED | unchanged; `UNMEASURED_PENDING_ICE_MERGE` |
| diffusive BBL input/output stream | PREREGISTERED, awaiting twins | stage 3: Kbb, Kmm, Krhs |
| TKE/EVD/IWM entry | not entered | blocked behind BBL acquisition |

## 6. ASKED / UNASKED and retained state

| action | classification | disposition |
|---|---|---|
| consumed-field V2 admission | ASKED | passed; false raw prediction retracted |
| EEN primitive discriminator | ASKED | completed; four records handed to GYRE |
| ORCA2 F-fold repair | ASKED within ownership | landed with closed-grid control |
| shared live-EEN repair | forbidden by ownership | not attempted |
| tripolar U provenance audit | ASKED | user-facing impact registered |
| OVERFLOW retry/reduced arm | ASKED | full JIT resource-incomplete; reduced row exact |
| tracer localization | ASKED | added without changing scorer |
| route shared handoffs | ASKED | tracer/transport closed in GYRE Round 22; partial-cell EEN re-routed to Lane 4 |
| diffusive BBL acquisition | ASKED continuation | twins prepared; no arithmetic change |
| user-shell MPI execution | ASKED | these two directories are the only requested runs |
| sandbox MPI/NEMO execution | forbidden | none |
| enter TKE before BBL | out of order | not done |
| shipped NEMO edit, push, or deletion | forbidden | none |

The GYRE probe `/tmp/codex-orca2-r21`, the pre-U-face worktree
`/tmp/codex-orca2-uface-before`, and the clean gate worktree
`/tmp/codex-orca2-p2n-gates` are preserved and flagged, not deleted.  No state
was borrowed from the GYRE probe beyond its requested diff description.

## Stop receipt

Run the two BBL directories above, one at a time, with their unchanged
`run.sh`, then resume for twin/schema/identity admission and the source-literal
diffusive-BBL walk.  This stop contains no legoESM BBL or TKE numerics.
