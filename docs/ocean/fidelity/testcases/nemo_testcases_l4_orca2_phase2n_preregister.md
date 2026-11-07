# NEMO testcase Lane 4 — ORCA2 Phase-2n preregistration

Date: 2026-09-06

Parent: `c862c49f85f56b9ce5951024083dd26ff5d776b0`

Status: **PREREGISTERED BEFORE LOCAL PAYLOAD SCORING.**  The user has reported
the two completed Phase-2m twins and their raw census; those statements are
inputs, not measurements produced by this gate.  The local checks below remain
fail-closed.  Production JIT, CPU, binary64, explicit scalar-libm, Rule 8/11/12,
and the rank-zero owned-cell convention are unchanged.

## P2N-A — oracle-extension admission

Twin A is the proposed `VARIANT_ORACLE_V2` extension and twin B is its
independent reproducibility witness.  Admission requires both complete at
`time.step=10` with `MPIRUN_RC=0`, identical 97-file inventories, and 97/97
raw-byte-identical oracle records.

Relative to the Phase-2j V2 baseline, every same-schema record must be raw
identical except the already source-proven pre-consumer workspace slot in
`oracle_transport_kt00000001_s1.bin`.  That record's schema is the 48-byte
`NEMO_L1_TRANSP_1` header followed by full-domain `zFu`, `zFv`, and `zFw`
arrays, each `jpi*jpj*jpk` binary64 values.  The executing vector form writes
this record before `tra_adv_trp` defines `zFw`; therefore:

- header, full owned `zFu`, and full owned `zFv` are consumed and must be
  bit-exact;
- `zFw` is explicitly `UNINFORMATIVE_PRE_CONSUMER_WORKSPACE` for this record;
- any differing bit in the header, owned `zFu`, or owned `zFv` voids the
  WRITE-only claim; and
- a binding plant in owned `zFu` must make admission exit nonzero.

The other 92 inherited records remain raw-identity rows.  The four ocean/SI3
restart shards must be raw identical to the accepted uninstrumented variant
control; the eight history data-variable payloads must be raw equal after only
the registered global `TimeStamp` exclusion.  Counts are reported from loops,
not literals.  The Phase-2m prediction of 93/93 inherited raw identity is
retracted under Rule 11 if and only if the sole exception above passes.

## P2N-B — EEN primitive discriminator

The four new streams are schema-walked first.  Then the ordered primitive rows
are scored on source-defined rank-zero cells, with target `0 / n` differing
binary64 cells:

1. `e3f_0vor`, after `dynvor.F90:918-950` including the F-fold exchange and
   zero fallback;
2. live `e3f_vor=e3f_0vor*(1+r3f*fe3mask)`, with `r3f` from
   `domqco.F90:233-246` and substitution at
   `domzgr_substitute.h90:125-130`;
3. `q=ff_f/e3f_vor`, the individual divisions used by the executing
   `dynspg_ts.F90:1329-1369` EEN arm; and
4. the eight U/V `zpvo` triads at `dynspg_ts.F90:1331-1342,1358-1369`.

Two candidates are reported separately: the shared production implementation
at this branch tip, and the GYRE Round-21 one-variable live-divisor arm.  The
latter is reconstructed only from the uncommitted diff in the explicitly
flagged `/tmp/codex-orca2-r21` worktree; no state, binary, or result is borrowed
from that tree.  Each first unequal cell is classified as north-fold row,
partial-cell bottom/coast, or interior from the oracle masks.  A one-ULP plant
in each candidate path must exit nonzero.  No EEN change lands in Lane 4:
shared external-mode debt is `GYRE_OWNER_SHARED_EXTERNAL_MODE`; only a proven
ORCA2 input/fold-construction defect may be fixed here after cross-card checks.

## P2N-C — Phase-2l review discriminators

The tripolar U-face shift at `4b7eb88ee73` is evaluated as one variable:

- current GYRE, LOCK_EXCHANGE, and OVERFLOW kt=1 gates must remain 0 ULP;
- an existing tripolar test/driver path is run on the pre-change and current
  implementation under the same inputs;
- changed `dx_u`, `dy_u`, `cos_alpha_u`, and `sin_alpha_u` values and the first
  affected consumer are enumerated; and
- impact registration covers `run_omip.py`, `run_omip_core2.py`,
  `run_tripole_20yr.py`, `run_coupled.py`, and the ORCA1 deck.  Every prior
  tripolar run using the shifted metrics is a user-facing provenance finding,
  not silently grandfathered evidence.

The runoff Rule-12 OVERFLOW row is retried with the same pre/post source and a
raised persistent JIT compilation budget.  If resource pressure persists, a
reduced-level arm must run both pre and post at the identical protocol.  Exact
`0 ULP` is required; otherwise the runoff claim is downgraded from 3/3 to 2/3.

The Phase-2l tracer gate gains localization only: per-level unequal counts,
bottom-versus-interior counts, and per-horizontal-row unequal fractions for T
and S.  The existing values, mask, operator, and scorer remain unchanged; a
plant must still fail.  Both shared tracer and transport reproducer handoffs
are routed to the GYRE lane without a Lane-4 shared-operator repair.

## P2N-D — continuation boundary

Only after A-C are recorded does the walk enter `trabbl.F90`, followed by the
`zdftke.F90` / `zdfevd.F90` / `zdfiwm.F90` entry.  NEMO operands are supplied
from the admitted variant oracle.  BBL/IWM/geothermal/runoff/north-fold debt is
Lane-4-owned; shared FCT/TKE/ZDF debt is registered and handed to GYRE.  SI3
remains `ORACLE_SUPPLIED / UNMEASURED_PENDING_ICE_MERGE`.  The first over-bar
boundary needing a decision stops the round.

## ASKED / UNASKED

| action | status | disposition |
|---|---|---|
| consumed-field admission exception | ASKED | P2N-A; fail on any consumed-bit change |
| EEN primitive walk and GYRE handoff | ASKED | P2N-B; no shared repair here |
| read `/tmp/codex-orca2-r21` | ASKED, diff only | only `git diff` may describe the arm |
| tripolar cross-card provenance | ASKED | P2N-C |
| retry OVERFLOW Rule 12 | ASKED | same-protocol pre/post, 0 ULP |
| tracer localization | ASKED | reporting-only gate extension |
| repair shared tracer/transport/EEN | forbidden by ownership | route to GYRE |
| sandbox MPI/NEMO execution | forbidden | none |
| shipped NEMO edits or deletion | forbidden | none |
