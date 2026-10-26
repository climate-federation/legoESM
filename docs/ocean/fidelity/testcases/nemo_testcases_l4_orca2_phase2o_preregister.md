# NEMO testcase Lane 4 — ORCA2 Phase-2o preregistration

Date: 2026-09-06

Parent: `b6a6189c9357`

Status: **PREREGISTERED BEFORE BBL PAYLOAD SCORING OR NUMERICAL CHANGE.**
The user-reported twin census is an input, not a locally produced result.
All local measurements remain fail-closed.  Production JIT, CPU, binary64,
explicit scalar-libm policy, rank-zero owned cells, and Rules 8/11/12 remain
the frozen protocol.  No sandbox MPI/NEMO execution is authorized.

## P2O-A — diffusive-BBL record admission

Twin A is the proposed `VARIANT_ORACLE_V2` extension and twin B is its
independent reproducibility witness.  Admission requires both launches to
report `MPIRUN_RC=0`, `RUN DONE`, and `time.step=10`; identical 98-file
inventories; and 98/98 raw-byte-identical oracle records.  Relative to the
Phase-2m V2 root, all 97 inherited records must be raw identical in this
realization.  The four restart shards must be byte-identical to the accepted
uninstrumented variant control.  The eight history data-variable payloads
must be raw equal after only the registered global `TimeStamp` exclusion.
Every count is computed by the validator, never printed as a literal.

`oracle_bbl_diffusive_kt00000001.bin` must have magic
`NEMO_L4_BBLDF_1`, the 13-int header frozen in the Phase-2n acquisition
preregistration, and derived payload count
`6*(jpi*jpj*jpk)+2*(jpi*jpj)`.  Its six T-grid volumes and two U/V fields
must be finite on source-owned cells and exactly zero in the writer's
canonical halo/land/inactive slots.  Header-count, owned-payload,
canonical-zero, twin-identity, ordinary-output, and history-payload plants
must each make their real validator path exit nonzero.

On admission, twin A is pinned as the V2 extension and twin B as its witness;
the pin manifests enumerate and hash every oracle record, restart shard, and
`ocean.output`.  No multi-megabyte payload is committed to Git.

## P2O-B — selected BBL arm and ordered alignment

The icebergs-off ORCA2 variant resolves `ln_trabbl=.true.`,
`nn_bbl_ldf=1`, `nn_bbl_adv=0`, and `rn_ahtbbl=1000 m2/s`
(`ORCA2_ICE_PISCES/EXPREF/namelist_cfg:268-271`; selector dispatch
`trabbl.F90:118-138`).  Thus only the diffusive arm is walked; there is no
second advective-BBL walk on this deck.  The repository is searched first for
an existing BBL implementation.  Any addition extends the one shared BBL
module and is selected by the NEMO option name; there is no ORCA2 fork.

The source-statement ladder is frozen as follows.  Every arithmetic statement
uses `nemo_source_round` before its next NEMO source-level consumer.

| row | NEMO source and time level | target / disposition |
|---|---|---|
| B0 | `trabbl.F90:118-127`, `:356-380`; `Kbb=1`, `Kmm=2`, `Krhs=3` | selectors exact |
| B1 | `tra_bbl_init`, `trabbl.F90:507-537` | `mbkt`, slope signs, BBL face thickness and `ahu_bbl_0/ahv_bbl_0` |
| B2 | `bbl`, `trabbl.F90:342-380` | bottom Kbb T/S, Kmm depth, EOS-80 alpha/beta, gated `ahu_bbl/ahv_bbl` exact |
| B3 | `tra_bbl_dif`, `trabbl.F90:187-190` | bottom Kbb tracer gather exact |
| B4 | `tra_bbl_dif`, `trabbl.F90:192-200` | each face difference, grouped U/V flux divergence, live Kmm `e3t`, and Krhs increment exact |
| B5 | stage-3 production dispatch | selected shared option executes once at the NEMO placement and leaves nonselecting cards unchanged |

The primary target is **0 / n unequal binary64 values** for `ahu_bbl`,
`ahv_bbl`, and post-BBL T/S Krhs on every rank-zero source-owned cell covered
by complete operands.  If a record omits a required halo operand, the gate
must name the exact first unavailable statement/cell and fail closed rather
than silently shrinking the claimed domain.  One-variable plants perturb a
nonzero coefficient, one pre-BBL bottom tracer, and one post-BBL Krhs value;
each must turn an exact row into nonzero exit.

The direct cross-card Rule-12 table is GYRE, LOCK_EXCHANGE, and OVERFLOW.
Resolved namelists must be cited.  All three are expected to select
`nn_bbl_ldf=0`, so the new diffusive branch must cause 0 ULP/state movement.
OVERFLOW's distinct `nn_bbl_adv=2` arm is explicitly not called “no BBL” and
must remain byte/ULP unchanged.  A faithful exact ORCA2 row stays even if a
downstream metric worsens; that movement becomes a named compensating debt.

## P2O-C — TKE/EVD/IWM entry audit

Only after BBL admission and the BBL walk, inspect the resolved stage-3
`zdf_phy` ordering and selectors in `zdfphy.F90`, `zdftke.F90`, `zdfevd.F90`,
and `zdfiwm.F90`.  Existing records are inventoried against every operand of
the first executed statement.  If they suffice, score the entry at 0 / n with
oracle-supplied upstream inputs.  If they do not, stop the numerical walk and
list the exact missing fields, time levels, source statements, and proposed
canonical WRITE-only frames.  Do not build a new instrument this round merely
to fill that gap.  Shared TKE/ZDF debt is routed to GYRE; only ORCA2-specific
IWM/geothermal selection or input handling is Lane-4-owned.

## ASKED / UNASKED

| action | status | disposition |
|---|---|---|
| admit and pin BBL twins | ASKED | P2O-A, fail closed |
| implement selected diffusive BBL | ASKED, ORCA2 owner | one shared selectable implementation |
| advective BBL on ORCA2 | UNASKED / inactive | `nn_bbl_adv=0`; no second walk |
| preserve OVERFLOW advective arm | ASKED by Rule 12 | exact no-movement control |
| enter TKE/EVD/IWM audit | ASKED | inspect first; no speculative dump build |
| repair shared TKE/ZDF | forbidden by ownership | register and route to GYRE |
| SI3 operators | pending external merge | remain `ORACLE_SUPPLIED / UNMEASURED_PENDING_ICE_MERGE` |
| sandbox MPI/NEMO execution | forbidden | none |
| shipped NEMO edit, deletion, or push | forbidden | none |
