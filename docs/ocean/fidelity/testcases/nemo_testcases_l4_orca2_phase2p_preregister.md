# NEMO testcase Lane 4 — ORCA2 Phase-2p preregistration

Date: 2026-09-06

Parent: `df39aea005fb`

Status: **PREREGISTERED BEFORE NEW EEN SCORING, NUMERICAL CHANGE, OR ZDF
BUILD.**  The independent-review ownership correction is an ASKED input.  All
new measurements use production JIT on CPU, binary64, explicit scalar-libm,
oracle-relative cellwise scoring, and the rank-zero reconstruction.  No
sandbox MPI/NEMO execution is authorized.

## P2P-A — partial-cell EEN ownership and statement ladder

The first retained live-`e3f_vor` departure is NEMO `(i,j,k)=(50,2,21)`.
`bottom_level[j=2,i=50]=21`, so it is a wet partial-cell bottom on the
southernmost wet row, not an inactive loop cell.  GYRE resolves full steps
(`l_zps=F`) and cannot exercise this arm.  The owner is corrected from
`GYRE_OWNER_SHARED_EXTERNAL_MODE` to **`ORCA2_OWNER_PARTIAL_CELL_EEN`**.
“Partial-cell bottom” is added to Lane 4's ORCA2-specific list beside north
fold and diffusive BBL.

Before changing arithmetic, the 10,160 unequal live-`e3f_vor` values are
localized by level, horizontal row, and mutually exclusive bottom/interior/
coast-or-land/fold classes.  The walk is frozen in source order:

| row | NEMO source | target |
|---|---|---|
| E0 | `dynvor.F90:918-950` | frozen `e3f_0vor`, including F-point fold and zero fallback |
| E1 | `domqco.F90:124-131,172-181,189-262` | selected RK3 `r3f` construction from the live SSH and reference column/area metrics |
| E2 | `domzgr_substitute.h90:48,130` | `e3f_vor=e3f_0vor*(1+r3f*fe3mask)` |
| E3 | `dynspg_ts.F90:1520-1569` | eight EEN `ffu/ffv` coefficient operands |

Each source statement is separated with `nemo_source_round`.  One-variable
arms vary only the first departing association or partial-cell/fold geometry.
The winning arm must give 0 / n on `e3f_0vor`, live `e3f_vor`, `q`, and all
eight `zpvo` streams before landing.  A one-bit oracle plant must exit
nonzero.

The implementation extends the one shared NEMO-QCO vorticity-thickness helper;
there is no ORCA2 code fork.  Rule 12 controls instantiate GYRE, LOCK_EXCHANGE,
and OVERFLOW and require 0 ULP movement.  GYRE/LOCK are full-step controls;
OVERFLOW is the separate partial-cell card and therefore the binding geometry
control.  If exact live thickness exposes a later shared scalar association in
the eight coefficients, the partial-cell fix stays and that later boundary is
routed to the shared external-mode owner.

## P2P-B — first ZDF discriminator

The existing ZDF stream is post-closure and cannot locate the first internal
statement.  This acquisition adds exactly one rank-zero stream immediately
after `zdf_sh2` and before `zdf_tke` in `zdf_phy`.  Under `key_RK3`, `zdf_phy`
is called once at whole-step entry with `(Kbb,Kmm,Krhs)=(1,1,3)`
(`stprk3.F90:150-165`); stage-3 `dyn_zdf`/`tra_zdf` are later consumers and are
not the dump site.

The stream `oracle_zdf_sh2_operands_kt00000001.bin` has:

1. 16-byte magic `NEMO_L4_ZSH2_1`;
2. 13 native int32 values: version, `kt`, `Kbb`, `Kmm`, `Krhs`, `jpi`, `jpj`,
   `jpk`, real storage bits, number of 3-D fields, number of 2-D fields,
   derived payload count, and `l_zdfsh2` as 0/1;
3. 21 full `(jpi,jpj,jpk)` binary64 Fortran-order fields, in order: `sh2`,
   `avm_k`, `avt_k`, `en`, `rn2`, `rn2b`, U(Kbb), U(Kmm), V(Kbb), V(Kmm),
   `e3uw`(Kbb), `e3uw`(Kmm), `e3vw`(Kbb), `e3vw`(Kmm), `umask`, `vmask`,
   `wumask`, `wvmask`, `gdepw`(Kmm), `e3t`(Kmm), `e3w`(Kmm); and
4. four full `(jpi,jpj)` binary64 fields: `taum`, `fr_i`, `rCdU_bot`, and
   real-valued `mbkt`.

The payload count is derived as `21*jpi*jpj*jpk + 4*jpi*jpj`.  The writer uses
its own `l4_zdf_unit`, `l4_zdf_ios`, filename, magic, counts, and zero-first
temporary arrays.  It is guarded by `lwp`, `kt==nit000`, and non-tiling; it
assigns no model array.  Every full field is copied only at an owned wet source
cell using the correct T/U/V/W grid mask.  Halo, land, below-bottom, and inactive
slots remain canonical +0.0.  BBL's writer locals are independently renamed to
`l4_bbl_*`; neither writer borrows Lane-1 locals.

The acquisition gate must derive and validate the header count, exact EOF,
field order, finiteness, canonical slots, and time levels.  Plants for header
count, one owned payload bit, and one canonical-zero slot must each make the
real validator exit nonzero.  Twin admission later requires the two complete
record inventories to be raw byte-identical, every inherited record raw
identical to the Phase-2n BBL V2 root, and ordinary restart/history identity
under only the existing timestamp exclusion.

The build uses `conda-scalarmath` and must show zero dynamic `_ZGV*` symbols.
Two separately staged `(jpni,jpnj)=(2,1)` CPU run directories use the unchanged
hash-guarded launcher pattern and are handed to the user shell.  Success of
this turn is only a prepared WRITE-only discriminator; numerical ZDF
certification waits for admitted twins.

## P2P-C — review wording and ownership register

The Phase-2n receipt will state the runoff evidence as **CONFIRMED 3/3 with the
third at reduced scope (levels 1, 2, 13, 25 of 101)**; the full 101-level
OVERFLOW retry remains PLAUSIBLE/backlog, not silently closed.  Every claim in
the Phase-2p handoff carries an explicit CONFIRMED or PLAUSIBLE label.

The GYRE lane's Round 22 result is registered as a CONFIRMED closure of the two
prior shared reproducer handoffs: CEN2 precursor and stage transports each
score 0 / 228,641 on Lane 4's gates.  Lane 4 does not relitigate or repair
those shared operators.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| correct `(50,2,21)` ownership | ASKED | Lane-4 partial-cell EEN |
| localize and walk live `e3f_vor` | ASKED | source-statement gate before change |
| alter shared EEN association after geometry closes | conditional / shared | register and route; no unauthorized repair |
| build first ZDF WRITE-only frame | ASKED | canonical rank-zero twin acquisition |
| build all downstream TKE/EVD/DDM/IWM frames now | UNASKED | first-divergence ordering; deferred |
| user-shell twin MPI | ASKED | prepare only; user executes later |
| sandbox MPI/NEMO | forbidden | none |
| shipped-tree edit, deletion, push | forbidden | none |
