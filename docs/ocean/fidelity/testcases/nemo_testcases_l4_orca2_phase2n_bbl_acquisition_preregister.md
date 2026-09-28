# NEMO testcase Lane 4 — Phase-2n diffusive-BBL acquisition preregistration

Date: 2026-09-06

Parent measurement commits: `c862c49f85f5` (Phase 2m) plus the Phase-2n
admission/discriminator commits through `4837d6187`.

Status: **PREREGISTERED BEFORE BUILD OR MPI EXECUTION.**  No BBL payload has
been produced or scored.  The shipped NEMO tree remains untouched.  The only
NEMO source change is a config-copy `MY_SRC/stprk3_stg.F90` WRITE-only writer;
it brackets the already selected `tra_bbl` call without assigning any model
array.

## Boundary and ownership

The icebergs-off ORCA2 variant resolves `ln_trabbl=.true.`,
`nn_bbl_ldf=1`, and `nn_bbl_adv=0`
(`cfgs/ORCA2_ICE_PISCES/EXPREF/namelist_cfg:268-269` plus the resolved
`ocean.output`).  NEMO therefore calls the diffusive arm at
`trabbl.F90:118-127`; the update is the bottom-cell statement at
`trabbl.F90:187-200`.  The legoESM ORCA2 recipe currently selects
`bbl_adv_option=0`, while the one existing shared BBL implementation is the
advective `nn_bbl_adv=2` arm.  Thus the first boundary after oracle-supplied
FCT output is an **ORCA2-owned missing selected arm**, not evidence about the
existing advective implementation.

## Frozen stream schema

At `kt=1`, RK stage 3 only, rank zero writes
`oracle_bbl_diffusive_kt00000001.bin` after the ordinary `tra_bbl` call.
It uses the existing canonical zero-first `T`, `U`, and `V` helpers, so halo,
land, and inactive cells are deterministic zero and no undefined slot is
admitted.

The stream is native little-endian NEMO binary:

1. 16-byte magic `NEMO_L4_BBLDF_1`;
2. 13 native int32 values: version, `kstp`, `kstg`, `Kbb`, `Kmm`, `Krhs`,
   `nn_bbl_ldf`, `nn_bbl_adv`, `jpi`, `jpj`, `jpk`, real storage bits, and
   payload count;
3. six full `(jpi,jpj,jpk)` binary64 Fortran-order fields: Kbb temperature,
   Kbb salinity, pre-BBL Krhs temperature, pre-BBL Krhs salinity, post-BBL
   Krhs temperature, post-BBL Krhs salinity; and
4. two full `(jpi,jpj)` binary64 Fortran-order fields: `ahu_bbl`, `ahv_bbl`.

The writer derives the payload count as
`6*(jpi*jpj*jpk) + 2*(jpi*jpj)`.  Admission requires the header count and
file size to agree with that expression, resolved selectors `(1,0)`, stage 3,
and all canonical-zero slots to be exactly zero.  A header-count plant and an
owned-payload plant must each make the validator exit nonzero.

## Twin and identity gates

The scalar-math instrumented executable must contain zero dynamic `_ZGV*`
symbols.  Two separately launched, otherwise identical CPU `2 x 1` MPI runs
are required.  The new 98-record inventories must be 98/98 raw byte-identical
between twins.  The 97 inherited records are compared to the admitted Phase-2m
V2 extension: raw identity except the previously registered pre-consumer
`zFw` slot in `oracle_transport_kt00000001_s1.bin`, for which header and all
owned consumed `zFu/zFv` cells must be exact.  Restart shards and history data
payloads must remain exact under the existing timestamp-only exclusion.

Success of the WRITE-only acquisition does not certify BBL numerics.  It only
admits the operands and result required for a later source-literal shared
`nn_bbl_ldf=1` implementation.  Per the user-shell rule, this turn stops with
both self-contained `run.sh` directories; no MPI/NEMO run occurs in the
sandbox.

## ASKED / UNASKED

| action | status | disposition |
|---|---|---|
| continue at BBL | ASKED | acquire the missing selected diffusive arm |
| config-local WRITE-only dump | ASKED | canonical, rank-zero, twin-admitted |
| MPI execution | ASKED, user shell only | prepare two launch directories and stop |
| change shared BBL numerics | UNASKED / deferred | no implementation before the oracle dump is admitted |
| proceed into TKE/EVD/IWM | blocked by ordered ladder | wait for BBL discriminator run |
| edit shipped NEMO | forbidden | none |
| delete prior artifacts/worktrees | forbidden | none |
