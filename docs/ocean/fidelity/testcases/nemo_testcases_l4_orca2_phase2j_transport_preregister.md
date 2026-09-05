# NEMO testcase Lane 4 — ORCA2 Phase-2j downstream preregistration

Date: 2026-09-06

Parent: `70bcc9a37` (VARIANT V2 pin)

Oracle: `VARIANT_ORACLE_V2`, twin A at
`variant_icebergs_off_phase2i_rhsrank0_a_10step_np2`.  The frozen inventory is
92 streams and is admissible only because an independent execution of the
same scalar-math binary and deck produced all 92 streams byte-for-byte.

## Operand-substitution boundary

The frozen EEN coefficients are already over bar.  They are a
`GYRE_OWNER_SHARED_EXTERNAL_MODE` debt and are not repaired in Lane 4.  To
continue without attributing that debt to later operators, every downstream
measurement starts from the oracle's own external-mode result.  In the first
transport discriminator those supplied values are the recorded `un_adv`,
`vn_adv`, `r1_hu(Kmm)`, `r1_hv(Kmm)`, `uu_b(Kmm)`, and `vv_b(Kmm)` operands.
They are labelled `ORACLE_SUPPLIED_EXTERNAL_MODE`, never a certified legoESM
external solve.

The first measured boundary is the stage-1 horizontal transport source
program in `stprk3_stg.F90:265-280`.  The writer immediately following it
records the live metric, Kmm thickness, Kmm velocity, correction, mask and
`zFu/zFv` result in
`oracle_rkstage1_transport_operands_kt00000001.bin` (`:283-310`).  The gate
will:

1. validate magic `NEMO_L2_TRPOP_2`, version 2, `kt=1`, `stage=1`, `Kmm=1`,
   `(jpi,jpj,jpk)=(94,152,31)`, binary64, and the derived
   `10*jpi*jpj + 8*jpi*jpj*jpk` payload;
2. strip exactly the two writer halos, retain the dummy bottom record in the
   schema, and score the 30 live levels only where the recorded U/V mask is
   one;
3. run the shared source-order expression under production JIT, CPU, fp64 and
   scalar-libm: materialize `velocity + correction*mask`, then
   `metric*e3`, then their product with `nemo_source_round` at each Fortran
   statement boundary;
4. report `unequal / n`, maximum absolute error and maximum ULP distance for U
   and V separately; and
5. mutate one live U result by one ULP through the same validator and require
   nonzero exit.

The pass condition is exactly 0 / n for both `zFu` and `zFv`.  A failure is
owned by `GYRE_OWNER_SHARED_STAGE_TRANSPORT` and is handed off without changing
shared transport code.  This discriminator does not claim the tripolar fold:
the recorded products precede the `tra_adv_fct` halo/fold exchange.  If it
passes, the next boundary is vector-form `tra_adv_trp` vertical transport
(`traadv.F90:138,201-242`) and then FCT's explicit `lbc_lnk` north-fold call
(`traadv_fct.F90:304-314`).  If the retained records cannot separate WZV,
fold, and FCT at source-defined cells, Lane 4 will prepare the minimum
rank-zero WRITE-only operand record and stop for a user-shell rerun.

## Remaining ordered ladder

After horizontal transport, the frozen order is: `zFw`/WZV; FCT tracer
advection; census-round BBL; then TKE/EVD/IWM entry.  Oracle bulk and SI3
exchange values remain `ORACLE_SUPPLIED`; SI3 operators remain
`UNMEASURED_PENDING_ICE_MERGE`.  ORCA2 may repair only north-fold, BBL, IWM,
geothermal, runoff, RGB and input-file semantics.  Any shared FCT, WZV, ZDF,
TKE, transport, or external-mode debt is registered with its boundary and
handed to the GYRE owner.

## ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| continue after EEN with oracle external outputs | ASKED | exact recorded operands substituted |
| cellwise production-JIT fp64/scalar-libm gate | ASKED | 0 / n bar and binding plant |
| fix shared transport if unequal | UNASKED and forbidden | hand to GYRE owner |
| infer north-fold certification from a pre-fold record | UNASKED | explicitly not claimed |
| add a WRITE-only discriminator if existing records are ambiguous | ASKED process | minimum schema only; user executes MPI |
| execute MPI | UNASKED and prohibited | no MPI in this measurement |
| delete superseded roots | UNASKED and forbidden | all retained |
