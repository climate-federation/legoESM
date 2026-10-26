# NEMO testcase Lane 4 — ORCA2 Phase-2b ocean-exchange preregistration

Date: 2026-09-04  
Parent: `f4ac8a2a3a1f4deaf857ba1fa6ba1ee9b645a50f`  
Prior receipt: `nemo_testcases_l4_orca2_phase2_receipt.md`  
Scope: first ocean boundary after the Phase-2 B2-I split decision

## Decision supplied by the user

The selected SI3 dynamics and thermodynamics remain
**UNMEASURED_PENDING_ICE_MERGE**.  The selected iceberg model remains
**UNMEASURED_PENDING_USER_DECISION**.  Neither operator will be recreated,
stubbed, disabled, or entered in this round.  The shipped ORCA2 oracle deck,
including `ln_icebergs=.true.`, is unchanged.

For the ocean-only ordered walk, the exact SI3/iceberg-to-ocean fields at
`kt=1` are `ORACLE_SUPPLIED` operands.  This is operand substitution, not a
claim that legoESM executed either producing operator.  The card remains the
full `180 x 148 x 30` global domain; detailed scoring remains restricted to
the rank-0 owned `90 x 148` strip obtained by removing the two-point halos
from NEMO's `94 x 152` rank-0 records.

## B2b-X — frozen first boundary and stopping result

The accepted `oracle_si3_exchange_frames.bin` is not the final ocean surface
input.  Its call is inside `ice_stp`, directly after `ice_update_tau`
(`cfgs/ORCA2_OMIP_L4/MY_SRC/icestp.F90:217-238`).  At that point its own
writer warns that the `utau`, `vtau`, and `emp` halos are not valid
(`:261-287`).

After `ice_stp` returns, the executed `sbc` path still performs, in order:

1. iceberg stepping at `src/OCE/SBC/sbcmod.F90:484`;
2. runoff at `sbcmod.F90:494`;
3. annual freshwater-budget correction at `sbcmod.F90:498`;
4. final stress/freshwater/runoff/non-solar-heat halo exchange at
   `sbcmod.F90:529-537`; and
5. interpolation of T-point stress to the U/V ocean operands at
   `sbcmod.F90:539-547`.

The iceberg crossing is material: `icb_thm` subtracts
`berg_grid%floating_melt` from `emp` and adds
`berg_grid%calving_hflx` to `qns`
(`src/OCE/ICB/icbthm.F90:286-295`).  The runoff path adds the prescribed
runoff-iceberg climatology to `rnf` (`src/OCE/SBC/sbcrnf.F90:133-144`), and
the resolved `nn_fwb=2` path modifies both `emp` and `qns`
(`src/OCE/SBC/sbcfwb.F90:292-295`).  Thus the accepted frame cannot be
silently promoted to the post-`sbc` ocean boundary.

**Preregistered result: B2b-X is MISSING_ORACLE_OPERAND and is the stopping
boundary.**  No legoESM stage-1 arithmetic may run until the replacement
record below is executed and validated.  This is a source-proven coverage
failure, not an over-bar numerical result and not an implementation choice.

## WRITE-only replacement record

A configuration-local writer will be added to the existing
`cfgs/ORCA2_OMIP_L4/MY_SRC/stprk3.F90` override immediately after
`CALL sbc(kstp,Nbb,Nbb)` and before EOS/BN2, ZDF, external mode, or an RK
stage.  It is armed only for `lwp .AND. kstp == nit000`, opens one rank-0
stream with `STATUS='REPLACE'`, writes, closes, and assigns no model array.

Frozen stream name: `oracle_ocean_surface_input_kt00000001.bin`  
Frozen magic: sixteen bytes, `NEMO_L4_SBCIN_1`  
Frozen integer header (native four-byte integers):
`version,kt,Kbb,jpi,jpj,jpts,nclasses,halo,nfull,nreduced2d,nhalo1,nreduced3d,bits`
Frozen header values: `1,1,1,94,152,2,10,2,20,12,1,2,64`

The binary64 payload is written in Fortran column-major order.  The 33 2-D
fields, in order, are:

`utau, vtau, utauU, vtauV, utau_b, vtau_b, utau_icb, vtau_icb, taum, wndm,
qsr, qns, qns_b, qsr_tot, qns_tot, emp, emp_b, sfx, sfx_b, emp_tot, fwfice,
rnf, rnf_b, fwficb, fr_i, snwice_mass, snwice_mass_b, snwice_fmass,
rCdU_ice, berg_grid%calving, berg_grid%calving_hflx,
berg_grid%floating_melt, berg_grid%stored_heat`.

The two 3-D families, in order, are `rnf_tsc,rnf_tsc_b`, each with `jpts=2`.
The NEMO allocation classes are part of the schema: 20 arrays are explicit
full `jpi*jpj`; 12 are `A2D(0)`; `rCdU_ice` alone is `A2D(1)`; and the two
runoff families are `A2D(0),jpts`.  With the resolved two-cell halo, the exact
payload count is therefore
`20*94*152 + (12+2*2)*(94-4)*(152-4) + 1*(94-2)*(152-2) = 512,680`
binary64 values.  The complete file size must be
`16 + 13*4 + 512680*8 = 4,101,508` bytes.  The writer declares the four
allocation-class counts from the actual write list; the gate mirrors this
derivation and rejects an unexplained flat-field constant.

This inventory contains the final fields actually consumed by ocean surface
momentum/heat/freshwater/salt paths, their RK3 before carries initialized by
`sbcmod.F90:549-575`, the explicit SI3 mass/drag exchange, and the iceberg
source/flux fields needed to label the substitution.  The runoff tracer
contents are recorded here because `sbc_rnf` produces them at this boundary
(`src/OCE/SBC/sbcrnf.F90:151-184`) and the later tracer surface operator
consumes both levels (`src/OCE/TRA/trasbc.F90:175-177`).  `sbc_tsc` is
deliberately absent: `tra_sbc` creates that tendency later, so it is not an
initialized post-`sbc` input.

## Replacement run and acceptance

The replacement is the accepted scalar-math, two-rank, ten-step oracle with
only this additional WRITE-only call and writer.  It retains `nn_fsbc=2`, so
SI3 and iceberg operators run on odd steps (`src/OCE/ICB/icbstp.F90:71-142`),
and retains the automatic smallest valid `jpni=2,jpnj=1` layout forced by the
iceberg implementation.  The run controls remain `nn_itend=10`,
`nn_stock=10`, `nn_istate=1`.

After user-shell execution, acceptance requires all of the following:

- launcher exit zero, `RUN DONE`, `time.step=10`, no NEMO error;
- new-record magic, every header field, derived file size, finite payload,
  and complete named-field walk valid;
- all pre-existing 90 oracle streams retain their accepted schemas;
- every ordinary output and all six restart shards are byte-identical to the
  accepted uninstrumented ten-step control, excluding only launcher/timing
  metadata by the already registered identity rule;
- a planted bad magic, bad derived field count, truncated payload, NaN, and
  one-ulp active-field mutation each traverse the validator and exit nonzero;
  and
- the rebuilt binary has no dynamic `_ZGV*` symbols and all configuration-local
  source and build/run artifacts are hash-pinned.

Only after those gates pass may Phase 2b resume at B3a using the final fields
as `ORACLE_SUPPLIED` inputs.  B3a will then walk NCAR/CORE bulk, runoff and
freshwater carry, RGB QSR, EOS/HPG, external mode, transports, FCT, BBL, and
TKE/EVD/IWM in NEMO execution order, stopping at the first over-bar boundary.

## ASKED / UNASKED

| item | status | handling |
|---|---|---|
| continue the ocean ladder without merged ice code | ASKED | exact oracle operands substitute only at the exchange boundary |
| SI3/iceberg operators remain unmeasured | ASKED | explicit owner labels; neither operator entered |
| do not change the shipped iceberg deck | ASKED | `ln_icebergs=.true.` retained |
| NEMO rerun needed for missing operand | ASKED process | prepare one self-contained run and stop for user-shell execution |
| post-`sbc` writer position | UNASKED source fact | first point at which all selected exchange modifiers and final halos have executed |
| include ocean before carries at kt=1 | UNASKED coverage choice | records exact RK3 surface forcing levels set by the same call |
| include both `rnf_tsc` levels, omit later `sbc_tsc` tendency | UNASKED source fact | records initialized runoff operands without reading a future tracer-stage result |
| rank-0 stream, no gather | ASKED inherited layout | `lwp` writer and documented halo removal |

### Pre-execution schema correction

The first committed draft classified all 33 2-D fields as `jpi*jpj` and
therefore preregistered the wrong byte count.  A source-allocation walk before
the build/run handoff found the distinction at
`src/OCE/SBC/sbc_oce.F90:185-216`, `src/OCE/SBC/sbc_ice.F90:145-161`, and
`src/OCE/SBC/sbcrnf.F90:84-89`.  The corrected header above declares full,
`A2D(0)`, and `A2D(1)` counts separately.  No record was measured under the
retracted draft schema.
