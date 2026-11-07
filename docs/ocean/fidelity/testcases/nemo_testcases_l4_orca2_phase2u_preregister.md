# NEMO testcase Lane 4 — ORCA2 Phase-2u preregistration

Date: 2026-09-06

Starting parent: `ccdc57c3c29c`

Status: **PREREGISTERED BEFORE MEASUREMENT.**  This round incorporates the
Phase-2s review's time-level naming correction, restores the ORCA2
`nn_eice=1` selector/source statement, and resumes the ordered TKE walk.  EVD
and IWM are entered only if every TKE boundary available from the admitted kt=2
frame reaches the exact bar.

## P2U-1 — Nbb selector name and registry

Rename the single selector `nemo_face_native_now2` to
`nemo_face_native_nbb2` everywhere it is accepted, constructed, or tested.
No compatibility alias is retained: an obsolete name must fail rather than
silently preserve Rule-1d ambiguity.  The stage field remains `step_entry`.

The registry proof is the executed key_RK3 call at `stprk3.F90:164-165`: the
MLF-form `zdf_phy(kstp,Nbb,Nnn,Nrhs)` is commented out and the live call is
`zdf_phy(kstp,Nbb,Nbb,Nrhs)`.  `stpmlf.F90:190` remains the contrasting live
MLF call.  Receipts must call the RK3 slot Nbb/step-entry, never "now".

Admission requires the production SH2 score and ORCA2/cross-card entry rows to
remain bit-identical after the pure rename.  The ZDF decoder must be shown to
consume its self-describing header without an external allocation table; the
allocation table is validation evidence only.

## P2U-2 — `nn_eice=1` identity restoration

ORCA2 inherits `nn_eice=1` from `namelist_ref:1255-1259`; the executed source
statement is `zdftke.F90:255`,
`zice_fra=TANH(fr_i*10._wp)`, first consumed at `:359`.  The current ORCA2 card
has `TKEConfig.eice=0`; change only that card to 1.

The existing shared mode-1 implementation is not source-faithful: it maps to
raw `fr_i`.  Replace that mode in the one shared TKE ice-attenuation mapping by
the NEMO-named statement, using the scalar-libm `tanh` policy and
`nemo_source_round` at the source-operation boundary.  Both C-grid and MPAS TKE
callers must consume that single mapping; KPP's separately documented
non-NEMO analogue is out of scope and unchanged.  Defaults remain zero.

The boundary confirms only at **0 / 8,794** wet rank-zero columns under
production JIT, CPU, fp64, scalar-libm.  A one-bit target and a wrong-selector
plant must exit nonzero.  GYRE inherits NEMO `nn_eice=1` at its
`namelist_ref:1255-1259` but its legoESM card remains unchanged this round;
LOCK and OVERFLOW do not execute the TKE arm.  Their established kt=1 gates
must remain **0 / n**.  Any movement stops the round under Rule 12.

## P2U-3 — ordered TKE walk

After the eice operand is exact, walk the admitted kt=2 fields in NEMO order:
surface Dirichlet, bottom Dirichlet, Langmuir/ice operand, SH2 and buoyancy RHS,
dissipation, matrix assembly, forward/back recurrences, `nn_mxl=3`, and avm/avt
assembly with `rn_emin/rn_emin0` floors.  Each scored statement uses NEMO's
recorded inputs, cellwise row-scale ULP, and one-variable arms.

The existing frame contains pre-closure operands but no post-solve `en`,
mixing-length, or post-closure avm/avt target.  If the next statement lacks an
oracle target, stop as `UNMEASURED_NEEDS_WRITE_ONLY_FRAME` and name the exact
arrays/call site; do not infer closure from a precursor.  A shared arithmetic
departure routes to `GYRE_OWNER_SHARED_TKE` with the reproducer.  An ORCA2 card
selector or forcing operand remains Lane 4.

That stop is reached at the no-Stokes Langmuir operand
`zdftke.F90:332`, `zWlc2=zcsd*taum`: the admitted frame has `taum` but no
oracle `zWlc2`.  Acquire one new kt=`nit000+1` rank-zero stream before making
another arithmetic claim.  Its frozen self-describing schema is magic
`NEMO_L4_TKEW_1`, base header `(version,kt,Kbb,Kmm,jpi,jpj,jpk,real_bits,
n3,n2,payload,nn_eice,nn_etau,nn_mxl,nn_pdl)`, and one header-derived extent
triple per field.  Payload order is:

1. 3-D: `zpelc`, `en_post_lc`, `pdlr`, `zdiag_pre_solve`,
   `zd_lw_pre_solve`, `zd_up_pre_solve`, `en_rhs_pre_solve`,
   `zdiag_after_forward`, `zd_lw_after_forward`, `en_post_solve`,
   `en_post_etau`, `mxlm`, `mxld`, `avm_post`, `avt_post`, `dissl_post`;
2. 2-D: `zice_fra`, `zWlc2`, `imlc_real`, `zhlc`, `zus3`.

The config-local writer is armed only on rank zero, at kt=2, outside tiles;
it allocates only while armed, zeroes all storage first, copies only the owned
wet water column, and assigns no model array.  Twin admission requires the
existing 100 records to remain raw-identical to the Phase-2s V2 root, the new
record to be raw-identical between independent twins, ordinary outputs to pass
the existing identity gate, and magic/extent/count/truncation/trailing/
canonical/twin plants all to exit nonzero.  This acquisition records targets;
it does not pre-judge which TKE statement is first non-bit.

## P2U-4 — conditional EVD/IWM entry

Only if TKE closes, continue through `zdfphy.F90:311-336`: copy closure Kz,
`zdf_evd`, DDM, then `zdf_iwm`.  The same exact/plant/ownership rules apply.
Otherwise EVD/IWM remain explicitly unentered after the first TKE stop.

## Rule 8 / 11 / 12 and ASKED / UNASKED

| action | classification | preregistered disposition |
|---|---|---|
| rename SH2 selector to Nbb | ASKED review fix | semantic rename only; old name becomes invalid |
| restore ORCA2 `eice: 0 -> 1` | ASKED identity restoration | card-only selector; no choice/default change |
| repair shared mode-1 mapping | ASKED missing faithful arm | one implementation, source-literal, scalar-libm |
| walk TKE then EVD/IWM | ASKED | fail closed at first non-bit/unmeasured boundary |
| acquire the registered TKE targets | normal in-scope follow-up to that stop | one WRITE-only rank-zero stream; twin MPI handed to user |
| edit shared TKE arithmetic beyond the missing mode-1 statement | UNASKED / GYRE-owned | forbidden in Lane 4 |
| alter GYRE/LOCK/OVERFLOW/KPP selections | UNASKED | must remain byte-identical |
| run MPI/NEMO, edit shipped NEMO, delete, push | forbidden | not planned |

Rule 8 keeps faithful selector repairs even if they expose the next debt.  Rule
11 replaces the misleading NOW label in code and receipts, not merely in prose.
Rule 12 requires production-path exactness and unchanged cross-card rows.
