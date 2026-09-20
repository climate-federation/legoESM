# NEMO testcase Lane 4 — ORCA2 Phase-2k WZV handoff

Date: 2026-09-06

Parent: `f54fa46926db03b6864aedfb54d80e9a9fcdf5c8`

Status: **SUPERSEDED / RULE-11 RETRACTION.**  The 93-stream
icebergs-off record set is reproducible and twin A is pinned as the
`VARIANT_ORACLE_V2` extension.  The stage-1 transport divergence, surface
runoff decrement, QCO W recurrence and final `pFw` product are each exact at
0 / 233,341 when evaluated on NEMO's resolved stage-1 clock.  Section 4's
claim that production hands a full-step denominator to a stage-interpolated
SSH delta is retracted: the arm paired operands that production never pairs.
The replacement compiled-production adjudication is recorded in the Phase-2l
receipt.  This document remains as preserved provenance, but it is not a
valid shared-clock handoff.

SI3 exchanges remain `ORACLE_SUPPLIED`; SI3 dynamics and thermodynamics remain
`UNMEASURED_PENDING_ICE_MERGE`.

## 1. VARIANT V2 extension pin

The accepted extension root is

`/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2j_wzv_a_10step_np2`.

Its independent witness is the adjacent `_b_` root.  Both unchanged launchers
finished ten steps with `MPIRUN_RC=0`, `LAUNCHER_RC=0`, `RUN DONE`, two MPI
ranks in the required `jpni=2,jpnj=1` layout, and 93 records.  The fail-closed
reproducibility gate requires the complete frozen inventory and reports
**93 / 93 raw-byte identical**.  Its one-byte WZV-record plant traverses the
same validator, exits 1, and reports 92 / 93.

The campaign rule remains atomic: **a record set is oracle evidence only after
two independent executions of the same binary and deck reproduce every byte
of the complete inventory**.  Twin A extends, rather than replaces, the
92-stream V2 root pinned in Phase 2j.  All 92 inherited records are also
byte-identical to that root.  Twin B is retained as the reproducibility
witness; earlier variant and pre-canonical roots remain preserved and flagged
superseded.

The 93-entry manifest is
`phase2k/variant_v2_extension_93_records.sha256`, SHA-256
`ea400055d19bb31be0242a397872df5441a9893d8bd94b2419ae029f96c3449d`.
The new record is 24,899,672 bytes and has SHA-256
`245be2ea348002b93358e5dd723f0b84198af47c3d38f0bc0bb1acb0fc3100fe`.

The inherited acceptance schema and all planted controls were rerun on views
which exclude only the newly appended record.  Both the O1 acquisition gate
and the legacy/surface gate pass.  Ordinary-output identity against
`variant_icebergs_off_uninstrumented_10step_np2` counts four restart shards
and eight history payloads dynamically.  The ordinary outputs use the same
campaign exclusions as Phase 1: launcher/timing provenance is excluded;
NetCDF global timestamps and the WRITE-only dump notices in `ocean.output`
are ignored, while every data payload is compared.  The pinned terminal
digests are unchanged:

| artifact | SHA-256 |
|---|---|
| ocean restart rank 0 | `28fbf31286d90f31e6f72083b942a322a4d2cbfb18db6bf4983d76bae82c7280` |
| ocean restart rank 1 | `4f7873be26a96f9694470e5e7b67b93e7956180045effe948e2d3080eca4a326` |
| SI3 restart rank 0 | `4015292b4663e27a8e44109c1c001603d1d82519339d8cc1285c6d4f268d8be2` |
| SI3 restart rank 1 | `f6bea2738a8e10286653d09ee8bf1bdd18a91111c9b22c3260200975f847a229` |
| `ocean.output` | `7590d8ff2afc830e73baa3238e98a87b7be44235179ce3f0c008ebab0aa7dea6` |

The inherited ten Phase-1 plants, six surface plants and three O1 plants all
report `PASS_NONZERO`.  The V2 extension therefore changes oracle coverage,
not ordinary model numerics.  The full-global ORCA2 card entry gate was also
rerun against the extension root: T, S, u, v and SSH remain 0 / n, the card
still resolves `ln_icebergs=F` with no iceberg inputs, and the gate exits 0.

## 2. Frozen WZV schema and levels

The writer is configuration-copy
`ORCA2_OMIP_L4/MY_SRC/traadv.F90:223-269`.  Its `lwp` guard makes rank zero the
only writer.  It assigns no model array and is never read by NEMO.  The gate
validates magic `NEMO_L4_WZVS1_1`, version 1, `kt=1`, `kstg=1`,
`(Kbb,Kmm,Kaa)=(1,1,3)`, `(jpi,jpj,jpk)=(94,152,31)`, stencil bounds
`i=2..92,j=2..150`, and binary64.  The payload count is derived from the write
list:

`2*(91*149*31) + 5*(94*152*31) + 4*(94*152) = 3,112,450` binary64 values.

The two raw stencil arrays are `pFu,pFv`.  The five full 3-D arrays are live
`e3t(Kmm)`, `e3t_0`, `tmask`, post-WZV `ww`, and `pFw`.  The four full 2-D
arrays are `r3t(Kbb)`, `r3t(Kaa)`, `r1_e1e2t`, and `rnf`.  Full canonical
arrays have zeroed halos and land; the raw stencil intentionally retains the
one west/south neighbor needed for every A2D(0) result.  The score strips two
halos, uses 30 active levels, validates the all-zero dummy `jpk` record, and
scores exactly the 233,341 wet T cells on the rank-zero subdomain.

| array | registered level |
|---|---|
| `pFu,pFv,e3t,tmask,r1_e1e2t,rnf` | stage 1, `Kmm=1` |
| `e3t_0` | static reference geometry |
| `r3t(Kbb)` | before, `Kbb=1` |
| `r3t(Kaa)` | after-stage external result, `Kaa=3` |
| `ww,pFw` | stage-1 transient; no prognostic slot |

## 3. Source walk and ORCA2-owned runoff

The gate runs production JIT on CPU, with explicit fp64 and scalar-libm.  It
uses the recorded `pFu/pFv` as `ORACLE_SUPPLIED_EXTERNAL_MODE` operands; no
external-mode certification is inferred.

NEMO's executed order is:

1. `traadv.F90:138,201-222` selects vertical transport for vector-invariant
   momentum and calls `wzv(...,np_transport)` for the tracer path;
2. `divhor.F90:116-123` differences the already materialized face transports,
   multiplies by `r1_e1e2t`, and divides by live `e3t(Kmm)`;
3. `divhor.F90:126` calls runoff, and resolved
   `ln_rnf_depth=ln_rnf_depth_ini=F` selects the surface-only decrement
   `rnf*r1_rho0/e3t(:,:,1,Kmm)` at `sbcrnf.F90:253-260`;
4. `divhor.F90:140-141` rematerializes `e3t*hdiv`;
5. the QCO arm at `sshwzv.F90:330-336` performs the bottom-up recurrence with
   `r1_Dt*e3t_0*(r3t(Kaa)-r3t(Kbb))`; and
6. `traadv.F90:225-226` forms `pFw=e1e2t*ww`.

| sub-boundary | result |
|---|---:|
| transport divergence | 0 / 233,341 |
| surface runoff-adjusted divergence | 0 / 233,341 |
| QCO bottom-up `ww`, using stage-1 `rDt` | 0 / 233,341 |
| `e1e2t*ww -> pFw` | 0 / 233,341 |

Each row has an independent one-ULP plant through its own scorer; all four
exit 1 at 1 / 233,341.  Removing only runoff makes `ww` differ in
58,810 / 233,341 cells, maximum absolute error
`1.6992512525769128e-07 m s-1`.  Thus runoff is an operative ORCA2 boundary,
not a zero-field pass.

The shared implementation now carries `freshwater.runoff` into the one
source-associated `nemo_transport_wzv_divergence_level` block.  The block is
used by the existing global WZV implementation and by the rank-subdomain
gate; there is no ORCA2 fork.  This is the authorized `ORCA2_OWNER_RUNOFF`
change.  Existing no-runoff WZV tests pass; the broader focused command found
one unrelated pre-existing partial-mesh fixture failure before entering WZV.

## 4. RETRACTED: invalid mixed state/clock ablation

NEMO sets stage 1 at `stprk3_stg.F90:118-124`:

`rDt = r1_3 * rn_Dt = 3,600 s`, then `r1_Dt = 1/rDt`.

The same file selects `rn_Dt/2` at stage 2 and `rn_Dt` at stage 3.  The live
tracer WZV call consumes that module variable through
`traadv.F90:222 -> sshwzv.F90:334-335`.

The old gate combined NEMO's explicitly materialized stage-1 HYB SSH
`2/3*ssh(Kbb)+1/3*ssha` with the full-step 10,800 s denominator.  NEMO pairs
that HYB value with 3,600 s.  legoESM production pairs its full-step SSH
endpoint with 10,800 s.  The mixed HYB/full-step pair, whose 233,341 / 233,341
failure was previously mislabelled a production clock arm, is formed by
neither model.  It is retained in the gate only under the key
`retracted_invalid_mixed_state_full_dt_ablation`, cannot set the gate result or
owner, and is labelled `RETRACTED_RULE_11_NOT_A_PRODUCTION_CONFIGURATION`.

The compiled production result and the real shared boundaries are reported in
the Phase-2l receipt.  There is no `GYRE_OWNER_SHARED_WZV_STAGE_CLOCK` debt.

## 5. Coverage at the stop

| boundary | disposition |
|---|---|
| O1 nine CORE inputs | VERIFIED 0 / 13,320 each |
| NCAR bulk | `ORACLE_SUPPLIED`; `LANE3B_OWNER` debt retained |
| SI3/ice exchange | `ORACLE_SUPPLIED`; producer `UNMEASURED_PENDING_ICE_MERGE` |
| RGB, EOS-80, SCO HPG | VERIFIED exact against V2 |
| frozen EEN external coefficients | CONFIRMED DEBT; `GYRE_OWNER_SHARED_EXTERNAL_MODE` |
| external result | `ORACLE_SUPPLIED_EXTERNAL_MODE` |
| stage-1 `zFu/zFv` | VERIFIED exact |
| `div_hor` and surface runoff | VERIFIED exact |
| WZV source program at NEMO stage clock | VERIFIED exact |
| WZV production stage-clock wiring | RETRACTED under Rule 11; invalid mixed operands |
| FCT (`ln_traadv_fct=T`, `nn_fct_h=2`, `nn_fct_v=2`, `nn_fct_imp=1`) | NOT ENTERED; stopped earlier |
| tripolar FCT north fold | NOT ENTERED |
| census-round BBL | NOT ENTERED |
| TKE/EVD/shared ZDF | NOT ENTERED |
| IWM/geothermal | NOT ENTERED |

No downstream operator inherits a claim across the over-bar WZV clock.

## 6. ASKED / UNASKED

| item | status | disposition |
|---|---|---|
| execute both prepared twins unchanged | ASKED | user shell; 93 / 93 exact |
| pin twin A only after complete reproducibility | ASKED | V2 extended and hash-bound |
| rerun identity and plants | ASKED | 4 restart, 8 history; legacy/surface/O1/WZV plants green |
| fix ORCA2 runoff in the shared NEMO identity | ASKED | one shared block, no card fork; exact |
| continue in NEMO order | ASKED | stopped at first production debt |
| repair shared WZV stage clock in Lane 4 | UNASKED and forbidden | obsolete request; handoff retracted by Phase 2l |
| infer FCT, fold, BBL, or TKE results past the debt | UNASKED | not entered |
| execute another NEMO MPI run | UNASKED and unnecessary | current record separates the debt |
| remove failed/retracted evidence | UNASKED and forbidden | dummy-level failure and full-step-clock misclassification retained and flagged |
| enter SI3 operators | UNASKED and forbidden pending merge | exchange remains oracle-supplied |

## 7. Next action after owner handoff

This stopping prescription is superseded.  Continue from the Phase-2l
compiled-production adjudication; do not implement a stage-clock repair from
this receipt.

This is a Phase-2 stopping receipt, not a Phase-2 closure and not SI3 or
legoESM-card completion evidence.
