# NEMO testcase Lane 4 — ORCA2 Phase-2t handoff receipt

Date: 2026-09-06

Parent: `608b50e4b04b`

Status: **STOPPED AT THE FIRST NON-BIT TKE STATEMENT.**  The Phase-2s twins
are admitted as one reproducible `VARIANT_ORACLE_V2` root, the non-vacuous
SH2 boundary is exact on every reconstructible rank-zero cell, and the ORCA2
bottom-TKE selector is restored and exact.  The next executed boundary is the
ice-fraction attenuation operand at `zdftke.F90:255`: NEMO resolves
`nn_eice=1`, while the legoESM ORCA2 card still supplies mode zero.  No shared
SH2/TKE arithmetic, SI3 operator, shipped NEMO source, or NEMO run was changed
or executed in the sandbox.

## 1. Combined VARIANT_ORACLE_V2 admission

**CONFIRMED:** the user-shell runs
`variant_icebergs_off_phase2s_zdf_een_{a,b}_10step_np2` both ended at
`time.step=10`, `MPIRUN_RC=0`, and `RUN DONE`, with 100 records each.  Their
launch logs have SHA-256
`998ed80dce556f042a705dd2f1ddaf5ba379d39c8c53d968139897f3e7f13676`
(A) and
`dade742233631c4f333a5076628b6a6236688ef84eab1b47fbf6e57b445f5cda`
(B).

The admission gate established:

| check | result |
|---|---:|
| independent twin records | **100 / 100 exact bytes** |
| inherited Phase-2q records excluding the versioned kt=1 ZDF stream | **94 / 94 exact bytes** |
| restored EEN records versus the Phase-2m witness | **4 / 4 exact bytes** |
| ocean restart shards versus the uninstrumented control | **2 / 2 exact bytes** |
| SI3 restart shards versus the uninstrumented control | **2 / 2 exact bytes** |
| history data-variable payloads, timestamp attributes excluded by the registered rule | **8 / 8 exact** |
| schemas walked to exact EOF | **100 / 100** |

**CONFIRMED:** the sole expected inherited-record difference is
`oracle_zdf_sh2_operands_kt00000001.bin`.  Its old header-dependent SHA-256
`1e23fa06140cb137c3f75b8354de231a76f2098becda35d0b874c229efa951f7`
is superseded by the self-describing record SHA-256
`0c0a1c3e89267c834f06c378a79ae39349d14c334abb9c79b692d14eba6d62ff`.
The new, non-vacuous kt=2 record is
`oracle_zdf_sh2_operands_kt00000002.bin`, SHA-256
`9642aa7a674d240fba64251bc27318b8207aaeabd50fb492c229fef644df6c27`.
Each ZDF header carries every field extent, an allocation-table hash, a
derived payload count, and the time-level indices; the decoder needs no side
knowledge and consumes each file to exact EOF.  The allocation table is used
for validation only, never to decode the record.

Twin A is now the single pinned `VARIANT_ORACLE_V2` root.  Twin B is the raw
identity witness.  The Phase-2m, 2n, 2p, and 2q roots remain retained and are
flagged **SUPERSEDED_AS_PRIMARY_ROOT**; the Phase-2m root remains the EEN
witness.  Nothing was deleted.

The combined validator also ran all inherited controls: ten Phase-1 inventory,
schema, finite-value, SI3-bit, and ordinary-output controls; six surface-input
controls; three O1 controls; and seven ZDF magic/extent/count/truncation/
trailing/canonical-zero/non-vacuity controls.  Every plant exited nonzero.
The binding ORCA2 entry plant separately exited 1 with
`T (1/399600)`.

Evidence:

- `scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2t_combined_schema_gate.py`
- `/data/abyssal/dbalwada/nemo-testcases-l4/phase2t/nemo_testcases_l4_orca2_phase2t_admission.json`
  (SHA-256 `8fc276f7e94462ba28fa5527b4da2ac3cd08b3bec4e43d83c4ae5b17d12c7525`)
- `/data/abyssal/dbalwada/nemo-testcases-l4/phase2t/nemo_testcases_l4_orca2_phase2t_schema_walk.json`
  (SHA-256 `f81732704ad249d62b4ad36c028cfb22357bf0125d3bece98252e5031cd1bcbf`)

These two per-run outputs exceed 20 KiB and were relocated byte-for-byte from
git to the Lane-4 data root in Phase 2v.  The small cited summaries remain in
git; no artifact was deleted.

## 2. Executed SH2 arm and non-vacuous score

**CONFIRMED:** NEMO's source expression is formally Kmm times Kbb, but the
executed ORCA2 RK3 arm uses the Nbb whole-step-entry slot twice:

1. `namelist_cfg:387-396` selects TKE/EVD/DDM/IWM rather than CST/OSM/RIC.
   `zdfphy.F90:56,221-222` therefore derives `l_zdfsh2=.TRUE.`; it is not an
   independently selectable namelist switch.
2. `zdfsh2.F90:80-89` formally multiplies Kmm and Kbb face-native shear and
   divides by Kmm and Kbb face thickness.
3. `stprk3.F90:163-165` comments out the MLF-form
   `zdf_phy(kstp,Nbb,Nnn,Nrhs)` and calls `zdf_phy(kstp,Nbb,Nbb,Nrhs)`.
4. `zdfphy.F90:264-269` passes those two actual arguments to `zdf_sh2`.

Thus both formals alias Nbb in the running executable.  The reviewer's
cross-level reading describes the formal expression under distinct MLF
arguments (`stpmlf.F90:190`), not this RK3 call site.

**CONFIRMED:** the kt=2 header reports `Kbb=3`, `Kmm=3`, `Krhs=1`; its U and V
Kbb/Kmm arrays are byte-identical and have 217,668 and 218,048 nonzero
gradient entries, respectively.  Under production JIT, CPU, fp64, and
scalar-libm policy, both the restored card tuple and a direct source-literal
call score:

| arm | unequal / n | max row-scale ULP | disposition |
|---|---:|---:|---|
| production restored tuple, record-complete stencil | **0 / 218,024** | 0 | AT-BAR |
| source-literal formal Kmm/Kbb, record-complete stencil | **0 / 218,024** | 0 | AT-BAR |

There is no first non-bit SH2 statement on a reconstructible scored cell and
therefore no shared-SH2 handoff.  The broader owned-cell diagnostic has 4,396
apparent differences among 222,506 cells: west edge 1,686, east edge 2,077,
north-fold edge 633, south edge 0, strict interior 0.  Those edge stencils
consume MPI/fold halo inputs that the WRITE-only writer canonicalized to zero,
so the operand record cannot reconstruct them.  They are
**UNINFORMATIVE_RECORD_INCOMPLETE**, not a physics score or waived debt.  The
one-bit target and wrong-selector plants both exit nonzero.

Evidence: `nemo_testcase_l4_orca2_phase2t_sh2_gate.py` and
`nemo_testcases_l4_orca2_phase2t_sh2.json` in the Lane-4 validation/result
directories.

## 3. Bottom TKE identity restoration

**CONFIRMED:** the ORCA2 reference resolves `ln_drg_OFF=.false.` at
`namelist_ref:812`.  Consequently `zdftke.F90:279-288` executes the
bottom-friction TKE boundary, evaluates the Kbb bottom velocities and masks
with `rCdU_bot`, and writes
`en(mbkt+1)=MAX(zebot,rn_emin)*ssmask`.  Because IWM is selected by
`namelist_cfg:396`, `zdftke.F90:841-844` sets `rn_emin=1e-10`.
`nn_bc_bot=1` at `namelist_ref:1261` is documented only for wave coupling; it
is read but does not guard the executed `tke_tke` branch.

The shared implementation already carried the per-column bathymetry-relative
bottom boundary.  Only the ORCA2 card selector changed:

```text
bottom_tke_bc: False -> True
```

This is a NEMO-identity restoration, not a new numerical option or a default
change.  At kt=1 the recorded Kbb bottom velocities are zero, so the exact
resolved target is `rn_emin`.  Production scores **0 / 8,794** wet bottom
rows; the target-bit and selector-false plants exit nonzero.

**CONFIRMED:** after this card-only restoration, the full ORCA2 entry gate is
unchanged: T, S, u, v are each **0 / 399,600**, and ssh is
**0 / 13,320**.  Cross-card selector gates are likewise unchanged: GYRE
**0 / 71,100**, LOCK_EXCHANGE **0 / 15,588**, OVERFLOW **0 / 111,700**;
total **0 / 198,388**.  Their shared kt=1 plant exits 1.

## 4. Ordered TKE walk and stop boundary

The existing frame closes the surface-input and SH2 precursor rows.  The
bottom boundary above is exact.  The next executed statement is:

```fortran
zdftke.F90:255  zice_fra(:) = TANH( fr_i(T1Di(0),jj) * 10._wp )
```

`nn_eice=1` is inherited from `namelist_ref:1255`; its first consumption is
the Langmuir source multiplier at `zdftke.F90:359`.  The legoESM ORCA2 card
currently supplies `eice=0`, hence zero rather than the NEMO tanh operand.

| boundary | unequal / n | first zero-based cell | candidate | oracle | owner |
|---|---:|---:|---:|---:|---|
| `nn_eice=1` ice-fraction attenuation | **1,779 / 8,794** | `(j=1,i=49)` | 0 | 0.9999999639936913 | `LANE4_ORCA2_FORCING_SELECTOR_THEN_GYRE_OWNER_SHARED_TKE` |

This is **CONFIRMED_CARD_SELECTOR_DEBT** and the requested fail-closed stop.
Lane 4 owns supplying the resolved ORCA2 forcing selector/operand.  Any change
to the shared TKE tanh transformation or downstream TKE arithmetic is routed
to GYRE.  Buoyancy production, dissipation, tridiagonal assembly/solve,
`nn_mxl=3`, and avm/avt assembly remain
**UNMEASURED_AFTER_FIRST_DEPARTURE**.  `nn_etau=1` also remains unwalked after
this stop.  No proxy target was manufactured.

The exact CPU reproducer attached to that boundary is:

```text
JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 PYTHONPATH=.:packages/core:packages/ocean \
  /home/dbalwada/legoESM/.venv/bin/python \
  scripts/validate/ocean_fidelity/orca2_l4/nemo_testcase_l4_orca2_phase2t_tke_entry_gate.py \
  --deck-root /data/abyssal/dbalwada/nemo-testcases-l4/inputs/ORCA2_ICE_v5.0.0 \
  --oracle-root /data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2s_zdf_een_a_10step_np2 \
  --mesh /data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_icebergs_off_phase2s_zdf_een_a_10step_np2/mesh_mask_0000.nc \
  --json-out docs/ocean/fidelity/testcases/nemo_testcases_l4_orca2_phase2t_tke_entry.json
```

The same command with `--plant` exits nonzero.  The targeted constructibility
and bottom-TKE unit selection also closes **2 passed, 72 deselected**.

### Coverage and time-level registry for the admitted ZDF frames

| field group | allocation/grid | kt=1 levels | kt=2 levels | disposition |
|---|---|---|---|---|
| `sh2`, `avt_k_pre`, `en_pre` | reduced W, `90x148x31` | Kbb=1, Kmm=1 | Kbb=3, Kmm=3 | VERIFIED exact EOF; SH2 kt=2 0/218,024 |
| `avm_k_pre`, `rn2`, `rn2b` | full W, `94x152x31` | before closure | before closure | VERIFIED record; post-closure absent |
| `u_Kbb/u_Kmm`, `v_Kbb/v_Kmm` | full U/V | 1/1 | 3/3 | VERIFIED; actual kt=2 aliases are identical |
| `e3uw/e3vw_Kbb,Kmm` | full UW/VW | 1/1 | 3/3 | VERIFIED operands |
| `umask/vmask/wumask/wvmask` | full face masks | static | static | VERIFIED operands |
| `gdepw_Kmm`, `e3t_Kmm`, `e3w_Kmm` | full W/T/W | Kmm=1 | Kmm=3 | VERIFIED operands |
| `taum` | reduced T, `90x148` | surface input | surface input | VERIFIED operand |
| `fr_i`, `rCdU_bot`, `mbkt_real` | full T, `94x152` | current/static | current/static | VERIFIED operands; `fr_i` is first TKE debt |

The kt=1 ZDF-entry stream additionally carries the pre-closure fields used by
the bottom gate.  No admitted stream contains post-solve `en`, mixing length,
or post-closure avm/avt, so those fields are explicitly UNMEASURED.

## 5. Independent BBL repeat

**CONFIRMED:** three independent production-JIT invocations generated
byte-identical reports, each SHA-256
`144bc0bd7df39c801174ea6383b872ece6cc4dbc014acb1c398669baeb4c24cf`.
Every execution printed:

```text
ahu_bbl                 0 / 8,489
ahv_bbl                 0 / 8,554
temperature_Krhs_post   0 / 231,519
salinity_Krhs_post      0 / 231,519
```

The paired coefficient, pre-tracer, and post-RHS plants each exited 1 through
their production scorers.  Rule-12 inactive-card rows remained exact in every
run: GYRE T/S **0 / 21,120**, LOCK T/S **0 / 7,800**, and OVERFLOW T/S
**0 / 60,600**.

## 6. Rule 8 / 11 / 12 accounting

| rule | status | evidence/disposition |
|---|---|---|
| Rule 8, production-path proof | **CONFIRMED** | SH2, bottom TKE, entry, and BBL gates invoke JIT production functions under fp64/scalar-libm; wrong selectors and one-bit targets fail |
| Rule 11, retractions | **CONFIRMED** | formal Kmm/Kbb is retained as source description, but the executed result is re-registered as Nbb/Nbb; record-incomplete edge cells are not printed as physics debt |
| Rule 12, shared-effect census | **CONFIRMED** | ORCA2 entry unchanged; GYRE/LOCK/OVERFLOW selector total 0/198,388; BBL inactive rows exact on all three cards |

**PLAUSIBLE, not claimed as certified:** once Lane 4 supplies `nn_eice=1`, the
shared scalar-libm tanh path should be directly testable against the same
record.  No downstream TKE result is inferred from that expectation.

## 7. ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| admit and pin Phase-2s twins | ASKED | complete; A primary, B raw witness |
| restore all EEN streams into the one root | ASKED | complete; 4/4 match Phase-2m |
| resolve SH2 Kmm/Kbb levels | ASKED | complete; formal Kmm/Kbb, executed Nbb/Nbb at step entry |
| restore `bottom_tke_bc` | ASKED identity restoration | complete, `False -> True`; general default unchanged |
| add the new/existing config-field row in the same commit | standing requirement | recorded above; no new field introduced |
| keep BBL defaults `0 / 0.0` | ASKED Decision 10 | resolved and unchanged; no silent on |
| walk TKE to the first non-bit statement | ASKED | stopped at `nn_eice=1`, 1,779/8,794 |
| repeat BBL gates and plants three times | ASKED | complete, identical reports |
| alter shared SH2/TKE arithmetic | UNASKED and Lane-4-forbidden | not done |
| enter SI3 operators | UNASKED pending ice-lane merge | not done; SI3 remains ORACLE_SUPPLIED |
| run MPI/NEMO in sandbox, edit shipped NEMO, delete, push | forbidden | not done |

The Phase-2s independent-review verdict had not been relayed when this receipt
was closed; no unreceived review finding is silently treated as resolved.

## 8. Handoff / next boundary

The next round must first make the ORCA2 card select NEMO's `nn_eice=1`
ice-fraction attenuation without an ORCA2 arithmetic fork, then re-score
`zdftke.F90:255` and continue in execution order.  If the shared scalar-libm
tanh path or later TKE statement is non-bit after the card operand is correct,
the gate and admitted record form the GYRE-owned reproducer.  Post-closure
walking will require additional canonical WRITE-only frames for `en`, mixing
length, and avm/avt.

No new MPI/NEMO run is needed for the present stop, and no `run.sh` is handed
off.
