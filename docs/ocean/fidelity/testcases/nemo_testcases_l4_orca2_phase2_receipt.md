# NEMO testcase Lane 4 — ORCA2 Phase-2 entry receipt

Date: 2026-09-04

Base: `5bbb9ee56ad885e35abd8b244831697194e0e0c7`

Preregistration: commit `325e49a74`

Card construction: commit `2c60066a9`

Result: **STOP at B2-I; B1 and B2-O pass**

This receipt closes only the first two preregistered boundaries. It does not
claim an ORCA2 integration, an ocean stage match, or an SI3 match. No NEMO
source or shipped configuration was changed, no NEMO rerun was made, and no
legoESM stage was executed.

## 1. Outcome

| boundary | result | evidence |
|---|---|---|
| B1 deck/card | **VERIFIED_WITH_EXPLICIT_UNMEASURED** | all consumed files hash-pinned; geometry, raw Coriolis, 3-D thicknesses, T/U/V/F masks and topology bit-exact |
| B2-O ocean kt=1 entry | **CONFIRMED, VERIFIED_BIT_EXACT** | `T 0/399600`, `S 0/399600`, `u 0/399600`, `v 0/399600`, `ssh 0/13320` unequal |
| B2-I SI3/iceberg kt=1 entry | **UNMEASURED_STOP** | the merged shared state does not carry the full selected SI3/iceberg registry; the card execution guard rejects it |
| B3a onward | **UNMEASURED / NOT ENTERED** | ordered stopping rule |

The comparison domain is the rank-0 owned strip only. The writer's local
record is `94 x 152 x 31`; two halos are stripped from every horizontal edge
and the dummy `jpk` record is checked zero and removed. The scored region is
the western `90 x 148 x 30` slice of the full legoESM `180 x 148 x 30`
owned domain. No rank-1 detailed payload is inferred.

The baseline gate was run with `JAX_PLATFORMS=cpu`, `JAX_ENABLE_X64=1`, the
production JAX implementation, fp64 policy, and the scalar-libm card stamp.
The JSON is under
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2/orca2_phase2_entry_gate.json`.

## 2. B1 — source-file card construction

The card is `ORCA2-zps`, a new consumer of the shared lat-lon C-grid NEMO
WS-RK3 program. There is no ORCA2-local step formula. The one shared dispatch
extension admits the executed vector-invariant/EEN/c2/no-Aimp program; it does
not implement an alternative operator.

The card consumes exactly three files at this boundary:

| file | SHA-256 |
|---|---|
| `ORCA_R2_zps_domcfg.nc` | `7125f7a54e8693ff8f1327b878d2258d258878bc7302ac123ac671f2339d2839` |
| `data_1m_potential_temperature_nomask.nc` | `ca00905c27078305e80130ea6dd07fc575de2ae3257fad38003b329458cf7f7a` |
| `data_1m_salinity_nomask.nc` | `ad648d972f0631bde7e1b598d98470a979b7f2649271fc9cf5ccfca158e31d0c` |

The archive remains immutable at the Phase-1-pinned SHA-256
`5d47eab85c591fe0fd7e63a80892f3264b6387edf93cbb975a71b2f2f5fdf1a4`.

NEMO source order was followed:

- `src/OCE/DOM/domhgr.F90:202-226` reads native coordinates, metrics and
  `ff_t/ff_f`; the shared tripole loader now retains those raw Coriolis fields.
- `src/OCE/DOM/domzgr.F90:95-173` reads `e3t_1d/e3w_1d`, wet levels and the
  native T/U/V/F 3-D scale factors. `src/OCE/DOM/depth_e3.F90:109-130`
  defines the scalar depth recurrence reproduced by the card.
- `src/OCE/DOM/dommsk.F90:146-172,207-243` defines the face masks,
  `rn_shlat=2` no-slip extension and file strait override. All four masks are
  bit-exact against the two Phase-1 `mesh_mask` shards after reassembly.
- `src/OCE/DOM/istate.F90:107-137` selects file T/S and zero U/V.
  `src/OCE/SBC/fldread.F90:181-227` fixes the temporal weights and the literal
  `before_weight*before + after_weight*after` operation order. The file's
  float32 values are promoted to binary64 before those operations.
- `src/ICE/iceistate.F90:262-393` diagnoses and distributes initial ice;
  `:398-427` subtracts globally averaged snow+ice+pond mass from both ocean
  levels. Reproducing the live pond contribution gives the exact oracle SSH
  `-0.09343505872684969 m` on wet cells.

Exact B1 comparisons cover `glamt/gphit` after NEMO-compatible radian
conversion; `e1t/e2t/e1u/e2u/e1v/e2v`; `ff_t/ff_f`;
`e3t_0/e3u_0/e3v_0/e3f_0`; `tmask/umask/vmask/fmask`; 30 prognostic levels;
and the north T-fold mapping. Nominal two-degree geometry is never
reconstructed.

### Fail-closed selected-feature register

These resolved ORCA2 mechanisms are named on the card but are not claimed to
be in the executable shared chain:

1. NEMO `fld_read` forcing/regridding and temporal interpolation;
2. staged GM/EIV;
3. linear implicit bottom drag;
4. differential internal-wave mixing;
5. spatial lateral viscosity;
6. freshwater-budget carry;
7. SI3 `jpl=5` layered/Prather state;
8. iceberg state.

`validate_nemo_testcase_card_for_execution` reports all eight and raises
nonzero. Thus construction and input comparison cannot accidentally execute a
hybrid third model.

## 3. B2-O — ocean step-entry identity

The accepted Phase-1 stream is
`instrumented_reviewfix_10step_np2/oracle_step_entry_kt00000001.bin`, magic
`NEMO_L1_ENTRY_1`, header `(kt,stage,Kbb,nx,ny,nz,halo,bits) =
(1,1,1,94,152,31,2,64)`. The validator checks magic, the complete header,
payload size, Fortran ordering, dummy record, halo removal, staggering and
then raw binary64 equality.

| field | card mapping | unequal / compared | disposition |
|---|---|---:|---|
| T | T-point `[:, :90, :30]` | `0 / 399600` | VERIFIED Kbb |
| S | T-point `[:, :90, :30]` | `0 / 399600` | VERIFIED Kbb |
| u | redundant U layout `[:, 1:91, :30]` | `0 / 399600` | VERIFIED Kbb |
| v | redundant V layout `[1:, :90, :30]` | `0 / 399600` | VERIFIED Kbb |
| ssh | T-point `[:, :90]` | `0 / 13320` | VERIFIED Kbb |

This is a byte-identity claim, not a tolerance claim. It confirms the ocean
part of the domain and initial-state pipeline on the only detailed oracle
subdomain recorded in Phase 1.

## 4. Coverage and time-level registry

### Card state and inputs

| family | representation/disposition at this stop |
|---|---|
| grid coordinates, metrics, raw `ff_t/ff_f`, fold descriptor | VERIFIED static |
| `e3t/e3u/e3v/e3f`, T/U/V/F masks, bottom levels, bathymetry and face depths | VERIFIED static |
| ocean `T,S,u,v,eta` | VERIFIED at kt=1 `Kbb=1` on rank-0 owned strip |
| ocean redundant edge rows/columns outside scored staggering | VERIFIED structurally; not claimed against absent rank-1 detailed records |
| ocean `H_bathy`, land mask and partial-cell coordinate carriers | VERIFIED static against domain/mesh files |
| barotropic seeds and future `Kmm/Krhs/Kaa` stage state | UNMEASURED; first consumed after this stop |
| TKE `en`, `avm/avt/avs` and closure carry | UNMEASURED at card entry; Phase-1 oracle coverage exists but no legoESM comparison was made |
| forcing descriptors/weights: NCAR/CORE, runoff, geothermal, IWM, viscosity, RGB chlorophyll | REGISTERED UNMEASURED; no substitute field was supplied |
| SI3 core: `v_i,v_s,a_i,t_su,u_ice,v_ice,oa_i,a_ip,v_ip,v_il` | UNMEASURED_STOP |
| SI3 10 ice-energy and 5 snow-energy layers; salinity profiles | UNMEASURED_STOP; absent from current shared card state |
| every Prather first/second-moment family for area, volume, age, energy, salinity and ponds | UNMEASURED_STOP; absent from current shared card state |
| SI3 stresses/deformation and snow-ice mass carry | UNMEASURED_STOP |
| iceberg gridded stores and 17 particle-state fields | UNMEASURED_STOP; no shared card state |
| TOP/PISCES | WAIVED by Phase-1 oracle definition; compile-excluded after no-feedback proof |
| XIOS | WAIVED by Phase-1 oracle definition |

The generic shared `DynamicSeaIceState` can carry five category means, but it
does not by itself represent the complete 210-variable SI3 restart registry
or 21-variable iceberg registry. Treating its missing layered/moment fields as
zero would violate the no-Frankenstein rule.

### Time levels

| field/family | registered level |
|---|---|
| input T/S and zero U/V | initialized into `Kbb`; copied to `Kmm` by `istate.F90:135-137` |
| initial SSH | `Kbb` and `Kmm`, including identical ice-mass adjustment at `iceistate.F90:403-427` |
| gate entry record | `Kbb=1`; header value is checked, never inferred from the filename |
| geometry, metrics, masks, reference scale factors | static/non-slot operands |
| future WS-RK3 stage 1 | preregistered `(Kbb,Kmm,Krhs,Kaa)=(1,1,3,3)`, NOT ENTERED |
| SI3 state | its current ice level before the first odd ice step, UNMEASURED; not assigned an ocean K slot |
| iceberg state | initialization state before `icb_stp`, UNMEASURED; not assigned an ocean K slot |

No dumped array lacks a time-level disposition at the boundary actually
measured.

## 5. Binding controls

The baseline exits 0. Every plant traverses the same production validator and
exits 1:

| plant | exit | observed binding failure |
|---|---:|---|
| raw grid | 1 | `geometry mismatch: e1t` |
| north fold | 1 | `NEMO mesh mismatch: vmask` |
| dummy level | 1 | `dummy-level plant` |
| interpolation reassociation | 1 | `T (57592/399600)` |
| T payload | 1 | `T (1/399600)` |
| S payload | 1 | `S (1/399600)` |
| zero velocity | 1 | `u (1/399600)` |
| unstripped halo | 1 | `T (-1/399600)` shape failure |
| coverage omission | 1 | `resolved-feature coverage registry` |

## 6. Ordered stopping decision

**CONFIRMED:** neither
`origin/fidelity/nemo-testcases-l3-si3dyn-codex` at `555b67f85837` nor
`origin/fidelity/nemo-testcases-l3-si3thd-codex` at `dad9dd304872` is an
ancestor of this branch. Those lines contain the certified Lane-3 aEVP,
Prather and coupled-column work, but they also contain distinct histories.
Selecting and reconciling them is a cross-lane integration decision, not
authorization to recreate their arithmetic here.

**CONFIRMED:** the accepted Phase-1 oracle has detailed SI3 thermo, ZDF,
reassociation, Prather and exchange streams plus terminal SI3/iceberg restart
shards, but no single complete kt=1 SI3+iceberg step-entry record analogous to
the ocean entry stream. Therefore a complete `0/n` B2-I assertion cannot be
manufactured from the existing payloads.

The next review decision is whether to land/reconcile the certified Lane-3
shared state first and then add a WRITE-only complete SI3/iceberg kt=1 entry
record, or explicitly split B2-I from the ocean ordered walk. Until that is
decided, the execution guard remains active and B3a is not entered. No NEMO
run directory is prepared because the required state schema depends on that
integration decision.

## 7. ASKED / UNASKED

| item | status | action |
|---|---|---|
| start from merged `5bbb9ee56ad8` | ASKED | exact ancestry retained |
| new ORCA2 card on shared NEMO WS-RK3 identity only | ASKED | shared recipe/grid/model code; no card-local step arithmetic |
| full global card; rank-0-owned detailed comparison | ASKED | `180x148x30` card; `90x148x30` score |
| CPU, fp64, scalar-libm, production JIT | ASKED | pinned in retained gate run/card |
| stop at first boundary requiring a decision | ASKED | stopped at B2-I |
| do not enter SI3 dynamics/thermo before the ocean step-1 chain | ASKED | neither SI3 operators nor ocean stages entered |
| add raw `ff_t/ff_f` preservation to shared tripole loader | UNASKED implementation fact | required by `domhgr`; exact and shared |
| admit vector/EEN/c2/no-Aimp in shared WS-RK3 composition guard | UNASKED implementation fact | selector-only extension; operator already canonical |
| initialize ocean SSH with SI3 mass before representing SI3 state | UNASKED source fact | required for ocean B2-O by `iceistate`; exact |
| mark eight selected mechanisms unresolved and block execution | UNASKED safety action | fail-closed; no silent omission |
| reconcile Lane-3 histories or define a new SI3 entry schema | UNASKED cross-lane decision | not taken |

## 8. Explicit needs after review

Before B3a can be measured:

1. choose and integrate the already certified shared Lane-3 state/operator
   ancestry without copying formulas into ORCA2;
2. extend that shared state to the complete selected SI3 `jpl=5`, 10-ice-layer,
   5-snow-layer, Prather-moment and iceberg registry, or document which
   certified fields deliberately remain separate;
3. add and run a rank-0, WRITE-only, complete kt=1 SI3/iceberg entry writer if
   B2-I remains a mandatory unified boundary;
4. clear B2-I at `0/n` with a binding plant and remove only the corresponding
   execution-guard items;
5. then implement/validate NEMO `fld_read` forcing in shared code: CORE/NCAR
   interpolation and weights, runoff/freshwater carry, geothermal, spatial
   viscosity, IWM, RGB chlorophyll and QSR;
6. walk the ocean stage order using the unchanged Lane-1/2 oracle-relative
   cellwise gates, scalar-libm policy, QCO 65-substep external-mode identity,
   census-round BBL geometry/raw-W ladder and isomorphism tripwire;
7. only after the ocean step-1 chain clears, enter Lane-3 SI3 dynamics and
   thermodynamics unchanged.

This is the independent-review stopping receipt. Phase 2 is not closed; it is
mechanically stopped at B2-I before any unverified numerical execution.
