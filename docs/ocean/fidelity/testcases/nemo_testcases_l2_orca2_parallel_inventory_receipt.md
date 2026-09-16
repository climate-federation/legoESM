# NEMO-testcases L2 ORCA2 parallel inventory receipt

Date: 2026-09-16. Status: **HELD AT PRE-KT1 ADMISSION**. The current tree
cannot construct a native ORCA2 testcase, while a completed native NEMO record
already exists; therefore this round made no numerical candidate and requests
no NEMO acquisition. (`orca2_inventory.json:2-24`,
`orca2_inventory.json:226-245`)

## 1. The newest ORCA2 spec

The newest receipt says: **resolve the ORCA2 integrator, then record native
production-step TKE boundaries and inputs with the same stamp, exact-EOF, and
one-ULP controls**. This is the governing spec, not one of the broader older
ORCA2 carry-forwards. (`round101_tke_statement_boundary_receipt.md:193-203`)

The compiled read-only target resolves that first clause: its preprocessing
card selects `key_RK3`, `key_qco`, `key_vco_1d3d`, and `key_si3`; its compiled
driver executes the external mode followed by RK3 stages 1, 2, and 3; and the
compiled closure calls `tke_tke` before `tke_avn`.
(`cpp_ORCA2_OMIP_L4.fcm:1`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/stprk3.f90:198-227`,
`ORCA2_OMIP_L4/BLD/ppsrc/nemo/zdftke.f90:188-194`)

## 2. Inventory

| item | classification today | evidence |
|---|---|---|
| ORCA2 NEMO card/config in this checkout | **MISSING**. The live testcase dispatch contains only LOCK, OVERFLOW, and GYRE; neither an ORCA2 builder nor the execution guard is present. | `nemo_testcase_recipe.py:1014-1026`; `orca2_inventory.json:6-20` |
| Historical legoESM ORCA2 source card | **BUILT, NOT RUNNABLE** at historical revision `b7ce08cc8afa`: it reads the native domain and monthly T/S deck, but carries six named unmeasured selected arms and its execution validator refuses any nonempty list. | `historical_nemo_testcase_recipe.py:1008-1030`; `historical_nemo_testcase_recipe.py:1252-1275`; `historical_nemo_testcase_recipe.py:1430-1459`; `orca2_inventory.json:27-40` |
| Read-only NEMO oracle target | **BUILT; NOT RUNNABLE AS STAGED**. `ORCA2_OMIP_L4` is the only executable ORCA2 target; its template `EXP00` lacks the domain and two initial-condition files named by the namelist. | `orca2_inventory.json:68-105`; `ORCA2_OMIP_L4/EXP00/namelist_cfg:19-55` |
| Other ORCA2 configs under the read-only oracle tree | **REFERENCE CONFIGS, NOT BUILT**. `ORCA2_ICE_PISCES`, `ORCA2_OFF_PISCES`, `ORCA2_OFF_TRC`, `ORCA2_SAS_ICE`, and `X3_ORCA2_ICE_PISCES` have `EXPREF` cards but no executable or `EXP00` namelist. | `orca2_inventory.json:43-91` |
| Raw ORCA2 record under `phase3/` | **MISSING**. The scan found 33 ORCA2-named derived artifacts and zero ORCA2 raw `.bin` records. | `orca2_inventory.json:107-117` |
| Existing record outside `phase3/` | **COMPLETE NATIVE RECORD**. Both independent Phase-2v runs ended at kt=10 with matching binary/deck/input provenance; each holds 101 oracle streams. | `orca2_inventory.json:226-245`; `phase2v_a_run.user.time.log:1-5`; `phase2v_b_run.user.time.log:1-5` |
| Native kt=1--3 oracle coverage | **PRESENT AND TWIN-EXACT** for three step entries, nine RK3 stage exits, and the kt=2 TKE walk: all 13 measured A/B hashes match. | `orca2_inventory.json:146-225` |
| legoESM tripolar grid | **BUILT** as the generic `create_tripole_grid` loader. It is necessary but does not itself make an ORCA2 recipe executable. | `tripole.py:401-419`; `orca2_inventory.json:6-20` |
| legoESM ORCA2 recipe | **MISSING FROM CURRENT TREE**; the historical constructor exists only in the separate ORCA2 lineage. | `orca2_inventory.json:6-20`; `orca2_inventory.json:27-40` |
| legoESM exact ORCA2 forcing | **MISSING FROM CURRENT TREE**; the historical lane has the `nemo_fld_read` implementation required by its card. | `orca2_inventory.json:6-20`; `orca2_inventory.json:27-40`; `historical_nemo_fld_read.py:1-14` |

The native NEMO namelist is a from-rest ORCA2-ICE-PISCES card with
`rn_Dt=10800`, domain file `ORCA_R2_zps_domcfg`, monthly potential-temperature
and salinity inputs, NCAR bulk forcing, SI3, and a 2-call surface/ice cadence.
(`ORCA2_OMIP_L4/EXP00/namelist_cfg:19-55`,
`ORCA2_OMIP_L4/EXP00/namelist_cfg:77-114`,
`cpp_ORCA2_OMIP_L4.fcm:1`, `phase2v_a_ocean.output:224-239`)

## 3. What this round measured

The read-only probe measured the admission predicates and record inventory; it
did not import or execute the ocean model and did not run NEMO.
Its unplanted arm reports `kt1_3_from_rest_possible=false` and
`acquisition_needed=false`.
(`nemo_testcase_l2_orca2_parallel_inventory.py:2-9`,
`orca2_inventory.json:2-24`)

| first-over-bar boundary | result today | disposition |
|---|---|---|
| pre-kt1 native-card admission | **REFUSE**: current ORCA2 dispatch, builder, execution guard, and exact forcing reader are absent | this is the first failed predicate; no numerical row exists (`orca2_inventory.json:6-24`) |
| kt=1, kt=2, kt=3 T/S/U/V/SSH and RK3 stages | **NOT EXECUTED** | **NOT MEASURED**; running a lat-lon or historical surrogate would not discharge the current native card (`orca2_inventory.json:21-25`) |

This is not a claim that ORCA2 has never been measured. Existing phase3
derived evidence records a bit-exact native entry but an explicit
`execution_ready=false` guard, a production-JIT W/transport DEBT row, and an
interior HPG AT-BAR row whose scope excludes the north fold, cyclic seam, and
level 31. (`round24_orca2_entry_stage.json:2-48`,
`round24_orca2_production_w.json:26-62`,
`round24_orca2_production_w.json:82-101`,
`round27_orca2_hpg_model_arm.json:6-17`,
`round27_orca2_hpg_model_arm.json:89-141`)

The older ORCA2 lineage also reached a production-JIT TKE first boundary:
`taum`, the ice attenuation, and `zWlc2` were exact, then the `zpelc`
accumulation differed in 39,290 / 242,135 owned cells with maximum four
row-scale ULP. This receipt registers that historical result as a prediction
for a future current-tree run, not a substitute for one.
(`phase2w_handoff_receipt.md:47-71`)

## 4. Record and acquisition disposition

The current probe independently validated the pinned kt=2 TKE stream's header
and extents through exact EOF: 61,038,700 bytes, binary64, 17 three-dimensional
and seven two-dimensional fields, with SHA-256
`31675493f022f71a609142f53bbe220c111b09e9a9a352926a1aff7358770a52`.
(`orca2_inventory.json:119-144`)

The older Phase-2w prose says 57,266,632 bytes even though its same paragraph
states the 17+7-field header/payload and pins the same SHA-256. The live file,
header-derived EOF, and pinned hash agree with 61,038,700 bytes; this receipt
therefore treats the older byte count as stale prose, not record corruption.
(`phase2w_handoff_receipt.md:10-18`,
`phase2w_handoff_receipt.md:35-45`, `orca2_inventory.json:119-144`)

The historical admission already bound magic, extent, count, truncation,
trailing-byte, canonical-slot, undefined-workspace, and twin controls, and its
TKE walk bound target-bit plants through the first differing statement.
(`phase2w_handoff_receipt.md:35-45`,
`phase2w_handoff_receipt.md:47-71`)

**ACQUISITION_NEEDED: NONE.** Re-acquiring NEMO would not repair the missing
current legoESM recipe. **Plan inference:** the newest same-stamp/one-ULP
wording is instead an OPEN consumer-side admission requirement: the current
reader must restamp this immutable record, prove exact EOF again, and fire a
one-ULP plant before any new row is promoted.
(`round101_tke_statement_boundary_receipt.md:193-203`,
`orca2_inventory.json:2-5`, `orca2_inventory.json:119-144`)

## 5. Controls and tests

The planted missing-record arm exited 2, printed a named `REFUSE:` line, and
created no output artifact. (`orca2_inventory_plant.txt:1-4`)

The receipt gate pinned all 44 file/line citations with no failures, unmapped
citations, or unused entries. Its one-line-shift plant changed the registered
`orca2_inventory.json:2-5` payload, produced the single expected failure, and
exited nonzero. (`orca2_citation_gate_summary.txt:1-7`)

Focused tests exercise literal dispatch discovery, historical guard discovery,
exact-EOF acceptance, trailing-byte refusal, and the fail-closed predicate.
The citation unit guards cover extraction, de-duplication, range parsing, and a
shift/refusal path. Final result: **6 passed in 0.07s**.
(`test_nemo_testcase_l2_orca2_parallel_inventory.py:51-81`,
`test_nemo_testcase_l2_orca2_parallel_receipt_gate.py:22-31`,
`orca2_final_tests.txt:1-6`)

Independent in-sandbox review: **independent review unavailable in-sandbox**.
The required command exited 1 before reviewing the diff because its app-server
client could not initialize in the read-only sandbox. (`orca2_review.txt:1-6`)

## 6. OPEN — ordered first-measurement plan

1. Forward-port the historical native constructor, exact forcing reader, and
   record readers onto a fresh held ORCA2 integration branch based on the
   current tip; do not merge historical production physics wholesale. Preserve
   the native domain/T/S inputs and the card's fail-closed guard.
   (`historical_nemo_testcase_recipe.py:1008-1030`,
   `historical_nemo_testcase_recipe.py:1252-1275`,
   `historical_nemo_testcase_recipe.py:1430-1459`)
2. Resolve each of the six named selected arms against the compiled ORCA2
   branch and existing record. Do not clear the execution guard merely to get
   a trajectory. (`orca2_inventory.json:27-40`)
3. Add a current-schema reader/admission bridge for the immutable Phase-2v
   entries, stages, and TKE frame. Require record provenance, exact EOF, and a
   one-ULP consumed-value plant; explicitly map which newest-spec boundaries
   the older TKE schema contains before deciding that another WRITE-only frame
   is necessary. (`orca2_inventory.json:119-225`,
   `round101_tke_statement_boundary_receipt.md:193-203`)
4. Only after the native card passes its execution validator, run from rest on
   CPU/fp64 for kt=1--3 and score T/S/U/V/SSH plus all three RK3 stage exits
   against the 12 native entry/stage frames. Stop at the first exact-bar
   violation and report the complete first-over-bar table. The record already
   supplies those frames. (`orca2_inventory.json:145-225`)
5. Re-enter the TKE walk at the first current production-step boundary. Treat
   the historical `zpelc` result as a prediction to confirm or refute, and do
   not promote isolated or historical arithmetic as a current production-step
   result. (`phase2w_handoff_receipt.md:47-71`)
6. Request a new NEMO acquisition only if the boundary-map audit proves that a
   required native operand is absent. No such absence was established in this
   inventory. (`orca2_inventory.json:2-5`, `orca2_inventory.json:119-225`)

No configuration or carried-state decision is needed for this first plan: the
native recorded card already fixes RK3, QCO/ZPS, NCAR forcing, SI3, and the
from-rest deck. (`cpp_ORCA2_OMIP_L4.fcm:1`,
`ORCA2_OMIP_L4/EXP00/namelist_cfg:19-55`,
`ORCA2_OMIP_L4/EXP00/namelist_cfg:77-114`,
`phase2v_a_ocean.output:224-239`)
