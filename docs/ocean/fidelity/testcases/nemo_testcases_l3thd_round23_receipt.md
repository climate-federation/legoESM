# Lane 3b round 23 receipt — ORCA1-ice real-geometry gate preparation

Date: 2026-09-06

Tracker: `climate-federation/legoESM#1699`

Parent: `2bbea6d612416c4a3d5e7fb589e3d85c57875a36`

Preregistration commit: `26ed581bb263`

Gate implementation commit: `d8583721d5e85de617714deb6c4db871efb30e50`

Status: **PREPARATION COMPLETE; EXACT PHYSICS SCORE UNMEASURED.**  The parser,
identity admission, coverage registry, duplicate-writer bridge, and controls
are ready.  The retained frames do not contain the state after frazil and
before ZDF, so no shared-column result is claimed.

## Decision 11 and pre-implementation search

User Decision 11 selects ORCA1's resolved identity: one category, 3+3 layers,
BL99/P07, salinity option 2 with `rn_sinew=0.75`, ponds and lateral melt off,
L16 landfast, aEVP, and Prather.  The five-category rung is dropped.

Before implementation, the search found and reused:

- the existing Lane-3 admission gate
  `scripts/validate/ocean_fidelity/testcases/nemo_orca2_si3_exact_input_gate.py`;
- `SI3ThermoConfig` and the shared `si3_column_step_arrays` implementation;
- the shared SI3 exchange operations in `ice/sea_ice.py` and
  `coupler/ocean_forcing.py`; and
- Lane 4's self-describing writer at remote commit `b7ce08cc8afa`, blob
  SHA-256 `b66823d02f35e60c9fa315c73ba9d51235b5bb88aa45247d0ed7142cb1d64bdf`.

No thermodynamic model, exchange module, or ORCA2 instrument was added.  This
round changes only the gate, its tests, and documentation.

## Executed identity and call order

The gate resolves all selectors from each run's `ocean.output` and fails
closed on a mismatch.  Both phase-2x roots resolve `jpl=1`, `nlay_i=3`,
`nlay_s=3`, P07, `nn_icesal=2`, `rn_sinew=0.75`, drainage/flushing on,
ponds/LEV/lateral melt/virtual ITD off, thermodynamic and open-water growth on,
L16 landfast, H79 strength, ridging/rafting, EVP+aEVP, and Prather.  It also
verifies `nn_fsbc=2`, `rDt_ice=21600 s`, and calls at ocean steps 1/3/5
(`icestp.F90:126,377-380`; `sbcmod.F90:474-477,604`).

For this namelist the registered NEMO order is:

| order | call | resolved disposition | NEMO 5.0.2 source |
|---:|---|---|---|
| 1 | `ice_thd_frazil` | executed | `icethd.F90:115` |
| 2 | active selection, `ice_thd_1d2d(1)` | executed | `icethd.F90:117-140` |
| 3 | `ice_thd_zdf` BL99/P07 | executed | `icethd.F90:143-148` |
| 4 | `ice_thd_dh` | executed, `ln_icedH=T` | `icethd.F90:150` |
| 5 | `ice_thd_temp` | executed | `icethd.F90:152` |
| 6 | `ice_thd_sal` option 2 | executed | `icethd.F90:154` |
| 7 | `ice_thd_temp` | executed | `icethd.F90:156` |
| 8 | `ice_thd_mono`, `ice_thd_da` | skipped | `icethd.F90:158-161` |
| 9 | `ice_thd_1d2d(2)` | executed | `icethd.F90:163` |
| 10 | ponds and category remap | skipped, `ln_pnd=F`, `jpl=1` | `icethd.F90:176-179` |
| 11 | `ice_thd_do` | executed, `ln_icedO=T` | `icethd.F90:181` |
| 12 | correction, aging, LBC | executed | `icethd.F90:183-190` |

## Frame registry and coverage

Lane 4's f0 frame is global state immediately before `ice_thd_frazil`; f1 is
global state after correction/LBC.  The eight flux arrays retain their
`ice_sbc_flx` ENTRY time level in both frames because `ice_thd` operates on
selected 1-D copies.  State-to-1-D conversion is at `icethd.F90:343-371`.

| fields | extent/time level | loaded | duplicate-writer bridge | physics |
|---|---|---|---|---|
| `a_i,v_i,v_s,sv_i,t_su` | 94x152x1, f0 PRE_FRAZIL / f1 POST_CORRECTION | VERIFIED | VERIFIED_BIT_IDENTICAL | UNMEASURED |
| `e_i,szv_i` | 94x152x3x1, same levels | VERIFIED | VERIFIED_BIT_IDENTICAL | UNMEASURED |
| `e_s` | 94x152x3x1, same levels | VERIFIED | VERIFIED_BIT_IDENTICAL | UNMEASURED |
| `qns_ice,qsr_ice,dqns_ice,evap_ice` | 90x148x1, `ice_sbc_flx` ENTRY | VERIFIED | WAIVED_NO_SECOND_WRITER | UNMEASURED |
| `qprec_ice` | 90x148, `ice_sbc_flx` ENTRY | VERIFIED | WAIVED_NO_SECOND_WRITER | UNMEASURED |
| `qml_ice,qcn_ice,qtr_ice_top` | 90x148x1, `ice_flx_other`/bulk ENTRY | VERIFIED | WAIVED_NO_SECOND_WRITER | UNMEASURED |

Every file is parsed from its own 16-field extent table.  Magic, version,
binary width, selector dimensions, field extents, derived payload count,
finite values, filename/header step and frame, sequence, and exact EOF are
checked.  State and flux allocations are not falsely assumed to have the same
halo extent.

## Measurements

The retained corrected twins are:

- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_a_10step_np2`
- `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_b_10step_np2`

Both reached `time.step=10`.  The six self-describing frames, legacy 40-frame
call-order stream, and five-frame ZDF-input stream are byte-identical between
A and B (8 / 8 files).  Their common state fields compare as follows:

| root | files/frames admitted | common values | non-bit | exact physics rows | result |
|---|---:|---:|---:|---:|---|
| phase2x A | 6 new + 40 legacy | 1,200,192 | **0** | 0 | `STOP_MISSING_OPERANDS` |
| phase2x B | 6 new + 40 legacy | 1,200,192 | **0** | 0 | `STOP_MISSING_OPERANDS` |

This confirms the writers agree; it does **not** certify SI3 thermodynamics or
exchange on the ORCA2 geometry.  Although PRE_FRAZIL active counts equal the
ZDF-input `npti` at kt1/3/5 (1777/1854/1880), count equality does not establish
unchanged indices, geometry, enthalpy, salinity, or snow temperature across
`ice_thd_frazil`.

The preregistered prediction of 1,700 PRE_FRAZIL points and 77 kt1 activations
is therefore **REFUTED** for the completed phase-2x twins.  The preregistration
is left unchanged as the before-measurement record.

The two failed phase-2w roots have no thermodynamic records.  Each exits 2 with
a named `GateError` (`no ORCA1-ice thermodynamic records`); neither is admitted
through an empty or partial score.

The A/B byte identity is a Lane-3 diagnostic, not a replacement for Lane 4's
oracle provenance receipt, which is not yet present on its fetched tip
`b7ce08cc8afa`.  The gate labels the records provisional for that reason.

## First unmeasured boundary and exact handoff

The first unmeasured boundary is **POST_FRAZIL_PRE_ZDF_1D**, after
`CALL ice_thd_1d2d(jl,1)` at `icethd.F90:140` and before initialization/ZDF at
`:143-148`.  Lane 4 should add one self-describing WRITE-only frame with global
cell identities `nptidx` and the same-time-level:

`a_i_1d,h_i_1d,h_s_1d,t_su_1d,e_i_1d,e_s_1d,s_i_1d,sz_i_1d,oa_i_1d,t_s_1d`.

It must also carry the 13 existing forcing operands:

`qns_ice_1d,qsr_ice_1d,dqns_ice_1d,qtr_ice_top_1d,t_bo_1d,sss_1d,evap_ice_1d,sprecip_1d,qprec_ice_1d,qcn_ice_1d,qsb_ice_bot_1d,fhld_1d,qlead_1d`.

Those conversions are `icethd.F90:343-435`.  For boundary-by-boundary scoring,
add equally self-describing outputs after ZDF, DH, TEMP1, SAL, TEMP2, DO, and
EXIT.  The legacy values exist but do not independently encode per-field
extents or the missing PRE_ZDF state.

Full exchange scoring also requires every rank-zero wet-cell operand/output,
plus `ssmask`, around:

- `blk_ice_1/blk_ice_2`, `sbcblk.F90:1218-1346`;
- `ice_flx_other`, `icesbc.F90:307-437`; and
- `ice_update_flx`, `icestp.F90:201-213` / `iceupdate.F90:105-194`.

The retained bulk record is one point and cannot support a cellwise exchange
claim.  This is a WRITE-only handoff; Lane 3 did not build an ORCA2 instrument.

## Controls and tests

Synthetic schema controls bind for magic, version, extent, field count,
payload count, truncated and trailing bytes, filename/header step, missing
frame, selector, and cadence.  All eight real-data row plants (`a_i`, `v_i`,
`v_s`, `sv_i`, `t_su`, `e_i`, `e_s`, `szv_i`) exit nonzero as
`STOP_COMMON_WRITER_NONBIT`, with exactly one non-bit value first reported at
the named kt1/f0 row.

- focused gate tests: **7 passed**;
- neighboring SI3 fidelity tests: **57 passed** in 69.46 s;
- constants ratchet restricted to the two touched Python files: **2 passed**;
- full constants ratchet: 3975 passed, 2 skipped, 6 failed.  The failures are
  pre-existing discovery/root and unrelated FV3/DINO literal rows; neither
  touched file failed.

## C1D blind spots and this vehicle

| behaviour the one-column rung could not exercise | disposition here |
|---|---|
| spatially varying ice/snow state and branch census | inputs exposed; physics UNMEASURED pending PRE_ZDF frame |
| spatially varying atmosphere/ice and ocean/ice exchange | 16 outputs loaded; UNMEASURED pending full operand frames and wet mask |
| fold/halo canonicalization | duplicate writers VERIFIED bit-identical; physical fold/LBC operator UNMEASURED |
| partial-cell influence on `t_bo`, basal heat and lead heat | UNMEASURED pending `ice_flx_other` operands |
| `nn_fsbc=2` cadence | VERIFIED for frame steps 1/3/5; thermodynamic temporal response UNMEASURED |
| landfast L16, aEVP, Prather | selectors VERIFIED; dynamics/transport remain Lane 3a scope |
| real-geometry thermodynamic sub-call response | first boundary identified; zero physics rows scored |

The C1D-certified identity remains: single-category HFN, 3+3 BL99/P07,
salinity 2 with `rn_sinew=0.75`, drainage/flushing, thermodynamic/open-water
growth, no ponds, no lateral melt, and its covered SI3 exchange path.  C1D did
not execute spatial dynamics, Prather transport, ridging/rafting, or landfast.

The coupled-slab kt2 row remains GYRE-owned.  No slab seed or prognostic
`uu_b/vv_b` code was changed or re-scored in this preparation-only round.

## Provenance and artifact hashes

Runtime summaries live outside Git at
`/data/abyssal/dbalwada/nemo-testcases-l3/round23_orca1ice_preparation/`
(344 KiB).  `artifacts.sha256` is SHA-256
`3bde03cd4fbcd6f0118e39f44d07baed1424253c56c1d620f8f5bef006c92f0b`.
The A/B identity manifest is
`6763bae8c556a3ee5302abd9ca7272e01160a8a74345256100427ed50e2b3af6`.
It pins:

| stream | bytes | SHA-256, identical A/B |
|---|---:|---|
| kt1 f0 | 2,453,052 | `0624d37f62f7c3c7de6f5e093bc2d76e8b4bae8d6ab0d47f5c5c4bbcb5daf29d` |
| kt1 f1 | 2,453,052 | `2eb52ee643e3d7ee334c3e10f37bda4e1afc56c7eb584ed403c24b85405b196e` |
| kt3 f0 | 2,453,052 | `639f317077cea460daa73bf8a27189f11a0c84a1d8013543bddd8f4af13feb23` |
| kt3 f1 | 2,453,052 | `72312a04e48c1d77ca3c775f519904eee14df6cd85b502e858c45d1615cc70ed` |
| kt5 f0 | 2,453,052 | `804a03cfa71c370770f882d7e5772b571f0b03ff5b18bd630c2e05986cba6bb4` |
| kt5 f1 | 2,453,052 | `c28486e4a2aa976c93925ea6d2abf67d742ce0bbec528c9ab0d72568cd1025b2` |
| legacy thermodynamics | 35,715,040 | `c37d7e003d37ae23e8ec59ae5b59ef3c780ab2cf1b50536fb81dfd1c7022ae44` |
| legacy ZDF inputs | 1,194,184 | `1d3a07447bbe86afd87bbc6175d5f6a19dd51c88db6f3b9833b84b1eba0c5508` |

Committed source hashes before this receipt are:

- preregistration: `2996c6cb90da0a5bfcf840a96dd1ec6615c887f3a938b1642a8d5bcc7015a316`;
- gate: `297eeac3fd1ffbd67b5f70e0bc7311c14f3fa793bbdf14296bdcb99ac2c64539`;
- unit tests: `b13a3849c63bc2ae4e4c9adff5ef436ecd0c6ba7f11b2f1931923a56130ce0db`.

No runtime binary or JSON is committed.

## ASKED / UNASKED and retained evidence

| action/choice | status | disposition |
|---|---|---|
| select ORCA1 single-category identity; drop five-category rung | ASKED, Decision 11 | applied |
| build parser/gate and dry-run failed/synthetic records | ASKED | complete |
| use completed phase2x twins before Lane-4 receipt | UNASKED as final oracle pin | diagnostic and explicitly provisional |
| request missing WRITE-only operands | ASKED conditional | exact handoff above |
| implement multi-category, landfast/aEVP/Prather dynamics | UNASKED / Lane 3a | not done |
| alter slab seed or carried barotropic state | GYRE-owned | not done |
| modify shipped NEMO, delete evidence, use GPU/`mpirun`, or push | forbidden | not done |

No files or run roots were deleted.  The failed phase2w A/B roots remain
flagged for future cleanup.  The previously flagged r13/r17/r18 diagnostic
roots remain flagged by the round-20 receipt; this round neither removes nor
reclassifies them.

