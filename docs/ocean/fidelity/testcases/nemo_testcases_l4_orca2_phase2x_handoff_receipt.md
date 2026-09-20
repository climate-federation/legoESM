# NEMO testcase Lane 4 — ORCA2 Phase-2x handoff receipt

Date: 2026-09-06

Parent: `b7ce08cc8afa5cf377922abf198cf1794fab8a73`

Status: **CONFIRMED STAGED; MPI NOT EXECUTED.**  This receipt stops at the two
replacement launchers.  It does not admit an oracle root and does not perform
legoESM card work.

## 1. Phase-2w failure retained

**CONFIRMED:** both Phase-2w twins are
`REJECTED_INIT_FAILURE`, retained unchanged at
`variant_orca1ice_phase2w_{a,b}_10step_np2`.  Both launchers returned 123
before timestep 1.  `ocean.output:782-804` resolves `ln_pnd=F` together with
the inherited `ln_pnd_LEV=T` and stops with:
`ice_thd_pnd_init: choose either none ... or only one pond scheme`.
The common `ocean.output` SHA-256 is
`6893af937db77d51bf3c6d8d5b73cc9630b37a07f2d83db5920a0516078f7b05`;
the common stdout SHA-256 is
`21405a6821e0bea33bd259dee5eda94d1561c6b373f3cb81f9b714198aeec1e4`.
The A/B timing-log hashes are respectively
`2fd8070e1a85668e6c31b204dd368e6af00862ded7fa9c499fc4f63272b8fc0a`
and `89e69f4cedc5b41c5beb24e24d0b561c24059a585dd2e57416836bdcb45eea04`.
Nothing was deleted.

## 2. Field-complete ORCA1 ice namelist resolution

**CONFIRMED:** the committed gate compares every active assignment made by
`/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg` with the complete
Phase-2x override.  It reports 91 ORCA1 rows, 90 exact rows, one registered
difference, and zero candidate-only rows.  The external row-by-row JSON is
`/data/abyssal/dbalwada/nemo-testcases-l4/phase2x/orca1ice_namelist_audit.json`
(SHA-256
`6eb443852d20068780b6293142a6a73b7e719830ed39ad9f236598f004bcac73`).
The `ln_pnd_LEV` plant exits nonzero (log SHA-256
`a61da4000eb4301695fcc6c6cc1520fc981590930bebc2a0b5b8a57ca1977333`).

| ORCA1 block | rows checked | disposition |
|---|---:|---|
| `nampar` | 8 | 8 exact |
| `namitd` | 3 | 3 exact |
| `namdyn` | 3 | 3 exact |
| `namdyn_rdgrft` | 12 | 12 exact |
| `namdyn_rhg` | 1 | 1 exact |
| `namdyn_adv` | 2 | 2 exact |
| `namsbc` | 2 | 2 exact |
| `namthd` | 1 | 1 exact |
| `namthd_zdf` | 2 | 2 exact |
| `namthd_do` | 1 | 1 exact |
| `namthd_sal` | 23 | 23 exact |
| `namthd_pnd` | 2 | 2 exact, including `ln_pnd_LEV=F` |
| `namini` | 30 | 29 exact; one deliberate difference below |
| `namdia` | 1 | 1 exact |

There is exactly one deliberate difference:

| field | ORCA1 | Phase-2x | reason |
|---|---:|---:|---|
| `namini.nn_iceini_file` | 1 | 0 | `Ice_initialization` is an ORCA1-grid file absent from the ORCA2 input deck; NEMO's analytic SST initialization is the smallest faithful ORCA2 initial-state choice |

All ORCA1 scalar initialization values and inactive `sn_hti/sn_hts/sn_ati/
sn_smi/sn_tmi/sn_tsu/sn_tms` descriptors are nevertheless copied exactly.
Ocean geometry, forcing, `ln_icebergs=F`, run length and `(jpni,jpnj)=(2,1)`
remain those of the accepted ORCA2 variant.  The replacement ice namelist
SHA-256 is
`6b647863137b518b95ff97f83975d9afcb3944b7f6e8d63e45a05f494f5edc89`.

## 3. Source-artifact hygiene

**CONFIRMED:** no full NEMO source remains under
`scripts/validate/ocean_fidelity/orca2_l4/phase2w_orca1ice_MY_SRC/`.
The repository carries only unified patches.  Each patch applies to the
hash-pinned shipped NEMO 5.0.2 source and reproduces the external applied file
byte-for-byte (`PATCH_REPLAY_EXACT 3/3`; durable log SHA-256
`f7a5c60d698016f3f2c24684ab1210925a4d213cb0a9e0eefa2a7cc0983d9fbf`).
The shipped checkout was not edited.

| file | shipped base SHA-256 | patch SHA-256 | applied SHA-256 |
|---|---|---|---|
| `icethd.F90` | `8c5242271435a79139a9a4976dde366da129b4213728cd47f4fef0c60431e851` | `c6dfdb95e0da0fda35ee446ecc8afb08d641e91c33b50c7e9e6480432d92f0bc` | `b66823d02f35e60c9fa315c73ba9d51235b5bb88aa45247d0ed7142cb1d64bdf` |
| `icedyn_rhg_evp.F90` | `f1c92a8e815f28e9178acdd8e9e307f598b8057b329e8c2e2abece627271b3fa` | `893811b95ceea5353d3dc235a0b73ac9264cfe7e1c9095cfd3ab8c1c433bf27c` | `bf61835a80faaa38a49c49644a11a16fb45be1049caed8aa55cb224f62f5d7b1` |
| `icedyn_adv_pra.F90` | `879e42b44b864674500eb993413c0f9796c9b35b262d0fb008187be7af5fc9c6` | `429f688ffe3d725f6e86342493e472f493634b9538e4f5ee4deffebbeda9c7bd` | `0ff0fc32c0845f23e093a2ad7957da89a3eef13ab1dc741005edc30e0785aa24` |

The Phase-2x applied files are under
`/data/abyssal/dbalwada/nemo-testcases-l4/build/phase2x_orca1ice/MY_SRC_final/`.
The original Phase-2w applied files remain separately pinned under `MY_SRC/`.

## 4. ORCA1 production-driver audit

**CONFIRMED no-op for the ORCA1 card:** commit `bb1c60beba4d` changes exactly
four old lines to four new lines in `scripts/run/run_omip_core2.py`:

```diff
- if int(tke_eice) not in (0, 1, 3):
+ if int(tke_eice) not in (0, 1, 2, 3):
- expected 0, 1 or 3
+ expected 0, 1, 2 or 3
- choices=[0, 1, 3]
+ choices=[0, 1, 2, 3]
- 1 = (1-fi); 0 = no attenuation
+ 1 = 1-tanh(10*fi); 2 = 1-fi; 0 = no attenuation
```

The default remains `None`, so `build_tripole_vmix_config("tke")` continues
to construct the production ORCA1 `TKEConfig(eice=3)`.  Loading the driver
immediately before and after that commit gives exact complete-config repr
hashes of
`14948b719a437c1e641397163f17255cb31a76c9c5252b06fe98710d8ff0d40b`
on both sides.  The committed audit reports `resolved_config_exact=true` and
`before_eice=after_eice=3`; its external JSON SHA-256 is
`bc24cba72be9663c72203bcd594a94a6f5d520854b1ae2bc6bcd66831777eeb9`.
Changing the post-edit config to mode 2 is a binding nonzero plant (log hash
`5bb0f19f6aeca54f2d9d2cd2518cd3e7278ccadfe5b0a8636a648c7d5f7d9849`).
The retained change therefore admits and documents a selectable value; it
does not change an ORCA1 resolved value or behavior.

## 5. SI3 frame coverage

**CONFIRMED WRITE-only coverage:** the new dynamics call is after strength and
landfast operands are formed and before the EVP loop
(`icedyn_rhg_evp.F90:254-260,335-381`).  Its 34 self-described fields are:

- entry `u_ice/v_ice`, `utau_ice/vtau_ice`, `strength`, aggregate
  `at_i/vt_i/vt_s`, `a_i/v_i/v_s`, and incoming `stress1/stress2/stress12`;
- `ht/hu/hv(:,:,Kmm)`, `icb_tmask/icb_umask/icb_vmask`, and
  `fast_tmask/fast_umask/fast_vmask`;
- `zaU/zaV`, atmosphere-ice stress, ocean-drag coefficients,
  `ztaux_base/ztauy_base/tau_icebfr`, and mass coefficients `zmU_t/zmV_t`;
- scalar `rn_lf_depfra/rn_lf_bfr/rn_lf_relax/rn_lf_tensile/rn_crhg`.

The T masks are read and the U/V masks derived at `icedyn.F90:115-125`.
The basal operands are formed at `icedyn_rhg_evp.F90:335-363`; they are not
assigned inside the `jter=1..nn_nevp` loop beginning at line 381.  Thus the
single pre-loop `ztaux_base/ztauy_base/tau_icebfr` frame is the exact invariant
operand for every subcycle.  `rn_lf_tensile` becomes `zkt` at lines 257-260
and enters `zs1/zs2/zs12` at lines 457-461 and 487-489.  Fast U/V masks enter
the velocity updates at lines 579,630,685,737.

**CONFIRMED thermodynamic composite coverage:** `NEMO_L3THD_001` records
global entry/exit `a_i/v_i/v_s/sv_i/oa_i/t_su/a_ip/v_ip/v_il/e_i/e_s/
szv_i` and, in exact `icethd.F90` execution order, compressed-column
`a_i,h_i,h_s,t_su,e_i,e_s,sz_i` after ZDF, thickness, temperature, salinity,
and second-temperature calls (`icethd.F90:140-176,241-284`).
`NEMO_L3ZIN_002` adds `qns/qsr/dqns/qtr_top/t_bo/sss/evap/sprecip/qprec/
qcn/qsb_bot/fhld/qml/t_s`; `NEMO_L3REA_001` adds `t_i/t_s` and the
reassociation operands (`icethd.F90:287-327`).  The ORCA1-target 16-field
global record separately captures category state and atmosphere-ice fluxes
at entry and exit for ocean kt 1,3,5.  Prather's 35 moments are captured at
entry/exit.  This covers the ice-thermo column and ice-dynamics entry requests;
no requested cheap operand remains missing.

All new full-domain writes pass through the canonical zero-first helpers and
are guarded by `lwp`; headers carry each field's extents and SIZE-derived
payload.  The Phase-2x schema gate derives the 34-field allocation table from
the header and its nine magic/truncation/trailing-byte plants bind.  Real
schema, twin reproducibility, and instrument-inertness are
**PLAUSIBLE_PENDING_MPI**, never claimed from synthetic data.

## 6. Build and staged twins

The ice namelist itself needed no rebuild.  The requested six-field WRITE-only
coverage extension did, so the binary was rebuilt.  The accepted clean serial
scalar-math build finished successfully; build-log SHA-256 is
`fa7ad9a9668565b7bd741097cd8723d1708bc749a5f0f7bee25d14d7e4400320`.
The binary SHA-256 is
`8e40bf0b595eabba1bd428775a45e87f3f42a778333374cbbfb26eeb6c685869`;
`nm -D` finds zero `_ZGV*` symbols.  An earlier overlapping build log is
retained as `build_phase2x_orca1ice.log` and flagged
`REJECTED_OVERLAPPED_BUILD_LOG` (SHA-256
`8660fc88db8b137190ceeab0f1ee963550ef213ac9b33aeb2ad6d514079a5881`);
it is not cited as build evidence.

Both fresh directories use copied XML/namelists, symlinks to the existing
hash-pinned ORCA2 input files, and an absolute symlink to the Phase-2x binary.
Their common deck-manifest SHA-256 is
`51da69b494a10fa3c3b119018329a94d963f1fe3e59b6834ea936055ab0df2b9`;
their common input-manifest SHA-256 is
`3dfe251754fa76c8b5053cda90a51ee10589d0fffc01a4e799c49cc36bbd17e5`.
The CPU launcher uses two MPI ranks, one OpenMP thread, `(jpni,jpnj)=(2,1)`,
the bash `time` keyword, `tee`, and records `MPIRUN_RC`/`RUN DONE`.  It refuses
the wrong directory, any pre-existing output, and any hash mismatch.

Run these one at a time from the user shell, unchanged:

1. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_a_10step_np2/run.sh`
2. `/data/abyssal/dbalwada/nemo-testcases-l4/runs/variant_orca1ice_phase2x_b_10step_np2/run.sh`

Launcher SHA-256 values are respectively
`b27a1302ad5c55f785c141831ffbb0d57dc7dc23b80bea26bbde5f770e7d4ca0`
and `4fbb1f42c7d68eae10db7e36dd7fbd4bc54fc4c5014fd77cc5818a86a8b41ed3`.
Do not pin either root until both runs finish, all records decode to exact EOF,
A/B identity is complete, inherited outputs are inert, and plants bind.

## ASKED / UNASKED

| action | classification | disposition |
|---|---|---|
| user-shell MPI execution | ASKED | not performed here; two launchers handed off |
| retain failed Phase-2w twins | ASKED | CONFIRMED retained and flagged |
| resolve every ORCA1-set ice row | ASKED | CONFIRMED 90 exact + one registered input-grid difference |
| keep `nn_iceini_file=0` | ASKED by ORCA2-input constraint | CONFIRMED only deliberate difference |
| patch only `ln_pnd_LEV` | explicitly rejected | not done; complete 91-row resolution used |
| replace committed full NEMO sources | ASKED | CONFIRMED patches plus external applied files |
| alter ORCA1 CORE2 behavior | decision-gated | CONFIRMED no resolved change; `eice=3` remains exact |
| extend missing SI3 operands | ASKED | CONFIRMED six WRITE-only dynamics fields added |
| rebuild for a namelist-only edit | UNASKED | not done; rebuild was required solely by the writer extension |
| run MPI/NEMO in sandbox | forbidden | not done |
| edit shipped NEMO, delete, push, or commit large artifacts | forbidden | none |

## Boundary

This is a run handoff, not admission.  Resume only after both Phase-2x twins
have been executed by the user shell.  No legoESM work was done after the
driver's no-op audit.
