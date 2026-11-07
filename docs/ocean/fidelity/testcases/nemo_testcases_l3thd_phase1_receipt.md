# NEMO testcase oracle receipt — lane 3b, SI3 thermodynamics phase 1

Tracker: `climate-federation/legoESM#1699`

## Verdict and scope

Rung 3.5 is **VERIFIED as an oracle artifact**.  This receipt does not make a
legoESM comparison.  The accepted NEMO 5.0.2 run completed all 8,760 hourly
steps on one CPU process, directly (no `mpirun`, no GPU), read the official
ERA5 North Greenland file, wrote the final ocean and ice restarts, wrote every
registered SI3 thermodynamics and exchange frame, and passed the fail-closed
coverage/phenomenology gate.  “VERIFIED” below means only that the named
artifact was measured by the committed gate; it is not a claim of model
fidelity.

The committed preregistration is
`docs/ocean/fidelity/testcases/nemo_testcases_l3thd_phase1_preregister.md`.
Its parent was `d5e259fad2c45c92de76229daddc1cc5ba55c30a`; the accepted run used
the instrumentation committed through `20810e5fe`.  The scope-exact dynamic
selector register and strengthened gate were preregistered in `19d8243fe`.

## Immutable oracle and copied case

- Shipped oracle: `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`, NEMO git
  `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`.
- Source case: `cfgs/C1D`; `git status --short -- cfgs/C1D` was empty after
  build and run.  No shipped C1D file or shipped configuration was edited or
  deleted.
- New case: `cfgs/C1D_OMIP_L3`, initially made by a copy of `cfgs/C1D`, then
  given its own cpp file, `EXP_SASICE` namelists, and `MY_SRC` overrides.
- Components: `OCE SAS ICE`.  The executable resolves `step.F90` and
  `nemogcm.F90` from `src/SAS`.
- Keys: `key_linssh key_vco_1d3d key_si3`.  The copied C1D `key_xios`,
  `key_RK3`, and `key_top` selections are not active in this executable.
- Accepted run directory:
  `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scope_gate2`
  under the required root `/data/abyssal/dbalwada/nemo-testcases-l3/`.

The component/key choice is an empirical discriminator, not an inferred
substitution.  With `key_RK3`, `src/OCE/nemogcm.F90:166` called the ocean RK3
stepper despite SAS sources being present; the one-metre ocean evolved until
salinity reached the C1D stop value 100 at step 3178.  The official ERA5 `sss`
is exactly 34 throughout.  Removing RK3 selected `src/SAS/nemogcm.F90` and
`src/SAS/step.F90`, which prescribe the ocean state through `namsbc_sas` as
the shipped ICE_ADV tests do.  Failed discriminator directories are retained:

- `c1d_omip_l3_sasice` — unresolved ocean singleton selections;
- `c1d_omip_l3_sasice_run1` — OCE/RK3 path, stopped at step 3178;
- `c1d_omip_l3_sasice_run2` — empty retained discriminator directory;
- `c1d_omip_l3_sasice_final` and `c1d_omip_l3_sasice_accepted` — complete
  physical runs rejected because an instrumentation `NEWUNIT` sentinel
  replaced the stream on every write;
- `c1d_omip_l3_sasice_gate` — the complete pre-2026-09-02-scope run, with
  full streams and its contemporary green gate, superseded because its
  resolved ridging distribution inherited the reference rather than ORCA1;
- `c1d_omip_l3_sasice_scope_gate` — stopped before integration because the
  forcing was initially staged at the root instead of the required `SAS/`
  subdirectory; retained unchanged.

None of the older roots contains `run.json`; their `ocean.output`, stream
sizes, restarts, and (where present) `oracle_gate.json` establish the statuses
above.  The final root contains `run.json`, the complete log, and the final
gate result.  No retained directory was deleted or reused to conceal a failed
launch.

## Official forcing provenance

Official SETTE archive listing URL:
`https://gws-access.jasmin.ac.uk/public/nemo-vol1/sette_inputs/r5.0.0/C1D_v5.0.0.tar.gz`.
The archive and extracted input were independently re-read locally before the
accepted gate:

| artifact | digest |
|---|---|
| `C1D_v5.0.0.tar.gz` | MD5 `9456e6a0a84d40630ad1804fd4061caf` (documented listing value) |
| `C1D_v5.0.0.tar.gz` | SHA-256 `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| archive `.sha256` sidecar | SHA-256 `9f0ac65e3f8c4a5a5f9f72a0cc9d80cbb9c88017f431c255e8640b1e2a1c063b` |
| `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | SHA-256 `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| accepted run's staged `SAS/ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | SHA-256 `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |

This ERA5 file is the only external input the accepted run reads.  The same
file supplies the bulk fields at `namelist_cfg:126-134` and prescribed ocean
and ice state at `:153-162`; `ocean.output:125631-125661` records its final
record reads.  No analytic or synthetic forcing was used.

## Resolved namelist: source of every selection

Resolution is configuration over reference (`icestp.F90:277-278` and the
per-block `READ_NML_REF`/`READ_NML_CFG` calls).  Each active line in the two
committed overlay files has its source on the same line.  The grouping below
is only an index to those one-line citations.

| block | resolved selection | source |
|---|---|---|
| `namrun`, `namdom`, `namsbc`, `namsbc_sas` | steps 1–8760, `rn_Dt=3600`, `ln_c1d=T`, `nn_fsbc=1`, `nn_ice=2`, official ERA5 descriptors | shipped `cfgs/C1D/EXP_SASICE/namelist_cfg:23-55,83-91,145-163` |
| `namsbc_blk` | NCAR on; ECMWF/COARE/MFS/ANDREAS off; constant ice-air `Cd=Ce=Ch=1e-3` | ORCA1 `EXPREF/namelist_cfg:129-142` sets NCAR and constant ice coefficients but never sets `ln_ECMWF`; false comes from `SHARED/namelist_ref:236`. Forcing/humidity remains shipped C1D `:93-135` |
| `nameos` | TEOS-10 on, EOS-80 off | ORCA1 `EXPREF/namelist_cfg:305-309` |
| `nampar` | `jpl=1`, `nlay_i=3`, `nlay_s=3`, `rn_amax_n/s=.99999`, thermodynamics on | ORCA1 ice deck `:24-32`; `jpl=1` is also the shipped C1D value |
| `namitd` | single-category HFN, `rn_himean=2`, `rn_himin=.05`, `rn_himax=99` | ORCA1 ice deck `:37-39`; exclusive/user choice and upper bound from ice ref `:42-46` |
| `namdyn` | `ln_dynALL=T`, `ln_dynRHGADV=F` | ORCA1 ice deck `:44-45`; inert here because `ln_c1d` skips dynamics at copied `icestp.F90:170-171` |
| `namdyn_rdgrft` | H79 strength, smoothing off, exponential redistribution/participation, ridging and rafting on, `rn_pstar=2e4`, `rn_crhg=20`, `rn_fpndrdg=rn_fpndrft=.5`, `rn_porordg=0` | ORCA1 ice deck `:52-65`, cited one deviation per overlay line |
| `namdyn_rhg` | EVP and adaptive EVP on, EAP off, `nn_nevp=100` | ORCA1 ice deck `:70`; resolved exclusive/default values from ice ref `:108-113` |
| `namdyn_adv` | Prather on, Ultimate-Macho off | ORCA1 ice deck `:75-76` |
| ice `namsbc` | `rn_Cd_io=.005`, snow-fraction option 2, blow fraction .66, per-category fluxes, conduction coupling off, `nn_qtrice=0` | ORCA1 ice deck `:81-82`; otherwise shipped ice ref `:136-150` |
| `namthd` | vertical thickness change on, lateral melt off, open-water growth and lead heat on | ORCA1 ice deck `:87`; remaining values from ice ref `:157-160` |
| `namthd_zdf` | BL99, P07 conductivity, U64 off, 3+3 layers, `rn_cnd_s=.5`, convergence diagnostic off | ORCA1 ice deck `:92-94`; BL99/P07 and extinction values from ice ref `:165-171` |
| `namthd_do` | `rn_hinew=.05`, frazil-wind option off | ORCA1 ice deck `:103`; ice ref `:194` |
| `namthd_sal` | option 2; flushing/drainage on; `rn_simin=.1`, `rn_sinew=.75`, `rn_sal_gd=5`, `rn_time_gd=1.73e6`, `rn_sal_fl=2`, `rn_time_fl=8.64e5` | ORCA1 ice deck `:108-124` |
| `namthd_pnd` | ponds and LEV off | ORCA1 ice deck `:151-152` |
| `namdia` | `ln_icediachk=T` | phase-1 oracle conservation ledger required by dossier §3 |

The resolved, runtime prints are in `ocean.output` because the stand-alone SAS
executable does not create a separate `ice.output`.  Relevant actual lines are:

- NCAR/ECMWF/constant ice transfer: `ocean.output:562-582`;
- `jpl=1`, `nlay_i=3`, `nlay_s=3`, thermodynamics on: `:618-623`;
- HFN mean/min/max `2/.05/99`: `:632-636`;
- H/A/O/lead switches: `:644-647`;
- BL99, P07, U64 off, `rn_cnd_s=.5`: `:652-660`;
- `rn_hinew=.05`: `:665-666`;
- salinity option 2 and every salinity coefficient above: `:674-683`;
- ponds off, surface boundary choices, dynamics selectors, conservation
  diagnostic: `:701-743,777-862,884`; specifically the exponential
  distribution and ridge/raft fractions are `:820-837`, aEVP/100 is
  `:843-846`, and Prather is `:861-862`.

This is exactly the 2026-09-02 scope: `jpl=1`, BL99 with 3+3 layers and P07,
salinity option 2, aEVP, Prather, ORCA1 ridging/rafting selections, and ponds
off.  The dynamic selectors are resolved and checked but are not exercised:
`ln_c1d` skips the dynamics call.  Landfast L16 resolves false from the
shipped reference and is outside this rung.  No alternative-option rung or
test is included.

### Finding: the requested salinity inconsistency is preserved

ORCA1 sets `nn_icesal=2` at `namelist_ice_cfg:108` and `rn_sinew=.75` at
`:116`; its next line says it “must be set to 0.30 if nn_icesal=2”.  This rung
keeps the target value `.75` unchanged.  `ocean.output:674,677` proves those
are the values NEMO actually used.  The inconsistency is a finding; it was not
silently corrected.

## Build and execution

- Build command: `./makenemo -n C1D_OMIP_L3 -d 'OCE SAS ICE' -m conda -j 8`.
- Compiler: GNU Fortran 15.2.0 (conda-forge), `-fdefault-real-8` in the
  compile transcript; executable SHA-256
  `0ed0ef1a48ce8f7b96cd69e5aea958b0ab83de76560c5c8618cacbaa3a0416bd`.
- Direct execution: `CUDA_VISIBLE_DEVICES='' OMP_NUM_THREADS=1
  OMPI_MCA_pml=ob1 OMPI_MCA_btl=self ./nemo.exe`.
  No MPI launcher was used.  OpenMPI/UCX emitted sandbox socket warnings, then
  NEMO exited with `STOP 0`; these did not stop file reads or the integration.
  GNU Fortran also summarized the non-trapping `IEEE_UNDERFLOW_FLAG` and
  `IEEE_DENORMAL` flags.  The completion gate reports those two flags and
  separately rejects invalid, divide-by-zero, overflow, non-finite tokens,
  NEMO error blocks, missing final reads, and any status other than `STOP 0`.
- Duration: `nn_it000=1`, `nn_itend=8760`, 3600 s per step.  The final input
  reads are dated 2018-12-31 (`ocean.output:125631-125661`), and both restarts
  are written at step 8760 (`:125625,125677`).
- fp64: the build uses default-real-8, both stream headers report 64-bit
  reals, and every required restart payload is NetCDF `double`.

## Write-only frame registry and time levels

The instruments do not read or alter any prognostic array.  They open
unformatted streams with `ACTION='WRITE'`, write a versioned header and raw
fp64 payload, and flush.  A saved logical open-state is separate from the
negative value returned by Fortran `NEWUNIT`; the regression test forbids the
terminal-frame truncation found in the two rejected measurements.

| id | registered frame | time level | source |
|---:|---|---|---|
| 0 | `ENTRY` global state | now, before `ice_thd_frazil`; distinct from the `*_b` copy made by `store_fields` | `icethd.F90:112` |
| 1 | `POST_ZDF` selected-category 1-D state | now | `icethd.F90:151-152` |
| 2 | `POST_DH` selected-category 1-D state | now | `icethd.F90:154-155` |
| 3 | `POST_TEMP1` selected-category 1-D state | now | `icethd.F90:157-158` |
| 4 | `POST_SAL` selected-category 1-D state | now | `icethd.F90:160-161` |
| 5 | `POST_TEMP2` selected-category 1-D state | now | `icethd.F90:163-164` |
| 6 | `POST_DO` global state | now, after open-water growth and before `ice_cor` | `icethd.F90:189-192` |
| 7 | `EXIT` global state | now, after correction and LBC | `icethd.F90:221-225` |

The global payload is `a_i,v_i,v_s,sv_i,oa_i,t_su,a_ip,v_ip,v_il,e_i,e_s,
szv_i`.  The selected-category payload is `a_i_1d,h_i_1d,h_s_1d,t_su_1d,
e_i_1d,e_s_1d,sz_i_1d`.  The gate saw exactly 70,080 frames = 8 stages ×
8,760 steps, no missing/trailing bytes, all headers in `(step,stage)` order,
and finite payloads.

After `ice_update_tau`, the second stream writes every Appendix-A SI3 exchange
array plus the ocean arrays in one registered frame (`icestp.F90:236-264`).
The gate saw exactly 8,760 exchange frames.  The C1D 1-D allocation uses
NEMO's reduced `A2D(0)` shapes; the gate's 260-value frame formula is tied to
the allocations at `sbc_ice.F90:123-151` and `sbc_oce.F90:196-224`.

## Gate result and documented phenomenology

Baseline status in `oracle_gate.json`: `VERIFIED`.  `run.json` records the
direct command, CPU/fp64 policy, exit code 0, 8,760-step duration, forcing
member, and that this scope-exact run supersedes the earlier `_gate` root.

| measured item | result |
|---|---:|
| thermodynamics frames | 70,080 |
| exchange frames | 8,760 |
| initial/final EXIT thickness | 2.0003133276837946 / 1.341543555314957 m |
| minimum thickness | 0.5550952376958682 m at step 6027 |
| maximum thickness | 2.445688622638321 m at step 3177 |
| maximum positive 24 h change | +0.013452422315101242 m |
| maximum negative 24 h change | -0.055198968840736606 m |
| exchange `fr_i` range | 0.9–0.9 |

The source comparison is deliberately limited to what the shipped/target
decks document.  ORCA1 calls `rn_himean=2.0` the “expected domain-average ice
thickness” and `rn_himin=.05` the remapping minimum (`namelist_ice_cfg:38-39`);
the resolved reference maximum is 99 m (`ocean.output:632-636`).  The shipped
thermodynamic chain labels `ice_thd_dh` “Growing/Melting”
(`icethd.F90:154`).  The measured 0.555–2.446 m stays inside [.05,99], and the
opposite-signed 24 h changes prove both growth and melt occur during the
documented 2018 annual forcing.  No shipped numerical gold time series or
tighter expected thickness envelope was found, so none is claimed.

## Appendix-A restart coverage register

The gate opens the final ice restart, enumerates every NetCDF variable,
requires fp64 and finite values for every required prognostic, and fails on
any unregistered name.  `oracle_gate.json` expands every waived moment name
individually.

| disposition | variables | reason |
|---|---|---|
| VERIFIED-loaded | `nn_fsbc,kt_ice,v_i,v_s,a_i,t_su,u_ice,v_ice,oa_i,a_ip,v_ip,v_il,e_s_l01..03,e_i_l01..03,sv_i,szv_i_l01..03,snwice_mass,snwice_mass_b` | present, NetCDF double, finite |
| WAIVED | `t_s_l01..03` | diagnostic temperature reconstructed from the verified snow enthalpy; `icerst.F90:153-159` writes `e_s`, not `t_s` |
| WAIVED | `cnd_ice,t1_ice` | written only for active Jules coupling (`icerst.F90:177-180`) |
| WAIVED | `stress1_i,stress2_i,stress12_i` | `ln_c1d` skips ice dynamics |
| WAIVED | mandatory Prather moments `sx/sy/sxx/syy/sxy` × `ice,sn,a,age,c0_l01..03,e_l01..03,sal` | `ln_c1d` skips Prather advection; writer inventory `icedyn_adv_pra.F90:1390-1477` |
| WAIVED | optional Prather moments `sx/sy/sxx/syy/sxy` × `si_l01..03,ap,vp,vl` | Prather inactive; option-4 salinity and ponds also inactive (`:1453-1499`) |
| metadata | `nav_lon,nav_lat,numcat,time_counter` | coordinate variables, not prognostics |

The restart gate's planted `PLANTED_UNACCOUNTED_RESTART_ARRAY` exits 1 and
prints `restart coverage unaccounted`.  Thus the enumeration can fail red.

All Appendix-A SI3 exchange arrays are written at every step:
`qns_ice,qsr_ice,qla_ice,dqla_ice,dqns_ice,tn_ice,alb_ice,qml_ice,qcn_ice,
qtr_ice_top,utau_ice,vtau_ice,emp_ice,evap_ice,devap_ice,qns_oce,qsr_oce,
qemp_oce,qemp_ice,qevap_ice,qprec_ice,emp_oce,wndm_ice,sstfrz,rCdU_ice,
snwice_mass,snwice_mass_b,snwice_fmass`, followed by
`utau,vtau,taum,qsr,qns,emp,sfx,fr_i`.  CICE-only exchange arrays
(`topmelt,botmelt,ss_iou,ss_iov,...`) are WAIVED because CICE is not built.

**Round-8 correction (2026-09-04):** the list above is a writer inventory,
not a pointwise value certification.  This phase-1 gate checked the payload was
finite and measured only `fr_i`; finite uninitialized `rCdU_ice` storage and
first-step `utau/vtau/emp` halos therefore passed.  The drift ticket
`nemo_testcases_l3thd_exchange_drift_ticket.md` accounts for every changed
byte, fixes the config-local writer, and supersedes this stream hash.  No phase-1
thermodynamics or restart result depended on those inactive bytes.

## Appendix-B constants drift recheck

NEMO values were re-read at `src/OCE/DOM/phycst.F90:48,57-66`, ocean
`rho0/rcp` at `src/OCE/TRA/eosbn2.F90:1898-1899`, P07 at
`src/ICE/icethd_zdf_bl99.F90:264-275`, and the actual snow conductivity from
`ocean.output:655`.  legoESM values are recorded only to preserve the dossier
drift table; no legoESM code was executed or compared to this oracle.

| quantity | legoESM | NEMO resolved/oracle | drift (NEMO − legoESM) |
|---|---:|---:|---:|
| ice specific heat J kg-1 K-1 | 2106.0 | 2096.7 | -9.3 |
| fusion J kg-1 | 333700 | 333360.1 | -339.9 |
| sublimation J kg-1 | 2834000 | 2834400 | +400 |
| fresh-ice conductivity W m-1 K-1 | 2.04 | 2.034396 | -0.005604 |
| snow conductivity W m-1 K-1 | 0.31 | **0.5 actual** | +0.19 |
| snow density kg m-3 | 330 | 330 | 0 |
| ice density kg m-3 | 917 | 917 | 0 |
| freshwater density kg m-3 | 1000 | 1000 | 0 |
| ocean reference density kg m-3 | 1025 default / 1026 NEMO pin | 1026 | +1 / 0 |
| ocean heat capacity J kg-1 K-1 | 3994 | 3991.86795711963 | -2.13204288037 |
| liquidus slope degC PSU-1 | .054 | .054 | 0 |
| lead/ocean albedo | .06 | .066 | +.006 |
| brine-conductivity coefficients | U64 `.13` | P07 `.09,.011` | formulation differs; no scalar drift claimed |
| ice/snow emissivity | .97 | .97 | 0 |

## Planted controls and unit tests

- Baseline gate: exit 0, `status=VERIFIED`.
- `--plant-restart`: exit 1,
  `PLANTED_UNACCOUNTED_RESTART_ARRAY` reported.
- `--plant-exchange`: exit 1, `exchange fr_i bound step 1` reported.
- `--plant-thickness`: exit 1, `HFN thickness bound violation` reported.
- `/home/dbalwada/legoESM/.venv/bin/python -m pytest -q
  tests/ocean/fidelity/test_nemo_si3thd_oracle_gate.py`: **8 passed**.

## Artifact SHA-256 ledger

| artifact | SHA-256 |
|---|---|
| accepted `nemo.exe` | `0ed0ef1a48ce8f7b96cd69e5aea958b0ab83de76560c5c8618cacbaa3a0416bd` |
| build `cpp.history` | `ff51708b09b5cfb1e926012d76e0d7c9e1b4277999374f0d2e5bd79c2cb22836` |
| build `arch_nemo.fcm` | `30f4928a3f466b0ac47405eedc3bbf84a09365e9c5ce22410520ea4b39ba820b` |
| `namelist_cfg` | `6151c0fdd2431d07c7897d5852846a620edd58c55255f11b7fb3569803b5342c` |
| `namelist_ice_cfg` | `da7b4fc5865edf6a6a912d6316a51f6b845aaf7e8b87e9da121c6f0a46278033` |
| `namelist_ref` read by run | `b04f2ce4d12aa247ae3b707483d25cf2d38d0b3cf8817d71d33e66918fd81cfd` |
| `namelist_ice_ref` read by run | `055ab5e6839c12dbae61bf1c3bcdda46a8f95cf275b7f294d68d7f8e73179307` |
| ORCA1 target `namelist_cfg` | `7cfe2d47d78a00553cb28fe72c7e2be8655f96f0ea22920f0b8f17f5b2a47a0b` |
| ORCA1 target `namelist_ice_cfg` | `9be4731d1835ebd042e590421fb43cec6d984f114f049728acf1a150edc90470` |
| resolved `output.namelist.dyn` | `3f1e09f4eae0f70c4edd424e86f6349a9b36353d31f45c4f0800fb2e4bf2db9e` |
| resolved `output.namelist.ice` | `d2173437dfb9cfd9de8f82a996d4c26374c4bfce4fe66f0265c693af453d24e1` |
| `oracle_si3_thd_frames.bin` (103,368,000 bytes) | `7fc9df2707a85581075e3c69b26784155151a55640a5693eb32c34fb710ea49b` |
| `oracle_si3_exchange_frames.bin` (18,571,200 bytes) | `998f4790a8c832962ed8437b9b48fc514555d55c8770108889b30636fb968f51` |
| final ice restart | `b61cb8443e14b3f0868ef125f621c0b1748aff7bc827dfab0d4044fc3f773b4e` |
| final ocean restart | `84ed40c5e5d46f9830f4c203b3e3a79f6dfffaf142dc4c44347648e2cd9f5265` |
| `ocean.output` | `3e47d39f061ca1f9fa111a46b01bb3d1dbcfa10be73422fced0abc9e4af25430` |
| `oracle_gate.json` | `7e8fcf463fecd83adacbbc94fc70ebe36719ada39edd7b5b42ee46823115b8b0` |
| `run.json` | `530862de5022827fd03183a3bf45f2934305b8a964bb88379408a897b57e680d` |
| `planted_controls.json` | `04fb074481033c4f52d36e1b02b9ec2d5e56c6f6ad6cfe33687a2e07f5f52959` |
| committed `icethd.F90` | `1de9604ea7198ef1094677df0c0cc648b8b4ea459f3b184c41d0002024ac5b75` |
| committed `icestp.F90` | `27fc1659865a065cf8ebee6dd889cb25413263942a0ce2ad79b9040bfe376633` |
| committed gate | `45c0cdaa8b0656dad5339fed6084643eddaf2219d727aa53052c5c3e5edca95d` |

## FLAGGED FOR FUTURE DELETION

Nothing was deleted from shipped NEMO.  The following copied/shipped choices
are deliberately excluded from the accepted build and are flagged rather
than removed from the source case:

- `key_xios` — lane-1 oracle pattern uses native I/O;
- `key_RK3` — calls the OCE stepper and defeats prescribed SAS state;
- `key_top` — TOP is outside this oracle-only SI3 thermodynamics rung;
- copied C1D experiments other than `EXP_SASICE`, and copied C1D analysis
  scripts — retained for provenance but not used by the accepted run.

No push was performed.  No legoESM matching or production change is part of
this receipt.
