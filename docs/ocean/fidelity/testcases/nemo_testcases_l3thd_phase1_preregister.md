# NEMO testcase lane 3b — SI3 thermodynamics phase-1 preregistration

Status: **PREREGISTERED — no C1D_OMIP_L3 model run has started.**  This is an
oracle-only receipt for tracker #1699.  It makes no legoESM comparison or
fidelity claim.

## Immutable provenance and execution

| item | preregistered value |
|---|---|
| NEMO source | `/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2`, commit `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796` |
| configuration | new copy `cfgs/C1D_OMIP_L3`; shipped `cfgs/C1D` remains byte-untouched |
| forcing archive | NEMO `sette_inputs/r5.0.0/C1D_v5.0.0.tar.gz`; official URL `https://gws-access.jasmin.ac.uk/public/nemo-vol1/sette_inputs/r5.0.0/C1D_v5.0.0.tar.gz`; MD5 `9456e6a0a84d40630ad1804fd4061caf`; SHA256 `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| forcing member | `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc`; SHA256 `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| target ice deck | `/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_ice_cfg`; SHA256 `9be4731d1835ebd042e590421fb43cec6d984f114f049728acf1a150edc90470` |
| target ocean deck | `/data/abyssal/dbalwada/ORCA1-omip/EXPREF/namelist_cfg`; its final SHA256 is recorded by the receipt |
| arithmetic | NEMO `REAL(wp)`; every binary frame must report `STORAGE_SIZE(1._wp)=64` |
| execution | one CPU process, direct `./nemo.exe`, no GPU, no `mpirun`; final scope-resolved run root `/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_sasice_scope_gate` |

The only input file the resolved run may open is the pinned ERA5 member.  The
shipped C1D initialization remains `ln_iceini=.true., nn_iceini_file=0`, which
constructs the initial ice from SST and namelist values
(`cfgs/SHARED/namelist_ice_ref:260-299`, `iceistate.F90:260-293`); importing
ORCA1's file-based initialization would require an unrelated ORCA input and is
therefore forbidden here.

## Exact resolved overlay

The copied configuration is built with the shipped `OCE SAS ICE` component
set and without the shipped C1D `key_RK3`/`key_top`, so `EXP_SASICE` resolves
`src/SAS/step.F90` plus `src/SAS/sbcssm.F90` and reads its `namsbc_sas`
ocean/ice boundary fields.  Keeping `key_RK3` makes `nemogcm.F90:166` call the
ocean RK3 stepper even when SAS sources are present.  That discriminator
evolved the one-metre ocean and stopped at step 3178 with salinity 100, while
the real ERA5 `sss` is exactly 34 throughout.  It is retained under the run
root as a finding and cannot satisfy this contract.  `key_xios` is also
excluded following lane 1; these three excluded keys are listed as FLAGGED
FOR FUTURE DELETION in the receipt rather than removed from shipped C1D.

Resolution is `namelist_ice_cfg` over `namelist_ice_ref`
(`icestp.F90:259-260`, with the per-block read pairs beginning at
`icestp.F90:345-346`).  Every changed row below is either the ORCA1 target or a
shipped C1D value retained explicitly.  The executable retains `key_si3`,
`key_linssh`, and `key_vco_1d3d`; `key_xios`, `key_RK3`, and `key_top` are
excluded from the copied configuration and listed in the receipt.

| block | required resolved value | source |
|---|---|---|
| ocean `namrun`, `namdom`, `namsbc` | `nn_it000=1`, `nn_itend=8760`, `nn_stock=8760`, `rn_Dt=3600`, `ln_c1d=T`, `nn_fsbc=1`, `nn_ice=2` | shipped `cfgs/C1D/EXP_SASICE/namelist_cfg:23-40,83-91` |
| ocean `namsbc_blk` | NCAR on; ECMWF/COARE/MFS/ANDREAS off; constant ice-air coefficients on with `Cd=Ce=Ch=1e-3` | ORCA1 `EXPREF/namelist_cfg:129-142` sets NCAR and constant ice coefficients but does not set `ln_ECMWF`; its false value is `SHARED/namelist_ref:236`. C1D forcing descriptors and humidity interpretation remain from `cfgs/C1D/EXP_SASICE/namelist_cfg:93-135` |
| ocean `nameos` | TEOS-10 on, EOS-80 off | ORCA1 `EXPREF/namelist_cfg:305-308` |
| ice `nampar` | `jpl=1`, `nlay_i=nlay_s=3`, `ln_icethd=T`, `rn_amax_n=rn_amax_s=0.99999`; `ln_icedyn` retains the resolved true value but is inert | ORCA1 ice cfg `:24-32`; dynamics is skipped by `ln_c1d` at `icestp.F90:167-168` |
| ice `namitd` | HFN on, user categories off, `rn_himean=2`, `rn_himin=0.05`, `rn_himax=99` | ORCA1 ice cfg `:37-39` over ice ref `:41-46` |
| ice `namdyn_rdgrft` | H79 strength, no strength smoothing, exponential redistribution and participation, ridging and rafting on, `rn_pstar=2e4`, `rn_crhg=20`, `rn_fpndrdg=rn_fpndrft=.5`, `rn_porordg=0` | ORCA1 ice cfg `:52-65`, one line per resolved deviation |
| ice `namdyn_rhg` | EVP and adaptive EVP on, EAP off, `nn_nevp=100` | ORCA1 ice cfg `:70` plus resolved ice ref `:108-113` |
| ice `namdyn_adv` | Prather on, Ultimate-Macho off | ORCA1 ice cfg `:75-76` |
| ice `namsbc` | `rn_Cd_io=5e-3`, `nn_snwfra=2`, `rn_snwblow=.66`, `nn_flxdist=-1`, conduction flux/emulation off, `nn_qtrice=0` | ORCA1 ice cfg `:79-82` over ice ref `:134-152` |
| ice `namthd` | thickness change on, lateral melt off, open-water growth on, lead heat on | ORCA1 ice cfg `:85-87` over ice ref `:155-160` |
| ice `namthd_zdf` | BL99 on; P07 on; U64 off; `rn_cnd_s=.5`, `rn_kappa_i=1`, `rn_kappa_s=10`, convergence check off | ORCA1 ice cfg `:90-94` over ice ref `:163-174` |
| ice `namthd_do` | `rn_hinew=.05`, frazil off | ORCA1 ice cfg `:101-103` over ice ref `:188-194` |
| ice `namthd_sal` | option 2; flushing/drainage on; `rn_simin=.1`, `rn_sinew=.75`, `rn_sal_gd=5`, `rn_time_gd=1.73e6`, `rn_sal_fl=2`, `rn_time_fl=8.64e5` | ORCA1 ice cfg `:106-124` |
| ice `namthd_pnd` | ponds off | ORCA1 ice cfg `:149-153` |
| ice `namdia` | `ln_icediachk=T` | deliberate oracle self-ledger instrumentation; dossier §3 requires this one diagnostic change |

**Registered finding, not corrected:** ORCA1 sets `rn_sinew=.75` while the
adjacent target-deck comment says it “must be set to 0.30 if nn_icesal=2”
(`namelist_ice_cfg:116-117`).  The run keeps `.75` because the deck is the
target.

## Write-only frame contract and time levels

`MY_SRC/icethd.F90` writes one stream containing one frame at each registered
stage for every one of the 8760 ice steps.  A frame contains a 16-byte magic,
format version, ocean step, stage code, dimensions, layer counts, 64-bit
storage declaration, then the fixed state registry.  Instrumentation assigns
no model array.  The expected stage order per step is:

| code | label | time-level disposition and source |
|---:|---|---|
| 0 | `ENTRY` | current pre-thermodynamic ice state at `ice_thd` entry, before `ice_thd_frazil` (`icethd.F90:86-115`) |
| 1 | `POST_ZDF` | current state immediately after `ice_thd_zdf` (`icethd.F90:148`) |
| 2 | `POST_DH` | current state immediately after `ice_thd_dh` (`icethd.F90:150`) |
| 3 | `POST_TEMP1` | current state immediately after the first `ice_thd_temp` (`icethd.F90:152`) |
| 4 | `POST_SAL` | current state immediately after `ice_thd_sal` (`icethd.F90:154`) |
| 5 | `POST_TEMP2` | current state immediately after the second `ice_thd_temp` (`icethd.F90:156`) |
| 6 | `POST_DO` | current state immediately after `ice_thd_do` (`icethd.F90:181`) |
| 7 | `EXIT` | current post-correction/halo thermodynamic state at routine exit (`icethd.F90:183-216`) |

SI3 has no leapfrog level for these arrays.  The registry labels all frames
`now` but preserves stage identity; `a_i_b` is separately the post-dynamics,
pre-thermodynamics copy made by `store_fields` at `icestp.F90:184`.
Unregistered stage codes, wrong order, missing/duplicate steps, non-finite
values, non-fp64 storage, or a frame count other than `8760*8` hard-fail.

After `ice_update_flx` (`icestp.F90:213`) and the same step's
`ice_update_tau` (`icestp.F90:233`), `MY_SRC/icestp.F90` writes one
exchange frame per step containing every SI3 array in `sbc_ice.F90:42-105`
except the explicitly CICE-only rows, plus `utau,vtau,taum,qsr,qns,emp,sfx,
fr_i,tn_ice`.  The exchange registry is fail-closed and all arrays must be
finite fp64 values.

The thermo stream uses payload code 0 for the geographic arrays at ENTRY,
POST_DO, and EXIT, and payload code 1 for the selected-column 1-D arrays
immediately after ZDF, DH, TEMP1, SAL, and TEMP2.  This records the actual
boundary state without calling a 1-D-to-2-D conversion and changing state.

## Restart coverage contract

The gate derives coverage from Appendix A and the active source arms, not from
a hand-picked subset.  Every variable discovered in the final ice restart must
have exactly one disposition and every required item below must exist:

* VERIFIED-loaded: `nn_fsbc`, `kt_ice`, `v_i`, `v_s`, `a_i`, `t_su`,
  `u_ice`, `v_ice`, `oa_i`, `a_ip`, `v_ip`, `v_il`, `e_s_l01..l03`,
  `sv_i`, `e_i_l01..l03`, `szv_i_l01..l03`, `snwice_mass`, and
  `snwice_mass_b`.
* WAIVED-inactive: `cnd_ice`, `t1_ice` (no Jules coupling); all EVP stresses
  and Prather moments (dynamics skipped by `ln_c1d`); pond moments (ponds off);
  `sxsi` layer-salt moments (`nn_icesal=2`, not option 4).
* WAIVED metadata: only coordinate/time aliases actually present in the file,
  each with a named reason.  Any other discovered variable is unaccounted and
  fails.

The gate must show two red planted controls: an added file-side restart name
fails coverage, and a nonzero perturbation to a finite exchange value fails the
registered exchange digest/invariant.  Unit tests invoke both failure paths.

## Preregistered run and phenomenology predicates

1. **CONFIRM build/run** iff the exact keys resolve, direct CPU execution exits
   zero at step 8760, the final ice restart exists, and logs contain no stop,
   floating exception, NaN, or infinite model value.  Otherwise **REFUTE**;
   no alternate forcing, precision, launcher, or shortened duration is allowed.
2. **CONFIRM seasonal growth/melt** iff the per-step thickness
   `SUM(v_i)/SUM(a_i)` has at least one positive and one negative 24-hour
   difference, its annual maximum occurs after a growth interval, and its
   annual minimum occurs after a melt interval.  This is a weighting-free
   one-column diagnostic.  Values are reported without a comparison claim.
3. **CONFIRM documented thickness bounds** iff every ice-present frame stays
   in the target HFN interval `0.05 <= h_i <= 99 m`, sourced from ORCA1
   `namelist_ice_cfg:39` and NEMO `namelist_ice_ref:46`, and the step-1 state
   instantiates the shipped northern initial thickness `2.0 m`
   (`namelist_ice_ref:267`).  No tighter undocumented observational range is
   invented.
4. **CONFIRM coverage/instrument** iff restart, namelist, frame, and exchange
   inventories pass and both planted controls fail for the registered reason.
5. The SAS executable's `ocean.output` (and generated `output.namelist.ice`)
   must print every resolved selector above; SAS does not create a separate
   `ice.output`.  The receipt may use those instantiated values, not deck
   comments, and must re-read NEMO
   `phycst.F90:39,48,57-66` for the Appendix-B constants table.

Scope: only the target deck's resolved `jpl=1`, BL99 3+3/P07,
`nn_icesal=2`, aEVP, Prather, ridging/rafting, and `ln_pnd=F` selection is
registered.  The dynamic selectors are resolution checks only: `ln_c1d`
skips their execution.  Landfast L16 remains the shipped-reference false
value and no additional-option rung or test is permitted in phase 1.

Blind spots: phase 1 validates oracle construction and trajectory completeness,
not legoESM equivalence; the column deliberately cannot exercise rheology,
advection, ridging, landfast ice, lateral melting, ponds, or multi-category
remapping.
