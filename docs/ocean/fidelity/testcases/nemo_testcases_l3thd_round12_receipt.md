# SI3 lane 3b round 12 receipt — rung 3.6 construction stop

Date: 2026-09-05  
Tracker: `climate-federation/legoESM#1699`  
Prerequisite commit: `1eeb4f1c8c40`  
Construction gate/code commit: `2d319abb1842`  
State: **CONSTRUCTION DEBT — USER DECISION REQUIRED**

This round followed the independently reviewed rung-3.6 design until its first
registered boundary failed.  It did not tune past that boundary.  The copied
NEMO oracle has a one-metre bathymetry and the ORCA1 ice deck starts with
two-metre, 90%-concentration ice.  SI3's required initial snow+ice mass
adjustment lowers the one-cell domain's sea level by exactly
`1.6666666666666667 m`; NEMO consequently enters `kt=1 PRE_SSM` with
`ssh=-1.6666666666666667 m` and wet-layer `e3t=-0.6666666666666667 m`.
The requested 8,760-step run therefore is not a valid ocean column.  NEMO's
own `stp_ctl` aborts at step 36 after `|U|=1.1436e6 m s-1` and
`|V|=1.2422e6 m s-1`.

The first ordered stop is
`INITIAL_STATE.positive_wet_layer_thickness`, before `POST_SSM`, `eos_fzp`,
or any legoESM exchange/ocean operator.  Accordingly, no rung-3.6 numerical
fidelity or full-year trajectory claim is made.  The card/top-drag/RK3
extensions were not implemented after this stop: doing so could not certify
an invalid oracle and would violate the design's explicit instruction to stop
at the first boundary requiring a scientific choice.

## Rule 0 — owner from executing NEMO source

The observed value is the literal execution of the selected identity:

- the resolved deck selects `rn_bathy=1 m` at the copied config
  `EXP00/namelist_cfg:19` and exact ERA5 SSH zero at `:24`;
- ORCA1 selects `ln_ice_embd=.FALSE.` at its `namelist_cfg:115`; the copied
  config retains it at `EXP00/namelist_cfg:116`;
- SI3 forms `snwice_mass` and its before level at
  `src/ICE/iceistate.F90:400-401`;
- for the executing levitating branch it computes the area-mean water
  displacement at `iceistate.F90:408-411`, prints it at `:421-422`, and
  subtracts it from both ocean levels at `:424-426`;
- the QCO call immediately refreshes the vertical geometry at
  `iceistate.F90:431-433`.  `domqco.F90:125-128,160` maps that SSH into the
  active z-star scale factor.

The first exchange frame gives `snwice_mass_b=1710.0000000000002 kg m-2`.
With the resolved `rho0=1026 kg m-3`, its displacement is the observed
`1.666666666666667 m`.  This is not an instrumentation artifact: PRE_SSM's
independent `ssh` and `e3t` operands obey `e3t = 1 m + ssh` bit-for-bit.
Changing `ln_ice_embd` does not solve the one-cell geometry: the alternative
branch at `iceistate.F90:403-406` subtracts the same local load in a one-cell
domain.  The physical free choice is therefore the slab depth (or, outside the
approved identity, its initial ice/SSH state).

## Oracle construction and provenance

Nothing in the shipped NEMO tree was modified.  The work configuration is a
new copy under
`/data/abyssal/dbalwada/nemo-testcases-l3/nemo502_si3bulk_scalarmath_a_src/cfgs/C1D_OMIP_L3_COUPLED_SM`.
It was built from NEMO revision `dcc7fb8c1779fa8409e41e4ce3ab7d45b9ceb796`
with components `OCE ICE` and keys
`key_si3 key_vco_1d3d key_RK3 key_qco`.  The scalar-math arch SHA-256 is
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`;
its flags include `-fdefault-real-8 -O3 -fno-tree-vectorize`.  The executable
SHA-256 is
`8d73061e82ca01a90250ffe5bf1f49eb528c0dc5ed1ea4bd408261815cc7960a`,
and `nm -D` found zero `_ZGV*` symbols.  It ran directly on CPU without
`mpirun`.

The requested run root is
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_coupled_sm_round12_oracle`.
Its `ocean.output` SHA-256 is
`7e7f3b0e90d894c1afb87997e9cac0cb1313b060de57611d1baa091df2cad145`.
The machine-readable construction result remains outside git as
`construction_gate.json` (SHA-256
`a897b837e9dac2e2d9ddc685161c03febc8c7d575be85cd13065bbff435fedef`).
The PRE/POST_SSM stream SHA-256 is
`7eeaf4da3baf99eefb60aa7043619ed75832580fa24d8d9ebc7e721f533fa1ab`.
Its registry stores `kt`, runtime `Kbb/Kmm`, stage, count, and fp64 width;
the first row is `kt=1,Kbb=1,Kmm=1,PRE`.  The config-local writer is
WRITE-only and has SHA-256
`64cd75f07a673a4584ed93c35adffa24fc23ad33c43846a5fec8d170d29e5589`.

The copied resolved namelists have SHA-256
`4538f9c8d497ea703a348837876af176056780346f25d41246a3427fa76bf182`
(ocean cfg) and
`d4da9b215b1d762b7a776324ee7ce2ea6a2c0ee18452b434bc9cec65cc8fbce6`
(ice cfg).  NEMO's printed resolved outputs are
`70ead05456120d8164f25b1329b3c26f8cdf93863c8b367d326ea382fb9fcd68`
(`output.namelist.dyn`) and
`d2173437dfb9cfd9de8f82a996d4c26374c4bfce4fe66f0265c693af453d24e1`
(`output.namelist.ice`).  The config-local MY_SRC manifest digest is
`c1c8b6ac8dfbc3a689c277b8441c0e90137258eea23695d4d8885c115079be38`;
the two new initialization sources are
`usrdef_nam.F90=fa3973e79d77ff172d433c0b25748add58d276e72bb94425e2a2a35720cae240`
and
`usrdef_istate.F90=305b07628245068b1dcb764fd9e376800bfdae82f444216377fcc0474b5c85e4`.

## Immutable and derived inputs

Every scientific file opened before the stop is pinned here.  The official
C1D archive re-verifies as MD5
`9456e6a0a84d40630ad1804fd4061caf` and SHA-256
`54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382`.

| role | file | SHA-256 |
|---|---|---|
| atmospheric forcing, actually read | `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| ORCA1 Zenodo archive | `INPUTS/orca1_inputs.zip` | `8ba4023bcc168f34d91d4d040e20c506b5428cf3dbb0aaa5a13f4cb04cbf4241` |
| RGB chlorophyll, actually read | `merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc` | `f43c5a1e8ce75e52bfc8edfa4318e68215c76c4b252cb0fb70fa00b39a40d6fe` |
| exact ERA5-derived T/S/U/V input, actually read | `C1D_OMIP_L3_COUPLED_init_v2.nc` | `0ded4378da5d454d1658a889190428cd19404e2e9fd8f27a6781d1c635e838fa` |
| column bilinear weights, actually read | `weights_reg05_C1D_OMIP_L3_bilinear.nc` | `715c51c4528feb7cb1f1d409584e5257d1b680f350682e2444b478217ec3b4dc` |
| WEIGHTS target geometry | bootstrap `domain_cfg.nc` | `68dd5e953355e672275563a6f402e78ff5de8c6feba27af00d4e56f2114ad515` |
| exact WEIGHTS namelist | committed config and `/data` copy | `301c9d77dc7317c58b56893ea6d76730bde2b68ce8b11f402fb98abc0126771b` |

The exact-record generator printed fp32 values
`sst=-1.690032958984375`, `sss=34`, `ssu=0`, `ssv=0`; all four are bit-identical
in `kt=1 PRE_SSM`.  The first, superseded fixed-time-axis file is retained as
`C1D_OMIP_L3_COUPLED_init.nc` and was not read by the accepted construction.

NEMO `tools/WEIGHTS` ran the documented `scripgrid -> scrip -> scripshape`
sequence with `map_method='bilinear'` and `normalize_opt='frac'`.  Binary hashes
are `c1345d3e23c82b1019842a20b69bed46573bcf0ae7ed301d778aa221f1551a14`
(`scripgrid.exe`),
`ed758bca7fce3fbfe9b2a5b64c245bdd3d957336fe39d7a34a86b9d8ccb0a6ca`
(`scrip.exe`), and
`db22052c1e67a9429b454da5f0a758865cc57cce221184661d12de00ed3cedde`
(`scripshape.exe`).  The two grid intermediates and SCRIP map are respectively
`f119dc87bdbc5c26a8da6e9b405bf222a3ed0fa1a7051dd2b853218ee087dd85`,
`9eef2cb6da3a7e2153e802136747d2a06bfaf8a1831e762c98b00347c6ba520a`,
and `5de28c973f90779a53b9d494b3dabc56ed20e268c457cbfa54d15774db3ad9c7`.
The shaped target uses source indices `250475,250476,251196,251197` with
weights `0,0,1.01807451e-13,1`.  This proves the sanctioned NEMO construction,
but P-WEIGHT remains **UNMEASURED**: the ordered stop precedes the registered
`POST_TRA_QSR` dump, so the interpolated CHLA value is not claimed checked.

## Gate and controls

`nemo_rung36_construction_gate.py` reports the normalized `1e-15` row and bit
identity for T/S/U/V and the source-defined geometry identity.  T, S, U, V,
and `e3t = bathy + ssh` are bit-identical.  The positivity invariant is DEBT.
The unplanted command exits 1.  Three row-binding plants—temperature,
salinity, and geometry—each change the target's fp64 bits, make that named row
red (geometry makes its defining identity red), and exit 1.  The focused unit
suite reports `3 passed`.

The repository-wide hardcoded-constant ratchet ran over all 3,398 cases.  It
reported `3391 passed, 2 skipped, 5 failed`; all five failures are pre-existing
files untouched by this lane
(`fv3_native_physics_coupling.py`, `test_fv3_physics_coupling.py`, and three
DINO geometry tests).  Neither Round-12 script is named in a failure.

## Coverage and stopping disposition

| boundary/register group | disposition |
|---|---|
| exact T/S/U/V ingestion and runtime indices | VERIFIED, bit-identical |
| SI3 load-adjusted SSH and QCO e3 relation | VERIFIED as source execution |
| positive wet-layer geometry | DEBT: `-0.6666666666666667 m` |
| `POST_SSM`, `POST_FZP`, update-flux/tau/FWB, RGB, RK3, SSH substeps, top drag | UNMEASURED: after the ordered construction stop |
| complete exchange-field and restart coverage | UNMEASURED: no valid full-year oracle exists |
| legoESM exchange card and continuous column | UNMEASURED / deliberately not implemented after stop |

The abort-run's inherited phase-1 streams are retained for diagnosis only:
exchange SHA-256
`7b0b18f1988b84967045dfb8649c076cd0192007b8477020fce29fe7638431eb`,
thermodynamics SHA-256
`5a013d3c58918196edd618c6418dc253b1e90aa9c7011ca4f6c2d8f2f4a5cbb2`,
and bulk operands SHA-256
`f3b1e81bb0eec4f2f03c657d9f623cab04f12c2b697000e5da5a1018e6b457a5`.
They cover only 36 unstable steps and certify nothing.

## Decision required

A deeper single wet layer is the only in-scope repair that retains the exact
ERA5 SSH, phase-1 ice initial state, and ORCA1 `ln_ice_embd` selection.  Its
exact depth changes the slab heat capacity and momentum response, so this
receipt does not choose it post hoc.  The next execution needs the user to
select and preregister `rn_bathy` (for example a 10 m mixed-layer slab), after
which the copied config must be rebuilt and the ordered walk restarted at
`INITIAL_STATE`.  Retaining 1 m would require an out-of-scope change to initial
SSH/ice mass or a NEMO behavior change.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| implement the reviewed design and stop at the first boundary needing a decision | ASKED | stopped at initial positive-thickness invariant |
| generate REG05-column weights with NEMO tools and pin every used input | ASKED | completed; hashes above |
| exact ERA5 IC, scalar math, CPU, no MPI, zero `_ZGV*` | ASKED | completed and verified |
| run all 8,760 steps | ASKED | attempted; NEMO abort at step 36, so not completed or claimed |
| silently deepen the slab after observing the failure | UNASKED | not done; changes coupled-column heat capacity |
| offset initial SSH, reduce initial ice, or change ORCA1 `ln_ice_embd` | UNASKED | not done; changes the approved identity and `ln_ice_embd` does not help a one-cell domain |
| implement/score downstream legoESM operators after an invalid initial boundary | UNASKED | not done under the explicit stopping rule |
| duplicate Pierre's lead-heat path | UNASKED | not done; overlap remains registered for the future valid run |
| modify/delete shipped NEMO, delete retained artifacts, push, GPU, or `mpirun` | UNASKED | none performed |

## Flagged for future deletion

Nothing was deleted.  The following are retained but stale/diagnostic:

- `c1d_omip_l3_coupled_sm_round12_inputs/C1D_OMIP_L3_COUPLED_init.nc`
  (wrong fixed time axis; never read by the accepted construction);
- `c1d_omip_l3_coupled_sm_round12_domain_bootstrap/` (domain-only bootstrap;
  its config deliberately stopped after writing `domain_cfg.nc`);
- `c1d_omip_l3_coupled_sm_round12_smoke1/` (failed fixed-axis input);
- `c1d_omip_l3_coupled_sm_round12_smoke2/` (accepted four-step smoke only);
- `c1d_omip_l3_coupled_sm_round12_oracle/` (requested run, aborted at step 36;
  retained as the evidence root, not a full-year oracle).
