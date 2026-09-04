# SI3 lane 3b Round 13 receipt — coupled 10 m measured prefix

Date: 2026-09-06  
Tracker: `climate-federation/legoESM#1699`  
State: **MEASURED PREFIX AT BAR; RUNG 3.6 NOT CLOSED**

## Outcome

The accepted retained oracle prefix is
`/data/abyssal/dbalwada/nemo-testcases-l3/c1d_omip_l3_coupled10m_r13_oracle_i`.
It completed 8,760 hourly CPU steps (`STOP 0`) with a scalar-math executable
containing zero dynamic `_ZGV*` symbols.  Geometry, all ten stream schemas,
and 448,950 pointwise operator/input/bridge rows pass.  Of those rows, 448,949
are bit-identical; the sole non-bit row is one `POST_FZP.t_bo` value with
normalised error `2.0955593187140483e-16`, below the fixed `1e-15` bar.

This is not a completed rung-3.6 verdict.  The measured execution prefix ends
after `POST_TRA_QSR`.  `POST_SBC_STAGGER`, `POST_ZDF_DRG_COEFF`,
`PRE_DYN_SPG_TS`, `SSH_SUBSTEP`, `POST_STP2D`, and `PRE_DYN_ZDF_SOLVE` remain
**UNMEASURED** and are emitted as such by the gate.  Consequently there is no
measured first over-bar row in the completed prefix, and no claim is made about
the unmeasured coupled dynamics.  The gate's honest label is
`MEASURED_PREFIX_AT_BAR`, never `matched` or `faithful`.

The production integration is also incomplete: the new source-ordered
operators are invoked by the card gate but the general coupled driver does not
yet construct this exact one-layer RK3-WS/off-horizontal-operator ocean or
route `rCdU_top` through its shared implicit top-drag consumer.  This is a
construction debt, not an at-bar result for a production trajectory.

## Construction and provenance

`rn_bathy=10 m` is the user's free construction parameter, not an ORCA1
quantity.  The copied `jpk=2` configuration sets W depths to
`[0,rn_bathy]`; NEMO's `depth_to_e3_1d` then derives a 10 m layer.  The initial
levitating-ice adjustment is the area-weighted global mean at
`iceistate.F90:410-411,424-427`; it equals the local displacement only because
this domain has one wet column.  The gate measures
`ssh=-1.6666666666666667 m` and
`e3t=8.333333333333332 m` versus the source-expanded
`8.333333333333334 m` (normalised `2.13e-16`).

The executable was built from the retained NEMO copy
`.../nemo502_si3bulk_scalarmath_a_src/cfgs/C1D_OMIP_L3_COUPLED10M_R13B_SM`
with `arch-conda-scalarmath.fcm` SHA-256
`132f7a0500c4f0e86d8d3bf7864974a82e1dea5d83166dcfdfaf409e2ca04561`.
The effective flags include `-O3 -fno-tree-vectorize -fdefault-real-8`.

| actually read/generated input | provenance | SHA-256 |
|---|---|---|
| `C1D_v5.0.0.tar.gz` | NEMO SETTE r5.0.0 archive; md5 `9456e6a0a84d40630ad1804fd4061caf` | `54a2ceefd9126e180676e68eaa28ded85cc3b93ea3f0dda0fb964a035a4fc382` |
| `ERA5_NorthGreenland_surface_84N_-36E_1h_y2018.nc` | official C1D archive | `e5ec49445d2569019c45dec24255b9c7daf050079444b0e6e6d86a5b82317afe` |
| `merged_ESACCI_BIOMER4V1R1_CHL_REG05.nc` | pinned Zenodo ORCA1 input | `f43c5a1e8ce75e52bfc8edfa4318e68215c76c4b252cb0fb70fa00b39a40d6fe` |
| `weights_reg05_C1D_OMIP_L3_bilinear.nc` | sanctioned NEMO WEIGHTS bilinear column mapping | `715c51c4528feb7cb1f1d409584e5257d1b680f350682e2444b478217ec3b4dc` |
| `C1D_OMIP_L3_COUPLED_init_v2.nc` | derived exact ERA5 record-1 T/S/U/V/SSH input | `0ded4378da5d454d1658a889190428cd19404e2e9fd8f27a6781d1c635e838fa` |

The ORCA1 chlorophyll overlay is `ln_qsr_rgb=T`, `nn_chldta=1`, and
`sn_chl=merged_ESACCI_BIOMER4V1R1_CHL_REG05` from ORCA1
`namelist_cfg:118,160-170`.  `fld_read` reads the gridded field at
`traqsr.F90:312-315`; the oracle dump registers the resulting column value and
lookup index.  In this single wet layer the bottom `wmask` is zero, so
`traqsr.F90:390-415` deposits every band in that layer; chlorophyll is covered
but numerically inert in this geometry.

A post-hoc review check independently replays both parts of the NEMO input
path.  It applies the four generated weights in the exact assignment order of
`fld_interp` (`fldread.F90:1475-1484`; accumulation at `:1482`), constructs the monthly record centres
as in `fldread.F90:890-917`, and applies the hourly before/after interpolation
at `fldread.F90:181-186,225-228`.  All 8,760 dumped `CHLA` operands are
bit-identical to that replay.  This closes the sanctioned-WEIGHTS value check;
it is labelled post-hoc because the independent replay was added after the
first measured-prefix review.

The card now binds both clocks explicitly: the ocean step is 3,600 s and
`nn_fsbc=4` makes the SI3 step 14,400 s.  The nested 3,600 s phase-1 column
clock is not reused as the coupled ice cadence.

## Ordered measurements

| boundary | rows | bit-identical | maximum normalised error | status |
|---|---:|---:|---:|---|
| seven `POST_SSM` fields | 61,320 | 61,320 | 0 | AT_BAR |
| `POST_FZP.t_bo` | 2,190 | 2,189 | `2.09556e-16` | AT_BAR |
| 14 `POST_UPDATE_FLX` fields | 30,660 | 30,660 | 0 | AT_BAR |
| seven `POST_UPDATE_TAU` fields | 61,320 | 61,320 | 0 | AT_BAR |
| three `POST_FWB` fields | 26,280 | 26,280 | 0 | AT_BAR |
| two `POST_TRA_SBC_RK3` fields | 52,560 | 52,560 | 0 | AT_BAR |
| two `POST_TRA_QSR` fields | 17,520 | 17,520 | 0 | AT_BAR |
| independently replayed `fld_read` chlorophyll | 8,760 | 8,760 | 0 | AT_BAR |
| eight exchange-card mappings | 70,080 | 70,080 | 0 | AT_BAR |
| 15 producer-to-exchange bridges | 118,260 | 118,260 | 0 | AT_BAR |

The executing `ice_update_flx` branch is ORCA1's explicit
`ln_cndflx=.false.` (`namelist_ice_cfg:81`), selecting
`iceupdate.F90:109-113,132-134`; mass uses `emp_oce` at `:178`.  Retry E
corrected the registry to include `qsr_ice` and `emp_oce`.  Tau follows the
implicit-drag arm at `iceupdate.F90:361-407`.  FWB selects case 1 and
volume-type 1 at `sbcfwb.F90:224-239,292-295`; its named
`cdelay='fwb1'` reduction is current on the first call and one ice refresh
behind thereafter.  The synchronous private arm fails first at `kt=5`.
RK3 T/S uses `trasbc.F90:282-315`; the one-layer RGB row uses
`traqsr.F90:371-415`.

The card boundary consumes `POST_FWB.emp/qns`, not the earlier `icestp`
exchange record.  This distinction is non-vacuous: across the retained year,
the largest pre/post-FWB differences are `3.32124e-4 kg m-2 s-1` (`emp`) and
`1.58647 W m-2` (`qns`).  A stale-bridge plant is red while the corrected
post-FWB card mappings are bit-identical.

Rule 1d is enforced rather than merely recorded.  `sbc_ssm` must have two
frames per `kt`, with `Kbb=Kmm=1` on odd steps and `3` on even steps; ZDF must
occur at `kt=1,5,9,...,8757`; `tra_sbc` must follow the exact alternating
three-stage `(Kbb,Kmm,Krhs)` cycle; and `tra_qsr` must use `Kmm=2`, alternate
`Krhs=3/1`, and cross-link exactly to the stage-3 tracer record.  The compact
first/last values and formulas are retained in the gate JSON.  A unit test
proves that positive but wrong SSM time levels fail.

The exchange register is now producer-bound explicitly.  Every-step
`rCdU_ice/utau/vtau/taum`, four-step-carried
`tn_ice/alb_ice/snwice_mass/snwice_mass_b/snwice_fmass/sfx/fr_i`, the
FWB-mutated `qns/emp` cadence, and refresh-time `qml_ice/qcn_ice` all compare
bit-for-bit from their registered producer frames to the exchange stream.
Thirty-five fields are VERIFIED.  `sstfrz` is WAIVED: it remains its allocated
zero in this in-process C1D run; only the inactive external-coupler send path
sets it (`sbccpl.F90:2724-2726`), while the active ice-base `eos_fzp` row is
scored separately.  A corrupted nonzero `utau` bridge plant is red.

The 36-field exchange-stream coverage register is one-to-one and all fields
are `VERIFIED` by the combined bulk, thermodynamic, update, tau, FWB, and card
rows.  This statement certifies field registration/producer-to-card flow; it
does not certify the six unmeasured ocean-dynamics consumers listed above.

Pierre's unmerged `lead_freeze_source` change overlaps the earlier shared
`_nemo_si3_ice_flx_other` transcription of `icesbc.F90:357-405`.  Round 13 did
not add a second lead-heat implementation: `ice_update_flx` consumes the
already-produced `fhld/qlead` ledger.  The overlap remains flagged for merge.

## Controls and tests

All eleven row-level plants exit nonzero and turn their named row red:
`ssm_sample`, `fzp_operand`, `update_heat`, `fwb_mass`, `fwb_immediate`,
`trasbc_heat`, `chl_input`, `qsr_flux`, `producer_bridge`, `fwb_bridge`, and
`freshwater_sign`.

Focused tests: **12 passed**.  The hard-coded-constant ratchet invoked on every
touched Python file reports **7 passed, 1 skipped** (the skipped item is the
ratchet's constants-module exemption).  A repository-wide invocation reports
3,395 passed, 2 skipped, and five pre-existing failures in untouched
FV3/DINO-radius sites; none is attributed to this lane.

Key retained hashes: `exchange_gate_rule1d_bridges_v2.json`
`41d9a3b575f61d53380d8974d1a78edcc68ec2e4e31e9ba3fc8ef25f8951c142`,
`header_validity_bridge_corrected.json`
`e51a8a5597ed4f49cf798d127758fe2fa6c78ad32b4b1d7cc2de63d1efe6d6cc`,
executable `decf2144977b20bd0f8aeddc7adeb588d9a6fefbb909033049cfab48a2e77cfb`,
ocean log `53bc133cb27f3391cf8813b5b39ec5534ab258b3e02a7d0f878e480d8fb27018`.
Restart hashes are `45aecc2778fdf786d082363fcc869d63a99f810a801939c7c47903ce72b7b16c`
(ocean) and `c756f3e2117ad84ba73fa17a8b00470aba1b6824f504f86f8159a23777288b79`
(ice).
The retained eleven-plant manifest digest (SHA-256 over the sorted per-file
SHA-256 lines under `plants_rule1d_bridges/`) is
`94b458417c9d0e90a5920a525cc892e06f38b792d9fc64885a5db0177e2b066c`.

## Retained attempts / flagged for future deletion

No file was deleted.  Runs `_oracle`, `_b`, `_c`, `_d`, `_e`, `_f`, `_g`, and
`_h` are retained.  The first is the invalid shallow-L75 construction; `_c`
has the obsolete 56-field update header; `_d` has 58 fields but lacks the two
resolved-branch operands; later roots are valid staged instrumentation but
superseded by `_i`.  They are **FLAGGED FOR FUTURE DELETION**, not removed.
The shipped NEMO tree and all immutable ORCA1/C1D inputs were not modified.

## Codex-internal review disposition

No external review artifact is claimed.  Two codex-internal adversarial
passes returned HOLD on the first prefix.  The wrong pre-FWB bridge, missing
runtime-index validation, unit chlorophyll placeholder, optional stream
schemas, ambiguous 3,600/14,400 s clocks, and metadata-only exchange promotions
are corrected and remeasured above.  The reviewers' production-driver/top-drag construction finding is
confirmed and remains explicitly UNVERIFIED; it is the same scope boundary as
the six missing oracle/consumer frames, not silently treated as complete.

## ASKED / UNASKED

| choice | state | disposition |
|---|---|---|
| `rn_bathy=10 m` as a free construction parameter | ASKED | implemented and source-bounded |
| scalar-math CPU oracle, zero `_ZGV*`, full year | ASKED | completed |
| generated NEMO-WEIGHTS bilinear column file | ASKED (sanctioned generated input) | hash-pinned; no hand interpolation |
| fp64 plus `transcendentals="libm"` explicitly | ASKED | coupled card binds it |
| resolved ORCA1 `ln_cndflx=.false.` identity | ASKED by scope | implemented; true arm is not constructible |
| delayed FWB collective time level | UNASKED discovery | preregistered before correction; synchronous arm retained as red control |
| post-FWB exchange-card bridge and stale-bridge plant | UNASKED review correction | preregistered before rescore; corrected arm bit-identical |
| independent chlorophyll WEIGHTS/time replay | ASKED | 8,760/8,760 bit-identical; formal replay labelled post-hoc |
| certify full split-explicit SSH and implicit drag consumers | ASKED | **not completed; UNMEASURED debt** |
| wire the exact card through the production coupled driver | ASKED | **not completed; UNVERIFIED construction debt** |
| modify shipped NEMO, modify immutable inputs, delete attempts, GPU, `mpirun`, push | UNASKED | not done |
| claim a completed rung or trajectory equivalence | UNASKED | explicitly withheld |

## Commit identity

Implementation commit: `c53e40cebda3`; honest-scope correction:
`3b62ef0d851c`; post-review preregistration: `176e428b8a28`; corrected bridge,
schema, clock, and chlorophyll replay: `ee56aef34f81`.  This receipt's final
commit and bundle hashes are filled by the end-of-round handoff.
