# Receipt — SMT-RUNGS round 7: SMT-7 (GM eddy-induced transport + MLE) survey and deck proposal

**Status: STOPPED_FOR_DECISION (survey only, no model code, no acquisition).**
Three findings change what SMT-7 can prove. (1) On this deck NEMO's mixed-layer-eddy
(MLE) scheme is LIVE and takes a branch ORCA2 never takes: the 500 m top level makes the
10 m reference level not exist, so the mixed layer is the whole water column. (2) legoESM
takes a different branch on the same state (mixed layer = the 500 m top level, so its MLE
streamfunction is exactly zero). (3) legoESM's RK3 lane refuses a non-zero GM coefficient
outright, so the eddy-induced transport cannot run on this deck today. Base `e7778df27`;
evidence `/data/abyssal/dbalwada/nemo-testcases-l2/phase3/smtrungs_rounds/round7/`
(`survey.json` sha256 `2db7434c…`, `survey_plant.json` `2fd343ed…`).

**ORCA2 pointer.** ORCA2 rung 3 = rung 2 + exactly two switches (`diff` of the rung-2 and
rung-3 `namelist_cfg` shows only `ln_mle` and `ln_ldfeiv` changed). Sections 1-2 are what
the ORCA2 rung-3 walk has to reproduce on ORCA2's own grid.

## Round 7 — 1. NEMO's program (compiled SMT-5 build `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/`)

Where each piece runs inside one RK3 step (before-level slopes and the coefficient are
formed once per step; the transports are added at stage 3 only):

| step | statement | citation |
|---|---|---|
| before N² and expansion coefficients, once per step | `CALL bn2( ts(:,:,:,:,Nbb), rab_b, rn2b, Nbb )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:142` |
| slopes (certified for SMT-3) | `CALL ldf_slp( kstp, rhd, rn2b, Nbb, Nbb )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:159` |
| eiv coefficient, once per step, before-level, only for `nn_aei_ijk_t = 21` | `IF( l_ldftra_time .OR. l_ldfeiv_time ) CALL ldf_tra( kstp, Nbb, Nbb )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:163` |
| the stage-3 transport hook | `At stage 3 only` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traadv.f90:253` |
| eiv transport added to the advecting transport (non-triad operator) | `IF( ln_ldfeiv .AND. .NOT. ln_traldf_triad )` then `CALL ldf_eiv_trp(...)` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traadv.f90:254` and `:256` |
| MLE transport added, with the potential density recomputed at `Kmm` | `CALL eos( ts, Kmm, rhd, rhop )` then `CALL tra_mle_trp(...)` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traadv.f90:261` and `:262` |
| vertical transport re-diagnosed from the augmented horizontal transport | `CALL wzv( ..., pFu, pFv, ww, np_transport )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traadv.f90:268` |
| FCT then advects with the augmented transport (stage 3 only does FCT) | `CALL tra_adv( ..., zFu, zFv, zFw, kstg )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3_stg.f90:474` |

So both parameterisations are advection: a bolus transport added before the tracer scheme,
at stage 3, with the vertical transport rebuilt from the total; neither is a tendency of
its own. `ln_ldfeiv_dia` is not a namelist key in 5.0.2: the diagnostic flag is set
from the requested output names (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/stprk3.f90:298`
is the first of them) and `ldf_eiv_dia` only calls `iom_put`; requesting none of those
outputs keeps it off and moves no state.

**Eddy-induced velocity (ldftra).**

| item | statement | citation |
|---|---|---|
| requires isoneutral laplacian | `IF( .NOT.( ln_traldf_iso .OR. ln_traldf_triad ) )` stops | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:611` |
| ceiling coefficient | `aei0 = zUfac * rn_Le**inn`, `zUfac = r1_2 * rn_Ue` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:622` and `:624` |
| `nn_aei_ijk_t = 21`: time-varying 2-D field, evaluated every step | `l_ldfeiv_time = .TRUE.` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:663` |
| coefficient call, `Kmm` = the before level | `CALL ldf_eiv( kt, aei0, aeiu, aeiv, Kmm )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:474` |
| column sum of the slope term (N² times slope², both slopes, per w-level) | `zah(ji,jj) = zah(ji,jj) + zn2 * ( wslpi*wslpi ...` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:758` |
| the thickness offset `zhw` starts at 5 m | `zhw(:,:) = 5._wp` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:729` |
| **Rossby-radius denominator uses `gphit`**, not `ff_t` | `zfw = MAX( ABS( 2. * omega * SIN( rad * gphit(ji,jj) ) ) , 1.e-10 )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:765` |
| radius clamp 2-40 km | `zRo(ji,jj) = MAX( 2.e3 , MIN( .4 * zn(ji,jj) / zfw, 40.e3 ) )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:767` |
| coefficient = radius² times inverse eddy time | `zaeiw = zRo * zRo * SQRT( zah / zhw ) * ssmask` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:769` |
| tropical taper uses `ff_t` | `zzaei = MIN( 1._wp, ABS( ff_t(ji,jj) * z1_f20 ) ) * zaeiw` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:775` |
| cap at `aei0` | `zaeiw(ji,jj) = MIN( zzaei , paei0 )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:776` |
| T to U face average, masked by `ssumask` | `zaeiu = 0.5 * ( zaeiw(ji,jj) + zaeiw(ji+1,jj) ) * ssumask` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:780` |
| stream function (stage-3 RK3 form): quarter-weighted slope pair times coefficient pair | `zpsi_uw(ji,jj,2) = - r1_4 * e2u(ji,jj) * ( wslpi(jk+1) + wslpi(ji+1,jk+1) ) ...` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:947` |
| transport increment | `pFu = pFu - ( zpsi_uw(ji,jj,1) - zpsi_uw(ji,jj,2) )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:954` |

With `nn_aht_ijk_t = 20` (the SMT-3 value) the Redi coefficient does not read `aeiu`
(that branch is `nn_aht_ijk_t = 21` only, `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:485`),
so switching the eddy-induced transport on does not change the Redi diffusion.

**Mixed-layer eddies (tramle, `nn_mle = 1`).**

| item | statement | citation |
|---|---|---|
| reference level 10 m with a tolerance of 10% of the thinnest layer | `zrefdep = 10._wp - 0.1_wp * MINVAL( e3w_1d )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/domzgr.f90:384` |
| shallowest w-level below it, and the level above | `nlb10 = MINLOC( gdepw_1d, mask = gdepw_1d > zrefdep, dim = 1 )`, `nla10 = nlb10 - 1` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/domzgr.f90:385` and `:386` |
| ML initialised to the whole wet column | `inml_mle(ji,jj) = mbkt(ji,jj) + 1` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:238` |
| **the density scan runs only if `nla10 > 0`** | `IF ( nla10 > 0 ) THEN` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:240` |
| the criterion (potential density, vs the reference level) | `IF( rhop(ji,jj,jk) > rhop(ji,jj,nla10) + rn_rho_c_mle )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:242` |
| deepest level of the computation, capped at `jpkm1` | `ikmax = MIN( MAXVAL( inml_mle(:,:) ), jpkm1 )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:245` |
| ML depth and mean buoyancy (`rhop` is masked, so a land cell reads `rho0 - 0`) | `zmld = zmld + zc`, `zbm = zbm + zc * (rho0 - rhop) * r1_rho0` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:253` and `:254` |
| face depth = min of the two neighbours (`nn_mld_uv = 0`) | `zhu(ji,jj) = MIN( zmld(ji+1,jj), zmld(ji,jj) )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:261` |
| buoyancy, divided by the larger of the top thickness and ML depth | `zbm = + grav * zbm / MAX( e3t(1)*(1+r3t*tmask), zmld )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:277` |
| stream function magnitude | `zpsim_u = rc_f * zhu * zhu * e2_e1u * ( zbm(ji+1) - zbm(ji) ) * MIN( 111.e3, e1u )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:301` |
| `nn_conv = 1` gate (OFF in ORCA2 rung 3) | `IF( MIN( zn2(ji,jj) , zn2(ji+1,jj) ) < 0._wp )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:311` |
| structure function: depth over ML depth, both stretched by `r3t` | `zcuw = 1._wp - ( gdepw_1d(jk+1)*(1+r3t(ji+1)) + gdepw_1d(jk+1)*(1+r3t(ji)) ) * zhu` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:333` |
| and its polynomial, clipped at 0 | `zmuw = MAX( 0._wp , ( 1._wp - zcuw ) * ( 1._wp + r5_21 * zcuw ) )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:337` |
| stream function at the lower w-level, masked by both `wumask` | `zpsi_uw(ji,jj,2) = zpsim_u(ji,jj) * zmuw * wumask(jk+1) * wumask(1)` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:340` |
| transport increment | `pFu = pFu + ( zpsi_uw(ji,jj,1) - zpsi_uw(ji,jj,2) )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:346` |
| `rc_f` from the REFERENCE latitude, not the local one | `rc_f = rn_ce / ( 5.e3_wp * 2._wp * omega * SIN( rad * rn_lat ) )` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:697` |
| `rhop` under the simplified EOS has no pressure term | `prhop(ji,jj,jk) = ( rho0 + zn ) * ztm` | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:904` |

**What ORCA2 rung 3 resolves them to** (`orca2_hierarchy/rung3/record/`):

| setting | ORCA2 rung 3 | citation |
|---|---|---|
| `ln_ldfeiv` | `.true.` | `orca2_rung3/namelist_cfg:334` |
| `nn_aei_ijk_t` | `21` (Treguier et al. 1997 growth-rate law) | `orca2_rung3/namelist_cfg:336` |
| `rn_Ue` / `rn_Le` | `0.03 m/s` / `200 km`, so `aei0 = 3000 m2/s` | `orca2_rung3/ocean.output:642` and `:643`; ceiling printed at `orca2_rung3/ocean.output:649` |
| `ln_mle` | `.true.` | `orca2_rung3/namelist_cfg:329` |
| `nn_mle` / `rn_ce` / `rn_lat` | `1` / `0.06` / `20.` deg (from `namelist_ref`, printed) | `orca2_rung3/ocean.output:749`, `:750`, `:753` |
| `nn_mld_uv` / `nn_conv` / `rn_rho_c_mle` | `0` (min) / `0` (always MLE) / `0.01 kg/m3` | `orca2_rung3/ocean.output:754`, `:755`, `:756` |
| lateral operator | iso-neutral Madec (non-triad), so the eiv term enters at `traadv` | `orca2_rung3/ocean.output:616` |
| eiv transport actually runs | `ldf_eiv_trp : eddy induced advection` | `orca2_rung3/ocean.output:1048` |

ORCA2 rung 3 has no coefficient file: aei is the time-varying law above (`rn_Ue`, `rn_Le`
only set the 3000 m2/s ceiling). ORCA2's time step is 10800 s; the seamount deck's is 2880 s.

## Round 7 — 2. legoESM's existing code (searched: `grep -rn "mle\|eiv\|bolus\|treguier" packages/ocean src scripts`)

| item | NEMO statement | legoESM | verdict |
|---|---|---|---|
| **MLE mixed-layer branch** | `nla10 = 0` skips the scan: ML = whole column (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:240`) | `iref = jnp.clip(searchsorted(...) - 1, 0, nlev-1)` turns `nla10 = 0` into level 0 and the scan then runs (`packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py:292`); the docstring claim "the -1 cannot underflow" (`packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py:290`) is false when `0.1*min(e3w) > ref_depth` | **GAP, measured**: legoESM ML = 500 m, NEMO ML = 4000-5000 m (section 3) |
| MLE density | `rhop` from the card's S-EOS (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/eosbn2.f90:904`) | `rho_pot = make_eos_fn()(T, S, 0)`, the default EOS, not the card's (`packages/ocean/legoesm/ocean/physics/combined.py:107`); `seos_cfg` is passed to the function (`packages/ocean/legoesm/ocean/physics/combined.py:277`) but not used here | **GAP** (hidden EOS) |
| MLE constants | `rho0 = 1026`, `grav = 9.80665` from the deck | `rho0=constants.rho_ocean` and `grav=constants.g` (`packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_latlon_cgrid.py:250` and `:251`), not the card's constants | **GAP** (sanctioned in the docstring for ORCA1; the SMT card pins NEMO constants) |
| MLE convection gate | ORCA2 rung 3: `nn_conv = 0` | default `no_mle_in_convection: bool = True` (`packages/ocean/legoesm/ocean/physics/lateral_mixing/mle.py:109`, ORCA1 value) | **GAP unless the card sets it False explicitly** |
| MLE placement / advection | stage 3 only, added to the advecting transport, FCT carries it, vertical transport rebuilt by `wzv` (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traadv.f90:262`) | a physics tendency from the step-entry state with a centred flux and its own continuity cumsum (`packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_latlon_cgrid.py:345` and `:363`); a SANCTIONED DEPARTURE in the file header | **GAP**; the SMT card sets `mle=None` today (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:2809`) |
| MLE streamfunction chain (rc_f, `H^2 e2u db`, min 111 km, `mu`) | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:301`, `:333`, `:337` | shared kernels `mle_streamfunction_magnitude`, `mle_vertical_structure` (`packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_latlon_cgrid.py:279`) | MATCH by reading; the structure function uses live `gdepw` from cell thickness (`packages/ocean/legoesm/ocean/physics/lateral_mixing/mle_latlon_cgrid.py:305`), NEMO uses the 1-D `gdepw_1d`: equal on full cells, differs in a partial bottom cell (UNVERIFIED size) |
| eiv Treguier coefficient | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:765`, `:775`, `:776` | `compute_treguier_kappa_gm_nemo_native` (`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:1562`) takes ONE `f_coriolis` for both the radius and the taper (`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:1721` and `:1731`) | **GAP**: NEMO's radius uses `gphit` (km read as degrees on this beta-plane), the taper uses `ff_t` |
| eiv T to U/V face average and stream function | `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:780` and `:947` | `nemo_kappa_gm_to_faces` (`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:1753`), `nemo_eiv_bolus_transport` (`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:2152`) | MATCH by reading (multiplication order differs from `r1_4 * e2u * (..)*(..)`: a last-bit question, UNVERIFIED) |
| eiv through the advection scheme | stage 3 only (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/traadv.f90:253`) | `gm_bolus_advection="through_fct"` exists (`packages/ocean/legoesm/ocean/physics/lateral_mixing/gm_redi_latlon_cgrid.py:2835`) and is added to the advecting flux (`packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:407`) | the path exists, but not for RK3: next row |
| **RK3 lane with a non-zero GM coefficient** | the deck is `rk3_ws` | `rk3_ws` RAISES for `kappa_GM != 0` ("does not yet support staged GM bolus transports", `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4928`); the unsplit step raises for `through_fct` as well | **GAP, blocking**: SMT-7 cannot run on the existing card; the ORCA2 zps card lists the same feature as unmeasured (`packages/ocean/legoesm/ocean/fidelity/nemo_testcase_recipe.py:1813`) |
| guard coverage | | the guard reads `kappa_GM` only; a Treguier-enabled config with `kappa_GM = 0.0` is not seen by it (read from source, not run) | PLAUSIBLE hole: the bolus could reach a lane that places it wrongly |
| defaults off | ref `ln_mle`, `ln_ldfeiv` = `.false.` | `OceanPhysicsConfig.mle = None`, `GMRediConfig(kappa_GM = 0.0)` in every shipped seamount card | MATCH (a card must select both explicitly) |

Callers of `mle=` in a certified seamount card: none (`mle=None`); the Treguier path is
used by the ORCA2 zps card with `kappa_GM` resolved to a non-zero Treguier ceiling.

## Round 7 — 3. What this deck does with them (REPLAY of NEMO's statements on a NEMO-written record)

Instrument: `scripts/validate/ocean_fidelity/testcases/nemo_testcase_l1_vortex_smtrungs_round7_smt7_survey.py`
(fp64 numpy transcription of the cited statements; input = the SMT-6 mesh and the day-30
and day-100 restart `tn`/`sshn`; S-EOS `rn_a0 = 0.28`, `rho0 = 1026`, `grav = 9.80665`
from the run's `ocean.output`; ORCA2 rung-3 values). It was NOT validated against a NEMO
`psiu_mle` dump (none exists for this deck): the numbers are PLAUSIBLE until the SMT-7
acquisition writes one.

- **Vertical grid.** 11 levels of 500 m (`gdepw_1d` = 0, 500, ..., 5000), so
  `zrefdep = 10 - 0.1*500 = -40 m`, `nlb10 = 1`, `nla10 = 0`: the scan at
  `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/tramle.f90:240` is skipped and
  `rn_rho_c_mle` is never read. Changing it cannot open or close anything on this grid.
- **NEMO's mixed layer = the whole wet column** (capped at `jpkm1`): 4000.0 to 5000.0 m
  over 3721 wet columns, in both restarts. legoESM's own function on the same state
  returns 500.0 m everywhere (one level), so its streamfunction is exactly zero.
- **Size, NEMO branch (replay).** Largest stream function on a wet face: 1.08e8 m3/s
  (day 100). Largest bolus velocity (increment over `e2u*e3u`): 3.3 m/s in a full 500 m
  cell, 3.4 m/s in a partial cell; all 3660 wet U faces are non-zero. At `rn_Dt = 2880 s`
  and `e1u = 30 km` that is a horizontal Courant number near 0.3 per step. This is the
  literal ORCA2 setting applied to a 5 km "mixed layer" (`rc_f*H^2` scaling): large by
  construction, not a defect of the replay.
- **Plant.** Replacing `MINVAL(e3w_1d)` by 10 m in the `zrefdep` line gives `nla10 = 1`:
  the scan runs, the ML is 500 m, the stream function is exactly 0.0 (the structure
  function is sampled only at depths 0 and `H`), and the run exits 3.
- **Eiv Rossby radius.** On this beta-plane `gphit` runs -930 to +930 (kilometres,
  `VORTEX_SMT5_VEC_R8_OMIP_L1/MY_SRC/usrdef_hgr.F90:130`), and NEMO reads it as degrees at
  `VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:765`, while `ff_t` is the 38.5 N
  beta-plane value `VORTEX_SMT5_VEC_R8_OMIP_L1/MY_SRC/usrdef_hgr.F90:177`
  (7.4e-5 to 1.07e-4). Radius from `gphit` against radius from `ff_t`: they differ on 630
  of 3721 wet columns (largest relative difference 18%); 3091 and 3721 columns sit on
  the 40 km clamp respectively. The effect on `aei` is real but bounded by the ceiling
  (`aei0 = 3000 m2/s`); how much is not measured here because the slopes were not replayed.
- **Eiv liveness (PROXY, not NEMO's slopes).** The initial vortex gives centred horizontal
  density differences on 698 U columns (anomaly `VORTEX_SMT5_VEC_R8_OMIP_L1/MY_SRC/usrdef_istate.F90:89`),
  and the proxy slope reaches the `rn_slpmax = 0.01` cap. So the eddy-induced transport is
  expected to be live and the coefficient near its 3000 m2/s ceiling in the vortex (PLAUSIBLE;
  unlike the BBL on the seamount, there is no gate that closes it).

## Round 7 — 4. SMT-7 deck proposal

SMT-7 = SMT-6 deck + two explicit namelist blocks (the SMT-6 `namelist_cfg` has neither;
both switches default `.false.`, so today they are inherited, not chosen):

```
&namtra_mle
   ln_mle        = .true.
   nn_mle        = 1          ! ORCA2 rung 3
   rn_ce         = 0.06       ! ORCA2 rung 3 (ref)
   rn_lat        = 20.        ! ORCA2 rung 3 (ref)
   nn_mld_uv     = 0          ! ORCA2 rung 3 (ref)
   nn_conv       = 0          ! ORCA2 rung 3 (ref)  -- legoESM default is the OTHER value
   rn_rho_c_mle  = 0.01       ! ORCA2 rung 3 (ref); never read on this grid (nla10 = 0)
/
&namtra_eiv
   ln_ldfeiv     = .true.
   nn_aei_ijk_t  = 21         ! ORCA2 rung 3
   rn_Ue         = 0.03       ! ORCA2 rung 3, m/s
   rn_Le         = 200.e+3    ! ORCA2 rung 3, m  (aei0 = 3000 m2/s)
   ln_eke_equ    = .false.
/
```

Everything else is the SMT-6 deck. The values above are ORCA2's own; nothing is invented.

**Decision (a), what SMT-7 is allowed to prove about MLE.**

| option | deck | what it certifies |
|---|---|---|
| **A (pick)** | ORCA2 values literally; ML = whole column (the `nla10 = 0` branch) | the stream-function chain, `zhu`/`zbm`/`mu`, `ikmax`, the masks and the stage-3 placement, at large amplitude; NOT the 10 m density criterion (ORCA2 exercises that; no seamount arm can) |
| B | A, plus a second MLE arm on a thin-top-layer vertical grid so `nla10 > 0` | also the criterion; changes the vertical grid under the whole SMT history (a new deck family), so a separate lane, not a rung |
| C | drop MLE from SMT-7, EIV only | EIV certified alone; MLE stays an ORCA2-only deliverable |

**Decision (b), the eiv coefficient law.**

| option | deck | what it certifies |
|---|---|---|
| **A (pick)** | `nn_aei_ijk_t = 21` as ORCA2 (needs a card field that feeds `gphit` to the radius) | the Treguier chain incl. NEMO's own radius, ceiling, taper and face average |
| B | `nn_aei_ijk_t = 0` (constant 3000 m2/s) | the transport and placement only; not ORCA2's law |

**Decision (c), split arms (D93: one switch per rung).** ORCA2 flips both switches at
once, but they are two modules with separate failure modes. Pick: three NEMO runs (minutes
each), `SMT-7e` = SMT-6 + eiv, `SMT-7m` = SMT-6 + MLE, `SMT-7` = both, so an eventual
non-bit statement is attributable. Alternative: only the combined deck.

**Decision (d), what the acquisition writes** (a "which quantities" choice). Pick: add the
existing `iom_put` fields `aeiu_3d`, `aeiv_3d` (`VORTEX_SMT5_VEC_R8_OMIP_L1/BLD/ppsrc/nemo/ldftra.f90:538`
is the 2-D pair's sibling) and `psiu_mle`, `psiv_mle` to the deck's output definition,
so the replay in round 8 compares NEMO's own coefficient and stream function; no source
change, no state change. Do NOT request the `uoce_eiv`/`ueiv_*` names (they switch
`ldf_eiv_dia` on).

**legoESM work SMT-7 needs before an acquisition is useful** (none exists): (1) an RK3
stage-3 eiv arm: the raise at `packages/ocean/legoesm/ocean/dynamics/ocean_model_latlon_cgrid.py:4928`
has to be replaced by a card-selected staged bolus that reaches the stage-3 advecting
transport and rebuilds `ww`; (2) a card field giving the Treguier radius its `gphit`
operand (decision (b)); (3) an MLE arm with the `nla10 = 0` branch, the card's EOS and
constants, and the stage-3 transport placement (decision (a)); (4) card selection of
both on the seamount card; (5) the acquisition `run.sh` (SMT-6 pattern) with the two blocks.

## Choices made this round (end-of-task list)

- Survey scope = the four items asked, read-only: ASKED.
- Evidence = a replay on the SMT-6 day-30/day-100 restarts with NEMO's constants; the eiv
  slope-dependent number is a PROXY and says so: UNASKED measurement convention (revert =
  restate with another record).
- Added one probe script + one test file (the instrument is committed, not heredoc): UNASKED,
  justified by the repo's "commit the probe" rule.
- No deck, card, default or model code changed: ASKED.

## Review and gates

- Citation gate (from heading "## Round 7 —"): see `round7/citation_gate.json`; the
  planted shift control is recorded next to it.
- New tests: `tests/ocean/fidelity/test_nemo_testcase_l1_vortex_smtrungs_round7_smt7_survey.py`
  (run with `packages/*` and `src` on `PYTHONPATH`).
- Review: see the final commit message / `round7/codex_review.txt`.

UNVERIFIED: that the replay equals NEMO's `psiu_mle` (no dump exists); the bolus
velocities (3.3-3.4 m/s) are replay numbers; the size of the Rossby-radius effect on
`aeiu` (slopes not replayed); that the stage-3 `tend` placement of legoESM's MLE is the
step-entry call (read from source, not traced); whether a Treguier-enabled `kappa_GM = 0.0`
card slips the RK3 guard; the partial-cell `gdepw` difference in the structure function.
