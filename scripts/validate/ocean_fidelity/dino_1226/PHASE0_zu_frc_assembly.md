# PHASE 0 — NEMO's zu_frc build (CONFIRMED, Rule 0, read from source)

Oracle: `cfgs/DINO/MY_SRC/dynspg_ts.F90` (DINO = MLF `#else` branch; `key_qco`
set ⇒ line 336-339 SUM branch, non-GPU). stpmlf.F90 dyn sequence (RHS = Nrhs):

    dyn_adv (303) -> dyn_vor (309) -> dyn_ldf (313) -> dyn_hpg (318) -> dyn_spg (326) -> [dyn_zdf 385, AFTER spg]

So `puu(:,:,:,Krhs)` AT ENTRY to dyn_spg_ts = adv + vor + ldf + hpg (NO zdf).

## zu_frc assembly order inside dyn_spg_ts (DINO MLF, key_qco):
1. **depth-mean base** (:337, key_qco SUM branch, REST weights — Q1 convention):
       zu_frc = SUM_k( e3u_0[k] * puu[k,Krhs] * umask[k] ) * r1_hu_0
   i.e. e3u_0 / r1_hu_0 (REST thickness), of (adv+vor+ldf+hpg) depth-averaged.
2. (:350-352) baroclinic split puu[Krhs]-=zu_frc  -- does NOT change zu_frc.
3. **2-D Coriolis removal** (:361-369): dyn_cor_2D(puu_b(Kmm),pvv_b(Kmm))->zu_trd;
       zu_frc -= zu_trd * ssumask     (een at Kmm on the barotropic velocity)
4. **drag correction** (:382-383 dyn_drg_init, under ln_drgimp): += pCdU-scaled
       baroclinic-residual bottom-drag term. INCREMENT dumped = drg_dump_zu_frc_inc.bin
5. (:406-420) ssh_ib inverse-barometer -- ln_apr_dyn; DINO has no atm pressure => ~0.
6. **wind stress** (:437-444): zu_frc += r1_rho0 * utauU * r1_hu(Kmm)  (centred:
       zztmp*(utau_b+utauU)). INCREMENT dumped = wnd_dump_zu_frc_inc.bin.
Final zu_frc dumped = spg_dump_zu_frc.bin.

## Stage-dump -> zu_frc-constituent map (RUN_SEQDUMP_Y20_1R, kt=00230401):
| dump | contents | role in zu_frc |
|---|---|---|
| stp_dump_03_dynadv _du | Krhs after adv | in the depth-mean base |
| stp_dump_04_dynvor _du | Krhs after +vor | in the depth-mean base |
| stp_dump_05_dynldf _du | Krhs after +ldf | in the depth-mean base |
| stp_dump_06_dynhpg _du | Krhs after +hpg = puu(Krhs) at dyn_spg ENTRY | **REST-weighted depth-mean = base of zu_frc** |
| cor2d_dump_zu_trd_substep1 | zu_trd (2-D Coriolis) | the -Coriolis removal (step 3) |
| drg_dump_zu_frc_inc | drag increment | step 4 |
| wnd_dump_zu_frc_inc | wind increment | step 6 |
| spg_dump_zu_frc | FINAL zu_frc | the whole = F_slow_u target |
| stp_dump_07_dynspg _u/_ub | u_new, puu_b after spg reconcile | N6 output (not a zu_frc constituent) |

DINO namelist: ln_drgimp (check), ln_apr_dyn=F (no ssh_ib), ln_bt_fw=F (centred).
