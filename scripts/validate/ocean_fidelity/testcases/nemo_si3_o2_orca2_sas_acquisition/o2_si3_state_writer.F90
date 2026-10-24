MODULE o2_si3_state_writer
   !!======================================================================
   !!                  ***  MODULE o2_si3_state_writer  ***
   !! Write-only binary observer for SI3 ladder rung O2 (ORCA2_SAS_ICE).
   !!======================================================================
   USE, INTRINSIC :: iso_fortran_env, ONLY : int32, real64
   USE par_oce
   USE dom_oce
   USE oce     , ONLY : ssh
   USE phycst
   USE sbc_oce
   USE sbc_ice , ONLY : snwice_mass, snwice_mass_b, snwice_fmass
   USE par_ice
   USE ice
   USE icedyn_adv_pra
   USE in_out_manager
   USE lib_mpp

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: o2_record_istate_scalars
   PUBLIC :: o2_write_istate_snapshot
   PUBLIC :: o2_write_si3_state_snapshot

   CHARACTER(len=8), PARAMETER :: cl_istate_magic = 'O2SI3I01'
   CHARACTER(len=8), PARAMETER :: cl_state_magic  = 'O1SI3R01'
   INTEGER(int32),   PARAMETER :: n_version       = 1_int32

   REAL(real64), ALLOCATABLE, SAVE, DIMENSION(:,:) :: z_ssh_before_kbb, z_ssh_before_kmm
   REAL(real64), ALLOCATABLE, SAVE, DIMENSION(:,:) :: z_t_bo_ini, z_ztfrz_ini, z_zmsk_ini
   REAL(real64), ALLOCATABLE, SAVE, DIMENSION(:,:) :: z_ht_i_ini, z_ht_s_ini, z_at_i_ini
   REAL(real64), ALLOCATABLE, SAVE, DIMENSION(:,:) :: z_sm_i_ini, z_tm_i_ini, z_t_su_ini, z_tm_s_ini
   REAL(real64), ALLOCATABLE, SAVE, DIMENSION(:,:) :: z_apnd_ini, z_hpnd_ini, z_hlid_ini, z_amax_2d_ini

CONTAINS

   SUBROUTINE o2_pad_a2d_2d( psrc, pdst )
      REAL(wp), DIMENSION(Nis0:Nie0, Njs0:Nje0), INTENT(in ) :: psrc
      REAL(real64), DIMENSION(jpi, jpj),         INTENT(out) :: pdst
      pdst(:,:) = 0.0_real64
      pdst(Nis0:Nie0, Njs0:Nje0) = REAL(psrc(Nis0:Nie0, Njs0:Nje0), real64)
   END SUBROUTINE o2_pad_a2d_2d

   SUBROUTINE o2_pad_a2d_3d( psrc, pdst )
      REAL(wp), DIMENSION(Nis0:Nie0, Njs0:Nje0, jpl), INTENT(in ) :: psrc
      REAL(real64), DIMENSION(jpi, jpj, jpl),         INTENT(out) :: pdst
      pdst(:,:,:) = 0.0_real64
      pdst(Nis0:Nie0, Njs0:Nje0, 1:jpl) = REAL(psrc(Nis0:Nie0, Njs0:Nje0, 1:jpl), real64)
   END SUBROUTINE o2_pad_a2d_3d

   SUBROUTINE o2_write_opt_3d( iu, parr )
      INTEGER,                                      INTENT(in) :: iu
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:),      INTENT(in) :: parr
      REAL(real64), DIMENSION(jpi, jpj, jpl)                   :: z3d
      IF( ALLOCATED(parr) ) THEN
         z3d(:,:,:) = REAL(parr(1:jpi, 1:jpj, 1:jpl), real64)
      ELSE
         z3d(:,:,:) = 0.0_real64
      ENDIF
      WRITE(iu) z3d
   END SUBROUTINE o2_write_opt_3d

   SUBROUTINE o2_write_opt_4d_s( iu, parr )
      INTEGER,                                      INTENT(in) :: iu
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:),    INTENT(in) :: parr
      REAL(real64), DIMENSION(jpi, jpj, nlay_s, jpl)           :: z4d
      IF( ALLOCATED(parr) ) THEN
         z4d(:,:,:,:) = REAL(parr(1:jpi, 1:jpj, 1:nlay_s, 1:jpl), real64)
      ELSE
         z4d(:,:,:,:) = 0.0_real64
      ENDIF
      WRITE(iu) z4d
   END SUBROUTINE o2_write_opt_4d_s

   SUBROUTINE o2_write_opt_4d_i( iu, parr )
      INTEGER,                                      INTENT(in) :: iu
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:,:),    INTENT(in) :: parr
      REAL(real64), DIMENSION(jpi, jpj, nlay_i, jpl)           :: z4d
      IF( ALLOCATED(parr) ) THEN
         z4d(:,:,:,:) = REAL(parr(1:jpi, 1:jpj, 1:nlay_i, 1:jpl), real64)
      ELSE
         z4d(:,:,:,:) = 0.0_real64
      ENDIF
      WRITE(iu) z4d
   END SUBROUTINE o2_write_opt_4d_i

   SUBROUTINE o2_record_istate_scalars( kt, Kbb, Kmm, pztfrz, pzmsk, pzht_i, pzht_s, pzat_i, &
      &                                 pzsm_i, pztm_i, pzt_su, pztm_s, pzapnd, pzhpnd, pzhlid )
      INTEGER,                      INTENT(in) :: kt, Kbb, Kmm
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) :: pztfrz, pzmsk, pzht_i, pzht_s, pzat_i
      REAL(wp), DIMENSION(jpi,jpj), INTENT(in) :: pzsm_i, pztm_i, pzt_su, pztm_s, pzapnd, pzhpnd, pzhlid

      IF( ALLOCATED(z_ssh_before_kbb) ) THEN
         IF( SIZE(z_ssh_before_kbb, 1) /= jpi .OR. SIZE(z_ssh_before_kbb, 2) /= jpj ) THEN
            DEALLOCATE( z_ssh_before_kbb, z_ssh_before_kmm, z_t_bo_ini, z_ztfrz_ini, z_zmsk_ini, &
               &        z_ht_i_ini, z_ht_s_ini, z_at_i_ini, z_sm_i_ini, z_tm_i_ini, z_t_su_ini, &
               &        z_tm_s_ini, z_apnd_ini, z_hpnd_ini, z_hlid_ini, z_amax_2d_ini )
         ENDIF
      ENDIF
      IF( .NOT. ALLOCATED(z_ssh_before_kbb) ) THEN
         ALLOCATE( z_ssh_before_kbb(jpi,jpj), z_ssh_before_kmm(jpi,jpj) )
         ALLOCATE( z_t_bo_ini(jpi,jpj), z_ztfrz_ini(jpi,jpj), z_zmsk_ini(jpi,jpj) )
         ALLOCATE( z_ht_i_ini(jpi,jpj), z_ht_s_ini(jpi,jpj), z_at_i_ini(jpi,jpj) )
         ALLOCATE( z_sm_i_ini(jpi,jpj), z_tm_i_ini(jpi,jpj), z_t_su_ini(jpi,jpj), z_tm_s_ini(jpi,jpj) )
         ALLOCATE( z_apnd_ini(jpi,jpj), z_hpnd_ini(jpi,jpj), z_hlid_ini(jpi,jpj), z_amax_2d_ini(jpi,jpj) )
      ENDIF

      IF( kt < 0 ) RETURN
      z_ssh_before_kbb(:,:) = REAL(ssh(1:jpi, 1:jpj, Kbb), real64)
      z_ssh_before_kmm(:,:) = REAL(ssh(1:jpi, 1:jpj, Kmm), real64)
      CALL o2_pad_a2d_2d( t_bo, z_t_bo_ini )
      z_ztfrz_ini(:,:)   = REAL(pztfrz(1:jpi, 1:jpj), real64)
      z_zmsk_ini(:,:)    = REAL(pzmsk(1:jpi, 1:jpj), real64)
      z_ht_i_ini(:,:)    = REAL(pzht_i(1:jpi, 1:jpj), real64)
      z_ht_s_ini(:,:)    = REAL(pzht_s(1:jpi, 1:jpj), real64)
      z_at_i_ini(:,:)    = REAL(pzat_i(1:jpi, 1:jpj), real64)
      z_sm_i_ini(:,:)    = REAL(pzsm_i(1:jpi, 1:jpj), real64)
      z_tm_i_ini(:,:)    = REAL(pztm_i(1:jpi, 1:jpj), real64)
      z_t_su_ini(:,:)    = REAL(pzt_su(1:jpi, 1:jpj), real64)
      z_tm_s_ini(:,:)    = REAL(pztm_s(1:jpi, 1:jpj), real64)
      z_apnd_ini(:,:)    = REAL(pzapnd(1:jpi, 1:jpj), real64)
      z_hpnd_ini(:,:)    = REAL(pzhpnd(1:jpi, 1:jpj), real64)
      z_hlid_ini(:,:)    = REAL(pzhlid(1:jpi, 1:jpj), real64)
      z_amax_2d_ini(:,:) = REAL(rn_amax_2d(1:jpi, 1:jpj), real64)
   END SUBROUTINE o2_record_istate_scalars

   SUBROUTINE o2_write_istate_snapshot( kt, Kbb, Kmm, parea, pzsshadj, &
      &                                 pthres_sst, phti_n, phti_s, phts_n, phts_s, &
      &                                 pati_n, pati_s, psmi_n, psmi_s, ptmi_n, ptmi_s, &
      &                                 ptsu_n, ptsu_s, ptms_n, ptms_s, papd_n, papd_s, &
      &                                 phpd_n, phpd_s, phld_n, phld_s )
      INTEGER,  INTENT(in) :: kt, Kbb, Kmm
      REAL(wp), INTENT(in) :: parea, pzsshadj
      REAL(wp), INTENT(in) :: pthres_sst, phti_n, phti_s, phts_n, phts_s
      REAL(wp), INTENT(in) :: pati_n, pati_s, psmi_n, psmi_s, ptmi_n, ptmi_s
      REAL(wp), INTENT(in) :: ptsu_n, ptsu_s, ptms_n, ptms_s, papd_n, papd_s
      REAL(wp), INTENT(in) :: phpd_n, phpd_s, phld_n, phld_s

      INTEGER :: iu, ios
      CHARACTER(len=256) :: cl_fname
      INTEGER(int32), DIMENSION(22) :: i_meta
      REAL(real64),   DIMENSION(26) :: r_meta
      REAL(real64),   DIMENSION(jpi, jpj) :: z2d

      IF( .NOT. ALLOCATED(z_ssh_before_kbb) ) THEN
         CALL ctl_stop( 'STOP', 'o2_write_istate_snapshot: o2_record_istate_scalars was not called first' )
         RETURN
      ENDIF

      WRITE(cl_fname, '("o2_si3_records/o2_si3_istate_rank", I4.4, ".bin")') mpprank
      OPEN( NEWUNIT=iu, FILE=TRIM(cl_fname), FORM='UNFORMATTED', ACCESS='STREAM', &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) THEN
         CALL ctl_stop( 'STOP', 'o2_write_istate_snapshot: failed to open '//TRIM(cl_fname) )
         RETURN
      ENDIF

      i_meta( 1) = INT(kt, int32)
      i_meta( 2) = INT(mpprank, int32)
      i_meta( 3) = INT(narea, int32)
      i_meta( 4) = INT(jpi, int32)
      i_meta( 5) = INT(jpj, int32)
      i_meta( 6) = INT(jpl, int32)
      i_meta( 7) = INT(nlay_i, int32)
      i_meta( 8) = INT(nlay_s, int32)
      i_meta( 9) = INT(nn_hls, int32)
      i_meta(10) = INT(Ni0glo, int32)
      i_meta(11) = INT(Nj0glo, int32)
      i_meta(12) = INT(nimpp, int32)
      i_meta(13) = INT(njmpp, int32)
      i_meta(14) = INT(Nis0, int32)
      i_meta(15) = INT(Nie0, int32)
      i_meta(16) = INT(Njs0, int32)
      i_meta(17) = INT(Nje0, int32)
      i_meta(18) = INT(Kbb, int32)
      i_meta(19) = INT(Kmm, int32)
      i_meta(20) = INT(nn_iceini_file, int32)
      i_meta(21) = MERGE(1_int32, 0_int32, ln_iceini)
      i_meta(22) = MERGE(1_int32, 0_int32, ln_ice_embd)

      r_meta( 1) = REAL(pthres_sst, real64)
      r_meta( 2) = REAL(phti_n, real64)
      r_meta( 3) = REAL(phti_s, real64)
      r_meta( 4) = REAL(phts_n, real64)
      r_meta( 5) = REAL(phts_s, real64)
      r_meta( 6) = REAL(pati_n, real64)
      r_meta( 7) = REAL(pati_s, real64)
      r_meta( 8) = REAL(psmi_n, real64)
      r_meta( 9) = REAL(psmi_s, real64)
      r_meta(10) = REAL(ptmi_n, real64)
      r_meta(11) = REAL(ptmi_s, real64)
      r_meta(12) = REAL(ptsu_n, real64)
      r_meta(13) = REAL(ptsu_s, real64)
      r_meta(14) = REAL(ptms_n, real64)
      r_meta(15) = REAL(ptms_s, real64)
      r_meta(16) = REAL(papd_n, real64)
      r_meta(17) = REAL(papd_s, real64)
      r_meta(18) = REAL(phpd_n, real64)
      r_meta(19) = REAL(phpd_s, real64)
      r_meta(20) = REAL(phld_n, real64)
      r_meta(21) = REAL(phld_s, real64)
      r_meta(22) = REAL(rn_amax_n, real64)
      r_meta(23) = REAL(rn_amax_s, real64)
      r_meta(24) = REAL(rt0, real64)
      r_meta(25) = REAL(parea, real64)
      r_meta(26) = REAL(pzsshadj, real64)

      WRITE(iu) cl_istate_magic
      WRITE(iu) n_version
      WRITE(iu) i_meta
      WRITE(iu) r_meta

      ! Domain & surface inputs (13 x 2D)
      WRITE(iu) REAL(tmask(1:jpi, 1:jpj, 1), real64)
      WRITE(iu) REAL(umask(1:jpi, 1:jpj, 1), real64)
      WRITE(iu) REAL(vmask(1:jpi, 1:jpj, 1), real64)
      WRITE(iu) REAL(ssmask(1:jpi, 1:jpj), real64)
      CALL o2_pad_a2d_2d( smask0,   z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( smask0_i, z2d ) ; WRITE(iu) z2d
      WRITE(iu) REAL(e1e2t(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(ff_t(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(gphit(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(sst_m(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(sss_m(1:jpi, 1:jpj), real64)
      WRITE(iu) z_ssh_before_kbb
      WRITE(iu) z_ssh_before_kmm

      ! Pre-ice_var_itd O2 scalar initial state (14 x 2D)
      WRITE(iu) z_t_bo_ini
      WRITE(iu) z_ztfrz_ini
      WRITE(iu) z_zmsk_ini
      WRITE(iu) z_ht_i_ini
      WRITE(iu) z_ht_s_ini
      WRITE(iu) z_at_i_ini
      WRITE(iu) z_sm_i_ini
      WRITE(iu) z_tm_i_ini
      WRITE(iu) z_t_su_ini
      WRITE(iu) z_tm_s_ini
      WRITE(iu) z_apnd_ini
      WRITE(iu) z_hpnd_ini
      WRITE(iu) z_hlid_ini
      WRITE(iu) z_amax_2d_ini

      ! Post-ice_istate aggregated volumes, snow/ice mass, and levitating-ice SSH (10 x 2D)
      WRITE(iu) REAL(SUM(v_i(1:jpi, 1:jpj, 1:jpl), dim=3), real64)
      WRITE(iu) REAL(SUM(v_s(1:jpi, 1:jpj, 1:jpl), dim=3), real64)
      WRITE(iu) REAL(at_i(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(SUM(a_ip(1:jpi, 1:jpj, 1:jpl), dim=3), real64)
      WRITE(iu) REAL(SUM(v_ip(1:jpi, 1:jpj, 1:jpl), dim=3), real64)
      WRITE(iu) REAL(SUM(v_il(1:jpi, 1:jpj, 1:jpl), dim=3), real64)
      WRITE(iu) REAL(snwice_mass(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(snwice_mass_b(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(ssh(1:jpi, 1:jpj, Kbb), real64)
      WRITE(iu) REAL(ssh(1:jpi, 1:jpj, Kmm), real64)

      CLOSE(iu)
   END SUBROUTINE o2_write_istate_snapshot

   SUBROUTINE o2_write_si3_state_snapshot( cd_stage, kt, ld_rst )
      CHARACTER(len=*), INTENT(in)           :: cd_stage
      INTEGER,          INTENT(in)           :: kt
      LOGICAL,          INTENT(in), OPTIONAL :: ld_rst

      INTEGER :: iu, ios
      LOGICAL :: ll_rst
      CHARACTER(len=256) :: cl_fname
      INTEGER(int32), DIMENSION(26) :: i_meta
      REAL(real64),   DIMENSION(3)  :: r_meta
      REAL(real64),   DIMENSION(jpi, jpj)      :: z2d
      REAL(real64),   DIMENSION(jpi, jpj, jpl) :: z3d

      ll_rst = lrst_ice
      IF( PRESENT(ld_rst) ) ll_rst = ld_rst

      IF( TRIM(cd_stage) == 'entry' ) THEN
         WRITE(cl_fname, '("o2_si3_records/o2_si3_state_entry_rank", I4.4, ".bin")') mpprank
      ELSE
         WRITE(cl_fname, '("o2_si3_records/o2_si3_state_step", I4.4, "_rank", I4.4, ".bin")') kt, mpprank
      ENDIF

      OPEN( NEWUNIT=iu, FILE=TRIM(cl_fname), FORM='UNFORMATTED', ACCESS='STREAM', &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) THEN
         CALL ctl_stop( 'STOP', 'o2_write_si3_state_snapshot: failed to open '//TRIM(cl_fname) )
         RETURN
      ENDIF

      i_meta( 1) = INT(kt, int32)
      i_meta( 2) = INT(mpprank, int32)
      i_meta( 3) = INT(narea, int32)
      i_meta( 4) = INT(jpi, int32)
      i_meta( 5) = INT(jpj, int32)
      i_meta( 6) = INT(jpl, int32)
      i_meta( 7) = INT(nlay_i, int32)
      i_meta( 8) = INT(nlay_s, int32)
      i_meta( 9) = INT(nn_hls, int32)
      i_meta(10) = INT(Ni0glo, int32)
      i_meta(11) = INT(Nj0glo, int32)
      i_meta(12) = INT(nimpp, int32)
      i_meta(13) = INT(njmpp, int32)
      i_meta(14) = INT(Nis0, int32)
      i_meta(15) = INT(Nie0, int32)
      i_meta(16) = INT(Njs0, int32)
      i_meta(17) = INT(Nje0, int32)
      i_meta(18) = INT(nn_fsbc, int32)
      i_meta(19) = INT(nn_icesal, int32)
      i_meta(20) = INT(nn_snwfra, int32)
      i_meta(21) = MERGE(1_int32, 0_int32, ln_pnd)
      i_meta(22) = MERGE(1_int32, 0_int32, ln_pnd_LEV)
      i_meta(23) = MERGE(1_int32, 0_int32, ln_pnd_TOPO)
      i_meta(24) = MERGE(1_int32, 0_int32, ln_pnd_lids)
      i_meta(25) = MERGE(1_int32, 0_int32, ln_cpl)
      i_meta(26) = MERGE(1_int32, 0_int32, ll_rst)

      r_meta(1) = REAL(rn_Dt, real64)
      r_meta(2) = REAL(rDt_ice, real64)
      r_meta(3) = REAL((2.0_wp * hi_mean(jpl) - hi_max(jpl-1)) / 3.0_wp, real64)

      WRITE(iu) cl_state_magic
      WRITE(iu) n_version
      WRITE(iu) i_meta
      WRITE(iu) r_meta

      ! HFN category bounds
      WRITE(iu) REAL(hi_max(0:jpl), real64)
      WRITE(iu) REAL(hi_mean(1:jpl), real64)

      ! Section A: Grid masks and ocean surface state (5 x 2D)
      WRITE(iu) REAL(tmask(1:jpi, 1:jpj, 1), real64)
      WRITE(iu) REAL(umask(1:jpi, 1:jpj, 1), real64)
      WRITE(iu) REAL(vmask(1:jpi, 1:jpj, 1), real64)
      WRITE(iu) REAL(sst_m(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(sss_m(1:jpi, 1:jpj), real64)

      ! Section B: Prognostic state and restart variables
      WRITE(iu) REAL(v_i(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(v_s(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(sv_i(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(a_i(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(t_su(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(oa_i(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(a_ip(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(v_ip(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(v_il(1:jpi, 1:jpj, 1:jpl), real64)
      CALL o2_pad_a2d_3d( cnd_ice, z3d ) ; WRITE(iu) z3d
      CALL o2_pad_a2d_3d( t1_ice,  z3d ) ; WRITE(iu) z3d
      CALL o2_pad_a2d_3d( t_si,    z3d ) ; WRITE(iu) z3d

      WRITE(iu) REAL(u_ice(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(v_ice(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(snwice_mass(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(snwice_mass_b(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(snwice_fmass(1:jpi, 1:jpj), real64)

      WRITE(iu) REAL(e_s(1:jpi, 1:jpj, 1:nlay_s, 1:jpl), real64)
      WRITE(iu) REAL(e_i(1:jpi, 1:jpj, 1:nlay_i, 1:jpl), real64)
      WRITE(iu) REAL(szv_i(1:jpi, 1:jpj, 1:nlay_i, 1:jpl), real64)

      ! Section C: All 5 Prather moments of all 11 extensive families
      CALL o2_write_opt_3d( iu, sxice  ) ; CALL o2_write_opt_3d( iu, syice  )
      CALL o2_write_opt_3d( iu, sxxice ) ; CALL o2_write_opt_3d( iu, syyice ) ; CALL o2_write_opt_3d( iu, sxyice )

      CALL o2_write_opt_3d( iu, sxsn   ) ; CALL o2_write_opt_3d( iu, sysn   )
      CALL o2_write_opt_3d( iu, sxxsn  ) ; CALL o2_write_opt_3d( iu, syysn  ) ; CALL o2_write_opt_3d( iu, sxysn  )

      CALL o2_write_opt_3d( iu, sxa    ) ; CALL o2_write_opt_3d( iu, sya    )
      CALL o2_write_opt_3d( iu, sxxa   ) ; CALL o2_write_opt_3d( iu, syya   ) ; CALL o2_write_opt_3d( iu, sxya   )

      CALL o2_write_opt_3d( iu, sxage  ) ; CALL o2_write_opt_3d( iu, syage  )
      CALL o2_write_opt_3d( iu, sxxage ) ; CALL o2_write_opt_3d( iu, syyage ) ; CALL o2_write_opt_3d( iu, sxyage )

      CALL o2_write_opt_3d( iu, sxsal  ) ; CALL o2_write_opt_3d( iu, sysal  )
      CALL o2_write_opt_3d( iu, sxxsal ) ; CALL o2_write_opt_3d( iu, syysal ) ; CALL o2_write_opt_3d( iu, sxysal )

      CALL o2_write_opt_3d( iu, sxap   ) ; CALL o2_write_opt_3d( iu, syap   )
      CALL o2_write_opt_3d( iu, sxxap  ) ; CALL o2_write_opt_3d( iu, syyap  ) ; CALL o2_write_opt_3d( iu, sxyap  )

      CALL o2_write_opt_3d( iu, sxvp   ) ; CALL o2_write_opt_3d( iu, syvp   )
      CALL o2_write_opt_3d( iu, sxxvp  ) ; CALL o2_write_opt_3d( iu, syyvp  ) ; CALL o2_write_opt_3d( iu, sxyvp  )

      CALL o2_write_opt_3d( iu, sxvl   ) ; CALL o2_write_opt_3d( iu, syvl   )
      CALL o2_write_opt_3d( iu, sxxvl  ) ; CALL o2_write_opt_3d( iu, syyvl  ) ; CALL o2_write_opt_3d( iu, sxyvl  )

      CALL o2_write_opt_4d_s( iu, sxc0  ) ; CALL o2_write_opt_4d_s( iu, syc0  )
      CALL o2_write_opt_4d_s( iu, sxxc0 ) ; CALL o2_write_opt_4d_s( iu, syyc0 ) ; CALL o2_write_opt_4d_s( iu, sxyc0 )

      CALL o2_write_opt_4d_i( iu, sxe   ) ; CALL o2_write_opt_4d_i( iu, sye   )
      CALL o2_write_opt_4d_i( iu, sxxe  ) ; CALL o2_write_opt_4d_i( iu, syye  ) ; CALL o2_write_opt_4d_i( iu, sxye  )

      CALL o2_write_opt_4d_i( iu, sxsi  ) ; CALL o2_write_opt_4d_i( iu, sysi  )
      CALL o2_write_opt_4d_i( iu, sxxsi ) ; CALL o2_write_opt_4d_i( iu, syysi ) ; CALL o2_write_opt_4d_i( iu, sxysi )

      ! Section D: ice_var_glo2eqv intensive equivalent variables
      WRITE(iu) REAL(h_i(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(h_s(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(o_i(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(h_ip(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(h_il(1:jpi, 1:jpj, 1:jpl), real64)
      CALL o2_pad_a2d_3d( a_ip_frac, z3d ) ; WRITE(iu) z3d
      CALL o2_pad_a2d_3d( a_ip_eff,  z3d ) ; WRITE(iu) z3d
      WRITE(iu) REAL(s_i(1:jpi, 1:jpj, 1:jpl), real64)
      WRITE(iu) REAL(sz_i(1:jpi, 1:jpj, 1:nlay_i, 1:jpl), real64)
      WRITE(iu) REAL(t_i(1:jpi, 1:jpj, 1:nlay_i, 1:jpl), real64)
      WRITE(iu) REAL(t_s(1:jpi, 1:jpj, 1:nlay_s, 1:jpl), real64)

      ! Section E: ice_var_agg category-aggregated variables (21 x 2D)
      WRITE(iu) REAL(vt_i(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(vt_s(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(at_i(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(at_ip(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(vt_ip(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(vt_il(1:jpi, 1:jpj), real64)
      WRITE(iu) REAL(ato_i(1:jpi, 1:jpj), real64)
      CALL o2_pad_a2d_2d( et_s,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( et_i,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( at_ip_eff, z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( tm_su,     z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( st_i,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( hm_i,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( hm_s,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( tm_si,     z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( om_i,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( sm_i,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( tm_i,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( tm_s,      z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( hm_ip,     z2d ) ; WRITE(iu) z2d
      CALL o2_pad_a2d_2d( hm_il,     z2d ) ; WRITE(iu) z2d

      CLOSE(iu)
   END SUBROUTINE o2_write_si3_state_snapshot

END MODULE o2_si3_state_writer
