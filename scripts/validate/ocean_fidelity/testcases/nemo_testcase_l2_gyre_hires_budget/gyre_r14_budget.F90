MODULE gyre_r14_budget
   !!======================================================================
   !! WRITE-only GYRE r4_smag surface-V online production budget.
   !! The operator builds this file into a cloned configuration; it is not
   !! part of NEMO's engine tree and no array here feeds the model state.
   !!======================================================================
   USE par_kind       , ONLY : wp
   USE dom_oce        , ONLY : jpi, jpj, jpkm1, vmask
   USE in_out_manager , ONLY : lwp, numout
   USE lib_mpp        , ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE

   INTEGER, PARAMETER :: jr14_first_step = 697
   INTEGER, PARAMETER :: jr14_last_step  = 4320
   INTEGER, PARAMETER :: jr14_steps_day  = 24
   INTEGER, PARAMETER :: jr14_terms      = 8
   REAL(wp), PARAMETER :: rr14_closure_bar = 1.0e-12_wp

   LOGICAL, SAVE :: lr14_active = .FALSE.
   LOGICAL, SAVE :: lr14_enabled = .FALSE.
   LOGICAL, SAVE :: lr14_configured = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: zr14_entry(:,:), zr14_previous(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: zr14_hpg(:,:), zr14_vor(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: zr14_keg(:,:), zr14_zad(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: zr14_ldf(:,:), zr14_vertical(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: zr14_pre_zdf(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: zr14_stress(:,:), zr14_after_zdf(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: zr14_sum(:,:,:)

   PUBLIC :: r14_begin, r14_after_hpg, r14_after_vor
   PUBLIC :: r14_after_keg, r14_after_zad, r14_after_ldf
   PUBLIC :: r14_is_active, r14_zdf_pre_row, r14_zdf_row, r14_finish

CONTAINS

   SUBROUTINE r14_configure
      CHARACTER(LEN=8) :: cvalue
      INTEGER :: ilength, istatus

      IF( lr14_configured ) RETURN
      CALL GET_ENVIRONMENT_VARIABLE( 'GYRE_R14_BUDGET', cvalue, ilength, istatus )
      IF( istatus /= 0 .OR. ilength /= 1 ) THEN
         CALL ctl_stop( 'round-14 requires GYRE_R14_BUDGET=0 or 1' )
      ENDIF
      SELECT CASE( cvalue(1:1) )
      CASE( '0' )
         lr14_enabled = .FALSE.
      CASE( '1' )
         lr14_enabled = .TRUE.
      CASE DEFAULT
         CALL ctl_stop( 'round-14 requires GYRE_R14_BUDGET=0 or 1' )
      END SELECT
      lr14_configured = .TRUE.
      IF( lwp ) WRITE(numout,*) 'GYRE_R14_BUDGET_ENABLED ', lr14_enabled
   END SUBROUTINE r14_configure

   SUBROUTINE r14_allocate
      IF( .NOT.ALLOCATED(zr14_entry) ) THEN
         ALLOCATE( zr14_entry(jpi,jpj), zr14_previous(jpi,jpj), &
            &      zr14_hpg(jpi,jpj), zr14_vor(jpi,jpj),       &
            &      zr14_keg(jpi,jpj), zr14_zad(jpi,jpj),       &
            &      zr14_ldf(jpi,jpj), zr14_vertical(jpi,jpj),  &
            &      zr14_pre_zdf(jpi,jpj), zr14_stress(jpi,jpj),   &
            &      zr14_after_zdf(jpi,jpj), &
            &      zr14_sum(jpi,jpj,jr14_terms) )
         zr14_sum(:,:,:) = 0._wp
      ENDIF
   END SUBROUTINE r14_allocate

   SUBROUTINE r14_begin( kt, pv_entry )
      INTEGER, INTENT(in) :: kt
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_entry
      CALL r14_configure
      lr14_active = lr14_enabled .AND. lwp .AND. &
         &          kt >= jr14_first_step .AND. kt <= jr14_last_step
      IF( lr14_active ) THEN
         CALL r14_allocate
         zr14_entry(:,:) = pv_entry(:,:)
         zr14_previous(:,:) = 0._wp
         zr14_hpg(:,:) = 0._wp
         zr14_vor(:,:) = 0._wp
         zr14_keg(:,:) = 0._wp
         zr14_zad(:,:) = 0._wp
         zr14_ldf(:,:) = 0._wp
         zr14_vertical(:,:) = 0._wp
         zr14_pre_zdf(:,:) = 0._wp
         zr14_stress(:,:) = 0._wp
         zr14_after_zdf(:,:) = 0._wp
      ENDIF
   END SUBROUTINE r14_begin

   LOGICAL FUNCTION r14_is_active()
      r14_is_active = lr14_active
   END FUNCTION r14_is_active

   SUBROUTINE r14_take_boundary( pv_now, pv_term )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_now
      REAL(wp), DIMENSION(:,:), INTENT(out) :: pv_term
      IF( lr14_active ) THEN
         pv_term(:,:) = pv_now(:,:) - zr14_previous(:,:)
         zr14_previous(:,:) = pv_now(:,:)
      ENDIF
   END SUBROUTINE r14_take_boundary

   SUBROUTINE r14_after_hpg( pv_now )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_now
      IF( lr14_active ) THEN
         ! dyn_hpg SETS Krhs because it is first in every RK3 stage
         ! (dynhpg.F90:160-166,320-334); it does not add to the stage slot.
         zr14_hpg(:,:) = pv_now(:,:)
         zr14_previous(:,:) = pv_now(:,:)
      ENDIF
   END SUBROUTINE r14_after_hpg

   SUBROUTINE r14_after_vor( pv_now )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_now
      IF( lr14_active ) CALL r14_take_boundary( pv_now, zr14_vor )
   END SUBROUTINE r14_after_vor

   SUBROUTINE r14_after_keg( pv_now )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_now
      IF( lr14_active ) CALL r14_take_boundary( pv_now, zr14_keg )
   END SUBROUTINE r14_after_keg

   SUBROUTINE r14_after_zad( pv_now )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_now
      IF( lr14_active ) CALL r14_take_boundary( pv_now, zr14_zad )
   END SUBROUTINE r14_after_zad

   SUBROUTINE r14_after_ldf( pv_now )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_now
      IF( lr14_active ) CALL r14_take_boundary( pv_now, zr14_ldf )
   END SUBROUTINE r14_after_ldf

   SUBROUTINE r14_zdf_pre_row( kj, pv_pre_zdf )
      INTEGER, INTENT(in) :: kj
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_pre_zdf
      IF( lr14_active ) zr14_pre_zdf(:,kj) = pv_pre_zdf(:,1)
   END SUBROUTINE r14_zdf_pre_row

   SUBROUTINE r14_zdf_row( kj, pv_no_stress, pv_full )
      INTEGER, INTENT(in) :: kj
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_no_stress, pv_full
      IF( lr14_active ) THEN
         ! The vertical row is a measured source boundary, never the closure
         ! remainder: the independently captured post-explicit/pre-ZDF state
         ! is subtracted from the no-surface-stress implicit solution.  Thus an
         ! omitted explicit contributor cannot telescope into this row.
         zr14_vertical(:,kj) = pv_no_stress(:,1) - zr14_pre_zdf(:,kj)
         zr14_stress(:,kj) = pv_full(:,1) - pv_no_stress(:,1)
         zr14_after_zdf(:,kj) = pv_full(:,1)
      ENDIF
   END SUBROUTINE r14_zdf_row

   SUBROUTINE r14_finish( kt, prdt, pv_final )
      INTEGER, INTENT(in) :: kt
      REAL(wp), INTENT(in) :: prdt
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pv_final
      INTEGER :: iday, iunit, ios
      CHARACTER(LEN=16) :: cmagic
      CHARACTER(LEN=96) :: cfile
      REAL(wp), DIMENSION(jpi,jpj,jr14_terms) :: zday

      IF( .NOT.lr14_active ) RETURN
      zr14_sum(:,:,1) = zr14_sum(:,:,1) + prdt * zr14_zad(:,:)
      zr14_sum(:,:,2) = zr14_sum(:,:,2) + prdt * ( zr14_hpg(:,:) + zr14_keg(:,:) )
      zr14_sum(:,:,3) = zr14_sum(:,:,3) + prdt * zr14_vor(:,:)
      zr14_sum(:,:,4) = zr14_sum(:,:,4) + prdt * zr14_ldf(:,:)
      zr14_sum(:,:,5) = zr14_sum(:,:,5) + zr14_vertical(:,:)
      zr14_sum(:,:,6) = zr14_sum(:,:,6) + zr14_stress(:,:)
      zr14_sum(:,:,7) = zr14_sum(:,:,7) + pv_final(:,:) - zr14_after_zdf(:,:)
      zr14_sum(:,:,8) = zr14_sum(:,:,8) + pv_final(:,:) - zr14_entry(:,:)

      IF( MOD(kt,jr14_steps_day) == 0 ) THEN
         iday = ( kt - 1 ) / jr14_steps_day + 1
         zday(:,:,:) = zr14_sum(:,:,:) / REAL(jr14_steps_day,wp)
         WRITE(cfile,'("oracle_gyre_r14_budget_day",I4.4,".bin")') iday
         OPEN( NEWUNIT=iunit, FILE=TRIM(cfile), ACCESS='STREAM', &
            &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
         IF( ios /= 0 ) CALL ctl_stop( 'round-14 cannot open budget record' )
         cmagic = 'NEMO_R14_GYRBUD'
         WRITE(iunit) cmagic, 1, iday, kt, jpi, jpj, jpkm1, &
            &         STORAGE_SIZE(1._wp), jr14_terms, jr14_steps_day
         WRITE(iunit) prdt, rr14_closure_bar
         WRITE(iunit) vmask(:,:,1)
         WRITE(iunit) zday(:,:,:)
         CLOSE(iunit,IOSTAT=ios)
         IF( ios /= 0 ) CALL ctl_stop( 'round-14 cannot close budget record' )
         WRITE(numout,*) 'GYRE_R14_BUDGET_DUMP ', iday, kt, TRIM(cfile)
         zr14_sum(:,:,:) = 0._wp
      ENDIF
      lr14_active = .FALSE.
   END SUBROUTINE r14_finish

END MODULE gyre_r14_budget
