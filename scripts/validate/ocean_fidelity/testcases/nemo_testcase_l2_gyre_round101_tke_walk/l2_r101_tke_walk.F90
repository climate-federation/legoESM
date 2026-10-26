MODULE l2_r101_tke_walk
   !! WRITE-only kt=2 statement boundaries inside the active TKE solve.
   !! No value written here is ever read by NEMO.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r101_tke_begin, r101_entry, r101_after_boundaries_row, &
      & r101_after_langmuir_row, r101_rhs_row, &
      & r101_post_sweep_row, r101_tke_finish

   CHARACTER(LEN=16), PARAMETER :: r101_magic = 'NEMO_L2_R101TKE'
   INTEGER, SAVE :: r101_unit = -1
   INTEGER, SAVE :: r101_counts(4) = 0
   LOGICAL, SAVE :: r101_active = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r101_entry_field
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r101_boundaries
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r101_langmuir
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r101_rhs
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r101_post_sweep

#  include "do_loop_substitute.h90"

CONTAINS

   SUBROUTINE r101_tke_begin(kt,Kbb,Kmm)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm
      INTEGER :: ios
      r101_active = lwp .AND. kt == nit000+1
      IF(.NOT.r101_active) RETURN
      IF(r101_unit /= -1) CALL ctl_stop('round101: nested TKE statement record')
      IF(l_istiled) CALL ctl_stop('round101: whole-array writer refuses tiling')
      IF(STORAGE_SIZE(1) /= 32) CALL ctl_stop('round101: writer requires 32-bit default integer')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round101: writer requires binary64 wp')
      OPEN(NEWUNIT=r101_unit,FILE='oracle_tke_statement_walk_kt00000002.bin', &
         & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round101: cannot open TKE statement record')
      ALLOCATE(r101_entry_field(ntsi:ntei,ntsj:ntej,jpk), &
         & r101_boundaries(ntsi:ntei,ntsj:ntej,jpk), &
         & r101_langmuir(ntsi:ntei,ntsj:ntej,jpk), &
         & r101_rhs(ntsi:ntei,ntsj:ntej,jpk), &
         & r101_post_sweep(ntsi:ntei,ntsj:ntej,jpk))
      r101_entry_field(:,:,:) = 0._wp
      r101_boundaries(:,:,:) = 0._wp
      r101_langmuir(:,:,:) = 0._wp
      r101_rhs(:,:,:) = 0._wp
      r101_post_sweep(:,:,:) = 0._wp
      r101_counts(:) = 0
      WRITE(r101_unit) r101_magic
      WRITE(r101_unit) 1,kt,Kbb,Kmm,jpi,jpj,jpk,jpkm1,ntsi,ntei,ntsj,ntej, &
         & STORAGE_SIZE(1._wp)
   END SUBROUTINE r101_tke_begin

   SUBROUTINE r101_entry(p_en)
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_en
      IF(.NOT.r101_active) RETURN
      r101_entry_field(:,:,:) = p_en(ntsi:ntei,ntsj:ntej,:)
   END SUBROUTINE r101_entry

   SUBROUTINE r101_after_boundaries_row(jj,p_en)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_en
      IF(.NOT.r101_active) RETURN
      IF(jj /= ntsj+r101_counts(1)) &
         & CALL ctl_stop('round101: boundary rows are missing, duplicated, or out of order')
      r101_boundaries(:,jj,:) = p_en(ntsi:ntei,jj,:)
      r101_counts(1) = r101_counts(1) + 1
   END SUBROUTINE r101_after_boundaries_row

   SUBROUTINE r101_after_langmuir_row(jj,p_en)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_en
      IF(.NOT.r101_active) RETURN
      IF(jj /= ntsj+r101_counts(2)) &
         & CALL ctl_stop('round101: Langmuir rows are missing, duplicated, or out of order')
      r101_langmuir(:,jj,:) = p_en(ntsi:ntei,jj,:)
      r101_counts(2) = r101_counts(2) + 1
   END SUBROUTINE r101_after_langmuir_row

   SUBROUTINE r101_rhs_row(jj,p_en)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_en
      IF(.NOT.r101_active) RETURN
      IF(jj /= ntsj+r101_counts(3)) &
         & CALL ctl_stop('round101: RHS rows are missing, duplicated, or out of order')
      r101_rhs(:,jj,:) = p_en(ntsi:ntei,jj,:)
      r101_counts(3) = r101_counts(3) + 1
   END SUBROUTINE r101_rhs_row

   SUBROUTINE r101_post_sweep_row(jj,p_en)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_en
      IF(.NOT.r101_active) RETURN
      IF(jj /= ntsj+r101_counts(4)) &
         & CALL ctl_stop('round101: post-sweep rows are missing, duplicated, or out of order')
      r101_post_sweep(:,jj,:) = p_en(ntsi:ntei,jj,:)
      r101_counts(4) = r101_counts(4) + 1
   END SUBROUTINE r101_post_sweep_row

   SUBROUTINE r101_tke_finish
      INTEGER :: expected_rows
      IF(.NOT.r101_active) RETURN
      expected_rows = ntej-ntsj+1
      IF(ANY(r101_counts /= expected_rows)) &
         & CALL ctl_stop('round101: a TKE statement boundary did not capture every row')
      WRITE(r101_unit) r101_entry_field, r101_boundaries, r101_langmuir, &
         & r101_rhs, r101_post_sweep
      CLOSE(r101_unit)
      r101_unit = -1
      r101_active = .FALSE.
      DEALLOCATE(r101_entry_field,r101_boundaries,r101_langmuir,r101_rhs, &
         & r101_post_sweep)
   END SUBROUTINE r101_tke_finish

END MODULE l2_r101_tke_walk
