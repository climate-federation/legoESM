MODULE l2_orca2_tke_boundary
   !! WRITE-only native en_after_boundaries frame for the ORCA2 Phase-2v card.
   !! Values stored here are never read by NEMO model arithmetic.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000, numout
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: o2b_begin, o2b_after_boundaries_row, o2b_finish

   CHARACTER(LEN=16), PARAMETER :: o2b_magic = 'NEMO_L4_TKEB_1'
   LOGICAL, SAVE :: o2b_active = .FALSE.
   INTEGER, SAVE :: o2b_kt = -1, o2b_kbb = -1, o2b_kmm = -1
   INTEGER, SAVE :: o2b_nn_eice = -1, o2b_nn_etau = -1
   INTEGER, SAVE :: o2b_nn_mxl = -1, o2b_nn_pdl = -1
   INTEGER, SAVE :: o2b_rows = 0
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: o2b_field

#  include "do_loop_substitute.h90"

CONTAINS

   SUBROUTINE o2b_begin(kt,Kbb,Kmm,nn_eice,nn_etau,nn_mxl,nn_pdl)
      INTEGER, INTENT(in) :: kt, Kbb, Kmm
      INTEGER, INTENT(in) :: nn_eice, nn_etau, nn_mxl, nn_pdl
      INTEGER :: ios
      o2b_active = lwp .AND. .NOT.ln_tile .AND. kt == nit000+1
      IF(.NOT.o2b_active) RETURN
      IF(ALLOCATED(o2b_field)) &
         & CALL ctl_stop('orca2-boundary: nested TKE boundary frame')
      IF(STORAGE_SIZE(1) /= 32) &
         & CALL ctl_stop('orca2-boundary: writer requires 32-bit default integer')
      IF(STORAGE_SIZE(1._wp) /= 64) &
         & CALL ctl_stop('orca2-boundary: writer requires binary64 wp')
      ALLOCATE(o2b_field(jpi,jpj,jpk),STAT=ios)
      IF(ios /= 0) CALL ctl_stop('orca2-boundary: cannot allocate boundary frame')
      o2b_field(:,:,:) = 0._wp
      o2b_kt = kt
      o2b_kbb = Kbb
      o2b_kmm = Kmm
      o2b_nn_eice = nn_eice
      o2b_nn_etau = nn_etau
      o2b_nn_mxl = nn_mxl
      o2b_nn_pdl = nn_pdl
      o2b_rows = 0
   END SUBROUTINE o2b_begin


   SUBROUTINE o2b_after_boundaries_row(jj,p_en)
      INTEGER, INTENT(in) :: jj
      REAL(wp), DIMENSION(A2D(0),jpk), INTENT(in) :: p_en
      IF(.NOT.o2b_active) RETURN
      IF(jj /= ntsj+o2b_rows) &
         & CALL ctl_stop('orca2-boundary: rows missing, duplicated, or out of order')
      o2b_field(:,jj,:) = p_en(:,jj,:)
      o2b_rows = o2b_rows + 1
   END SUBROUTINE o2b_after_boundaries_row


   SUBROUTINE o2b_finish
      INTEGER :: expected_rows, ios, payload, unit
      INTEGER, DIMENSION(3) :: extents
      CHARACTER(LEN=96) :: filename
      IF(.NOT.o2b_active) RETURN
      expected_rows = ntej-ntsj+1
      IF(o2b_rows /= expected_rows) &
         & CALL ctl_stop('orca2-boundary: not every TKE row was captured')
      extents = SHAPE(o2b_field)
      payload = PRODUCT(extents)
      WRITE(filename,'("oracle_tke_boundary_kt",I8.8,".bin")') o2b_kt
      OPEN(NEWUNIT=unit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('orca2-boundary: cannot open boundary frame')
      WRITE(unit) o2b_magic, 1, o2b_kt, o2b_kbb, o2b_kmm, jpi, jpj, jpk, &
         & STORAGE_SIZE(1._wp), 1, 0, payload, o2b_nn_eice, o2b_nn_etau, &
         & o2b_nn_mxl, o2b_nn_pdl
      WRITE(unit) extents
      WRITE(unit) o2b_field
      CLOSE(unit)
      DEALLOCATE(o2b_field)
      o2b_active = .FALSE.
      WRITE(numout,*) 'ORCA2_TKE_BOUNDARY_DUMP ', o2b_kt, o2b_kbb, o2b_kmm
   END SUBROUTINE o2b_finish

END MODULE l2_orca2_tke_boundary
