MODULE zdftke_round326_writer
   !! WRITE-only ORCA2 round-326 record of the SI3 fields that tke_avn's
   !! nn_mxlice = 2 surface mixing length reads (zdftke.F90, CASE( 2 ):
   !! ((1-fr_i)*zraug*taum + fr_i*hm_i*2)*tmask(:,:,1)).  Called from zdf_tke
   !! just before tke_avn, so it copies the values tke_avn then reads.
   !! Nothing here is read by model arithmetic.
   USE par_kind, ONLY : wp
   USE dom_oce
   USE sbc_oce , ONLY : fr_i, taum, nn_fsbc
#if defined key_si3
   USE ice     , ONLY : hm_i
#endif
   USE in_out_manager
   USE lib_mpp
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: r326_write

   INTEGER, PARAMETER :: jp_r326_last = 10   ! last recorded step counted from nit000

CONTAINS

   SUBROUTINE r326_write( kt, knn_mxlice, ld_mxl0 )
      !! One file per rank per step: hm_i, fr_i, taum and tmask(:,:,1) on the
      !! inner domain (ntsi:ntei, ntsj:ntej, the loop tke_avn runs), zero on
      !! the halo.
      INTEGER, INTENT(in) :: kt, knn_mxlice
      LOGICAL, INTENT(in) :: ld_mxl0
      INTEGER :: iunit, ios, ji, jj
      CHARACTER(LEN=96) :: clfile
      CHARACTER(LEN=16) :: clmagic
      REAL(wp), ALLOCATABLE, DIMENSION(:,:) :: zwrk
      IF( kt < nit000 .OR. kt > nit000 + jp_r326_last - 1 )   RETURN
#if ! defined key_si3
      CALL ctl_stop( 'round326 hm_i writer requires key_si3' )
#else
      IF( l_istiled )   CALL ctl_stop( 'round326 hm_i writer refuses tiling' )
      IF( STORAGE_SIZE(1._wp) /= 64 )   CALL ctl_stop( 'round326 hm_i writer requires fp64 wp' )
      WRITE(clfile,'("oracle_tke_hmi_kt",I8.8,"_r",I4.4,".bin")') kt, narea-1
      OPEN( NEWUNIT=iunit, FILE=TRIM(clfile), ACCESS='STREAM', FORM='UNFORMATTED',   &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 )   CALL ctl_stop( 'round326 hm_i writer cannot open output' )
      clmagic = 'NEMO_L4_HMI_1'
      WRITE(iunit) clmagic
      WRITE(iunit) 1, kt, narea-1, nimpp, njmpp, jpi, jpj, STORAGE_SIZE(1._wp), 4,   &
         &         knn_mxlice, MERGE(1,0,ld_mxl0), nn_fsbc
      ALLOCATE( zwrk(jpi,jpj) )
#define R326_2D(name,value) WRITE(iunit) name ; WRITE(iunit) 2,jpi,jpj,1,1,1,1 ; WRITE(iunit) value
      zwrk(:,:) = 0._wp
      DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj) = hm_i(ji,jj)
      END DO ; END DO
      R326_2D('hm_i            ',zwrk)
      zwrk(:,:) = 0._wp
      DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj) = fr_i(ji,jj)
      END DO ; END DO
      R326_2D('fr_i            ',zwrk)
      zwrk(:,:) = 0._wp
      DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj) = taum(ji,jj)
      END DO ; END DO
      R326_2D('taum            ',zwrk)
      zwrk(:,:) = 0._wp
      DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj) = tmask(ji,jj,1)
      END DO ; END DO
      R326_2D('tmask_1         ',zwrk)
#undef R326_2D
      CLOSE(iunit)
      DEALLOCATE( zwrk )
      WRITE(numout,*) 'ORCA2_R326_TKE_HMI ', kt, narea-1, TRIM(clfile)
#endif
   END SUBROUTINE r326_write

END MODULE zdftke_round326_writer
