MODULE vortex_r16_stage_terms
   !!======================================================================
   !! Read-only stage-1/2/3 momentum-term writer for VORTEX round 16
   !! (lane round 200), the FLUX-FORM card.
   !!
   !! Round 192's writer (vortex_r8_stage_terms) covers the VECTOR card and
   !! refuses any stage but 2 and 3.  The flux card's kt=2 velocity error is
   !! already made at STAGE 1 -- where the only momentum statement NEMO
   !! executes is the flux-form advection call
   !! (stprk3_stg.F90, CASE(1): `IF( .NOT.ln_dynadv_vec ) CALL dyn_adv(...)`)
   !! -- and the operands of that call are the advective transports zFu, zFv
   !! and zFw, which no existing record carries.  This writer therefore opens
   !! at EVERY stage and adds the three transports to what round 192 wrote.
   !!
   !! One self-describing stream per stage is opened after that stage's WZV
   !! block and closed after the common barotropic correction.  Every write
   !! happens AFTER a compiled statement has completed; this module performs
   !! no arithmetic on any field NEMO consumes.
   !!======================================================================
   USE par_kind       , ONLY : wp
   USE par_oce        , ONLY : jpi, jpj, jpk
   USE oce            , ONLY : ww, ssh
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: vortex_r16_stage_begin
   PUBLIC :: vortex_r16_stage_rhs
   PUBLIC :: vortex_r16_stage_state
   PUBLIC :: vortex_r16_stage_finish

   INTEGER, SAVE :: nunit = -1
   INTEGER, SAVE :: nstage = 0
   LOGICAL, SAVE :: lopen = .FALSE.

CONTAINS

   SUBROUTINE vortex_r16_stage_begin( kstp, kstg, Kbb, Kmm, Kaa, Krhs, puu, pvv, &
      &                               pFu, pFv, pFw )
      INTEGER, INTENT(in) :: kstp, kstg, Kbb, Kmm, Kaa, Krhs
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      REAL(wp), DIMENSION(:,:,:)  , INTENT(in) :: pFu, pFv, pFw
      INTEGER :: ios, ngrps
      CHARACTER(LEN=16) :: clmagic
      CHARACTER(LEN=128) :: clfile

      IF( lopen ) CALL ctl_stop( 'VORTEX R16 stage writer opened twice' )
      IF( kstg < 1 .OR. kstg > 3 ) &
         CALL ctl_stop( 'VORTEX R16 stage writer accepts stages 1, 2 and 3 only' )
      IF( STORAGE_SIZE(1._wp) /= 64 ) &
         CALL ctl_stop( 'VORTEX R16 stage writer requires 64-bit wp' )

      WRITE(clfile,'("oracle_stage_flux_terms_kt",I8.8,"_s",I1,".bin")') kstp, kstg
      OPEN( NEWUNIT=nunit, FILE=TRIM(clfile), ACCESS='STREAM', &
         &  FORM='UNFORMATTED', STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) CALL ctl_stop( 'cannot open VORTEX R16 stage-term file' )

      nstage = kstg
      lopen = .TRUE.
      SELECT CASE( kstg )
      CASE( 1 )      ;   ngrps = 15
      CASE( 2 )      ;   ngrps = 19
      CASE DEFAULT   ;   ngrps = 21
      END SELECT
      clmagic = 'NEMO_L1_STGFLX1'
      WRITE(nunit) clmagic, 1, kstp, kstg, Kbb, Kmm, Kaa, Krhs, &
         &         jpi, jpj, jpk, ngrps, 0, 0, 0, STORAGE_SIZE(1._wp)
      CALL write3( 'kmm_u', puu(:,:,:,Kmm) )
      CALL write3( 'kmm_v', pvv(:,:,:,Kmm) )
      CALL write2( 'ssh_kmm', ssh(:,:,Kmm) )
      CALL write3( 'ww', ww )
      CALL write3( 'zfu', pFu )
      CALL write3( 'zfv', pFv )
      CALL write3( 'zfw', pFw )
      CALL vortex_r16_stage_rhs( 'base', Krhs, puu, pvv )
   END SUBROUTINE vortex_r16_stage_begin


   SUBROUTINE vortex_r16_stage_rhs( cdname, Krhs, puu, pvv )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      INTEGER, INTENT(in) :: Krhs
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      IF( .NOT. lopen ) RETURN
      CALL write3( TRIM(cdname)//'_u', puu(:,:,:,Krhs) )
      CALL write3( TRIM(cdname)//'_v', pvv(:,:,:,Krhs) )
   END SUBROUTINE vortex_r16_stage_rhs


   SUBROUTINE vortex_r16_stage_state( cdname, Kaa, puu, pvv )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      INTEGER, INTENT(in) :: Kaa
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      IF( .NOT. lopen ) RETURN
      CALL write3( TRIM(cdname)//'_u', puu(:,:,:,Kaa) )
      CALL write3( TRIM(cdname)//'_v', pvv(:,:,:,Kaa) )
   END SUBROUTINE vortex_r16_stage_state


   SUBROUTINE vortex_r16_stage_finish( kstg, Kaa, puu, pvv )
      INTEGER, INTENT(in) :: kstg, Kaa
      REAL(wp), DIMENSION(:,:,:,:), INTENT(in) :: puu, pvv
      IF( .NOT. lopen ) RETURN
      IF( kstg /= nstage ) CALL ctl_stop( 'VORTEX R16 stage writer stage drift' )
      CALL vortex_r16_stage_state( 'out', Kaa, puu, pvv )
      CLOSE(nunit)
      WRITE(numout,*) 'VORTEX_R16_STAGE_FLUX_TERM_DUMP stage ', kstg
      nunit = -1
      nstage = 0
      lopen = .FALSE.
   END SUBROUTINE vortex_r16_stage_finish


   SUBROUTINE write3( cdname, pfield )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      clname = cdname
      WRITE(nunit) clname, 3, SIZE(pfield,1), SIZE(pfield,2), SIZE(pfield,3)
      WRITE(nunit) pfield
   END SUBROUTINE write3


   SUBROUTINE write2( cdname, pfield )
      CHARACTER(LEN=*), INTENT(in) :: cdname
      REAL(wp), DIMENSION(:,:), INTENT(in) :: pfield
      CHARACTER(LEN=16) :: clname
      clname = cdname
      WRITE(nunit) clname, 2, SIZE(pfield,1), SIZE(pfield,2), 1
      WRITE(nunit) pfield
   END SUBROUTINE write2

END MODULE vortex_r16_stage_terms
