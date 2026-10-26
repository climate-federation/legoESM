MODULE zdfphy_round79b_writer
   !! WRITE-only ORCA2 round-79b record of the vertical-mixing chain.
   !!
   !! It captures NEMO's own diffusivities at the five boundaries of the
   !! compiled zdf_phy, in NEMO's execution order, together with the TKE
   !! closure internals a mixing-length comparison needs.  Nothing here is
   !! read by model arithmetic: every routine only copies out of live arrays
   !! and the final one writes a file.
   USE oce
   USE dom_oce
   USE zdf_oce, ONLY : avt, avs, avm, avt_k, avm_k, en, avtb, avmb, avtb_2d,  &
      &                ln_zdfiwm, ln_zdfddm, ln_zdfevd
   USE sbcrnf , ONLY : rnfmsk, nkrnf, ln_rnf_mouth
   USE in_out_manager
   USE lib_mpp
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: r79_active, r79_arm, r79_after_tke, r79_after_rnf
   PUBLIC :: r79_after_evd, r79_after_ddm, r79_write
   PUBLIC :: r79_tke_mxl, r79_tke_scalars

   INTEGER, PARAMETER :: jp_r79_first = 1    ! first recorded step counted from nit000
   INTEGER, PARAMETER :: jp_r79_last  = 10   ! last  recorded step counted from nit000

   LOGICAL, SAVE :: l_r79 = .FALSE.
   INTEGER, SAVE :: n_r79_mxl = -1
   REAL(wp), SAVE :: z_r79_ediff = 0._wp, z_r79_rmxl_min = 0._wp
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) ::                           &
      &   b_avt_tke, b_avm_tke, b_avt_rnf, b_avt_evd, b_avm_evd,              &
      &   b_avt_ddm, b_avs_ddm, b_avm_ddm, b_mxlm, b_mxld, b_dissl

CONTAINS

   LOGICAL FUNCTION r79_active()
      r79_active = l_r79
   END FUNCTION r79_active

   SUBROUTINE r79_arm( kt )
      !! Arm the recorder for this step and clear the per-step buffers.
      INTEGER, INTENT(in) :: kt
      l_r79 = ( kt >= nit000 + jp_r79_first - 1 .AND. kt <= nit000 + jp_r79_last - 1 )
      IF( .NOT. l_r79 )   RETURN
      IF( l_istiled )   CALL ctl_stop( 'round79b zdf writer refuses tiling' )
      IF( STORAGE_SIZE(1._wp) /= 64 )   CALL ctl_stop( 'round79b zdf writer requires fp64 wp' )
      IF( .NOT. ALLOCATED(b_avt_tke) ) THEN
         ALLOCATE( b_avt_tke(jpi,jpj,jpk), b_avm_tke(jpi,jpj,jpk),            &
            &      b_avt_rnf(jpi,jpj,jpk), b_avt_evd(jpi,jpj,jpk),            &
            &      b_avm_evd(jpi,jpj,jpk), b_avt_ddm(jpi,jpj,jpk),            &
            &      b_avs_ddm(jpi,jpj,jpk), b_avm_ddm(jpi,jpj,jpk),            &
            &      b_mxlm(jpi,jpj,jpk),    b_mxld(jpi,jpj,jpk),               &
            &      b_dissl(jpi,jpj,jpk) )
      ENDIF
      b_avt_tke(:,:,:) = 0._wp   ;   b_avm_tke(:,:,:) = 0._wp
      b_avt_rnf(:,:,:) = 0._wp   ;   b_avt_evd(:,:,:) = 0._wp
      b_avm_evd(:,:,:) = 0._wp   ;   b_avt_ddm(:,:,:) = 0._wp
      b_avs_ddm(:,:,:) = 0._wp   ;   b_avm_ddm(:,:,:) = 0._wp
      b_mxlm(:,:,:)    = 0._wp   ;   b_mxld(:,:,:)    = 0._wp
      b_dissl(:,:,:)   = 0._wp
   END SUBROUTINE r79_arm

   SUBROUTINE r79_tke_scalars( knn_mxl, pediff, prmxl_min )
      !! The closure's resolved selectors, which are private to zdftke.
      INTEGER , INTENT(in) :: knn_mxl
      REAL(wp), INTENT(in) :: pediff, prmxl_min
      IF( .NOT. l_r79 )   RETURN
      n_r79_mxl      = knn_mxl
      z_r79_ediff    = pediff
      z_r79_rmxl_min = prmxl_min
   END SUBROUTINE r79_tke_scalars

   SUBROUTINE r79_tke_mxl( kj, pmxlm, pmxld, pdissl )
      !! One i-k slice of the closure's two mixing lengths, plus the
      !! dissipation length scale, captured inside tke_avn's jj loop.
      INTEGER , INTENT(in) :: kj
      REAL(wp), INTENT(in) :: pmxlm(ntsi:ntei,jpk), pmxld(ntsi:ntei,jpk)
      REAL(wp), INTENT(in) :: pdissl(ntsi:ntei,ntsj:ntej,jpk)
      INTEGER :: ji, jk
      IF( .NOT. l_r79 )   RETURN
      DO jk = 1, jpk
         DO ji = ntsi, ntei
            b_mxlm (ji,kj,jk) = pmxlm(ji,jk)
            b_mxld (ji,kj,jk) = pmxld(ji,jk)
            b_dissl(ji,kj,jk) = pdissl(ji,kj,jk)
         END DO
      END DO
   END SUBROUTINE r79_tke_mxl

   SUBROUTINE r79_after_tke()
      INTEGER :: ji, jj, jk
      IF( .NOT. l_r79 )   RETURN
      b_avm_tke(:,:,:) = avm(:,:,:)
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         b_avt_tke(ji,jj,jk) = avt(ji,jj,jk)
      END DO ; END DO ; END DO
   END SUBROUTINE r79_after_tke

   SUBROUTINE r79_after_rnf()
      INTEGER :: ji, jj, jk
      IF( .NOT. l_r79 )   RETURN
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         b_avt_rnf(ji,jj,jk) = avt(ji,jj,jk)
      END DO ; END DO ; END DO
   END SUBROUTINE r79_after_rnf

   SUBROUTINE r79_after_evd()
      INTEGER :: ji, jj, jk
      IF( .NOT. l_r79 )   RETURN
      b_avm_evd(:,:,:) = avm(:,:,:)
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         b_avt_evd(ji,jj,jk) = avt(ji,jj,jk)
      END DO ; END DO ; END DO
   END SUBROUTINE r79_after_evd

   SUBROUTINE r79_after_ddm()
      INTEGER :: ji, jj, jk
      IF( .NOT. l_r79 )   RETURN
      b_avm_ddm(:,:,:) = avm(:,:,:)
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         b_avt_ddm(ji,jj,jk) = avt(ji,jj,jk)
         b_avs_ddm(ji,jj,jk) = avs(ji,jj,jk)
      END DO ; END DO ; END DO
   END SUBROUTINE r79_after_ddm

   SUBROUTINE r79_write( kt, Kbb, Kmm )
      !! Flush one file per rank per step, after zdf_iwm has run.
      INTEGER, INTENT(in) :: kt, Kbb, Kmm
      INTEGER :: iunit, ios, ji, jj, jk
      CHARACTER(LEN=96) :: clfile
      CHARACTER(LEN=16) :: clmagic
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: zwrk
      REAL(wp), ALLOCATABLE, DIMENSION(:,:)   :: zwrk2
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: zscal
      IF( .NOT. l_r79 )   RETURN
      IF( .NOT. ALLOCATED(b_avt_tke) )   CALL ctl_stop( 'round79b zdf writer was not armed' )
      IF( n_r79_mxl < 0 )   CALL ctl_stop( 'round79b zdf writer never saw the closure selectors' )
      WRITE(clfile,'("oracle_zdf_vmix_kt",I8.8,"_r",I4.4,".bin")') kt, narea-1
      OPEN( NEWUNIT=iunit, FILE=TRIM(clfile), ACCESS='STREAM', FORM='UNFORMATTED',   &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 )   CALL ctl_stop( 'round79b zdf writer cannot open output' )
      clmagic = 'NEMO_L4_ZDFV_1'
      WRITE(iunit) clmagic
      WRITE(iunit) 1, kt, Kbb, Kmm, narea-1, nimpp, njmpp, jpi, jpj, jpk,           &
         &         STORAGE_SIZE(1._wp), 24, n_r79_mxl,                              &
         &         MERGE(1,0,ln_zdfiwm), MERGE(1,0,ln_zdfddm)
      ALLOCATE( zwrk(jpi,jpj,jpk), zwrk2(jpi,jpj), zscal(1,1,jpk) )
#define R79_3D(name,value) WRITE(iunit) name ; WRITE(iunit) 3,jpi,jpj,jpk,1,1,1 ; WRITE(iunit) value
#define R79_2D(name,value) WRITE(iunit) name ; WRITE(iunit) 2,jpi,jpj,1,1,1,1   ; WRITE(iunit) value
#define R79_1D(name,value) WRITE(iunit) name ; WRITE(iunit) 1,1,1,jpk,1,1,1     ; WRITE(iunit) value
      R79_3D('avt_after_tke   ',b_avt_tke)
      R79_3D('avm_after_tke   ',b_avm_tke)
      R79_3D('avt_after_rnf   ',b_avt_rnf)
      R79_3D('avt_after_evd   ',b_avt_evd)
      R79_3D('avm_after_evd   ',b_avm_evd)
      R79_3D('avt_after_ddm   ',b_avt_ddm)
      R79_3D('avs_after_ddm   ',b_avs_ddm)
      R79_3D('avm_after_ddm   ',b_avm_ddm)
      zwrk(:,:,:) = 0._wp
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj,jk) = avt(ji,jj,jk)
      END DO ; END DO ; END DO
      R79_3D('avt_after_iwm   ',zwrk)
      R79_3D('avm_after_iwm   ',avm)
      zwrk(:,:,:) = 0._wp
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj,jk) = avs(ji,jj,jk)
      END DO ; END DO ; END DO
      R79_3D('avs_after_iwm   ',zwrk)
      zwrk(:,:,:) = 0._wp
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj,jk) = avt_k(ji,jj,jk)
      END DO ; END DO ; END DO
      R79_3D('avt_k           ',zwrk)
      R79_3D('avm_k           ',avm_k)
      zwrk(:,:,:) = 0._wp
      DO jk = 1, jpk ; DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk(ji,jj,jk) = en(ji,jj,jk)
      END DO ; END DO ; END DO
      R79_3D('en              ',zwrk)
      R79_3D('dissl           ',b_dissl)
      R79_3D('mxlm            ',b_mxlm)
      R79_3D('mxld            ',b_mxld)
      R79_3D('wmask           ',wmask)
      R79_3D('tmask           ',tmask)
      zwrk2(:,:) = 0._wp
      IF( ln_rnf_mouth ) THEN
         DO jj = 1, jpj ; DO ji = 1, jpi
            zwrk2(ji,jj) = rnfmsk(ji,jj)
         END DO ; END DO
      ENDIF
      R79_2D('rnfmsk          ',zwrk2)
      zwrk2(:,:) = 0._wp
      DO jj = ntsj, ntej ; DO ji = ntsi, ntei
         zwrk2(ji,jj) = avtb_2d(ji,jj)
      END DO ; END DO
      R79_2D('avtb_2d         ',zwrk2)
      zscal(1,1,:) = avtb(:)
      R79_1D('avtb            ',zscal)
      zscal(1,1,:) = avmb(:)
      R79_1D('avmb            ',zscal)
      zscal(:,:,:) = 0._wp
      zscal(1,1,1) = z_r79_ediff
      zscal(1,1,2) = z_r79_rmxl_min
      zscal(1,1,3) = REAL( nkrnf, wp )
      zscal(1,1,4) = MERGE( 1._wp, 0._wp, ln_rnf_mouth )
      zscal(1,1,5) = MERGE( 1._wp, 0._wp, ln_zdfevd )
      R79_1D('closure_scalars ',zscal)
#undef R79_3D
#undef R79_2D
#undef R79_1D
      CLOSE(iunit)
      DEALLOCATE( zwrk, zwrk2, zscal )
      WRITE(numout,*) 'ORCA2_R79B_ZDF_VMIX ', kt, narea-1, TRIM(clfile)
   END SUBROUTINE r79_write

END MODULE zdfphy_round79b_writer
