MODULE dynadv_round62_writer
   !! WRITE-only ORCA2 round-62 split of the executing vector-advection arm.
   USE oce
   USE dom_oce
   USE sbc_oce, ONLY : ln_vortex_force
   USE zdf_oce, ONLY : ln_zad_Aimp
   USE in_out_manager
   USE lib_mpp
   IMPLICIT NONE
   PRIVATE

   PUBLIC :: dynadv_round62_arm, dynadv_round62_before
   PUBLIC :: dynadv_round62_after_keg, dynadv_round62_after_zad

   INTEGER, SAVE :: r62_stage = 0
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_before_u, r62_before_v
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r62_after_keg_u, r62_after_keg_v

CONTAINS

   SUBROUTINE dynadv_round62_arm( kstg )
      INTEGER, INTENT(in) :: kstg
      r62_stage = kstg
   END SUBROUTINE dynadv_round62_arm

   SUBROUTINE dynadv_round62_before( kt, Kmm, Krhs, puu, pvv )
      INTEGER, INTENT(in) :: kt, Kmm, Krhs
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu, pvv
      IF( kt /= nit000 .OR. r62_stage /= 2 ) RETURN
      IF( ALLOCATED(r62_before_u) ) CALL ctl_stop( 'round62 vector writer already armed' )
      ALLOCATE( r62_before_u(jpi,jpj,jpk), r62_before_v(jpi,jpj,jpk) )
      ALLOCATE( r62_after_keg_u(jpi,jpj,jpk), r62_after_keg_v(jpi,jpj,jpk) )
      r62_before_u(:,:,:) = puu(:,:,:,Krhs)
      r62_before_v(:,:,:) = pvv(:,:,:,Krhs)
      r62_after_keg_u(:,:,:) = 0._wp
      r62_after_keg_v(:,:,:) = 0._wp
   END SUBROUTINE dynadv_round62_before

   SUBROUTINE dynadv_round62_after_keg( kt, Krhs, puu, pvv )
      INTEGER, INTENT(in) :: kt, Krhs
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu, pvv
      IF( kt /= nit000 .OR. r62_stage /= 2 ) RETURN
      IF( .NOT.ALLOCATED(r62_before_u) ) CALL ctl_stop( 'round62 vector writer was not armed' )
      r62_after_keg_u(:,:,:) = puu(:,:,:,Krhs)
      r62_after_keg_v(:,:,:) = pvv(:,:,:,Krhs)
   END SUBROUTINE dynadv_round62_after_keg

   SUBROUTINE dynadv_round62_after_zad( kt, Kmm, Krhs, kscheme, puu, pvv )
      INTEGER, INTENT(in) :: kt, Kmm, Krhs, kscheme
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpt), INTENT(in) :: puu, pvv
      INTEGER :: r62_unit, r62_ios, ji, jj, jk
      CHARACTER(LEN=96) :: r62_file
      CHARACTER(LEN=16) :: r62_magic
      REAL(wp), ALLOCATABLE, DIMENSION(:,:,:) :: r62_payload
      IF( kt /= nit000 .OR. r62_stage /= 2 ) RETURN
      IF( .NOT.ALLOCATED(r62_after_keg_u) ) CALL ctl_stop( 'round62 KEG frame is absent' )
      IF( STORAGE_SIZE(1._wp) /= 64 ) CALL ctl_stop( 'round62 vector writer requires fp64 wp' )
      IF( l_istiled ) CALL ctl_stop( 'round62 vector writer refuses tiling' )
      WRITE(r62_file,'("oracle_vector_adv_split_kt",I8.8,"_s2_r",I4.4,".bin")') kt, narea-1
      OPEN( NEWUNIT=r62_unit, FILE=TRIM(r62_file), ACCESS='STREAM', FORM='UNFORMATTED', &
         & STATUS='REPLACE', ACTION='WRITE', IOSTAT=r62_ios )
      IF( r62_ios /= 0 ) CALL ctl_stop( 'round62 vector writer cannot open output' )
      r62_magic = 'NEMO_L4_VADV_1'
      WRITE(r62_unit) r62_magic
      WRITE(r62_unit) 1, kt, r62_stage, Kmm, Krhs, narea-1, nimpp, njmpp, &
         & ntsi, ntei, ntsj, ntej, jpi, jpj, jpk, jpkm1, STORAGE_SIZE(1._wp), &
         & kscheme, MERGE(1,0,ln_vortex_force), MERGE(1,0,ln_zad_Aimp)
#define R62_3D(name,value) WRITE(r62_unit) name ; WRITE(r62_unit) 3,jpi,jpj,jpk ; WRITE(r62_unit) value
#define R62_2D(name,value) WRITE(r62_unit) name ; WRITE(r62_unit) 2,jpi,jpj,1   ; WRITE(r62_unit) value
      R62_3D('before_keg_u    ',r62_before_u)
      R62_3D('before_keg_v    ',r62_before_v)
      R62_3D('after_keg_u     ',r62_after_keg_u)
      R62_3D('after_keg_v     ',r62_after_keg_v)
      R62_3D('after_zad_u     ',puu(:,:,:,Krhs))
      R62_3D('after_zad_v     ',pvv(:,:,:,Krhs))
      R62_3D('uu_Kmm          ',puu(:,:,:,Kmm))
      R62_3D('vv_Kmm          ',pvv(:,:,:,Kmm))
      R62_3D('ww              ',ww)
      ALLOCATE( r62_payload(jpi,jpj,jpk) )
      r62_payload(:,:,:) = 0._wp
      R62_3D('wsd_effective   ',r62_payload)
      DO jk = 1, jpk ; DO jj = 1, jpj ; DO ji = 1, jpi
         r62_payload(ji,jj,jk) = e3u_3d(ji,jj,jk) * (1._wp + r3u(ji,jj,Kmm) * umask(ji,jj,jk))
      END DO ; END DO ; END DO
      R62_3D('e3u_Kmm         ',r62_payload)
      DO jk = 1, jpk ; DO jj = 1, jpj ; DO ji = 1, jpi
         r62_payload(ji,jj,jk) = e3v_3d(ji,jj,jk) * (1._wp + r3v(ji,jj,Kmm) * vmask(ji,jj,jk))
      END DO ; END DO ; END DO
      R62_3D('e3v_Kmm         ',r62_payload)
      R62_2D('e1e2t           ',e1e2t)
      R62_2D('e1e2u           ',e1e2u)
      R62_2D('e1e2v           ',e1e2v)
      R62_2D('r1_e1u          ',r1_e1u)
      R62_2D('r1_e2v          ',r1_e2v)
      R62_2D('r1_e1e2u        ',r1_e1e2u)
      R62_2D('r1_e1e2v        ',r1_e1e2v)
      R62_3D('umask           ',umask)
      R62_3D('vmask           ',vmask)
#undef R62_3D
#undef R62_2D
      CLOSE(r62_unit)
      DEALLOCATE( r62_payload, r62_before_u, r62_before_v, r62_after_keg_u, r62_after_keg_v )
      WRITE(numout,*) 'ORCA2_R62_VECTOR_ADV_SPLIT ', kt, r62_stage, narea-1, TRIM(r62_file)
   END SUBROUTINE dynadv_round62_after_zad

END MODULE dynadv_round62_writer
