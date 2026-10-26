MODULE l1_r53_ens
   !! Round-53 WRITE-only kt=3 stage-2 ENS operand recorder.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : lwp, nit000
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r53_ens_begin, r53_ens_zwz, r53_ens_before, r53_ens_after, &
      & r53_ens_finish

   CHARACTER(LEN=16), PARAMETER :: r53_magic = 'NEMO_L1_R53ENS01'
   INTEGER, PARAMETER :: r53_field_count = 12
   INTEGER, SAVE :: r53_unit = -1
   INTEGER, SAVE :: r53_pre_count = 0, r53_post_count = 0
   INTEGER, SAVE :: r53_before_count = 0, r53_after_count = 0
   INTEGER, SAVE :: r53_kt = -1, r53_kmm = -1, r53_kvor = -1
   LOGICAL, SAVE :: r53_active = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r53_zwz_pre, r53_zwz_post
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r53_zuav, r53_upair
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r53_uproduct, r53_ubefore
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r53_uafter, r53_zvau
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r53_vpair, r53_vproduct
   REAL(wp), ALLOCATABLE, SAVE, DIMENSION(:,:,:) :: r53_vbefore, r53_vafter

CONTAINS

   SUBROUTINE put3(name, value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: value
      CHARACTER(LEN=16) :: field
      field = name
      WRITE(r53_unit) field
      WRITE(r53_unit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3)
      WRITE(r53_unit) value
   END SUBROUTINE put3

   SUBROUTINE r53_ens_begin(kt, Kmm, kvor, is_cme)
      INTEGER, INTENT(in) :: kt, Kmm, kvor
      LOGICAL, INTENT(in) :: is_cme
      r53_active = lwp .AND. kt == nit000+2 .AND. Kmm == 3 .AND. is_cme
      IF(.NOT.r53_active) RETURN
      IF(r53_unit /= -1) CALL ctl_stop('round53: nested ENS record')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round53: writer requires fp64')
      r53_kt = kt ; r53_kmm = Kmm ; r53_kvor = kvor
      ALLOCATE(r53_zwz_pre(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_zwz_post(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_zuav(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_upair(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_uproduct(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_ubefore(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_uafter(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_zvau(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_vpair(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_vproduct(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_vbefore(ntsi:ntei,ntsj:ntej,jpkm1), &
         & r53_vafter(ntsi:ntei,ntsj:ntej,jpkm1))
      r53_zwz_pre=0._wp ; r53_zwz_post=0._wp
      r53_zuav=0._wp ; r53_upair=0._wp ; r53_uproduct=0._wp
      r53_ubefore=0._wp ; r53_uafter=0._wp
      r53_zvau=0._wp ; r53_vpair=0._wp ; r53_vproduct=0._wp
      r53_vbefore=0._wp ; r53_vafter=0._wp
      r53_pre_count=0 ; r53_post_count=0
      r53_before_count=0 ; r53_after_count=0
   END SUBROUTINE r53_ens_begin

   SUBROUTINE r53_ens_zwz(phase, ji, jj, jk, value)
      INTEGER, INTENT(in) :: phase, ji, jj, jk
      REAL(wp), INTENT(in) :: value
      IF(.NOT.r53_active) RETURN
      IF(ji < ntsi .OR. ji > ntei .OR. jj < ntsj .OR. jj > ntej) RETURN
      SELECT CASE(phase)
      CASE(1)
         r53_zwz_pre(ji,jj,jk)=value ; r53_pre_count=r53_pre_count+1
      CASE(2)
         r53_zwz_post(ji,jj,jk)=value ; r53_post_count=r53_post_count+1
      CASE DEFAULT
         CALL ctl_stop('round53: invalid zwz phase')
      END SELECT
   END SUBROUTINE r53_ens_zwz

   SUBROUTINE r53_ens_before(ji,jj,jk,zuav,upair,uproduct,ubefore, &
      & zvau,vpair,vproduct,vbefore)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: zuav,upair,uproduct,ubefore
      REAL(wp), INTENT(in) :: zvau,vpair,vproduct,vbefore
      IF(.NOT.r53_active) RETURN
      r53_zuav(ji,jj,jk)=zuav ; r53_upair(ji,jj,jk)=upair
      r53_uproduct(ji,jj,jk)=uproduct ; r53_ubefore(ji,jj,jk)=ubefore
      r53_zvau(ji,jj,jk)=zvau ; r53_vpair(ji,jj,jk)=vpair
      r53_vproduct(ji,jj,jk)=vproduct ; r53_vbefore(ji,jj,jk)=vbefore
      r53_before_count=r53_before_count+1
   END SUBROUTINE r53_ens_before

   SUBROUTINE r53_ens_after(ji,jj,jk,uafter,vafter)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: uafter,vafter
      IF(.NOT.r53_active) RETURN
      r53_uafter(ji,jj,jk)=uafter ; r53_vafter(ji,jj,jk)=vafter
      r53_after_count=r53_after_count+1
   END SUBROUTINE r53_ens_after

   SUBROUTINE r53_ens_finish
      INTEGER :: ios, expected
      IF(.NOT.r53_active) RETURN
      expected=(ntei-ntsi+1)*(ntej-ntsj+1)*jpkm1
      IF(r53_pre_count /= expected .OR. r53_post_count /= expected .OR. &
         & r53_before_count /= expected .OR. r53_after_count /= expected) &
         & CALL ctl_stop('round53: incomplete ENS operand record')
      OPEN(NEWUNIT=r53_unit,FILE='oracle_r53_ens_kt00000003_s2.bin', &
         & ACCESS='STREAM',FORM='UNFORMATTED',STATUS='REPLACE', &
         & ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round53: cannot open ENS record')
      WRITE(r53_unit) r53_magic
      WRITE(r53_unit) 1,r53_kt,2,r53_kmm,r53_kvor,ntei-ntsi+1, &
         & ntej-ntsj+1,jpkm1,ntsi,ntsj,STORAGE_SIZE(1._wp),r53_field_count
      CALL put3('zwz_prediv',r53_zwz_pre)
      CALL put3('zwz_postdiv',r53_zwz_post)
      CALL put3('zuav',r53_zuav)
      CALL put3('zwz_pair_u',r53_upair)
      CALL put3('product_u',r53_uproduct)
      CALL put3('rhs_before_u',r53_ubefore)
      CALL put3('rhs_after_u',r53_uafter)
      CALL put3('zvau',r53_zvau)
      CALL put3('zwz_pair_v',r53_vpair)
      CALL put3('product_v',r53_vproduct)
      CALL put3('rhs_before_v',r53_vbefore)
      CALL put3('rhs_after_v',r53_vafter)
      CLOSE(r53_unit) ; r53_unit=-1 ; r53_active=.FALSE.
      DEALLOCATE(r53_zwz_pre,r53_zwz_post,r53_zuav,r53_upair, &
         & r53_uproduct,r53_ubefore,r53_uafter,r53_zvau,r53_vpair, &
         & r53_vproduct,r53_vbefore,r53_vafter)
   END SUBROUTINE r53_ens_finish

END MODULE l1_r53_ens
