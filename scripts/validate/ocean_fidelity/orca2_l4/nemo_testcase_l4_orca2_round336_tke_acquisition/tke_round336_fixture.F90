PROGRAM tke_round336_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE par_kind, ONLY : wp
   USE par_oce
   USE oce
   USE dom_oce
   USE sbc_oce
   USE zdfdrg
   USE zdfmxl
   USE ice
   USE sbcwave
   USE zdf_oce
   USE zdftke, ONLY : zdf_tke_observed => zdf_tke, r336_fixture_config
   USE zdftke_reference, ONLY : zdf_tke_reference => zdf_tke, &
      & r336_reference_config => r336_fixture_config
   IMPLICIT NONE
   INTEGER :: i,j,k,rank
   CHARACTER(LEN=16) :: argument
   REAL(wp), ALLOCATABLE :: sh2(:,:,:),avt_observed(:,:,:),avt_reference(:,:,:)
   REAL(wp), ALLOCATABLE :: avm_observed(:,:,:),avm_reference(:,:,:)
   REAL(wp), ALLOCATABLE :: en_initial(:,:,:),en_reference(:,:,:)

   CALL GET_COMMAND_ARGUMENT(1,argument); READ(argument,*) rank; narea=rank+1
   ALLOCATE(uu(jpi,jpj,jpk,3),vv(jpi,jpj,jpk,3),rn2(jpi,jpj,jpk),rn2b(jpi,jpj,jpk))
   ALLOCATE(mbkt(jpi,jpj),mikt(jpi,jpj),e3t_3d(jpi,jpj,jpk),e3w_1d(jpk), &
      & gdepw_1d(jpk),r3t(jpi,jpj,3),gphit(jpi,jpj),ssmask(jpi,jpj), &
      & tmask(jpi,jpj,jpk),umask(jpi,jpj,jpk),vmask(jpi,jpj,jpk),wmask(jpi,jpj,jpk))
   ALLOCATE(fr_i(Nis0:Nie0,Njs0:Nje0),taum(Nis0:Nie0,Njs0:Nje0), &
      & utau(Nis0:Nie0,Njs0:Nje0),vtau(Nis0:Nie0,Njs0:Nje0))
   ALLOCATE(rCdU_bot(Nis0:Nie0,Njs0:Nje0),rCdU_top(Nis0:Nie0,Njs0:Nje0))
   ALLOCATE(nmln(Nis0:Nie0,Njs0:Nje0),hmlp(Nis0:Nie0,Njs0:Nje0))
   ALLOCATE(hm_i(Nis0:Nie0,Njs0:Nje0),h_i(jpi,jpj,1))
   ALLOCATE(hsw(Nis0:Nie0,Njs0:Nje0),phioc(Nis0:Nie0,Njs0:Nje0), &
      & ut0sd(Nis0:Nie0,Njs0:Nje0),vt0sd(Nis0:Nie0,Njs0:Nje0))
   ALLOCATE(en(Nis0:Nie0,Njs0:Nje0,jpk),avtb(jpk),avmb(jpk), &
      & avtb_2d(Nis0:Nie0,Njs0:Nje0))
   ALLOCATE(sh2(Nis0:Nie0,Njs0:Nje0,jpk), &
      & avt_observed(Nis0:Nie0,Njs0:Nje0,jpk), &
      & avt_reference(Nis0:Nie0,Njs0:Nje0,jpk), &
      & avm_observed(jpi,jpj,jpk),avm_reference(jpi,jpj,jpk), &
      & en_initial(Nis0:Nie0,Njs0:Nje0,jpk), &
      & en_reference(Nis0:Nie0,Njs0:Nje0,jpk))

   uu=0.02_wp+REAL(rank,wp)*0.001_wp
   vv=-0.01_wp-REAL(rank,wp)*0.001_wp
   rn2=1.e-5_wp; rn2b=8.e-6_wp
   mbkt=jpkm1; mikt=1; gphit=45._wp; ssmask=1._wp
   tmask=1._wp; umask=1._wp; vmask=1._wp; wmask=1._wp
   tmask(1,:,:)=0._wp; umask(1,:,:)=0._wp; vmask(1,:,:)=0._wp; wmask(1,:,:)=-0._wp
   DO k=1,jpk
      e3t_3d(:,:,k)=8._wp+REAL(k,wp)
      e3w_1d(k)=8.5_wp+REAL(k,wp)
      gdepw_1d(k)=REAL(k-1,wp)*10._wp
   END DO
   r3t=0._wp
   fr_i=0.2_wp+0.01_wp*REAL(rank,wp); hm_i=1.5_wp+0.1_wp*REAL(rank,wp)
   taum=0.05_wp+0.001_wp*REAL(rank,wp); utau=0.01_wp; vtau=0.02_wp
   rCdU_bot=-0.002_wp; rCdU_top=-0.001_wp; nmln=2; hmlp=25._wp
   h_i=hm_i(ntsi,ntsj); hsw=0._wp; phioc=0._wp; ut0sd=0._wp; vt0sd=0._wp
   avtb=1.e-5_wp; avmb=2.e-5_wp; avtb_2d=1._wp
   DO k=1,jpk; DO j=Njs0,Nje0; DO i=Nis0,Nie0
      sh2(i,j,k)=2.e-6_wp*REAL(i+j+k+rank,wp)
      en_initial(i,j,k)=1.e-4_wp+1.e-6_wp*REAL(i+j+k+rank,wp)
      avt_observed(i,j,k)=1.e-5_wp+1.e-7_wp*REAL(i+j+k+rank,wp)
   END DO; END DO; END DO
   avm_observed=3.e-5_wp
   avm_observed(1,1,1)=-0._wp
   avt_observed(ntsi,ntsj,jpk)=-0._wp
   en_initial(ntsi,ntsj,jpk)=-0._wp

   CALL r336_reference_config()
   CALL r336_fixture_config()
   en=en_initial; avm_reference=avm_observed; avt_reference=avt_observed
   CALL zdf_tke_reference(2,3,3,sh2,avm_reference,avt_reference)
   en_reference=en
   en=en_initial
   CALL zdf_tke_observed(2,3,3,sh2,avm_observed,avt_observed)
   CALL require_bits(en,en_reference,'en')
   CALL require_bits(avm_observed,avm_reference,'avm')
   CALL require_bits(avt_observed,avt_reference,'avt')
   WRITE(*,*) 'R336_PATCHED_TKE_FIXTURE_PASS',rank

CONTAINS

   SUBROUTINE require_bits(left,right,name)
      REAL(wp), INTENT(in) :: left(:,:,:),right(:,:,:)
      CHARACTER(LEN=*), INTENT(in) :: name
      INTEGER :: ii,jj,kk
      INTEGER(int64) :: left_bits,right_bits
      DO kk=1,SIZE(left,3); DO jj=1,SIZE(left,2); DO ii=1,SIZE(left,1)
         left_bits=TRANSFER(left(ii,jj,kk),left_bits)
         right_bits=TRANSFER(right(ii,jj,kk),right_bits)
         IF(left_bits/=right_bits) THEN
            WRITE(*,*) 'fixture mismatch ',TRIM(name),ii,jj,kk
            ERROR STOP 72
         ENDIF
      END DO; END DO; END DO
   END SUBROUTINE require_bits
END PROGRAM tke_round336_fixture
