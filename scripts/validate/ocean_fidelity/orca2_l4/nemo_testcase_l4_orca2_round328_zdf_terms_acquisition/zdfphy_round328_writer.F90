MODULE zdfphy_round328_writer
   !! WRITE-only kt=1 zdf_phy boundary recorder for ORCA2 round 328.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : nit000
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE lib_mpp, ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r328_zp_begin, r328_zp_ddm_cell, r328_zp_boundary2
   PUBLIC :: r328_zp_iwm_level, r328_zp_finish

   INTEGER, SAVE :: nunit = -1, nx = 0, ny = 0, nz = 0
   INTEGER, SAVE :: nkt = -1, nkbb = -1, nkmm = -1, nkrhs = -1
   INTEGER, SAVE :: nddm = 0, niwm = 0
   LOGICAL, SAVE :: lactive = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: boundary(:,:,:,:,:), tke(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r3t_kmm(:,:), ht0(:,:), mbkt_value(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: ddm(:,:,:,:), iwm(:,:,:,:)

CONTAINS

   FUNCTION checksum2(value) RESULT(checksum)
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER(int64) :: checksum, word
      INTEGER :: ji, jj
      checksum = 0_int64
      DO jj=1,SIZE(value,2); DO ji=1,SIZE(value,1)
         word = TRANSFER(value(ji,jj),word)
         checksum = IEOR(checksum,word)
      END DO; END DO
   END FUNCTION checksum2

   FUNCTION checksum3(value) RESULT(checksum)
      REAL(wp), INTENT(in) :: value(:,:,:)
      INTEGER(int64) :: checksum, word
      INTEGER :: ji, jj, jk
      checksum = 0_int64
      DO jk=1,SIZE(value,3); DO jj=1,SIZE(value,2); DO ji=1,SIZE(value,1)
         word = TRANSFER(value(ji,jj,jk),word)
         checksum = IEOR(checksum,word)
      END DO; END DO; END DO
   END FUNCTION checksum3

   SUBROUTINE put2(name,value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:)
      CHARACTER(LEN=24) :: field
      field=name
      WRITE(nunit) field
      WRITE(nunit) 2,SIZE(value,1),SIZE(value,2),1,ntsi,ntsj,2
      WRITE(nunit) checksum2(value)
      WRITE(nunit) value
   END SUBROUTINE put2

   SUBROUTINE put3(name,value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:,:)
      CHARACTER(LEN=24) :: field
      field=name
      WRITE(nunit) field
      WRITE(nunit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),ntsi,ntsj,2
      WRITE(nunit) checksum3(value)
      WRITE(nunit) value
   END SUBROUTINE put3

   SUBROUTINE r328_zp_begin(kt,Kbb,Kmm,Krhs,pavm,pavt,pavs,pen,pavm_k, &
      & pavt_k,prn2,pr3t,pht0,pmbkt,pwmask)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Krhs
      REAL(wp), INTENT(in) :: pavm(:,:,:),pavt(:,:,:),pavs(:,:,:)
      REAL(wp), INTENT(in) :: pen(:,:,:),pavm_k(:,:,:),pavt_k(:,:,:)
      REAL(wp), INTENT(in) :: prn2(:,:,:),pr3t(:,:),pht0(:,:),pwmask(:,:,:)
      INTEGER, INTENT(in) :: pmbkt(:,:)
      lactive=kt == nit000 .AND. Kbb == 1 .AND. Kmm == 1 .AND. Krhs == 3
      IF(.NOT.lactive) RETURN
      IF(ALLOCATED(boundary)) CALL ctl_stop('round328 zdfphy: nested record')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round328 zdfphy: requires fp64')
      nx=SIZE(pavm,1); ny=SIZE(pavm,2); nz=SIZE(pavm,3)
      IF(ANY(SHAPE(pavt) /= SHAPE(pavm)) .OR. ANY(SHAPE(pavs) /= SHAPE(pavm))) &
         & CALL ctl_stop('round328 zdfphy: boundary-1 shape mismatch')
      ALLOCATE(boundary(3,3,nx,ny,nz),tke(5,nx,ny,nz), &
         & r3t_kmm(nx,ny),ht0(nx,ny),mbkt_value(nx,ny), &
         & ddm(3,nx,ny,nz),iwm(3,nx,ny,nz))
      boundary=0._wp; tke=0._wp; ddm=0._wp; iwm=0._wp
      boundary(1,1,:,:,:)=pavm; boundary(1,2,:,:,:)=pavt; boundary(1,3,:,:,:)=pavs
      tke(1,:,:,:)=pen; tke(2,:,:,:)=pavm_k; tke(3,:,:,:)=pavt_k
      tke(4,:,:,:)=prn2; tke(5,:,:,:)=pwmask
      r3t_kmm=pr3t; ht0=pht0; mbkt_value=REAL(pmbkt,wp)
      nkt=kt; nkbb=Kbb; nkmm=Kmm; nkrhs=Krhs; nddm=0; niwm=0
   END SUBROUTINE r328_zp_begin

   SUBROUTINE r328_zp_ddm_cell(ji,jj,jk,pheat,psalt,pmax)
      INTEGER, INTENT(in) :: ji,jj,jk
      REAL(wp), INTENT(in) :: pheat,psalt,pmax
      INTEGER :: i,j,k
      IF(.NOT.lactive) RETURN
      i=ji-ntsi+1; j=jj-ntsj+1; k=jk-1
      IF(i<1 .OR. i>nx .OR. j<1 .OR. j>ny .OR. k<1 .OR. k>nz) &
         & CALL ctl_stop('round328 zdfphy: DDM cell outside record')
      ddm(1,i,j,k)=pheat; ddm(2,i,j,k)=psalt; ddm(3,i,j,k)=pmax
      nddm=nddm+1
   END SUBROUTINE r328_zp_ddm_cell

   SUBROUTINE r328_zp_boundary2(pavm,pavt,pavs)
      REAL(wp), INTENT(in) :: pavm(:,:,:),pavt(:,:,:),pavs(:,:,:)
      IF(.NOT.lactive) RETURN
      IF(nddm /= nx*ny*nz) CALL ctl_stop('round328 zdfphy: incomplete DDM operands')
      IF(ANY(SHAPE(pavm) /= (/nx,ny,nz/))) CALL ctl_stop('round328 zdfphy: boundary-2 shape')
      boundary(2,1,:,:,:)=pavm; boundary(2,2,:,:,:)=pavt; boundary(2,3,:,:,:)=pavs
   END SUBROUTINE r328_zp_boundary2

   SUBROUTINE r328_zp_iwm_level(jk,pwave,pemx,preb)
      INTEGER, INTENT(in) :: jk
      REAL(wp), INTENT(in) :: pwave(:,:),pemx(:,:),preb(:,:)
      INTEGER :: k
      IF(.NOT.lactive) RETURN
      k=jk-1
      IF(k<1 .OR. k>nz .OR. ANY(SHAPE(pwave) /= (/nx,ny/)) .OR. &
         & ANY(SHAPE(pemx) /= SHAPE(pwave)) .OR. ANY(SHAPE(preb) /= SHAPE(pwave))) &
         & CALL ctl_stop('round328 zdfphy: IWM level shape')
      iwm(1,:,:,k)=pwave; iwm(2,:,:,k)=pemx; iwm(3,:,:,k)=preb
      niwm=niwm+1
   END SUBROUTINE r328_zp_iwm_level

   SUBROUTINE r328_zp_finish(pavm,pavt,pavs)
      REAL(wp), INTENT(in) :: pavm(:,:,:),pavt(:,:,:),pavs(:,:,:)
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      IF(.NOT.lactive) RETURN
      IF(niwm /= nz) CALL ctl_stop('round328 zdfphy: incomplete IWM operands')
      IF(ANY(SHAPE(pavm) /= (/nx,ny,nz/))) CALL ctl_stop('round328 zdfphy: boundary-3 shape')
      boundary(3,1,:,:,:)=pavm; boundary(3,2,:,:,:)=pavt; boundary(3,3,:,:,:)=pavs
      WRITE(filename,'("oracle_zdfphy_kt",I8.8,"_r",I4.4,".bin")') nkt,narea-1
      OPEN(NEWUNIT=nunit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round328 zdfphy: cannot open record')
      magic='NEMO_L4_R328ZP'
      WRITE(nunit) magic,1,nkt,nkbb,nkmm,nkrhs,narea-1,nx,ny,nz,ntsi,ntsj, &
         & STORAGE_SIZE(1._wp),23
      CALL put3('b1_avm',boundary(1,1,:,:,:)); CALL put3('b1_avt',boundary(1,2,:,:,:))
      CALL put3('b1_avs',boundary(1,3,:,:,:)); CALL put3('tke_en',tke(1,:,:,:))
      CALL put3('tke_avm_k',tke(2,:,:,:)); CALL put3('tke_avt_k',tke(3,:,:,:))
      CALL put3('tke_rn2',tke(4,:,:,:)); CALL put2('tke_r3t_kmm',r3t_kmm)
      CALL put2('tke_ht_0',ht0); CALL put2('tke_mbkt',mbkt_value)
      CALL put3('tke_wmask',tke(5,:,:,:)); CALL put3('b2_avm',boundary(2,1,:,:,:))
      CALL put3('b2_avt',boundary(2,2,:,:,:)); CALL put3('b2_avs',boundary(2,3,:,:,:))
      CALL put3('ddm_heat_inc',ddm(1,:,:,:)); CALL put3('ddm_salt_inc',ddm(2,:,:,:))
      CALL put3('ddm_max_inc',ddm(3,:,:,:)); CALL put3('b3_avm',boundary(3,1,:,:,:))
      CALL put3('b3_avt',boundary(3,2,:,:,:)); CALL put3('b3_avs',boundary(3,3,:,:,:))
      CALL put3('iwm_zav_wave',iwm(1,:,:,:)); CALL put3('iwm_zemx_iwm',iwm(2,:,:,:))
      CALL put3('iwm_zreb',iwm(3,:,:,:))
      CLOSE(nunit); nunit=-1
      DEALLOCATE(boundary,tke,r3t_kmm,ht0,mbkt_value,ddm,iwm)
      lactive=.FALSE.
   END SUBROUTINE r328_zp_finish

END MODULE zdfphy_round328_writer
