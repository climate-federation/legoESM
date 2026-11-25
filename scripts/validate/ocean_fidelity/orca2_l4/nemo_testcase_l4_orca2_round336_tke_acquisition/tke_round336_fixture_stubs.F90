MODULE par_kind
   INTEGER, PARAMETER :: wp=KIND(1.0D0)
END MODULE par_kind

MODULE par_oce
   INTEGER, PARAMETER :: jpi=7,jpj=6,jpk=5,jpkm1=4
END MODULE par_oce

MODULE oce
   USE par_kind, ONLY : wp
   USE par_oce
   REAL(wp), ALLOCATABLE :: uu(:,:,:,:),vv(:,:,:,:),rn2(:,:,:),rn2b(:,:,:)
END MODULE oce

MODULE phycst
   USE par_kind, ONLY : wp
   REAL(wp), PARAMETER :: rho0=1026._wp,grav=9.80665_wp,rpi=3.141592653589793_wp
   REAL(wp), PARAMETER :: rsmall=1.e-20_wp,vkarmn=0.4_wp
END MODULE phycst

MODULE dom_oce
   USE par_kind, ONLY : wp
   USE par_oce
   INTEGER, PARAMETER :: Nis0=3,Nie0=5,Njs0=3,Nje0=4
   INTEGER, PARAMETER :: ntsi=Nis0,ntei=Nie0,ntsj=Njs0,ntej=Nje0
   INTEGER :: narea=1
   REAL(wp) :: rn_Dt=10800._wp
   LOGICAL :: ln_isfcav=.FALSE.
   INTEGER, ALLOCATABLE :: mbkt(:,:),mikt(:,:)
   REAL(wp), ALLOCATABLE :: e3t_3d(:,:,:),e3w_1d(:),gdepw_1d(:),r3t(:,:,:)
   REAL(wp), ALLOCATABLE :: gphit(:,:),ssmask(:,:),tmask(:,:,:),umask(:,:,:)
   REAL(wp), ALLOCATABLE :: vmask(:,:,:),wmask(:,:,:)
END MODULE dom_oce

MODULE sbc_oce
   USE par_kind, ONLY : wp
   USE dom_oce, ONLY : Nis0,Nie0,Njs0,Nje0
   LOGICAL :: ln_cpl=.FALSE.,ln_phioc=.FALSE.,ln_sdw=.FALSE.
   INTEGER :: nn_ice=2
   REAL(wp), ALLOCATABLE :: fr_i(:,:),taum(:,:),utau(:,:),vtau(:,:)
END MODULE sbc_oce

MODULE zdfdrg
   USE par_kind, ONLY : wp
   LOGICAL :: ln_drg_OFF=.FALSE.
   REAL(wp), ALLOCATABLE :: rCdU_bot(:,:),rCdU_top(:,:)
END MODULE zdfdrg

MODULE zdfmxl
   USE par_kind, ONLY : wp
   INTEGER, ALLOCATABLE :: nmln(:,:)
   REAL(wp), ALLOCATABLE :: hmlp(:,:)
END MODULE zdfmxl

MODULE zdftke_round326_writer
CONTAINS
   SUBROUTINE r326_write(kt,nn_mxlice,ln_mxl0)
      INTEGER, INTENT(in) :: kt,nn_mxlice
      LOGICAL, INTENT(in) :: ln_mxl0
   END SUBROUTINE r326_write
END MODULE zdftke_round326_writer

MODULE ice
   USE par_kind, ONLY : wp
   REAL(wp), ALLOCATABLE :: hm_i(:,:),h_i(:,:,:)
END MODULE ice

MODULE in_out_manager
   TYPE :: cfctl_type
      LOGICAL :: l_prtctl=.FALSE.
   END TYPE cfctl_type
   INTEGER, PARAMETER :: nit000=1,numout=6
   INTEGER :: numnam_ref=10,numnam_cfg=11,numond=12,numror=13,numrow=14,nitrst=0
   LOGICAL :: lwp=.FALSE.,lwm=.FALSE.,ln_rstart=.FALSE.
   TYPE(cfctl_type) :: sn_cfctl
END MODULE in_out_manager

MODULE iom
   USE par_kind, ONLY : wp
CONTAINS
   LOGICAL FUNCTION iom_use(name)
      CHARACTER(LEN=*), INTENT(in) :: name
      iom_use=.FALSE.
   END FUNCTION iom_use
   SUBROUTINE iom_put(name,value)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:,:)
   END SUBROUTINE iom_put
END MODULE iom

MODULE lib_mpp
CONTAINS
   SUBROUTINE ctl_stop(message1,message2)
      CHARACTER(LEN=*), INTENT(in) :: message1
      CHARACTER(LEN=*), OPTIONAL, INTENT(in) :: message2
      WRITE(*,*) 'CTL_STOP ',TRIM(message1)
      ERROR STOP 71
   END SUBROUTINE ctl_stop
END MODULE lib_mpp

MODULE prtctl
   USE par_kind, ONLY : wp
CONTAINS
   SUBROUTINE prt_ctl(tab3d_1,clinfo1,tab3d_2,clinfo2)
      REAL(wp), INTENT(in) :: tab3d_1(:,:,:)
      REAL(wp), OPTIONAL, INTENT(in) :: tab3d_2(:,:,:)
      CHARACTER(LEN=*), INTENT(in) :: clinfo1
      CHARACTER(LEN=*), OPTIONAL, INTENT(in) :: clinfo2
   END SUBROUTINE prt_ctl
END MODULE prtctl

MODULE sbcwave
   USE par_kind, ONLY : wp
   LOGICAL :: cpl_phioc=.FALSE.,cpl_sdrftx=.FALSE.
   REAL(wp), ALLOCATABLE :: hsw(:,:),phioc(:,:),ut0sd(:,:),vt0sd(:,:)
END MODULE sbcwave

MODULE zdf_oce
   USE par_kind, ONLY : wp
   REAL(wp), ALLOCATABLE :: en(:,:,:),avtb(:),avmb(:),avtb_2d(:,:)
END MODULE zdf_oce

MODULE dynzdf_round328_writer
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE par_kind, ONLY : wp
CONTAINS
   INTEGER(int64) FUNCTION checksum2(value)
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER :: i,j
      INTEGER(int64) :: word
      checksum2=0_int64
      DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j),word); checksum2=IEOR(checksum2,word)
      END DO; END DO
   END FUNCTION checksum2
   INTEGER(int64) FUNCTION checksum3(value)
      REAL(wp), INTENT(in) :: value(:,:,:)
      INTEGER :: i,j,k
      INTEGER(int64) :: word
      checksum3=0_int64
      DO k=1,SIZE(value,3); DO j=1,SIZE(value,2); DO i=1,SIZE(value,1)
         word=TRANSFER(value(i,j,k),word); checksum3=IEOR(checksum3,word)
      END DO; END DO; END DO
   END FUNCTION checksum3
END MODULE dynzdf_round328_writer
