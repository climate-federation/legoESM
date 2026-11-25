MODULE par_kind
   INTEGER, PARAMETER :: wp=KIND(1.0D0)
END MODULE par_kind

MODULE par_oce
   INTEGER, PARAMETER :: jpi=7,jpj=6,jpk=5,jpkm1=4,jpts=2,jpt=3
   INTEGER, PARAMETER :: jp_tem=1,jp_sal=2
END MODULE par_oce

MODULE oce
   USE par_kind, ONLY : wp
   USE par_oce
   REAL(wp) :: rDt=10800._wp
   REAL(wp), ALLOCATABLE :: wi(:,:,:)
END MODULE oce

MODULE dom_oce
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi,jpj,jpk,jpt
   INTEGER :: narea=1
   INTEGER, PARAMETER :: ntsi=3,ntei=5,ntsj=3,ntej=4
   INTEGER :: ntile=1
   LOGICAL :: l_istiled=.FALSE.
   REAL(wp), ALLOCATABLE :: e3t_3d(:,:,:),e3w_1d(:),r3t(:,:,:),tmask(:,:,:)
END MODULE dom_oce

MODULE phycst
END MODULE phycst

MODULE zdf_oce
   USE par_kind, ONLY : wp
   LOGICAL :: ln_zdfddm=.TRUE.,ln_zad_Aimp=.FALSE.
   REAL(wp), ALLOCATABLE :: avt(:,:,:),avs(:,:,:)
END MODULE zdf_oce

MODULE zdfmfc
   USE par_kind, ONLY : wp
   LOGICAL :: ln_zdfmfc=.FALSE.
   REAL(wp), ALLOCATABLE :: edmfm(:,:,:),edmftra(:,:,:,:)
END MODULE zdfmfc

MODULE sbc_oce
END MODULE sbc_oce

MODULE ldftra
END MODULE ldftra

MODULE ldfslp
   USE par_kind, ONLY : wp
   LOGICAL :: l_ldfslp=.TRUE.,ln_traldf_msc=.TRUE.
   REAL(wp), ALLOCATABLE :: akz(:,:,:),ah_wslp2(:,:,:)
END MODULE ldfslp

MODULE trd_oce
   LOGICAL :: l_trdtra=.FALSE.
   INTEGER, PARAMETER :: jptra_zdf=1
END MODULE trd_oce

MODULE trdtra
   USE par_kind, ONLY : wp
CONTAINS
   SUBROUTINE trd_tra(kt,Kmm,Krhs,kind,tracer,code,trend)
      INTEGER, INTENT(in) :: kt,Kmm,Krhs,tracer,code
      CHARACTER(LEN=*), INTENT(in) :: kind
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: trend
   END SUBROUTINE trd_tra
END MODULE trdtra

MODULE eosbn2
   USE par_kind, ONLY : wp
   LOGICAL :: ln_SEOS=.FALSE.
   REAL(wp) :: rn_b0=1._wp
END MODULE eosbn2

MODULE in_out_manager
   INTEGER, PARAMETER :: nit000=1,numout=6
   LOGICAL :: ln_timing=.FALSE.,lwp=.FALSE.
END MODULE in_out_manager

MODULE prtctl
   USE par_kind, ONLY : wp
   TYPE :: cfctl_type
      LOGICAL :: l_prtctl=.FALSE.
   END TYPE cfctl_type
   TYPE(cfctl_type) :: sn_cfctl
CONTAINS
   SUBROUTINE prt_ctl(tab3d_1,clinfo1,mask1,tab3d_2,clinfo2,mask2,clinfo3)
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: tab3d_1,mask1
      REAL(wp), DIMENSION(:,:,:), OPTIONAL, INTENT(in) :: tab3d_2,mask2
      CHARACTER(LEN=*), INTENT(in) :: clinfo1
      CHARACTER(LEN=*), OPTIONAL, INTENT(in) :: clinfo2,clinfo3
   END SUBROUTINE prt_ctl
END MODULE prtctl

MODULE lbclnk
END MODULE lbclnk

MODULE lib_mpp
CONTAINS
   SUBROUTINE ctl_stop(message)
      CHARACTER(LEN=*), INTENT(in) :: message
      WRITE(*,*) 'CTL_STOP ',TRIM(message)
      ERROR STOP 71
   END SUBROUTINE ctl_stop
END MODULE lib_mpp

MODULE timing
CONTAINS
   SUBROUTINE timing_start(name)
      CHARACTER(LEN=*), INTENT(in) :: name
   END SUBROUTINE timing_start
   SUBROUTINE timing_stop(name)
      CHARACTER(LEN=*), INTENT(in) :: name
   END SUBROUTINE timing_stop
END MODULE timing
