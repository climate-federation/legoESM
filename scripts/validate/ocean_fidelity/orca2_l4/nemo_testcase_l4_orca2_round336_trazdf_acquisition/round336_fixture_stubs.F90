MODULE par_kind
   INTEGER, PARAMETER :: wp=KIND(1.0D0)
END MODULE par_kind
MODULE par_oce
   INTEGER, PARAMETER :: jpi=7,jpj=6,jpk=5,jpkm1=4,jpts=2,jpt=3
   INTEGER, PARAMETER :: jp_tem=1,jp_sal=2
END MODULE par_oce
MODULE dom_oce
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi,jpj,jpk,jpt
   INTEGER :: narea=1
   INTEGER, PARAMETER :: ntsi=3,ntei=5,ntsj=3,ntej=4
   REAL(wp), ALLOCATABLE :: e3t_3d(:,:,:),e3w_1d(:),r3t(:,:,:),tmask(:,:,:)
END MODULE dom_oce
MODULE in_out_manager
   INTEGER, PARAMETER :: nit000=1
END MODULE in_out_manager
MODULE zdf_oce
   USE par_kind, ONLY : wp
   REAL(wp), ALLOCATABLE :: avt(:,:,:),avs(:,:,:)
END MODULE zdf_oce
MODULE ldfslp
   USE par_kind, ONLY : wp
   REAL(wp), ALLOCATABLE :: akz(:,:,:)
END MODULE ldfslp
MODULE lib_mpp
CONTAINS
   SUBROUTINE ctl_stop(message)
      CHARACTER(LEN=*), INTENT(in) :: message
      WRITE(*,*) 'CTL_STOP ',TRIM(message)
      ERROR STOP 71
   END SUBROUTINE ctl_stop
END MODULE lib_mpp
