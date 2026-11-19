! gSAM 1.8.8 tracesini oracle (SRC/RAD_RRTM/rad.f90 lines 1219-1431, #included VERBATIM
! as tracesini.inc by build.sh).  The module below stands in for the gSAM modules
! tracesini reads (grid, params, rad state, MPI helpers); ggr is gSAM's own value
! (SRC/consts.f90).  Built with -fdefault-real-8 (gSAM's default reals in fp64).
module sam_stub
  use parkind, only : kind_rb, kind_im
  use rrlw_ncpar, only : maxAbsorberNameLength, status, getAbsorberIndex
  implicit none
  integer :: nz, nzm
  real, allocatable :: pres(:), presi(:)
  logical :: masterproc = .true., dompi = .false.
  character(len=512) :: rundatadir
  integer :: year = 2000
  real(8), parameter :: ggr = 9.79764d0
  real :: nxco2 = 1.
  logical :: notracegases = .false., docurrentco2 = .false.
  real(kind_rb), allocatable, dimension(:) :: o3, co2, ch4, n2o, o2, cfc11, cfc12, cfc22, ccl4
contains
  real function co2_ppm(y)
    integer, intent(in) :: y
    co2_ppm = 0.
  end function co2_ppm
  subroutine task_abort()
    stop 'task_abort'
  end subroutine task_abort
  subroutine task_bcast_real8(r, a, n)
    integer :: r, n
    real(kind_rb) :: a(n)
  end subroutine task_bcast_real8
#include "tracesini.inc"
end module sam_stub

program oracle_driver
  ! usage: oracle_driver <rundatadir> <in.txt> <out.txt>
  !  in.txt : nzm ; then nzm lines pres(k) [hPa, k=1 surface-side]; then nz lines presi(k) [hPa]
  !  out.txt: nzm+1 lines: k, o3(k) [VMR]  (k = nzm+1 is the extra layer above the model top)
  use sam_stub
  implicit none
  character(len=512) :: fin, fout
  integer :: k, u
  call get_command_argument(1, rundatadir)
  call get_command_argument(2, fin)
  call get_command_argument(3, fout)
  open(newunit=u, file=trim(fin), status='old')
  read(u,*) nzm
  nz = nzm + 1
  allocate(pres(nzm), presi(nz))
  do k = 1, nzm
    read(u,*) pres(k)
  end do
  do k = 1, nz
    read(u,*) presi(k)
  end do
  close(u)
  allocate(o3(nzm+1), co2(nzm+1), ch4(nzm+1), n2o(nzm+1), o2(nzm+1), cfc11(nzm+1), &
           cfc12(nzm+1), cfc22(nzm+1), ccl4(nzm+1))
  call tracesini()
  open(newunit=u, file=trim(fout), status='replace')
  do k = 1, nzm+1
    write(u,'(i5,1x,es26.17e3)') k, o3(k)
  end do
  close(u)
  print *, 'ORACLE_DONE ', nzm
end program oracle_driver
