! Harness module for the MFCC_FILE build of the E3SM gw oracle: holds E3SM's own
! gw_init_beres (components/eam/src/physics/cam/gw_drag.F90:513-571, tag v3.0.1,
! extracted VERBATIM by build.sh into gw_init_beres.inc) and the gw_drag module
! variables it reads (maxh / maxuh parameters, gw_drag.F90:102,104, also verbatim
! in gw_beres_dims.inc).  Replaced infrastructure: ioFileMod getfil (the path is
! used as given), masterproc (single process), no SPMD broadcast.
module ioFileMod
  implicit none
contains
  subroutine getfil(fulpath, locfn)
    character(len=*), intent(in) :: fulpath
    character(len=*), intent(out) :: locfn
    locfn = fulpath
  end subroutine getfil
end module ioFileMod

module gw_beres_table
  use shr_kind_mod, only: r8 => shr_kind_r8
  use cam_logfile, only: iulog
  use cam_abortutils, only: endrun
  implicit none
#include "gw_beres_dims.inc"
  logical, parameter :: masterproc = .true.
  integer :: pgwv = -1
  character(len=256) :: gw_drag_file = ''
contains
#include "gw_init_beres.inc"
end module gw_beres_table
