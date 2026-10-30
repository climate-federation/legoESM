! Write-only dump buffer for the instrumented oracle copy (patch_dump.py).
module thom_dump
  implicit none
  integer, parameter :: ndmp = 10
  real, allocatable :: dmp(:,:,:)
end module thom_dump
