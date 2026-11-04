! Write-only hooks for the dump build (patch_dump.py): print intermediate
! fields of one mynnedmf call as "dump_<name> n v1 .. vn".  The patched copy
! only adds these calls; the generator checks its final outputs are bitwise
! the unpatched oracle's.
module mynn_dump_mod
  use ccpp_kind_types, only: kind_phys
  implicit none
  logical :: mynn_dump_on = .false.
contains
  subroutine dump1(name, x)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: x(:)
    if (mynn_dump_on) write(*,'(a,1x,i4,*(1x,es26.17e3))') 'dump_'//name, size(x), x
  end subroutine dump1
  subroutine dump0(name, x)
    character(len=*), intent(in) :: name
    real(kind_phys), intent(in) :: x
    if (mynn_dump_on) write(*,'(a,1x,i4,1x,es26.17e3)') 'dump_'//name, 1, x
  end subroutine dump0
end module mynn_dump_mod
