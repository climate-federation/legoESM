! STUBS for CIME infrastructure needed to compile cime5.6.47 shr_flux_mod.F90
! standalone. None of these touch the shr_flux_atmOcn numerics.
module shr_log_mod
  use shr_kind_mod
  implicit none
  integer(SHR_KIND_IN) :: shr_log_Level = 0
  integer(SHR_KIND_IN) :: shr_log_Unit  = 6
contains
  pure function shr_log_errMsg(file, line)
    character(len=SHR_KIND_CX)   :: shr_log_errMsg
    character(len=*), intent(in) :: file
    integer         , intent(in) :: line
    write(shr_log_errMsg,'(a,a,i6)') 'ERROR in ', trim(file), line
  end function shr_log_errMsg
end module shr_log_mod

module shr_sys_mod
  implicit none
contains
  subroutine shr_sys_abort(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(*,*) 'SHR_SYS_ABORT: ', trim(msg)
    stop 99
  end subroutine shr_sys_abort
end module shr_sys_mod

module water_isotopes
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
contains
  ! isotope flux not audited: returns 0 (outputs evap_16O/HDO/18O ignored)
  subroutine wiso_flxoce(iso, rbot, zbot, wtbot, ts, rocn, ustar, re, ssq, qflx, qbot, qe)
    integer,  intent(in)  :: iso
    real(r8), intent(in)  :: rbot, zbot, wtbot, ts, rocn, ustar, re, ssq, qbot, qe
    real(r8), intent(out) :: qflx
    qflx = 0.0_r8
  end subroutine wiso_flxoce
end module water_isotopes
