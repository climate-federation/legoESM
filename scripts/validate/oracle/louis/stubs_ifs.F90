! Minimal OpenIFS infrastructure stubs so the REAL arpifs/phys_ec/vdfexcu.F90
! compiles standalone.  No physics here.  Real parphy.F90 and yoevdfs.F90 are
! compiled from the oracle tree; physical constants (YOMCST) are set by the driver
! to legoESM values.
module parkind1
  implicit none
  integer, parameter :: jpim = selected_int_kind(9)
  integer, parameter :: jprb = selected_real_kind(13,300)
  integer, parameter :: jprd = selected_real_kind(13,300)
  integer, parameter :: jprm = selected_real_kind(6,37)
end module parkind1

module yomhook
  use parkind1, only: jprd
  implicit none
  logical :: lhook = .false.
  integer, parameter :: jphook = jprd
contains
  subroutine dr_hook(name, k, h)
    character(len=*), intent(in) :: name
    integer, intent(in) :: k
    real(jphook), intent(inout) :: h
  end subroutine dr_hook
end module yomhook

module yomcst
  use parkind1, only: jprb
  implicit none
  real(jprb) :: rg, rd, rcpd, retv, rlvtt
end module yomcst

module yoegwd
  use parkind1, only: jpim
  implicit none
  type tegwd
    logical :: lrdiff_strato = .false.
    integer(jpim) :: ndiff_strato = 0
  end type tegwd
end module yoegwd

module yoephy
  use iso_c_binding, only: c_ptr
  implicit none
  type tephy
    type(c_ptr) :: ysurf
  end type tephy
end module yoephy

module yomdyncore
  use parkind1, only: jprb
  implicit none
  real(jprb) :: rplrg = 1.0_jprb, rpldare = 1.0_jprb
end module yomdyncore

module spp_gen_mod
  use parkind1, only: jprb, jpim
  implicit none
  type spp_pert
    integer(jpim) :: mp = 1
    real(jprb) :: mu(1) = 0.0_jprb, xmag(1) = 0.0_jprb
  end type spp_pert
end module spp_gen_mod

module spp_mod
  use parkind1, only: jpim
  use spp_gen_mod, only: spp_pert
  implicit none
  type tspp_ptr
    integer(jpim) :: vdexc = 0, rkap = 0, rkap1 = 0, rkap2 = 0, rkap3 = 0
  end type tspp_ptr
  type tspp_sm
    integer(jpim) :: nrftotal = 1
    type(spp_pert) :: pn(1)
  end type tspp_sm
  type tspp_config
    logical :: lspp = .false.
    type(tspp_ptr) :: pptr
    type(tspp_sm) :: sm
  end type tspp_config
end module spp_mod

subroutine surf_inq(ydsurf, prepust)
  use iso_c_binding, only: c_ptr
  use parkind1, only: jprb
  implicit none
  type(c_ptr), intent(in) :: ydsurf
  real(jprb), optional, intent(out) :: prepust
  if (present(prepust)) prepust = 1.0e-4_jprb   ! only feeds ZUST (mixed-layer branch, disabled)
end subroutine surf_inq
