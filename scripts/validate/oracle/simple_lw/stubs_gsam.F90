! Minimal gSAM 1.8.8 module stubs so SRC/rad_simple.f90 compiles standalone.
! Only the names rad_simple references (grid/vars/params/rad). No physics:
! all coefficients (coef, coef1, f0, xk, cp_spec, 0.008) stay hardcoded in
! the unmodified oracle routine. Profiles are 1-D in z (gSAM reference state).
module grid
  implicit none
  integer :: nx = 1, ny = 1, nz, nzm
  real :: dz, dtn = 1.0
  real, allocatable :: adz(:), zi(:)
end module grid
module vars
  implicit none
  real, allocatable :: qcl(:,:,:), qci(:,:,:), qv(:,:,:), t(:,:,:)
  real, allocatable :: rho(:), rhow(:), radlwdn(:), radqrlw(:)
end module vars
module params
  implicit none
  logical :: dolongwave = .true.
end module params
module rad
  implicit none
  real, allocatable :: qrad(:,:,:)
end module rad
