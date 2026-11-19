! Stubs for the standalone WRF v4.8.0 module_cu_tiedtke build.
! module_model_constants: only the 7 names module_cu_tiedtke USEs
!   (r_d, r_v, cp, xlv, xls, xlf, g).
!   -DLEGOCONST : legoESM packages/core/legoesm/constants.py values, water-only
!                 (xls = xlv, xlf = 0: legoESM convection has no ice phase).
!   default     : stock WRF share/module_model_constants.F values.
module module_model_constants
  implicit none
#ifdef LEGOCONST
  real, parameter :: g = 9.80616      ! const-ok: legoESM constants.g fed to oracle
  real, parameter :: r_d = 287.05     ! const-ok: legoESM constants.R_d
  real, parameter :: r_v = 461.51     ! const-ok: legoESM constants.R_v
  real, parameter :: cp = 1004.64     ! const-ok: legoESM constants.c_pd
  real, parameter :: xlv = 2.501e6    ! const-ok: legoESM constants.L_v
  real, parameter :: xls = 2.501e6    ! water-only: L_s := L_v
  real, parameter :: xlf = 0.0        ! water-only: no fusion heat
#else
  real, parameter :: g = 9.81
  real, parameter :: r_d = 287.
  real, parameter :: cp = 7.*r_d/2.
  real, parameter :: r_v = 461.6
  real, parameter :: xls = 2.85e6
  real, parameter :: xlv = 2.5e6
  real, parameter :: xlf = 3.50e5
#endif
end module module_model_constants

module tdbg
  implicit none
  integer, parameter :: mc = 64, ml = 80
  real :: d_ptte(mc,ml), d_qte(mc,ml), d_cte(mc,ml), d_vom(mc,ml), d_vol(mc,ml), d_prec(mc)
  integer :: d_ktype(mc)
  real :: d_mfub0(mc), d_zdqpbl(mc), d_zdqcv(mc), d_zqumqe(mc)
  integer :: d_ktype0(mc), d_kcbot0(mc), d_ldcum0(mc)
  real :: d_zcape(mc), d_zheat(mc), d_zrelh(mc), d_mfub(mc)
  integer :: d_kcbot(mc), d_kctop(mc), d_ldcum(mc), d_idtop(mc), d_lodd(mc), d_icum
  real :: d_mfu(mc,ml), d_mfd(mc,ml), d_tu(mc,ml), d_qu(mc,ml), d_lu(mc,ml), d_lude(mc,ml)
  real :: d_dmfup(mc,ml), d_uu(mc,ml), d_vu(mc,ml), d_ud(mc,ml), d_vd(mc,ml)
end module tdbg
