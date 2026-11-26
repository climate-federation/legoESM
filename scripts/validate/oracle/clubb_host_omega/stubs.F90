! Stand-ins for the CAM and CLUBB modules the host-omega lines read.
! Constants are legoESM's (packages/core/legoesm/constants.py), as in the
! clubb_glue oracle; CLUBB's grid is a plain derived type with the fields
! grid_class.F90 reads (zm, zt, nz, weights_zt2zm).
module shr_kind_mod
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
end module shr_kind_mod

module clubb_precision
  use shr_kind_mod, only: shr_kind_r8
  integer, parameter :: core_rknd = shr_kind_r8
end module clubb_precision

module ppgrid
  integer, parameter :: pcols = 1, pver = PVER_VAL, pverp = pver + 1
end module ppgrid

module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  real(r8), parameter :: gravit = 9.80616_r8   ! legoESM constants.g
  real(r8), parameter :: cpair  = 1004.64_r8   ! legoESM constants.c_pd
end module physconst

module physics_types
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver, pverp
  type physics_state
     real(r8) :: zi(pcols,pverp), zm(pcols,pver), pmid(pcols,pver), pdel(pcols,pver)
     real(r8) :: omega(pcols,pver), q(pcols,pver,3)
  end type physics_state
end module physics_types

module grid_x
  use clubb_precision, only: core_rknd
  use ppgrid, only: pverp
  type grid
     integer :: nz
     real(core_rknd) :: zm(pverp), zt(pverp), weights_zt2zm(2,pverp)
  end type grid
  type(grid) :: gr
end module grid_x
