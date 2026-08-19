! geopk/p_grad_c/one_grad_p oracle shim — the THREE dyn_core module
! scalars the extracted bodies read, plus the one constants_mod value.
!
! Everything else these three routines need (fv_grid_bounds_type,
! fv_grid_type incl. dg%is_initialized / rdxc / rdyc / rdx / rdy /
! grid / agrid / dxa / dya / edge_*, fv_flags_type%duogrid,
! great_circle_dist) already lives in the SHARED, unmodified
! fv3_swcore_shim.F90 — reuse it, do not duplicate.
!
! UNCERTAIN U1 (spec 6): dyn_core.F90:24 does
! ``use constants_mod, only: rdgas, radius, cp_air, pi`` and NO
! constants*.F90 exists anywhere under the Zenodo 8327578 symmetryclean
! tree, so the production numeric value of cp_air (and rdgas/kappa) is
! NOT verifiable from the sources.  cp_air is therefore a module
! VARIABLE set from the input header, not a parameter: both the Fortran
! oracle and the python port read the SAME header number, which makes
! the certificate INSENSITIVE to the true production value.  Pin the
! production values from the Zenodo run's namelist/fms.out before any
! production-fidelity claim (see also U6).
module geopk_shim_mod
  implicit none

  ! constants_mod (dyn_core.F90:24); geopk is the only one of the three
  ! extracted bodies that reads it (:2775).  NOT a parameter — see U1.
  real :: cp_air = -9.e9
  ! dyn_core.F90:72 module scalars, set once per dyn_core call at :244-248
  real :: ptk = -9.e9        ! ptop**akap        (dyn_core.F90:248)
  real :: peln1 = -9.e9      ! log(ptop)         (0. under -DSW_DYNAMICS)
  real :: rgrav = -9.e9      ! declared with them; unread by the three bodies

contains

  ! ptk uses the `**` OPERATOR while pk(i,j,k) inside geopk's k loop uses
  ! exp(akap*log(p)) (:2746).  These are DIFFERENT operations and may
  ! differ in the last bit — the port must mirror each form at its own
  ! site.  The -9.e9 poison above makes a driver that forgets this call
  ! produce obviously wrong output instead of a silent zero.
  subroutine set_dyncore_scalars(ptop, akap, cpair)
    real, intent(in) :: ptop, akap, cpair
    peln1 = log(ptop)        ! NON-SW build (dyn_core.F90:246)
    ptk = ptop ** akap       ! dyn_core.F90:248 — `**`, NOT exp(akap*log())
    cp_air = cpair
  end subroutine set_dyncore_scalars

end module geopk_shim_mod
