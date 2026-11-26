! Oracle: CAM cam_cesm2_1_rel_60 clubb_intr.F90 host omega -> CLUBB wm_zt and
! wm_zm (:1626-1674 grid and density, :1677-1680 wm_zt, :1778 wm_zm =
! zt2zm_api(wm_zt)), with zt2zm_api = CLUBB grid_class linear_interpolated_azm
! on the grid_type = 3 grid CLUBB builds from CAM's momentum heights (zt at
! the midpoints, ghost level below the surface; l_cubic_interp = .false. in
! CAM6).  All physics lines are verbatim includes (build.sh).
! Input file: ncol nlev, then per column zi(pverp) zm(pver) pmid pdel omega
! (CAM order, top first).  Output: per column wm_zt(nlev+1) and wm_zm(nlev+1)
! (CLUBB order, 1 = ghost / surface).
program wm_oracle
  use shr_kind_mod, only: r8 => shr_kind_r8
  use clubb_precision, only: core_rknd
  use ppgrid, only: pcols, pver, pverp
  use physconst, only: gravit, cpair
  use physics_types, only: physics_state
  use grid_x, only: gr
  implicit none
  integer, parameter :: ixcldice = 3, t_above = 1, t_below = 2
  real(core_rknd), parameter :: one = 1.0_core_rknd
  type(physics_state) :: state1
  integer :: ncol_in, nlev_in, ic, i, k, nlev, top_lev
  real(r8) :: exner_clubb(pcols,pver), thv(pcols,pver), radf_clubb(pcols,pver), qrl(pcols,pver)
  real(r8) :: grid_dx(pcols), grid_dy(pcols), sfc_elevation, host_dx, host_dy
  real(r8), dimension(pverp) :: zi_g, zt_g, p_in_Pa, exner, rho_ds_zt, invrs_rho_ds_zt, rho_in, &
       thv_ds_zt, rfrzm, radf, qrl_clubb, rho_zt, wm_zt, wm_zm
  real(r8) :: dz_g(pver)
  character(len=512) :: arg
  character(len=*), parameter :: fmt = '(*(es27.17e3))'

  top_lev = 1                      ! L32 top mid-level is above 100 Pa: CAM6 top_lev = 1
  nlev = pver + 1 - top_lev
  i = 1
  exner_clubb = 1._r8; thv = 300._r8; radf_clubb = 0._r8; qrl = 0._r8
  grid_dx = 1.e5_r8; grid_dy = 1.e5_r8; state1%q = 0._r8
  call get_command_argument(1, arg); open(10, file=trim(arg), status='old')
  read(10,*) ncol_in, nlev_in
  if (nlev_in /= pver) stop 'nlev'
  do ic = 1, ncol_in
     read(10,*) state1%zi(1,:); read(10,*) state1%zm(1,:)
     read(10,*) state1%pmid(1,:); read(10,*) state1%pdel(1,:); read(10,*) state1%omega(1,:)
     zi_g = 0._r8; zt_g = 0._r8; rho_in = 1._r8; wm_zt = -999._r8; wm_zm = -999._r8
     include 'inc_grid.inc'
     include 'inc_wm.inc'
     call setup_grid()
     include 'inc_wm_zm.inc'
     write(*,fmt) wm_zt(1:nlev+1)
     write(*,fmt) wm_zm(1:nlev+1)
  end do
  close(10)
contains

  subroutine setup_grid()
    ! grid_class setup_grid_heights, grid_type = 3, momentum_heights = zi_g
    integer :: k
    gr%nz = nlev + 1
    gr%zm(1:gr%nz) = zi_g(1:gr%nz)
    include 'inc_zt_grid_type3.inc'
    do k = 1, gr%nz
       gr%weights_zt2zm(t_above:t_below,k) = interpolated_azmk_imp(k)
    end do
  end subroutine setup_grid

  function interpolated_azmk_imp(m_lev) result(azt_weight)
    integer, intent(in) :: m_lev
    real(core_rknd), dimension(2) :: azt_weight
    real(core_rknd) :: factor
    integer :: k
    include 'inc_weights_zt2zm.inc'
  end function interpolated_azmk_imp

  function linear_interpolated_azm(azt)
    real(core_rknd), intent(in), dimension(gr%nz) :: azt
    real(core_rknd), dimension(gr%nz) :: linear_interpolated_azm
    integer :: k
    include 'inc_linear_interpolated_azm.inc'
  end function linear_interpolated_azm

  elemental real(core_rknd) function linear_interp_factor(factor, var_high, var_low)
    real(core_rknd), intent(in) :: factor, var_high, var_low
    include 'inc_linear_interp_factor.inc'
  end function linear_interp_factor

  function zt2zm_api(azt) result(azm)
    ! CLUBB's zt2zm_api -> redirect_interpolated_azm, l_cubic_interp = .false.
    real(r8), intent(in) :: azt(:)
    real(r8) :: azm(size(azt))
    azm = azt
    azm(1:gr%nz) = linear_interpolated_azm(azt(1:gr%nz))
  end function zt2zm_api
end program wm_oracle
