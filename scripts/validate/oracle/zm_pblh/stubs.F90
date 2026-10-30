! Minimal CAM infrastructure stubs for compiling CAM cam_cesm2_1_rel_60
! src/physics/cam/zm_conv.F90 standalone.  Provenance and modes: make_fixture.py docstring.

module shr_kind_mod
  implicit none
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
  integer, parameter :: shr_kind_in = kind(1)
end module shr_kind_mod

module spmd_utils
  implicit none
  logical :: masterproc = .true.
end module spmd_utils

module ppgrid
  implicit none
  integer, parameter :: pcols = PCOLS_VAL
  integer, parameter :: pver = PVER_VAL
  integer, parameter :: pverp = pver + 1
end module ppgrid

module cam_logfile
  implicit none
  integer :: iulog = 6
end module cam_logfile

module cam_abortutils
  implicit none
contains
  subroutine endrun(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) write(*,*) 'ENDRUN: ', trim(msg)
    stop 1
  end subroutine endrun
end module cam_abortutils

! physconst: module VARIABLES (not parameters) so the driver can select either
! legoESM's constants (faithfulness run) or CESM2.1 shr_const (stock run).
module physconst
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  real(r8) :: cpair, epsilo, gravit, latice, latvap, tmelt, rair, cpwv, cpliq, rh2o, zvir
  real(r8) :: h2otrip = 273.16_r8
contains
  subroutine set_constants(mode)
    integer, intent(in) :: mode   ! 1 = legoESM constants.py, 2 = CESM2.1 shr_const
    real(r8) :: rgas
    if (mode == 1) then
       gravit = 9.80616_r8; cpair = 1004.64_r8; latvap = 2.501e6_r8; latice = 3.337e5_r8
       tmelt = 273.15_r8; rair = 287.05_r8; rh2o = 461.51_r8
       cpwv = 1846.0_r8; cpliq = 4218.0_r8
       epsilo = rair/rh2o
    else
       rgas = 6.02214e26_r8*1.38065e-23_r8
       gravit = 9.80616_r8; cpair = 1.00464e3_r8; latvap = 2.501e6_r8; latice = 3.337e5_r8
       tmelt = 273.15_r8; rair = rgas/28.966_r8; rh2o = rgas/18.016_r8
       cpwv = 1.810e3_r8; cpliq = 4.188e3_r8
       epsilo = 18.016_r8/28.966_r8
    end if
    zvir = rh2o/rair - 1._r8
  end subroutine set_constants
end module physconst

module phys_control
  implicit none
contains
  logical function cam_physpkg_is(name)
    character(len=*), intent(in) :: name
    cam_physpkg_is = (trim(name) == 'cam6')
  end function cam_physpkg_is
end module phys_control

module phys_grid
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
contains
  subroutine get_rlat_all_p(lchnk, ncol, rlat)
    integer, intent(in) :: lchnk, ncol
    real(r8), intent(out) :: rlat(:)
    rlat = 0._r8
  end subroutine get_rlat_all_p
  real(r8) function get_rlat_p(lchnk, icol)
    integer, intent(in) :: lchnk, icol
    get_rlat_p = 0._r8
  end function get_rlat_p
  real(r8) function get_rlon_p(lchnk, icol)
    integer, intent(in) :: lchnk, icol
    get_rlon_p = 0._r8
  end function get_rlon_p
end module phys_grid

module constituents
  implicit none
contains
  character(len=3) function cnst_get_type_byind(ind)
    integer, intent(in) :: ind
    cnst_get_type_byind = 'wet'   ! CLDLIQ/CLDICE are moist mixing ratios in CAM6
  end function cnst_get_type_byind
end module constituents

! cloud_fraction::cldfrc_fice copied verbatim (cloud_fraction.F90:740-811);
! top_lev = trop_cloud_top_lev = 1 on L32 (trop_cloud_top_press = 1 hPa < p_top 2.255 hPa).
module cloud_fraction
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver
  implicit none
  integer :: top_lev = 1
contains
  subroutine cldfrc_fice(ncol, t, fice, fsnow)
    use physconst, only: tmelt
    integer,  intent(in)  :: ncol
    real(r8), intent(in)  :: t(pcols,pver)
    real(r8), intent(out) :: fice(pcols,pver)
    real(r8), intent(out) :: fsnow(pcols,pver)
    real(r8) :: tmax_fice, tmin_fice, tmax_fsnow, tmin_fsnow
    integer :: i,k
    tmax_fice = tmelt - 10._r8
    tmin_fice = tmax_fice - 30._r8
    tmax_fsnow = tmelt
    tmin_fsnow = tmelt - 5._r8
    fice(:,:top_lev-1) = 0._r8
    fsnow(:,:top_lev-1) = 0._r8
    do k=top_lev,pver
       do i=1,ncol
          if (t(i,k) > tmax_fice) then
             fice(i,k) = 0.0_r8
          else if (t(i,k) < tmin_fice) then
             fice(i,k) = 1.0_r8
          else
             fice(i,k) =(tmax_fice - t(i,k)) / (tmax_fice - tmin_fice)
          end if
          if (t(i,k) > tmax_fsnow) then
             fsnow(i,k) = 0.0_r8
          else if (t(i,k) < tmin_fsnow) then
             fsnow(i,k) = 1.0_r8
          else
             fsnow(i,k) =(tmax_fsnow - t(i,k)) / (tmax_fsnow - tmin_fsnow)
          end if
       end do
    end do
  end subroutine cldfrc_fice
end module cloud_fraction

! zm_microphysics: only the derived types (verbatim, zm_microphysics.F90:79-184);
! zm_mphy itself is dead in CAM6 (zmconv_microp=.false.) and patched out (P1).
module zm_microphysics
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  private
  public :: ptr2d, zm_aero_t, zm_conv_t
  type :: ptr2d
     real(r8), pointer :: val(:,:)
  end type ptr2d
  type :: zm_aero_t
     character(len=5) :: scheme
     integer :: nbulk = 0
     integer :: idxsul = -1, idxdst1 = -1, idxdst2 = -1, idxdst3 = -1, idxdst4 = -1, idxbcphi = -1
     real(r8),    allocatable :: num_to_mass_aer(:)
     type(ptr2d), allocatable :: mmr_bulk(:)
     real(r8),    allocatable :: mmrg_bulk(:,:,:)
     integer                  :: nmodes = 0
     integer,     allocatable :: nspec(:)
     type(ptr2d), allocatable :: num_a(:)
     type(ptr2d), allocatable :: mmr_a(:,:)
     real(r8),    allocatable :: numg_a(:,:,:)
     real(r8),    allocatable :: mmrg_a(:,:,:,:)
     real(r8),    allocatable :: voltonumblo(:), voltonumbhi(:)
     real(r8),    allocatable :: specdens(:,:), spechygro(:,:)
     integer :: mode_accum_idx = -1, mode_aitken_idx = -1, mode_coarse_idx = -1
     integer :: coarse_dust_idx = -1, coarse_nacl_idx = -1
     type(ptr2d), allocatable :: dgnum(:)
     real(r8),    allocatable :: dgnumg(:,:,:)
     real(r8) :: sigmag_aitken
  end type zm_aero_t
  type :: zm_conv_t
     real(r8), allocatable :: qi(:,:), qliq(:,:), qice(:,:), wu(:,:), sprd(:,:), qrain(:,:), qsnow(:,:)
     real(r8), allocatable :: qnl(:,:), qni(:,:), qnr(:,:), qns(:,:), frz(:,:)
     real(r8), allocatable :: autolm(:,:), accrlm(:,:), bergnm(:,:), fhtimm(:,:), fhtctm(:,:), fhmlm(:,:)
     real(r8), allocatable :: hmpim(:,:), accslm(:,:), dlfm(:,:), autoln(:,:), accrln(:,:), bergnn(:,:)
     real(r8), allocatable :: fhtimn(:,:), fhtctn(:,:), fhmln(:,:), accsln(:,:), activn(:,:), dlfn(:,:)
     real(r8), allocatable :: autoim(:,:), accsim(:,:), difm(:,:), nuclin(:,:), autoin(:,:), accsin(:,:)
     real(r8), allocatable :: hmpin(:,:), difn(:,:), cmel(:,:), cmei(:,:), trspcm(:,:), trspcn(:,:)
     real(r8), allocatable :: trspim(:,:), trspin(:,:), dcape(:), lambdadpcu(:,:), mudpcu(:,:)
     real(r8), allocatable :: di(:,:), dnl(:,:), dni(:,:), qide(:,:), qncde(:,:), qnide(:,:)
  end type zm_conv_t
end module zm_microphysics

! wv_saturation: qsat_water / qsat with a runtime curve switch.
!  sat_mode 1 = legoESM legoesm.thermo.saturation_specific_humidity, reproduced
!               exactly: Tetens 611.2*exp(17.67 Tc/(Tc+243.5)) with T clipped at
!               150 K, w = eps*e/(softplus(p-e-1)+1), smooth cap
!               w = 1 - softplus(20(1-w))/20, q = w/(1+w).  Used for BOTH calls
!               (that is what the port does).
!  sat_mode 2 = CESM2.1 wv_saturation: qsat_water = Goff-Gratch over water
!               (wv_sat_methods, real code); qsat = linear interpolation of the
!               1-K table of svp_trans (water/ice blend over 20 K below tmelt),
!               verbatim estblf; then svp_to_qsat and es=min(es,p).
!  sat_mode 5 = what the port does since the evaporation moved to CAM's mixed-
!               phase qsat: qsat_water as mode 1; qsat = CAM's wv_sat_svp_trans
!               (unmodified) over the legoESM Tetens/Clausius-Clapeyron curves
!               (patch W1, index 91), then mode 1's smooth conversion.
!  sat_mode 6 = what the port does since zm_conv_evap moved to CAM's own qsat:
!               qsat_water as mode 1; qsat = CESM2.1 (mode 2: estblf on the 1-K
!               table of the Goff-Gratch svp_trans, svp_to_qsat).
module wv_saturation
  use shr_kind_mod, only: r8 => shr_kind_r8
  use wv_sat_methods, only: wv_sat_methods_init, wv_sat_svp_water, wv_sat_svp_trans, &
                            svp_to_qsat => wv_sat_svp_to_qsat, &
                            Lego_idx, lego_tfrz, lego_ls, lego_rv
  implicit none
  private
  public :: qsat, qsat_water, wv_sat_init_stub, sat_mode
  integer :: sat_mode = 1
  real(r8) :: eps_l = 0.622_r8, tfrz_l = 273.15_r8
  real(r8), parameter :: tmin = 127.16_r8, tmax = 375.16_r8, ttrice = 20.00_r8, tboil = 373.16_r8
  real(r8), allocatable :: estbl(:)
  interface
     pure function c_log1p(x) bind(C, name='log1p')
       use iso_c_binding, only: c_double
       real(c_double), value, intent(in) :: x
       real(c_double) :: c_log1p
     end function c_log1p
  end interface
contains
  subroutine wv_sat_init_stub(mode, tmelt, h2otrip, epsilo)
    use physconst, only: latvap, latice, rh2o
    integer, intent(in) :: mode
    real(r8), intent(in) :: tmelt, h2otrip, epsilo
    character(len=256) :: errstring
    integer :: i, plenest
    sat_mode = mode
    eps_l = epsilo
    tfrz_l = tmelt
    ! patch W1 curve constants: legoESM T_freeze, L_s = L_v + L_f, R_v
    lego_tfrz = tmelt
    lego_ls = latvap + latice
    lego_rv = rh2o
    call wv_sat_methods_init(r8, tmelt, h2otrip, tboil, ttrice, epsilo, errstring)
    if (len_trim(errstring) > 0) then
       write(*,*) trim(errstring); stop 2
    end if
    plenest = ceiling(tmax-tmin) + 2
    if (allocated(estbl)) deallocate(estbl)
    allocate(estbl(plenest))
    do i = 1, plenest
       estbl(i) = wv_sat_svp_trans(tmin + real(i-1,r8))
    end do
  end subroutine wv_sat_init_stub

  elemental real(r8) function softplus(x)
    real(r8), intent(in) :: x
    softplus = max(x, 0._r8) + c_log1p(exp(-abs(x)))   ! = jax.nn.softplus (logaddexp(x,0))
  end function softplus

  elemental subroutine lego_qsat(t, p, es, qs)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    real(r8) :: tc, w
    tc = max(t, 150._r8) - tfrz_l
    es = 611.2_r8*exp(17.67_r8*tc/(tc + 243.5_r8))
    w = eps_l*es/(softplus(p - es - 1._r8) + 1._r8)
    w = 1._r8 - softplus(20._r8*(1._r8 - w))/20._r8
    qs = w/(1._r8 + w)
  end subroutine lego_qsat

  ! sat_mode 3: same Tetens curve, but CAM's hard e->q conversion (no softplus
  ! floor, no smooth 1 kg/kg cap): isolates the AD smoothing inside legoesm.thermo.
  elemental subroutine tetens_hard(t, p, es, qs)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    real(r8) :: tc
    tc = max(t, 150._r8) - tfrz_l
    es = 611.2_r8*exp(17.67_r8*tc/(tc + 243.5_r8))
    if (p - es <= 0._r8) then
       qs = 1._r8
    else
       qs = eps_l*es/(p - (1._r8 - eps_l)*es)
    end if
  end subroutine tetens_hard

  ! sat_mode 5, qsat only: CAM's svp_trans liquid->ice blend (wv_sat_methods,
  ! unmodified) over the legoESM curves (patch W1), then legoESM's smooth
  ! e->r conversion and q = r/(1+r) = legoesm.thermo.saturation_specific_humidity_trans.
  elemental subroutine lego_qsat_trans(t, p, es, qs)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    real(r8) :: w
    es = wv_sat_svp_trans(t, Lego_idx)
    w = eps_l*es/(softplus(p - es - 1._r8) + 1._r8)
    w = 1._r8 - softplus(20._r8*(1._r8 - w))/20._r8
    qs = w/(1._r8 + w)
  end subroutine lego_qsat_trans

  elemental real(r8) function estblf(t) result(es)
    real(r8), intent(in) :: t
    integer  :: i
    real(r8) :: t_tmp, weight
    t_tmp = max(min(t,tmax)-tmin, 0._r8)
    i = int(t_tmp) + 1
    weight = t_tmp - aint(t_tmp, r8)
    es = (1._r8 - weight)*estbl(i) + weight*estbl(i+1)
  end function estblf

  elemental subroutine qsat_water(t, p, es, qs)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    if (sat_mode == 1 .or. sat_mode == 5 .or. sat_mode == 6) then
       call lego_qsat(t, p, es, qs)
    else if (sat_mode == 3) then
       call tetens_hard(t, p, es, qs)
    else
       es = wv_sat_svp_water(t)
       qs = svp_to_qsat(es, p)
       es = min(es, p)
    end if
  end subroutine qsat_water

  elemental subroutine qsat(t, p, es, qs)
    real(r8), intent(in) :: t, p
    real(r8), intent(out) :: es, qs
    if (sat_mode == 1) then
       call lego_qsat(t, p, es, qs)
    else if (sat_mode == 5) then
       call lego_qsat_trans(t, p, es, qs)
    else if (sat_mode == 3) then
       call tetens_hard(t, p, es, qs)
    else if (sat_mode == 4) then
       ! split test: Goff-Gratch over WATER for qsat too (no ice blend)
       es = wv_sat_svp_water(t)
       qs = svp_to_qsat(es, p)
       es = min(es, p)
    else
       es = estblf(t)
       qs = svp_to_qsat(es, p)
       es = min(es, p)
    end if
  end subroutine qsat
end module wv_saturation
