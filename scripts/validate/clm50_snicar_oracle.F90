! CLM5.0 (CESM2.1.5, the CAM6 land model) SNICAR oracle.
!
! Compiles the UNMODIFIED CESM2.1.5 SnowSnicarMod.F90 (SNICAR_RT, SnowAge_grain,
! SnowOptics_init, SnowAge_init) against the minimal stub modules below, and
! drives one snow column through a prescribed temperature / liquid-water history:
! CLM5.0 grain growth evolves the effective radius, and SNICAR_RT returns the
! VIS/NIR snow albedo (direct at each requested cosz, and diffuse) for every
! aerosol case.  Optics file / aging table are the CLM5.0 namelist defaults
! (snicar_optics_5bnd_c090915.nc, snicar_drdt_bst_fit_60_c070416.nc).
!
! Build (netcdf-fortran):  see scripts/validate/probe_snow_albedo_snicar.py
!   (shr_assert.h is replaced by an empty-macro header: assertions only).
!
! stdin (list-directed):
!   optics_path
!   aging_path
!   dtime_s  nlay  init_forc_t_K  out_every_steps
!   swe(1:nlay)   [kg/m2, top first]
!   dz(1:nlay)    [m]
!   albsfc_vis albsfc_nir
!   ncase ; then per case: top-layer conc(8) and lower-layer conc(8) [kg/kg]
!           species order = CLM5.0 mss_cnc_aer: bcphi bcpho ocphi ocpho dst1-4
!   ncz ; cosz(1:ncz)
!   nseg ; per seg: nsteps liq_frac forc_t_K T_layer(1:nlay)_K T_soil_K
!          (layer-centre temperatures, top first; the caller builds them so that
!           CLM's own diagnosed top-layer gradient is the intended one)
! stdout, every out_every_steps and at step 0:
!   step  r_top  ( for case: (for cz: alb_dir_vis alb_dir_nir) alb_dif_vis alb_dif_nir )

module shr_kind_mod
  integer, parameter :: shr_kind_r8 = selected_real_kind(12)
end module shr_kind_mod
module shr_sys_mod
contains
  subroutine shr_sys_flush(u)
    integer, intent(in) :: u
  end subroutine shr_sys_flush
end module shr_sys_mod
module shr_log_mod
contains
  function shr_log_errMsg(f, l) result(s)
    character(len=*), intent(in) :: f
    integer, intent(in) :: l
    character(len=512) :: s
    write (s, '(a,":",i0)') trim(f), l
  end function shr_log_errMsg
end module shr_log_mod
module shr_const_mod
  use shr_kind_mod, only: r8 => shr_kind_r8
  real(r8), parameter :: SHR_CONST_PI = 3.14159265358979323846_r8
  real(r8), parameter :: SHR_CONST_RHOICE = 0.917e3_r8
end module shr_const_mod
module clm_varctl
  integer :: iulog = 6
  character(len=512) :: fsnowoptics = ' ', fsnowaging = ' '
end module clm_varctl
module clm_varcon
  use shr_kind_mod, only: r8 => shr_kind_r8
  character(len=16), parameter :: namec = 'column'
  real(r8), parameter :: tfrz = 273.15_r8
  real(r8), parameter :: spval = 1.e36_r8
end module clm_varcon
module clm_varpar
  integer, parameter :: nlevsno = 12, numrad = 2
end module clm_varpar
module abortutils
contains
  subroutine endrun(msg, decomp_index, clmlevel)
    character(len=*), intent(in), optional :: msg, clmlevel
    integer, intent(in), optional :: decomp_index
    if (present(msg)) write (0, *) 'ENDRUN: ', trim(msg)
    stop 2
  end subroutine endrun
end module abortutils
module decompMod
  type bounds_type
    integer :: begc = 1, endc = 1
  end type bounds_type
end module decompMod
module AerosolMod
  use shr_kind_mod, only: r8 => shr_kind_r8
  real(r8), public, parameter :: snw_rds_min = 54.526_r8   ! CESM2.1.5 AerosolMod
  real(r8), public :: fresh_snw_rds_max = 204.526_r8       ! clm5_0 namelist default
end module AerosolMod
module atm2lndType
  use shr_kind_mod, only: r8 => shr_kind_r8
  type atm2lnd_type
    real(r8), pointer :: forc_t_downscaled_col(:) => null()
  end type atm2lnd_type
end module atm2lndType
module WaterStateType
  use shr_kind_mod, only: r8 => shr_kind_r8
  type waterstate_type
    real(r8), pointer :: h2osno_col(:) => null(), frac_sno_eff_col(:) => null()
    real(r8), pointer :: h2osoi_liq_col(:,:) => null(), h2osoi_ice_col(:,:) => null()
    real(r8), pointer :: snw_rds_col(:,:) => null()
    real(r8), pointer :: snw_rds_top_col(:) => null(), sno_liq_top_col(:) => null()
  end type waterstate_type
end module WaterStateType
module WaterFluxType
  use shr_kind_mod, only: r8 => shr_kind_r8
  type waterflux_type
    real(r8), pointer :: qflx_snow_grnd_col(:) => null(), qflx_snofrz_lyr_col(:,:) => null()
  end type waterflux_type
end module WaterFluxType
module TemperatureType
  use shr_kind_mod, only: r8 => shr_kind_r8
  type temperature_type
    real(r8), pointer :: t_soisno_col(:,:) => null(), t_grnd_col(:) => null()
    real(r8), pointer :: snot_top_col(:) => null(), dTdz_top_col(:) => null()
  end type temperature_type
end module TemperatureType
module GridcellType
  use shr_kind_mod, only: r8 => shr_kind_r8
  type gc_t
    real(r8) :: latdeg(1) = 60._r8, londeg(1) = 0._r8
  end type gc_t
  type(gc_t) :: grc
end module GridcellType
module LandunitType
  type lu_t
    integer :: itype(1) = 1
  end type lu_t
  type(lu_t) :: lun
end module LandunitType
module ColumnType
  use shr_kind_mod, only: r8 => shr_kind_r8
  type col_t
    integer :: snl(1) = 0, landunit(1) = 1, gridcell(1) = 1
    real(r8) :: dz(1, -11:1) = 0._r8
  end type col_t
  type(col_t) :: col
end module ColumnType
module clm_time_manager
  use shr_kind_mod, only: r8 => shr_kind_r8
  integer :: nstep_g = 0
  real(r8) :: dtime_g = 1800._r8
contains
  integer function get_nstep()
    get_nstep = nstep_g
  end function get_nstep
  integer function get_step_size()
    get_step_size = nint(dtime_g)
  end function get_step_size
end module clm_time_manager
module fileutils
contains
  subroutine getfil(f, l, i)
    character(len=*), intent(in) :: f
    character(len=*), intent(out) :: l
    integer, intent(in) :: i
    l = f
  end subroutine getfil
end module fileutils
module spmdMod
  logical :: masterproc = .false.
end module spmdMod
module ncdio_pio
  use netcdf
  use shr_kind_mod, only: r8 => shr_kind_r8
  type file_desc_t
    integer :: id = -1
  end type file_desc_t
  interface ncd_io
    module procedure io1, io2, io3
  end interface
contains
  subroutine chk(s, what)
    integer, intent(in) :: s
    character(len=*), intent(in) :: what
    if (s /= nf90_noerr) then
      write (0, *) 'netcdf: ', trim(what), ': ', trim(nf90_strerror(s))
      stop 3
    end if
  end subroutine chk
  subroutine ncd_pio_openfile(f, fname, mode)
    type(file_desc_t), intent(inout) :: f
    character(len=*), intent(in) :: fname
    integer, intent(in) :: mode
    call chk(nf90_open(trim(fname), nf90_nowrite, f%id), fname)
  end subroutine ncd_pio_openfile
  subroutine ncd_pio_closefile(f)
    type(file_desc_t), intent(inout) :: f
    call chk(nf90_close(f%id), 'close')
  end subroutine ncd_pio_closefile
  subroutine io1(v, d, flag, f, posNOTonfile)
    character(len=*), intent(in) :: v, flag
    real(r8), intent(inout) :: d(:)
    type(file_desc_t), intent(in) :: f
    logical, intent(in), optional :: posNOTonfile
    integer :: vid
    call chk(nf90_inq_varid(f%id, v, vid), v)
    call chk(nf90_get_var(f%id, vid, d), v)
  end subroutine io1
  subroutine io2(v, d, flag, f, posNOTonfile)
    character(len=*), intent(in) :: v, flag
    real(r8), intent(inout) :: d(:,:)
    type(file_desc_t), intent(in) :: f
    logical, intent(in), optional :: posNOTonfile
    integer :: vid
    call chk(nf90_inq_varid(f%id, v, vid), v)
    call chk(nf90_get_var(f%id, vid, d), v)
  end subroutine io2
  subroutine io3(v, d, flag, f, posNOTonfile)
    character(len=*), intent(in) :: v, flag
    real(r8), intent(inout) :: d(:,:,:)
    type(file_desc_t), intent(in) :: f
    logical, intent(in), optional :: posNOTonfile
    integer :: vid
    call chk(nf90_inq_varid(f%id, v, vid), v)
    call chk(nf90_get_var(f%id, vid, d), v)
  end subroutine io3
end module ncdio_pio

program clm50_snicar_oracle
  use shr_kind_mod, only: r8 => shr_kind_r8
  use clm_varctl, only: fsnowoptics, fsnowaging
  use clm_varpar, only: nlevsno, numrad
  use clm_time_manager, only: nstep_g, dtime_g
  use decompMod, only: bounds_type
  use ColumnType, only: col
  use atm2lndType
  use WaterStateType
  use WaterFluxType
  use TemperatureType
  use SnowSnicarMod
  implicit none
  integer, parameter :: c = 1, nsoil = 1
  type(bounds_type) :: b
  type(atm2lnd_type) :: a2l
  type(waterstate_type) :: ws
  type(waterflux_type) :: wf
  type(temperature_type) :: ts
  integer :: nlay, ncase, ncz, nseg, iseg, nst, k, ic, iz, j, outev, step, top
  real(r8) :: dtime, forc_t0, lf, ft
  real(r8), allocatable :: tl(:)
  real(r8), allocatable :: swe(:), dzl(:), conc(:,:,:), cz(:)
  real(r8) :: albsfc(1, numrad), albout(1, numrad), flx(1, -nlevsno+1:1, numrad)
  real(r8) :: aer(1, -nlevsno+1:0, sno_nbr_aer), czc(1)
  integer :: rds(1, -nlevsno+1:0)
  integer :: filt(1) = (/1/)
  character(len=512) :: line

  read (*, '(a)') line; fsnowoptics = adjustl(line)
  read (*, '(a)') line; fsnowaging = adjustl(line)
  read (*, *) dtime, nlay, forc_t0, outev
  allocate (swe(nlay), dzl(nlay), tl(nlay + 1))
  read (*, *) swe
  read (*, *) dzl
  read (*, *) albsfc(1, :)
  read (*, *) ncase
  allocate (conc(ncase, 2, sno_nbr_aer))
  do ic = 1, ncase
    read (*, *) conc(ic, 1, :), conc(ic, 2, :)
  end do
  read (*, *) ncz
  allocate (cz(ncz))
  read (*, *) cz
  dtime_g = dtime

  call SnowOptics_init()
  call SnowAge_init()

  allocate (a2l%forc_t_downscaled_col(1))
  allocate (ws%h2osno_col(1), ws%frac_sno_eff_col(1), ws%snw_rds_top_col(1), ws%sno_liq_top_col(1))
  allocate (ws%h2osoi_liq_col(1, -nlevsno+1:nsoil), ws%h2osoi_ice_col(1, -nlevsno+1:nsoil))
  allocate (ws%snw_rds_col(1, -nlevsno+1:0))
  allocate (wf%qflx_snow_grnd_col(1), wf%qflx_snofrz_lyr_col(1, -nlevsno+1:0))
  allocate (ts%t_soisno_col(1, -nlevsno+1:nsoil), ts%t_grnd_col(1), ts%snot_top_col(1), ts%dTdz_top_col(1))

  top = 1 - nlay                         ! CLM index of top snow layer (snl+1)
  col%snl(1) = -nlay
  col%dz(1, :) = 0._r8
  col%dz(1, nsoil) = 0.1_r8
  ws%h2osoi_liq_col = 0._r8; ws%h2osoi_ice_col = 0._r8
  do j = 1, nlay
    col%dz(1, top + j - 1) = dzl(j)
  end do
  ws%h2osno_col(1) = sum(swe)
  ws%frac_sno_eff_col(1) = 1._r8
  wf%qflx_snow_grnd_col = 0._r8; wf%qflx_snofrz_lyr_col = 0._r8
  a2l%forc_t_downscaled_col(1) = forc_t0
  ws%snw_rds_col = 0._r8
  ws%snw_rds_col(1, top:0) = FreshSnowRadius(1, a2l)

  step = 0
  read (*, *) nseg
  tl = forc_t0
  call set_state(0._r8)
  call emit()
  do iseg = 1, nseg
    read (*, *) nst, lf, ft, tl
    a2l%forc_t_downscaled_col(1) = ft
    call set_state(lf)
    do k = 1, nst
      step = step + 1
      nstep_g = step
      call SnowAge_grain(b, 1, filt, 0, filt, wf, ws, ts, a2l)
      if (mod(step, outev) == 0) call emit()
    end do
  end do

contains

  subroutine set_state(liq)
    real(r8), intent(in) :: liq
    integer :: jj
    do jj = 1, nlay
      ts%t_soisno_col(1, top + jj - 1) = tl(jj)
      ws%h2osoi_liq_col(1, top + jj - 1) = liq * swe(jj)
      ws%h2osoi_ice_col(1, top + jj - 1) = (1._r8 - liq) * swe(jj)
    end do
    ts%t_soisno_col(1, nsoil) = tl(nlay + 1)
    ts%t_grnd_col(1) = ts%t_soisno_col(1, top)
  end subroutine set_state

  subroutine emit()
    real(r8) :: o(4 * ncase + 2 * ncz * ncase)
    integer :: n, jj
    rds = 0
    rds(1, top:0) = nint(ws%snw_rds_col(1, top:0))
    n = 0
    do ic = 1, ncase
      aer = 0._r8
      aer(1, top, :) = conc(ic, 1, :)
      do jj = top + 1, 0
        aer(1, jj, :) = conc(ic, 2, :)
      end do
      do iz = 1, ncz
        czc(1) = cz(iz)
        call SNICAR_RT(1, b, 1, filt, czc, 1, ws%h2osoi_liq_col(:, -nlevsno+1:0), &
             ws%h2osoi_ice_col(:, -nlevsno+1:0), rds, aer, albsfc, albout, flx, ws)
        o(n + 1:n + 2) = albout(1, :); n = n + 2
      end do
      czc(1) = 0.5_r8
      call SNICAR_RT(1, b, 1, filt, czc, 2, ws%h2osoi_liq_col(:, -nlevsno+1:0), &
           ws%h2osoi_ice_col(:, -nlevsno+1:0), rds, aer, albsfc, albout, flx, ws)
      o(n + 1:n + 2) = albout(1, :); n = n + 2
    end do
    write (*, '(i8,1x,f10.3,*(1x,f10.6))') step, ws%snw_rds_col(1, top), o(1:n)
  end subroutine emit

end program clm50_snicar_oracle
