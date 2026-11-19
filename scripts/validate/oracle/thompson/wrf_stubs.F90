! Minimal stand-ins for the WRF framework symbols that
! phys/module_mp_thompson.F and phys/module_mp_radar.F (WRF v4.8.0) reference.
! None of them changes the microphysics: they only provide messages, the
! three background effective radii, the timing hooks and the namelist
! switches of the table routines (never force-read; write only on request,
! see nl_get_write_thompson_tables).  Non-MPI build (no DM_PARALLEL), as a single-task WRF.
module module_wrf_error
  implicit none
  character(len=256) :: wrf_err_message
contains
  subroutine wrf_message(str)
    character(len=*), intent(in) :: str
    write(0, '(a)') trim(str)
  end subroutine wrf_message
end module module_wrf_error

module module_model_constants
  implicit none
  ! WRF v4.8.0 share/module_model_constants.F:62-64
  real, parameter :: RE_QC_BG = 2.49E-6
  real, parameter :: RE_QI_BG = 4.99E-6
  real, parameter :: RE_QS_BG = 9.99E-6
end module module_model_constants

module module_timing
  implicit none
contains
  subroutine start_timing
  end subroutine start_timing
  subroutine end_timing(str)
    character(len=*), intent(in) :: str
    write(0, '(a)') 'timing: '//trim(str)
  end subroutine end_timing
end module module_timing

subroutine wrf_debug(level, str)
  integer, intent(in) :: level
  character(len=*), intent(in) :: str
end subroutine wrf_debug

subroutine wrf_error_fatal(str)
  character(len=*), intent(in) :: str
  write(0, '(a)') 'FATAL: '//trim(str)
  stop 1
end subroutine wrf_error_fatal

subroutine nl_get_force_read_thompson(id, val)
  integer, intent(in) :: id
  logical, intent(out) :: val
  val = .false.
end subroutine nl_get_force_read_thompson

subroutine nl_get_write_thompson_tables(id, val)
  ! WRF's own table cache: with THOMPSON_WRITE_TABLES=1 the computed tables
  ! are written to qr_acr_qg_V4.dat / qr_acr_qsV2.dat / freezeH2O.dat in the
  ! working directory, and any later run there READS them (module_mp_thompson
  ! reads an existing file instead of recomputing; same doubles, unformatted).
  integer, intent(in) :: id
  logical, intent(out) :: val
  character(len=8) :: env
  call get_environment_variable('THOMPSON_WRITE_TABLES', env)
  val = (trim(env) == '1')
end subroutine nl_get_write_thompson_tables

subroutine nl_get_write_thompson_mp38table(id, val)
  integer, intent(in) :: id
  logical, intent(out) :: val
  val = .false.
end subroutine nl_get_write_thompson_mp38table

logical function wrf_dm_on_monitor()
  wrf_dm_on_monitor = .true.
end function wrf_dm_on_monitor

! table_ccnAct (aerosol-aware runs only, never called with mp_physics=8)
! USEs these two framework modules; nothing from them is referenced.
module module_domain
end module module_domain

module module_dm
end module module_dm
