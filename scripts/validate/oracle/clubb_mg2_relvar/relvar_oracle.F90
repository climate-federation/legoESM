! CAM6 qclvar/relvar oracle: the oracle's own arithmetic, line for line.
!   rcm, rcp2: CLUBB pdf_closure_module.F90 (CLUBB_CESM vendor_clubb_r8099_n03
!     :1048-1050, :1072-1074; identical in larson-group/clubb_release
!     54403a04 :1013-1020, :1232-1239), on the Fortran pdf_params.
!   qclvar, relvar: CAM clubb_intr.F90 cam_cesm2_1_rel_60 :1998 and
!     :2416-2425 with deep_scheme = 'ZM' (CAM6), so relvarmax = 10.
! Input (stdin-free): file arg 1, first line n, then n rows of
!   mixt_frac chi_1 chi_2 rc_1 rc_2 cloud_frac_1 cloud_frac_2
!   stdev_chi_1 stdev_chi_2 rcm_host [qclvar_out]
! With arg 3 = "qclvar", each row carries an 11th column, CLUBB's own
! qclvar_out (rcp2_zt captured from the Fortran core), and relvar uses it
! instead of the transcribed rcp2.
! Output: file arg 2, n rows "rcp2 relvar".
program relvar_oracle
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  real(r8), parameter :: zero_threshold = 0.0_r8, one = 1.0_r8
  integer :: n, i
  real(r8) :: mixt_frac, chi_1, chi_2, rc_1, rc_2, cloud_frac_1, cloud_frac_2, &
              stdev_chi_1, stdev_chi_2, rcm_host, rcm, rcp2, qclvar, relvar, relvarmax, &
              qclvar_out
  character(len=512) :: fin, fout, mode
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, mode)
  open(10, file=trim(fin), status='old')
  open(11, file=trim(fout), status='replace')
  read(10, *) n
  do i = 1, n
     if (trim(mode) == 'qclvar') then
        read(10, *) mixt_frac, chi_1, chi_2, rc_1, rc_2, cloud_frac_1, cloud_frac_2, &
                    stdev_chi_1, stdev_chi_2, rcm_host, qclvar_out
     else
        read(10, *) mixt_frac, chi_1, chi_2, rc_1, rc_2, cloud_frac_1, cloud_frac_2, &
                    stdev_chi_1, stdev_chi_2, rcm_host
     end if
     ! pdf_closure_module.F90 r8099 :1048-1050
     rcm        = mixt_frac * rc_1         + (one-mixt_frac) * rc_2
     rcm = max( zero_threshold, rcm )
     ! pdf_closure_module.F90 r8099 :1072-1074
     rcp2 = mixt_frac * ( chi_1*rc_1 + cloud_frac_1*stdev_chi_1**2 ) &
            + ( one-mixt_frac ) * ( chi_2*rc_2 + cloud_frac_2*stdev_chi_2**2 ) - rcm**2
     rcp2 = max( zero_threshold, rcp2 )
     ! clubb_intr.F90 :1998 (qclvar_out = rcp2_zt, advance_clubb_core :1798)
     if (trim(mode) == 'qclvar') then
        qclvar = min(1._r8, qclvar_out)
     else
        qclvar = min(1._r8, rcp2)
     end if
     ! clubb_intr.F90 :2415-2425, deep_scheme /= 'CLUBB_SGS'
     relvarmax = 10.0_r8
     relvar = relvarmax  ! default
     if (rcm_host /= 0 .and. qclvar /= 0) &
         relvar = min(relvarmax,max(0.001_r8,rcm_host**2/qclvar))
     write(11, '(2es26.17e3)') rcp2, relvar
  end do
  close(10); close(11)
end program relvar_oracle
