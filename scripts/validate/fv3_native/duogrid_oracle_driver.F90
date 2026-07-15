! Phase-3 duogrid halo oracle driver.
!
! Runs the VERBATIM global_grid_init (duo-grid variant global_grid.F90,
! mirror luanfs/FV3_container @ 7d06431e — the module legoESM's duogrid.py /
! cube_rmp lineage traces to) serially on the unit-radius... NOTE: the module
! hardcodes RADIUS = 6.3712e6 m (a variant quirk vs FMS 6371.0e3); dx/dy/da
! dumps carry that radius — consumers compare RATIOS or rescale.
!
! Dumps, for C{res} and gg%ng ghost layers, per tile n=1..6:
!   k2e_loc / k2e_coef            (A-grid rows/cols remap tables)
!   k2e_loc_b / k2e_coef_b        (B)
!   k2e_loc_c_x/c_y, coef         (C staggers)
!   k2e_loc_d_x/d_y, coef         (D staggers)
!   ext_x, ext_y, kik_x, kik_y    (extended vs kinked supergrid coords)
! as text records consumed by gen_duogrid_oracle.sh -> npz fixture.
!
! Build:
!   gfortran -O2 -fdefault-real-8 -c duogrid_shim.F90
!   gfortran -O2 -fdefault-real-8 -c <ref>/global_grid.F90
!   gfortran -O2 -fdefault-real-8 -o duogrid_oracle duogrid_oracle_driver.F90 *.o
program duogrid_oracle_driver
  use platform_mod, only: r8_kind
  use global_grid_mod
  implicit none

  type(global_grid_type) :: gg
  integer, parameter :: nres = 2
  integer, parameter :: res_list(nres) = (/ 12, 24 /)
  integer :: r, res, ng, n, i, j, k, u
  integer :: is, ie, isd, ied, sgis, sgie
  real(kind=r8_kind) :: tlon, tlat
  character(len=64) :: fname

  do r = 1, nres
    res = res_list(r)
    ng = 5              ! gen_k2e uses gg%ng-2 = 3 remap ghost rings
    tlon = 0.0_r8_kind
    tlat = 0.0_r8_kind

    gg%is_initialized = .false.
    call global_grid_init(gg, res, ng, 0, .false., tlon, tlat)

    is = 1;  ie = res
    isd = is - (ng - 2); ied = ie + (ng - 2)
    ! supergrid extended index space of ext_x/kik_x etc.
    sgis = 1 - 2*ng
    sgie = 2*res + 1 + 2*ng

    write(fname, '(A,I0,A)') 'duogrid_k2e_c', res, '.txt'
    open(newunit=u, file=trim(fname), status='replace', action='write')
    write(u,'(A)') '# duogrid k2e tables: rec stag n i j loc coef(1:k2e_nord)'
    write(u,'(A,I0,1X,I0,1X,I0)') '# res ng k2e_nord = ', res, ng, gg%k2e_nord
    do n = 1, 6
      do j = isd, ied
        do i = isd, ied
          if (gg%k2e_loc(i, j, n) /= -999) then
            write(u,'(A,1X,I1,1X,I5,1X,I5,1X,I5,1X,4ES26.17E3)') 'A', n, i, j, &
                 gg%k2e_loc(i, j, n), (gg%k2e_coef(k, i, j, n), k=1,gg%k2e_nord)
          end if
        end do
      end do
      do j = isd, ied + 1
        do i = isd, ied + 1
          if (gg%k2e_loc_b(i, j, n) /= -999) then
            write(u,'(A,1X,I1,1X,I5,1X,I5,1X,I5,1X,4ES26.17E3)') 'B', n, i, j, &
                 gg%k2e_loc_b(i, j, n), (gg%k2e_coef_b(k, i, j, n), k=1,gg%k2e_nord)
          end if
        end do
      end do
      do j = isd, ied
        do i = isd, ied + 1
          if (gg%k2e_loc_c_x(i, j, n) /= -999) then
            write(u,'(A,1X,I1,1X,I5,1X,I5,1X,I5,1X,4ES26.17E3)') 'CX', n, i, j, &
                 gg%k2e_loc_c_x(i, j, n), (gg%k2e_coef_c_x(k, i, j, n), k=1,gg%k2e_nord)
          end if
        end do
      end do
      do j = isd, ied + 1
        do i = isd, ied
          if (gg%k2e_loc_c_y(i, j, n) /= -999) then
            write(u,'(A,1X,I1,1X,I5,1X,I5,1X,I5,1X,4ES26.17E3)') 'CY', n, i, j, &
                 gg%k2e_loc_c_y(i, j, n), (gg%k2e_coef_c_y(k, i, j, n), k=1,gg%k2e_nord)
          end if
        end do
      end do
      do j = isd, ied
        do i = isd, ied + 1
          if (gg%k2e_loc_d_x(i, j, n) /= -999) then
            write(u,'(A,1X,I1,1X,I5,1X,I5,1X,I5,1X,4ES26.17E3)') 'DX', n, i, j, &
                 gg%k2e_loc_d_x(i, j, n), (gg%k2e_coef_d_x(k, i, j, n), k=1,gg%k2e_nord)
          end if
        end do
      end do
      do j = isd, ied + 1
        do i = isd, ied
          if (gg%k2e_loc_d_y(i, j, n) /= -999) then
            write(u,'(A,1X,I1,1X,I5,1X,I5,1X,I5,1X,4ES26.17E3)') 'DY', n, i, j, &
                 gg%k2e_loc_d_y(i, j, n), (gg%k2e_coef_d_y(k, i, j, n), k=1,gg%k2e_nord)
          end if
        end do
      end do
    end do
    close(u)

    write(fname, '(A,I0,A)') 'duogrid_coords_c', res, '.txt'
    open(newunit=u, file=trim(fname), status='replace', action='write')
    write(u,'(A)') '# duogrid ext/kik supergrid coords: n i j ext_x ext_y kik_x kik_y'
    write(u,'(A,I0,1X,I0)') '# res ng = ', res, ng
    do n = 1, 6
      do j = sgis, sgie
        do i = sgis, sgie
          write(u,'(I1,1X,I6,1X,I6,1X,4ES26.17E3)') n, i, j, &
               gg%ext_x(i, j, n), gg%ext_y(i, j, n), &
               gg%kik_x(i, j, n), gg%kik_y(i, j, n)
        end do
      end do
    end do
    close(u)

    call global_grid_end(gg)
  end do

  write(*,*) 'duogrid_oracle: wrote k2e + coord files for C12, C24'
end program duogrid_oracle_driver
