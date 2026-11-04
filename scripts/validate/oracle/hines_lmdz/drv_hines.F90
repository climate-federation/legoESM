! Oracle driver: LMDZ r5891 hines_gwd (compiled unmodified) on columns read
! from argv(1); writes the tendencies d_u_hin/dtime, d_v_hin/dtime (LMDZ
! bottom-up order) and the top fluxes zustrhi/zvstrhi to argv(2).
! Input: "ncol nlev" / "dtime rd rg rcpd rpi" / per column: lat_deg,
! paprs(nlev+1), pplay(nlev), t, u, v (each one line, bottom-up).
PROGRAM drv_hines
  USE dimphy
  USE yomcst_mod_h
  IMPLICIT NONE
  INTEGER :: ncol, nlev, i
  REAL :: dtime
  REAL, ALLOCATABLE :: lat(:), paprs(:, :), pplay(:, :), t(:, :), u(:, :), v(:, :)
  REAL, ALLOCATABLE :: du(:, :), dv(:, :), dt(:, :), zus(:), zvs(:)
  CHARACTER(len=1024) :: fin, fout
  CALL get_command_argument(1, fin)
  CALL get_command_argument(2, fout)
  OPEN (10, file=trim(fin), status='old')
  READ (10, *) ncol, nlev
  READ (10, *) dtime, rd, rg, rcpd, rpi
  klon = ncol; klev = nlev; kidia = 1; kfdia = ncol
  ALLOCATE (lat(ncol), paprs(ncol, nlev+1), pplay(ncol, nlev), t(ncol, nlev), &
            u(ncol, nlev), v(ncol, nlev), du(ncol, nlev), dv(ncol, nlev), &
            dt(ncol, nlev), zus(ncol), zvs(ncol))
  DO i = 1, ncol
    READ (10, *) lat(i)
    READ (10, *) paprs(i, :)
    READ (10, *) pplay(i, :)
    READ (10, *) t(i, :)
    READ (10, *) u(i, :)
    READ (10, *) v(i, :)
  END DO
  CLOSE (10)
  CALL hines_gwd(ncol, nlev, dtime, paprs, pplay, lat, t, u, v, zus, zvs, dt, du, dv)
  OPEN (11, file=trim(fout), status='replace')
  DO i = 1, ncol
    WRITE (11, '(A,I4)') 'COL', i
    WRITE (11, '(*(ES25.17E3,1X))') du(i, :) / dtime
    WRITE (11, '(*(ES25.17E3,1X))') dv(i, :) / dtime
    WRITE (11, '(*(ES25.17E3,1X))') zus(i), zvs(i)
  END DO
  CLOSE (11)
  PRINT *, 'HINES_DRV_OK'
END PROGRAM drv_hines
