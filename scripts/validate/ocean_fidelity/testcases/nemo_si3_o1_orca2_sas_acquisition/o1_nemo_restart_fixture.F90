PROGRAM o1_nemo_restart_fixture
   ! Halo-free single-rank restart following iom_nf90.F90:135-154,729-760:
   ! Fortran dimensions (x,y,numcat,time) become NetCDF (time,numcat,y,x).
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   USE netcdf
   IMPLICIT NONE
   INTEGER, PARAMETER :: nx=180, ny=148, nc=5, n3=34, n2=4
   CHARACTER(LEN=16), PARAMETER :: names3(n3) = [ CHARACTER(LEN=16) :: &
      'v_i','v_s','a_i','t_su','oa_i','a_ip','v_ip','v_il', &
      'sv_i','e_s_l01','e_s_l02','e_s_l03','e_s_l04','e_s_l05', &
      'e_i_l01','e_i_l02','e_i_l03','e_i_l04','e_i_l05','e_i_l06', &
      'e_i_l07','e_i_l08','e_i_l09','e_i_l10','szv_i_l01','szv_i_l02', &
      'szv_i_l03','szv_i_l04','szv_i_l05','szv_i_l06','szv_i_l07', &
      'szv_i_l08','szv_i_l09','szv_i_l10' ]
   CHARACTER(LEN=16), PARAMETER :: names2(n2) = [ CHARACTER(LEN=16) :: &
      'u_ice','v_ice','snwice_mass','snwice_mass_b' ]
   INTEGER :: ncid, dx, dy, dc, dt, scalar_dims(1), dims3(4), dims2(3)
   INTEGER :: ids3(n3), ids2(n2), id_fsbc, id_kt
   INTEGER :: i, j, k, n, status
   REAL(real64) :: a3(nx,ny,nc,1), a2(nx,ny,1), scalar(1)

   CALL ck(nf90_create('nemo_writer_layout_restart.nc', NF90_CLOBBER + NF90_64BIT_OFFSET, ncid))
   CALL ck(nf90_def_dim(ncid, 'x', nx, dx))
   CALL ck(nf90_def_dim(ncid, 'y', ny, dy))
   CALL ck(nf90_def_dim(ncid, 'numcat', nc, dc))
   CALL ck(nf90_def_dim(ncid, 'time_counter', NF90_UNLIMITED, dt))
   dims3 = [dx,dy,dc,dt]
   dims2 = [dx,dy,dt]
   scalar_dims = [dt]
   DO n=1,n3
      CALL ck(nf90_def_var(ncid, TRIM(names3(n)), NF90_DOUBLE, dims3, ids3(n)))
   END DO
   DO n=1,n2
      CALL ck(nf90_def_var(ncid, TRIM(names2(n)), NF90_DOUBLE, dims2, ids2(n)))
   END DO
   CALL ck(nf90_def_var(ncid, 'nn_fsbc', NF90_DOUBLE, scalar_dims, id_fsbc))
   CALL ck(nf90_def_var(ncid, 'kt_ice', NF90_DOUBLE, scalar_dims, id_kt))
   CALL ck(nf90_put_att(ncid, NF90_GLOBAL, 'DOMAIN_size_global', [nx,ny]))
   CALL ck(nf90_put_att(ncid, NF90_GLOBAL, 'DOMAIN_halo_size_start', [0,0]))
   CALL ck(nf90_put_att(ncid, NF90_GLOBAL, 'DOMAIN_halo_size_end', [0,0]))
   CALL ck(nf90_enddef(ncid))

   DO n=1,n3
      DO k=1,nc; DO j=1,ny; DO i=1,nx
         a3(i,j,k,1) = REAL(n,real64) + REAL(i,real64)*1.0E-3_real64 &
                       + REAL(j,real64)*1.0E-6_real64 + REAL(k,real64)*1.0E-9_real64
      END DO; END DO; END DO
      CALL ck(nf90_put_var(ncid, ids3(n), a3))
   END DO
   DO j=1,ny; DO i=1,nx
      a2(i,j,1) = REAL(i,real64)*1.0E-3_real64 + REAL(j,real64)*1.0E-6_real64
   END DO; END DO
   DO n=1,n2
      CALL ck(nf90_put_var(ncid, ids2(n), a2 + REAL(n,real64)))
   END DO
   scalar = 1.0_real64
   CALL ck(nf90_put_var(ncid, id_fsbc, scalar))
   CALL ck(nf90_put_var(ncid, id_kt, scalar))
   CALL ck(nf90_close(ncid))
CONTAINS
   SUBROUTINE ck(code)
      INTEGER, INTENT(IN) :: code
      IF (code /= nf90_noerr) THEN
         WRITE(*,'(A)') TRIM(nf90_strerror(code))
         ERROR STOP 1
      END IF
   END SUBROUTINE ck
END PROGRAM o1_nemo_restart_fixture
