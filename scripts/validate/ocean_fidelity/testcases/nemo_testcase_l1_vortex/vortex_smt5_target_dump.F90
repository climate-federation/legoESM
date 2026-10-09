MODULE vortex_smt5_target_dump
   !!======================================================================
   !!  SMT-5 (Decision 107e): NEMO writes its OWN T/S damping inputs.
   !!
   !!  Called once from usr_def_istate after the analytical T and S are set.
   !!  Writes, in the run directory, the three files the SMT-5 deck reads:
   !!    data_1m_potential_temperature_nomask.nc  votemper(time_counter,z,y,x)
   !!    data_1m_salinity_nomask.nc               vosaline(time_counter,z,y,x)
   !!       12 identical monthly records of the analytical initial state
   !!       (Decision 107a,d); names, dimension names and axis order of
   !!       ORCA2 rung 1's sn_tem/sn_sal files; NF90_DOUBLE so the target is
   !!       the double-precision field itself, not a float32 rounding of it.
   !!    resto.nc                                 resto(z,y,x), NF90_DOUBLE
   !!       ptmask * (1/86400) s-1: uniform 1-day restoring on every wet cell,
   !!       0 on land (Decision 107b), ORCA2 resto.nc's layout.
   !!  Interior points only (Nis0:Nie0, Njs0:Nje0), all jpk levels: the global
   !!  domain fld_read/iom_get read with jpdom_global.  Single rank only.
   !!  NF90_NOCLOBBER: an existing file is a refusal, never silently replaced.
   !!  The model state is only READ here; nothing NEMO computes is changed.
   !!======================================================================
   USE par_oce
   USE in_out_manager
   USE lib_mpp
   USE netcdf
   IMPLICIT NONE
   PRIVATE
   PUBLIC   vortex_smt5_dump_target

CONTAINS

   SUBROUTINE vortex_smt5_dump_target( ptmask, pts )
      REAL(wp), DIMENSION(jpi,jpj,jpk)     , INTENT(in) ::   ptmask
      REAL(wp), DIMENSION(jpi,jpj,jpk,jpts), INTENT(in) ::   pts
      REAL(wp), DIMENSION(Ni_0,Nj_0,jpk) ::   zfld
      !
      IF( jpnij /= 1 )   CALL ctl_stop( 'STOP', 'vortex_smt5_dump_target: single-rank run only' )
      zfld(:,:,:) = pts(Nis0:Nie0,Njs0:Nje0,:,jp_tem)
      CALL smt5_write_records( 'data_1m_potential_temperature_nomask.nc', 'votemper', zfld )
      zfld(:,:,:) = pts(Nis0:Nie0,Njs0:Nje0,:,jp_sal)
      CALL smt5_write_records( 'data_1m_salinity_nomask.nc', 'vosaline', zfld )
      zfld(:,:,:) = ptmask(Nis0:Nie0,Njs0:Nje0,:) * ( 1._wp / 86400._wp )
      CALL smt5_write_resto( 'resto.nc', zfld )
      IF(lwp) WRITE(numout,*) 'vortex_smt5_dump_target: SMT5_TARGET_DUMP_WRITTEN 3 files'
   END SUBROUTINE vortex_smt5_dump_target


   SUBROUTINE smt5_write_records( cdfile, cdvar, pfld )
      CHARACTER(len=*)                  , INTENT(in) ::   cdfile, cdvar
      REAL(wp), DIMENSION(Ni_0,Nj_0,jpk), INTENT(in) ::   pfld
      INTEGER ::   incid, idx, idy, idz, idt, idtv, idv, jt
      !
      CALL smt5_chk( nf90_create( cdfile, NF90_NOCLOBBER, incid ), cdfile )
      CALL smt5_chk( nf90_def_dim( incid, 'x', Ni_0, idx ), cdfile )
      CALL smt5_chk( nf90_def_dim( incid, 'y', Nj_0, idy ), cdfile )
      CALL smt5_chk( nf90_def_dim( incid, 'z', jpk , idz ), cdfile )
      CALL smt5_chk( nf90_def_dim( incid, 'time_counter', NF90_UNLIMITED, idt ), cdfile )
      CALL smt5_chk( nf90_def_var( incid, 'time_counter', NF90_DOUBLE, (/ idt /), idtv ), cdfile )
      CALL smt5_chk( nf90_put_att( incid, idtv, 'units', 'month_number' ), cdfile )
      CALL smt5_chk( nf90_def_var( incid, cdvar, NF90_DOUBLE, (/ idx, idy, idz, idt /), idv ), cdfile )
      CALL smt5_chk( nf90_enddef( incid ), cdfile )
      DO jt = 1, 12
         CALL smt5_chk( nf90_put_var( incid, idtv, REAL(jt,wp), start=(/ jt /) ), cdfile )
         CALL smt5_chk( nf90_put_var( incid, idv, pfld, start=(/ 1, 1, 1, jt /),   &
            &                         count=(/ Ni_0, Nj_0, jpk, 1 /) ), cdfile )
      END DO
      CALL smt5_chk( nf90_close( incid ), cdfile )
   END SUBROUTINE smt5_write_records


   SUBROUTINE smt5_write_resto( cdfile, pfld )
      CHARACTER(len=*)                  , INTENT(in) ::   cdfile
      REAL(wp), DIMENSION(Ni_0,Nj_0,jpk), INTENT(in) ::   pfld
      INTEGER ::   incid, idx, idy, idz, idv
      !
      CALL smt5_chk( nf90_create( cdfile, NF90_NOCLOBBER, incid ), cdfile )
      CALL smt5_chk( nf90_def_dim( incid, 'x', Ni_0, idx ), cdfile )
      CALL smt5_chk( nf90_def_dim( incid, 'y', Nj_0, idy ), cdfile )
      CALL smt5_chk( nf90_def_dim( incid, 'z', jpk , idz ), cdfile )
      CALL smt5_chk( nf90_def_var( incid, 'resto', NF90_DOUBLE, (/ idx, idy, idz /), idv ), cdfile )
      CALL smt5_chk( nf90_enddef( incid ), cdfile )
      CALL smt5_chk( nf90_put_var( incid, idv, pfld ), cdfile )
      CALL smt5_chk( nf90_close( incid ), cdfile )
   END SUBROUTINE smt5_write_resto


   SUBROUTINE smt5_chk( kstatus, cdfile )
      INTEGER         , INTENT(in) ::   kstatus
      CHARACTER(len=*), INTENT(in) ::   cdfile
      IF( kstatus /= NF90_NOERR )   CALL ctl_stop( 'STOP', 'vortex_smt5_dump_target: '//TRIM(cdfile)   &
         &                                          //': '//TRIM(nf90_strerror(kstatus)) )
   END SUBROUTINE smt5_chk

END MODULE vortex_smt5_target_dump
