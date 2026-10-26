! Standalone harness driver for WRF v4.8.0 FAST_SBM routines (module_mp_fast_sbm.F,
! commit 06d4240ae989cc3e50af412bb472df3d9048783c). Reads one task from stdin, writes
! results to stdout as one value per line (es25.17). CGS units throughout (oracle's).
program sbm_driver
  use module_mp_SBM_Collision, only: coll_xxx_lwf, coll_xyx_lwf, courant_bott_KS
  use module_mp_SBM_Auxiliary, only: POLYSVP
  use sbm_extract, only: ONECOND1, FREEZ, J_W_MELT, FALFLUXHUCM_Z, ecoalmass
  implicit none
  character(len=16) :: task
  integer :: nkr, i, j, k, ncond, ikl, nz, nsub, isym1, isym2(3), isym3, isym4, isym5, diffu
  real, allocatable :: xl(:), rlec(:), ro1bl(:), vr1(:), ff1(:), ff1in(:), ff2(:,:), xi(:,:), &
       ff3(:), ff4(:), ff5(:), rho_z(:), p_z(:), z_z(:), vr(:,:), chem(:,:)
  real(8), allocatable :: xl_mg(:), g(:), g2(:), ck(:,:), chucm(:,:), fl(:), fl2(:), dm_rime(:), &
       xld(:), drad(:), vr1d(:), ecm(:,:)
  integer, allocatable :: ima(:,:)
  real(8) :: tt, qq, pp, del1in, del2in, div1, div2, es1n, es2n, ew1n, dtnew
  real :: ror, dt, dtcond, colv, aa1, bb1, aa2, bb2, tin, ro, tfall
  real :: afreezmy, bfreezmy, bfreezmax
  integer :: krfreez

  read(*,*) task
  select case (trim(task))

  case ('COLL', 'TABLES')
    read(*,*) nkr
    allocate(xl(nkr), xl_mg(nkr), g(nkr), ck(nkr,nkr), chucm(nkr,nkr), ima(nkr,nkr), fl(nkr))
    read(*,*) (xl(i), i=1,nkr)
    read(*,*) ((ck(i,j), j=1,nkr), i=1,nkr)
    read(*,*) (g(i), i=1,nkr)
    chucm = 0.d0; ima = 0
    call courant_bott_KS(xl, nkr, chucm, ima, 1.0d0)
    xl_mg = dble(xl) * 1.0d3
    if (trim(task) == 'TABLES') then
      do i = 1, nkr
        do j = 1, nkr
          write(*,'(2i4,i4,es26.17e3)') i, j, ima(i,j), chucm(i,j)
        end do
      end do
    else
      fl = 1.0d0
      call coll_xxx_lwf(g, fl, ck, xl_mg, chucm, ima, 1.0d0, nkr)
      write(*,'(es26.17e3)') (g(i), i=1,nkr)
    end if

  case ('RIME')
    read(*,*) nkr
    allocate(xl(nkr), xl_mg(nkr), g(nkr), g2(nkr), ck(nkr,nkr), chucm(nkr,nkr), ima(nkr,nkr), &
             fl(nkr), fl2(nkr), dm_rime(nkr))
    read(*,*) (xl(i), i=1,nkr)
    read(*,*) ((ck(i,j), j=1,nkr), i=1,nkr)
    read(*,*) (g(i), i=1,nkr)      ! gx = collector (ice)
    read(*,*) (g2(i), i=1,nkr)     ! gy = collected (liquid)
    chucm = 0.d0; ima = 0
    call courant_bott_KS(xl, nkr, chucm, ima, 1.0d0)
    xl_mg = dble(xl) * 1.0d3
    fl = 0.0d0; fl2 = 1.0d0; dm_rime = 0.d0
    ! COAL_BOTT_NEW l.8462: call coll_xyx_lwf(g3,g1,rf3,rf1,cwsl,xs_mg,xl_mg,chucm,ima,prdkrn1,nkr,1,dm_rime)
    call coll_xyx_lwf(g, g2, fl, fl2, ck, xl_mg, xl_mg, chucm, ima, 1.0d0, nkr, 1, dm_rime)
    write(*,'(es26.17e3)') (g(i), i=1,nkr)
    write(*,'(es26.17e3)') (g2(i), i=1,nkr)

  case ('COND')
    ! FAST_SBM l.4576-4765 condensation loop (warm path only; nucleation not called).
    read(*,*) nkr, ncond
    allocate(xl(nkr), rlec(nkr), ro1bl(nkr), vr1(nkr), ff1(nkr), ff1in(nkr))
    read(*,*) dt, tt, qq, pp, ror, colv
    read(*,*) aa1, bb1, aa2, bb2
    read(*,*) (xl(i), i=1,nkr)
    read(*,*) (rlec(i), i=1,nkr)
    read(*,*) (ro1bl(i), i=1,nkr)
    read(*,*) (vr1(i), i=1,nkr)
    read(*,*) (ff1(i), i=1,nkr)
    dtcond = dt / real(ncond)
    dtnew = 0.0
    diffu = 1      ! HARNESS: DIFFU gate forced on (oracle sets 0 when dynamics left S unchanged)
    do ikl = 1, ncond
      dtcond = min(dt - real(dtnew), dtcond)
      dtnew = dtnew + dtcond
      es1n = aa1*dexp(-bb1/tt)
      es2n = aa2*dexp(-bb2/tt)
      ew1n = qq*pp/(0.622+0.378*qq)
      div1 = ew1n/es1n
      del1in = ew1n/es1n - 1.0
      div2 = ew1n/es2n
      del2in = ew1n/es2n - 1.0
      if (div1 > div2 .and. tt <= 265) diffu = 0
      if (diffu == 1) then
        ff1in = ff1
        isym1 = 0; isym2 = 0; isym3 = 0; isym4 = 0; isym5 = 0
        if (maxval(ff1) > 0) isym1 = 1
        if (isym1 == 1) then
          call ONECOND1(tt, qq, pp, ror, vr1, real(pp), del1in, del2in, div1, div2, &
                        ff1, ff1in, xl, rlec, ro1bl, aa1, bb1, aa2, bb2, 0.00033, 0.0, &
                        colv, dtcond, 3, nkr, isym1, isym2, isym3, isym4, isym5, 1, 1, 1, &
                        0.0, 1000.0, 1)
        end if
      end if
    end do
    write(*,'(es26.17e3)') tt
    write(*,'(es26.17e3)') qq
    write(*,'(es26.17e3)') del1in
    write(*,'(es26.17e3)') (ff1(i), i=1,nkr)

  case ('FREEZ')
    read(*,*) nkr
    allocate(xl(nkr), ff1(nkr), ff2(nkr,3), xi(nkr,3), ff3(nkr), ff4(nkr), ff5(nkr))
    read(*,*) dt, tin, ro, colv
    read(*,*) (xl(i), i=1,nkr)
    read(*,*) (ff1(i), i=1,nkr)
    ff2 = 0.; xi = 0.; ff3 = 0.; ff4 = 0.; ff5 = 0.
    ! FAST_SBM DATA l.4090-4099: KRFREEZ=21, BFREEZMAX=0.66, AFREEZMY=0.3333E-04, BFREEZMY=0.66
    krfreez = 21; bfreezmax = 0.66; afreezmy = 0.3333e-04; bfreezmy = 0.66
    call FREEZ(ff1, xl, ff2, xi, ff3, xl, ff4, xl, ff5, xl, tin, dt, ro, colv, &
               afreezmy, bfreezmy, bfreezmax, krfreez, 3, nkr)
    write(*,'(es26.17e3)') tin
    write(*,'(es26.17e3)') (ff1(i), i=1,nkr)
    write(*,'(es26.17e3)') (ff2(i,2), i=1,nkr)
    write(*,'(es26.17e3)') (ff5(i), i=1,nkr)

  case ('MELT')
    ! every ice category in: crystal habits 1-3 (FF2), snow FF3, graupel FF4, hail FF5
    read(*,*) nkr
    allocate(xl(nkr), ff1(nkr), ff2(nkr,3), xi(nkr,3), ff3(nkr), ff4(nkr), ff5(nkr))
    read(*,*) dt, tin, ro, colv
    read(*,*) (xl(i), i=1,nkr)
    read(*,*) ((ff2(i,k), i=1,nkr), k=1,3)
    read(*,*) (ff3(i), i=1,nkr)
    read(*,*) (ff4(i), i=1,nkr)
    read(*,*) (ff5(i), i=1,nkr)
    ff1 = 0.; xi = 0.
    call J_W_MELT(ff1, xl, ff2, xi, ff3, xl, ff4, xl, ff5, xl, tin, dt, ro, colv, 3, nkr)
    write(*,'(es26.17e3)') tin
    write(*,'(es26.17e3)') (ff1(i), i=1,nkr)
    write(*,'(es26.17e3)') ((ff2(i,k), i=1,nkr), k=1,3)
    write(*,'(es26.17e3)') (ff3(i), i=1,nkr)
    write(*,'(es26.17e3)') (ff4(i), i=1,nkr)
    write(*,'(es26.17e3)') (ff5(i), i=1,nkr)

  case ('FALL')
    read(*,*) nz, nkr
    allocate(rho_z(nz), p_z(nz), z_z(nz), vr(nz,nkr), chem(nz,nkr))
    read(*,*) dt
    read(*,*) (rho_z(k), k=1,nz)
    read(*,*) (p_z(k), k=1,nz)
    read(*,*) (z_z(k), k=1,nz)
    read(*,*) ((vr(k,i), i=1,nkr), k=1,nz)
    read(*,*) ((chem(k,i), i=1,nkr), k=1,nz)
    ! report NSUB per bin exactly as FALFLUXHUCM_Z computes it (l.5345-5352)
    do i = 1, nkr
      tfall = 1.e10
      do k = 1, nz
        tfall = amin1(tfall, z_z(k)/(vr(k,i)+1.e-20))
      end do
      nsub = int(2.0*dt/tfall) + 1
      write(*,'(i8)') nsub
    end do
    call FALFLUXHUCM_Z(chem, vr, rho_z, p_z, z_z, dt, 1, nz, nkr)
    write(*,'(es26.17e3)') ((chem(k,i), i=1,nkr), k=1,nz)

  case ('ECOAL')    ! not used by make_fixture.py (ECOALMASSM not ported)
    read(*,*) nkr
    allocate(xld(nkr), drad(nkr), vr1d(nkr), ecm(nkr,nkr))
    read(*,*) (xld(i), i=1,nkr)
    read(*,*) (drad(i), i=1,nkr)
    read(*,*) (vr1d(i), i=1,nkr)
    ! BREAKINIT_KS l.8754-8763
    do j = 1, nkr
      do i = 1, nkr
        ecm(i,j) = ecoalmass(xld(i), xld(j), drad, vr1d, nkr)
      end do
    end do
    do j = 25, 31
      ecm(nkr,j) = 0.1d-29
    end do
    write(*,'(es26.17e3)') ((ecm(i,j), j=1,nkr), i=1,nkr)

  case ('POLYSVP')
    read(*,*) tin
    write(*,'(es26.17e3)') POLYSVP(tin, 0)

  case default
    print *, 'unknown task ', task
    stop 2
  end select
end program sbm_driver
