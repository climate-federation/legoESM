! CAM6 (cam_cesm2_1_rel_60, SHA a03b84b7) physics SEQUENCING oracle, standalone.
!
! What is verbatim from the oracle (line refs into /burg/glab/users/pg2328/oracles/CAM):
!   * physics_update arithmetic: src/physics/cam/physics_types.F90:304-347 (u, v, q + qneg3 /
!     number clamp), 372-380 (cldlim zeroing after 'convect_deep'), 411-417 (T += s*dt/cpair).
!     geopotential_t (:421-431) is a NO-OP stub here: it only refreshes s/zm/zi, which no stub
!     scheme reads.
!   * qneg3 numeric core: src/physics/cam/qneg_module.F90:275-295 (q<qmin -> qmin; logging dropped).
!   * physics_ptend_scale: physics_types.F90:824-883 (multiply every active tendency by fac).
!   * tphysbc control flow: physpkg.F90:1918-1925 (convect_deep_tend + physics_update(ztodt)),
!     2033-2189 (macmic loop: clubb_tend_cam(cld_macmic_ztodt) -> ptend_scale(1/N) ->
!     physics_update(ztodt); microp_driver_tend(cld_macmic_ztodt) -> ptend_scale(1/N) ->
!     physics_update(ztodt); prec_str averaged over sub-steps), 2272-2279 (radiation_tend +
!     physics_update);  tphysac physpkg.F90:1496-1500 (gw_tend + physics_update).
!   * clubb_intr detrainment block: src/physics/cam/clubb_intr.F90:2343-2376 (zmconv_microp=F
!     branch; number concentrations dropped, no number tracers here), evaluated on state1 =
!     state after the CLUBB update (clubb_intr.F90:2251), all inside the sub-step ptend.
!   * held radiation: radiation.F90:1224-1233,1253-1258 (Q*dp stored, /pdel on held steps;
!     pdel is constant in this no-dynamics harness, so identity).
! Schemes are analytic STUBS identical to ../stubs.py (unit under test = the glue).
! usage: oracle_seq <input> <output> rad_post gw_post dlf_clubb qneg_on  (1 = CAM, 0 = old legoESM
! order).  Harness of tests/atmosphere/hydrostatic/unit/test_cam6_sequencing_oracle.py.
module seq_mod
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: pcnst = 3, ixq = 1, ixcldliq = 2, ixcldice = 3
  integer :: pcols, pver, nmac
  real(r8) :: cpair, latvap, latice, gravit, ztodt
  real(r8) :: qmin(pcnst) = (/1.e-12_r8, 0._r8, 0._r8/)
  real(r8), allocatable :: pm(:), dp(:), q0ref(:,:)
  logical :: qneg_on
  type ptend_t
    character(len=24) :: name
    logical :: ls, lu, lv, lq(pcnst)
    real(r8), allocatable :: s(:,:), u(:,:), v(:,:), q(:,:,:)
  end type
  type state_t
    real(r8), allocatable :: t(:,:), u(:,:), v(:,:), q(:,:,:)
  end type
contains
  subroutine ptend_init(p, name)
    type(ptend_t), intent(out) :: p
    character(len=*), intent(in) :: name
    p%name = name; p%ls = .false.; p%lu = .false.; p%lv = .false.; p%lq = .false.
    allocate(p%s(pcols,pver), p%u(pcols,pver), p%v(pcols,pver), p%q(pcols,pver,pcnst))
    p%s = 0._r8; p%u = 0._r8; p%v = 0._r8; p%q = 0._r8
  end subroutine

  subroutine qneg3(ncol, lver, m, q)          ! qneg_module.F90:275-295 core
    integer, intent(in) :: ncol, lver, m
    real(r8), intent(inout) :: q(:,:)
    integer :: i, k
    do k = 1, lver
      do i = 1, ncol
        if (q(i,k) < qmin(m)) then
          q(i,k) = qmin(m)
        end if
      end do
    end do
  end subroutine

  subroutine physics_update(state, ptend, dt)  ! physics_types.F90:304-347,372-380,411-417
    type(state_t), intent(inout) :: state
    type(ptend_t), intent(inout) :: ptend
    real(r8), intent(in) :: dt
    integer :: k, m, ncol
    ncol = pcols
    if (ptend%lu) then
      do k = 1, pver
        state%u(:ncol,k) = state%u(:ncol,k) + ptend%u(:ncol,k) * dt
      end do
    end if
    if (ptend%lv) then
      do k = 1, pver
        state%v(:ncol,k) = state%v(:ncol,k) + ptend%v(:ncol,k) * dt
      end do
    end if
    do m = 1, pcnst
      if (ptend%lq(m)) then
        do k = 1, pver
          state%q(:ncol,k,m) = state%q(:ncol,k,m) + ptend%q(:ncol,k,m) * dt
        end do
        if (qneg_on) call qneg3(ncol, pver, m, state%q(:,:,m))
      end if
    end do
    if (qneg_on .and. ptend%lq(ixcldliq) .and. (ptend%name == 'convect_deep' .or. ptend%name == 'zm_conv_tend')) then
      where (state%q(:ncol,:,ixcldliq) < 1.e-36_r8) state%q(:ncol,:,ixcldliq) = 0._r8
    end if
    if (qneg_on .and. ptend%lq(ixcldice) .and. (ptend%name == 'convect_deep' .or. ptend%name == 'zm_conv_tend')) then
      where (state%q(:ncol,:,ixcldice) < 1.e-36_r8) state%q(:ncol,:,ixcldice) = 0._r8
    end if
    if (ptend%ls) then
      do k = 1, pver
        state%t(:ncol,k) = state%t(:ncol,k) + ptend%s(:ncol,k)*dt/cpair
      end do
    end if
    ! geopotential_t: no-op stub (s/zm/zi unused by the stubs)
    ptend%ls = .false.; ptend%lu = .false.; ptend%lv = .false.; ptend%lq = .false.
  end subroutine

  subroutine ptend_scale(p, fac)               ! physics_types.F90:824-883
    type(ptend_t), intent(inout) :: p
    real(r8), intent(in) :: fac
    integer :: m
    if (p%lu) p%u = p%u * fac
    if (p%lv) p%v = p%v * fac
    if (p%ls) p%s = p%s * fac
    do m = 1, pcnst
      if (p%lq(m)) p%q(:,:,m) = p%q(:,:,m) * fac
    end do
  end subroutine

  subroutine copy_state(a, b)
    type(state_t), intent(in) :: a
    type(state_t), intent(out) :: b
    allocate(b%t, source=a%t); allocate(b%u, source=a%u)
    allocate(b%v, source=a%v); allocate(b%q, source=a%q)
  end subroutine

  ! ---------------- stubs (identical to ../stubs.py) ----------------
  subroutine stub_conv(st, p, dlf, keep_dlf_in_ptend)
    type(state_t), intent(in) :: st
    type(ptend_t), intent(out) :: p
    real(r8), intent(out) :: dlf(:,:)
    logical, intent(in) :: keep_dlf_in_ptend
    real(r8) :: w(pver), dT(pcols,pver)
    integer :: i
    call ptend_init(p, 'convect_deep')
    w = (pm / 1.0e5_r8) ** 4
    do i = 1, pcols
      dT(i,:) = 0.02_r8 * st%q(i,:,ixq) * w
      dlf(i,:) = 0.2_r8 * cpair / latvap * dT(i,:)
      p%q(i,:,ixq) = -cpair / latvap * dT(i,:) - dlf(i,:)
      p%u(i,:) = -1.0e-5_r8 * st%u(i,:) * w
    end do
    p%s = cpair * dT
    p%ls = .true.; p%lu = .true.; p%lq(ixq) = .true.
    if (keep_dlf_in_ptend) then
      p%q(:,:,ixcldliq) = dlf; p%lq(ixcldliq) = .true.
    end if
  end subroutine

  subroutine stub_turb(st, dt, p)
    type(state_t), intent(in) :: st
    real(r8), intent(in) :: dt
    type(ptend_t), intent(out) :: p
    integer, parameter :: NB = 10
    real(r8) :: fac, xb, sdp
    integer :: i, k0
    call ptend_init(p, 'clubb')
    fac = (1.0_r8 - exp(-dt / 3600.0_r8)) / dt
    k0 = pver - NB + 1
    sdp = sum(dp(k0:pver))
    do i = 1, pcols
      xb = sum(st%t(i,k0:pver) * dp(k0:pver)) / sdp
      p%s(i,k0:pver) = (xb - st%t(i,k0:pver)) * fac
      xb = sum(st%q(i,k0:pver,ixq) * dp(k0:pver)) / sdp
      p%q(i,k0:pver,ixq) = (xb - st%q(i,k0:pver,ixq)) * fac
      xb = sum(st%q(i,k0:pver,ixcldliq) * dp(k0:pver)) / sdp
      p%q(i,k0:pver,ixcldliq) = (xb - st%q(i,k0:pver,ixcldliq)) * fac
      xb = sum(st%u(i,k0:pver) * dp(k0:pver)) / sdp
      p%u(i,k0:pver) = (xb - st%u(i,k0:pver)) * fac
      p%s(i,pver) = p%s(i,pver) + 20.0_r8 * gravit / (cpair * dp(pver))
      p%q(i,pver,ixq) = p%q(i,pver,ixq) + 100.0_r8 * gravit / (latvap * dp(pver))
    end do
    p%s = cpair * p%s                          ! s-tendency, J/kg/s
    p%ls = .true.; p%lu = .true.; p%lq(ixq) = .true.; p%lq(ixcldliq) = .true.
  end subroutine

  subroutine stub_micro(st, dt, p, precip)
    type(state_t), intent(in) :: st
    real(r8), intent(in) :: dt
    type(ptend_t), intent(out) :: p
    real(r8), intent(out) :: precip(:)
    real(r8) :: cond(pver), auto(pver), dep(pver), b(pver)
    integer :: i
    call ptend_init(p, 'micro_mg')
    b = 0._r8; b(21:25) = 2.0e-10_r8
    do i = 1, pcols
      cond = max((st%q(i,:,ixq) - q0ref(i,:)) / 1800.0_r8, -max(st%q(i,:,ixcldliq), 0._r8) / dt)
      auto = st%q(i,:,ixcldliq) * (1.0_r8 - exp(-dt / 1200.0_r8)) / dt
      dep = st%q(i,:,ixcldice) * (1.0_r8 - exp(-dt / 3000.0_r8)) / dt
      p%q(i,:,ixq) = -cond
      p%q(i,:,ixcldliq) = cond - auto - b
      p%q(i,:,ixcldice) = -dep
      p%s(i,:) = cpair * (latvap * cond / cpair)
      precip(i) = sum((auto + b + dep) * dp) / gravit
    end do
    p%ls = .true.; p%lq = .true.
  end subroutine

  subroutine stub_rad(st, qrs)
    type(state_t), intent(in) :: st
    real(r8), intent(out) :: qrs(:,:)          ! s-tendency J/kg/s
    qrs = cpair * (-(st%t - 180.0_r8) / (40.0_r8 * 86400.0_r8) &
          + 1.0e-5_r8 * (st%q(:,:,ixcldliq) + st%q(:,:,ixcldice)) / 1.0e-4_r8)
  end subroutine

  subroutine stub_gw(st, p)
    type(state_t), intent(in) :: st
    type(ptend_t), intent(out) :: p
    call ptend_init(p, 'gw')
    p%u(:,1:8) = -st%u(:,1:8) * (st%t(:,1:8) - 150.0_r8) * 1.0e-8_r8
    p%s(:,1:8) = cpair * (-st%u(:,1:8) * p%u(:,1:8) / cpair)
    p%ls = .true.; p%lu = .true.
  end subroutine
end module seq_mod

program oracle_seq
  use seq_mod
  implicit none
  type(state_t) :: state, state0, state1
  type(ptend_t) :: ptend, ptend_gw_pre
  real(r8), allocatable :: dlf(:,:), qrs(:,:), qrs_pre(:,:), prec(:), prec_mac(:), dum1(:,:)
  real(r8), allocatable :: Tin(:,:), qin(:,:), qcin(:,:), qiin(:,:), uin(:,:)
  integer :: rad_post, gw_post, dlf_clubb, iqn, nstep, it, i, ios
  character(len=16) :: arg
  character(len=1024) :: fin, fout
  real(r8) :: cld_macmic_ztodt
  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  call get_command_argument(3, arg); read(arg,*) rad_post
  call get_command_argument(4, arg); read(arg,*) gw_post
  call get_command_argument(5, arg); read(arg,*) dlf_clubb
  call get_command_argument(6, arg); read(arg,*) iqn
  qneg_on = (iqn == 1)
  open(10, file=trim(fin), status='old')
  read(10,*) pcols, pver, nmac
  read(10,*) cpair; read(10,*) latvap; read(10,*) latice; read(10,*) gravit; read(10,*) ztodt
  allocate(pm(pver), dp(pver), q0ref(pcols,pver))
  allocate(Tin(pcols,pver), qin(pcols,pver), qcin(pcols,pver), qiin(pcols,pver), uin(pcols,pver))
  read(10,*) pm; read(10,*) dp
  do i = 1, pcols
    read(10,*) Tin(i,:); read(10,*) qin(i,:); read(10,*) qcin(i,:); read(10,*) qiin(i,:)
    read(10,*) uin(i,:); read(10,*) q0ref(i,:)
  end do
  close(10)
  allocate(state%t(pcols,pver), state%u(pcols,pver), state%v(pcols,pver), state%q(pcols,pver,pcnst))
  state%t = Tin; state%u = uin; state%v = 0._r8
  state%q(:,:,ixq) = qin; state%q(:,:,ixcldliq) = qcin; state%q(:,:,ixcldice) = qiin
  allocate(dlf(pcols,pver), qrs(pcols,pver), qrs_pre(pcols,pver), prec(pcols), prec_mac(pcols), dum1(pcols,pver))
  cld_macmic_ztodt = ztodt / nmac
  open(20, file=trim(fout), status='replace')
  do nstep = 0, 1                          ! two physics steps; radiation_do('sw') true at 0 only
    call copy_state(state, state0)
    if (rad_post == 0 .and. nstep == 0) call stub_rad(state0, qrs)   ! legoESM: pre-physics state
    if (gw_post == 0) call stub_gw(state0, ptend_gw_pre)
    ! ---- tphysbc: convect_deep_tend -> physics_update (physpkg.F90:1918-1925)
    call stub_conv(state, ptend, dlf, dlf_clubb == 0)
    call physics_update(state, ptend, ztodt)
    ! ---- macmic loop (physpkg.F90:2033-2189)
    prec_mac = 0._r8
    do it = 1, nmac
      call stub_turb(state, cld_macmic_ztodt, ptend)          ! clubb_tend_cam
      if (dlf_clubb == 1) then                                 ! clubb_intr.F90:2251,2343-2376
        call copy_state(state, state1)
        state1%t = state1%t + ptend%s * cld_macmic_ztodt / cpair
        state1%q = state1%q + ptend%q * cld_macmic_ztodt
        where (state1%t > 268.15_r8)
          dum1 = 0.0_r8
        elsewhere (state1%t < 238.15_r8)
          dum1 = 1.0_r8
        elsewhere
          dum1 = (268.15_r8 - state1%t) / 30._r8
        end where
        ptend%q(:,:,ixcldliq) = ptend%q(:,:,ixcldliq) + dlf * (1._r8 - dum1)
        ptend%q(:,:,ixcldice) = ptend%q(:,:,ixcldice) + dlf * dum1
        ptend%s = ptend%s + dlf * dum1 * latice
        ptend%lq(ixcldice) = .true.
        deallocate(state1%t, state1%u, state1%v, state1%q)
      end if
      call ptend_scale(ptend, 1._r8 / nmac)
      call physics_update(state, ptend, ztodt)
      call stub_micro(state, cld_macmic_ztodt, ptend, prec)   ! microp_driver_tend
      call ptend_scale(ptend, 1._r8 / nmac)
      call physics_update(state, ptend, ztodt)
      prec_mac = prec_mac + prec
    end do
    prec = prec_mac / nmac
    ! ---- radiation_tend -> physics_update (physpkg.F90:2272-2279)
    if (rad_post == 1 .and. nstep == 0) call stub_rad(state, qrs)
    call ptend_init(ptend, 'radheat'); ptend%s = qrs; ptend%ls = .true.
    call physics_update(state, ptend, ztodt)
    ! ---- tphysac: gw_tend -> physics_update (physpkg.F90:1496-1500)
    if (gw_post == 1) then
      call stub_gw(state, ptend)
      call physics_update(state, ptend, ztodt)
    else
      call physics_update(state, ptend_gw_pre, ztodt)
    end if
    do i = 1, pcols
      write(20,'(A,I2,I3,1000ES26.17E3)') 'PREC', nstep, i, prec(i)
    end do
    deallocate(state0%t, state0%u, state0%v, state0%q)
  end do
  do i = 1, pcols
    write(20,'(A,I3,1000ES26.17E3)') 'T ', i, state%t(i,:)
    write(20,'(A,I3,1000ES26.17E3)') 'U ', i, state%u(i,:)
    write(20,'(A,I3,1000ES26.17E3)') 'Q ', i, state%q(i,:,ixq)
    write(20,'(A,I3,1000ES26.17E3)') 'QC', i, state%q(i,:,ixcldliq)
    write(20,'(A,I3,1000ES26.17E3)') 'QI', i, state%q(i,:,ixcldice)
  end do
  close(20)
  print *, 'ORACLE_SEQ_DONE', rad_post, gw_post, dlf_clubb, iqn
end program
