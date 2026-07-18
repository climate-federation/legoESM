! BOUNDED-conventions gridstruct oracle driver: feed the kinked
! mpp-state grid corners (python export, full B lattice incl whatever
! wedge convention the caller supplies), run the verbatim metric +
! angle init with bounded_domain=.true., dump every field the stepper
! consumes.
!
! stdin: n ng, then records
!   GRD i j lon lat        (B corners, fort isd..ied+1 square)
! stdout: records "<NAME> i j val" (2D) / "<NAME> i j k val" (3D last)
program fv3_boundedgs_driver
  use bounded_gs_shim_mod
  use bounded_gs_extract_mod
  implicit none

  integer :: n, ng, npx, npy, i, j, k, ios
  character(len=16) :: tag
  type(shim_atm_type), target :: Atm
  real(kind=R_GRID) :: a, b
  integer :: isd, ied, jsd, jed

  read (*, *) n, ng
  npx = n + 1; npy = n + 1
  Atm%bd%is = 1;  Atm%bd%ie = n
  Atm%bd%js = 1;  Atm%bd%je = n
  Atm%bd%ng = ng
  Atm%bd%isd = 1 - ng; Atm%bd%ied = n + ng
  Atm%bd%jsd = 1 - ng; Atm%bd%jed = n + ng
  Atm%ng = ng
  isd = Atm%bd%isd; ied = Atm%bd%ied
  jsd = Atm%bd%jsd; jed = Atm%bd%jed

  allocate (Atm%gridstruct%grid(isd:ied + 1, jsd:jed + 1, 2))
  allocate (Atm%gridstruct%agrid(isd:ied, jsd:jed, 2))
  allocate (Atm%gridstruct%grid_64(isd:ied + 1, jsd:jed + 1, 2))
  allocate (Atm%gridstruct%agrid_64(isd:ied, jsd:jed, 2))
  allocate (Atm%gridstruct%area(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%area_c(isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%area_64(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%area_c_64(isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%dx(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%dy(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%dx_64(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%dy_64(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%dxa(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%dya(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%dxa_64(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%dya_64(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%dxc(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%dyc(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%dxc_64(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%dyc_64(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%sina(isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%cosa(isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%sina_64(isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%cosa_64(isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%rsina(Atm%bd%is:Atm%bd%ie + 1, Atm%bd%js:Atm%bd%je + 1))
  allocate (Atm%gridstruct%rsin2(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%rsin_u(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%rsin_v(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%cosa_u(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%cosa_v(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%cosa_s(isd:ied, jsd:jed))
  allocate (Atm%gridstruct%sina_u(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%sina_v(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%divg_u(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%divg_v(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%del6_u(isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%del6_v(isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%sin_sg(isd:ied, jsd:jed, 9))
  allocate (Atm%gridstruct%cos_sg(isd:ied, jsd:jed, 9))
  allocate (Atm%gridstruct%ec1(3, isd:ied, jsd:jed))
  allocate (Atm%gridstruct%ec2(3, isd:ied, jsd:jed))
  allocate (Atm%gridstruct%ew(3, isd:ied + 1, jsd:jed, 2))
  allocate (Atm%gridstruct%es(3, isd:ied, jsd:jed + 1, 2))
  allocate (Atm%gridstruct%ee1(3, isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%ee2(3, isd:ied + 1, jsd:jed + 1))
  allocate (Atm%gridstruct%en1(3, isd:ied, jsd:jed + 1))
  allocate (Atm%gridstruct%en2(3, isd:ied + 1, jsd:jed))
  allocate (Atm%gridstruct%eww(3, 4))
  allocate (Atm%gridstruct%ess(3, 4))
  allocate (Atm%gridstruct%l2c_u(Atm%bd%is:Atm%bd%ie + 1, Atm%bd%js:Atm%bd%je))
  allocate (Atm%gridstruct%l2c_v(Atm%bd%is:Atm%bd%ie, Atm%bd%js:Atm%bd%je + 1))
  allocate (Atm%gridstruct%edge_s(npx), Atm%gridstruct%edge_n(npx))
  allocate (Atm%gridstruct%edge_w(npy), Atm%gridstruct%edge_e(npy))
  allocate (Atm%gridstruct%edge_vect_s(isd:ied))
  allocate (Atm%gridstruct%edge_vect_n(isd:ied))
  allocate (Atm%gridstruct%edge_vect_w(jsd:jed))
  allocate (Atm%gridstruct%edge_vect_e(jsd:jed))
  allocate (Atm%ak(2), Atm%bk(2))
  Atm%gridstruct%grid = -9.9d9

  do
    read (*, *, iostat=ios) tag, i, j, a, b
    if (ios /= 0) exit
    select case (trim(tag))
    case ("GRD")
      Atm%gridstruct%grid(i, j, 1) = a
      Atm%gridstruct%grid(i, j, 2) = b
    case default
      stop "unknown record"
    end select
  end do

  call tools_metrics(Atm, npx, npy)
  call grid_area_bounded(Atm, 2)
  ! upstream OVERLOAD_R8 layout: fv_grid_tools computes into the *_64
  ! arrays and grid_utils_init consumes them (copy-back to working
  ! precision at its end).  The excerpted metric stage writes the plain
  ! arrays, so mirror them into *_64 here (exact: everything is r8
  ! under -fdefault-real-8).
  Atm%gridstruct%grid_64 = Atm%gridstruct%grid
  Atm%gridstruct%agrid_64 = Atm%gridstruct%agrid
  Atm%gridstruct%area_64 = Atm%gridstruct%area
  Atm%gridstruct%area_c_64 = Atm%gridstruct%area_c
  Atm%gridstruct%dx_64 = Atm%gridstruct%dx
  Atm%gridstruct%dy_64 = Atm%gridstruct%dy
  Atm%gridstruct%dxa_64 = Atm%gridstruct%dxa
  Atm%gridstruct%dya_64 = Atm%gridstruct%dya
  Atm%gridstruct%dxc_64 = Atm%gridstruct%dxc
  Atm%gridstruct%dyc_64 = Atm%gridstruct%dyc
  call grid_utils_init(Atm, npx, npy, 1, .true., 0, 4)

  call dump2("DX  ", Atm%gridstruct%dx, isd, jsd)
  call dump2("DY  ", Atm%gridstruct%dy, isd, jsd)
  call dump2("DXA ", Atm%gridstruct%dxa, isd, jsd)
  call dump2("DYA ", Atm%gridstruct%dya, isd, jsd)
  call dump2("DXC ", Atm%gridstruct%dxc, isd, jsd)
  call dump2("DYC ", Atm%gridstruct%dyc, isd, jsd)
  call dump2("AREA", Atm%gridstruct%area, isd, jsd)
  call dump2("ARC ", Atm%gridstruct%area_c, isd, jsd)
  call dump2("SINA", Atm%gridstruct%sina, isd, jsd)
  call dump2("COSA", Atm%gridstruct%cosa, isd, jsd)
  call dump2r("RSNA", Atm%gridstruct%rsina, Atm%bd%is, Atm%bd%js)
  call dump2r("RSN2", Atm%gridstruct%rsin2, isd, jsd)
  call dump2r("RSNU", Atm%gridstruct%rsin_u, isd, jsd)
  call dump2r("RSNV", Atm%gridstruct%rsin_v, isd, jsd)
  call dump2r("CSAU", Atm%gridstruct%cosa_u, isd, jsd)
  call dump2r("CSAV", Atm%gridstruct%cosa_v, isd, jsd)
  call dump2r("CSAS", Atm%gridstruct%cosa_s, isd, jsd)
  call dump2r("SNAU", Atm%gridstruct%sina_u, isd, jsd)
  call dump2r("SNAV", Atm%gridstruct%sina_v, isd, jsd)
  call dump2r("DVGU", Atm%gridstruct%divg_u, isd, jsd)
  call dump2r("DVGV", Atm%gridstruct%divg_v, isd, jsd)
  call dump2r("DL6U", Atm%gridstruct%del6_u, isd, jsd)
  call dump2r("DL6V", Atm%gridstruct%del6_v, isd, jsd)
  do k = 1, 9
    call dump2rk("SSG ", Atm%gridstruct%sin_sg(:, :, k), isd, jsd, k)
    call dump2rk("CSG ", Atm%gridstruct%cos_sg(:, :, k), isd, jsd, k)
  end do
  call dump2("AGX ", Atm%gridstruct%agrid(:, :, 1), isd, jsd)
  call dump2("AGY ", Atm%gridstruct%agrid(:, :, 2), isd, jsd)
  write (*, '(A,2ES26.17E3)') "DAMN ", Atm%gridstruct%da_min, &
    Atm%gridstruct%da_min_c

contains

  subroutine dump2(name, f, ilo, jlo)
    character(len=*), intent(in) :: name
    real(kind=R_GRID), intent(in) :: f(:, :)
    integer, intent(in) :: ilo, jlo
    integer :: ii, jj
    do jj = 1, size(f, 2)
      do ii = 1, size(f, 1)
        write (*, '(A,2I6,ES26.17E3)') name, ii + ilo - 1, jj + jlo - 1, &
          f(ii, jj)
      end do
    end do
  end subroutine dump2

  subroutine dump2r(name, f, ilo, jlo)
    character(len=*), intent(in) :: name
    real, intent(in) :: f(:, :)
    integer, intent(in) :: ilo, jlo
    integer :: ii, jj
    do jj = 1, size(f, 2)
      do ii = 1, size(f, 1)
        write (*, '(A,2I6,ES26.17E3)') name, ii + ilo - 1, jj + jlo - 1, &
          f(ii, jj)
      end do
    end do
  end subroutine dump2r

  subroutine dump2rk(name, f, ilo, jlo, kk)
    character(len=*), intent(in) :: name
    real, intent(in) :: f(:, :)
    integer, intent(in) :: ilo, jlo, kk
    integer :: ii, jj
    do jj = 1, size(f, 2)
      do ii = 1, size(f, 1)
        write (*, '(A,3I6,ES26.17E3)') name, ii + ilo - 1, jj + jlo - 1, &
          kk, f(ii, jj)
      end do
    end do
  end subroutine dump2rk

end program fv3_boundedgs_driver
