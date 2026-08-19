! VERBATIM extract certificate for the COMPOSED duo halo-exchange chain
! (ext_scalar A/B + ext_vector D/C) -- authoritative Zenodo 8327578
! symmetryclean tree, pinned at:
!   /burg-archive/glab/users/pg2328/fv3_oracle_pinned/atmos_cubed_sphere-symmetryclean
!   tools/fv_duogrid.F90  sha256
!   c3c745c4071581ef23d78291873b5ba92a6cfdfc5711e5e80ede0be1e613795e
!
! Blocks copied VERBATIM (line ranges of the pinned file):
!   tools/fv_duogrid.F90: 505- 569  ext_scalar_3d          (staged copy, hooks)
!   tools/fv_duogrid.F90: 626- 975  ext_vector             (staged copy, hooks)
!   tools/fv_duogrid.F90: 977-1137  cube_rmp               (byte-verbatim)
!   tools/fv_duogrid.F90:1177-1420  fill_corners_domain_decomp_2d (byte-verbatim)
!   tools/fv_duogrid.F90:1422-1705  fill_corners_domain_decomp_3d (byte-verbatim)
!   tools/fv_duogrid.F90:2590-2674  cubed_a2c_halo         (byte-verbatim)
!   tools/fv_duogrid.F90:2676-2763  cubed_a2d_halo         (byte-verbatim)
!   tools/fv_duogrid.F90:2765-2844  c2l_ord2_cgrid         (byte-verbatim)
!
! DEVIATIONS -- complete list; everything else is byte-identical:
!   D1  module wrapper: the bodies live in extchain_extract_mod and resolve
!       their dependencies through explicit `use` of the REAL linked modules
!       (fv_arrays_mod, lib_grid_mod, mpp/mpp_domains/mpp_parameter,
!       duogrid_mod public fill_corner_region + lagrange_poly_interp,
!       fv_grid_utils_mod c2l_ord2) instead of duogrid_mod internal
!       module scope.  NO shims: mpp_update_domains, mpp_send/recv,
!       fill_corner_region etc. are the real FMS / the real fv_duogrid.o.
!   D2  the two ENTRY POINTS are renamed ext_scalar_3d_staged /
!       ext_vector_staged and take one extra `tag` argument (dump naming).
!   D3  inserted `call extchain_dump3(...)` stage-dump lines, every one
!       marked `! EXTCHAIN-HOOK`.  Pure I/O; no numerics.
!   D4  NOTHING ELSE.  In particular the private helpers cube_rmp,
!       fill_corners_domain_decomp_2d/3d, cubed_a2c_halo, cubed_a2d_halo,
!       c2l_ord2_cgrid are byte-verbatim (whitespace included).
!
! CERTIFICATION: the staged chain is NOT trusted by construction -- the
! driver (fv3_extchain_oracle_driver.F90) runs the REAL linked
! duogrid_mod::ext_scalar / ext_vector on an identical input copy and
! records max|staged - composed| per family; the fixture builder and the
! consuming pytest require exact 0.0 before any staged dump is trusted.
!
! Verbatim-ness is machine-checked: tests/grids/test_fv3_extchain_oracle.py
! re-reads the pinned line ranges and diffs them against this file with the
! documented deviation lines stripped (skips when the pinned tree is absent).

module extchain_dump_mod
  implicit none
  private
  public :: extchain_dump_open, extchain_dump_close, extchain_dump3, &
            extchain_dump2, extchain_mf_note
  integer :: dump_unit = -1, mf_unit = -1
  integer(kind=8) :: dump_off = 0
contains
  subroutine extchain_dump_open(datname, mfname)
    character(len=*), intent(in) :: datname, mfname
    open (newunit=dump_unit, file=trim(datname), access='stream', &
          form='unformatted', status='replace', action='write')
    open (newunit=mf_unit, file=trim(mfname), status='replace', &
          action='write')
    dump_off = 0
  end subroutine extchain_dump_open
  subroutine extchain_dump_close()
    close (dump_unit)
    close (mf_unit)
    dump_unit = -1
    mf_unit = -1
  end subroutine extchain_dump_close
  subroutine extchain_dump3(name, arr)
    character(len=*), intent(in) :: name
    real, intent(in) :: arr(:, :, :)
    write (dump_unit) arr
    write (mf_unit, '(A,3(1X,I6),1X,I15)') trim(name), shape(arr), dump_off
    dump_off = dump_off + int(size(arr), 8)*8
  end subroutine extchain_dump3
  subroutine extchain_dump2(name, arr)
    character(len=*), intent(in) :: name
    real, intent(in) :: arr(:, :)
    write (dump_unit) arr
    write (mf_unit, '(A,2(1X,I6),1X,I6,1X,I15)') trim(name), shape(arr), &
      1, dump_off
    dump_off = dump_off + int(size(arr), 8)*8
  end subroutine extchain_dump2
  subroutine extchain_mf_note(text)
    character(len=*), intent(in) :: text
    write (mf_unit, '(A)') trim(text)
  end subroutine extchain_mf_note
end module extchain_dump_mod

module extchain_extract_mod
  use fv_arrays_mod, only: duogrid_type, fv_grid_bounds_type, &
                           fv_flags_type, fv_grid_type
  use lib_grid_mod, only: R_GRID
  use mpp_mod, only: FATAL, mpp_error, mpp_pe, mpp_npes, mpp_root_pe, &
                     mpp_send, mpp_recv, mpp_sync
  use mpp_domains_mod, only: mpp_update_domains, domain2d, mpp_get_tile_id
  use mpp_parameter_mod, only: DGRID_NE, CGRID_NE, NORTH, EAST
  use duogrid_mod, only: fill_corner_region, lagrange_poly_interp
  use fv_grid_utils_mod, only: c2l_ord2
  use extchain_dump_mod, only: extchain_dump3
  implicit none
  public
  interface fill_corners_domain_decomp
    module procedure fill_corners_domain_decomp_2d
    module procedure fill_corners_domain_decomp_3d
  end interface fill_corners_domain_decomp

contains
  subroutine ext_scalar_3d_staged(var, dg, bd, domain, istag, jstag, tag)   ! EXTCHAIN renamed (deviation D2)
    type(duogrid_type), intent(INOUT) :: dg
    type(domain2d), intent(INOUT)         :: domain
    type(fv_grid_bounds_type), intent(IN) :: bd
    integer, intent(IN) :: istag, jstag
    character(len=*), intent(in) :: tag   ! EXTCHAIN-HOOK arg (deviation D2)
    real, dimension(:, :, :) :: var
    !--- local
    integer :: i, j, isd, ied, jsd, jed, k, npz
    integer :: dims(3)

    integer, dimension(:, :), allocatable :: k2e_loc_u
    real(kind=R_GRID), dimension(:, :, :), allocatable :: k2e_coef_u

    isd = bd%isd
    ied = bd%ied
    jsd = bd%jsd
    jed = bd%jed

    if ((istag == 0 .and. jstag == 1) .or. (istag == 1 .and. jstag == 0)) then
      call mpp_error(FATAL, 'ext_scalar_3d for stag fields not implemented')
    end if

    !--- update domains
    if (istag == 1 .and. jstag == 1) then
      allocate (k2e_coef_u(dg%k2e_nord, isd:ied + 1, jsd:jed + 1))
      allocate (k2e_loc_u(isd:ied + 1, jsd:jed + 1))
!      k2e_loc_u = dg%k2e_loc_b
!      k2e_coef_u = dg%k2e_coef_b

      do i = isd, ied + 1
      do j = jsd, jed + 1
        k2e_loc_u(i, j) = dg%k2e_loc_b(i, j)
        k2e_coef_u(:, i, j) = dg%k2e_coef_b(:, i, j)
      end do
      end do
      call mpp_update_domains(var, domain, complete=.true., position=NORTH + EAST)
    call extchain_dump3('S1_'//trim(tag), var)   ! EXTCHAIN-HOOK (deviation D3)
    end if

    if (istag == 0 .and. jstag == 0) then
      allocate (k2e_coef_u(dg%k2e_nord, isd:ied, jsd:jed))
      allocate (k2e_loc_u(isd:ied, jsd:jed))
      do i = isd, ied
      do j = jsd, jed
        k2e_loc_u(i, j) = dg%k2e_loc(i, j)
        k2e_coef_u(:, i, j) = dg%k2e_coef(:, i, j)
      end do
      end do
      !k2e_loc_u = dg%k2e_loc
      !k2e_coef_u = dg%k2e_coef
      call mpp_update_domains(var, domain, complete=.true.)
    call extchain_dump3('S1_'//trim(tag), var)   ! EXTCHAIN-HOOK (deviation D3)
    end if

    !--- remap to ext
    dims = shape(var)
    npz = dims(3)

    do k = 1, npz
      call cube_rmp(var(:, :, k), dg, bd, domain, istag, jstag, k2e_loc_u, k2e_coef_u)
    end do
    call extchain_dump3('S2_'//trim(tag), var)   ! EXTCHAIN-HOOK (deviation D3)
    call fill_corners_domain_decomp(var, dg, bd, domain, istag, jstag)
    call fill_corner_region(var, bd, dg, istag, jstag)

    deallocate (k2e_coef_u)
    deallocate (k2e_loc_u)
  end subroutine ext_scalar_3d_staged   ! EXTCHAIN renamed (deviation D2)

  subroutine ext_vector_staged(u_in, v_in, dg, bd, domain, gridstruct, flagstruct, ieu_stag, jeu_stag, iev_stag, jev_stag, tag)   ! EXTCHAIN renamed (deviation D2)
    type(duogrid_type), intent(INOUT) :: dg
    type(fv_grid_bounds_type), intent(IN) :: bd
    integer, intent(IN) :: ieu_stag, jeu_stag, iev_stag, jev_stag
    character(len=*), intent(in) :: tag   ! EXTCHAIN-HOOK arg (deviation D2)
    type(domain2d), intent(INOUT)         :: domain
    type(fv_flags_type), intent(IN) :: flagstruct
    type(fv_grid_type), intent(INOUT) :: gridstruct
    real, dimension(bd%isd:bd%ied + ieu_stag, bd%jsd:bd%jed + jeu_stag, flagstruct%npz) :: u_in
    real, dimension(bd%isd:bd%ied + iev_stag, bd%jsd:bd%jed + jev_stag, flagstruct%npz) :: v_in
    !--- local
    real, dimension(bd%isd:bd%ied + ieu_stag, bd%jsd:bd%jed + jeu_stag, flagstruct%npz) :: u
    real, dimension(bd%isd:bd%ied + iev_stag, bd%jsd:bd%jed + jev_stag, flagstruct%npz) :: v
    real, dimension(bd%isd - 1:bd%ied + 1 + ieu_stag, bd%jsd - 1:bd%jed + 1 + jeu_stag, flagstruct%npz) :: up1
    real, dimension(bd%isd - 1:bd%ied + 1 + iev_stag, bd%jsd - 1:bd%jed + 1 + jev_stag, flagstruct%npz) :: vp1
    integer :: is, ie, js, je, isd, ied, jsd, jed, isdp1, iedp1, jsdp1, jedp1, ng, npz
    integer :: i, j, k
    real, dimension(:, :, :), allocatable :: ull, vll, ullp1, vllp1

    real, dimension(:, :, :, :), allocatable :: c2l, l2c
    integer, dimension(:, :), allocatable :: k2e_loc_u
    real(kind=R_GRID), dimension(:, :, :), allocatable :: k2e_coef_u
    integer, dimension(:, :), allocatable :: k2e_loc_v
    real(kind=R_GRID), dimension(:, :, :), allocatable :: k2e_coef_v

    !--- assign parameters
    is = bd%is
    ie = bd%ie
    js = bd%js
    je = bd%je
    isd = bd%isd
    ied = bd%ied
    jsd = bd%jsd
    jed = bd%jed
    isdp1 = bd%isd - 1
    iedp1 = bd%ied + 1
    jsdp1 = bd%jsd - 1
    jedp1 = bd%jed + 1
    ng = bd%ng

    npz = flagstruct%npz

    allocate (ull(isd:ied, jsd:jed, npz))
    allocate (vll(isd:ied, jsd:jed, npz))
    allocate (ullp1(isd - 1:ied + 1, jsd - 1:jed + 1, npz))
    allocate (vllp1(isd - 1:ied + 1, jsd - 1:jed + 1, npz))
    ull = -99999.
    vll = -99999.
    allocate (c2l(2, 2, isd:ied + iev_stag, jsd:jed + jev_stag))
    allocate (l2c(2, 2, isd:ied + iev_stag, jsd:jed + jev_stag))
    c2l = -99999.
    l2c = -99999.

    ! Since we are transforming C/D variables to A using c2l, we only use the A-grid
    ! remapping coeff. C/D extensions coeff are available, using them however with the current
    ! logic produced more noised compared to the first method.
    ! To CHECK:  do C/D on the fly without passing by A.

!Agrid
    if ((ieu_stag == 0 .and. jeu_stag == 0 .and. iev_stag == 0 .and. jev_stag == 0)) then  !AGRID CGRID
      allocate (k2e_coef_u(dg%k2e_nord, isd:ied, jsd:jed))
      allocate (k2e_loc_u(isd:ied, jsd:jed))
      allocate (k2e_coef_v(dg%k2e_nord, isd:ied, jsd:jed))
      allocate (k2e_loc_v(isd:ied, jsd:jed))
      k2e_loc_u = -99999
      k2e_coef_u = -99999
      k2e_loc_v = -99999
      k2e_coef_v = -99999
      do j = jsd, jed
      do i = isd, ied
        c2l(:, :, i, j) = dg%a_c2l(:, :, i, j)
        l2c(:, :, i, j) = dg%a_l2c(:, :, i, j)
        k2e_loc_u(i, j) = dg%k2e_loc(i, j)
        k2e_coef_u(:, i, j) = dg%k2e_coef(:, i, j)
        k2e_loc_v(i, j) = dg%k2e_loc(i, j)
        k2e_coef_v(:, i, j) = dg%k2e_coef(:, i, j)
      end do
      end do

    else

      allocate (k2e_coef_u(dg%k2e_nord, isd - 1:ied + 1, jsd - 1:jed + 1))
      allocate (k2e_loc_u(isd - 1:ied + 1, jsd - 1:jed + 1))
      allocate (k2e_coef_v(dg%k2e_nord, isd - 1:ied + 1, jsd - 1:jed + 1))
      allocate (k2e_loc_v(isd - 1:ied + 1, jsd - 1:jed + 1))
      k2e_loc_u = -99999
      k2e_coef_u = -99999
      k2e_loc_v = -99999
      k2e_coef_v = -99999
      do j = jsd - 1, jed + 1
      do i = isd - 1, ied + 1
        !c2l (:,:,i,j) = dg%a_c2l(:,:,i,j)
        !l2c(:,:,i,j) = dg%a_l2c(:,:,i,j)
        k2e_loc_u(i, j) = dg%k2e_loc(i, j)
        k2e_coef_u(:, i, j) = dg%k2e_coef(:, i, j)
        k2e_loc_v(i, j) = dg%k2e_loc(i, j)
        k2e_coef_v(:, i, j) = dg%k2e_coef(:, i, j)
      end do
      end do

    end if
    if (ieu_stag == 0 .and. jeu_stag == 0 .and. iev_stag == 0 .and. jev_stag == 0) then  !AGRID
      do k = 1, npz
        do j = js, je
          do i = is, ie
            ull(i, j, k) = c2l(1, 1, i, j)*u_in(i, j, k) + c2l(1, 2, i, j)*v_in(i, j, k)
          end do
        end do
        do j = js, je
          do i = is, ie
            vll(i, j, k) = c2l(2, 1, i, j)*u_in(i, j, k) + c2l(2, 2, i, j)*v_in(i, j, k)
          end do
        end do
      end do
    end if

    if (ieu_stag == 0 .and. jeu_stag == 1 .and. iev_stag == 1 .and. jev_stag == 0) then  !DGRID
      call mpp_update_domains(u_in, v_in, domain, gridtype=DGRID_NE) ! update interior pe first
    call extchain_dump3('S1u_'//trim(tag), u_in)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S1v_'//trim(tag), v_in)   ! EXTCHAIN-HOOK (deviation D3)
      !call c2l_ord2(u_in, v_in, ull, vll, gridstruct, npz, 0, bd, .false.) !zonal-merdional vel
      call c2l_ord2(u_in, v_in, ull, vll, gridstruct, npz, 0, bd, .true.) !zonal-merdional vel

      do k = 1, npz
        do j = js, je + 1
          do i = is, ie + 1
            ullp1(i, j, k) = ull(i, j, k)
          end do
        end do
        do j = js, je + 1
          do i = is, ie + 1
            vllp1(i, j, k) = vll(i, j, k)
          end do
        end do
      end do

    end if

    if (ieu_stag == 1 .and. jeu_stag == 0 .and. iev_stag == 0 .and. jev_stag == 1) then  !CGRID
      call mpp_update_domains(u_in, v_in, domain, gridtype=CGRID_NE) !update interior pe first
    call extchain_dump3('S1u_'//trim(tag), u_in)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S1v_'//trim(tag), v_in)   ! EXTCHAIN-HOOK (deviation D3)
      call c2l_ord2_cgrid(u_in, v_in, ull, vll, gridstruct, npz, 0, bd, .true.) !zonal-merdional vel
      do k = 1, npz
        do j = js, je + 1
          do i = is, ie + 1
            ullp1(i, j, k) = ull(i, j, k)
          end do
        end do
        do j = js, je + 1
          do i = is, ie + 1
            vllp1(i, j, k) = vll(i, j, k)
          end do
        end do
      end do

    end if  !C-grid

    !Update latlonwinds haloes
      !!!!!!!!!!!!!!!!!!!!!!!!!!!
    if ((ieu_stag == 0 .and. jeu_stag == 0 .and. iev_stag == 0 .and. jev_stag == 0)) then  ! AGRID
      call mpp_update_domains(ull, domain, complete=.false.)
      call mpp_update_domains(vll, domain, complete=.true.)
    else !CDGRID
    call extchain_dump3('S2u_'//trim(tag), ull)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S2v_'//trim(tag), vll)   ! EXTCHAIN-HOOK (deviation D3)
      call mpp_update_domains(ullp1, dg%domain_for_duo, complete=.false.)
      call mpp_update_domains(vllp1, dg%domain_for_duo, complete=.true.)
    call extchain_dump3('S3u_'//trim(tag), ullp1)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S3v_'//trim(tag), vllp1)   ! EXTCHAIN-HOOK (deviation D3)
    end if

    !--- remap to ext
      !!!!!!!!!!!!!!!!!!
    if ((ieu_stag == 0 .and. jeu_stag == 0 .and. iev_stag == 0 .and. jev_stag == 0)) then   !AGRID
      do k = 1, npz
        call cube_rmp(ull(:, :, k), dg, bd, domain, 0, 0, k2e_loc_u, k2e_coef_u)
        call cube_rmp(vll(:, :, k), dg, bd, domain, 0, 0, k2e_loc_v, k2e_coef_v)
      end do
      call fill_corners_domain_decomp(ull, dg, bd, domain, 0, 0)
      call fill_corners_domain_decomp(vll, dg, bd, domain, 0, 0)

    else !CDGRID

      do k = 1, npz
        call cube_rmp(ullp1(:, :, k), dg, dg%bd, dg%domain_for_duo, 0, 0, k2e_loc_u, k2e_coef_u)
        call cube_rmp(vllp1(:, :, k), dg, dg%bd, dg%domain_for_duo, 0, 0, k2e_loc_v, k2e_coef_v)
      end do
      call fill_corners_domain_decomp(ullp1, dg, dg%bd, dg%domain_for_duo, 0, 0)
      call fill_corners_domain_decomp(vllp1, dg, dg%bd, dg%domain_for_duo, 0, 0)
    call extchain_dump3('S4u_'//trim(tag), ullp1)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S4v_'//trim(tag), vllp1)   ! EXTCHAIN-HOOK (deviation D3)

      if (ieu_stag == 1 .and. jeu_stag == 0 .and. iev_stag == 0 .and. jev_stag == 1) then  !CGRID
        call cubed_a2c_halo(npz, ullp1, vllp1, up1, vp1, dg)
      else
        call cubed_a2d_halo(npz, ullp1, vllp1, up1, vp1, dg)
      end if
    call extchain_dump3('S5u_'//trim(tag), ullp1)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S5v_'//trim(tag), vllp1)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S5up_'//trim(tag), up1)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S5vp_'//trim(tag), vp1)   ! EXTCHAIN-HOOK (deviation D3)

      do k = 1, npz
      do j = jsd, jed + jeu_stag
      do i = isd, ied + ieu_stag
        u(i, j, k) = up1(i, j, k)
      end do
      end do
      do j = jsd, jed + jev_stag
      do i = isd, ied + iev_stag
        v(i, j, k) = vp1(i, j, k)
      end do
      end do
      end do
    end if

    if (ieu_stag == 0 .and. jeu_stag == 0 .and. iev_stag == 0 .and. jev_stag == 0) then  !AGRID
      !--- convert to wind
      ! need to update the interior pes as well, so no switch
      ! south
      do k = 1, npz
        do j = jsd, js - 1
          do i = isd, ied
            u_in(i, j, k) = l2c(1, 1, i, j)*ull(i, j, k) + l2c(1, 2, i, j)*vll(i, j, k)
          end do
        end do
        do j = jsd, js - 1
          do i = isd, ied
            v_in(i, j, k) = l2c(2, 1, i, j)*ull(i, j, k) + l2c(2, 2, i, j)*vll(i, j, k)
          end do
        end do
      end do

      ! north
      do k = 1, npz
        do j = je + 1, jed
          do i = isd, ied
            u_in(i, j, k) = l2c(1, 1, i, j)*ull(i, j, k) + l2c(1, 2, i, j)*vll(i, j, k)
          end do
        end do
        do j = je + 1, jed
          do i = isd, ied
            v_in(i, j, k) = l2c(2, 1, i, j)*ull(i, j, k) + l2c(2, 2, i, j)*vll(i, j, k)
          end do
        end do
      end do

      ! west
      do k = 1, npz
        do j = js, je
          do i = isd, is - 1
            u_in(i, j, k) = l2c(1, 1, i, j)*ull(i, j, k) + l2c(1, 2, i, j)*vll(i, j, k)
          end do
        end do
        do j = js, je
          do i = isd, is - 1
            v_in(i, j, k) = l2c(2, 1, i, j)*ull(i, j, k) + l2c(2, 2, i, j)*vll(i, j, k)
          end do
        end do
      end do

      ! east
      do k = 1, npz
        do j = js, je
          do i = ie + 1, ied
            u_in(i, j, k) = l2c(1, 1, i, j)*ull(i, j, k) + l2c(1, 2, i, j)*vll(i, j, k)
          end do
        end do
        do j = js, je
          !do i = ie+ieu_stag+1,ied+iev_stag
          do i = ie + 1, ied
            v_in(i, j, k) = l2c(2, 1, i, j)*ull(i, j, k) + l2c(2, 2, i, j)*vll(i, j, k)
          end do
        end do
      end do

    else ! CD

      if (dg%rmp_s) then
        ! south
        do k = 1, npz
          do j = jsd, js - 1
            do i = isd, ied + ieu_stag
              u_in(i, j, k) = u(i, j, k)
            end do
          end do
          do j = jsd, js - 1
            do i = isd, ied + iev_stag
              v_in(i, j, k) = v(i, j, k)
            end do
          end do
        end do
      end if

      if (dg%rmp_n) then
        ! north
        do k = 1, npz
          do j = je + jeu_stag + 1, jed + jeu_stag
            do i = isd, ied + ieu_stag
              u_in(i, j, k) = u(i, j, k)
            end do
          end do
          !do j = je+jeu_stag+1,jed+jev_stag
          do j = je + jev_stag + 1, jed + jev_stag
            do i = isd, ied + iev_stag
              v_in(i, j, k) = v(i, j, k)
            end do
          end do
        end do
      end if

      if (dg%rmp_w) then
        ! west
        do k = 1, npz
          do j = jsd, jed + jeu_stag
            do i = isd, is - 1
              u_in(i, j, k) = u(i, j, k)
            end do
          end do
          do j = jsd, jed + jev_stag
            do i = isd, is - 1
              v_in(i, j, k) = v(i, j, k)
            end do
          end do
        end do
      end if

      if (dg%rmp_e) then
        ! east
        do k = 1, npz
          do j = jsd, jed + jeu_stag
            do i = ie + ieu_stag + 1, ied + ieu_stag
              u_in(i, j, k) = u(i, j, k)
            end do
          end do
          do j = jsd, jed + jev_stag
            !do i = ie+ieu_stag+1,ied+iev_stag
            do i = ie + iev_stag + 1, ied + iev_stag
              v_in(i, j, k) = v(i, j, k)
            end do
          end do
        end do
      end if

    end if !if C-grid

    call extchain_dump3('S6u_'//trim(tag), u_in)   ! EXTCHAIN-HOOK (deviation D3)
    call extchain_dump3('S6v_'//trim(tag), v_in)   ! EXTCHAIN-HOOK (deviation D3)
!Fill corner region
!!!!!!!!!!!!!!!!!!!
    call fill_corner_region(u_in, bd, dg, ieu_stag, jeu_stag)
    call fill_corner_region(v_in, bd, dg, iev_stag, jev_stag)

    !--- deallocate wk arrays
    deallocate (ull)
    deallocate (vll)
    deallocate (ullp1)
    deallocate (vllp1)
    deallocate (k2e_coef_u)
    deallocate (k2e_coef_v)
    deallocate (k2e_loc_u)
    deallocate (k2e_loc_v)
    deallocate (c2l)
    deallocate (l2c)

  end subroutine ext_vector_staged   ! EXTCHAIN renamed (deviation D2)

  subroutine cube_rmp(var, dg, bd, domain, istag, jstag, k2e_loc, k2e_coef)
    type(duogrid_type), intent(in) :: dg
    type(fv_grid_bounds_type), intent(IN) :: bd
    type(domain2d), intent(INOUT) :: domain
    integer, intent(in) :: istag, jstag
    !integer, intent(in) :: ext_x, ext_y
    real, dimension(bd%isd:bd%ied + istag, bd%jsd:bd%jed + jstag) :: var
    ! local
    real, dimension(:, :), allocatable :: var_kik
    logical :: rmp_w, rmp_e, rmp_s, rmp_n, rmp_sw, rmp_se, rmp_ne, rmp_nw
    integer :: is, ie, js, je, isd, ied, jsd, jed, ng
    integer :: i, j, ii, n
    integer :: loc, lo, offset, jmin, jmax
    !   real, dimension(:,:), allocatable :: k2e_loc
    !   real, dimension(:,:,:), allocatable :: k2e_coef
    !  integer, dimension(bd%isd:bd%ied+ext_x, bd%jsd:bd%jed+ext_y) :: k2e_loc
    !  real, dimension(dg%k2e_nord,bd%isd:bd%ied+ext_x, bd%jsd:bd%jed+ext_y) :: k2e_coef
    integer, dimension(bd%isd:bd%ied + istag, bd%jsd:bd%jed + jstag) :: k2e_loc
    real(kind=R_GRID), dimension(dg%k2e_nord, bd%isd:bd%ied + istag, bd%jsd:bd%jed + jstag) :: k2e_coef

    !--- assign parameters
    is = bd%is
    ie = bd%ie
    js = bd%js
    je = bd%je
    isd = bd%isd
    ied = bd%ied
    jsd = bd%jsd
    jed = bd%jed
    ng = bd%ng

    rmp_w = dg%rmp_w
    rmp_e = dg%rmp_e
    rmp_s = dg%rmp_s
    rmp_n = dg%rmp_n

    rmp_sw = dg%rmp_sw
    rmp_se = dg%rmp_se
    rmp_ne = dg%rmp_ne
    rmp_nw = dg%rmp_nw

    offset = dg%k2e_nord/2

    !--- allocata wk array
    allocate (var_kik(isd:ied + istag, jsd:jed + jstag))

    !--- copy neighbor to var_kik
    do ii = 1, ng
      do i = isd, ied + istag
        j = js - ii
        var_kik(i, j) = var(i, j)
        j = je + jstag + ii
        var_kik(i, j) = var(i, j)
      end do
      do j = js, je + jstag
        i = is - ii
        var_kik(i, j) = var(i, j)
        i = ie + istag + ii
        var_kik(i, j) = var(i, j)
      end do
    end do

    !--- south
    if (rmp_s) then
      do ii = 1, ng

        j = js - ii

        do i = is, ie + istag

          loc = k2e_loc(i, j)
          lo = loc - offset

          var(i, j) = 0.
          do n = 1, dg%k2e_nord
            var(i, j) = var(i, j) + var_kik(lo + n, j)*k2e_coef(n, i, j)
          end do

        end do

      end do
    end if

    !--- north
    if (rmp_n) then
      do ii = 1, ng

        j = je + jstag + ii

        do i = is, ie + istag

          loc = k2e_loc(i, j)
          lo = loc - offset

          var(i, j) = 0.
          do n = 1, dg%k2e_nord
            var(i, j) = var(i, j) + var_kik(lo + n, j)*k2e_coef(n, i, j)
          end do

        end do

      end do
    end if

    !--- west
    if (rmp_w) then
      jmin = js
      jmax = je + jstag
! For few pe layouts we can extend the rmp process
! to fill the pe corners at the edges; but for most of them
! the lo+n goes beyond the var_kik dimensions, meaning the remapped
! value at the corners needs to grab data outside of what is available on its pe
! so we end up filling the pe corner region from the neighbooring pe.

      do ii = 1, ng

        i = is - ii

        !do j = js,je+jstag
        do j = jmin, jmax

          loc = k2e_loc(i, j)
          lo = loc - offset

          var(i, j) = 0.
          do n = 1, dg%k2e_nord
            var(i, j) = var(i, j) + var_kik(i, lo + n)*k2e_coef(n, i, j)
          end do

        end do

      end do
    end if

    !--- east
    if (rmp_e) then
      do ii = 1, ng

        i = ie + istag + ii

        do j = js, je + jstag

          loc = k2e_loc(i, j)
          lo = loc - offset

          var(i, j) = 0.
          do n = 1, dg%k2e_nord
            var(i, j) = var(i, j) + var_kik(i, lo + n)*k2e_coef(n, i, j)
          end do

        end do

      end do
    end if

    !move this outside of cube_rmp to do the comms 3D
    !call fill_corners_domain_decomp(var, dg, bd, domain, istag, jstag)

    deallocate (var_kik)

  end subroutine cube_rmp

  subroutine fill_corners_domain_decomp_2d(var, dg, bd, domain, istag, jstag)
    type(duogrid_type), intent(in) :: dg
    type(domain2d), intent(IN)         :: domain
    type(fv_grid_bounds_type), intent(IN) :: bd
    integer, intent(in) :: istag, jstag
    real, dimension(bd%isd:, bd%jsd:) :: var
    integer :: is, ie, js, je, isd, ied, jsd, jed, ng
    integer :: i, j, layoutx, layouty, tope, frompe, gid, kk
    integer :: tile(1), npes_per_tile
    real, dimension(1:bd%ng, 1:bd%ng) :: corner

    is = bd%is
    ie = bd%ie
    js = bd%js
    je = bd%je
    isd = bd%isd
    ied = bd%ied
    jsd = bd%jsd
    jed = bd%jed
    ng = bd%ng

    tile = mpp_get_tile_id(domain)
    npes_per_tile = mpp_npes()/6.
    gid = mpp_pe()
    layoutx = dg%layout(1)
    layouty = dg%layout(2)

    if (layoutx + layouty > 2) then

!!!!!!!!!!!!!!!!!!
!!!!!! WEST !!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_w .and. layouty > 1) then

        do kk = 1, layouty - 1
          tope = (kk)*layoutx + (tile(1) - 1)*npes_per_tile
          frompe = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do i = 1, ng
              do j = 1, ng
                corner(i, j) = var(isd + i - 1, je - ng + j)
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(isd + i - 1, jsd + j - 1) = corner(i, j)
              end do
            end do
          end if

          frompe = (kk)*layoutx + (tile(1) - 1)*npes_per_tile
          tope = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do i = 1, ng
              do j = 1, ng
                corner(i, j) = var(isd + i - 1, js + jstag + j - 1)
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(isd + i - 1, je + jstag + 1 + j - 1) = corner(i, j)
              end do
            end do
          end if

        end do !enddo kk

      end if !endif rmp_w

!!!!!!!!!!!!!!!!!!
!!!!!! EAST !!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_e .and. layouty > 1) then

        do kk = 1, layouty - 1

          tope = (kk)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1
          frompe = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1

          if (gid == frompe) then
            do i = 1, ng
              do j = 1, ng
                corner(i, j) = var(ie + istag + 1 + i - 1, je - ng + j)
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(ie + istag + 1 + i - 1, jsd + j - 1) = corner(i, j)
              end do
            end do
          end if

          frompe = (kk)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1
          tope = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1

          if (gid == frompe) then
            do i = 1, ng
              do j = 1, ng
                corner(i, j) = var(ie + istag + 1 + i - 1, js + jstag + j - 1)
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(ie + istag + 1 + i - 1, je + jstag + 1 + j - 1) = corner(i, j)
              end do
            end do
          end if

        end do
      end if

!!!!!!!!!!!!!!!!!!
!!!!!!!SOUTH!!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_s .and. layoutx > 1) then

        do kk = 1, layoutx - 1

          tope = (kk) + (tile(1) - 1)*npes_per_tile
          frompe = (kk - 1) + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
          do i = 1, ng
            do j = 1, ng
              corner(i, j) = var(ie + istag - ng + i, jsd + j - 1)
              corner(i, j) = var(ie - ng + i, jsd + j - 1)
            end do
          end do
          call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(isd + i - 1, jsd + j - 1) = corner(i, j)
              end do
            end do
          end if

          frompe = (kk) + (tile(1) - 1)*npes_per_tile
          tope = (kk - 1) + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
          do i = 1, ng
            do j = 1, ng
              corner(i, j) = var(is + istag + i - 1, jsd + j - 1)
            end do
          end do
          call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(ie + istag + 1 + i - 1, jsd + j - 1) = corner(i, j)
              end do
            end do
          end if

        end do
      end if

!!!!!!!!!!!!!!!!!!
!!!!!!!NORTH!!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_n .and. layoutx > 1) then

        do kk = 1, layoutx - 1

          tope = (kk) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile
          frompe = (kk - 1) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile
          if (gid == frompe) then
            do i = 1, ng
              do j = 1, ng
                corner(i, j) = var(ie - ng + i, je + jstag + 1 + j - 1)
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(isd + i - 1, je + jstag + 1 + j - 1) = corner(i, j)
              end do
            end do
          end if

          frompe = (kk) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile
          tope = (kk - 1) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do i = 1, ng
              do j = 1, ng
                corner(i, j) = var(is + istag + i - 1, je + jstag + 1 + j - 1)
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do i = 1, ng
              do j = 1, ng
                var(ie + istag + 1 + i - 1, je + jstag + 1 + j - 1) = corner(i, j)
              end do
            end do
          end if

        end do ! kk
      end if ! rmpn

    end if ! layoux+layouty>2

  end subroutine fill_corners_domain_decomp_2d

  subroutine fill_corners_domain_decomp_3d(var, dg, bd, domain, istag, jstag)
    type(duogrid_type), intent(in) :: dg
    type(domain2d), intent(IN)         :: domain
    type(fv_grid_bounds_type), intent(IN) :: bd
    integer, intent(in) :: istag, jstag
    real, dimension(bd%isd:, bd%jsd:, 1:) :: var
    real, allocatable, dimension(:, :, :) :: corner
    ! real, dimension(1:bd%ng, 1:bd%ng) :: corner

    integer :: is, ie, js, je, isd, ied, jsd, jed, ng
    integer :: i, j, layoutx, layouty, tope, frompe, gid, k, kk
    integer :: tile(1), npes_per_tile, npz, dims(3)

    dims = shape(var)
    npz = dims(3)

    allocate (corner(1:bd%ng, 1:bd%ng, 1:npz))

    is = bd%is
    ie = bd%ie
    js = bd%js
    je = bd%je
    isd = bd%isd
    ied = bd%ied
    jsd = bd%jsd
    jed = bd%jed
    ng = bd%ng

    tile = mpp_get_tile_id(domain)
    npes_per_tile = mpp_npes()/6.
    gid = mpp_pe()
    layoutx = dg%layout(1)
    layouty = dg%layout(2)

    if (layoutx + layouty > 2) then

!!!!!!!!!!!!!!!!!!
!!!!!! WEST !!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_w .and. layouty > 1) then

        do kk = 1, layouty - 1
          tope = (kk)*layoutx + (tile(1) - 1)*npes_per_tile
          frompe = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(isd + i - 1, je - ng + j, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(isd + i - 1, jsd + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

          frompe = (kk)*layoutx + (tile(1) - 1)*npes_per_tile
          tope = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(isd + i - 1, js + jstag + j - 1, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(isd + i - 1, je + jstag + 1 + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

        end do !enddo kk

      end if !endif rmp_w

!!!!!!!!!!!!!!!!!!
!!!!!! EAST !!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_e .and. layouty > 1) then

        do kk = 1, layouty - 1

          tope = (kk)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1
          frompe = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1

          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(ie + istag + 1 + i - 1, je - ng + j, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(ie + istag + 1 + i - 1, jsd + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

          frompe = (kk)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1
          tope = (kk - 1)*layoutx + (tile(1) - 1)*npes_per_tile + layoutx - 1

          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(ie + istag + 1 + i - 1, js + jstag + j - 1, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(ie + istag + 1 + i - 1, je + jstag + 1 + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

        end do
      end if

!!!!!!!!!!!!!!!!!!
!!!!!!!SOUTH!!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_s .and. layoutx > 1) then

        do kk = 1, layoutx - 1

          tope = (kk) + (tile(1) - 1)*npes_per_tile
          frompe = (kk - 1) + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(ie - ng + i, jsd + j - 1, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(isd + i - 1, jsd + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

          frompe = (kk) + (tile(1) - 1)*npes_per_tile
          tope = (kk - 1) + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(is + istag + i - 1, jsd + j - 1, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(ie + istag + 1 + i - 1, jsd + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

        end do
      end if

!!!!!!!!!!!!!!!!!!
!!!!!!!NORTH!!!!!!
!!!!!!!!!!!!!!!!!!

      if (dg%rmp_n .and. layoutx > 1) then

        do kk = 1, layoutx - 1

          tope = (kk) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile
          frompe = (kk - 1) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile
          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(ie - ng + i, je + jstag + 1 + j - 1, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(isd + i - 1, je + jstag + 1 + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

          frompe = (kk) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile
          tope = (kk - 1) + (layoutx*(layouty - 1)) + (tile(1) - 1)*npes_per_tile

          if (gid == frompe) then
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  corner(i, j, k) = var(is + istag + i - 1, je + jstag + 1 + j - 1, k)
                end do
              end do
            end do
            call mpp_send(corner, size(corner), to_pe=tope)
          end if

          if (gid == tope) then
            call mpp_recv(corner, size(corner), from_pe=frompe)
            do k = 1, npz
              do i = 1, ng
                do j = 1, ng
                  var(ie + istag + 1 + i - 1, je + jstag + 1 + j - 1, k) = corner(i, j, k)
                end do
              end do
            end do
          end if

        end do ! kk
      end if ! rmpn

    end if ! layoux+layouty>2

    deallocate (corner)

  end subroutine fill_corners_domain_decomp_3d

  subroutine cubed_a2c_halo(npz, uatemp, vatemp, uc, vc, dg)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! HERE WE USE THE BD OF ng=4 domain, so isd,ied,jsd,jed are offsetted by one
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Purpose; Transform wind on A grid to C grid

    type(duogrid_type), intent(in), target :: dg
    integer, intent(in):: npz
    real, intent(inout), dimension(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed, npz):: uatemp, vatemp
    real, intent(inout):: uc(dg%bd%isd:dg%bd%ied + 1, dg%bd%jsd:dg%bd%jed, npz)
    real, intent(inout):: vc(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed + 1, npz)
! local:
    real v3(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)
    real ue(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges
    real ve(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges

    integer i, j, k

    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: ew, es
    real(kind=R_GRID), pointer, dimension(:, :, :)   :: vlon, vlat

    integer :: isd, ied, jsd, jed

    isd = dg%bd%isd
    ied = dg%bd%ied
    jsd = dg%bd%jsd
    jed = dg%bd%jed

    vlon => dg%vlon_ext
    vlat => dg%vlat_ext
    ew => dg%ew_ext
    es => dg%es_ext

! check if this ok
    call fill_corner_region(vatemp, dg%bd, dg, 0, 0)
    call fill_corner_region(uatemp, dg%bd, dg, 0, 0)

    do k = 1, npz
! Compute 3D wind on A grid
      do j = jsd, jed
        do i = isd, ied
          v3(1, i, j) = uatemp(i, j, k)*vlon(i, j, 1) + vatemp(i, j, k)*vlat(i, j, 1)
          v3(2, i, j) = uatemp(i, j, k)*vlon(i, j, 2) + vatemp(i, j, k)*vlat(i, j, 2)
          v3(3, i, j) = uatemp(i, j, k)*vlon(i, j, 3) + vatemp(i, j, k)*vlat(i, j, 3)
        end do
      end do

! A --> C
! Interpolate to cell edges
      do j = jsd, jed
        do i = isd + 1, ied
          ue(1, i, j) = 0.5*(v3(1, i - 1, j) + v3(1, i, j))
          ue(2, i, j) = 0.5*(v3(2, i - 1, j) + v3(2, i, j))
          ue(3, i, j) = 0.5*(v3(3, i - 1, j) + v3(3, i, j))
        end do
      end do

      do j = jsd + 1, jed
        do i = isd, ied
          ve(1, i, j) = 0.5*(v3(1, i, j - 1) + v3(1, i, j))
          ve(2, i, j) = 0.5*(v3(2, i, j - 1) + v3(2, i, j))
          ve(3, i, j) = 0.5*(v3(3, i, j - 1) + v3(3, i, j))
        end do
      end do

      do j = jsd, jed
        do i = isd + 1, ied
          uc(i, j, k) = ue(1, i, j)*ew(1, i, j, 1) + &
                        ue(2, i, j)*ew(2, i, j, 1) + &
                        ue(3, i, j)*ew(3, i, j, 1)
        end do
      end do
      do j = jsd + 1, jed
        do i = isd, ied
          vc(i, j, k) = ve(1, i, j)*es(1, i, j, 2) + &
                        ve(2, i, j)*es(2, i, j, 2) + &
                        ve(3, i, j)*es(3, i, j, 2)
        end do
      end do

    end do         ! k-loop

  end subroutine cubed_a2c_halo

  subroutine cubed_a2d_halo(npz, uatemp, vatemp, ud, vd, dg)
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! HERE WE USE THE BD OF ng=4 domain, so isd,ied,jsd,jed are offsetted by one
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
! Purpose; Transform wind on A grid to D grid

    type(duogrid_type), intent(IN), target :: dg
    integer, intent(in):: npz
    real, intent(inout), dimension(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed, npz):: uatemp, vatemp
    real, intent(inout):: ud(dg%bd%isd:dg%bd%ied, dg%bd%jsd:dg%bd%jed + 1, npz)
    real, intent(inout):: vd(dg%bd%isd:dg%bd%ied + 1, dg%bd%jsd:dg%bd%jed, npz)
! local:
    real v3(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)
    real ue(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges
    real ve(3, dg%bd%is - dg%bd%ng:dg%bd%ie + 1 + dg%bd%ng, dg%bd%js - dg%bd%ng:dg%bd%je + 1 + dg%bd%ng)    ! 3D winds at edges

    integer i, j, k

    real(kind=R_GRID), pointer, dimension(:, :, :, :) :: ew, es
    real(kind=R_GRID), pointer, dimension(:, :, :) :: vlon, vlat

    integer :: isd, ied, jsd, jed

    isd = dg%bd%isd
    ied = dg%bd%ied
    jsd = dg%bd%jsd
    jed = dg%bd%jed

    ud = -999.
    vd = -888.

    vlon => dg%vlon_ext
    vlat => dg%vlat_ext
    ew => dg%ew_ext
    es => dg%es_ext

! check if this ok
    call fill_corner_region(vatemp, dg%bd, dg, 0, 0)
    call fill_corner_region(uatemp, dg%bd, dg, 0, 0)

    do k = 1, npz
! Compute 3D wind on A grid
      do j = jsd, jed
        do i = isd, ied
          v3(1, i, j) = uatemp(i, j, k)*vlon(i, j, 1) + vatemp(i, j, k)*vlat(i, j, 1)
          v3(2, i, j) = uatemp(i, j, k)*vlon(i, j, 2) + vatemp(i, j, k)*vlat(i, j, 2)
          v3(3, i, j) = uatemp(i, j, k)*vlon(i, j, 3) + vatemp(i, j, k)*vlat(i, j, 3)
        end do
      end do

! A --> D
! Interpolate to cell edges
      do j = jsd + 1, jed
        do i = isd, ied
          ue(1, i, j) = 0.5*(v3(1, i, j - 1) + v3(1, i, j))
          ue(2, i, j) = 0.5*(v3(2, i, j - 1) + v3(2, i, j))
          ue(3, i, j) = 0.5*(v3(3, i, j - 1) + v3(3, i, j))
        end do
      end do

      do j = jsd, jed
        do i = isd + 1, ied
          ve(1, i, j) = 0.5*(v3(1, i - 1, j) + v3(1, i, j))
          ve(2, i, j) = 0.5*(v3(2, i - 1, j) + v3(2, i, j))
          ve(3, i, j) = 0.5*(v3(3, i - 1, j) + v3(3, i, j))
        end do
      end do

      do j = jsd + 1, jed
        do i = isd, ied
          ud(i, j, k) = ue(1, i, j)*es(1, i, j, 1) + &
                        ue(2, i, j)*es(2, i, j, 1) + &
                        ue(3, i, j)*es(3, i, j, 1)
        end do
      end do
      do j = jsd, jed
        do i = isd + 1, ied
          vd(i, j, k) = ve(1, i, j)*ew(1, i, j, 2) + &
                        ve(2, i, j)*ew(2, i, j, 2) + &
                        ve(3, i, j)*ew(3, i, j, 2)
        end do
      end do

    end do         ! k-loop

  end subroutine cubed_a2d_halo

  subroutine c2l_ord2_cgrid(u, v, ua, va, gridstruct, km, grid_type, bd, do_halo)
    type(fv_grid_bounds_type), intent(IN) :: bd
    integer, intent(in) :: km, grid_type
    real, intent(in) ::  u(bd%isd:bd%ied + 1, bd%jsd:bd%jed, km)
    real, intent(in) ::  v(bd%isd:bd%ied, bd%jsd:bd%jed + 1, km)
    type(fv_grid_type), intent(IN), target :: gridstruct
    logical, intent(in) :: do_halo
!
    real, intent(out):: ua(bd%isd:bd%ied, bd%jsd:bd%jed, km)
    real, intent(out):: va(bd%isd:bd%ied, bd%jsd:bd%jed, km)
!--------------------------------------------------------------
! Local
    real wu(bd%is - 1:bd%ie + 2, bd%js - 1:bd%je + 1)
    real wv(bd%is - 1:bd%ie + 1, bd%js - 1:bd%je + 2)
    real u1(bd%is - 1:bd%ie + 1), v1(bd%is - 1:bd%ie + 1)
    integer i, j, k
    integer :: is, ie, js, je

    real, dimension(:, :), pointer :: a11, a12, a21, a22
    real, dimension(:, :), pointer :: dx, dy, rdxa, rdya

    a11 => gridstruct%a11
    a12 => gridstruct%a12
    a21 => gridstruct%a21
    a22 => gridstruct%a22

    dx => gridstruct%dx
    dy => gridstruct%dy
    rdxa => gridstruct%rdxa
    rdya => gridstruct%rdya

    if (do_halo) then
      is = bd%is - 1
      ie = bd%ie + 1
      js = bd%js - 1
      je = bd%je + 1
    else
      is = bd%is
      ie = bd%ie
      js = bd%js
      je = bd%je
    end if

!$OMP parallel do default(none) shared(is,ie,js,je,km,grid_type,u,dx,v,dy,ua,va,a11,a12,a21,a22) &
!$OMP                          private(u1, v1, wu, wv)
    do k = 1, km
      if (grid_type < 4) then
        do j = js, je
          do i = is, ie + 1
            wu(i, j) = u(i, j, k)*dy(i, j)
          end do
        end do
        do j = js, je + 1
          do i = is, ie
            wv(i, j) = v(i, j, k)*dx(i, j)
          end do
        end do

        do j = js, je
          do i = is, ie
! Co-variant to Co-variant "vorticity-conserving" interpolation
            u1(i) = 2.*(wu(i, j) + wu(i + 1, j))/(dy(i, j) + dy(i + 1, j))
            v1(i) = 2.*(wv(i, j) + wv(i, j + 1))/(dx(i, j) + dx(i, j + 1))
! Cubed (cell center co-variant winds) to lat-lon:
            ua(i, j, k) = a11(i, j)*u1(i) + a12(i, j)*v1(i) !org
            va(i, j, k) = a21(i, j)*u1(i) + a22(i, j)*v1(i) !org
          end do
        end do
      else
! 2nd order:
        do j = js, je
          do i = is, ie
            va(i, j, k) = 0.5*(v(i, j, k) + v(i, j + 1, k))
            ua(i, j, k) = 0.5*(u(i, j, k) + u(i + 1, j, k))
          end do
        end do
      end if
    end do

  end subroutine c2l_ord2_cgrid

end module extchain_extract_mod
