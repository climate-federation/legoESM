! Drive CAM6 conv_water_4rad (compiled unmodified) on columns read from
! cw_in.bin and write totg_liq, totg_ice to cw_out.bin.
! cw_in.bin (stream, little endian): int32 ncol, int32 nlev, then nine
! float64 arrays in C order (ncol, nlev): ls_liq, ls_ice, pdel, ast,
! dp_frac, dp_icwmr, sh_frac, sh_icwmr, rei.  Namelist cw_nl.txt sets
! conv_water_in_rad and conv_water_frac_limit.
program cw_oracle
  use shr_kind_mod, only: r8 => shr_kind_r8
  use ppgrid, only: pcols, pver
  use physics_buffer, only: physics_buffer_desc, pbuf_get_index, store
  use physics_types, only: physics_state
  use conv_water, only: conv_water_readnl, conv_water_register, &
                        conv_water_init, conv_water_4rad
  implicit none
  type(physics_state), target :: state
  type(physics_buffer_desc), pointer :: pbuf(:)
  real(r8) :: totg_liq(pcols, pver), totg_ice(pcols, pver)
  real(r8), allocatable :: a(:,:,:)
  integer :: ncol, nlev, n, i, k, u
  character(len=8), parameter :: fld(6) = (/ 'AST     ', 'DP_FRAC ', 'ICWMRDP ', &
                                            'SH_FRAC ', 'ICWMRSH ', 'REI     ' /)

  allocate(pbuf(1))
  call conv_water_readnl('cw_nl.txt')
  call conv_water_register()
  call conv_water_init()

  open(newunit=u, file='cw_in.bin', access='stream', form='unformatted', status='old')
  read(u) ncol, nlev
  if (nlev /= pver .or. ncol > pcols) stop 2
  allocate(a(nlev, ncol, 9))
  read(u) a
  close(u)

  state%ncol = ncol
  state%q = 0._r8
  state%pdel = 0._r8
  do i = 1, ncol
    do k = 1, nlev
      state%q(i, k, 2) = a(k, i, 1)
      state%q(i, k, 3) = a(k, i, 2)
      state%pdel(i, k) = a(k, i, 3)
      do n = 1, 6
        store(i, k, pbuf_get_index(trim(fld(n)))) = a(k, i, 3 + n)
      end do
    end do
  end do
  store(:, :, pbuf_get_index('FICE')) = 0._r8

  totg_liq = 0._r8
  totg_ice = 0._r8
  call conv_water_4rad(state, pbuf, totg_liq, totg_ice)

  open(newunit=u, file='cw_out.bin', access='stream', form='unformatted', status='replace')
  write(u) ((totg_liq(i, k), k = 1, nlev), i = 1, ncol)
  write(u) ((totg_ice(i, k), k = 1, nlev), i = 1, ncol)
  close(u)
end program cw_oracle
