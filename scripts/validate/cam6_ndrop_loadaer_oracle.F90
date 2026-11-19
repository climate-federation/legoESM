! CAM6 ndrop.F90 loadaer (VERBATIM lines 1860-1963) + ndrop_init voltonumb (VERBATIM 158-161) oracle,
! for tests/atmosphere/hydrostatic/unit/test_cam6_modal_aerosol_oracle.py.
! Build: gfortran -O0 -ffree-line-length-none -fno-fast-math -J /tmp cam6_ndrop_loadaer_oracle.F90 -o o && ./o < cam6_ndrop_loadaer_oracle_cases.txt
! Shim: physics_state / physics_buffer_desc are dummies; rad_cnst_get_aer_mmr / _mode_num / _aer_props return the
! case's arrays (interstitial 'a' = case values, cloud-borne 'c' = 0, user decision 2026-10-16).
module oracle_shim
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: pcols = 1, iulog = 6
  real(r8) :: pi
  integer, parameter :: maxspec = 6
  integer :: nspec_amode(4)
  real(r8), target :: raer_mass(pcols,1,maxspec), qqcw_zero(pcols,1), raer_num(pcols,1)
  real(r8) :: dens(maxspec), hyg(maxspec)
  real(r8), allocatable :: voltonumblo_amode(:), voltonumbhi_amode(:), alogsig(:), sigmag_amode(:), &
                           dgnumlo_amode(:), dgnumhi_amode(:)
  type physics_state
    integer :: lchnk = 1
  end type physics_state
  type physics_buffer_desc
    integer :: dummy = 0
  end type physics_buffer_desc
contains
  subroutine endrun(msg)
    character(len=*), intent(in) :: msg
    write(*,*) 'ENDRUN ', msg
    stop 2
  end subroutine endrun
  subroutine rad_cnst_get_aer_mmr(list_idx, m, l, phase, state, pbuf, mmr)
    integer, intent(in) :: list_idx, m, l
    character(len=1), intent(in) :: phase
    type(physics_state), intent(in) :: state
    type(physics_buffer_desc), pointer :: pbuf(:)
    real(r8), pointer :: mmr(:,:)
    if (phase == 'a') then
      mmr => raer_mass(:,:,l)
    else
      mmr => qqcw_zero
    end if
  end subroutine rad_cnst_get_aer_mmr
  subroutine rad_cnst_get_mode_num(list_idx, m, phase, state, pbuf, num)
    integer, intent(in) :: list_idx, m
    character(len=1), intent(in) :: phase
    type(physics_state), intent(in) :: state
    type(physics_buffer_desc), pointer :: pbuf(:)
    real(r8), pointer :: num(:,:)
    if (phase == 'a') then
      num => raer_num
    else
      num => qqcw_zero
    end if
  end subroutine rad_cnst_get_mode_num
  subroutine rad_cnst_get_aer_props(list_idx, m, l, density_aer, hygro_aer)
    integer, intent(in) :: list_idx, m, l
    real(r8), intent(out) :: density_aer, hygro_aer
    density_aer = dens(l)
    hygro_aer = hyg(l)
  end subroutine rad_cnst_get_aer_props
  subroutine set_voltonumb(ntot_amode)
    integer, intent(in) :: ntot_amode
    integer :: m
    allocate(voltonumblo_amode(ntot_amode), voltonumbhi_amode(ntot_amode), alogsig(ntot_amode))
    do m = 1, ntot_amode
      alogsig(m)     = log(sigmag_amode(m))
      voltonumblo_amode(m) = 1._r8 / ( (pi/6._r8)*                          &
                             (dgnumlo_amode(m)**3._r8)*exp(4.5_r8*alogsig(m)**2._r8) )
      voltonumbhi_amode(m) = 1._r8 / ( (pi/6._r8)*                          &
                             (dgnumhi_amode(m)**3._r8)*exp(4.5_r8*alogsig(m)**2._r8) )
    end do
  end subroutine set_voltonumb

subroutine loadaer( &
   state, pbuf, istart, istop, k, &
   m, cs, phase, naerosol, &
   vaerosol, hygro)

   ! return aerosol number, volume concentrations, and bulk hygroscopicity

   ! input arguments
   type(physics_state), target, intent(in) :: state
   type(physics_buffer_desc),   pointer    :: pbuf(:)

   integer,  intent(in) :: istart      ! start column index (1 <= istart <= istop <= pcols)
   integer,  intent(in) :: istop       ! stop column index  
   integer,  intent(in) :: m           ! mode index
   integer,  intent(in) :: k           ! level index
   real(r8), intent(in) :: cs(:,:)     ! air density (kg/m3)
   integer,  intent(in) :: phase       ! phase of aerosol: 1 for interstitial, 2 for cloud-borne, 3 for sum

   ! output arguments
   real(r8), intent(out) :: naerosol(:)  ! number conc (1/m3)
   real(r8), intent(out) :: vaerosol(:)  ! volume conc (m3/m3)
   real(r8), intent(out) :: hygro(:)     ! bulk hygroscopicity of mode

   ! internal
   integer  :: lchnk               ! chunk identifier

   real(r8), pointer :: raer(:,:) ! interstitial aerosol mass, number mixing ratios
   real(r8), pointer :: qqcw(:,:) ! cloud-borne aerosol mass, number mixing ratios
   real(r8) :: specdens, spechygro

   real(r8) :: vol(pcols) ! aerosol volume mixing ratio
   integer  :: i, l
   !-------------------------------------------------------------------------------

   lchnk = state%lchnk

   do i = istart, istop
      vaerosol(i) = 0._r8
      hygro(i)    = 0._r8
   end do

   do l = 1, nspec_amode(m)

      call rad_cnst_get_aer_mmr(0, m, l, 'a', state, pbuf, raer)
      call rad_cnst_get_aer_mmr(0, m, l, 'c', state, pbuf, qqcw)
      call rad_cnst_get_aer_props(0, m, l, density_aer=specdens, hygro_aer=spechygro)

      if (phase == 3) then
         do i = istart, istop
            vol(i) = max(raer(i,k) + qqcw(i,k), 0._r8)/specdens
         end do
      else if (phase == 2) then
         do i = istart, istop
            vol(i) = max(qqcw(i,k), 0._r8)/specdens
         end do
      else if (phase == 1) then
         do i = istart, istop
            vol(i) = max(raer(i,k), 0._r8)/specdens
         end do
      else
         write(iulog,*)'phase=',phase,' in loadaer'
         call endrun('phase error in loadaer')
      end if

      do i = istart, istop
         vaerosol(i) = vaerosol(i) + vol(i)
         hygro(i)    = hygro(i) + vol(i)*spechygro
      end do

   end do

   do i = istart, istop
      if (vaerosol(i) > 1.0e-30_r8) then   ! +++xl add 8/2/2007
         hygro(i)    = hygro(i)/(vaerosol(i))
         vaerosol(i) = vaerosol(i)*cs(i,k)
      else
         hygro(i)    = 0.0_r8
         vaerosol(i) = 0.0_r8
      end if
   end do

   ! aerosol number
   call rad_cnst_get_mode_num(0, m, 'a', state, pbuf, raer)
   call rad_cnst_get_mode_num(0, m, 'c', state, pbuf, qqcw)
   if (phase == 3) then
      do i = istart, istop
         naerosol(i) = (raer(i,k) + qqcw(i,k))*cs(i,k)
      end do
   else if (phase == 2) then
      do i = istart, istop
         naerosol(i) = qqcw(i,k)*cs(i,k)
      end do
   else
      do i = istart, istop
         naerosol(i) = raer(i,k)*cs(i,k)
      end do
   end if
   ! adjust number so that dgnumlo < dgnum < dgnumhi
   do i = istart, istop
      naerosol(i) = max(naerosol(i), vaerosol(i)*voltonumbhi_amode(m))
      naerosol(i) = min(naerosol(i), vaerosol(i)*voltonumblo_amode(m))
   end do

end subroutine loadaer
end module oracle_shim

program loadaer_oracle
  use oracle_shim
  implicit none
  integer :: m, l, ncase, k, phase
  real(r8) :: cs(pcols,1), naerosol(pcols), vaerosol(pcols), hygro(pcols)
  type(physics_state) :: state
  type(physics_buffer_desc), pointer :: pbuf(:) => null()
  read(*,*) pi
  allocate(sigmag_amode(4), dgnumlo_amode(4), dgnumhi_amode(4))
  read(*,*) (sigmag_amode(m), m=1,4)
  read(*,*) (dgnumlo_amode(m), m=1,4)
  read(*,*) (dgnumhi_amode(m), m=1,4)
  call set_voltonumb(4)
  write(*,'(*(ES25.17E3))') (voltonumblo_amode(m), m=1,4), (voltonumbhi_amode(m), m=1,4)
  qqcw_zero = 0._r8
  read(*,*) ncase
  do k = 1, ncase
    read(*,*) m, nspec_amode(m), phase, cs(1,1), raer_num(1,1)
    read(*,*) (raer_mass(1,1,l), l=1,nspec_amode(m))
    read(*,*) (dens(l), l=1,nspec_amode(m))
    read(*,*) (hyg(l), l=1,nspec_amode(m))
    call loadaer(state, pbuf, 1, 1, 1, m, cs, phase, naerosol, vaerosol, hygro)
    write(*,'(*(ES25.17E3))') naerosol(1), vaerosol(1), hygro(1)
  end do
end program loadaer_oracle
