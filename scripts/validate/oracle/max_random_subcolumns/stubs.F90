! Minimal CAM infrastructure stubs for the McICA oracle (only what mcica_subcol_gen_sw uses),
! plus the module holding the harness-supplied uniforms.
module cam_abortutils
  implicit none
contains
  subroutine endrun(msg)
    character(len=*), intent(in), optional :: msg
    if (present(msg)) print *, 'ENDRUN: ', trim(msg)
    stop 2
  end subroutine endrun
end module cam_abortutils

module mcica_inject_data
  use shr_kind_mod, only: r8 => shr_kind_r8
  implicit none
  logical :: use_inject = .false.
  real(r8), allocatable :: cdf_inject_sw(:,:,:)
end module mcica_inject_data
