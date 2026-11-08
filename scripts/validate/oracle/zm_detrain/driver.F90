! CAM6 detrainment of ZM condensate (cam_cesm2_1_rel_60,
! a03b84b7c4e34f965b115686f22a043b85739e56): the loop clubb_intr.F90:2345-2386,
! #included verbatim from block.inc (extracted by build.sh).  Only the
! state1/ptend_loc members it touches are declared here.  zmconv_microp =
! .false. (namelist_defaults_cam.xml:1643) and dlf2 = 0 (clubb_intr.F90:2327).
! usage: detrain_driver <inputs.txt> <outputs.txt>
program detrain_driver
  implicit none
  integer, parameter :: r8 = selected_real_kind(12)
  integer, parameter :: ixcldliq = 1, ixcldice = 2, ixnumliq = 3, ixnumice = 4
  type st_t
     real(r8), allocatable :: t(:,:), pdel(:,:)
  end type
  type pt_t
     real(r8), allocatable :: q(:,:,:), s(:,:)
  end type
  type(st_t) :: state1
  type(pt_t) :: ptend_loc
  character(len=512) :: fin, fout
  integer :: ncol, pver, i, k
  logical :: zmconv_microp
  real(r8) :: latice, gravit, dum1
  real(r8), allocatable :: dlf(:,:), dlf2(:,:), dlfzm(:,:), difzm(:,:), &
                           dnlfzm(:,:), dnifzm(:,:), det_s(:), det_ice(:)

  call get_command_argument(1, fin)
  call get_command_argument(2, fout)
  open(10, file=trim(fin), status='old')
  read(10,*) ncol, pver
  read(10,*) latice, gravit
  allocate(state1%t(ncol,pver), state1%pdel(ncol,pver), ptend_loc%q(ncol,pver,4), &
           ptend_loc%s(ncol,pver), dlf(ncol,pver), dlf2(ncol,pver), dlfzm(ncol,pver), &
           difzm(ncol,pver), dnlfzm(ncol,pver), dnifzm(ncol,pver), det_s(ncol), det_ice(ncol))
  do i = 1, ncol
     read(10,*) state1%t(i,:)
     read(10,*) dlf(i,:)
     read(10,*) state1%pdel(i,:)
  end do
  close(10)
  zmconv_microp = .false.
  dlf2 = 0.0_r8
  dlfzm = 0.0_r8; difzm = 0.0_r8; dnlfzm = 0.0_r8; dnifzm = 0.0_r8
  ptend_loc%q = 0.0_r8; ptend_loc%s = 0.0_r8
  det_s = 0.0_r8; det_ice = 0.0_r8

#include "block.inc"

  open(20, file=trim(fout), status='replace')
  do i = 1, ncol
     write(20,'(1000es26.17e3)') ptend_loc%q(i,:,ixcldliq)
     write(20,'(1000es26.17e3)') ptend_loc%q(i,:,ixcldice)
     write(20,'(1000es26.17e3)') ptend_loc%q(i,:,ixnumliq)
     write(20,'(1000es26.17e3)') ptend_loc%q(i,:,ixnumice)
     write(20,'(1000es26.17e3)') ptend_loc%s(i,:)
     write(20,'(2es26.17e3)') det_s(i), det_ice(i)
  end do
  close(20)
  write(*,*) 'ORACLE_OK ', ncol
end program detrain_driver
