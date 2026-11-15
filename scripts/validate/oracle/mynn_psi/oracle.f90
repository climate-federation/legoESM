! Oracle for legoESM bulk_flux "mynn_sfc": the four full MYNN-SFC psi functions
! copied verbatim from CCPP module_sf_mynnsfc_driver.F90 (psi_opt = 0), wrapped in
! a driver that prints zeta, psim, psih on a fixed grid.  Build:
!   gfortran -O0 -ffree-line-length-none -fdefault-real-8 oracle.f90 -o oracle_r8   (_ORACLE_R8_*)
!   gfortran -O0 -ffree-line-length-none oracle.f90 -o oracle_r4lit                   (_ORACLE_R4LIT_*)
module mynn_psi
 implicit none
 integer, parameter :: kind_phys = 8
 real(kind_phys), parameter :: one = 1.0_kind_phys, two = 2.0_kind_phys, &
      ten = 10.0_kind_phys, four = 4.0_kind_phys
 real(kind_phys), parameter :: sqrt3   = 1.7320508075688773
 real(kind_phys), parameter :: atan1   = 0.785398163397     !in radians
 real(kind_phys), parameter :: p333    = 1.0/3.0
 contains
 real(kind_phys) function psim_stable_full(zolf)

    real(kind_phys) :: zolf

    !psim_stable_full=-6.1*log(zolf+(1+zolf**2.5)**(1./2.5))
    psim_stable_full=-6.1_kind_phys*log(zolf+(one+zolf**2.5_kind_phys)**0.4_kind_phys)

 end function

!>\ingroup mynn_sfc
 real(kind_phys) function psih_stable_full(zolf)

    real(kind_phys) :: zolf

    !psih_stable_full=-5.3*log(zolf+(1+zolf**1.1)**(1./1.1))
    psih_stable_full=-5.3_kind_phys*log(zolf+(one+zolf**1.1_kind_phys)**0.9090909090909090909_kind_phys)

 end function

!>\ingroup mynn_sfc
 real(kind_phys) function psim_unstable_full(zolf)

    real(kind_phys) :: zolf,x,ym,psimc,psimk

    x=(one-16._kind_phys*zolf)**.25_kind_phys
    !psimk=2*ALOG(0.5*(1+X))+ALOG(0.5*(1+X*X))-2.*ATAN(X)+2.*ATAN(1.)
    !psimk=2.*ALOG(0.5*(1+X))+ALOG(0.5*(1+X*X))-2.*ATAN(X)+2.*atan1
    psimk=two*LOG(0.5_kind_phys*(one+X))+LOG(0.5_kind_phys*(one+X*X))-two*ATAN(X)+two*atan1

    ym=(one-ten*zolf)**p333
    !psimc=(3./2.)*log((ym**2.+ym+1.)/3.)-sqrt(3.)*ATAN((2.*ym+1)/sqrt(3.))+4.*ATAN(1.)/sqrt(3.)
    psimc=1.5_kind_phys*log((ym**2 + ym+one)*p333)-sqrt3*ATAN((two*ym+1)/sqrt3)+four*atan1/sqrt3

    psim_unstable_full=(psimk+zolf**2*(psimc))/(1+zolf**2)

 end function

!>\ingroup mynn_sfc
 real(kind_phys) function psih_unstable_full(zolf)

    real(kind_phys) :: zolf,y,yh,psihc,psihk

    y=(one-16._kind_phys*zolf)**.5_kind_phys
    !psihk=2.*log((1+y)/2.)
    psihk=two*log((one+y)*0.5_kind_phys)

    yh=(one-34._kind_phys*zolf)**p333
    !psihc=(3./2.)*log((yh**2.+yh+1.)/3.)-sqrt(3.)*ATAN((2.*yh+1)/sqrt(3.))+4.*ATAN(1.)/sqrt(3.)
    psihc=1.5_kind_phys*log((yh**2+yh+one)*p333)-sqrt3*ATAN((two*yh+one)/sqrt3)+four*atan1/sqrt3

    psih_unstable_full=(psihk+zolf**2*(psihc))/(one+zolf**2)

 end function

 ! ==================================================================
end module mynn_psi

program oracle
 use mynn_psi
 implicit none
 real(kind_phys) :: z(12) = [1.0d-3, 0.01d0, 0.1d0, 0.37d0, 1.0d0, 2.5d0, 7.0d0, 9.9d0, 1.0d-6, 0.5d0, 3.0d0, 10.0d0]
 integer :: i
 do i = 1, size(z)
   write(*,'(A,3ES26.17)') 'S', z(i), psim_stable_full(z(i)), psih_stable_full(z(i))
   write(*,'(A,3ES26.17)') 'U', -z(i), psim_unstable_full(-z(i)), psih_unstable_full(-z(i))
 end do
end program oracle
