! Four analytic stability functions copied verbatim from MMM-physics
! sf_sfclayrev.F90 (WRF/MPAS revised MM5 surface layer, public domain);
! prints the oracle rows of tests/unit/test_wrf_revmm5_stability.py.
! Build (double, literals promoted as in an -r8 WRF build):
!   gfortran -fdefault-real-8 -ffree-line-length-none oracle.f90 -o oracle && ./oracle
module m
 implicit none
 integer, parameter :: kind_phys = 8
contains
 real(kind=kind_phys) function psim_stable_full(zolf)
 real(kind=kind_phys),intent(in):: zolf
 psim_stable_full=-6.1*log(zolf+(1+zolf**2.5)**(1./2.5))

 return
 end function psim_stable_full
 real(kind=kind_phys) function psih_stable_full(zolf)
 real(kind=kind_phys),intent(in):: zolf
 psih_stable_full=-5.3*log(zolf+(1+zolf**1.1)**(1./1.1))

 return
 end function psih_stable_full
 real(kind=kind_phys) function psim_unstable_full(zolf)
 real(kind=kind_phys),intent(in):: zolf
 real(kind=kind_phys):: psimc,psimk,x,y,ym
 x=(1.-16.*zolf)**.25
 psimk=2*ALOG(0.5*(1+X))+ALOG(0.5*(1+X*X))-2.*ATAN(X)+2.*ATAN(1.)

 ym=(1.-10.*zolf)**0.33
 psimc=(3./2.)*log((ym**2.+ym+1.)/3.)-sqrt(3.)*ATAN((2.*ym+1)/sqrt(3.))+4.*ATAN(1.)/sqrt(3.)

 psim_unstable_full=(psimk+zolf**2*(psimc))/(1+zolf**2.)

 return
 end function psim_unstable_full
 real(kind=kind_phys) function psih_unstable_full(zolf)
 real(kind=kind_phys),intent(in):: zolf
 real(kind=kind_phys):: psihc,psihk,y,yh
 y=(1.-16.*zolf)**.5
 psihk=2.*log((1+y)/2.)

 yh=(1.-34.*zolf)**0.33
 psihc=(3./2.)*log((yh**2.+yh+1.)/3.)-sqrt(3.)*ATAN((2.*yh+1)/sqrt(3.))+4.*ATAN(1.)/sqrt(3.)

 psih_unstable_full=(psihk+zolf**2*(psihc))/(1+zolf**2.)

 return
 end function psih_unstable_full
end module m
program p
 use m
 real(kind_phys) :: z(8) = (/0.001_kind_phys, 0.01_kind_phys, 0.1_kind_phys, 0.5_kind_phys, 1._kind_phys, 2._kind_phys, 5._kind_phys, 10._kind_phys/)
 integer :: i
 do i=1,8
  write(*,'(5es25.16)') z(i), psim_stable_full(z(i)), psih_stable_full(z(i)), psim_unstable_full(-z(i)), psih_unstable_full(-z(i))
 end do
end program
