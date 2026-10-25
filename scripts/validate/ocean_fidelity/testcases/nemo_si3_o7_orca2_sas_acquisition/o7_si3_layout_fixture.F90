PROGRAM o7_si3_layout_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY: real64
   USE par_oce
   USE par_ice
   USE o7_si3_writer
   IMPLICIT NONE
   INTEGER :: kt, i, j, l
   REAL(wp), DIMENSION(jpi,jpj) :: a, b, c, d, e, f, g, h
   REAL(wp), DIMENSION(jpi,jpj,jpl) :: x, y, z, p, q, r
   REAL(wp), DIMENSION(ntei-ntsi+1,ntej-ntsj+1) :: ai, bi, ci, di, ei, fi, gi, hi
   REAL(wp), DIMENSION(ntei-ntsi+1,ntej-ntsj+1,jpl) :: xi, yi, zi, pi, qi, ri

   DO l=1,jpl; DO j=1,jpj; DO i=1,jpi
      x(i,j,l)=REAL(i+10*j+100*l,wp); y(i,j,l)=-x(i,j,l)
      z(i,j,l)=0.5_wp*x(i,j,l); p(i,j,l)=2._wp*x(i,j,l)
      q(i,j,l)=3._wp*x(i,j,l); r(i,j,l)=4._wp*x(i,j,l)
   END DO; END DO; END DO
   DO j=1,jpj; DO i=1,jpi
      a(i,j)=REAL(i+10*j,wp); b(i,j)=-a(i,j); c(i,j)=0.5_wp*a(i,j)
      d(i,j)=2._wp*a(i,j); e(i,j)=3._wp*a(i,j); f(i,j)=4._wp*a(i,j)
      g(i,j)=5._wp*a(i,j); h(i,j)=6._wp*a(i,j)
   END DO; END DO
   a(ntsi,ntsj)=-0._wp
   ai=a(ntsi:ntei,ntsj:ntej); bi=b(ntsi:ntei,ntsj:ntej); ci=c(ntsi:ntei,ntsj:ntej)
   di=d(ntsi:ntei,ntsj:ntej); ei=e(ntsi:ntei,ntsj:ntej); fi=f(ntsi:ntei,ntsj:ntej)
   gi=g(ntsi:ntei,ntsj:ntej); hi=h(ntsi:ntei,ntsj:ntej)
   xi=x(ntsi:ntei,ntsj:ntej,:); yi=y(ntsi:ntei,ntsj:ntej,:); zi=z(ntsi:ntei,ntsj:ntej,:)
   pi=p(ntsi:ntei,ntsj:ntej,:); qi=q(ntsi:ntei,ntsj:ntej,:); ri=r(ntsi:ntei,ntsj:ntej,:)

   DO kt=1,2
      CALL o7_set_context(kt,2,1,0,1,1,5400._wp,1._wp/5400._wp,1026._wp,3991.86795711963_wp)
      CALL o7_set_coeffs(1.4e-3_wp,1.4e-3_wp,1.4e-3_wp)
      CALL o7_blk1_entry(ai,bi,ci,di,ei,fi,gi,hi)
      CALL o7_blk1_exit(ai,bi,ci,di,ei,fi,gi,a,b)
      CALL o7_tau_after(a,b)
      CALL o7_alb_entry(x,y,z,p,q,r,ai,a)
      CALL o7_alb_exit(xi)
      CALL o7_blk2_entry(x,y,z,pi,q,ai,bi,ci,di,ei,fi,gi,hi,ai,bi,ci,di,ei,fi,gi,hi,ai,bi,ci,di)
      CALL o7_blk2_exit(xi,yi,zi,pi,qi,ri,xi,ai,yi,zi,pi,qi,ri,xi,yi,zi,bi,ci,di,ei,fi,gi,hi,ai,bi,ci,di,ei,fi,gi)
      CALL o7_flx_entry(a,b,c,d,e,f,gi,hi,ai,bi,ci,di,ei,fi,gi,hi,ai,bi)
      CALL o7_flx_exit(ai,bi,ci,di)
      CALL o7_alb_entry(x,y,z,p,q,r,ai,a)
      CALL o7_alb_exit(xi)
      CALL o7_finish_step
   END DO
END PROGRAM o7_si3_layout_fixture
