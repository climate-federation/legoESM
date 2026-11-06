c     ORACLE: A.W. Vreman's reference implementation, downloaded verbatim from
c     http://www.vremanresearch.nl/Vreman_Subgridmodel_Fortran.txt (2026-10-19).
c     AUDIT EDITS (harness only, marked "AUDIT"): the template's pseudo-loop
c     "do n= ...." becomes "do n=1,npts"; arrays/vol/eddyv become dummy args;
c     the isotropic del**2 widths are replaced by the anisotropic h1**2,h2**2,
c     h3**2 exactly as the author's own comment prescribes; coef is passed in
c     (author hard-codes csma=0.17 -> coef=2.5*csma**2) so both sides share c.
c     The b_ij / abeta / bbeta / threshold / sqrt lines are unchanged.
       subroutine eddyvis(npts,du1dx1,du1dx2,du1dx3,du2dx1,du2dx2,
     $   du2dx3,du3dx1,du3dx2,du3dx3,h1,h2,h3,coefin,eddyv)
       implicit none
       integer npts
       real*8 du1dx1(npts),du1dx2(npts),du1dx3(npts)
       real*8 du2dx1(npts),du2dx2(npts),du2dx3(npts)
       real*8 du3dx1(npts),du3dx2(npts),du3dx3(npts)
       real*8 h1(npts),h2(npts),h3(npts),eddyv(npts),coefin
       integer n
       real*8 csma,coef,del,d1,d2,d3
       real*8 d1v1,d2v1,d3v1,d1v2,d2v2,d3v2,d1v3,d2v3,d3v3
       real*8 b11,b12,b13,b22,b23,b33
       real*8 abeta,bbeta
c      AUDIT: coef from caller (author: csma=0.17d0; coef=2.5d0*csma*csma)
       coef=coefin
       do n=1,npts
c      AUDIT: anisotropic form per the author's comment (d_i = h_i**2)
        d1=h1(n)*h1(n)
        d2=h2(n)*h2(n)
        d3=h3(n)*h3(n)

        d1v1=du1dx1(n)
        d2v1=du1dx2(n)
        d3v1=du1dx3(n)        

        d1v2=du2dx1(n)
        d2v2=du2dx2(n)
        d3v2=du2dx3(n)       

        d1v3=du3dx1(n)
        d2v3=du3dx2(n)
        d3v3=du3dx3(n)

        b11=d1*d1v1*d1v1+d2*d2v1*d2v1+d3*d3v1*d3v1
        b12=d1*d1v1*d1v2+d2*d2v1*d2v2+d3*d3v1*d3v2        
        b13=d1*d1v1*d1v3+d2*d2v1*d2v3+d3*d3v1*d3v3
        b22=d1*d1v2*d1v2+d2*d2v2*d2v2+d3*d3v2*d3v2
        b23=d1*d1v2*d1v3+d2*d2v2*d2v3+d3*d3v2*d3v3
        b33=d1*d1v3*d1v3+d2*d2v3*d2v3+d3*d3v3*d3v3

        abeta=d1v1**2+d1v2**2+d1v3**2+
     $        d2v1**2+d2v2**2+d2v3**2+
     $        d3v1**2+d3v2**2+d3v3**2        
        bbeta=b11*b22-(b12**2)+b11*b33-(b13**2)+b22*b33-(b23**2)

        if (bbeta.lt.1d-12) then
         eddyv(n)=0d0
        else
         eddyv(n)=coef*sqrt(bbeta/abeta)
        end if
       end do
       return
       end

c     AUDIT driver: stream binary I/O. Input file: int32 npts, then 12 arrays of
c     npts float64 (a11 a12 a13 a21 a22 a23 a31 a32 a33 h1 h2 h3; a_cd=du_c/dx_d),
c     then float64 coef. Output: npts float64 eddyv.
       program drv
       implicit none
       integer npts, i
       real*8, allocatable :: a(:,:), ev(:)
       real*8 coef
       character(len=512) fin, fout
       call get_command_argument(1, fin)
       call get_command_argument(2, fout)
       open(10, file=fin, access='stream', form='unformatted')
       read(10) npts
       allocate(a(npts,12), ev(npts))
       do i=1,12
        read(10) a(:,i)
       end do
       read(10) coef
       close(10)
       call eddyvis(npts,a(:,1),a(:,2),a(:,3),a(:,4),a(:,5),a(:,6),
     $   a(:,7),a(:,8),a(:,9),a(:,10),a(:,11),a(:,12),coef,ev)
       open(11, file=fout, access='stream', form='unformatted',
     $   status='replace')
       write(11) ev
       close(11)
       print *, 'VREMAN_ORACLE_DONE', npts, coef
       end
