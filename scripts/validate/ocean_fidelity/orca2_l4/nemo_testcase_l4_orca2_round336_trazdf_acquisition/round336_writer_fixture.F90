PROGRAM round336_writer_fixture
   USE par_kind, ONLY : wp
   USE par_oce
   USE dom_oce
   USE zdf_oce
   USE ldfslp
   USE round336_trazdf_writer, ONLY : r336_begin,r336_capture
   IMPLICIT NONE
   REAL(wp), ALLOCATABLE :: pt(:,:,:,:,:)
   INTEGER :: i,j,k,n,rank
   CHARACTER(LEN=16) :: arg

   CALL GET_COMMAND_ARGUMENT(1,arg); READ(arg,*) rank; narea=rank+1
   ALLOCATE(e3t_3d(jpi,jpj,jpk),e3w_1d(jpk),r3t(jpi,jpj,jpt),tmask(jpi,jpj,jpk))
   ALLOCATE(avt(jpi,jpj,jpk),avs(jpi,jpj,jpk),akz(jpi,jpj,jpk))
   ALLOCATE(pt(jpi,jpj,jpk,jpts,jpt))
   DO k=1,jpk
      e3w_1d(k)=10._wp+REAL(k,wp)
      DO j=1,jpj; DO i=1,jpi
         e3t_3d(i,j,k)=8._wp+REAL(k,wp)+0.01_wp*REAL(i+j+rank,wp)
         tmask(i,j,k)=1._wp
         avt(i,j,k)=1.e-5_wp*REAL(1+i+j+k+rank,wp)
         avs(i,j,k)=1.5e-5_wp*REAL(1+i+j+k+rank,wp)
         akz(i,j,k)=2.e-6_wp*REAL(1+k,wp)
      END DO; END DO
   END DO
   tmask(4,3,4:5)=0._wp
   avt(1,1,1)=-0._wp; avs(1,1,1)=-0._wp; akz(1,1,1)=-0._wp
   DO n=1,jpt; DO j=1,jpj; DO i=1,jpi
      r3t(i,j,n)=1.e-5_wp*REAL(i-j+n+rank,wp)
   END DO; END DO; END DO
   pt=-0._wp
   DO n=1,jpts; DO k=1,jpk; DO j=1,jpj; DO i=1,jpi
      pt(i,j,k,n,3)=2._wp+0.1_wp*REAL(i+j+k+n+rank,wp)
      pt(i,j,k,n,1)=1.e-6_wp*REAL(i-j+k+n+rank,wp)
   END DO; END DO; END DO; END DO
   CALL r336_begin(2,1,pt)
   CALL reference_solve(3,2,1,1,10800._wp,pt)
   CALL r336_capture(2,3,2,1,1,10800._wp,pt,.FALSE.,.FALSE.,.TRUE.,.TRUE.,.TRUE.)
   WRITE(*,*) 'R336_TRAZDF_FIXTURE_PASS',rank

CONTAINS

   SUBROUTINE reference_solve(Kbb,Kmm,Krhs,Kaa,p2dt,p)
      INTEGER, INTENT(in) :: Kbb,Kmm,Krhs,Kaa
      REAL(wp), INTENT(in) :: p2dt
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(inout) :: p
      REAL(wp) :: zwi(jpi,jpk),zwt(jpi,jpk),zwd(jpi,jpk),zws(jpi,jpk),zrhs
      INTEGER :: ii,jj,kk,nn
      DO jj=ntsj,ntej
         DO nn=1,jpts
            IF(nn==jp_tem) THEN
               DO kk=2,jpk; DO ii=ntsi,ntei
                  zwt(ii,kk)=avt(ii,jj,kk)+akz(ii,jj,kk)
               END DO; END DO
            ELSE
               DO kk=2,jpk; DO ii=ntsi,ntei
                  zwt(ii,kk)=avs(ii,jj,kk)+akz(ii,jj,kk)
               END DO; END DO
            ENDIF
            zwt(:,1)=0._wp
            DO kk=1,jpkm1; DO ii=ntsi,ntei
               zwi(ii,kk)=-p2dt*zwt(ii,kk)/(e3w_1d(kk)*(1._wp+r3t(ii,jj,Kmm)))
               zws(ii,kk)=-p2dt*zwt(ii,kk+1)/(e3w_1d(kk+1)*(1._wp+r3t(ii,jj,Kmm)))
               zwd(ii,kk)=e3t_3d(ii,jj,kk)*(1._wp+r3t(ii,jj,Kaa)*tmask(ii,jj,kk))- &
                  & (zwi(ii,kk)+zws(ii,kk))
            END DO; END DO
            DO ii=ntsi,ntei; zwt(ii,1)=zwd(ii,1); END DO
            DO kk=2,jpkm1; DO ii=ntsi,ntei
               zwt(ii,kk)=zwd(ii,kk)-zwi(ii,kk)*zws(ii,kk-1)/zwt(ii,kk-1)
            END DO; END DO
            DO ii=ntsi,ntei
               p(ii,jj,1,nn,Kaa)=e3t_3d(ii,jj,1)*(1._wp+r3t(ii,jj,Kbb)*tmask(ii,jj,1))* &
                  & p(ii,jj,1,nn,Kbb)+p2dt*e3t_3d(ii,jj,1)* &
                  & (1._wp+r3t(ii,jj,Kmm)*tmask(ii,jj,1))*p(ii,jj,1,nn,Krhs)
            END DO
            DO kk=2,jpkm1; DO ii=ntsi,ntei
               zrhs=e3t_3d(ii,jj,kk)*(1._wp+r3t(ii,jj,Kbb)*tmask(ii,jj,kk))* &
                  & p(ii,jj,kk,nn,Kbb)+p2dt*e3t_3d(ii,jj,kk)* &
                  & (1._wp+r3t(ii,jj,Kmm)*tmask(ii,jj,kk))*p(ii,jj,kk,nn,Krhs)
               p(ii,jj,kk,nn,Kaa)=zrhs-zwi(ii,kk)/zwt(ii,kk-1)*p(ii,jj,kk-1,nn,Kaa)
            END DO; END DO
            DO ii=ntsi,ntei
               p(ii,jj,jpkm1,nn,Kaa)=p(ii,jj,jpkm1,nn,Kaa)/zwt(ii,jpkm1)*tmask(ii,jj,jpkm1)
            END DO
            DO kk=jpk-2,1,-1; DO ii=ntsi,ntei
               p(ii,jj,kk,nn,Kaa)=(p(ii,jj,kk,nn,Kaa)-zws(ii,kk)* &
                  & p(ii,jj,kk+1,nn,Kaa))/zwt(ii,kk)*tmask(ii,jj,kk)
            END DO; END DO
         END DO
      END DO
   END SUBROUTINE reference_solve
END PROGRAM round336_writer_fixture
