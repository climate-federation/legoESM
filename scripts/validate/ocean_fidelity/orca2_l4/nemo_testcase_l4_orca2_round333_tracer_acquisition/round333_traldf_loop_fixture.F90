PROGRAM round333_traldf_loop_fixture
   USE, INTRINSIC :: iso_fortran_env, ONLY : int64
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi,jpj,jpk,jpts
   USE round333_traldf_writer
   IMPLICIT NONE
   REAL(wp) :: seed(jpi,jpj,jpk,jpts,3),ref(jpi,jpj,jpk,jpts),obs(jpi,jpj,jpk,jpts)
   INTEGER(int64) :: ref_bits(jpi*jpj*jpk*jpts),obs_bits(jpi*jpj*jpk*jpts)
   INTEGER :: i,j,k,n

   DO n=1,jpts; DO k=1,jpk; DO j=1,jpj; DO i=1,jpi
      seed(i,j,k,n,1)=REAL(1000*n+100*k+10*j+i,wp)/97._wp
      seed(i,j,k,n,2)=0._wp
      seed(i,j,k,n,3)=seed(i,j,k,n,1)
   END DO; END DO; END DO; END DO
   seed(1,1,1,1,1)=-0._wp
   CALL reference_kernel(seed,ref)
   CALL r333_ldf_arm(2,3,3,2,1)
   CALL r333_ldf_begin(2,3,2,1,seed)
   CALL observed_kernel(seed,obs)
   ref_bits=TRANSFER(ref,ref_bits); obs_bits=TRANSFER(obs,obs_bits)
   IF(ANY(ref_bits/=obs_bits)) ERROR STOP 'patched traldf loop fixture changed arithmetic'
   IF(.NOT.ANY(r333_horizontal_seen==1._wp) .OR. .NOT.ANY(r333_rhs_seen==1._wp)) &
      & ERROR STOP 'patched traldf loop fixture did not execute observers'
   WRITE(*,'(A)') 'R333_TRALDF_PATCHED_LOOP_FIXTURE_PASS independent-reference'

CONTAINS

   SUBROUTINE reference_kernel(pt,rhs)
      REAL(wp), INTENT(in) :: pt(jpi,jpj,jpk,jpts,3)
      REAL(wp), INTENT(out) :: rhs(jpi,jpj,jpk,jpts)
      REAL(wp) :: dit(jpi,jpj,2),djt(jpi,jpj,2),dkt(jpi,jpj,2)
      REAL(wp) :: fu(jpi,jpj),fv(jpi,jpj),fw(jpi,jpj),upper
      INTEGER :: i,j,k,n,ik,ikp1,tmp
      rhs=pt(:,:,:,:,1); dit=0._wp; djt=0._wp; dkt=0._wp; fw=0._wp
      DO n=1,jpts
         ik=1; ikp1=2
         DO k=1,jpk-1
            IF(k>1) THEN; tmp=ik; ik=ikp1; ikp1=tmp; ENDIF
            DO j=1,jpj-1; DO i=1,jpi-1
               dit(i,j,ik)=pt(i+1,j,k,n,3)-pt(i,j,k,n,3)
               djt(i,j,ik)=pt(i,j+1,k,n,3)-pt(i,j,k,n,3)
               IF(k<jpk) THEN
                  dit(i,j,ikp1)=pt(i+1,j,k+1,n,3)-pt(i,j,k+1,n,3)
                  djt(i,j,ikp1)=pt(i,j+1,k+1,n,3)-pt(i,j,k+1,n,3)
               ENDIF
            END DO; END DO
            DO j=1,jpj; DO i=1,jpi
               dkt(i,j,ik)=MERGE(pt(i,j,k,n,3)-pt(i,j,k+1,n,3),0._wp,k<jpk)
               dkt(i,j,ikp1)=MERGE(pt(i,j,k,n,3)-pt(i,j,k+1,n,3),0._wp,k<jpk)
            END DO; END DO
            DO j=1,jpj-1; DO i=1,jpi-1
               fu(i,j)=2._wp*dit(i,j,ik)+0.125_wp*(dkt(i+1,j,ik)+dkt(i,j,ikp1))
               fv(i,j)=3._wp*djt(i,j,ik)+0.25_wp*(dkt(i,j+1,ik)+dkt(i,j,ikp1))
            END DO; END DO
            DO j=2,jpj-1; DO i=2,jpi-1
               upper=0.5_wp*(dit(i,j,ik)+djt(i,j,ik))
               rhs(i,j,k,n)=rhs(i,j,k,n)+(((fu(i,j)-fu(i-1,j)) + &
                  & (fv(i,j)-fv(i,j-1)))+(fw(i,j)-upper))*0.125_wp
               fw(i,j)=upper
            END DO; END DO
         END DO
      END DO
   END SUBROUTINE reference_kernel

   SUBROUTINE observed_kernel(pt,rhs)
      REAL(wp), INTENT(in) :: pt(jpi,jpj,jpk,jpts,3)
      REAL(wp), INTENT(out) :: rhs(jpi,jpj,jpk,jpts)
      REAL(wp) :: dit(jpi,jpj,2),djt(jpi,jpj,2),dkt(jpi,jpj,2)
      REAL(wp) :: fu(jpi,jpj),fv(jpi,jpj),fw(jpi,jpj),upper
      INTEGER :: i,j,k,n,ik,ikp1,tmp
      rhs=pt(:,:,:,:,1); dit=0._wp; djt=0._wp; dkt=0._wp; fw=0._wp
      DO n=1,jpts
         ik=1; ikp1=2
         DO k=1,jpk-1
            IF(k>1) THEN; tmp=ik; ik=ikp1; ikp1=tmp; ENDIF
            DO j=1,jpj-1; DO i=1,jpi-1
               dit(i,j,ik)=pt(i+1,j,k,n,3)-pt(i,j,k,n,3)
               djt(i,j,ik)=pt(i,j+1,k,n,3)-pt(i,j,k,n,3)
               IF(k<jpk) THEN
                  dit(i,j,ikp1)=pt(i+1,j,k+1,n,3)-pt(i,j,k+1,n,3)
                  djt(i,j,ikp1)=pt(i,j+1,k+1,n,3)-pt(i,j,k+1,n,3)
               ENDIF
            END DO; END DO
            DO j=1,jpj; DO i=1,jpi
               dkt(i,j,ik)=MERGE(pt(i,j,k,n,3)-pt(i,j,k+1,n,3),0._wp,k<jpk)
               dkt(i,j,ikp1)=MERGE(pt(i,j,k,n,3)-pt(i,j,k+1,n,3),0._wp,k<jpk)
            END DO; END DO
            CALL r333_ldf_grad(n,k,1,jpi-1,1,jpj-1,dit(1:jpi-1,1:jpj-1,ik), &
               & djt(1:jpi-1,1:jpj-1,ik),1,jpi,1,jpj,dkt(:,:,ik),dkt(:,:,ikp1))
            DO j=1,jpj-1; DO i=1,jpi-1
               fu(i,j)=2._wp*dit(i,j,ik)+0.125_wp*(dkt(i+1,j,ik)+dkt(i,j,ikp1))
               fv(i,j)=3._wp*djt(i,j,ik)+0.25_wp*(dkt(i,j+1,ik)+dkt(i,j,ikp1))
            END DO; END DO
            CALL r333_ldf_horizontal(n,k,1,jpi-1,1,jpj-1,fu(1:jpi-1,1:jpj-1),fv(1:jpi-1,1:jpj-1))
            CALL r333_ldf_vertical_before(n,k,2,jpi-1,2,jpj-1,fw(2:jpi-1,2:jpj-1), &
               & rhs(2:jpi-1,2:jpj-1,k,n))
            DO j=2,jpj-1; DO i=2,jpi-1
               upper=0.5_wp*(dit(i,j,ik)+djt(i,j,ik))
               rhs(i,j,k,n)=rhs(i,j,k,n)+(((fu(i,j)-fu(i-1,j)) + &
                  & (fv(i,j)-fv(i,j-1)))+(fw(i,j)-upper))*0.125_wp
               fw(i,j)=upper
            END DO; END DO
            CALL r333_ldf_vertical_after(n,k,2,jpi-1,2,jpj-1,fw(2:jpi-1,2:jpj-1), &
               & rhs(2:jpi-1,2:jpj-1,k,n))
         END DO
      END DO
   END SUBROUTINE observed_kernel
END PROGRAM round333_traldf_loop_fixture
