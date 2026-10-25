PROGRAM round333_writer_fixture
   USE, INTRINSIC :: ieee_arithmetic, ONLY : ieee_is_negative
   USE dom_oce
   USE round332_writer, ONLY : r333_tracer_open,r333_tracer_rhs,r333_tracer_finish
   IMPLICIT NONE
   INTEGER :: rank,row
   CHARACTER(LEN=8) :: argument
   REAL(wp) :: pT(jpi,jpj,jpk),pS(jpi,jpj,jpk)
   REAL(wp) :: pbbT(jpi,jpj,jpk),pbbS(jpi,jpj,jpk)
   REAL(wp) :: pmmT(jpi,jpj,jpk),pmmS(jpi,jpj,jpk)
   REAL(wp) :: paaT(jpi,jpj,jpk),paaS(jpi,jpj,jpk)
   REAL(wp) :: rbb(jpi,jpj),rmm(jpi,jpj),raa(jpi,jpj)

   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   narea=rank+1
   pT=-0._wp; pS=-0._wp
   pT(3,2,2)=REAL(rank+1,wp); pS(3,2,2)=REAL(rank+11,wp)
   CALL r333_tracer_open(2,3,3,2,1,1,pT,pS)
   DO row=1,5
      pT(3,2,2)=pT(3,2,2)+REAL(row,wp)
      pS(3,2,2)=pS(3,2,2)+REAL(row,wp)
      CALL r333_tracer_rhs(pT,pS)
   END DO
   pbbT=pT; pbbS=pS; pmmT=pT; pmmS=pS; paaT=pT; paaS=pS
   rbb=-0._wp; rmm=-0._wp; raa=-0._wp
   rbb(3,2)=REAL(rank+21,wp)
   CALL r333_tracer_finish(pbbT,pbbS,pmmT,pmmS,paaT,paaS,rbb,rmm,raa)
   IF(.NOT.ieee_is_negative(pT(1,1,1))) ERROR STOP 'fixture lost 3-D negative zero'
   IF(.NOT.ieee_is_negative(rbb(1,1))) ERROR STOP 'fixture lost 2-D negative zero'
   WRITE(*,'(A,1X,I0,1X,L1)') 'R333_WRITER_FIXTURE_PASS',rank, &
      & ieee_is_negative(paaT(1,1,1))
END PROGRAM round333_writer_fixture
