PROGRAM round333_traldf_writer_fixture
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi,jpj,jpk,jpts
   USE dom_oce, ONLY : narea
   USE round333_traldf_writer
   IMPLICIT NONE
   REAL(wp) :: pt(jpi,jpj,jpk,jpts,3),p3(jpi,jpj,jpk),p2(jpi,jpj),p1(jpk)
   CHARACTER(LEN=8) :: argument
   INTEGER :: rank,jn
   CALL GET_COMMAND_ARGUMENT(1,argument); READ(argument,*) rank; narea=rank+1
   pt=0._wp; p3=1._wp+rank; p2=2._wp+rank; p1=3._wp+rank
   pt(1,1,1,1,1)=-0._wp; pt(2,2,2,2,3)=4._wp
   CALL r333_ldf_arm(2,3,3,2,1)
   CALL r333_ldf_begin(2,3,2,1,pt)
   DO jn=1,jpts
      r333_dit(2,2,1,jn)=-0._wp; r333_djt(2,2,1,jn)=jn
      r333_dkt(2,2,1,jn)=2*jn; r333_A11(2,2,1,jn)=3*jn
      r333_A22(2,2,1,jn)=4*jn; r333_A13(2,2,1,jn)=5*jn
      r333_A23(2,2,1,jn)=6*jn; r333_hmsku(2,2,1,jn)=7*jn
      r333_hmskv(2,2,1,jn)=8*jn; r333_fu(2,2,1,jn)=9*jn
      r333_fv(2,2,1,jn)=10*jn; r333_vmsku(2,2,1,jn)=11*jn
      r333_vmskv(2,2,1,jn)=12*jn; r333_ahu_w(2,2,1,jn)=13*jn
      r333_ahv_w(2,2,1,jn)=14*jn; r333_A31(2,2,1,jn)=15*jn
      r333_A32(2,2,1,jn)=16*jn; r333_fw_lower(2,2,1,jn)=-0._wp
      r333_fw_upper(2,2,1,jn)=17*jn
      r333_horizontal_seen(2,2,1,jn)=1._wp
      r333_vertical_seen(2,2,1,jn)=1._wp; r333_rhs_seen(2,2,1,jn)=1._wp
   END DO
   pt(2,2,1,:,1)=pt(2,2,1,:,1)+1._wp
   CALL r333_ldf_finish(2,3,2,1,pt,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3, &
      & p2,p2,p2,p2,p2,p2,p2,p2,p2,p2,p2,p1)
   CALL r333_ldf_disarm(2,3)
   WRITE(*,'(A,1X,I0)') 'R333_TRALDF_FIXTURE_PASS',rank
END PROGRAM round333_traldf_writer_fixture
