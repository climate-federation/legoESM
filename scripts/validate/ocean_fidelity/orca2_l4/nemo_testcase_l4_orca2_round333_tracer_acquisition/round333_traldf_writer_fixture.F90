PROGRAM round333_traldf_writer_fixture
   USE par_kind, ONLY : wp
   USE par_oce, ONLY : jpi,jpj,jpk,jpts
   USE dom_oce, ONLY : narea
   USE round333_traldf_writer
   IMPLICIT NONE
   REAL(wp) :: pt(jpi,jpj,jpk,jpts,3),p3(jpi,jpj,jpk),p2(jpi,jpj),p1(jpk)
   REAL(wp) :: pah(jpi,jpj,jpk),pakz_raw(jpi,jpj,jpk),pakz(jpi,jpj,jpk)
   REAL(wp) :: fu(jpi,jpj),fv(jpi,jpj),fw0(jpi,jpj),fw1(jpi,jpj)
   REAL(wp) :: zu,zv,zahu,zahv
   CHARACTER(LEN=8) :: argument
   INTEGER :: rank,jn,ji,jj,jk
   CALL GET_COMMAND_ARGUMENT(1,argument); READ(argument,*) rank; narea=rank+1
   pt=0._wp; p3=1._wp+rank; p2=2._wp+rank; p1=3._wp+rank
   pt(1,1,1,1,1)=-0._wp; pt(2,2,2,2,3)=4._wp
   CALL r333_ldf_arm(2,3,3,2,1)
   CALL r333_ldf_begin(2,3,2,1,pt)
   pah=0._wp; pakz_raw=0._wp; pakz=0._wp
   DO jk=2,jpk-1; DO jj=2,jpj-1; DO ji=2,jpi-1
      zu=p3(ji,jj,jk)/MAX((p3(ji,jj,jk-1)+p3(ji-1,jj,jk)) + &
         & (p3(ji-1,jj,jk-1)+p3(ji,jj,jk)),1._wp)
      zv=p3(ji,jj,jk)/MAX((p3(ji,jj,jk-1)+p3(ji,jj-1,jk)) + &
         & (p3(ji,jj-1,jk-1)+p3(ji,jj,jk)),1._wp)
      zahu=((p3(ji,jj,jk-1)+p3(ji-1,jj,jk)) + &
         & (p3(ji-1,jj,jk-1)+p3(ji,jj,jk)))*zu
      zahv=((p3(ji,jj,jk-1)+p3(ji,jj-1,jk)) + &
         & (p3(ji,jj-1,jk-1)+p3(ji,jj,jk)))*zv
      pah(ji,jj,jk)=zahu*p3(ji,jj,jk)*p3(ji,jj,jk) + &
         & zahv*p3(ji,jj,jk)*p3(ji,jj,jk)
      pakz_raw(ji,jj,jk)=( &
         & ((p3(ji,jj,jk)+p3(ji,jj,jk-1))/(p2(ji,jj)*p2(ji,jj)) + &
         &  (p3(ji-1,jj,jk)+p3(ji-1,jj,jk-1))/(p2(ji-1,jj)*p2(ji-1,jj))) + &
         & ((p3(ji,jj,jk)+p3(ji,jj,jk-1))/(p2(ji,jj)*p2(ji,jj)) + &
         &  (p3(ji,jj-1,jk)+p3(ji,jj-1,jk-1))/(p2(ji,jj-1)*p2(ji,jj-1))) )*0.25_wp
      pakz(ji,jj,jk)=MAX(0.25_wp*(pakz_raw(ji,jj,jk)+pah(ji,jj,jk)/ &
         & ((p1(jk)*(1._wp+p2(ji,jj)))*(p1(jk)*(1._wp+p2(ji,jj)))))-0.5_wp,0._wp) * &
         & ((p1(jk)*(1._wp+p2(ji,jj)))*(p1(jk)*(1._wp+p2(ji,jj))))*4._wp
   END DO; END DO; END DO
   CALL r333_ldf_a33_first(2,2,jpi-1,2,jpj-1,pah,p3,p3,p3,p3,p3,p3,p3)
   CALL r333_ldf_a33_raw(2,jpi-1,2,jpj-1,pakz_raw)
   CALL r333_ldf_a33_lap(2,2,jpi-1,2,jpj-1,pakz,p1,p2,0.25_wp,4._wp)
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
      fu=0._wp; fv=0._wp; fw0=0._wp; fw1=0._wp
      fu(2,2)=REAL(jn,wp); fv(2,2)=2._wp*REAL(jn,wp)
      fw0(2,2)=-0._wp; fw1(2,2)=0.5_wp*REAL(jn,wp)
      CALL r333_ldf_horizontal(jn,1,1,jpi-1,1,jpj-1,fu(1:jpi-1,1:jpj-1),fv(1:jpi-1,1:jpj-1))
      pt(2,2,1,jn,1)=4._wp+(((fu(2,2)-fu(1,2))+(fv(2,2)-fv(2,1))) + &
         & (fw0(2,2)-fw1(2,2)))*p2(2,2)/p3(2,2,1)
      CALL r333_ldf_vertical_before(jn,1,2,2,2,2,fw0(2:2,2:2),RESHAPE((/4._wp/),(/1,1/)))
      CALL r333_ldf_vertical_after(jn,1,2,2,2,2,fw1(2:2,2:2),pt(2:2,2:2,1,jn,1))
   END DO
   CALL r333_ldf_finish(2,3,2,1,pt,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,p3,pah,pakz, &
      & p2,p2,p2,p2,p2,p2,p2,p2,p2,p2,p2,p2,p2,p1)
   CALL r333_ldf_disarm(2,3)
   WRITE(*,'(A,1X,I0)') 'R333_TRALDF_FIXTURE_PASS',rank
END PROGRAM round333_traldf_writer_fixture
