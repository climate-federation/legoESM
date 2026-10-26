PROGRAM round340_surface_fixture_program
   USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_is_negative
   USE par_oce, ONLY: wp, jpi, jpj, jpk, jpts
   USE lib_mpp, ONLY: mpprank
   USE sbc_oce
   USE sbcrnf, ONLY: rnf_tsc
   USE sbcssr, ONLY: erp, qrp, coefice
   USE sbcfwb, ONLY: a_fwb, a_fwb_b
   USE round340_surface_writer, ONLY: r340_surface_frame
   USE round340_state_writer, ONLY: r340_state_frame
   IMPLICIT NONE
   INTEGER :: rank
   REAL(wp), ALLOCATABLE :: pts(:,:,:,:), pu(:,:,:), pv(:,:,:), pssh(:,:)
   CHARACTER(LEN=8) :: argument
   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   mpprank=rank
   ALLOCATE(sst_m(jpi,jpj),sss_m(jpi,jpj),ssu_m(jpi,jpj),ssv_m(jpi,jpj), &
      & ssh_m(jpi,jpj),e3t_m(jpi,jpj),frq_m(jpi,jpj),utau(jpi,jpj), &
      & vtau(jpi,jpj),utauU(jpi,jpj),vtauV(jpi,jpj),emp(jpi,jpj), &
      & rnf(jpi,jpj),fr_i(jpi,jpj))
   ALLOCATE(qsr(2:jpi-1,2:jpj-1),qns(2:jpi-1,2:jpj-1), &
      & sfx(2:jpi-1,2:jpj-1), &
      & taum(2:jpi-1,2:jpj-1),wndm(2:jpi-1,2:jpj-1), &
      & rnf_tsc(2:jpi-1,2:jpj-1,jpts),erp(2:jpi-1,2:jpj-1), &
      & qrp(2:jpi-1,2:jpj-1),coefice(2:jpi-1,2:jpj-1))
   ALLOCATE(pts(jpi,jpj,jpk,jpts),pu(jpi,jpj,jpk),pv(jpi,jpj,jpk),pssh(jpi,jpj))
   sst_m=-0._wp ; sss_m=-0._wp ; ssu_m=-0._wp ; ssv_m=-0._wp
   ssh_m=-0._wp ; e3t_m=-0._wp ; frq_m=-0._wp
   utau=-0._wp ; vtau=-0._wp ; utauU=-0._wp ; vtauV=-0._wp ; rnf=-0._wp ; fr_i=-0._wp
   qsr=-0._wp ; qns=-0._wp ; emp=-0._wp ; sfx=-0._wp
   taum=-0._wp ; wndm=-0._wp ; rnf_tsc=-0._wp ; erp=-0._wp ; qrp=-0._wp ; coefice=-0._wp
   a_fwb=-0._wp ; a_fwb_b=-0._wp
   pts=-0._wp ; pu=-0._wp ; pv=-0._wp ; pssh=-0._wp
   qsr(2,2)=REAL(rank+1,wp)
   CALL r340_surface_frame(1,rank)
   CALL r340_state_frame(1,rank,1,pts,pu,pv,pssh)
   IF(.NOT.ieee_is_negative(sst_m(1,1))) ERROR STOP 'lost negative zero'
   WRITE(*,'(A,1X,I0)') 'R340_SURFACE_FIXTURE_PASS',rank
END PROGRAM round340_surface_fixture_program
