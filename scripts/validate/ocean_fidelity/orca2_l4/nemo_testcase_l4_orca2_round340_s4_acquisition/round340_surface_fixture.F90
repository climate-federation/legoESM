PROGRAM round340_surface_fixture_program
   USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_is_negative
   USE par_oce, ONLY: wp, jpi, jpj
   USE lib_mpp, ONLY: mpprank
   USE sbc_oce
   USE sbcmod, ONLY: r340_surface_fixture
   IMPLICIT NONE
   INTEGER :: rank
   CHARACTER(LEN=8) :: argument
   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   mpprank=rank
   ALLOCATE(sst_m(jpi,jpj),sss_m(jpi,jpj),ssu_m(jpi,jpj),ssv_m(jpi,jpj), &
      & ssh_m(jpi,jpj),e3t_m(jpi,jpj),frq_m(jpi,jpj),utau(jpi,jpj), &
      & vtau(jpi,jpj),utauU(jpi,jpj),vtauV(jpi,jpj))
   ALLOCATE(emp(jpi,jpj))
   ALLOCATE(qsr(2:jpi-1,2:jpj-1),qns(2:jpi-1,2:jpj-1), &
      & sfx(2:jpi-1,2:jpj-1), &
      & taum(2:jpi-1,2:jpj-1),wndm(2:jpi-1,2:jpj-1))
   sst_m=-0._wp ; sss_m=-0._wp ; ssu_m=-0._wp ; ssv_m=-0._wp
   ssh_m=-0._wp ; e3t_m=-0._wp ; frq_m=-0._wp
   utau=-0._wp ; vtau=-0._wp ; utauU=-0._wp ; vtauV=-0._wp
   qsr=-0._wp ; qns=-0._wp ; emp=-0._wp ; sfx=-0._wp
   taum=-0._wp ; wndm=-0._wp
   qsr(2,2)=REAL(rank+1,wp)
   CALL r340_surface_fixture(1,rank)
   IF(.NOT.ieee_is_negative(sst_m(1,1))) ERROR STOP 'lost negative zero'
   WRITE(*,'(A,1X,I0)') 'R340_SURFACE_FIXTURE_PASS',rank
END PROGRAM round340_surface_fixture_program
