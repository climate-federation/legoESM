PROGRAM round339_writer_fixture
   USE, INTRINSIC :: ieee_arithmetic, ONLY : ieee_is_negative
   USE par_kind, ONLY : wp
   USE dom_oce
   USE in_out_manager, ONLY : numout
   USE sbcmod, ONLY : r339_fixture_write
   IMPLICIT NONE
   INTEGER :: rank
   CHARACTER(LEN=8) :: argument
   REAL(wp) :: field(jpi,jpj)
   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   narea=rank+1 ; nimpp=1+90*rank ; njmpp=1
   ntsi=3 ; ntei=jpi-2 ; ntsj=3 ; ntej=jpj-2 ; numout=6
   field=-0._wp
   field(ntsi,ntsj)=REAL(rank+1,wp)/4._wp
   CALL r339_fixture_write(1,field,2)
   IF(.NOT.ieee_is_negative(field(1,1))) ERROR STOP 'fixture lost negative zero'
   WRITE(*,'(A,1X,I0)') 'R339_WRITER_FIXTURE_PASS',rank
END PROGRAM round339_writer_fixture

