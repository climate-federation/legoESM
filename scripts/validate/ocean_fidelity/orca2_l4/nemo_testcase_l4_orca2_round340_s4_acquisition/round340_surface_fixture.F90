PROGRAM round340_surface_fixture_program
   USE, INTRINSIC :: ieee_arithmetic, ONLY: ieee_is_negative
   USE par_oce, ONLY: wp, jpi, jpj, jpk, jpts
   USE dom_oce, ONLY: Nis0, Nie0, Njs0, Nje0
   USE lib_mpp, ONLY: mpprank
   USE sbc_oce
   USE sbcrnf, ONLY: rnf_tsc
   USE sbcssr, ONLY: erp, qrp, coefice
   USE sbcmod, ONLY: r340_surface_probe
   USE fldread, ONLY: r340_fld_probe
   USE sbcice_if, ONLY: r340_iif_probe
   USE sbcblk, ONLY: r340_s1_probe
   USE stprk3, ONLY: r340_entry_probe
   USE stprk3_stg, ONLY: r340_stage_probe
   USE sbcfwb, ONLY: r340_fwb_probe
   IMPLICIT NONE
   INTEGER :: rank, kid
   REAL(wp), ALLOCATABLE :: pts(:,:,:,:), pu(:,:,:), pv(:,:,:), pssh(:,:)
   REAL(wp) :: z(2,1), fwb_value
   CHARACTER(LEN=8) :: argument
   CALL GET_COMMAND_ARGUMENT(1,argument)
   READ(argument,*) rank
   mpprank=rank
   ALLOCATE(sst_m(jpi,jpj),sss_m(jpi,jpj),ssu_m(jpi,jpj),ssv_m(jpi,jpj), &
      & ssh_m(jpi,jpj),e3t_m(jpi,jpj),frq_m(jpi,jpj),utau(jpi,jpj), &
      & vtau(jpi,jpj),utauU(jpi,jpj),vtauV(jpi,jpj),emp(jpi,jpj), &
      & rnf(jpi,jpj),fr_i(jpi,jpj))
   ALLOCATE(qsr(Nis0:Nie0,Njs0:Nje0),qns(Nis0:Nie0,Njs0:Nje0), &
      & sfx(Nis0:Nie0,Njs0:Nje0), &
      & taum(Nis0:Nie0,Njs0:Nje0),wndm(Nis0:Nie0,Njs0:Nje0), &
      & rnf_tsc(Nis0:Nie0,Njs0:Nje0,jpts),erp(Nis0:Nie0,Njs0:Nje0), &
      & qrp(Nis0:Nie0,Njs0:Nje0),coefice(Nis0:Nie0,Njs0:Nje0))
   ALLOCATE(pts(jpi,jpj,jpk,jpts),pu(jpi,jpj,jpk),pv(jpi,jpj,jpk),pssh(jpi,jpj))
   sst_m=-0._wp ; sss_m=-0._wp ; ssu_m=-0._wp ; ssv_m=-0._wp
   ssh_m=-0._wp ; e3t_m=-0._wp ; frq_m=-0._wp
   utau=-0._wp ; vtau=-0._wp ; utauU=-0._wp ; vtauV=-0._wp ; rnf=-0._wp ; fr_i=-0._wp
   qsr=-0._wp ; qns=-0._wp ; emp=-0._wp ; sfx=-0._wp
   taum=-0._wp ; wndm=-0._wp ; rnf_tsc=-0._wp ; erp=-0._wp ; qrp=-0._wp ; coefice=-0._wp
   pts=-0._wp ; pu=-0._wp ; pv=-0._wp ; pssh=-0._wp
   z=-0._wp
   qsr(Nis0,Njs0)=REAL(rank+1,wp)
   DO kid=0,11
      CALL r340_surface_probe(1,kid)
   END DO
   CALL r340_fld_probe(1,10800)
   DO kid=0,3
      CALL r340_iif_probe(1,kid,1,1,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z,z)
   END DO
   CALL r340_s1_probe(1,z)
   CALL r340_entry_probe(1,1,pts,pu,pv,pssh)
   CALL r340_stage_probe(1,1,pts,pu,pv,pssh)
   CALL r340_fwb_probe(fwb_value)
   IF(.NOT.ieee_is_negative(sst_m(1,1))) ERROR STOP 'lost negative zero'
   WRITE(*,'(A,1X,I0)') 'R340_SURFACE_FIXTURE_PASS',rank
END PROGRAM round340_surface_fixture_program
