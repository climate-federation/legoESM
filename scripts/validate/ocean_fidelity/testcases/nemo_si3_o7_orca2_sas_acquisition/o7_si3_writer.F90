MODULE o7_si3_writer
   USE, INTRINSIC :: iso_fortran_env, ONLY: int32, real64
   USE par_oce, ONLY: wp, jpi, jpj, nn_hls, ntsi, ntei, ntsj, ntej
   USE par_ice, ONLY: jpl
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: o7_set_context, o7_set_coeffs
   PUBLIC :: o7_blk1_entry, o7_blk1_exit, o7_tau_after
   PUBLIC :: o7_alb_entry, o7_alb_exit, o7_blk2_entry, o7_blk2_exit
   PUBLIC :: o7_flx_entry, o7_flx_exit

   INTEGER :: current_kt = -1, current_kmm = -1, current_nit000 = -1
   INTEGER :: current_rank = -1, current_jpni = -1, current_jpnj = -1
   REAL(wp) :: current_rdt = 0._wp, current_r1dt = 0._wp
   REAL(wp) :: current_rho0 = 0._wp, current_rcp = 0._wp
   REAL(wp) :: current_cd = 0._wp, current_ce = 0._wp, current_ch = 0._wp

CONTAINS

   SUBROUTINE o7_set_context( kt, kmm, nit000, rank, jpni, jpnj, rdt, r1dt, rho0, rcp )
      INTEGER, INTENT(in) :: kt, kmm, nit000, rank, jpni, jpnj
      REAL(wp), INTENT(in) :: rdt, r1dt, rho0, rcp
      current_kt = kt ; current_kmm = kmm ; current_nit000 = nit000
      current_rank = rank ; current_jpni = jpni ; current_jpnj = jpnj
      current_rdt = rdt ; current_r1dt = r1dt ; current_rho0 = rho0 ; current_rcp = rcp
   END SUBROUTINE o7_set_context

   SUBROUTINE o7_set_coeffs( cd, ce, ch )
      REAL(wp), INTENT(in) :: cd, ce, ch
      current_cd = cd ; current_ce = ce ; current_ch = ch
   END SUBROUTINE o7_set_coeffs

   SUBROUTINE open_record( kind, nfields, unit )
      INTEGER, INTENT(in) :: kind, nfields
      INTEGER, INTENT(out) :: unit
      CHARACTER(LEN=8), PARAMETER :: magic = 'O7SI3R01'
      CHARACTER(LEN=128) :: path
      INTEGER(int32), DIMENSION(20) :: meta
      REAL(real64), DIMENSION(7) :: scalars
      INTEGER :: ni, nj
      ni = ntei - ntsi + 1 ; nj = ntej - ntsj + 1
      WRITE(path,'("o7_si3_records/o7_kind",I2.2,"_step",I4.4,"_rank",I4.4,".bin")') &
         kind, current_kt, current_rank
      OPEN( NEWUNIT=unit, FILE=TRIM(path), ACCESS='stream', FORM='unformatted', STATUS='replace', ACTION='write' )
      meta = [ kind, current_kt, current_kmm, current_rank, jpi, jpj, jpl, nn_hls, &
               ntsi, ntei, ntsj, ntej, ni, nj, nfields, 1, current_nit000, &
               current_jpni, current_jpnj, 0 ]
      scalars = [ current_rdt, current_r1dt, current_rho0, current_rcp, &
                  current_cd, current_ce, current_ch ]
      WRITE(unit) magic
      WRITE(unit) INT(1, int32)
      WRITE(unit) meta
      WRITE(unit) scalars
   END SUBROUTINE open_record

   SUBROUTINE put_i2( unit, a )
      INTEGER, INTENT(in) :: unit
      REAL(wp), DIMENSION(:,:), INTENT(in) :: a
      INTEGER :: ni, nj
      ni = ntei-ntsi+1 ; nj = ntej-ntsj+1
      IF( SIZE(a,1) == ni .AND. SIZE(a,2) == nj ) THEN
         WRITE(unit) a
      ELSEIF( SIZE(a,1) == jpi .AND. SIZE(a,2) == jpj ) THEN
         WRITE(unit) a(nn_hls+1:jpi-nn_hls,nn_hls+1:jpj-nn_hls)
      ELSE
         ERROR STOP 'O7 put_i2 extent is neither A2D(0) nor full halo'
      ENDIF
   END SUBROUTINE put_i2

   SUBROUTINE put_f2( unit, a )
      INTEGER, INTENT(in) :: unit
      REAL(wp), DIMENSION(:,:), INTENT(in) :: a
      IF( SIZE(a,1) /= jpi .OR. SIZE(a,2) /= jpj ) ERROR STOP 'O7 put_f2 requires full halo'
      WRITE(unit) a
   END SUBROUTINE put_f2

   SUBROUTINE put_i3( unit, a )
      INTEGER, INTENT(in) :: unit
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: a
      INTEGER :: ni, nj
      ni = ntei-ntsi+1 ; nj = ntej-ntsj+1
      IF( SIZE(a,3) /= jpl ) ERROR STOP 'O7 put_i3 category extent differs from jpl'
      IF( SIZE(a,1) == ni .AND. SIZE(a,2) == nj ) THEN
         WRITE(unit) a
      ELSEIF( SIZE(a,1) == jpi .AND. SIZE(a,2) == jpj ) THEN
         WRITE(unit) a(nn_hls+1:jpi-nn_hls,nn_hls+1:jpj-nn_hls,1:jpl)
      ELSE
         ERROR STOP 'O7 put_i3 extent is neither A2D(0) nor full halo'
      ENDIF
   END SUBROUTINE put_i3

   SUBROUTINE o7_blk1_entry( wndi, wndj, tair, qair, slp, tm_su, rhoa, smask0 )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: wndi, wndj, tair, qair, slp, tm_su, rhoa, smask0
      INTEGER :: u
      CALL open_record(1, 8, u)
      CALL put_i2(u,wndi); CALL put_i2(u,wndj); CALL put_i2(u,tair); CALL put_i2(u,qair)
      CALL put_i2(u,slp); CALL put_i2(u,tm_su); CALL put_i2(u,rhoa); CALL put_i2(u,smask0)
      CLOSE(u)
   END SUBROUTINE o7_blk1_entry

   SUBROUTINE o7_blk1_exit( wndm, zsipt, cd, ch, ce, theta, q, utau, vtau )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: wndm, zsipt, cd, ch, ce, theta, q, utau, vtau
      INTEGER :: u
      CALL open_record(2, 9, u)
      CALL put_i2(u,wndm); CALL put_i2(u,zsipt); CALL put_i2(u,cd); CALL put_i2(u,ch)
      CALL put_i2(u,ce); CALL put_i2(u,theta); CALL put_i2(u,q); CALL put_i2(u,utau); CALL put_i2(u,vtau)
      CLOSE(u)
   END SUBROUTINE o7_blk1_exit

   SUBROUTINE o7_tau_after( utau, vtau )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: utau, vtau
      INTEGER :: u
      CALL open_record(3, 2, u); CALL put_f2(u,utau); CALL put_f2(u,vtau); CLOSE(u)
   END SUBROUTINE o7_tau_after

   SUBROUTINE o7_alb_entry( t_su, h_i, h_s, a_ip_eff, h_ip, h_il, cloud, smask0 )
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: t_su, h_i, h_s, a_ip_eff, h_ip, h_il
      REAL(wp), DIMENSION(:,:), INTENT(in) :: cloud, smask0
      INTEGER :: u
      CALL open_record(4, 8, u)
      CALL put_i3(u,t_su); CALL put_i3(u,h_i); CALL put_i3(u,h_s); CALL put_i3(u,a_ip_eff)
      CALL put_i3(u,h_ip); CALL put_i3(u,h_il); CALL put_i2(u,cloud); CALL put_i2(u,smask0)
      CLOSE(u)
   END SUBROUTINE o7_alb_entry

   SUBROUTINE o7_alb_exit( alb )
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: alb
      INTEGER :: u
      CALL open_record(5, 1, u); CALL put_i3(u,alb); CLOSE(u)
   END SUBROUTINE o7_alb_exit

   SUBROUTINE o7_blk2_entry( t_su,h_s,h_i,alb,a_i_b,tair,qair,slp,qlw,precip,snow,qsr,rhoa, &
      wndm,ch,ce,theta,qzu,sst,emp,at_i_b,qns_oce,qsr_oce,cloud,smask0 )
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: t_su,h_s,h_i,alb,a_i_b
      REAL(wp), DIMENSION(:,:), INTENT(in) :: tair,qair,slp,qlw,precip,snow,qsr,rhoa,wndm,ch,ce
      REAL(wp), DIMENSION(:,:), INTENT(in) :: theta,qzu,sst,emp,at_i_b,qns_oce,qsr_oce,cloud,smask0
      INTEGER :: u
      CALL open_record(6, 25, u)
      CALL put_i3(u,t_su); CALL put_i3(u,h_s); CALL put_i3(u,h_i); CALL put_i3(u,alb); CALL put_i3(u,a_i_b)
      CALL put_i2(u,tair); CALL put_i2(u,qair); CALL put_i2(u,slp); CALL put_i2(u,qlw)
      CALL put_i2(u,precip); CALL put_i2(u,snow); CALL put_i2(u,qsr); CALL put_i2(u,rhoa)
      CALL put_i2(u,wndm); CALL put_i2(u,ch); CALL put_i2(u,ce); CALL put_i2(u,theta); CALL put_i2(u,qzu)
      CALL put_i2(u,sst); CALL put_i2(u,emp); CALL put_i2(u,at_i_b); CALL put_i2(u,qns_oce)
      CALL put_i2(u,qsr_oce); CALL put_i2(u,cloud); CALL put_i2(u,smask0)
      CLOSE(u)
   END SUBROUTINE o7_blk2_entry

   SUBROUTINE o7_blk2_exit( qsr_i,qns_i,dqns_i,qla_i,dqla_i,evap_i,devap_i,qprec_i,qevap_i,qtr_i, &
      zsq,zsipt,zqlw,zdqlw,zqsb,zdqsb,tprecip,sprecip,emp_oce,emp_ice,emp_tot,qemp_oce,qemp_ice, &
      qns_tot,qsr_tot,zcptrain,zcptsnw,zcptn,zevap,zsnw )
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: qsr_i,qns_i,dqns_i,qla_i,dqla_i,evap_i,devap_i
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: qevap_i,qtr_i,zsq,zsipt,zqlw,zdqlw,zqsb,zdqsb
      REAL(wp), DIMENSION(:,:), INTENT(in) :: tprecip,sprecip,emp_oce,emp_ice,emp_tot,qemp_oce,qemp_ice
      REAL(wp), DIMENSION(:,:), INTENT(in) :: qns_tot,qsr_tot,qprec_i,zcptrain,zcptsnw,zcptn,zevap,zsnw
      INTEGER :: u
      CALL open_record(7, 30, u)
      CALL put_i3(u,qsr_i); CALL put_i3(u,qns_i); CALL put_i3(u,dqns_i); CALL put_i3(u,qla_i)
      CALL put_i3(u,dqla_i); CALL put_i3(u,evap_i); CALL put_i3(u,devap_i); CALL put_i3(u,qevap_i)
      CALL put_i3(u,qtr_i); CALL put_i3(u,zsq); CALL put_i3(u,zsipt)
      CALL put_i3(u,zqlw); CALL put_i3(u,zdqlw); CALL put_i3(u,zqsb); CALL put_i3(u,zdqsb)
      CALL put_i2(u,tprecip); CALL put_i2(u,sprecip); CALL put_i2(u,emp_oce); CALL put_i2(u,emp_ice)
      CALL put_i2(u,emp_tot); CALL put_i2(u,qemp_oce); CALL put_i2(u,qemp_ice); CALL put_i2(u,qns_tot)
      CALL put_i2(u,qsr_tot); CALL put_i2(u,qprec_i); CALL put_i2(u,zcptrain); CALL put_i2(u,zcptsnw); CALL put_i2(u,zcptn)
      CALL put_i2(u,zevap); CALL put_i2(u,zsnw)
      CLOSE(u)
   END SUBROUTINE o7_blk2_exit

   SUBROUTINE o7_flx_entry( uice,vice,ssu,ssv,utau,vtau,smask0,at_i,at_i_b,vt_i,qsr_oce,qns_oce, &
      qemp_oce,frq,e3t,sst,tbo,amax )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: uice,vice,ssu,ssv,utau,vtau,smask0,at_i,at_i_b,vt_i
      REAL(wp), DIMENSION(:,:), INTENT(in) :: qsr_oce,qns_oce,qemp_oce,frq,e3t,sst,tbo,amax
      INTEGER :: u
      CALL open_record(8, 18, u)
      CALL put_f2(u,uice); CALL put_f2(u,vice); CALL put_f2(u,ssu); CALL put_f2(u,ssv)
      CALL put_f2(u,utau); CALL put_f2(u,vtau); CALL put_i2(u,smask0); CALL put_i2(u,at_i)
      CALL put_i2(u,at_i_b); CALL put_i2(u,vt_i); CALL put_i2(u,qsr_oce); CALL put_i2(u,qns_oce)
      CALL put_i2(u,qemp_oce); CALL put_i2(u,frq); CALL put_i2(u,e3t); CALL put_i2(u,sst)
      CALL put_i2(u,tbo); CALL put_i2(u,amax)
      CLOSE(u)
   END SUBROUTINE o7_flx_entry

   SUBROUTINE o7_flx_exit( qsb, fhld, qlead, frq )
      REAL(wp), DIMENSION(:,:), INTENT(in) :: qsb, fhld, qlead, frq
      INTEGER :: u
      CALL open_record(9, 4, u); CALL put_i2(u,qsb); CALL put_i2(u,fhld); CALL put_i2(u,qlead); CALL put_i2(u,frq); CLOSE(u)
   END SUBROUTINE o7_flx_exit

END MODULE o7_si3_writer
