MODULE vortex_r23_ldf_terms
   !! Read-only internal tracer-LDF writer for lane round 227.
   USE par_kind       , ONLY : wp
   USE par_oce        , ONLY : jpi, jpj, jpk, jp_tem
   USE in_out_manager , ONLY : numout
   USE lib_mpp        , ONLY : ctl_stop

   IMPLICIT NONE
   PRIVATE

   PUBLIC :: r227_slope_active, r227_iso_active
   PUBLIC :: r227_slope_begin, r227_slope_finish
   PUBLIC :: r227_iso_begin, r227_iso_finish
   PUBLIC :: r227_slope_e3u, r227_slope_e3v
   PUBLIC :: r227_raw_u, r227_raw_v, r227_bound_u, r227_bound_v
   PUBLIC :: r227_hraw_u, r227_hraw_v, r227_wraw_i, r227_wraw_j
   PUBLIC :: r227_dit, r227_djt, r227_dkt
   PUBLIC :: r227_A11, r227_A22, r227_A13, r227_A23
   PUBLIC :: r227_hmsku, r227_hmskv, r227_fu, r227_fv
   PUBLIC :: r227_vmsku, r227_vmskv, r227_ahu_w, r227_ahv_w
   PUBLIC :: r227_A31, r227_A32, r227_fw_lower, r227_fw_upper

   LOGICAL, SAVE :: r227_slope_active = .FALSE.
   LOGICAL, SAVE :: r227_iso_active = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: r227_slope_e3u(:,:,:), r227_slope_e3v(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_raw_u(:,:,:), r227_raw_v(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_bound_u(:,:,:), r227_bound_v(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_hraw_u(:,:,:), r227_hraw_v(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_wraw_i(:,:,:), r227_wraw_j(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_rhs_before(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_dit(:,:,:), r227_djt(:,:,:), r227_dkt(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_A11(:,:,:), r227_A22(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_A13(:,:,:), r227_A23(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_hmsku(:,:,:), r227_hmskv(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_fu(:,:,:), r227_fv(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_vmsku(:,:,:), r227_vmskv(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_ahu_w(:,:,:), r227_ahv_w(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_A31(:,:,:), r227_A32(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r227_fw_lower(:,:,:), r227_fw_upper(:,:,:)

CONTAINS

   SUBROUTINE r227_slope_begin
      IF( r227_slope_active ) CALL ctl_stop( 'Round-227 slope writer opened twice' )
      ALLOCATE( r227_slope_e3u(jpi,jpj,jpk), r227_slope_e3v(jpi,jpj,jpk) )
      ALLOCATE( r227_raw_u(jpi,jpj,jpk), r227_raw_v(jpi,jpj,jpk) )
      ALLOCATE( r227_bound_u(jpi,jpj,jpk), r227_bound_v(jpi,jpj,jpk) )
      ALLOCATE( r227_hraw_u(jpi,jpj,jpk), r227_hraw_v(jpi,jpj,jpk) )
      ALLOCATE( r227_wraw_i(jpi,jpj,jpk), r227_wraw_j(jpi,jpj,jpk) )
      r227_slope_e3u = 0._wp ; r227_slope_e3v = 0._wp
      r227_raw_u = 0._wp ; r227_raw_v = 0._wp
      r227_bound_u = 0._wp ; r227_bound_v = 0._wp
      r227_hraw_u = 0._wp ; r227_hraw_v = 0._wp
      r227_wraw_i = 0._wp ; r227_wraw_j = 0._wp
      r227_slope_active = .TRUE.
   END SUBROUTINE r227_slope_begin


   SUBROUTINE r227_slope_finish( kt, Kbb, Kmm, prd, pn2, uslp, vslp, wslpi, wslpj )
      INTEGER, INTENT(in) :: kt, Kbb, Kmm
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: prd, pn2, uslp, vslp, wslpi, wslpj
      INTEGER :: unit, ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=128) :: file
      IF( .NOT. r227_slope_active ) RETURN
      WRITE(file,'("oracle_ldf_slope_kt",I8.8,".bin")') kt
      OPEN( NEWUNIT=unit, FILE=TRIM(file), ACCESS='STREAM', FORM='UNFORMATTED', &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) CALL ctl_stop( 'cannot open Round-227 slope record' )
      magic = 'NEMO_L1_LDFSLP1'
      WRITE(unit) magic, 1, kt, 0, Kbb, Kmm, 0, 0, jpi, jpj, jpk, 16, 0, 0, 0, STORAGE_SIZE(1._wp)
      CALL write3( unit, 'prd', prd )
      CALL write3( unit, 'pn2', pn2 )
      CALL write3( unit, 'e3u_kmm', r227_slope_e3u )
      CALL write3( unit, 'e3v_kmm', r227_slope_e3v )
      CALL write3( unit, 'raw_u', r227_raw_u )
      CALL write3( unit, 'raw_v', r227_raw_v )
      CALL write3( unit, 'bound_u', r227_bound_u )
      CALL write3( unit, 'bound_v', r227_bound_v )
      CALL write3( unit, 'hraw_u', r227_hraw_u )
      CALL write3( unit, 'hraw_v', r227_hraw_v )
      CALL write3( unit, 'wraw_i', r227_wraw_i )
      CALL write3( unit, 'wraw_j', r227_wraw_j )
      CALL write3( unit, 'uslp', uslp )
      CALL write3( unit, 'vslp', vslp )
      CALL write3( unit, 'wslpi', wslpi )
      CALL write3( unit, 'wslpj', wslpj )
      CLOSE(unit, IOSTAT=ios)
      IF( ios /= 0 ) CALL ctl_stop( 'cannot close Round-227 slope record' )
      DEALLOCATE( r227_slope_e3u, r227_slope_e3v, r227_raw_u, r227_raw_v )
      DEALLOCATE( r227_bound_u, r227_bound_v, r227_hraw_u, r227_hraw_v )
      DEALLOCATE( r227_wraw_i, r227_wraw_j )
      r227_slope_active = .FALSE.
      WRITE(numout,*) 'ROUND227_LDF_SLOPE_DUMP ', kt, TRIM(file)
   END SUBROUTINE r227_slope_finish


   SUBROUTINE r227_iso_begin( pt, Krhs )
      INTEGER, INTENT(in) :: Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pt
      IF( r227_iso_active ) CALL ctl_stop( 'Round-227 ISO writer opened twice' )
      ALLOCATE( r227_rhs_before(jpi,jpj,jpk) )
      ALLOCATE( r227_dit(jpi,jpj,jpk), r227_djt(jpi,jpj,jpk), r227_dkt(jpi,jpj,jpk) )
      ALLOCATE( r227_A11(jpi,jpj,jpk), r227_A22(jpi,jpj,jpk), r227_A13(jpi,jpj,jpk), r227_A23(jpi,jpj,jpk) )
      ALLOCATE( r227_hmsku(jpi,jpj,jpk), r227_hmskv(jpi,jpj,jpk), r227_fu(jpi,jpj,jpk), r227_fv(jpi,jpj,jpk) )
      ALLOCATE( r227_vmsku(jpi,jpj,jpk), r227_vmskv(jpi,jpj,jpk), r227_ahu_w(jpi,jpj,jpk), r227_ahv_w(jpi,jpj,jpk) )
      ALLOCATE( r227_A31(jpi,jpj,jpk), r227_A32(jpi,jpj,jpk), r227_fw_lower(jpi,jpj,jpk), r227_fw_upper(jpi,jpj,jpk) )
      r227_rhs_before = pt(:,:,:,jp_tem,Krhs)
      r227_dit = 0._wp ; r227_djt = 0._wp ; r227_dkt = 0._wp
      r227_A11 = 0._wp ; r227_A22 = 0._wp ; r227_A13 = 0._wp ; r227_A23 = 0._wp
      r227_hmsku = 0._wp ; r227_hmskv = 0._wp ; r227_fu = 0._wp ; r227_fv = 0._wp
      r227_vmsku = 0._wp ; r227_vmskv = 0._wp ; r227_ahu_w = 0._wp ; r227_ahv_w = 0._wp
      r227_A31 = 0._wp ; r227_A32 = 0._wp ; r227_fw_lower = 0._wp ; r227_fw_upper = 0._wp
      r227_iso_active = .TRUE.
   END SUBROUTINE r227_iso_begin


   SUBROUTINE r227_iso_finish( kt, Kbb, Kmm, Krhs, pt, e3t, e3u, e3v, tmask, umask, vmask, wmask, &
      &                       ahtu, ahtv, uslp, vslp, wslpi, wslpj, ah_wslp2, akz, &
      &                       r3t, r3u, r3v, e2_e1u, e1_e2v, e2u, e1v, e1t, e2t, e1e2t, r1_e1e2t, e3w_1d )
      INTEGER, INTENT(in) :: kt, Kbb, Kmm, Krhs
      REAL(wp), DIMENSION(:,:,:,:,:), INTENT(in) :: pt
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: e3t, e3u, e3v, tmask, umask, vmask, wmask
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: ahtu, ahtv, uslp, vslp, wslpi, wslpj, ah_wslp2, akz
      REAL(wp), DIMENSION(:,:), INTENT(in) :: r3t, r3u, r3v, e2_e1u, e1_e2v, e2u, e1v, e1t, e2t, e1e2t, r1_e1e2t
      REAL(wp), DIMENSION(:), INTENT(in) :: e3w_1d
      INTEGER :: unit, ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=128) :: file
      IF( .NOT. r227_iso_active ) RETURN
      WRITE(file,'("oracle_ldf_iso_kt",I8.8,".bin")') kt
      OPEN( NEWUNIT=unit, FILE=TRIM(file), ACCESS='STREAM', FORM='UNFORMATTED', &
         &  STATUS='REPLACE', ACTION='WRITE', IOSTAT=ios )
      IF( ios /= 0 ) CALL ctl_stop( 'cannot open Round-227 ISO record' )
      magic = 'NEMO_L1_LDFISO1'
      WRITE(unit) magic, 1, kt, 3, Kbb, Kmm, 0, Krhs, jpi, jpj, jpk, 50, 0, 0, 0, STORAGE_SIZE(1._wp)
      CALL write3( unit, 't_kbb', pt(:,:,:,jp_tem,Kbb) )
      CALL write3( unit, 'rhs_before', r227_rhs_before )
      CALL write3( unit, 'rhs_after', pt(:,:,:,jp_tem,Krhs) )
      CALL write3( unit, 'rhs_increment', pt(:,:,:,jp_tem,Krhs) - r227_rhs_before )
      CALL write3( unit, 'e3t_3d', e3t ) ; CALL write3( unit, 'e3u_3d', e3u ) ; CALL write3( unit, 'e3v_3d', e3v )
      CALL write3( unit, 'tmask', tmask ) ; CALL write3( unit, 'umask', umask )
      CALL write3( unit, 'vmask', vmask ) ; CALL write3( unit, 'wmask', wmask )
      CALL write3( unit, 'ahtu', ahtu ) ; CALL write3( unit, 'ahtv', ahtv )
      CALL write3( unit, 'uslp', uslp ) ; CALL write3( unit, 'vslp', vslp )
      CALL write3( unit, 'wslpi', wslpi ) ; CALL write3( unit, 'wslpj', wslpj )
      CALL write3( unit, 'ah_wslp2', ah_wslp2 ) ; CALL write3( unit, 'akz', akz )
      CALL write3( unit, 'dit', r227_dit ) ; CALL write3( unit, 'djt', r227_djt ) ; CALL write3( unit, 'dkt', r227_dkt )
      CALL write3( unit, 'A11', r227_A11 ) ; CALL write3( unit, 'A22', r227_A22 )
      CALL write3( unit, 'A13', r227_A13 ) ; CALL write3( unit, 'A23', r227_A23 )
      CALL write3( unit, 'hmsku', r227_hmsku ) ; CALL write3( unit, 'hmskv', r227_hmskv )
      CALL write3( unit, 'fu', r227_fu ) ; CALL write3( unit, 'fv', r227_fv )
      CALL write3( unit, 'vmsku', r227_vmsku ) ; CALL write3( unit, 'vmskv', r227_vmskv )
      CALL write3( unit, 'ahu_w', r227_ahu_w ) ; CALL write3( unit, 'ahv_w', r227_ahv_w )
      CALL write3( unit, 'A31', r227_A31 ) ; CALL write3( unit, 'A32', r227_A32 )
      CALL write3( unit, 'fw_lower', r227_fw_lower ) ; CALL write3( unit, 'fw_upper', r227_fw_upper )
      CALL write2( unit, 'r3t_kmm', r3t ) ; CALL write2( unit, 'r3u_kmm', r3u ) ; CALL write2( unit, 'r3v_kmm', r3v )
      CALL write2( unit, 'e2_e1u', e2_e1u ) ; CALL write2( unit, 'e1_e2v', e1_e2v )
      CALL write2( unit, 'e2u', e2u ) ; CALL write2( unit, 'e1v', e1v )
      CALL write2( unit, 'e1t', e1t ) ; CALL write2( unit, 'e2t', e2t )
      CALL write2( unit, 'e1e2t', e1e2t ) ; CALL write2( unit, 'r1_e1e2t', r1_e1e2t )
      CALL write1( unit, 'e3w_1d', e3w_1d )
      CLOSE(unit, IOSTAT=ios)
      IF( ios /= 0 ) CALL ctl_stop( 'cannot close Round-227 ISO record' )
      DEALLOCATE( r227_rhs_before, r227_dit, r227_djt, r227_dkt )
      DEALLOCATE( r227_A11, r227_A22, r227_A13, r227_A23, r227_hmsku, r227_hmskv, r227_fu, r227_fv )
      DEALLOCATE( r227_vmsku, r227_vmskv, r227_ahu_w, r227_ahv_w, r227_A31, r227_A32, r227_fw_lower, r227_fw_upper )
      r227_iso_active = .FALSE.
      WRITE(numout,*) 'ROUND227_LDF_ISO_DUMP ', kt, TRIM(file)
   END SUBROUTINE r227_iso_finish


   SUBROUTINE write3( unit, name, field )
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:,:), INTENT(in) :: field
      CHARACTER(LEN=16) :: label
      label = name
      WRITE(unit) label, 3, SIZE(field,1), SIZE(field,2), SIZE(field,3)
      WRITE(unit) field
   END SUBROUTINE write3

   SUBROUTINE write2( unit, name, field )
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:,:), INTENT(in) :: field
      CHARACTER(LEN=16) :: label
      label = name
      WRITE(unit) label, 2, SIZE(field,1), SIZE(field,2), 1
      WRITE(unit) field
   END SUBROUTINE write2

   SUBROUTINE write1( unit, name, field )
      INTEGER, INTENT(in) :: unit
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), DIMENSION(:), INTENT(in) :: field
      CHARACTER(LEN=16) :: label
      label = name
      WRITE(unit) label, 1, SIZE(field,1), 1, 1
      WRITE(unit) field
   END SUBROUTINE write1

END MODULE vortex_r23_ldf_terms
