MODULE dynzdf_round328_writer
   !! WRITE-only kt=1 stage-3 dyn_zdf source-order recorder for ORCA2 round 328.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : nit000
   USE iso_fortran_env, ONLY : int64
   USE lib_mpp,        ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r328_begin, r328_state_row, r328_matrix_row, r328_solve_row, r328_finish

   INTEGER, PARAMETER :: jp_state = 4, jp_matrix = 5
   INTEGER, SAVE :: nunit = -1, nx = 0, ny = 0
   INTEGER, SAVE :: nkt = -1, nkbb = -1, nkmm = -1, nkrhs = -1, nkaa = -1
   INTEGER, SAVE :: nstate_count(jp_state) = 0
   INTEGER, SAVE :: nmatrix_count(2,jp_matrix) = 0, nsolve_count(2) = 0
   LOGICAL, SAVE :: lactive = .FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: before_u(:,:,:), before_v(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: rhs_u(:,:,:), rhs_v(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: avm_in(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r3u_kmm_in(:,:), r3v_kmm_in(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: r3u_kaa_in(:,:), r3v_kaa_in(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: baro_u(:,:), baro_v(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: rcdu_bot_in(:,:), rcdu_top_in(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: utau_in(:,:), vtau_in(:,:)
   REAL(wp), ALLOCATABLE, SAVE :: state_u(:,:,:,:), state_v(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: matrix(:,:,:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: solve_u(:,:,:), solve_v(:,:,:)

CONTAINS

   FUNCTION checksum2(value) RESULT(checksum)
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER(int64) :: checksum, word
      INTEGER :: ji, jj
      checksum = 0_int64
      DO jj=1,SIZE(value,2); DO ji=1,SIZE(value,1)
         word = TRANSFER(value(ji,jj),word)
         checksum = IEOR(checksum,word)
      END DO; END DO
   END FUNCTION checksum2

   FUNCTION checksum3(value) RESULT(checksum)
      REAL(wp), INTENT(in) :: value(:,:,:)
      INTEGER(int64) :: checksum, word
      INTEGER :: ji, jj, jk
      checksum = 0_int64
      DO jk=1,SIZE(value,3); DO jj=1,SIZE(value,2); DO ji=1,SIZE(value,1)
         word = TRANSFER(value(ji,jj,jk),word)
         checksum = IEOR(checksum,word)
      END DO; END DO; END DO
   END FUNCTION checksum3

   SUBROUTINE put2(name, value, i0, j0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER, INTENT(in) :: i0, j0
      CHARACTER(LEN=24) :: field
      field = name
      WRITE(nunit) field
      WRITE(nunit) 2, SIZE(value,1), SIZE(value,2), 1, i0, j0, 1
      WRITE(nunit) checksum2(value)
      WRITE(nunit) value
   END SUBROUTINE put2

   SUBROUTINE put3(name, value, i0, j0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:,:)
      INTEGER, INTENT(in) :: i0, j0
      CHARACTER(LEN=24) :: field
      field = name
      WRITE(nunit) field
      WRITE(nunit) 3, SIZE(value,1), SIZE(value,2), SIZE(value,3), i0, j0, 1
      WRITE(nunit) checksum3(value)
      WRITE(nunit) value
   END SUBROUTINE put3

   SUBROUTINE r328_begin(kt, Kbb, Kmm, Krhs, Kaa, puu, pvv, pavm, &
      & pr3u_kmm, pr3v_kmm, pr3u_kaa, pr3v_kaa, puub, pvvb, pbot, ptop, putau, pvtau)
      INTEGER, INTENT(in) :: kt, Kbb, Kmm, Krhs, Kaa
      REAL(wp), INTENT(in) :: puu(:,:,:,:), pvv(:,:,:,:), pavm(:,:,:)
      REAL(wp), INTENT(in) :: pr3u_kmm(:,:), pr3v_kmm(:,:)
      REAL(wp), INTENT(in) :: pr3u_kaa(:,:), pr3v_kaa(:,:)
      REAL(wp), INTENT(in) :: puub(:,:), pvvb(:,:), pbot(:,:), ptop(:,:)
      REAL(wp), INTENT(in) :: putau(:,:), pvtau(:,:)
      lactive = kt == nit000 .AND. Kbb == 1 .AND. Kmm == 2 .AND. Krhs == 3 .AND. Kaa == 3
      IF(.NOT.lactive) RETURN
      IF(nunit /= -1 .OR. ALLOCATED(before_u)) CALL ctl_stop('round328: nested dyn_zdf record')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round328: writer requires fp64')
      nx = ntei-ntsi+1 ; ny = ntej-ntsj+1
      nkt=kt ; nkbb=Kbb ; nkmm=Kmm ; nkrhs=Krhs ; nkaa=Kaa
      ALLOCATE(before_u(nx,ny,jpkm1), before_v(nx,ny,jpkm1), &
         & rhs_u(nx,ny,jpkm1), rhs_v(nx,ny,jpkm1), &
         & avm_in(nx+1,ny+1,jpk), &
         & r3u_kmm_in(nx,ny), r3v_kmm_in(nx,ny), &
         & r3u_kaa_in(nx,ny), r3v_kaa_in(nx,ny), &
         & baro_u(nx,ny), baro_v(nx,ny), &
         & rcdu_bot_in(nx+1,ny+1), rcdu_top_in(nx+1,ny+1), &
         & utau_in(nx,ny), vtau_in(nx,ny), &
         & state_u(jp_state,nx,ny,jpkm1), state_v(jp_state,nx,ny,jpkm1), &
         & matrix(2,jp_matrix,3,nx,ny,jpkm1), &
         & solve_u(nx,ny,jpkm1), solve_v(nx,ny,jpkm1))
      before_u=puu(ntsi:ntei,ntsj:ntej,1:jpkm1,Kbb)
      before_v=pvv(ntsi:ntei,ntsj:ntej,1:jpkm1,Kbb)
      rhs_u=puu(ntsi:ntei,ntsj:ntej,1:jpkm1,Krhs)
      rhs_v=pvv(ntsi:ntei,ntsj:ntej,1:jpkm1,Krhs)
      avm_in=pavm(ntsi:ntei+1,ntsj:ntej+1,1:jpk)
      r3u_kmm_in=pr3u_kmm(ntsi:ntei,ntsj:ntej)
      r3v_kmm_in=pr3v_kmm(ntsi:ntei,ntsj:ntej)
      r3u_kaa_in=pr3u_kaa(ntsi:ntei,ntsj:ntej)
      r3v_kaa_in=pr3v_kaa(ntsi:ntei,ntsj:ntej)
      baro_u=puub(ntsi:ntei,ntsj:ntej) ; baro_v=pvvb(ntsi:ntei,ntsj:ntej)
      rcdu_bot_in=pbot(ntsi:ntei+1,ntsj:ntej+1)
      rcdu_top_in=ptop(ntsi:ntei+1,ntsj:ntej+1)
      utau_in=putau(ntsi:ntei,ntsj:ntej) ; vtau_in=pvtau(ntsi:ntei,ntsj:ntej)
      state_u=0._wp ; state_v=0._wp ; matrix=0._wp
      solve_u=0._wp ; solve_v=0._wp
      nstate_count=0 ; nmatrix_count=0 ; nsolve_count=0
   END SUBROUTINE r328_begin

   SUBROUTINE r328_state_row(phase, jj, urow, vrow)
      INTEGER, INTENT(in) :: phase, jj
      REAL(wp), INTENT(in) :: urow(:,:), vrow(:,:)
      INTEGER :: jlocal
      IF(.NOT.lactive) RETURN
      IF(phase < 1 .OR. phase > jp_state) CALL ctl_stop('round328: invalid state phase')
      jlocal=jj-ntsj+1
      IF(jlocal < 1 .OR. jlocal > ny) CALL ctl_stop('round328: state row outside owned domain')
      IF(ANY(SHAPE(urow) /= (/nx,jpkm1/)) .OR. ANY(SHAPE(vrow) /= SHAPE(urow))) &
         & CALL ctl_stop('round328: wrong state row shape')
      state_u(phase,:,jlocal,:)=urow ; state_v(phase,:,jlocal,:)=vrow
      nstate_count(phase)=nstate_count(phase)+1
   END SUBROUTINE r328_state_row

   SUBROUTINE r328_matrix_row(component, phase, jj, lower, diagonal, upper)
      INTEGER, INTENT(in) :: component, phase, jj
      REAL(wp), INTENT(in) :: lower(:,:), diagonal(:,:), upper(:,:)
      INTEGER :: jlocal
      IF(.NOT.lactive) RETURN
      IF(component < 1 .OR. component > 2 .OR. phase < 1 .OR. phase > jp_matrix) &
         & CALL ctl_stop('round328: invalid matrix phase')
      jlocal=jj-ntsj+1
      IF(jlocal < 1 .OR. jlocal > ny) CALL ctl_stop('round328: matrix row outside owned domain')
      IF(ANY(SHAPE(lower) /= (/nx,jpkm1/)) .OR. ANY(SHAPE(diagonal) /= SHAPE(lower)) &
         & .OR. ANY(SHAPE(upper) /= SHAPE(lower))) CALL ctl_stop('round328: wrong matrix row shape')
      matrix(component,phase,1,:,jlocal,:)=lower
      matrix(component,phase,2,:,jlocal,:)=diagonal
      matrix(component,phase,3,:,jlocal,:)=upper
      nmatrix_count(component,phase)=nmatrix_count(component,phase)+1
   END SUBROUTINE r328_matrix_row

   SUBROUTINE r328_solve_row(component, jj, value)
      INTEGER, INTENT(in) :: component, jj
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER :: jlocal
      IF(.NOT.lactive) RETURN
      jlocal=jj-ntsj+1
      IF(jlocal < 1 .OR. jlocal > ny .OR. ANY(SHAPE(value) /= (/nx,jpkm1/))) &
         & CALL ctl_stop('round328: wrong solve row')
      SELECT CASE(component)
      CASE(1); solve_u(:,jlocal,:)=value
      CASE(2); solve_v(:,jlocal,:)=value
      CASE DEFAULT; CALL ctl_stop('round328: invalid solve component')
      END SELECT
      nsolve_count(component)=nsolve_count(component)+1
   END SUBROUTINE r328_solve_row

   SUBROUTINE r328_finish(puu, pvv, Kaa)
      REAL(wp), INTENT(in) :: puu(:,:,:,:), pvv(:,:,:,:)
      INTEGER, INTENT(in) :: Kaa
      INTEGER :: ios, component, phase, term
      CHARACTER(LEN=160) :: filename
      CHARACTER(LEN=24) :: name
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=1), PARAMETER :: comp_name(2)=(/'u','v'/)
      CHARACTER(LEN=4), PARAMETER :: matrix_name(3)=(/'low ','diag','up  '/)
      CHARACTER(LEN=10), PARAMETER :: state_name(4)=(/'explicit  ','baro_sub  ','bottom_rhs','top_rhs   '/)
      CHARACTER(LEN=12), PARAMETER :: phase_name(5)=(/'base_zdf    ','after_zad   ','after_ldf   ','bottom_diag ','top_diag    '/)
      IF(.NOT.lactive) RETURN
      IF(ANY(nstate_count /= ny) .OR. ANY(nmatrix_count /= ny) &
         & .OR. ANY(nsolve_count /= ny)) &
         & CALL ctl_stop('round328: incomplete dyn_zdf record')
      WRITE(filename,'("oracle_dynzdf_kt",I8.8,"_s3_r",I4.4,".bin")') nkt,narea-1
      OPEN(NEWUNIT=nunit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='NEW',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round328: cannot open dyn_zdf record')
      magic='NEMO_L4_R328ZDF'
      WRITE(nunit) magic
      WRITE(nunit) 1,nkt,3,nkbb,nkmm,nkrhs,nkaa,narea-1,nx,ny,jpkm1,ntsi,ntsj, &
         & STORAGE_SIZE(1._wp),57
      CALL put3('before_u',before_u,ntsi,ntsj); CALL put3('before_v',before_v,ntsi,ntsj)
      CALL put3('rhs_u',rhs_u,ntsi,ntsj); CALL put3('rhs_v',rhs_v,ntsi,ntsj)
      CALL put3('avm_input',avm_in,ntsi,ntsj)
      CALL put2('r3u_kmm',r3u_kmm_in,ntsi,ntsj); CALL put2('r3v_kmm',r3v_kmm_in,ntsi,ntsj)
      CALL put2('r3u_kaa',r3u_kaa_in,ntsi,ntsj); CALL put2('r3v_kaa',r3v_kaa_in,ntsi,ntsj)
      CALL put2('barotropic_u',baro_u,ntsi,ntsj); CALL put2('barotropic_v',baro_v,ntsi,ntsj)
      CALL put2('rCdU_bot',rcdu_bot_in,ntsi,ntsj); CALL put2('rCdU_top',rcdu_top_in,ntsi,ntsj)
      CALL put2('utauU',utau_in,ntsi,ntsj); CALL put2('vtauV',vtau_in,ntsi,ntsj)
      DO phase=1,4
         name=TRIM(state_name(phase))//'_'//comp_name(1); CALL put3(name,state_u(phase,:,:,:),ntsi,ntsj)
         name=TRIM(state_name(phase))//'_'//comp_name(2); CALL put3(name,state_v(phase,:,:,:),ntsi,ntsj)
      END DO
      DO component=1,2; DO phase=1,jp_matrix; DO term=1,3
         name=comp_name(component)//'_'//TRIM(phase_name(phase))//'_'//TRIM(matrix_name(term))
         CALL put3(name,matrix(component,phase,term,:,:,:),ntsi,ntsj)
      END DO; END DO; END DO
      CALL put3('solve_u',solve_u,ntsi,ntsj); CALL put3('solve_v',solve_v,ntsi,ntsj)
      CALL put3('corrected_u',puu(ntsi:ntei,ntsj:ntej,1:jpkm1,Kaa),ntsi,ntsj)
      CALL put3('corrected_v',pvv(ntsi:ntei,ntsj:ntej,1:jpkm1,Kaa),ntsi,ntsj)
      CLOSE(nunit); nunit=-1; lactive=.FALSE.
      DEALLOCATE(before_u,before_v,rhs_u,rhs_v,avm_in,r3u_kmm_in,r3v_kmm_in, &
         & r3u_kaa_in,r3v_kaa_in,baro_u,baro_v, &
         & rcdu_bot_in,rcdu_top_in,utau_in,vtau_in,state_u,state_v,matrix,solve_u,solve_v)
   END SUBROUTINE r328_finish

END MODULE dynzdf_round328_writer
