MODULE tke_round331_writer
   !! WRITE-only kt=1 TKE source-order recorder for ORCA2 round 331.
   !! Captured values are never read back into NEMO state.
   USE dom_oce
   USE in_out_manager, ONLY : nit000
   USE dynzdf_round328_writer, ONLY : checksum2, checksum3
   USE lib_mpp, ONLY : ctl_stop
   IMPLICIT NONE
   PRIVATE
   PUBLIC :: r331_begin, r331_sh2, r331_drag, r331_mxl, r331_tke_inputs
   PUBLIC :: r331_after_bc, r331_after_matrix, r331_after_solve, r331_finish

   INTEGER, SAVE :: nunit=-1,nx=0,ny=0,nz=0
   INTEGER, SAVE :: nkt=-1,nkbb=-1,nkmm=-1,nkrhs=-1
   INTEGER, SAVE :: ninput=0,nbc=0,nmatrix=0,nsolve=0
   LOGICAL, SAVE :: lactive=.FALSE.,lsh2=.FALSE.,ldrag(2)=.FALSE.,lmxl=.FALSE.
   REAL(wp), ALLOCATABLE, SAVE :: sh2_output(:,:,:),drag_output(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: mxl_output(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: sh2_input(:,:,:),drag_input(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: mxl_input(:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: after_bc(:,:,:),matrix(:,:,:,:)
   REAL(wp), ALLOCATABLE, SAVE :: after_solve(:,:,:)

CONTAINS

   SUBROUTINE put2(name,value,k0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:)
      INTEGER, INTENT(in) :: k0
      CHARACTER(LEN=24) :: field
      field=name
      WRITE(nunit) field
      WRITE(nunit) 2,SIZE(value,1),SIZE(value,2),1,ntsi,ntsj,k0
      WRITE(nunit) checksum2(value)
      WRITE(nunit) value
   END SUBROUTINE put2

   SUBROUTINE put3(name,value,k0)
      CHARACTER(LEN=*), INTENT(in) :: name
      REAL(wp), INTENT(in) :: value(:,:,:)
      INTEGER, INTENT(in) :: k0
      CHARACTER(LEN=24) :: field
      field=name
      WRITE(nunit) field
      WRITE(nunit) 3,SIZE(value,1),SIZE(value,2),SIZE(value,3),ntsi,ntsj,k0
      WRITE(nunit) checksum3(value)
      WRITE(nunit) value
   END SUBROUTINE put3

   SUBROUTINE r331_begin(kt,Kbb,Kmm,Krhs,pinterior)
      INTEGER, INTENT(in) :: kt,Kbb,Kmm,Krhs
      REAL(wp), INTENT(in) :: pinterior(:,:,:)
      lactive=kt == nit000 .AND. Kbb == 1 .AND. Kmm == 1 .AND. Krhs == 3
      IF(.NOT.lactive) RETURN
      IF(ALLOCATED(sh2_output)) CALL ctl_stop('round331 TKE: nested record')
      IF(STORAGE_SIZE(1._wp) /= 64) CALL ctl_stop('round331 TKE: requires fp64')
      nx=SIZE(pinterior,1); ny=SIZE(pinterior,2); nz=SIZE(pinterior,3)
      ALLOCATE(sh2_output(nx,ny,nz),drag_output(2,nx,ny), &
         & mxl_output(2,nx,ny),sh2_input(nx,ny,nz),drag_input(2,nx,ny), &
         & mxl_input(2,nx,ny),after_bc(nx,ny,nz+1), &
         & matrix(4,nx,ny,nz+1),after_solve(nx,ny,nz+1))
      sh2_output=0._wp; drag_output=0._wp; mxl_output=0._wp
      sh2_input=0._wp; drag_input=0._wp; mxl_input=0._wp
      after_bc=0._wp; matrix=0._wp; after_solve=0._wp
      nkt=kt; nkbb=Kbb; nkmm=Kmm; nkrhs=Krhs
      ninput=0; nbc=0; nmatrix=0; nsolve=0
      lsh2=.FALSE.; ldrag=.FALSE.; lmxl=.FALSE.
   END SUBROUTINE r331_begin

   SUBROUTINE r331_sh2(value)
      REAL(wp), INTENT(in) :: value(:,:,:)
      IF(.NOT.lactive) RETURN
      IF(ANY(SHAPE(value) /= (/nx,ny,nz/))) CALL ctl_stop('round331 TKE: sh2 shape')
      sh2_output=value; lsh2=.TRUE.
   END SUBROUTINE r331_sh2

   SUBROUTINE r331_drag(which,value)
      INTEGER, INTENT(in) :: which
      REAL(wp), INTENT(in) :: value(:,:)
      IF(.NOT.lactive) RETURN
      IF(which<1 .OR. which>2 .OR. ANY(SHAPE(value) /= (/nx,ny/))) &
         & CALL ctl_stop('round331 TKE: drag shape')
      drag_output(which,:,:)=value; ldrag(which)=.TRUE.
   END SUBROUTINE r331_drag

   SUBROUTINE r331_mxl(pnmln,phmlp)
      INTEGER, INTENT(in) :: pnmln(:,:)
      REAL(wp), INTENT(in) :: phmlp(:,:)
      IF(.NOT.lactive) RETURN
      IF(ANY(SHAPE(pnmln) /= (/nx,ny/)) .OR. ANY(SHAPE(phmlp) /= (/nx,ny/))) &
         & CALL ctl_stop('round331 TKE: mxl shape')
      mxl_output(1,:,:)=REAL(pnmln,wp); mxl_output(2,:,:)=phmlp
      lmxl=.TRUE.
   END SUBROUTINE r331_mxl

   SUBROUTINE r331_tke_inputs(jj,psh2,pbottom,ptop,pnmln,phmlp)
      INTEGER, INTENT(in) :: jj
      REAL(wp), INTENT(in) :: psh2(:,:),pbottom(:),ptop(:),phmlp(:)
      INTEGER, INTENT(in) :: pnmln(:)
      INTEGER :: j
      IF(.NOT.lactive) RETURN
      j=jj-ntsj+1
      IF(j<1 .OR. j>ny .OR. ANY(SHAPE(psh2) /= (/nx,nz/)) .OR. &
         & SIZE(pbottom)/=nx .OR. SIZE(ptop)/=nx .OR. SIZE(pnmln)/=nx .OR. &
         & SIZE(phmlp)/=nx) CALL ctl_stop('round331 TKE: input row shape')
      sh2_input(:,j,:)=psh2; drag_input(1,:,j)=pbottom; drag_input(2,:,j)=ptop
      mxl_input(1,:,j)=REAL(pnmln,wp); mxl_input(2,:,j)=phmlp
      ninput=ninput+1
   END SUBROUTINE r331_tke_inputs

   SUBROUTINE r331_after_bc(jj,pen)
      INTEGER, INTENT(in) :: jj
      REAL(wp), INTENT(in) :: pen(:,:)
      INTEGER :: j
      IF(.NOT.lactive) RETURN
      j=jj-ntsj+1
      IF(j<1 .OR. j>ny .OR. ANY(SHAPE(pen) /= (/nx,nz+1/))) &
         & CALL ctl_stop('round331 TKE: BC row shape')
      after_bc(:,j,:)=pen; nbc=nbc+1
   END SUBROUTINE r331_after_bc

   SUBROUTINE r331_after_matrix(jj,pen,pdiag,pup,plow)
      INTEGER, INTENT(in) :: jj
      REAL(wp), INTENT(in) :: pen(:,:),pdiag(:,:),pup(:,:),plow(:,:)
      INTEGER :: j
      IF(.NOT.lactive) RETURN
      j=jj-ntsj+1
      IF(j<1 .OR. j>ny .OR. ANY(SHAPE(pen) /= (/nx,nz+1/)) .OR. &
         & ANY(SHAPE(pdiag) /= SHAPE(pen)) .OR. ANY(SHAPE(pup) /= SHAPE(pen)) .OR. &
         & ANY(SHAPE(plow) /= SHAPE(pen))) CALL ctl_stop('round331 TKE: matrix row shape')
      matrix(1,:,j,:)=pen; matrix(2,:,j,:)=pdiag
      matrix(3,:,j,:)=pup; matrix(4,:,j,:)=plow
      nmatrix=nmatrix+1
   END SUBROUTINE r331_after_matrix

   SUBROUTINE r331_after_solve(jj,pen)
      INTEGER, INTENT(in) :: jj
      REAL(wp), INTENT(in) :: pen(:,:)
      INTEGER :: j
      IF(.NOT.lactive) RETURN
      j=jj-ntsj+1
      IF(j<1 .OR. j>ny .OR. ANY(SHAPE(pen) /= (/nx,nz+1/))) &
         & CALL ctl_stop('round331 TKE: solve row shape')
      after_solve(:,j,:)=pen; nsolve=nsolve+1
   END SUBROUTINE r331_after_solve

   SUBROUTINE r331_finish(pen,pavm,pavt,pwmask)
      REAL(wp), INTENT(in) :: pen(:,:,:),pavm(:,:,:),pavt(:,:,:),pwmask(:,:,:)
      INTEGER :: ios
      CHARACTER(LEN=16) :: magic
      CHARACTER(LEN=96) :: filename
      IF(.NOT.lactive) RETURN
      IF(.NOT.lsh2 .OR. .NOT.ALL(ldrag) .OR. .NOT.lmxl) &
         & CALL ctl_stop('round331 TKE: incomplete input outputs')
      IF(ninput/=ny .OR. nbc/=ny .OR. nmatrix/=ny .OR. nsolve/=ny) &
         & CALL ctl_stop('round331 TKE: incomplete row capture')
      IF(ANY(SHAPE(pen) /= (/nx,ny,nz+1/)) .OR. &
         & ANY(SHAPE(pavm) /= (/nx,ny,nz/)) .OR. ANY(SHAPE(pavt) /= (/nx,ny,nz/)) .OR. &
         & ANY(SHAPE(pwmask) /= (/nx,ny,nz+1/))) CALL ctl_stop('round331 TKE: finish shape')
      WRITE(filename,'("oracle_tke331_kt",I8.8,"_r",I4.4,".bin")') nkt,narea-1
      OPEN(NEWUNIT=nunit,FILE=TRIM(filename),ACCESS='STREAM',FORM='UNFORMATTED', &
         & STATUS='REPLACE',ACTION='WRITE',IOSTAT=ios)
      IF(ios /= 0) CALL ctl_stop('round331 TKE: cannot open record')
      magic='NEMO_L4_R331TK'
      WRITE(nunit) magic,1,nkt,nkbb,nkmm,nkrhs,narea-1,nx,ny,nz,ntsi,ntsj, &
         & STORAGE_SIZE(1._wp),20
      CALL put3('sh2_output',sh2_output,2)
      CALL put2('drag_bottom_output',drag_output(1,:,:),1)
      CALL put2('drag_top_output',drag_output(2,:,:),1)
      CALL put2('mxl_nmln_output',mxl_output(1,:,:),1)
      CALL put2('mxl_hmlp_output',mxl_output(2,:,:),1)
      CALL put3('tke_sh2_input',sh2_input,2)
      CALL put2('tke_drag_bottom_input',drag_input(1,:,:),1)
      CALL put2('tke_drag_top_input',drag_input(2,:,:),1)
      CALL put2('tke_nmln_input',mxl_input(1,:,:),1)
      CALL put2('tke_hmlp_input',mxl_input(2,:,:),1)
      CALL put3('after_bc_en',after_bc,1)
      CALL put3('matrix_rhs_en',matrix(1,:,:,:),1)
      CALL put3('matrix_diag',matrix(2,:,:,:),1)
      CALL put3('matrix_upper',matrix(3,:,:,:),1)
      CALL put3('matrix_lower',matrix(4,:,:,:),1)
      CALL put3('after_solve_en',after_solve,1)
      CALL put3('after_tke_en',pen,1)
      CALL put3('after_tke_avm',pavm,2)
      CALL put3('after_tke_avt',pavt,2)
      CALL put3('tke_wmask',pwmask,1)
      CLOSE(nunit); nunit=-1
      DEALLOCATE(sh2_output,drag_output,mxl_output,sh2_input,drag_input,mxl_input, &
         & after_bc,matrix,after_solve)
      lactive=.FALSE.
   END SUBROUTINE r331_finish

END MODULE tke_round331_writer
