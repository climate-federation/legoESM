PROGRAM o5_si3_layout_fixture
   !! Template for the bounds-checked patched-ppsrc fixture.  run.sh replaces
   !! every O5_PATCHED_* marker with the corresponding CALL statement copied
   !! verbatim from the patched, preprocessed NEMO sources before compilation.
   USE, INTRINSIC :: ieee_arithmetic, ONLY : ieee_quiet_nan, ieee_value
   USE par_kind, ONLY : wp
   USE o5_si3_writer
   IMPLICIT NONE
   INTEGER, PARAMETER :: nx=184, ny=152, nc=5, nf=56
   INTEGER, PARAMETER :: Nis0=3, Nie0=182, Njs0=3, Nje0=150
   INTEGER, PARAMETER :: frames(6) = [1, 2, 3, 50, 99, 100]
   INTEGER :: i, j, k, kt, Kmm, jf, jter, mpprank
   INTEGER :: jpi, jpj, jpl, nn_hls, ncom_stp, nit000, jpni, jpnj, nn_npfchk
   INTEGER :: ntsi, ntei, ntsj, ntej
   LOGICAL :: ln_nnogather
   REAL(wp) :: rDt_ice, base, nan
   REAL(wp), TARGET :: bank(nf,nx,ny)
   REAL(wp), TARGET :: tmask_store(nx,ny,1), umask_store(nx,ny,1), vmask_store(nx,ny,1)
   REAL(wp) :: a_i(nx,ny,nc), v_i(nx,ny,nc)
   REAL(wp) :: pdivu_i(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: pshear_i(Nis0:Nie0,Njs0:Nje0)
   REAL(wp) :: pdelta_i(Nis0:Nie0,Njs0:Nje0)
   REAL(wp), POINTER :: fimask(:,:), at_i(:,:), strength(:,:)
   REAL(wp), POINTER :: tmask(:,:,:)
   REAL(wp), POINTER :: u_ice(:,:), v_ice(:,:), u_ice_b(:,:), v_ice_b(:,:)
   REAL(wp), POINTER :: zs1(:,:), zs2(:,:), zs12(:,:), vt_i(:,:), vt_s(:,:)
   REAL(wp), POINTER :: vt_ip(:,:), vt_il(:,:), utau_ice(:,:), vtau_ice(:,:)
   REAL(wp), POINTER :: drag_io(:,:), u_oce(:,:), v_oce(:,:), ssh_m(:,:)
   REAL(wp), POINTER :: snwice_mass(:,:), snwice_mass_b(:,:), zsshdyn(:,:), ff_t(:,:)
   REAL(wp), POINTER :: umask(:,:,:), vmask(:,:,:)
   REAL(wp), POINTER :: fast_umask(:,:), fast_vmask(:,:), fast_tmask(:,:)
   REAL(wp), POINTER :: icb_umask(:,:), icb_vmask(:,:), icb_tmask(:,:)
   REAL(wp), POINTER :: e1t(:,:), e2t(:,:), e1u(:,:), e2u(:,:), e1v(:,:), e2v(:,:)
   REAL(wp), POINTER :: e1f(:,:), e2f(:,:), e1e2t(:,:), e1e2u(:,:), e1e2v(:,:), e1e2f(:,:)
   REAL(wp), POINTER :: r1_e1u(:,:), r1_e2v(:,:), r1_e1e2t(:,:), r1_e1e2u(:,:)
   REAL(wp), POINTER :: r1_e1e2v(:,:), r1_e1e2f(:,:), zdelta(:,:), zp_delt(:,:)
   REAL(wp), POINTER :: pstress1_i(:,:), pstress2_i(:,:), pstress12_i(:,:)

   fimask=>bank(1,:,:); at_i=>bank(2,:,:); tmask=>tmask_store; strength=>bank(4,:,:)
   u_ice=>bank(5,:,:); v_ice=>bank(6,:,:); u_ice_b=>bank(7,:,:); v_ice_b=>bank(8,:,:)
   zs1=>bank(9,:,:); zs2=>bank(10,:,:); zs12=>bank(11,:,:); vt_i=>bank(12,:,:)
   vt_s=>bank(13,:,:); vt_ip=>bank(14,:,:); vt_il=>bank(15,:,:); utau_ice=>bank(16,:,:)
   vtau_ice=>bank(17,:,:); drag_io=>bank(18,:,:); u_oce=>bank(19,:,:); v_oce=>bank(20,:,:)
   ssh_m=>bank(21,:,:); snwice_mass=>bank(22,:,:); snwice_mass_b=>bank(23,:,:)
   zsshdyn=>bank(24,:,:); ff_t=>bank(25,:,:)
   umask=>umask_store; vmask=>vmask_store
   fast_umask=>bank(28,:,:); fast_vmask=>bank(29,:,:); fast_tmask=>bank(30,:,:)
   icb_umask=>bank(31,:,:); icb_vmask=>bank(32,:,:); icb_tmask=>bank(33,:,:)
   e1t=>bank(34,:,:); e2t=>bank(35,:,:); e1u=>bank(36,:,:); e2u=>bank(37,:,:)
   e1v=>bank(38,:,:); e2v=>bank(39,:,:); e1f=>bank(40,:,:); e2f=>bank(41,:,:)
   e1e2t=>bank(42,:,:); e1e2u=>bank(43,:,:); e1e2v=>bank(44,:,:); e1e2f=>bank(45,:,:)
   r1_e1u=>bank(46,:,:); r1_e2v=>bank(47,:,:); r1_e1e2t=>bank(48,:,:)
   r1_e1e2u=>bank(49,:,:); r1_e1e2v=>bank(50,:,:); r1_e1e2f=>bank(51,:,:)
   zdelta=>bank(52,:,:); zp_delt=>bank(53,:,:); pstress1_i=>bank(54,:,:)
   pstress2_i=>bank(55,:,:); pstress12_i=>bank(56,:,:)

   jpi=nx; jpj=ny; jpl=nc; nn_hls=2; ncom_stp=1; nit000=1
   jpni=2; jpnj=1; nn_npfchk=2; ln_nnogather=.TRUE.; rDt_ice=10800._wp
   ntsi=Nis0; ntei=Nie0; ntsj=Njs0; ntej=Nje0; Kmm=2
   nan=ieee_value(0._wp,ieee_quiet_nan)
   DO mpprank=0,1
      base=REAL(mpprank,wp)+1._wp
      bank=base
      tmask=base; umask=base; vmask=base
      pdivu_i=base; pshear_i=base; pdelta_i=base
      DO k=1,nc; DO j=1,ny; DO i=1,nx
         a_i(i,j,k)=base+REAL(10000*k+100*j+i,wp)
         v_i(i,j,k)=-a_i(i,j,k)
      END DO; END DO; END DO
      fast_umask=nan; fast_vmask=nan; icb_umask=nan; icb_vmask=nan
      fast_umask(1:nx-1,1:ny-1)=base; fast_vmask(1:nx-1,1:ny-1)=base
      icb_umask(1:nx-1,1:ny-1)=base; icb_vmask(1:nx-1,1:ny-1)=base
      at_i(3,3)=-0._wp
      DO kt=1,3
! O5_PATCHED_CONTEXT
         IF (kt == 1) THEN
! O5_PATCHED_FIMASK
         END IF
! O5_PATCHED_STRENGTH_ENTRY
! O5_PATCHED_STRENGTH_RAW
! O5_PATCHED_STRENGTH_SMOOTH
         zs1=base; zs2=base; zs12=base
! O5_PATCHED_AEVP_ENTRY_STATE
! O5_PATCHED_AEVP_ENTRY_MASKS
! O5_PATCHED_AEVP_ENTRY_METRICS
         zs1=nan; zs2=nan; zs12=nan
         zs1(2:nx,2:ny)=base; zs2(2:nx,2:ny)=base; zs12(1:nx-1,1:ny-1)=base
         DO jf=1,SIZE(frames)
            jter=frames(jf)
! O5_PATCHED_FRAME_DELTA
! O5_PATCHED_FRAME_STRESS
! O5_PATCHED_FRAME_VELOCITY
         END DO
! O5_PATCHED_AEVP_EXIT
! O5_PATCHED_RDGRFT_ENTRY
! O5_PATCHED_RDGRFT_EXIT
      END DO
   END DO
END PROGRAM o5_si3_layout_fixture
