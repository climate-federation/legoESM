PROGRAM o5_si3_layout_fixture
   !! Bounds-checked executable fixture for every O5 observer path.
   USE par_kind, ONLY : wp
   USE o5_si3_writer
   IMPLICIT NONE
   INTEGER, PARAMETER :: nx=8, ny=7, nc=5
   INTEGER, PARAMETER :: frames(6) = [1, 2, 3, 50, 99, 100]
   INTEGER :: i, j, k, kt, jf
   REAL(wp) :: a3(nx,ny,nc), b3(nx,ny,nc)
   REAL(wp) :: f(64,nx,ny)

   DO k=1,nc; DO j=1,ny; DO i=1,nx
      a3(i,j,k) = REAL(10000*k + 100*j + i,wp)
      b3(i,j,k) = -a3(i,j,k)
   END DO; END DO; END DO
   DO k=1,64; DO j=1,ny; DO i=1,nx
      f(k,i,j) = REAL(100000*k + 100*j + i,wp)
   END DO; END DO; END DO
   f(1,3,3) = -0.0_wp

   DO kt=1,3
      CALL o5_context(kt, 2, 0, nx, ny, nc, 2, 1, 1, 1, 1, .TRUE., 2, 10800.0_wp)
      IF (kt == 1) CALL o5_fimask(f(1,:,:))
      CALL o5_strength_entry(a3,b3,f(2,:,:),f(3,:,:))
      CALL o5_strength_raw(f(4,:,:),f(5,:,:))
      CALL o5_strength_smooth(f(6,:,:))
      CALL o5_aevp_entry_state(f(7,:,:),f(8,:,:),f(9,:,:),f(10,:,:),f(11,:,:),f(12,:,:), &
         f(13,:,:),f(14,:,:),f(15,:,:),f(16,:,:),f(17,:,:),f(18,:,:),f(19,:,:),f(20,:,:), &
         f(21,:,:),f(22,:,:),f(23,:,:),f(24,:,:),f(25,:,:),f(26,:,:),f(27,:,:),f(28,:,:),f(29,:,:))
      CALL o5_aevp_entry_masks(f(30,:,:),f(31,:,:),f(32,:,:),f(33,:,:),f(34,:,:), &
         f(35,:,:),f(36,:,:),f(37,:,:),f(38,:,:))
      CALL o5_aevp_entry_metrics(f(39,:,:),f(40,:,:),f(41,:,:),f(42,:,:),f(43,:,:),f(44,:,:), &
         f(45,:,:),f(46,:,:),f(47,:,:),f(48,:,:),f(49,:,:),f(50,:,:),f(51,:,:),f(52,:,:), &
         f(53,:,:),f(54,:,:),f(55,:,:),f(56,:,:))
      DO jf=1,SIZE(frames)
         CALL o5_frame_delta(frames(jf),f(57,:,:),f(58,:,:))
         CALL o5_frame_stress(frames(jf),f(59,:,:),f(60,:,:),f(61,:,:))
         CALL o5_frame_velocity(frames(jf),f(62,:,:),f(63,:,:))
      END DO
      CALL o5_aevp_exit(f(1,:,:),f(2,:,:),f(3,:,:),f(4,:,:),f(5,:,:),f(6,:,:),f(7,:,:),f(8,:,:))
      CALL o5_rdgrft_entry(a3,b3)
      CALL o5_rdgrft_exit(a3,b3)
   END DO
END PROGRAM o5_si3_layout_fixture
