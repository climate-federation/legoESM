! Column driver for the WRF v4.8.0 km_opt=3 Smagorinsky eddy coefficients.
! A doubly periodic 3x3 tile of IDENTICAL columns (horizontally homogeneous, flat,
! map factors 1, w = 0) is handed to the unmodified module_diffusion_em routines
! compute_diff_metrics -> cal_deform_and_div -> calculate_km_kh (km_opt = 3,
! damp_opt = 0, mix_isotropic = 0).  Eta-level metrics (dnw, dn, fnm, fnp, cf1-3)
! follow dyn_em/module_initialize_scm_xy.F; th_phy and the surface p8w, t8w
! follow phy_prep.  Input / output: plain text, one column after another, bottom-up.
PROGRAM drv_smag
  USE module_big_step_utilities_em, ONLY: grid_config_rec_type
  USE module_model_constants, ONLY: g, rcp, p1000mb
  USE module_diffusion_em, ONLY: compute_diff_metrics, cal_deform_and_div, calculate_km_kh
  IMPLICIT NONE
  TYPE(grid_config_rec_type) :: cf
  INTEGER :: ncol, nlev, ic, k, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme
  INTEGER :: its, ite, jts, jte, kts, kte, i, j
  INTEGER, PARAMETER :: NMOIST = 4, NNBA = 9
  REAL :: dt, c_s, mub, cf1, cf2, cf3, cof1, cof2, z0, z1, z2, w1, w2
  REAL, ALLOCATABLE :: zw(:), uc(:), vc(:), tc(:), pc(:), pw(:), qv(:), qc(:), qi(:), znw(:)
  REAL, ALLOCATABLE :: dnw(:), dn(:), fnm(:), fnp(:), ubase(:), vbase(:)
  REAL, ALLOCATABLE, DIMENSION(:,:,:) :: u, v, w, ph, phb, z, rdz, rdzw, zx, zy, div, &
       d11, d22, d33, d12, d13, d23, th, t, p, p8w, t8w, xkmh, xkmv, xkhh, xkhv, bn2, tke, &
       dlk, xkmv_meso
  REAL, ALLOCATABLE :: moist(:,:,:,:), nba(:,:,:,:), msf(:,:), hpbl(:,:)
  CHARACTER(LEN=512) :: fin, fout

  CALL get_command_argument(1, fin)
  CALL get_command_argument(2, fout)
  OPEN(10, FILE=fin, STATUS='old')
  OPEN(11, FILE=fout, STATUS='replace')
  READ(10,*) ncol, nlev
  READ(10,*) dt, c_s, mub
  cf%c_s = c_s
  ids = 1; ide = 4; jds = 1; jde = 4; kds = 1; kde = nlev + 1
  its = 1; ite = 4; jts = 1; jte = 4; kts = 1; kte = nlev + 1
  ims = -2; ime = 7; jms = -2; jme = 7; kms = 1; kme = nlev + 1
  ALLOCATE(zw(kme), uc(nlev), vc(nlev), tc(nlev), pc(nlev), pw(kme), qv(nlev), qc(nlev), &
           qi(nlev), znw(kme), dnw(kme), dn(kme), fnm(kme), fnp(kme), ubase(kme), vbase(kme))
  ALLOCATE(u(ims:ime,kms:kme,jms:jme)); u = 0.0
  v = u; w = u; ph = u; phb = u; z = u; rdz = u; rdzw = u; zx = u; zy = u; div = u
  d11 = u; d22 = u; d33 = u; d12 = u; d13 = u; d23 = u; th = u; t = u; p = u; p8w = u
  t8w = u; xkmh = u; xkmv = u; xkhh = u; xkhv = u; bn2 = u; tke = u; dlk = u; xkmv_meso = u
  ALLOCATE(moist(ims:ime,kms:kme,jms:jme,NMOIST), nba(ims:ime,kms:kme,jms:jme,NNBA), &
           msf(ims:ime,jms:jme), hpbl(ims:ime,jms:jme))
  msf = 1.0; hpbl = 0.0; ubase = 0.0; vbase = 0.0

  DO ic = 1, ncol
    READ(10,*) zw
    READ(10,*) uc
    READ(10,*) vc
    READ(10,*) tc
    READ(10,*) pc
    READ(10,*) pw
    READ(10,*) qv
    READ(10,*) qc
    READ(10,*) qi
    READ(10,*) znw
    ! eta metrics: module_initialize_scm_xy.F lines 333-349
    dnw = 0.0; dn = 0.0; fnm = 0.0; fnp = 0.0
    DO k = 1, kde - 1
      dnw(k) = znw(k+1) - znw(k)
    END DO
    DO k = 2, kde - 1
      dn(k) = 0.5*(dnw(k) + dnw(k-1))
      fnp(k) = .5*dnw(k)/dn(k)
      fnm(k) = .5*dnw(k-1)/dn(k)
    END DO
    cof1 = (2.*dn(2)+dn(3))/(dn(2)+dn(3))*dnw(1)/dn(2)
    cof2 =     dn(2)        /(dn(2)+dn(3))*dnw(1)/dn(3)
    cf1  = fnp(2) + cof1
    cf2  = fnm(2) - cof1 - cof2
    cf3  = cof2
    u = 0.0; v = 0.0; w = 0.0; phb = 0.0; moist = 0.0; nba = 0.0; tke = 0.0; dlk = 0.0
    p8w = 0.0; t8w = 0.0
    DO j = jms, jme
    DO i = ims, ime
      DO k = 1, kde
        ph(i,k,j) = g*zw(k)
        p8w(i,k,j) = pw(k)
      END DO
      DO k = 1, nlev
        u(i,k,j) = uc(k); v(i,k,j) = vc(k); t(i,k,j) = tc(k); p(i,k,j) = pc(k)
        th(i,k,j) = tc(k)/(pc(k)/p1000mb)**rcp          ! phy_prep th_phy
        moist(i,k,j,2) = qv(k); moist(i,k,j,3) = qc(k); moist(i,k,j,4) = qi(k)
      END DO
    END DO
    END DO
    CALL compute_diff_metrics(cf, ph, phb, z, rdz, rdzw, zx, zy, 1.0, 1.0, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    ! phy_prep (module_big_step_utilities_em.F 4915-4923): surface p8w, t8w
    ! extrapolated linearly in z from the two lowest mass levels
    z0 = zw(1); z1 = 0.5*(zw(1) + zw(2)); z2 = 0.5*(zw(2) + zw(3))
    w1 = (z0 - z2)/(z1 - z2)
    w2 = 1. - w1
    p8w(:,1,:) = w1*pc(1) + w2*pc(2)
    t8w(:,1,:) = w1*tc(1) + w2*tc(2)
    zx = 0.0; zy = 0.0      ! flat: exact zeros (the metric differences are of identical columns)
    CALL cal_deform_and_div(cf, u, v, w, div, d11, d22, d33, d12, d13, d23, nba, NNBA, &
         ubase, vbase, msf, msf, msf, msf, msf, msf, 1.0, 1.0, dn, dnw, rdz, rdzw, &
         fnm, fnp, cf1, cf2, cf3, zx, zy, &
         ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, its, ite, jts, jte, kts, kte)
    CALL calculate_km_kh(cf, dt, 0.0, 0.0, 0, xkmh, xkmv, xkhh, xkhv, bn2, 0.0, 0.0, div, &
         d11, d22, d33, d12, d13, d23, tke, p8w, t8w, th, t, p, moist, dn, dnw, 1.0, 1.0, &
         rdz, rdzw, 0, NMOIST, cf1, cf2, cf3, .FALSE., mub, msf, msf, zx, zy, hpbl, dlk, &
         xkmv_meso, ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
         its, ite, jts, jte, kts, kte)
    WRITE(11,'(A,I6)') 'COL', ic
    WRITE(11,'(*(ES26.17E3))') d13(2,1:kde,2)
    WRITE(11,'(*(ES26.17E3))') d23(2,1:kde,2)
    WRITE(11,'(*(ES26.17E3))') bn2(2,1:nlev,2)
    WRITE(11,'(*(ES26.17E3))') xkmv(2,1:nlev,2)
    WRITE(11,'(*(ES26.17E3))') xkhv(2,1:nlev,2)
    WRITE(11,'(*(ES26.17E3))') cf1, cf2, cf3, t8w(2,1,2)
  END DO
  CLOSE(10); CLOSE(11)
  PRINT '(A)', 'SMAG_DRV_OK'
END PROGRAM drv_smag
