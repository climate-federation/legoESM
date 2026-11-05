! Minimal stand-ins for the WRF framework modules that dyn_em/module_diffusion_em.F
! (WRF v4.8.0) USEs.  Only the subroutines the harness calls are exercised
! (compute_diff_metrics, cal_deform_and_div, calculate_km_kh -> calculate_N2 +
! smag_km); the rest of the module only needs these names to compile.

MODULE module_state_description
  IMPLICIT NONE
  INTEGER, PARAMETER :: p_m11 = 1, p_m22 = 2, p_m33 = 3, p_m12 = 4, p_m13 = 5, &
                        p_m23 = 6, p_r12 = 7, p_r13 = 8, p_r23 = 9
  INTEGER, PARAMETER :: P_QNS = 1, P_QNR = 1, P_QNG = 1, P_QT = 1, P_QNH = 1, P_QVOLG = 1
END MODULE module_state_description

MODULE module_big_step_utilities_em
  IMPLICIT NONE
  ! WRF moist() species slots: PARAM_FIRST_SCALAR = 2 (slot 1 is a dummy).
  INTEGER, PARAMETER :: param_first_scalar = 2, p_qv = 2, p_qc = 3, p_qi = 4
  TYPE grid_config_rec_type
    LOGICAL :: periodic_x = .TRUE., periodic_y = .TRUE., open_xs = .FALSE., &
               open_xe = .FALSE., open_ys = .FALSE., open_ye = .FALSE., &
               specified = .FALSE., nested = .FALSE., polar = .FALSE., &
               mix_full_fields = .TRUE., chem_mix2_off = .FALSE., &
               moist_mix2_off = .FALSE., scalar_mix2_off = .FALSE., &
               tke_mix2_off = .FALSE., tracer_mix2_off = .FALSE.
    INTEGER :: bl_pbl_physics = 0, cu_physics = 0, diff_opt = 2, isfflx = 0, &
               km_opt = 3, m_opt = 0, sfs_opt = 0, shcu_physics = 0, &
               spec_bdy_width = 0, use_theta_m = 0
    REAL :: c_k = 0.15, c_s = 0.25, tke_drag_coefficient = 0.0, tke_heat_flux = 0.0
  END TYPE grid_config_rec_type
END MODULE module_big_step_utilities_em

MODULE module_bc
  IMPLICIT NONE
CONTAINS
  SUBROUTINE set_physical_bc3d(dat, variable, config_flags, &
                               ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                               its, ite, jts, jte, kts, kte, its2, ite2, jts2, jte2, kts2, kte2)
    USE module_big_step_utilities_em, ONLY: grid_config_rec_type
    INTEGER, INTENT(IN) :: ids, ide, jds, jde, kds, kde, ims, ime, jms, jme, kms, kme, &
                           its, ite, jts, jte, kts, kte, its2, ite2, jts2, jte2, kts2, kte2
    REAL, DIMENSION(ims:ime, kms:kme, jms:jme), INTENT(INOUT) :: dat
    CHARACTER(LEN=1), INTENT(IN) :: variable
    TYPE(grid_config_rec_type), INTENT(IN) :: config_flags
    STOP 'set_physical_bc3d is not on the harness path'
  END SUBROUTINE set_physical_bc3d
  SUBROUTINE set_physical_bc2d(dat, variable, config_flags, &
                               ids, ide, jds, jde, ims, ime, jms, jme, &
                               its, ite, jts, jte, its2, ite2, jts2, jte2)
    USE module_big_step_utilities_em, ONLY: grid_config_rec_type
    INTEGER, INTENT(IN) :: ids, ide, jds, jde, ims, ime, jms, jme, &
                           its, ite, jts, jte, its2, ite2, jts2, jte2
    REAL, DIMENSION(ims:ime, jms:jme), INTENT(INOUT) :: dat
    CHARACTER(LEN=1), INTENT(IN) :: variable
    TYPE(grid_config_rec_type), INTENT(IN) :: config_flags
    STOP 'set_physical_bc2d is not on the harness path'
  END SUBROUTINE set_physical_bc2d
END MODULE module_bc

! module_diffusion_em calls wrf_error_fatal as an EXTERNAL procedure.
SUBROUTINE wrf_error_fatal(str)
  CHARACTER(LEN=*), INTENT(IN) :: str
  WRITE(0,'(A)') 'WRF_FATAL: '//TRIM(str)
  STOP 3
END SUBROUTINE wrf_error_fatal
