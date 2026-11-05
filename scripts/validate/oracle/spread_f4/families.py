"""F4 family registry: committed generator + its args, oracle exes to intercept,
committed fixture, committed test + the test's own metric, tests that compare
port vs oracle (jit/gap/grad tests excluded: not oracle comparisons)."""
import os
WT = os.environ.get("F4_WT", "/burg/glab/users/pg2328/wt_f4")   # VOTE V32: stacked-branch worktrees
OR = "/burg/glab/users/pg2328/oracles"
OA = "/burg/glab/users/pg2328/oracle_audit"
B = "/burg/glab/users/pg2328/oracle_audit/f4/build"
G = f"{WT}/scripts/validate/oracle"
FX = f"{WT}/tests/atmosphere/oracle_fixtures"

FAMILIES = {
    "zm_detrain": dict(
        build=f"bash -x {G}/zm_detrain/build.sh {OR}/CAM {B}/zm_detrain",
        generator=f"{G}/zm_detrain/make_fixture.py",
        argv=["--driver", f"{B}/zm_detrain/detrain_driver", "--out", "{OUT}"],
        exes={"detrain_driver"}, fixture=f"{FX}/zm_detrain_cam.npz",
        test=f"{WT}/tests/atmosphere/test_zm_detrain_oracle.py", fix_attr="FIXTURE",
        metrics=["_rel"], select="hard_limit"),
    "kuo_mahfouf": dict(
        build="true",  # the generator builds the oracle itself (traced by gen_intercept)
        generator=f"{G}/kuo_mahfouf/make_fixture.py",
        argv=["--oracle", f"{OR}/MOIST_CONVECTION_KUO", "--zm-columns",
              f"{OA}/zm_deep_convection/inputs/columns.npz", "--out", "{OUT}"],
        exes={"kuo_harm"}, fixture=f"{FX}/kuo_mahfouf.npz",
        test=f"{WT}/tests/atmosphere/test_kuo_mahfouf_oracle.py", fix_attr="FIXTURE",
        metrics=["_err"], select="hard_limit"),
    "louis": dict(
        build=f"bash -x {G}/louis/build.sh {OR}/openifs {OR}/ICON {B}/louis",
        generator=f"{G}/louis/make_fixture.py",
        argv=["--inputs", f"{OA}/local_closures_louis_tke_clubb_lite/inputs.npz",
              "--build", f"{B}/louis", "--out", "{OUT}"],
        exes={"ifs_exe", "icon_exe"}, fixture=f"{FX}/louis.npz",
        test=f"{WT}/tests/atmosphere/hydrostatic/unit/test_louis_oracle.py", fix_attr="FIXTURE",
        metrics=["_err"], select="match_openifs or match_icon", exclude_src=r"ri_pin"),
    "kain_fritsch": dict(
        build=f"bash -x {G}/kain_fritsch/build.sh {OR}/WRF {B}/kain_fritsch",
        generator=f"{G}/kain_fritsch/make_fixture.py",
        argv=[f"{OA}/zm_deep_convection/inputs/columns.npz", f"{B}/kain_fritsch", "{OUT}"],
        exes={"kf_driver.x"}, fixture=f"{FX}/kain_fritsch.npz",
        test=f"{WT}/tests/atmosphere/test_kain_fritsch_oracle.py", fix_attr="_FIX",
        metrics=["_rel"], select="matches_wrf or internal_state or sweep_columns"),
    "mynn25": dict(
        build=(f"mkdir -p {B}/mynn25_const && $PY {G}/mynn25/make_fixture.py constants {B}/mynn25_const/module_model_constants.F90"
               f" && bash -x {G}/mynn25/build.sh {OR}/WRF {B}/mynn25 "
               f"{B}/mynn25_const/module_model_constants.F90"),
        generator=f"{G}/mynn25/make_fixture.py",
        argv=["run", "--audit", f"{OA}/ysu_pbl/inputs.npz", "--ci",
              f"{OR}/WRF/phys/MYNN-EDMF/tests/data", "--build", f"{B}/mynn25", "--out", "{OUT}"],
        exes={"mynn_driver"}, fixture=f"{FX}/mynn25.npz",
        test=f"{WT}/tests/atmosphere/hydrostatic/unit/test_mynn25_oracle.py", fix_attr="FIXTURE",
        metrics=["_err"], select="outputs_match or intermediates_match"),
    "rad_boundary": dict(
        build=f"bash -x {G}/rad_boundary/build.sh {OR}/cime5.6.47 {B}/rad_boundary",
        generator=f"{G}/rad_boundary/make_fixture.py",
        argv=[f"{B}/rad_boundary/oracle_driver", f"{OR}/cime5.6.47", "{OUT}"],
        exes={"oracle_driver"}, fixture=f"{FX}/rad_boundary.npz",
        test=f"{WT}/tests/atmosphere/test_rad_boundary_oracle.py", fix_attr="_FIX",
        metrics=["_rel"], select="match"),
}

_MG2 = f"{G}/morrison_mg2_warm_rain/make_fixture.py"
_MG2T = f"{WT}/tests/atmosphere/microphysics/unit"
for _tag, _flag, _test, _sel in (
        ("mg2_prog_nc", "--prognostic-nc", "test_morrison_prognostic_nc.py", "match"),
        ("mg2_prog_nc_act", "--prognostic-nc-act", "test_morrison_prognostic_nc_act.py",
         "budget_matches_mg2_stage"),
        ("mg2_prog_nc_sed", "--prognostic-nc-sed", "test_morrison_prognostic_nc_sed.py",
         "sedimentation_matches_mg2")):
    FAMILIES[_tag] = dict(
        build="true", generator=_MG2,
        argv=["--cam", f"{OR}/CAM", "--build", f"{B}/{_tag}/{{MODE}}", "--out", "{OUT}", _flag],
        exes={"mg2_driver", "mg2_tap", "mg2_procs", "mg2_rime", "mg2_ice"},
        fixture=f"{WT}/tests/atmosphere/microphysics/oracle_fixtures/"
                f"{'mg2_prognostic_nc' + _tag[len('mg2_prog_nc'):]}.npz",
        test=f"{_MG2T}/{_test}", fix_attr="FIXTURE", metrics=["_rel"], select=_sel,
        local_ref=True)   # VOTE V9 (b)

FAMILIES["mg2_prog_nc_sed"]["metrics"] = ["_rel", "_rel_flux"]

# F4 part 3: gray radiation and HS94 Rayleigh friction vs Isca v1.0 (VOTES V20-V23)
FAMILIES["gray_isca"] = dict(
    build=f"bash -x {G}/gray_isca/build.sh {OR}/Isca_v1.0 {B}/gray_isca",
    generator=f"{G}/gray_isca/make_fixture.py",
    argv=[f"{B}/gray_isca/drv_gray", f"{OR}/Isca_v1.0",
          f"{OA}/morrison_mg_warm_rain_sedimentation/out/columns.npz", "{OUT}"],
    exes={"drv_gray"}, fixture=f"{FX}/gray_isca.npz",
    test=f"{WT}/tests/atmosphere/test_gray_isca_oracle.py", fix_attr="FIXTURE",
    metrics=["_rel"], select="matches_isca")
FAMILIES["rayleigh_hs_isca"] = dict(
    build=f"bash -x {G}/hs_isca/build.sh {OR}/Isca_v1.0 {B}/hs_isca",
    generator=f"{G}/hs_isca/make_fixture.py",
    argv=[f"{B}/hs_isca/drv_hs", f"{OR}/Isca_v1.0",
          f"{OA}/simple_gwd_hines_lindzen_rayleigh_spectral/inputs/cols.npz", "{OUT}"],
    exes={"drv_hs"}, fixture=f"{FX}/rayleigh_hs_isca.npz",
    test=f"{WT}/tests/atmosphere/test_rayleigh_hs_isca_oracle.py", fix_attr="FIXTURE",
    metrics=["_rel"], select="matches_isca")
FAMILIES["hines_lmdz"] = dict(
    build=f"bash -x {G}/hines_lmdz/build.sh {OR}/LMDZ/modipsl/modeles/LMDZ/libf/phylmd {B}/hines_lmdz",
    generator=f"{G}/hines_lmdz/make_fixture.py",
    argv=[f"{B}/hines_lmdz/drv_hines", f"{OR}/LMDZ/modipsl/modeles/LMDZ/libf/phylmd",
          f"{OA}/simple_gwd_hines_lindzen_rayleigh_spectral/inputs/cols.npz", "{OUT}"],
    exes={"drv_hines"}, fixture=f"{FX}/hines_lmdz.npz",
    test=f"{WT}/tests/atmosphere/hydrostatic/unit/test_hines_lmdz_oracle.py", fix_attr="FIXTURE",
    metrics=["_rel"], select="matches_lmdz")


# Beres (Beres 2004 / E3SM gw_beres_src) on the real gw_drag_file table (VOTES V27-V33)
FAMILIES["gw_beres_table"] = dict(
    build=(f"set +u && source /etc/profile.d/modules.sh && module load netcdf-c/4.9.3 netcdf-fortran/4.6.2 && set -u"
           f" && bash -x {G}/gw_beres_table/build.sh {OR}/E3SM_v3.0.1 {B}/gw_beres_table"
           f" {OA}/nonorographic_gwd_e3sm_cam/fortran/src/interpolate_data.F90"),
    generator=f"{G}/gw_beres_table/make_fixture.py",
    argv=["--drv", f"{B}/gw_beres_table/gw_spec_oracle_nc", "--mfcc-file",
          f"{OR}/inputdata/atm/waccm/gw/newmfspectra40_dc25.nc", "--out", "{OUT}"],
    exes={"gw_spec_oracle_nc"}, fixture=f"{FX}/gw_spec_e3sm_beres_table.npz",
    test=f"{WT}/tests/atmosphere/hydrostatic/unit/test_gw_spec_e3sm_oracle.py",
    fix_attr="FIXTURE_TABLE", metrics=["_rel_err", "_rel_err_field"], select="beres_real_table")
