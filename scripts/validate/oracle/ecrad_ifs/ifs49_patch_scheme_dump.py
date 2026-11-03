"""Harness dumps of the ecRad call inside the 49r1 RADIATION_SCHEME.

Inserts write-only DUMP_R/DUMP_I calls (module ECRAD49_DUMP, active only when
ECRAD49_STAGE_DUMP names a directory) immediately before and after the single
"CALL RADIATION(" statement of a COPY of arpifs/phys_radi/radiation_scheme.F90:
before = every input object exactly as handed to ecRad (prefix ifs_in_),
after = every allocated flux array ecRad returns (prefix ifs_out_). Only the
USE line and the DUMP calls are added; no oracle statement changes. Refuses
to patch if the anchors are not found exactly once.

usage: python -I ifs49_patch_scheme_dump.py <radiation_scheme.F90>   (edits in place)
"""
import sys

IN_R = {
    "SINGLE_LEVEL": "COS_SZA SKIN_TEMPERATURE SW_ALBEDO SW_ALBEDO_DIRECT LW_EMISSIVITY",
    "THERMODYNAMICS": "PRESSURE_HL TEMPERATURE_HL H2O_SAT_LIQ",
    "GAS": "MIXING_RATIO",
    "YLCLOUD": "Q_LIQ Q_ICE RE_LIQ RE_ICE FRACTION OVERLAP_PARAM FRACTIONAL_STD",
    "AEROSOL": "MIXING_RATIO",
}
OUT_R = ("LW_UP LW_DN SW_UP SW_DN LW_UP_CLEAR LW_DN_CLEAR SW_UP_CLEAR SW_DN_CLEAR SW_DN_DIRECT "
         "SW_DN_DIRECT_CLEAR LW_DERIVATIVES SW_DN_SURF_BAND SW_DN_DIRECT_SURF_BAND SW_DN_SURF_CLEAR_BAND "
         "SW_DN_DIRECT_SURF_CLEAR_BAND SW_DN_DIFFUSE_SURF_CANOPY SW_DN_DIRECT_SURF_CANOPY LW_DN_SURF_CANOPY "
         "CLOUD_COVER_LW CLOUD_COVER_SW")


def main(path):
    src = open(path).read()
    anchor_use = "USE RADIATION_SAVE,           ONLY : SAVE_INPUTS, SAVE_FLUXES\n"
    anchor_call = ("CALL RADIATION(KLON, KLEV, KIDIA, KFDIA, RAD_CONFIG,&\n"
                   "     &  SINGLE_LEVEL, THERMODYNAMICS, GAS, YLCLOUD, AEROSOL, FLUX)\n")
    for a in (anchor_use, anchor_call):
        if src.count(a) != 1:
            sys.exit(f"anchor not found exactly once: {a!r}")
    before = []
    # q_liq/q_ice/re_liq/re_ice are POINTER views into the cloud type's allocatable arrays
    ptr = {"Q_LIQ", "Q_ICE", "RE_LIQ", "RE_ICE"}
    for obj, names in IN_R.items():
        for n in names.split():
            test = "ASSOCIATED" if (obj == "YLCLOUD" and n in ptr) else "ALLOCATED"
            before.append(f"IF ({test}({obj}%{n})) CALL DUMP_R('ifs_in_{obj.lower()}_{n.lower()}', {obj}%{n})")
    before += [
        "CALL DUMP_R('ifs_in_single_level_solar_irradiance', SINGLE_LEVEL%SOLAR_IRRADIANCE)",
        "CALL DUMP_R('ifs_in_single_level_spectral_solar_cycle_multiplier', SINGLE_LEVEL%SPECTRAL_SOLAR_CYCLE_MULTIPLIER)",
        "IF (ALLOCATED(SINGLE_LEVEL%SPECTRAL_SOLAR_SCALING)) CALL DUMP_R('ifs_in_single_level_spectral_solar_scaling', SINGLE_LEVEL%SPECTRAL_SOLAR_SCALING)",
        "IF (ALLOCATED(SINGLE_LEVEL%ISEED)) CALL DUMP_I('ifs_in_single_level_iseed', SINGLE_LEVEL%ISEED)",
        "CALL DUMP_I('ifs_in_gas_iunits', GAS%IUNITS)",
        "CALL DUMP_R('ifs_in_rad_config_cloud_inhom_decorr_scaling', RAD_CONFIG%CLOUD_INHOM_DECORR_SCALING)",
    ]
    after = [f"IF (ALLOCATED(FLUX%{n})) CALL DUMP_R('ifs_out_flux_{n.lower()}', FLUX%{n})" for n in OUT_R.split()]
    src = src.replace(anchor_use, anchor_use + "USE ECRAD49_DUMP,             ONLY : DUMP_R, DUMP_I  ! harness dump (write-only)\n")
    src = src.replace(anchor_call, "! >>> harness dump (write-only)\n" + "\n".join(before) + "\n! <<<\n"
                      + anchor_call + "! >>> harness dump (write-only)\n" + "\n".join(after) + "\n! <<<\n")
    open(path, "w").write(src)


if __name__ == "__main__":
    main(sys.argv[1])
