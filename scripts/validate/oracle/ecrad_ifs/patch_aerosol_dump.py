"""Harness stage dumps around add_aerosol_optics in the 49r1 radiation_interface.F90.

Writes the gas optics entering the aerosol stage and the combined gas+aerosol
optics leaving it (od_sw, ssa_sw, g_sw, od_lw; Fortran shape (ng, nlev, col))
as aer_in_<var> / aer_out_<var>, plus the relative-humidity inputs
thermodynamics%h2o_sat_liq and pressure_hl (module ECRAD49_DUMP, active only
when ECRAD49_STAGE_DUMP is set). Only WRITE statements are added; no oracle
statement changes. The anchor is the single non-monochromatic
"call add_aerosol_optics(" of radiation_interface.F90.

usage: python -I patch_aerosol_dump.py <radiation_interface.F90>   (edits in place)
"""
import re
import sys

IN_VARS = ("od_sw", "ssa_sw", "od_lw")
OUT_VARS = ("od_sw", "ssa_sw", "g_sw", "od_lw")


def main(path):
    lines = open(path).read().split("\n")
    out, n_call, n_use = [], 0, 0
    i = 0
    while i < len(lines):
        line = lines[i]
        if re.match(r"\s*use radiation_aerosol_optics, only : add_aerosol_optics\s*$", line):
            out.append(line)
            out.append("    use ecrad49_dump, only : dump_r")
            n_use += 1
            i += 1
            continue
        if re.match(r"\s*call add_aerosol_optics\(nlev,istartcol,iendcol,", line):
            ind = re.match(r"(\s*)", line).group(1)
            out += [f"{ind}call dump_r('aer_in_{v}', {v}(:,:,istartcol:iendcol))" for v in IN_VARS]
            out += [f"{ind}call dump_r('aer_in_h2o_sat_liq', thermodynamics%h2o_sat_liq(istartcol:iendcol,:))",
                    f"{ind}call dump_r('aer_in_pressure_hl', thermodynamics%pressure_hl(istartcol:iendcol,:))",
                    f"{ind}call dump_r('aer_in_mixing_ratio', aerosol%mixing_ratio(istartcol:iendcol,:,:))"]
            out.append(line)
            while lines[i].rstrip().endswith("&"):
                i += 1
                out.append(lines[i])
            out += [f"{ind}call dump_r('aer_out_{v}', {v}(:,:,istartcol:iendcol))" for v in OUT_VARS]
            n_call += 1
            i += 1
            continue
        out.append(line)
        i += 1
    if (n_use, n_call) != (1, 1):
        raise SystemExit(f"{path}: anchors found use={n_use} call={n_call}, expected 1 and 1")
    open(path, "w").write("\n".join(out))


if __name__ == "__main__":
    main(sys.argv[1])
