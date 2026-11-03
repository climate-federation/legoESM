"""Harness stage dumps in the 49r1 radiation_ifs_rrtm.F90 gas_optics routine.

Inserts DUMP_R/DUMP_I calls (module ECRAD49_DUMP, active only when
ECRAD49_STAGE_DUMP is set) after each RRTMG call, writing that call's outputs
under the prefixes lw_prep_, lw_coef_, lw_od_, sw_coef_, sw_od_ (the SW
setcoef reuses the LW arrays, so the prefixes keep both). Only WRITE
statements are added; no oracle statement changes.

usage: python -I patch_gas_dump.py <radiation_ifs_rrtm.F90>   (edits in place)
"""
import re
import sys

STAGES = {
    "RRTM_PREPARE_GASES": ("lw_prep_", "r:ZCOLDRY r:ZWBRODL r:ZWKL r:ZWX r:ZPAVEL r:ZTAVEL r:ZPZ r:ZTZ i:IREFLECT"),
    "RRTM_SETCOEF_140GP": ("lw_coef_", "r:ZFAC00 r:ZFAC01 r:ZFAC10 r:ZFAC11 r:ZFORFAC r:ZFORFRAC i:INDFOR i:JP i:JT i:JT1 "
                           "r:ZCOLH2O r:ZCOLCO2 r:ZCOLO3 r:ZCOLN2O r:ZCOLCH4 r:ZCOLO2 r:ZCO2MULT r:ZCOLBRD "
                           "i:ILAYTROP i:ILAYSWTCH i:ILAYLOW r:ZSELFFAC r:ZSELFFRAC i:INDSELF i:INDMINOR "
                           "r:ZSCALEMINOR r:ZSCALEMINORN2 r:ZMINORFRAC r:ZRAT_H2OCO2 r:ZRAT_H2OCO2_1 r:ZRAT_H2OO3 "
                           "r:ZRAT_H2OO3_1 r:ZRAT_H2ON2O r:ZRAT_H2ON2O_1 r:ZRAT_H2OCH4 r:ZRAT_H2OCH4_1 "
                           "r:ZRAT_N2OCO2 r:ZRAT_N2OCO2_1 r:ZRAT_O3CO2 r:ZRAT_O3CO2_1"),
    "RRTM_GAS_OPTICAL_DEPTH": ("lw_od_", "r:ZOD_LW r:ZPFRAC"),
    "SRTM_SETCOEF": ("sw_coef_", "i:ILAYTROP r:ZCOLCH4 r:ZCOLCO2 r:ZCOLH2O r:ZCOLMOL r:ZCOLO2 r:ZCOLO3 r:ZFORFAC "
                     "r:ZFORFRAC i:INDFOR r:ZSELFFAC r:ZSELFFRAC i:INDSELF r:ZFAC00 r:ZFAC01 r:ZFAC10 r:ZFAC11 "
                     "i:JP i:JT i:JT1"),
    "SRTM_GAS_OPTICAL_DEPTH": ("sw_od_", "r:ZOD_SW r:ZSSA_SW r:ZINCSOL"),
}


def dump_lines(prefix, spec):
    out = []
    for item in spec.split():
        k, n = item.split(":")
        out.append(f"    CALL DUMP_{k.upper()}('{prefix}{n.lower()}', {n})")
    return out


def main(path):
    lines = open(path).read().split("\n")
    start = next(i for i, l in enumerate(lines) if re.match(r"\s*subroutine gas_optics\(", l))
    end = next(i for i, l in enumerate(lines) if re.match(r"\s*end subroutine gas_optics\b", l))
    body = lines[start:end]
    use_i = [i for i, l in enumerate(body) if re.match(r"\s*use radiation_gas\s*$", l)]
    if len(use_i) != 1:
        raise SystemExit("use radiation_gas anchor not unique in gas_optics")
    out, seen = [], set()
    i = 0
    while i < len(body):
        out.append(body[i])
        if i == use_i[0]:
            out.append("    use ecrad49_dump, only : dump_r, dump_i")
        m = re.match(r"\s*CALL\s+(\w+)", body[i], re.I)
        if m and m.group(1).upper() in STAGES:
            name = m.group(1).upper()
            while body[i].rstrip().endswith("&"):          # to the end of the statement
                i += 1
                out.append(body[i])
            out += dump_lines(*STAGES[name])
            seen.add(name)
        i += 1
    if seen != set(STAGES):
        raise SystemExit(f"calls not found: {set(STAGES) - seen}")
    lines[start:end] = out
    open(path, "w").write("\n".join(lines))


if __name__ == "__main__":
    main(sys.argv[1])
