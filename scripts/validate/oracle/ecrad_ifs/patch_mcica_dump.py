"""Harness stage dumps in the 49r1 McICA solvers (radiation_mcica_lw/sw.F90).

After each CALL CLOUD_GENERATOR, writes the generator outputs od_scaling
(ng,nlev) and total_cloud_cover for that column as <lw|sw>_gen_<var>_c<jcol>
(module ECRAD49_DUMP, active only when ECRAD49_STAGE_DUMP is set). Only WRITE
statements are added.

usage: python -I patch_mcica_dump.py <radiation_mcica_lw.F90> <radiation_mcica_sw.F90>
"""
import re
import sys


def patch(path, tag):
    lines = open(path).read().split("\n")
    out, n_use, n_call = [], 0, 0
    i = 0
    while i < len(lines):
        out.append(lines[i])
        if re.match(r"\s*use radiation_cloud_generator, only: cloud_generator\s*$", lines[i]):
            out.append("    use ecrad49_dump, only : dump_r_col")
            n_use += 1
        if re.match(r"\s*call cloud_generator\(", lines[i]):
            while lines[i].rstrip().endswith("&"):
                i += 1
                out.append(lines[i])
            ind = re.match(r"(\s*)", lines[i]).group(1).replace("&", " ")
            out += [f"{ind}call dump_r_col('{tag}_gen_od_scaling', jcol, od_scaling)",
                    f"{ind}call dump_r_col('{tag}_gen_total_cloud_cover', jcol, total_cloud_cover)"]
            n_call += 1
        i += 1
    if (n_use, n_call) != (1, 1):
        raise SystemExit(f"{path}: anchors found use={n_use} call={n_call}, expected 1 and 1")
    open(path, "w").write("\n".join(out))


if __name__ == "__main__":
    patch(sys.argv[1], "lw")
    patch(sys.argv[2], "sw")
