"""Parse ECMWF's published L137 model-level table into the hybrid A/B input of
the aerosol-arm harness (vote B1b: GPHPRE(VP00) reference eta needs the model
A/B; no L137 table is in the 49r1 sources).

Source page: https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions
The table rows are n, a [Pa], b, ph, pf, ...; n = 0..137 are the half levels,
top-down, the IFS VAH(0:137), VBH(0:137) order. Values are copied as printed
(no rounding or derivation); the sha256 of the html is checked.

usage: python -I ifs49_l137_ab.py <l137.html> <sha256> <out dir>
writes <out dir>/l137_vah.txt and l137_vbh.txt (ifs49_driver RD1 format)
"""
import hashlib
import re
import sys


def main(html, sha, out):
    raw = open(html, "rb").read()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        sys.exit(f"sha256 mismatch: {got} != {sha}")
    cells = re.findall(r'<td class="confluenceTd">([^<]*)</td>', raw.decode("utf-8"))
    rows = {}
    for i in range(0, len(cells) - 2):
        if re.fullmatch(r"\d+", cells[i].strip()) and re.fullmatch(r"-?\d+\.\d+", cells[i + 1].strip()) \
                and re.fullmatch(r"-?\d+\.\d+", cells[i + 2].strip()):
            n = int(cells[i])
            if n not in rows and 0 <= n <= 137:
                rows[n] = (cells[i + 1].strip(), cells[i + 2].strip())
    if sorted(rows) != list(range(138)):
        sys.exit(f"expected rows 0..137, got {len(rows)}")
    if rows[0] != ("0.000000", "0.000000") or rows[137][1] != "1.000000":
        sys.exit(f"unexpected end rows {rows[0]} {rows[137]}")
    for k, name in ((0, "l137_vah"), (1, "l137_vbh")):
        with open(f"{out}/{name}.txt", "w") as f:
            f.write(f"{name} 1 138\n")
            f.write("\n".join(rows[n][k] for n in range(138)) + "\n")
    print(f"L137_OK a[1]={rows[1][0]} b[136]={rows[136][1]} source sha256 {got}")


if __name__ == "__main__":
    main(*sys.argv[1:4])
