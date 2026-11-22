"""Build the per-deck Cariolle ozone-chemistry coefficient file (votes C1-C3, O1).

Source: the public OpenIFS 48r1 climate set, T63 (48r1_climate.v020_63.tar.gz,
md5 4a6ce8fd39928776dd2836dacb09ef7c; DECLARED DEPARTURE: 48r1 data with the
49r1 code, no public 49r1 set), files 63l_2/o3chem_l137/O3CHEM{01..12}_H1.0,
read as 49r1 climate/updo3ch.F90 reads them: big-endian sequential
unformatted; month, n_coef, then per coefficient (ich, nlev, nlat), a name
record, and per level nlat REAL*8 zonal values on the T63 Gaussian latitudes
(north to south; the 63l_2 grid is TL63, 64 latitudes).  The coefficient levels are the IFS L137 levels; their
standard full-level pressures are the 'pf' column of ECMWF's published L137
table (https://confluence.ecmwf.int/display/UDOC/L137+model+level+definitions,
the same page scripts/validate/oracle/ecrad_ifs/ifs49_l137_ab.py parses).

Each coefficient is interpolated linearly in ln(standard pressure) from the
L137 levels to the deck's standard full-level pressures (its vertical
coordinate at p_s = 101325 Pa), constant beyond the ends (vote C2a).
Latitude stays on the T63 grid; the run interpolates to cell latitudes (O2a).

usage: python -P build_ifs_cariolle_coefficients.py <o3chem_l137 dir>
           <l137.html> <l137.html sha256> <vertical_coord> <out.npz>
(writes <out>.json beside it: npz and source sha256s)
"""
import hashlib
import json
import re
import struct
import sys
from pathlib import Path

import numpy as np

from legoesm import constants

# 49r1 sudimf1.F90:224-230: the hybrid-linear (HLO) scheme reads 7 fields
# (k1..k7); the 8th (Cl_y, TEECL2 term) exists only for the non-HLO paths.
N_COEF = 7


def _records(raw):
    out, i = [], 0
    while i < len(raw):
        (n,) = struct.unpack(">i", raw[i:i + 4])
        body = raw[i + 4:i + 4 + n]
        (n2,) = struct.unpack(">i", raw[i + 4 + n:i + 8 + n])
        if n2 != n:
            raise ValueError("record marker mismatch")
        out.append(body)
        i += 8 + n
    return out


def read_o3chem(path):
    """-> (month, coef (7, nlev, nlat)) exactly as updo3ch.F90:301-348 reads."""
    recs = _records(Path(path).read_bytes())
    (month,) = struct.unpack(">i", recs[0])
    (ncoef,) = struct.unpack(">i", recs[1])
    if ncoef != N_COEF:
        raise ValueError(f"{path}: {ncoef} coefficients, expected {N_COEF}")
    k, coef = 2, None
    for jch in range(1, ncoef + 1):
        ich, nlev, nlat = struct.unpack(">3i", recs[k])
        if ich != jch:
            raise ValueError(f"{path}: coefficient {ich} out of order")
        if coef is None:
            coef = np.empty((ncoef, nlev, nlat))
        k += 2                                   # header + name record
        for jl in range(nlev):
            coef[jch - 1, jl] = np.frombuffer(recs[k], dtype=">f8", count=nlat)
            k += 1
    if k != len(recs):
        raise ValueError(f"{path}: {len(recs) - k} trailing records")
    return month, coef


def l137_pf(html, sha):
    raw = Path(html).read_bytes()
    got = hashlib.sha256(raw).hexdigest()
    if got != sha:
        sys.exit(f"sha256 mismatch: {got} != {sha}")
    rows = re.findall(r"<tr>(.*?)</tr>", raw.decode("utf-8"), flags=re.S)
    pf = {}
    for r in rows:
        cells = [c.strip() for c in re.findall(r'<td class="confluenceTd">([^<]*)</td>', r)]
        if len(cells) >= 5 and re.fullmatch(r"\d+", cells[0]) and 1 <= int(cells[0]) <= 137:
            pf[int(cells[0])] = float(cells[4]) * 100.0    # pf printed in hPa
    if sorted(pf) != list(range(1, 138)):
        sys.exit("L137 table: full-level rows 1..137 not all found")
    return np.array([pf[n] for n in range(1, 138)])


def gaussian_latitudes_deg(nlat):
    """Gaussian latitudes of the file's grid (TL63: 64 = N32), north to south."""
    x, _ = np.polynomial.legendre.leggauss(nlat)
    return np.rad2deg(np.arcsin(x))[::-1]


def deck_standard_pressures(vertical_coord):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    if vertical_coord != "cam_l32":
        sys.exit(f"vertical_coord {vertical_coord!r} not supported by this builder")
    from legoesm.grids.vertical import make_cam6_l32_levels
    sig = make_cam6_l32_levels()
    return np.asarray(sig.pressure_at_full(jnp.asarray([constants.p_atm_std]))).reshape(-1)


def main(src_dir, html, sha, vertical_coord, out):
    src = Path(src_dir)
    files = [src / f"O3CHEM{m:02d}_H1.0" for m in range(1, 13)]
    coef = []
    for m, f in enumerate(files, start=1):
        month, c = read_o3chem(f)
        if month != m:
            sys.exit(f"{f}: month record {month} != {m}")
        coef.append(c)
    coef = np.stack(coef)                                   # (12, 7, 137, 64)
    pf = l137_pf(html, sha)
    if coef.shape[2] != pf.size:
        sys.exit(f"unexpected coefficient shape {coef.shape}")
    p_deck = deck_standard_pressures(vertical_coord)
    lnp, lnq = np.log(pf), np.log(p_deck)
    out_c = np.empty(coef.shape[:2] + (p_deck.size, coef.shape[3]))
    for i in range(12):
        for j in range(N_COEF):
            for la in range(coef.shape[3]):
                out_c[i, j, :, la] = np.interp(lnq, lnp, coef[i, j, :, la])
    np.savez(out, coef=out_c, lat_deg=gaussian_latitudes_deg(coef.shape[3]), p_std_full=p_deck,
             vertical_coord=np.array(vertical_coord),
             source_sha256=np.array([hashlib.sha256(f.read_bytes()).hexdigest() for f in files]),
             l137_html_sha256=np.array(sha))
    out = Path(out)
    meta = {
        "data_set": "openifs-48r1-climate.v020-T63 (DEPARTURE: 48r1 data, 49r1 code)",
        "generator": "scripts/data/build_ifs_cariolle_coefficients.py",
        "npz_sha256": hashlib.sha256(out.read_bytes()).hexdigest(),
        "source_files": {f.name: hashlib.sha256(f.read_bytes()).hexdigest() for f in files},
        "l137_html_sha256": sha,
        "vertical_coord": vertical_coord,
        "shapes": {"coef": list(out_c.shape)},
    }
    out.with_suffix(".json").write_text(json.dumps(meta, indent=1, sort_keys=True) + "\n")
    print("CARIOLLE_OK", out, out_c.shape)


if __name__ == "__main__":
    main(*sys.argv[1:6])
