"""Generate tests/atmosphere/oracle_fixtures/gw_spec_e3sm_beres_table.npz: the Beres
arms of ``gw_spec_e3sm/make_fixture.py`` re-run on the REAL E3SM/CAM source-spectrum table.

Table: ``atm/waccm/gw/newmfspectra40_dc25.nc`` (E3SM v3.0.1 / CAM default
``gw_drag_file``), CESM inputdata
https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/waccm/gw/newmfspectra40_dc25.nc
(svn last-changed revision 15), sha256 below (the E3SM inputdata copy is identical).
The oracle (``gw_spec_oracle_nc``, the MFCC_FILE build of ``driver.F90``) reads it
with E3SM's own ``gw_init_beres`` (gw_drag.F90:513-571, verbatim) and dumps the table
it loaded; the fixture stores that table, so the test needs no netcdf file.

Arms and columns = ``gw_spec_e3sm/make_fixture.ARMS`` convective_frontal and oro_frontal_convective
on both column sets (unchanged; only the table differs from the seeded harness table).

Self-checks, every run: the file sha256 equals the pinned value; the table the
Fortran read equals the file's ``mfcc`` variable read in Python (dims (PS, MW, HD) ->
Fortran (HD, MW, PS), phase-speed slice start ngwv_file-pgwv+1) bitwise; every column
set launches nonzero Beres momentum tendencies.

    bash build.sh /path/to/E3SM_v3.0.1 /tmp/gwspec_build      # needs nf-config on PATH
    python make_fixture.py --drv /tmp/gwspec_build/gw_spec_oracle_nc \
        --mfcc-file <inputdata>/atm/waccm/gw/newmfspectra40_dc25.nc \
        --out tests/atmosphere/oracle_fixtures/gw_spec_e3sm_beres_table.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "gw_spec_e3sm"))
import make_fixture as M  # noqa: E402   (gw_spec_e3sm generator: columns, arms, oracle I/O)

TABLE_SHA256 = "8f1528d85b3642b5024ba186f45720ecaf7998adcf9fbcb9cd21399c67d3cc16"
TABLE_URL = ("https://svn-ccsm-inputdata.cgd.ucar.edu/trunk/inputdata/atm/waccm/gw/"
             "newmfspectra40_dc25.nc")
NGWV_FILE = 40    # gw_drag.F90:531
ARM_NAMES = ("convective_frontal", "oro_frontal_convective")


def file_table(path, pgwv):
    with xr.open_dataset(path) as ds:
        a = np.asarray(ds["mfcc"].transpose("HD", "MW", "PS").values, np.float64)
    lo = NGWV_FILE - pgwv
    return a[:, :, lo:lo + 2 * pgwv + 1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--drv", required=True)
    ap.add_argument("--mfcc-file", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    sha = hashlib.sha256(Path(a.mfcc_file).read_bytes()).hexdigest()
    assert sha == TABLE_SHA256, (sha, TABLE_SHA256)
    data = {"l36": M.build_l36(), "l32": M.build_l32()}
    out = {f"{ds}/{k}": v for ds, d in data.items() for k, v in d.items()}
    meta_arms, table = {}, None
    for name in ARM_NAMES:
        arm, sets = M.ARMS[name]
        assert arm["mfcc"], name
        for ds in sets:
            o = M.run_oracle(a.drv, data[ds], arm, str(a.mfcc_file))
            t = o["mfcc"].reshape((M.MAXH, 2 * M.MAXUH + 1, 2 * arm["pgwv"] + 1), order="F")
            assert table is None or t.tobytes() == table.tobytes()
            table = t
            assert np.abs(o["ptend_u"]).max() > 0.0, (name, ds)
            for k in ("ptend_u", "ptend_v", "ptend_s", "kidx"):
                out[f"{name}/{ds}/{k}"] = o[k]
            if np.any(data[ds]["phis"] != 0.0):
                o0 = M.run_oracle(a.drv, dict(data[ds], phis=0.0 * data[ds]["phis"]), arm,
                                  str(a.mfcc_file))
                out[f"{name}/{ds}/ptend_s_nophis"] = o0["ptend_s"]
        meta_arms[name] = dict(datasets=sets, dt=arm["dt"], ours=arm["ours"], mfcc=True,
                               oracle={k: arm[k] for k in arm if k != "ours"})
    pgwv = M.ARMS[ARM_NAMES[0]][0]["pgwv"]
    ref = file_table(a.mfcc_file, pgwv)
    assert ref.tobytes() == table.tobytes(), "Fortran gw_init_beres read != file mfcc slice"
    out["mfcc"] = table
    meta = dict(oracle=f"E3SM-Project/E3SM {M.E3SM_TAG} {M.E3SM_SHA} (gw_init_beres on the file)",
                kwv=M.E3SM_KWV,
                constants=dict(g=M.constants.g, R_d=M.constants.R_d, c_pd=M.constants.c_pd),
                mfcc=dict(file="atm/waccm/gw/newmfspectra40_dc25.nc", url=TABLE_URL,
                          svn_revision=15, sha256=TABLE_SHA256, pgwv=pgwv,
                          table_sha256=hashlib.sha256(table.tobytes()).hexdigest()),
                legoesm_sha=subprocess.run(["git", "-C", str(M.REPO), "rev-parse", "HEAD"],
                                           capture_output=True, text=True).stdout.strip(),
                arms=meta_arms)
    out["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, Path(a.out).stat().st_size, "bytes; mfcc max", float(table.max()))


if __name__ == "__main__":
    main()
