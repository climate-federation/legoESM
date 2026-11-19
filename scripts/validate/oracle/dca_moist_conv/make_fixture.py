"""Generate the Manabe moist-convective-adjustment oracle fixture by RUNNING
GFDL atmos_phys ``moist_conv``.

Oracle: NOAA-GFDL/atmos_phys tag ``2021.02`` (commit
46817a12232ed54d1a5d5a4394a695621f575812),
``atmos_param/moist_conv/moist_conv.F90`` compiled unmodified by ``build.sh``
(FMS stubs with legoESM constants and saturation curve).  coldT = .false., no
cloud tracers, namelist defaults (HC 1, TOLmin 0.02, TOLmax 0.10, ITSMOD 30,
do_simple F); dtinv = 1/1800 s.

Columns: the 19 columns of the 2026-10-19 tier-2 audit (same file as the SBM
fixture, ``../sbm_qe/make_fixture.py``) -- on which moist_conv acts on two,
with one block and one pass each -- plus the constructed columns of
:func:`extra_columns` (same recipe as the audit's ``satunst_*`` columns, on the
``trop_ocean_deep`` grid), which make moist_conv re-test, iterate, find several
blocks and reach the bottom level.

Usage (CPU node; gfortran 8.5.0)::

    bash scripts/validate/oracle/dca_moist_conv/build.sh <atmos_phys 2021.02> <build dir>
    python scripts/validate/oracle/dca_moist_conv/make_fixture.py --build <build dir> \\
        --columns <columns.npz> --atmos-phys <atmos_phys 2021.02> \\
        --out tests/atmosphere/oracle_fixtures/dca_moist_conv.npz
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "sbm_qe"))
from make_fixture import DT, _sha256, write_input  # noqa: E402  (shared column writer)

ORACLE = dict(repo="NOAA-GFDL/atmos_phys", tag="2021.02",
              sha="46817a12232ed54d1a5d5a4394a695621f575812",
              file="atmos_param/moist_conv/moist_conv.F90")


EXTRA = (  # name, surface T [K], lapse [K/m], supersaturated bands [(p_top, p_bot) Pa]
    ("satunst_sfc", 300.0, 9.5e-3, ((600e2, 1.1e5),)),
    ("satunst_two", 300.0, 9.5e-3, ((380e2, 620e2), (760e2, 1.1e5))),
    ("satunst_steep", 305.0, 9.7e-3, ((250e2, 1.1e5),)),
    ("satunst_super", 305.0, 12.0e-3, ((250e2, 1.1e5),)),
)


def extra_columns(c):
    """Supersaturated (1.002 q_sat), super-moist-adiabatic bands under 0.6 q_sat
    elsewhere and 3e-6 above 100 hPa, as the audit built satunst_trop/_mid."""
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.thermo import saturation_specific_humidity
    names = [str(n) for n in c["names"]]
    i0 = names.index("trop_ocean_deep")
    pf0, ph0, T0 = c["p_full"][i0], c["p_half"][i0], c["T"][i0]
    out = {k: [c[k]] for k in ("T", "q", "p_full", "p_half")}
    for nm, Ts, gam, bands in EXTRA:
        Ts = float(T0[-1]) if Ts is None else Ts
        Tn = np.maximum(Ts * (pf0 / ph0[-1]) ** (constants.R_d * gam / constants.g), 200.0)
        qs = np.asarray(saturation_specific_humidity(jnp.asarray(Tn), jnp.asarray(pf0)))
        band = np.zeros(pf0.shape, bool)
        for lo, hi in bands:
            band |= (pf0 >= lo) & (pf0 <= hi)
        qn = np.where(pf0 < 100e2, 3e-6, np.where(band, 1.002 * qs, 0.6 * qs))
        for k, v in (("T", Tn), ("q", qn), ("p_full", pf0), ("p_half", ph0)):
            out[k].append(v[None])
        names.append(nm)
    out = {k: np.concatenate(v) for k, v in out.items()}
    out["names"] = np.array(names)
    return out


def read_output(path):
    lines = Path(path).read_text().split("\n")
    head = lines[0].split()
    cols = []
    i = 1
    while i < len(lines) and lines[i].startswith("COL"):
        sc = [float(x) for x in lines[i + 1].split()]
        cols.append(dict(rain=sc[0], snow=sc[1],
                         dT=np.array([float(x) for x in lines[i + 2].split()]),
                         dq=np.array([float(x) for x in lines[i + 3].split()]),
                         conv=np.array([int(x) for x in lines[i + 4].split()])))
        i += 5
    return cols, dict(notes=int(head[3]), warnings=int(head[4]))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--build", required=True)
    ap.add_argument("--columns", required=True)
    ap.add_argument("--atmos-phys", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    c = extra_columns(dict(np.load(a.columns)))
    with tempfile.TemporaryDirectory() as d:
        write_input(Path(d) / "in.txt", c)
        r = subprocess.run([str(Path(a.build) / "drv_mca.x"), "in.txt", "out.txt"],
                           cwd=d, capture_output=True, text=True, check=True)
        assert "DRV_MCA_OK" in r.stdout, r.stdout + r.stderr
        cols, counts = read_output(Path(d) / "out.txt")
    head = subprocess.run(["git", "-C", a.atmos_phys, "rev-parse", "HEAD"],
                          capture_output=True, text=True, check=True).stdout.strip()
    assert head == ORACLE["sha"], head
    gf = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    meta = dict(ORACLE, oracle_file_sha256=_sha256(Path(a.atmos_phys) / ORACLE["file"]), gfortran=gf,
                flags="-O0 -cpp -fdefault-real-8 -fdefault-double-8 -fno-fast-math",
                constants="legoESM (sbm_qe/stubs_common.F90 default)",
                saturation="legoESM Tetens + legoESM q_sat and exact dq_sat/dT (sat_mode 1)",
                coldT=False, cloud_tracers=False, dt=DT, altol_doublings=counts["notes"],
                max_iteration_warnings=counts["warnings"], columns_sha256=_sha256(a.columns),
                columns="oracle_audit/sbm_dca_adjustment_convection/inputs/columns.npz")
    out = {k: c[k] for k in ("T", "q", "p_full", "p_half")}
    out["names"] = np.array([str(n) for n in c["names"]])
    out["ora_rain"] = np.array([col["rain"] for col in cols])
    out["ora_snow"] = np.array([col["snow"] for col in cols])
    for k in ("dT", "dq", "conv"):
        out[f"ora_{k}"] = np.stack([col[k] for col in cols])
    out["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez(a.out, **out)
    print("wrote", a.out, "levels flagged per column", out["ora_conv"].sum(axis=1).tolist())


if __name__ == "__main__":
    main()
