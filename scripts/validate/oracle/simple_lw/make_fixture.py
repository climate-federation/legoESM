"""Generate the simple-LW oracle fixture by RUNNING the compiled gSAM rad_simple.

Oracle: gSAM 1.8.8 release tarball
(http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz), ``SRC/rad_simple.f90``
compiled unmodified by ``build.sh`` (gfortran, fp64 via -fdefault-real-8).
The stubs in ``stubs_gsam.F90`` only declare the module variables the routine
reads; every coefficient stays hardcoded in the oracle.

Columns (bottom-up, exactly the tier-2 audit set):

* DYCOMS RF01 on a uniform 5 m grid and ASTEX209 on a uniform 25 m grid,
  from the repo's ``data/les_cases/<case>/snd`` soundings, interpolated to
  cell centres, anelastic reference from
  ``spectral_les_moist.make_anelastic_reference`` and saturation-adjusted
  with ``les_suite.scm_coupling.saturation_adjust``;
* RF01 on a stretched grid built with gSAM's own ``setgrid.f90`` algorithm;
* edge columns: an ice/liquid split, a clear column, a dry column with no
  inversion, a cell with ``q_t`` exactly at the 8 g/kg isoline, and a cell
  with negative condensate (and both together).

The face density ``rhow`` is gSAM ``setdata.f90`` lines 491-496
(non-lat-lon branch): hydrostatic, from the reference pressure at the cell
centres, extrapolated linearly to the bottom and top faces.

usage: python make_fixture.py --tgz <gsam1.8.8.tar.gz> --build <dir> \
           --out tests/atmosphere/oracle_fixtures/simple_lw_gsam.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[3]
COL_FIELDS = ("zi", "adz", "zc", "rho", "rhow", "qv", "qcl", "qci")


def _jnp():
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    return jnp


def sam_grid_from_centres(zc):
    """gSAM setgrid.f90 old-style grd branch (lines 41-80): centres -> zi, adz, dz."""
    nzm = zc.size
    nz = nzm + 1
    z = zc.copy()
    adzw = np.zeros(nz)
    adz = np.zeros(nzm)
    zi = np.zeros(nz)
    dz = 0.5 * (z[0] + z[1])
    adzw[1:nzm] = (z[1:] - z[:-1]) / dz
    adzw[0] = 1.0
    adzw[nzm] = adzw[nzm - 1]
    adz[0] = 1.0
    adz[1:nzm - 1] = 0.5 * (z[2:] - z[:-2]) / dz
    adz[nzm - 1] = adzw[nzm - 1]
    for k in range(1, nz):
        zi[k] = zi[k - 1] + adz[k - 1] * dz
    dz = zi[1] - zi[0]
    adz = (zi[1:] - zi[:-1]) / dz
    z = 0.5 * (zi[1:] + zi[:-1])
    return zi, adz, dz, z


def sounding_column(case, zi, adz, dz, zc):
    jnp = _jnp()
    from legoesm import constants
    from legoesm.atmosphere.dynamics.les.spectral_les_moist import make_anelastic_reference
    from legoesm.atmosphere.forcing.sam_case_forcing import read_sam_snd
    from legoesm.atmosphere.les_suite.scm_coupling import saturation_adjust
    snd = read_sam_snd(ROOT / "data" / "les_cases" / case / "snd")
    thl = np.interp(zc, snd.z, snd.theta)
    qt = np.interp(zc, snd.z, snd.q_v)
    ref = make_anelastic_reference(zc, zi, float(snd.pres0) * 100.0, thl, qt)
    _th, qv, qc = saturation_adjust(jnp.asarray(thl), jnp.asarray(qt), ref.exner_c, ref.p_c)
    rho = np.asarray(ref.rho_c)
    pres_hpa = np.asarray(ref.p_c) / 100.0
    nzm = zc.size
    rhow = np.zeros(nzm + 1)
    rhow[1:nzm] = (pres_hpa[:-1] - pres_hpa[1:]) / (zc[1:] - zc[:-1]) / constants.g * 100.0
    rhow[0] = 2 * rho[0] - rhow[1]
    rhow[nzm] = 2 * rho[nzm - 1] - rhow[nzm - 1]
    return dict(zi=zi, adz=adz, dz=dz, zc=zc, rho=rho, rhow=rhow,
                qv=np.asarray(qv), qcl=np.asarray(qc), qci=np.zeros(nzm))


def build_columns():
    cols = {}
    zi = np.arange(0.0, 1500.0 + 1e-9, 5.0)
    zc = 0.5 * (zi[1:] + zi[:-1])
    rf01 = sounding_column("DYCOMS_RF01", zi, np.ones(300), 5.0, zc)
    cols["rf01_5m"] = rf01
    zi = np.arange(0.0, 2500.0 + 1e-9, 25.0)
    zc = 0.5 * (zi[1:] + zi[:-1])
    cols["astex209_25m"] = sounding_column("ASTEX209", zi, np.ones(100), 25.0, zc)
    zc_s = np.cumsum(4.0 * 1.012 ** np.arange(160)) - 2.0
    zi_s, adz_s, dz_s, zc_s2 = sam_grid_from_centres(zc_s)
    cols["rf01_stretched"] = sounding_column("DYCOMS_RF01", zi_s, adz_s, dz_s, zc_s2)
    c = dict(rf01)
    c["qci"] = np.where(np.arange(300) > 160, 0.3 * rf01["qcl"], 0.0)
    c["qcl"] = rf01["qcl"] - c["qci"]
    cols["rf01_ice_split"] = c
    cols["rf01_clear"] = dict(rf01, qcl=np.zeros(300))
    cols["dry_no_inversion_cloudy"] = dict(rf01, qv=rf01["qv"] * 0.5)
    qcl = rf01["qcl"].copy()
    qcl[20] = -1e-6
    qv = rf01["qv"].copy()
    qv[200] = 0.008
    qcl_eq = qcl.copy()
    qcl_eq[200] = 0.0
    cols["edge_eq_threshold_neg_cond"] = dict(rf01, qv=qv, qcl=qcl_eq)
    cols["edge_eq_threshold_only"] = dict(rf01, qv=qv)
    cols["edge_neg_cond_only"] = dict(rf01, qcl=qcl)
    return cols


def _f17(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


def write_input(path, cols):
    lines = [str(len(cols))]
    for c in cols.values():
        lines += [f"{c['adz'].size} {float(c['dz'])!r}", _f17(c["adz"]), _f17(c["zi"]),
                  _f17(c["rho"]), _f17(c["rhow"]), _f17(c["qv"]), _f17(c["qcl"]),
                  _f17(c["qci"])]
    Path(path).write_text("\n".join(lines) + "\n")


def read_output(path, ncol):
    rows = [ln for ln in Path(path).read_text().splitlines() if ln.strip()]
    out = []
    i = 0
    for _ in range(ncol):
        assert rows[i].startswith("COL"), rows[i]
        flux = np.array(rows[i + 1].split(), dtype=np.float64)
        hr = np.array(rows[i + 2].split(), dtype=np.float64)
        out.append((flux, hr))
        i += 3
    return out


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--tgz", required=True, type=Path)
    ap.add_argument("--build", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    a = ap.parse_args(argv)
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.tgz), str(a.build)], check=True)
    cols = build_columns()
    fin = a.build / "simple_in.txt"
    write_input(fin, cols)
    subprocess.run([str(a.build / "drv_simple")], cwd=a.build, check=True)
    res = read_output(a.build / "simple_out.txt", len(cols))
    fx = {"col_names": np.array(list(cols))}
    for (name, c), (flux, hr) in zip(cols.items(), res):
        for k in COL_FIELDS:
            fx[f"{name}__{k}"] = np.asarray(c[k], dtype=np.float64)
        fx[f"{name}__dz"] = np.float64(c["dz"])
        fx[f"{name}__ora_flux"] = flux     # net upward LW at faces 1..nzm [W/m2]
        fx[f"{name}__ora_hr"] = hr         # FTHRL [K/s] at cells 1..nzm
    meta = dict(
        oracle="gSAM", tag="1.8.8", file="SRC/rad_simple.f90",
        tarball="http://rossby.msrc.sunysb.edu/GSAM/gsam1.8.8.tar.gz",
        tarball_sha256=hashlib.sha256(a.tgz.read_bytes()).hexdigest(),
        file_sha256=hashlib.sha256((a.build / "rad_simple.f90").read_bytes()).hexdigest(),
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        legoesm_sha=subprocess.run(["git", "-C", str(HERE), "rev-parse", "HEAD"],
                                   capture_output=True, text=True).stdout.strip(),
        generator=("python scripts/validate/oracle/simple_lw/make_fixture.py "
                   f"--tgz <gsam1.8.8.tar.gz> --build <dir> --out {a.out.name}"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["tarball_sha256"])


if __name__ == "__main__":
    main()
