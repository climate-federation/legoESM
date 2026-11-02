"""Generate tests/core/oracle_fixtures/surfacefluxes_psi.npz by RUNNING CliMA
SurfaceFluxes.jl's universal functions (psi_m, psi_h) on a zeta grid.

    bash build.sh <julia> <SurfaceFluxes.jl checkout> <envs>
    python make_fixture.py --julia <julia> --env151 <envs>/jl151 --env0120 <envs>/jl0120 \\
        --out tests/core/oracle_fixtures/surfacefluxes_psi.npz

Parameter sets equal legoESM's coefficients (sf_psi_v151.jl, sf_psi_v0120.jl,
the scripts the tier-2 audit ran): Businger(Pr0=1, a=5, b=16) = "dyer1974";
Gryanik(Pr0=0.98, a_m=a_h=5, b_m=0.3, b_h=0.4, b_unstable=16);
Grachev(Pr0=1, a=5, b_m=5/6.5, b_h=5, c_h=3, b_unstable=16) (v1.5.1); and
Beljaars(Pr0=1, a=1, b=0.667, c=5, d=0.35) (v0.12.0).  Zeta grid: 401 points
log-spaced over +-[1e-6, 100] plus 0 (oracle_audit .../inputs/zeta_grid_flat.txt
construction).
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import tempfile
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
EXPECT = {"151": "SF_V151_OK 1.5.1 ClimaParams 1.3.3",
          "0120": "SF_V0120_OK 0.12.0 ClimaParams 0.10.35"}
SHA = {"151": "fd42723427b5264b3e5f8ee9f622cc7b2e94fdae",
       "0120": "167c5144c0171cc425432ad26d6569552a73ee11"}
SETS = {"dyer1974": ("151", "businger_lego"), "gryanik2020": ("151", "gryanik_lego"),
        "grachev2007_sheba": ("151", "grachev_lego"),
        "beljaars_holtslag1991": ("0120", "beljaars_lego")}


def zeta_grid():
    pos = np.geomspace(1e-6, 100.0, 200)
    return np.concatenate([-pos[::-1], [0.0], pos])


def check_loaded_source(path, ver):
    """The SurfaceFluxes source Julia loaded (pkgdir) is the pinned SHA, unmodified."""
    sha = subprocess.run(["git", "-C", path, "rev-parse", "HEAD"], check=True,
                         capture_output=True, text=True).stdout.strip()
    if sha != SHA[ver]:
        raise SystemExit(f"{path} is at {sha}, expected {SHA[ver]}")
    if subprocess.run(["git", "-C", path, "diff", "--quiet", "HEAD", "--", "src"]).returncode:
        raise SystemExit(f"{path} src/ has local changes")


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--julia", required=True)
    ap.add_argument("--env151", required=True)
    ap.add_argument("--env0120", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    z = zeta_grid()
    save = {"zeta": z}
    with tempfile.TemporaryDirectory() as td:
        zf = Path(td) / "zeta.txt"
        np.savetxt(zf, z, fmt="%.17e")
        for ver, env in (("151", a.env151), ("0120", a.env0120)):
            out = Path(td) / f"sf{ver}"
            r = subprocess.run([a.julia, f"--project={env}", str(HERE / f"sf_psi_v{ver}.jl"),
                                str(out), str(zf)], check=True, env=dict(os.environ),
                               capture_output=True, text=True)
            lines = r.stdout.strip().splitlines()
            ok = lines[-1]
            src = [ln.split(" ", 1)[1] for ln in lines if ln.startswith("SF_SRC ")]
            if len(src) != 1:
                raise SystemExit(f"oracle run did not report its SurfaceFluxes source: {lines}")
            check_loaded_source(src[0], ver)
            want = EXPECT[ver]
            if not ok.startswith(want):
                raise SystemExit(f"oracle environment reports {ok!r}, expected {want!r}")
            save[f"run_v{ver}"] = np.array(ok)
            jv = subprocess.run([a.julia, "--version"], check=True, capture_output=True,
                                text=True).stdout.strip()
            if jv != "julia version 1.10.10":
                raise SystemExit(f"{jv} != julia version 1.10.10")
            for scheme, (v, name) in SETS.items():
                if v != ver:
                    continue
                arr = np.loadtxt(f"{out}_{name}.txt")
                if not np.array_equal(arr[:, 0], z):
                    raise SystemExit(f"zeta grid not round-tripped for {name}")
                save[f"{scheme}_psi_m"] = arr[:, 1]
                save[f"{scheme}_psi_h"] = arr[:, 2]
            save[f"params_v{ver}"] = np.array(Path(f"{out}_params.txt").read_text())
    meta = dict(oracle="CliMA SurfaceFluxes.jl",
                v151=f"v1.5.1 {SHA['151']}", v0120=f"v0.12.0 {SHA['0120']}",
                julia="1.10.10", sets={k: v[1] for k, v in SETS.items()})
    save["meta"] = np.array(json.dumps(meta))
    np.savez_compressed(a.out, **save)
    print("wrote", a.out)


if __name__ == "__main__":
    main()
