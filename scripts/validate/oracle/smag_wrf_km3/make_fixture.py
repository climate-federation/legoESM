"""Generate tests/atmosphere/oracle_fixtures/smag_wrf_km3.npz by RUNNING WRF's
km_opt=3 Smagorinsky eddy coefficients.

Oracle: wrf-model/WRF v4.8.0 (06d4240ae989cc3e50af412bb472df3d9048783c)
``dyn_em/module_diffusion_em.F`` compiled UNMODIFIED (``build.sh``) with framework
stubs (``stubs.F90``) and called by ``driver.F90`` on a doubly periodic tile of
identical columns: ``compute_diff_metrics`` -> ``cal_deform_and_div`` ->
``calculate_km_kh`` (``km_opt = 3``, ``mix_isotropic = 0``, ``damp_opt = 0``).
Compared: ``defor13``/``defor23`` (w-levels), ``BN2``, ``xkmv``, ``xkhv`` (mass levels).

Constants: WRF's own share/module_model_constants.F with legoESM's g, R_d, c_pd, R_v,
L_v, p_ref, T_freeze and ``SmagorinskyConfig().wrf_prandtl`` substituted (VOTE: our
constants written into the oracle).  ``c_s`` and ``mix_upper_bound`` are run inputs.

Columns (VOTE D13): the 26 tier-2 turbulence audit columns
(oracle_audit/local_closures_louis_tke_clubb_lite/inputs.npz, dt from the file), qc =
qi = 0, plus CLOUDY variants of four of them (VOTE S6) that exercise the moist BN2
branch.  Water enters WRF as mixing ratios r_x = q_x / (1 - q_v - q_c - q_i) (VOTE D8).
WRF eta at w-levels: (p_half - p_top) / (p_s - p_top) (VOTE D10).

usage (compute node):
    python make_fixture.py constants <WRF> <out module_model_constants.F>
    bash build.sh <WRF> <build dir> <module_model_constants.F>
    python make_fixture.py run <build dir>/drv_smag <WRF> <inputs.npz> <out.npz>
"""
import json
import re
import subprocess
import sys
import tempfile
from pathlib import Path

import numpy as np

from legoesm import constants as C
from legoesm.atmosphere.physics.turbulence.config import SmagorinskyConfig

SHA = "06d4240ae989cc3e50af412bb472df3d9048783c"
ROWS_W = ("defor13", "defor23")
ROWS_M = ("bn2", "xkmv", "xkhv")
CLOUDY = ("trop_ocean_deep", "midlat_supercell", "cold_deep_ocean", "all_snow_land")
DT_SHORT = 60.0      # s, second oracle pass
N_CLOUD = 5          # cloudy levels per variant: the N most humid
QC_CLOUD = 2.0e-4    # kg/kg liquid on them (above the WRF qc_cr = 1e-5 test)
QI_CLOUD = 2.0e-5    # kg/kg ice on those with T < T_freeze


def write_constants(wrf, out):
    cfg = SmagorinskyConfig()
    vals = {"g": C.g, "r_d": C.R_d, "cp": C.c_pd, "r_v": C.R_v, "p1000mb": C.p_ref,
            "XLV": C.L_v, "SVPT0": C.T_freeze, "prandtl": cfg.wrf_prandtl}
    s = (Path(wrf) / "share/module_model_constants.F").read_text()
    for k, v in vals.items():
        pat = re.compile(rf"(PARAMETER\s*::\s*{k}\s*=)\s*[^!\n]+", re.IGNORECASE)
        s, n = pat.subn(lambda m: f"{m.group(1)} {float(v)!r}", s)
        assert n == 1, (k, n)
    Path(out).write_text("! legoESM constants substituted by make_fixture.py\n" + s)
    print("CONSTANTS_OK", out)


def f17(a):
    return " ".join(repr(float(x)) for x in np.ravel(a))


def build_columns(src):
    """Top-down legoESM columns (+ cloudy variants) with qc, qi."""
    from legoesm.thermo import saturation_mixing_ratio
    names = [str(n) for n in src["names"]]
    cols = []
    for i, nm in enumerate(names):
        base = {k: np.asarray(src[k][i], dtype=np.float64)
                for k in ("T", "q", "u", "v", "p_full", "p_half", "zi")}
        base["qc"] = np.zeros_like(base["T"])
        base["qi"] = np.zeros_like(base["T"])
        cols.append((nm, base))
        if nm in CLOUDY:
            cl = dict(base)
            r = base["q"] / (1.0 - base["q"])
            rh = r / np.asarray(saturation_mixing_ratio(base["T"], base["p_full"]))
            order = np.argsort(rh)[::-1]
            wet = np.zeros(rh.size, bool)
            wet[order[:N_CLOUD]] = True
            cl["qc"] = np.where(wet, QC_CLOUD, 0.0)
            cl["qi"] = np.where(wet & (base["T"] < C.T_freeze), QI_CLOUD, 0.0)
            # most humid level supersaturated by 1 %: the qv >= qvs trigger
            k = order[0]
            r_s = float(saturation_mixing_ratio(base["T"][k], base["p_full"][k]))
            r_new = 1.01 * r_s
            q = base["q"].copy()
            q[k] = r_new * (1.0 - cl["qc"][k] - cl["qi"][k]) / (1.0 + r_new)
            cl["q"] = q
            cols.append((nm + "_cloudy", cl))
    return cols


def _oracle(oracle, cols, nlev, dt, cfg):
    s = [f"{len(cols)} {nlev}", f17([dt, cfg.wrf_c_s, cfg.mix_upper_bound])]
    for _, c in cols:
        qt = c["q"] + c["qc"] + c["qi"]
        ph = c["p_half"][::-1]
        znw = (ph - ph[-1]) / (ph[0] - ph[-1])
        s += [f17(c["zi"][::-1]), f17(c["u"][::-1]), f17(c["v"][::-1]), f17(c["T"][::-1]),
              f17(c["p_full"][::-1]), f17(ph)]
        s += [f17((c[k] / (1.0 - qt))[::-1]) for k in ("q", "qc", "qi")]
        s += [f17(znw)]
    with tempfile.TemporaryDirectory() as td:
        fin, fout = Path(td) / "in.txt", Path(td) / "out.txt"
        fin.write_text("\n".join(s) + "\n")
        r = subprocess.run([oracle, str(fin), str(fout)], cwd=td, capture_output=True,
                           text=True, check=True)
        assert "SMAG_DRV_OK" in r.stdout, r.stdout + r.stderr
        lines = fout.read_text().splitlines()
    res = {k: [] for k in ROWS_W + ROWS_M + ("cf",)}
    for i in range(len(cols)):
        blk = lines[i * 7:(i + 1) * 7]
        assert blk[0].split() == ["COL", str(i + 1)], blk[0]
        for j, k in enumerate(ROWS_W + ROWS_M + ("cf",)):
            res[k].append(np.array(blk[1 + j].split(), dtype=np.float64)[::-1]
                          if k != "cf" else np.array(blk[1 + j].split(), dtype=np.float64))
    res = {k: np.stack(v) for k, v in res.items()}   # back to top-down
    return res


def run(oracle, wrf, inputs, out):
    cfg = SmagorinskyConfig()
    src = np.load(inputs, allow_pickle=True)
    dt = float(src["dt"])
    cols = build_columns(src)
    nlev = cols[0][1]["T"].size
    res = _oracle(oracle, cols, nlev, dt, cfg)
    # second pass at a short step: the cap moves up, so K leaves the floor/cap and
    # the deformation average is exercised (VOTE U1)
    short = _oracle(oracle, cols, nlev, DT_SHORT, cfg)
    res["xkmv_short"], res["xkhv_short"] = short["xkmv"], short["xkhv"]

    # known-value self-checks (WRF definitions, every run)
    zi = np.stack([c["zi"] for _, c in cols])
    dz = zi[:, :-1] - zi[:, 1:]
    mub_cap = cfg.mix_upper_bound * dz ** 2 / dt
    assert np.all(res["defor13"][:, 0] == 0.0) and np.all(res["defor13"][:, -1] == 0.0)
    u = np.stack([c["u"] for _, c in cols])
    rdz = 2.0 / (zi[:, :-2] - zi[:, 2:])
    want = (u[:, :-1] - u[:, 1:]) * rdz
    assert np.max(np.abs(res["defor13"][:, 1:-1] - want)) <= 1e-12 * np.max(np.abs(want)) + 1e-300
    # 1e-12: the 1-ulp spread runs perturb z and dt inside the oracle only
    assert np.all(res["xkmv"] >= 1e-6 * dz ** 2 * (1 - 1e-12))
    assert np.all(res["xkmv"] <= mub_cap * (1 + 1e-12))
    assert np.all(res["xkhv"] <= mub_cap * (1 + 1e-12))
    assert np.allclose(res["bn2"][:, 0], res["bn2"][:, 1], rtol=0, atol=0)   # top copy

    meta = {"oracle": "WRF v4.8.0 dyn_em/module_diffusion_em.F (unmodified): "
                      "compute_diff_metrics, cal_deform_and_div, calculate_km_kh km_opt=3",
            "repo": "https://github.com/wrf-model/WRF", "sha": SHA,
            "wrf_head": subprocess.run(["git", "-C", wrf, "rev-parse", "HEAD"],
                                       capture_output=True, text=True, check=True).stdout.strip(),
            "generator": "scripts/validate/oracle/smag_wrf_km3/{build.sh,make_fixture.py,"
                         "driver.F90,stubs.F90}",
            "dt": dt, "dt_short": DT_SHORT, "wrf_c_s": cfg.wrf_c_s, "wrf_prandtl": cfg.wrf_prandtl,
            "mix_upper_bound": cfg.mix_upper_bound, "inputs": str(inputs)}
    assert meta["wrf_head"] == SHA
    st = {k: np.stack([c[k] for _, c in cols]) for k in
          ("T", "q", "qc", "qi", "u", "v", "p_full", "p_half", "zi")}
    np.savez(out, meta=json.dumps(meta), names=np.array([n for n, _ in cols]), dt=dt,
             **st, **res)
    print("FIXTURE_OK", out, res["xkmv"].shape)


if __name__ == "__main__":
    if sys.argv[1] == "constants":
        write_constants(*sys.argv[2:4])
    elif sys.argv[1] == "run":
        run(*sys.argv[2:6])
    else:
        raise SystemExit(f"unknown mode {sys.argv[1]!r}")
