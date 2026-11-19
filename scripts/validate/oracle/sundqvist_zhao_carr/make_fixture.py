"""Generate the Sundqvist / Zhao-Carr precipitation oracle fixture by RUNNING the oracle.

Oracle: NCAR/ccpp-physics v7.0.1 (commit
da75531bf90a799e4c0dcb988cc591d82bd024de),
``physics/MP/Zhao_Carr/zhaocarr_precpd.f`` (``zhaocarr_precpd_run``) and
``physics/hooks/machine.F`` compiled UNMODIFIED (``build.sh``), with the
``funcphys::fpvs`` lookup table replaced by legoESM's liquid Tetens curve
(``stub_funcphys.F90``) -- the build the 2026-10-19 tier-2 audit ran.  The
physical constants passed in are legoESM's (g, L_v, L_f, c_pd, R_d/R_v).

Columns (CAM L32, level 0 = model top):
  * the six tier-1 audit warm columns (reused from
    ``scripts/validate/oracle/morrison_mg2_warm_rain/make_fixture.py``);
  * ``cold_mixed`` (the tier-2 audit's cold deck), ``winter_snow`` (an ice deck
    over a -15..0 degC layer over a melting/sub-saturated boundary layer) and
    ``top_only`` (condensate only at the model top: a skipped column),
    ``trace_snow`` (trace snow: the oracle's sr = 0 below 1e-13 kg m-2) and
    ``caps`` (an ice deck over a warm spike, a cloudy cell just above RH_crit and
    a cold dry cell: the melting cap, the 0.01 cloud-cover floor and the
    over-evaporation cap bind) and ``floor`` (one thin warm cloud at RH 0.802:
    the 0.01 cloud-cover floor sets the release where its exponent is not capped);
  * kernel-only: ``edge`` (an RH 1.02 cell) and ``super_neg`` (super-saturated
    cells and negative condensate), which the whole scheme's condensation
    surrogate would alter.

Each parameter set in ``RUNS`` is one oracle call over every column.

Usage (CPU node; needs gfortran and the ccpp-physics checkout)::

    python scripts/validate/oracle/sundqvist_zhao_carr/make_fixture.py \\
        --ccpp /burg/glab/users/pg2328/oracles/ccpp-physics --build <dir> \\
        --out tests/atmosphere/oracle_fixtures/sundqvist_zhao_carr.npz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
MG2_GEN = HERE.parent / "morrison_mg2_warm_rain" / "make_fixture.py"
COL_FIELDS = ("T", "q_v", "q_c", "p_half", "p_full", "rho")
# Whole-scheme columns are sub-saturated everywhere (the condensation surrogate,
# which has no oracle, is then exactly zero); "edge" holds an RH 1.02 cell.
WHOLE_CALL = ("trop_deep", "subtrop_sc", "midlat_cont", "trade_cu",
              "polar_dry", "cold_mixed", "winter_snow", "top_only", "trace_snow",
              "caps", "floor")
KERNEL_ONLY = ("edge", "super_neg")
# name -> (dt [s], u00k, prautco, psautco, evpco, wminco liquid, wminco ice)
RUNS = {
    "A": (600.0, 0.8, 1.0e-3, 8.0e-4, 2.0e-5, 1.0e-5, 3.0e-5),
    "B": (1800.0, 0.85, 1.5e-4, 5.0e-4, 5.0e-4, 0.0, 0.0),
    "C": (600.0, 0.8, 1.0e-3, 0.0, 0.0, 0.0, 0.0),
}
SOURCE = "physics/MP/Zhao_Carr/zhaocarr_precpd.f"


def _load_mg2_generator():
    spec = importlib.util.spec_from_file_location("mg2_make_fixture", MG2_GEN)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def build_columns():
    """All fixture columns as dicts with ``name`` + COL_FIELDS (level 0 = top)."""
    from legoesm import constants as C
    from legoesm.thermo import saturation_specific_humidity
    import jax.numpy as jnp

    def qsat(T, p):
        return np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(p)))

    warm = _load_mg2_generator().build_columns()
    cols = [{"name": c["name"], **{k: np.asarray(c[k], float) for k in COL_FIELDS}}
            for c in warm]
    mid = next(c for c in cols if c["name"] == "midlat_cont")
    ph, pf = mid["p_half"], mid["p_full"]

    def add(name, T, rh, qc):
        qv = rh * qsat(T, pf)
        cols.append(dict(name=name, T=T, q_v=qv, q_c=qc, p_half=ph, p_full=pf,
                         rho=pf / (C.R_d * T * (1.0 + (C.R_v / C.R_d - 1.0) * qv))))

    # cold mixed deck (tier-2 audit recipe): T - 22 K, RH 0.985 in 750-350 hPa
    T = np.maximum(mid["T"] - 22.0, 205.0)
    deck = (pf > 3.5e4) & (pf < 7.5e4)
    rh = np.where(pf < 2.0e4, 0.2, np.where(deck, 0.985, 0.6))
    add("cold_mixed", T, rh, np.where(deck, 2.0e-4, 0.0))

    # winter snow: ice deck 350-550 hPa, -15..0 C layer 550-750 hPa holding
    # condensate under it, sub-saturated melting layer below 850 hPa
    T = np.maximum(mid["T"] - 10.0, 205.0)
    ice = (pf > 3.5e4) & (pf < 5.5e4)
    mix = (pf >= 5.5e4) & (pf < 7.5e4)
    rh = np.where(ice, 0.97, np.where(mix, 0.9, np.where(pf > 8.5e4, 0.6, 0.75)))
    rh = np.where(pf < 2.0e4, 0.2, rh)
    qc = np.where(ice, 4.0e-4, np.where(mix, 1.0e-4, 0.0))
    add("winter_snow", T, rh, qc)

    # condensate only at the model top: the oracle skips the column
    qc = np.zeros_like(pf)
    qc[0] = 1.0e-5
    add("top_only", mid["T"].copy(), np.full_like(pf, 0.5), qc)

    # trace snow: 1e-17 kg/kg of cold cloud (eligible only where wminco = 0)
    T = np.full_like(pf, 250.0)
    add("trace_snow", T, np.full_like(pf, 0.95), np.where(pf > 5.0e4, 1.0e-17, 0.0))

    # caps: ice deck 350-500 hPa (245 K); first level below 520 hPa a 290 K
    # spike (snow melts at the cap); next level liquid cloud at RH 0.802 (cloud
    # cover b < 0.01 for RH_crit 0.8); next level 230 K and dry (rain
    # evaporation limited by the saturation deficit); the rest midlat_cont
    T = mid["T"].copy()
    rh = np.full_like(pf, 0.5)
    qc = np.zeros_like(pf)
    deck = (pf > 3.5e4) & (pf < 5.0e4)
    T = np.where(deck, 245.0, T)
    rh = np.where(deck, 0.95, rh)
    qc = np.where(deck, 5.0e-4, qc)
    k = int(np.argmax(pf > 5.2e4))
    T[k] = 290.0
    T[k + 1], rh[k + 1], qc[k + 1] = 290.0, 0.802, 5.0e-4
    T[k + 2], rh[k + 2] = 230.0, 0.7
    add("caps", T, rh, qc)

    # floor: a single thin warm cloud (2e-6 kg/kg) at RH 0.802, dry elsewhere
    T = mid["T"].copy()
    rh = np.full_like(pf, 0.5)
    qc = np.zeros_like(pf)
    k = int(np.argmin(np.abs(pf - 9.0e4)))
    rh[k], qc[k] = 0.802, 2.0e-6
    add("floor", T, rh, qc)

    # kernel-only: super-saturated cloud, a negative-condensate cell that vapour
    # fills, one it cannot fill, and condensate below the presence threshold
    T = mid["T"].copy()
    rh = np.full_like(pf, 0.7)
    qc = np.zeros_like(pf)
    sup = (pf > 6.0e4) & (pf < 7.5e4)
    rh = np.where(sup, 1.03, rh)
    qc = np.where(sup, 3.0e-4, qc)
    k_fill = int(np.argmin(np.abs(pf - 8.0e4)))
    k_drain = int(np.argmin(np.abs(pf - 3.0e4)))
    k_tiny = int(np.argmin(np.abs(pf - 5.0e4)))
    qc[k_fill] = -1.0e-6
    qc[k_tiny] = 5.0e-21
    add("super_neg", T, rh, qc)
    cols[-1]["q_v"][k_drain] = 1.0e-7
    cols[-1]["q_c"][k_drain] = -2.0e-6
    return cols


def write_inputs(cols, path, run):
    """Driver input (drv_zc.F90): one call, every column, levels BOTTOM->TOP."""
    from legoesm import constants as C
    dt, u00k, prautco, psautco, evpco, wl, wi = RUNS[run]
    eps = C.R_d / C.R_v
    km = len(cols[0]["T"])
    with open(path, "w") as f:
        hvap, hfus = C.L_v, C.L_f  # latent-ok: Zhao-Carr oracle constant (hvap, hfus inputs)
        f.write(f"{len(cols)} {km} {dt!r} {C.g!r} {hvap!r} {hfus!r} "
                f"{C.T_zero_zhao_carr!r} {C.c_pd!r} {eps!r} {eps - 1.0!r} "
                f"{prautco!r} {psautco!r} {evpco!r} {wl!r} {wi!r}\n")
        for c in cols:
            dp = np.diff(c["p_half"])
            for k in range(km - 1, -1, -1):
                f.write(" ".join(f"{x:.17e}" for x in (
                    dp[k], c["p_full"][k], c["q_v"][k], c["q_c"][k], c["T"][k],
                    u00k)) + "\n")


def read_outputs(path, ncol, km):
    rows = [list(map(float, ln.split())) for ln in open(path)]
    out, i = [], 0
    for _ in range(ncol):
        rn, sr = rows[i]
        blk = np.array(rows[i + 1:i + 1 + km])[::-1]      # back to top -> bottom
        i += 1 + km
        out.append(dict(rn=rn, sr=sr, q=blk[:, 0], cwm=blk[:, 1], t=blk[:, 2]))
    return out


def _git(path, *args):
    return subprocess.run(["git", "-C", str(path), *args], capture_output=True,
                          text=True, check=True).stdout.strip()


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--ccpp", type=Path, required=True)
    ap.add_argument("--build", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    import jax
    jax.config.update("jax_enable_x64", True)
    subprocess.run(["bash", str(HERE / "build.sh"), str(a.ccpp), str(a.build)], check=True)
    cols = build_columns()
    names = [c["name"] for c in cols]
    km = len(cols[0]["T"])
    fx = {"names": np.array(names)}
    for k in COL_FIELDS:
        fx[f"col_{k}"] = np.stack([c[k] for c in cols])
    for run, vals in RUNS.items():
        fx[f"run_{run}"] = np.array(vals)
        fin, fout = a.build / f"in_{run}.txt", a.build / f"out_{run}.txt"
        write_inputs(cols, fin, run)
        subprocess.run([str(a.build / "drv_zc"), str(fin), str(fout)], check=True)
        res = read_outputs(fout, len(cols), km)
        for k in ("q", "cwm", "t"):
            fx[f"ora_{run}_{k}"] = np.stack([r[k] for r in res])
        for k in ("rn", "sr"):
            fx[f"ora_{run}_{k}"] = np.array([r[k] for r in res])
    src = (a.ccpp / SOURCE).read_bytes()
    meta = dict(
        oracle="NCAR/ccpp-physics", tag="v7.0.1", sha=_git(a.ccpp, "rev-parse", "HEAD"),
        files=[SOURCE, "physics/hooks/machine.F"],
        source_sha256=hashlib.sha256(src).hexdigest(),
        fpvs="stub: legoESM liquid Tetens (stub_funcphys.F90)",
        flags="-O0 -fno-fast-math -ffpe-trap=invalid,zero,overflow -fdefault-real-8 -fdefault-double-8",
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        whole_call=list(WHOLE_CALL), kernel_only=list(KERNEL_ONLY),
        run_fields="dt u00k prautco psautco evpco wminco_liquid wminco_ice",
        legoesm_sha=_git(HERE, "rev-parse", "HEAD"),
        generator=("python scripts/validate/oracle/sundqvist_zhao_carr/make_fixture.py "
                   f"--ccpp <ccpp-physics> --build <dir> --out {a.out.name}"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


if __name__ == "__main__":
    main()
