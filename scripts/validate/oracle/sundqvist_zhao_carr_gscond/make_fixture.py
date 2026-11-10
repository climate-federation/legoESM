"""Generate the Sundqvist / Zhao-Carr CONDENSATION oracle fixture by RUNNING the oracle.

Oracle: NCAR/ccpp-physics v7.0.1 (commit
da75531bf90a799e4c0dcb988cc591d82bd024de),
``physics/MP/Zhao_Carr/zhaocarr_gscond.f`` (``zhaocarr_gscond_run``) and
``physics/hooks/machine.F`` compiled UNMODIFIED (``build.sh``), with
``funcphys::fpvs`` replaced by legoESM's liquid Tetens curve (the stub the
precpd oracle uses).  Two-time-level branch (``dtf = dt``).  The physical
constants passed in are legoESM's (L_v, L_f, c_pd, R_d, R_v, g).

Columns: the precpd fixture's 13 columns
(``../sundqvist_zhao_carr/make_fixture.py``), level 0 = model top.  The
previous-step state ``tp, qp, psp`` is built from prescribed other-process
tendencies ``A_t, A_q, dps/dt`` (``TENDENCIES``): ``tp = T - A_t dt``,
``qp = max(q - A_q dt, epsq)``, ``psp = ps - dps/dt dt``; ``mixed`` draws
per-cell values in the same ranges from a fixed seed.

Each run in ``RUNS`` x ``TENDENCIES`` is one oracle call over every column.
``chain_<run>_*``: the whole scheme's oracle for each precpd run of
``../sundqvist_zhao_carr`` (its ``RUNS``): gscond on a fresh start
(``tp = T``, ``qp = max(q, epsq)``, ``psp = ps``), then the precpd oracle on
gscond's output, over the precpd fixture's whole-call columns.  The
oracle's round-off spread is measured by 20 reruns with every float input
moved by a random -1/0/+1 ulp (``SPREAD_SEEDS``); seed 0 is also rerun
unperturbed and must be bitwise identical.

Usage (CPU node; needs gfortran and the ccpp-physics checkout)::

    python scripts/validate/oracle/sundqvist_zhao_carr_gscond/make_fixture.py \\
        --ccpp /burg/glab/users/pg2328/oracles/ccpp-physics --build <dir> \\
        --out tests/atmosphere/oracle_fixtures/sundqvist_zhao_carr_gscond.npz
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import subprocess
import zlib
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
PRECPD_GEN = HERE.parent / "sundqvist_zhao_carr" / "make_fixture.py"
SOURCE = "physics/MP/Zhao_Carr/zhaocarr_gscond.f"
EPSQ = 2.0e-12
# name -> (dt [s], u)
RUNS = {"A": (600.0, 0.8), "B": (1800.0, 0.85)}
# name -> (A_t [K/s], A_q [kg/kg/s], dps/dt [Pa/s]); None = per-cell draw
TENDENCIES = {
    "moisten": (-1.0e-4, 1.0e-8, 1.0e-2),
    "dry": (2.0e-4, -2.0e-8, -1.0e-2),
    "mixed": None,
}
MIXED_RANGE = (2.0e-4, 2.0e-8, 1.0e-2)
MIXED_SEED = 20261104
SPREAD_SEEDS = 20
OUT_FIELDS = ("t", "q", "cwm", "tp", "qp")


def _load_precpd_generator():
    spec = importlib.util.spec_from_file_location("zc_precpd_make_fixture", PRECPD_GEN)
    gen = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(gen)
    return gen


def previous_state(cols, tend, dt):
    """(tp, qp, psp) stacked over columns for one tendency set."""
    T = np.stack([c["T"] for c in cols])
    q = np.stack([c["q_v"] for c in cols])
    ps = np.array([c["p_half"][-1] for c in cols])
    if tend is None:
        rng = np.random.default_rng(MIXED_SEED)
        at = rng.uniform(-1, 1, T.shape) * MIXED_RANGE[0]
        aq = rng.uniform(-1, 1, T.shape) * MIXED_RANGE[1]
        dps = rng.uniform(-1, 1, ps.shape) * MIXED_RANGE[2]
    else:
        at, aq, dps = tend
    tp = T - at * dt
    qp = np.maximum(q - aq * dt, EPSQ)
    psp = ps - dps * dt
    return tp, qp, psp


def write_inputs(path, dt, u, T, q, cwm, pf, tp, qp, ps, psp):
    from legoesm import constants as C
    eps = C.R_d / C.R_v
    ncol, km = T.shape
    hvap, hfus = C.L_v, C.L_f  # latent-ok: Zhao-Carr oracle constant (hvap, hfus inputs)
    with open(path, "w") as f:
        f.write(f"{ncol} {km} {dt!r} {hvap!r} {C.g!r} {hfus!r} "
                f"{C.T_zero_zhao_carr!r} {C.R_d!r} {C.c_pd!r} {eps!r} {eps - 1.0!r} "
                f"{C.R_v!r}\n")
        for i in range(ncol):
            f.write(f"{ps[i]:.17e} {psp[i]:.17e}\n")
            for k in range(km - 1, -1, -1):
                f.write(" ".join(f"{x:.17e}" for x in (
                    pf[i, k], q[i, k], cwm[i, k], T[i, k], tp[i, k], qp[i, k], u)) + "\n")


def read_outputs(path, ncol, km):
    a = np.loadtxt(path).reshape(ncol, km, len(OUT_FIELDS))[:, ::-1, :]
    return {k: a[:, :, j] for j, k in enumerate(OUT_FIELDS)}


def _ulp(x, rng):
    s = rng.integers(-1, 2, x.shape)
    return np.where(s > 0, np.nextafter(x, np.inf),
                    np.where(s < 0, np.nextafter(x, -np.inf), x))


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
    cols = _load_precpd_generator().build_columns()
    fx = {"names": np.array([c["name"] for c in cols])}
    for k in ("T", "q_v", "q_c", "p_half", "p_full"):
        fx[f"col_{k}"] = np.stack([c[k] for c in cols])
    T, q, cwm, pf = fx["col_T"], fx["col_q_v"], fx["col_q_c"], fx["col_p_full"]
    ps = fx["col_p_half"][:, -1]
    ncol, km = T.shape

    def run_oracle(tag, *inputs):
        fin, fout = a.build / f"in_{tag}.txt", a.build / f"out_{tag}.txt"
        write_inputs(fin, *inputs)
        subprocess.run([str(a.build / "drv_gscond"), str(fin), str(fout)], check=True)
        return read_outputs(fout, ncol, km)

    for run, (dt, u) in RUNS.items():
        fx[f"run_{run}"] = np.array([dt, u])
        for tn, tend in TENDENCIES.items():
            key = f"{run}_{tn}"
            tp, qp, psp = previous_state(cols, tend, dt)
            fx[f"prev_{key}_tp"], fx[f"prev_{key}_qp"], fx[f"prev_{key}_psp"] = tp, qp, psp
            base = (T, q, cwm, pf, tp, qp, ps, psp)
            ref = run_oracle(key, dt, u, *base)
            again = run_oracle(key + "_rerun", dt, u, *base)
            for k in OUT_FIELDS:
                fx[f"ora_{key}_{k}"] = ref[k]
                if not np.array_equal(ref[k], again[k]):
                    raise SystemExit(f"oracle rerun not bitwise: {key} {k}")
            rng = np.random.default_rng(zlib.crc32(key.encode()))
            spread = {k: 0.0 for k in OUT_FIELDS}
            for s in range(SPREAD_SEEDS):
                pert = run_oracle(f"{key}_s{s}", dt, u, *(_ulp(x, rng) for x in base))
                for k in OUT_FIELDS:
                    d = np.max(np.abs(pert[k] - ref[k])) / max(np.max(np.abs(ref[k])), 1e-300)
                    spread[k] = max(spread[k], float(d))
            fx[f"spread_{key}"] = np.array([spread[k] for k in OUT_FIELDS])
    # Whole-scheme chain: gscond (fresh start) -> precpd, the oracle order.
    pgen = _load_precpd_generator()
    subprocess.run(["bash", str(PRECPD_GEN.parent / "build.sh"), str(a.ccpp),
                    str(a.build)], check=True)
    for run, vals in pgen.RUNS.items():
        dt, u = vals[0], vals[1]
        tp, qp, psp = T, np.maximum(q, EPSQ), ps
        g = run_oracle(f"chain_{run}_gscond", dt, u, T, q, cwm, pf, tp, qp, ps, psp)
        cols2 = [dict(c, T=g["t"][i], q_v=g["q"][i], q_c=g["cwm"][i])
                 for i, c in enumerate(cols)]
        fin, fout = a.build / f"chain_{run}_precpd_in.txt", a.build / f"chain_{run}_precpd_out.txt"
        pgen.write_inputs(cols2, fin, run)
        subprocess.run([str(a.build / "drv_zc"), str(fin), str(fout)], check=True)
        res = pgen.read_outputs(fout, ncol, km)
        for k in ("q", "cwm", "t"):
            fx[f"chain_{run}_{k}"] = np.stack([r[k] for r in res])
        for k in ("rn", "sr"):
            fx[f"chain_{run}_{k}"] = np.array([r[k] for r in res])
        fx[f"chain_{run}_params"] = np.array(vals)
    src = (a.ccpp / SOURCE).read_bytes()
    meta = dict(
        oracle="NCAR/ccpp-physics", tag="v7.0.1", sha=_git(a.ccpp, "rev-parse", "HEAD"),
        files=[SOURCE, "physics/hooks/machine.F"],
        source_sha256=hashlib.sha256(src).hexdigest(),
        fpvs="stub: legoESM liquid Tetens (../sundqvist_zhao_carr/stub_funcphys.F90)",
        flags="-O0 -fno-fast-math -ffpe-trap=invalid,zero,overflow -fdefault-real-8 -fdefault-double-8",
        gfortran=subprocess.run(["gfortran", "--version"], capture_output=True,
                                text=True).stdout.split("\n")[0],
        runs="dt u", tendencies={k: v for k, v in TENDENCIES.items()},
        mixed_range=MIXED_RANGE, mixed_seed=MIXED_SEED, spread_seeds=SPREAD_SEEDS,
        out_fields=list(OUT_FIELDS),
        chain_runs=list(pgen.RUNS), chain_whole_call=list(pgen.WHOLE_CALL),
        legoesm_sha=_git(HERE, "rev-parse", "HEAD"),
        generator=("python scripts/validate/oracle/sundqvist_zhao_carr_gscond/make_fixture.py "
                   f"--ccpp <ccpp-physics> --build <dir> --out {a.out.name}"))
    fx["meta"] = np.array(json.dumps(meta))
    a.out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(a.out, **fx)
    print("FIXTURE_OK", a.out, a.out.stat().st_size, "bytes", meta["sha"])


if __name__ == "__main__":
    main()
