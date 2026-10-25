"""Generate the Kuo oracle fixture by RUNNING Mahfouf's MOIST_CONVECTION_KUO.

Oracle: github AJFMAHFOUF/MOIST_CONVECTION_KUO (no tags; master HEAD pinned at
79b78f58f793fbf7849bca51d8ec0a92f39ec170), ``src/kuo_schemes.f90`` Kuo-1965 and
Kuo-Anthes, turned into a column driver by ``build.sh`` (I/O-only edits,
legoESM constants and saturation curve).  fp64.

Columns (the inputs of the 2026-10-19 tier-2 audit, rebuilt here by
:func:`build_columns` from the oracle checkout, this repository and the tier-1
ZM column file ``oracle_audit/zm_deep_convection/inputs/columns.npz``, whose
sha256 is recorded in ``meta``):

* ``g60``: L60 columns --
  - ECMWF tropical profiles shipped with the oracle (``data_in/profiles.dat``,
    own ``w``; the oracle's mean moisture convergence
    ``input_model_mean_values.dat``): every profile with divergence inside the
    oracle's cloud, plus every 50th profile;
  - the 60-level sounding of ``tests/unit/test_kuo_oracle_faithful.py`` at its
    own humidity and at RH 0.98 (``w`` = moisture convergence);
  - two soundings built from it with a saturated +30 K layer at 800 hPa, which
    make the oracle's parcel stop and restart condensing (a second LCL), with
    ``w`` changing sign between the two LCLs (both orders).
* ``g32``: the 15 tier-1 ZM L32 columns (3 MPAS AMIP checkpoint columns, 12
  constructed soundings) and 3 edge columns (subsiding ``w``, zero convergence,
  mixed-sign convergence); moisture convergence = the oracle mean profile
  interpolated in sigma; ``w`` = 0.01 m/s.

Every column is run at ``klev_max = 1`` (the parcel is scanned to the model top,
as legoESM's port does).  The oracle's own ``klev_max = 20`` is also run on
every ECMWF profile and must give identical tendencies (it does not bind there).

Usage (CPU node; gfortran 8.5.0)::

    python scripts/validate/oracle/kuo_mahfouf/make_fixture.py \\
        --oracle <MOIST_CONVECTION_KUO> --zm-columns <zm_deep_convection/inputs/columns.npz> \\
        --out tests/atmosphere/oracle_fixtures/kuo_mahfouf.npz

(it compiles the driver with ``build.sh`` into a temporary directory first).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

ORACLE = dict(repo="AJFMAHFOUF/MOIST_CONVECTION_KUO", tag=None,
              sha="79b78f58f793fbf7849bca51d8ec0a92f39ec170",
              file="src/kuo_schemes.f90")
DT = 900.0  # oracle time step (does not enter the Kuo tendencies)
HDR = ("cvgu", "zint", "zint_t", "zint_q", "rhmean", "bkuo", "rain", "rainA", "zcape", "w_lcl", "n_icond2")
PROF = ("icond", "tc", "qvc", "dT_k65", "dq_k65", "dT_anth", "dq_anth", "gz")
REENTRY = dict(amplitude_K=30.0, p_center_Pa=800e2, width_Pa=20e2, rh=0.99, w_abs=0.01)


def _sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def run_oracle(exe, cols, klev_max):
    ncol, nlev = cols["T"].shape
    with tempfile.TemporaryDirectory() as d:
        lines = [f"{ncol} {nlev} {klev_max}"]
        for i in range(ncol):
            lines.append(f"{cols['p_half'][i, -1]:.17e}")
            for k in range(nlev):
                lines.append(" ".join(f"{x:.17e}" for x in (
                    cols["T"][i, k], cols["q"][i, k], cols["p_full"][i, k],
                    cols["p_half"][i, k], cols["w"][i, k], cols["ptenq"][i, k])))
        (Path(d) / "in.txt").write_text("\n".join(lines) + "\n")
        subprocess.run([str(exe), "in.txt", "out.txt"], cwd=d, check=True)
        out = (Path(d) / "out.txt").read_text().split("\n")
    assert [int(x) for x in out[0].split()] == [ncol, nlev, klev_max]
    hdr = np.zeros((ncol, len(HDR)))
    prof = np.zeros((ncol, nlev, len(PROF)))
    i = 1
    for c in range(ncol):
        hdr[c] = [float(x) for x in out[i].split()]
        i += 1
        for k in range(nlev):
            prof[c, k] = [float(x) for x in out[i].split()]
            i += 1
    o = {k: hdr[:, j] for j, k in enumerate(HDR)}
    o.update({k: prof[:, :, j] for j, k in enumerate(PROF)})
    for v in o.values():
        assert np.all(np.isfinite(v))
    return o


def read_ecmwf_profiles(oracle):
    """The 1535 ECMWF L60 profiles shipped with the oracle and its mean
    moisture-convergence profile (columns 1-5 of input_model_mean_values.dat:
    level, -, -, ptent, ptenq)."""
    lines = (Path(oracle) / "data_in/profiles.dat").read_text().split("\n")
    prof, i = [], 0
    while i < len(lines):
        if "PROFILE" not in lines[i]:
            i += 1
            continue
        ps = float(lines[i + 1].split()[4])
        rows = np.array([[float(x) for x in lines[i + 3 + k].split()] for k in range(60)])
        prof.append((ps, rows))
        i += 63
    ptenq = np.array([float(ln.split()[4]) for ln in
                      (Path(oracle) / "data_in/input_model_mean_values.dat").read_text().split("\n") if ln.strip()])
    assert len(prof) == 1535 and ptenq.shape == (60,)
    return prof, ptenq


def build_columns(oracle, zm_columns):
    """Rebuild the audit's column sets A60, B32, C60k1 and C60rh98k1."""
    prof, ptenq_mean = read_ecmwf_profiles(oracle)
    n = len(prof)
    A = dict(T=np.zeros((n, 60)), q=np.zeros((n, 60)), p_full=np.zeros((n, 60)),
             p_half=np.zeros((n, 61)), w=np.zeros((n, 60)), ptenq=np.tile(ptenq_mean, (n, 1)))
    for j, (ps, r) in enumerate(prof):  # columns: t, qv, p, ph, w, u, v
        A["T"][j], A["q"][j], A["p_full"][j], A["w"][j] = r[:, 0], r[:, 1], r[:, 2], r[:, 4]
        A["p_half"][j, :60], A["p_half"][j, 60] = r[:, 3], ps
    A["names"] = np.array([f"ecmwf_prof{j + 1}" for j in range(n)])

    z = np.load(zm_columns)
    s900 = prof[899][1][:, 2] / prof[899][0]
    T, q, pf, ph = (np.array(z[k]) for k in ("T", "q", "p_full", "p_half"))
    names = [str(x) for x in z["names"]]
    sig = pf / ph[:, -1:]
    pq = np.array([np.interp(sig[i], s900, ptenq_mean) for i in range(len(T))])
    w = np.full_like(T, 0.01)
    j0 = names.index("trop_ocean_deep")
    pk = 3.0e-3 / 86400.0
    p0 = pf[j0]
    mixed = pk * np.exp(-((p0 - 850e2) / 120e2) ** 2) - 1.3 * pk * np.exp(-((p0 - 300e2) / 70e2) ** 2)
    for nm, ww, pp in (("edge_subsiding_w", -w[j0], pq[j0]), ("edge_zero_convergence", w[j0], 0.0 * pq[j0]),
                       ("edge_mixed_sign_convergence", w[j0], mixed)):
        T, q, pf, ph = (np.vstack([a, a[j0:j0 + 1]]) for a in (T, q, pf, ph))
        w, pq = np.vstack([w, ww[None]]), np.vstack([pq, pp[None]])
        names.append(nm)
    B = dict(T=T, q=np.maximum(q, 0.0), p_full=pf, p_half=ph, w=w, ptenq=pq, names=np.array(names))

    import sys
    import jax
    jax.config.update("jax_enable_x64", True)  # the sounding is built with jnp
    sys.path.insert(0, str(Path(__file__).resolve().parents[4] / "tests" / "unit"))
    import test_kuo_oracle_faithful as tk  # the unit-test sounding (w = moisture convergence)
    C = {}
    for nm, cols in (("unit_test_sounding", tk._build_sounding()),
                     ("unit_test_rh098", tk._build_sounding_with_rh(0.98))):
        Tt, qt, pft, pht, pqt = (np.asarray(x) for x in cols)
        C[nm] = dict(T=Tt, q=qt, p_full=pft, p_half=pht, w=pqt.copy(), ptenq=pqt, names=np.array([nm]))
    return {"A60": A, "B32": B, "C60k1": C["unit_test_sounding"], "C60rh98k1": C["unit_test_rh098"]}


def reentry_columns(c):
    """Second-LCL soundings: a saturated warm layer on the unit-test sounding.
    Their oracle condensation pattern (cond, not, cond going up) is asserted in
    :func:`main`."""
    from legoesm.thermo import saturation_specific_humidity
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp

    r = REENTRY
    pf = c["p_full"][0]
    bump = np.exp(-((pf - r["p_center_Pa"]) / r["width_Pa"]) ** 2)
    T = c["T"][0] + r["amplitude_K"] * bump
    qs = np.asarray(saturation_specific_humidity(jnp.asarray(T), jnp.asarray(pf)))
    q = np.where(bump > 0.3, r["rh"] * qs, c["q"][0])
    w_lo_up = np.where(pf >= r["p_center_Pa"], r["w_abs"], -r["w_abs"])  # ascent below, descent above
    out = {k: np.repeat(c[k][:1], 2, axis=0) for k in ("p_full", "p_half", "ptenq")}
    out["T"] = np.stack([T, T])
    out["q"] = np.stack([q, q])
    out["w"] = np.stack([w_lo_up, -w_lo_up])
    out["names"] = np.array(["reentry_w_up_then_down", "reentry_w_down_then_up"])
    return out


def cat(*sets):
    return {k: np.concatenate([s[k] for s in sets]) for k in ("T", "q", "p_full", "p_half", "w", "ptenq", "names")}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--oracle", required=True)
    ap.add_argument("--zm-columns", required=True)
    ap.add_argument("--out", required=True)
    a = ap.parse_args()
    here = Path(__file__).resolve().parent
    build = Path(tempfile.mkdtemp(prefix="kuo_oracle_build_"))
    r = subprocess.run(["bash", str(here / "build.sh"), a.oracle, str(build)],
                       capture_output=True, text=True, check=True)
    assert "BUILD_OK" in r.stdout, r.stdout + r.stderr
    exe = build / "kuo_harm"
    load = build_columns(a.oracle, a.zm_columns)

    # ECMWF profiles: klev_max 20 (oracle) must equal klev_max 1 on the tendencies
    A = load["A60"]
    o20, o1 = run_oracle(exe, A, 20), run_oracle(exe, A, 1)
    for k in ("dT_k65", "dq_k65", "dT_anth", "dq_anth", "icond"):
        assert np.array_equal(o20[k], o1[k]), k
    mixed = np.any((o1["icond"] == 2) & (A["ptenq"] <= 0.0), axis=1)
    keep = mixed | (np.arange(len(mixed)) % 50 == 0)
    A_sub = {k: v[keep] for k, v in A.items()}

    re = reentry_columns(load["C60k1"])
    g60 = cat(A_sub, load["C60k1"], load["C60rh98k1"], re)
    g32 = load["B32"]
    out = {}
    for gname, cols in (("g60", g60), ("g32", g32)):
        o = run_oracle(exe, cols, 1)
        for k in ("T", "q", "p_full", "p_half", "w", "ptenq"):
            out[f"{gname}_{k}"] = cols[k]
        out[f"{gname}_names"] = np.array([str(n) for n in cols["names"]])
        for k, v in o.items():
            out[f"{gname}_ora_{k}"] = v

    # coverage the test relies on
    n60 = list(out["g60_names"])
    for nm in ("reentry_w_up_then_down", "reentry_w_down_then_up"):
        j = n60.index(nm)
        cond = (out["g60_ora_icond"][j][::-1] >= 1).astype(int)  # surface first
        assert np.any((cond[2:] == 1) & (cond[1:-1] == 0) & (cond[:-2] == 1)), nm
    assert np.sum(mixed) > 0 and np.any(out["g60_ora_bkuo"] < 0.0)
    gf = subprocess.run(["gfortran", "--version"], capture_output=True, text=True).stdout.split("\n")[0]
    meta = dict(ORACLE, oracle_file_sha256=_sha256(Path(a.oracle) / ORACLE["file"]), gfortran=gf,
                flags=[ln for ln in r.stdout.split("\n") if ln.startswith("FLAGS: ")][0][7:],
                data_in_sha256={f: _sha256(Path(a.oracle) / "data_in" / f)
                                for f in ("profiles.dat", "input_model_mean_values.dat")},
                constants="legoESM (driver CONST_LEGO block)",
                saturation="legoESM Tetens + q_sat with softplus floor/cap and exact dq_sat/dT (thermo_lego.f90)",
                klev_max=1, klev_max_20_identical_on_ecmwf_profiles=True, dt=DT, entrainment=5.0e-5,
                n_ecmwf_mixed_sign=int(np.sum(mixed)), reentry=REENTRY,
                zm_columns_sha256=_sha256(a.zm_columns),
                driver_diff_sha256=_sha256(here / "driver_vs_oracle.diff"),
                thermo_lego_sha256=_sha256(here / "thermo_lego.f90"),
                driver_sha256=_sha256(build / "kuo_driver.F90"))
    out["meta"] = np.array(json.dumps(meta, sort_keys=True))
    np.savez_compressed(a.out, **out)
    print("wrote", a.out, {g: out[f"{g}_T"].shape for g in ("g60", "g32")},
          "firing", {g: int(np.sum(out[f"{g}_ora_n_icond2"] > 0)) for g in ("g60", "g32")})


if __name__ == "__main__":
    main()
