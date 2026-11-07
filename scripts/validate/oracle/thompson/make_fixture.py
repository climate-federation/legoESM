"""Generate the WRF v4.8.0 Thompson oracle fixture by RUNNING the oracle.

Oracle: WRF ``phys/module_mp_thompson.F`` (tag v4.8.0, commit
06d4240ae989cc3e50af412bb472df3d9048783c) and ``phys/module_mp_radar.F``,
compiled UNMODIFIED in fp64 (``build.sh``), called through its own
``thompson_init`` + ``mp_gt_driver`` entry points as ``mp_physics=8`` (no
aerosol or hail arrays passed), see ``thompson_driver.F90``.  A second,
write-only instrumented build (``patch_dump.py``) also returns mp_thompson's
tendency arrays; its final states must be bitwise identical to the
unpatched build's, which this script checks.

Inputs (all in the oracle's own variables: water species are MIXING
RATIOS, numbers per kg, level 0 = bottom):

* ``cols``: CAM L32 levels with published-sounding shapes (the five columns
  of the 2026-10-19 tier-2 audit, rebuilt analytically here, plus columns
  aimed at the branches those five miss), vapour from the oracle's own
  Flatau fits; run at dt = 300 s and dt = 60 s (the scheme sends warm
  riming to rain only above 120 s).
* ``snow`` / ``ice``: the audit's single-level point families (snow alone,
  cloud ice alone, over temperature x content x ice saturation), dz = 1e9 m
  so levels are independent; dt = 300 s.

Usage (CPU node; needs gfortran and the WRF checkout)::

    bash scripts/validate/oracle/thompson/build.sh <WRF> <bld>
    python scripts/validate/oracle/thompson/make_fixture.py --wrf <WRF> \\
        --build <bld> --run <scratch dir> \\
        --out tests/atmosphere/microphysics/oracle_fixtures/thompson_wrf.npz
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import subprocess
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
DTS = (300.0, 60.0)
FIN = ("T", "p", "dz", "w", "qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr")
FOUT = ("T", "qv", "qc", "qi", "qr", "qs", "qg", "ni", "nr")
TEN = ("tten", "qvten", "qcten", "qiten", "qrten", "qsten", "qgten", "niten",
       "nrten", "ncten")
PPT = ("rainncv", "snowncv", "graupelncv")
WRF_SHA = "06d4240ae989cc3e50af412bb472df3d9048783c"
SOURCE_SHA256 = {
    "phys/module_mp_thompson.F": "fabf19e2a9073cff886e882b187080bfdf089d3fd40c0fce1d19bc93b1e5e802",
    "phys/module_mp_radar.F": "fa5e06a4d64bbc32089a1357bfa5c57e6c682e5b68c706842698c2b0f292e68e",
}
META = dict(
    oracle="WRF phys/module_mp_thompson.F (mp_physics=8)", tag="v4.8.0", sha=WRF_SHA,
    flags="-O2 -cpp -ffree-form -ffree-line-length-none -fdefault-real-8 -fdefault-double-8",
    driver="scripts/validate/oracle/thompson/thompson_driver.F90 (thompson_init + mp_gt_driver)",
)


def provenance(wrf):
    """Check the checkout is the pinned commit and the two compiled sources
    are byte-identical to it; record what was measured."""
    head = subprocess.run(["git", "-C", str(wrf), "rev-parse", "HEAD"], check=True,
                          capture_output=True, text=True).stdout.strip()
    if head != WRF_SHA:
        raise SystemExit(f"WRF checkout is at {head}, expected {WRF_SHA}")
    got = {k: hashlib.sha256((Path(wrf) / k).read_bytes()).hexdigest() for k in SOURCE_SHA256}
    if got != SOURCE_SHA256:
        raise SystemExit(f"oracle source sha256 mismatch: {got}")
    fc = subprocess.run(["gfortran", "--version"], check=True, capture_output=True,
                        text=True).stdout.splitlines()[0]
    return dict(checkout_head=head, source_sha256=got, compiler=fc)


def _sat(p, t, ice):
    """The oracle's RSLF/RSIF (module_mp_thompson.F:5383-5445), NumPy."""
    c = ((.609868993E03, .499320233E02, .184672631E01, .402737184E-1,
          .565392987E-3, .521693933E-5, .307839583E-7, .105785160E-9,
          .161444444E-12) if ice else
         (.611583699E03, .444606896E02, .143177157E01, .264224321E-1,
          .299291081E-3, .203154182E-5, .702620698E-8, .379534310E-11,
          -.321582393E-13))
    x = np.maximum(-80.0, t - 273.16)
    e = c[0] + x * (c[1] + x * (c[2] + x * (c[3] + x * (c[4] + x * (c[5] + x * (
        c[6] + x * (c[7] + x * c[8])))))))
    e = np.minimum(e, p * 0.15)
    den = np.maximum(1.e-4, p - e) if ice else p - e
    return 0.622 * e / den  # const-ok: the oracle's own epsilon literal (RSLF/RSIF)


def build_columns():
    """Columns as dicts of (nz,) arrays, level 0 = BOTTOM (oracle order)."""
    from legoesm import constants as C
    from legoesm.grids.vertical import CAM6_L32_HYAI, CAM6_L32_HYBI, CAM6_L32_P0
    hyai = np.array(CAM6_L32_HYAI)
    hybi = np.array(CAM6_L32_HYBI)
    nlev = len(hyai) - 1
    rho_i, rho_w = 890.0, 1000.0
    cols = []

    def base(ps, Ts, gam, Ttrop):
        ph = hyai * CAM6_L32_P0 + hybi * ps
        pf = 0.5 * (ph[1:] + ph[:-1])
        T = np.maximum(Ts * (pf / ps) ** (C.R_d * gam / C.g), Ttrop)
        return ph, pf, T

    def n_of(q, D, rho_x):  # per-kg number for mass q at volume diameter D
        return np.where(q > 0, q / (np.pi / 6.0 * rho_x * D ** 3), 0.0)

    def finish(name, ph, pf, T, rh_l, rh_i, qc, qr, Dr, qi, Di, qs, qg, w=None,
               nr=None, ni=None, qv=None):
        rho = pf / (C.R_d * T)
        dz = (ph[1:] - ph[:-1]) / (C.g * rho)
        if qv is None:
            qv = np.where(rh_i > 0, rh_i * _sat(pf, T, True), rh_l * _sat(pf, T, False))
        col = dict(T=T, p=pf, dz=dz, w=np.zeros(nlev) if w is None else w, qv=qv,
                   qc=qc, qi=qi, qr=qr, qs=qs, qg=qg,
                   ni=n_of(qi, Di, rho_i) if ni is None else ni,
                   nr=n_of(qr, Dr, rho_w) if nr is None else nr)
        cols.append((name, {k: np.asarray(v, dtype=np.float64)[::-1].copy()
                            for k, v in col.items()}))

    z = lambda: np.zeros(nlev)  # noqa: E731
    # 1 tropical deep convective (GATE/TOGA-COARE-shaped), melting ~560 hPa
    ph, pf, T = base(101000.0, 300.0, 6.2e-3, 195.0)
    P = pf / 100.0
    rh_l = np.where(P > 600, 0.85, 0.6)
    qc = np.where((P <= 650) & (P >= 420), 2e-4, 0.0)
    rh_l = np.where(qc > 0, 1.0, rh_l)
    qr = np.where(P >= 450, np.interp(P, [450, 560, 1100], [1e-4, 4e-4, 3e-4]), 0.0)
    qg = np.where((P <= 650) & (P >= 300), 3e-4, 0.0)
    qs = np.where((P <= 560) & (P >= 200), 2e-4, 0.0)
    qi = np.where((P <= 450) & (P >= 120), 5e-5, 0.0)
    rh_i = np.where((qi > 0) & (qc == 0), 1.02, 0.0)
    finish("trop_deep", ph, pf, T, rh_l, rh_i, qc, qr, 500e-6, qi, 60e-6, qs, qg)
    # 2 midlatitude frontal nimbostratus, freezing level ~800 hPa
    ph, pf, T = base(100000.0, 285.0, 6.5e-3, 215.0)
    P = pf / 100.0
    rh_l = np.where(P > 800, 0.95, 0.7)
    qc = np.where((P <= 760) & (P >= 620), 1e-4, 0.0)
    rh_l = np.where(qc > 0, 1.0, rh_l)
    qs = np.where((P <= 780) & (P >= 350), 3e-4, 0.0)
    qs = np.where((P > 780) & (P <= 860), 1e-4, qs)
    qi = np.where((P <= 620) & (P >= 280), 3e-5, 0.0)
    qg = np.where((P <= 720) & (P >= 600), 5e-5, 0.0)
    qr = np.where(P >= 780, 2e-4, 0.0)
    rh_i = np.where((qi > 0) & (qc == 0), 1.0, 0.0)
    finish("midlat_frontal", ph, pf, T, rh_l, rh_i, qc, qr, 400e-6, qi, 80e-6, qs, qg)
    # 3 Arctic mixed-phase stratocumulus (M-PACE/ISDAC-shaped), Ts 258 K
    ph, pf, T = base(101000.0, 258.0, 5.0e-3, 210.0)
    P = pf / 100.0
    qc = np.where((P <= 900) & (P >= 840), 2e-4, 0.0)
    rh_l = np.where(qc > 0, 1.0, 0.8)
    qi = np.where((P <= 900) & (P >= 820), 1e-5, 0.0)
    qs = np.where(P >= 840, 3e-5, 0.0)
    rh_i = np.where((qc == 0) & (P > 900), 1.0, 0.0)
    finish("arctic_mpsc", ph, pf, T, rh_l, rh_i, qc, z(), 500e-6, qi, 40e-6, qs, z())
    # 4 tropical upper troposphere: anvil cirrus, pristine ice-supersaturated
    # layer, detrained supercooled water below -40 C
    ph, pf, T = base(101000.0, 300.0, 6.2e-3, 192.0)
    P = pf / 100.0
    rh_l = np.full(nlev, 0.5)
    qi = np.where((P <= 300) & (P >= 230), 1e-5, 0.0)
    rh_i = np.where(qi > 0, 1.05, 0.0)
    rh_i = np.where((P < 230) & (P >= 150), 1.55, rh_i)
    qc = np.where((P <= 270) & (P >= 240), 5e-5, 0.0)
    rh_l = np.where(qc > 0, 1.0, rh_l)
    rh_i = np.where(qc > 0, 0.0, rh_i)
    finish("trop_cirrus", ph, pf, T, rh_l, rh_i, qc, z(), 500e-6, qi, 30e-6, z(), z())
    # 5 edge / threshold column: one targeted case per level
    ph, pf, T = base(100000.0, 290.0, 6.5e-3, 210.0)
    rh_l = np.full(nlev, 0.4)
    rh_i, qc, qr, qi, qs, qg = z(), z(), z(), z(), z(), z()
    Dr = np.full(nlev, 500e-6)
    Di = np.full(nlev, 50e-6)

    def at(t):
        return int(np.argmin(np.abs(T - t)))
    k = at(270.0); qr[k] = 3e-4; rh_l[k] = 1.0                         # Bigg window
    k = at(273.6); qi[k] = 2e-5; qs[k] = 2e-4; qg[k] = 2e-4; qc[k] = 1e-4; rh_l[k] = 1.0  # just above 0 C (warm riming)
    k = at(255.0); qi[k] = 2e-12; Di[k] = 20e-6; rh_i[k] = 1.05          # trace ice
    k = at(245.0); qi[k] = 1e-5; Di[k] = 0.5e-6; rh_i[k] = 1.03          # xDi < 5 um
    k = at(237.0); qi[k] = 1e-5; Di[k] = 1500e-6; rh_i[k] = 1.03         # xDi > 300 um
    k = at(262.0); qg[k] = 2e-8; qc[k] = 1e-4; rh_l[k] = 1.0             # near graupel gate
    k = at(250.0); qi[k] = 2e-5; qs[k] = 2e-4; rh_i[k] = 1.04            # ice/snow split
    finish("edge", ph, pf, T, rh_l, rh_i, qc, qr, Dr, qi, Di, qs, qg)
    # 6 heavy riming at -3..-8 C: Hallett-Mossop, rimed snow -> graupel
    ph, pf, T = base(100000.0, 284.0, 6.5e-3, 215.0)
    P = pf / 100.0
    rim = (T < 271.0) & (T > 262.0)
    qc = np.where(rim, 1.2e-3, np.where((T >= 271.0) & (T < 278.0), 4e-4, 0.0))
    rh_l = np.where(qc > 0, 1.0, 0.8)
    qs = np.where((T < 273.0) & (T > 245.0), 5e-4, 0.0)
    qg = np.where(rim, 1.5e-3, 0.0)
    qi = np.where((T < 266.0) & (T > 240.0), 2e-5, 0.0)
    qr = np.where(T >= 271.0, 3e-4, 0.0)
    rh_i = np.where((qc == 0) & (qi > 0), 1.01, 0.0)
    finish("riming", ph, pf, T, rh_l, rh_i, qc, qr, 800e-6, qi, 40e-6, qs, qg)
    # 7 dry column (no microphysics) with one negative vapour value
    ph, pf, T = base(101000.0, 295.0, 6.5e-3, 205.0)
    rh_l = np.full(nlev, 0.3)
    qv = rh_l * _sat(pf, T, False)
    qv[nlev // 2] = -1.0e-6                     # interior: neighbour-mean repair
    qv[1] = -1.0e-6                             # second level from the top: 1e-7 repair
    finish("dry", ph, pf, T, rh_l, z(), z(), z(), 500e-6, z(), 50e-6, z(), z(), qv=qv)
    # 8 warm rain shaft: big drops (break-up), sub-cloud evaporation incl. the
    # fast branch, fog layer with cloud sedimentation, rain number unset
    ph, pf, T = base(101300.0, 302.0, 6.0e-3, 200.0)
    P = pf / 100.0
    qc = np.where((P <= 850) & (P >= 650), 8e-4, 0.0)
    qc = np.where(P > 990, 3e-4, qc)                                      # fog
    rh_l = np.where(qc > 0, 1.0, np.where(P > 850, 0.55, 0.9))
    qr = np.where((P >= 650) & (P <= 950), 1.5e-3, 0.0)
    qr = np.where((P > 950) & (P <= 990), 5e-9, qr)                       # trace rain
    Dr = np.where(P < 800, 2.4e-3, 1.0e-3)
    nr = np.where((P > 900) & (P <= 950), 0.0, (qr > 0) * qr / (np.pi / 6 * 1000.0 * Dr ** 3))
    w = np.where(P > 995, 0.3, 0.05)            # fog level above the 0.1 m/s settling gate
    finish("warm_rain", ph, pf, T, rh_l, z(), qc, qr, Dr, z(), 50e-6, z(), z(), w=w, nr=nr)
    # 9 stratiform melting layer: subsaturated melting snow/graupel
    # (wet-bulb), warm rain-snow/graupel collisions, graupel-shielded evaporation
    ph, pf, T = base(100000.0, 290.0, 6.5e-3, 210.0)
    P = pf / 100.0
    melt = (T > 273.2) & (T < 280.0)
    qs = np.where((T < 279.0) & (T > 240.0), 6e-4, 0.0)
    qg = np.where((T < 281.0) & (T > 255.0), 8e-4, 0.0)
    qr = np.where(T > 272.0, 6e-4, 0.0)
    rh_l = np.where(melt, 0.8, np.where(T > 280.0, 0.7, 0.95))
    qc = np.where((T < 272.0) & (T > 262.0), 1e-4, 0.0)
    rh_l = np.where(qc > 0, 1.0, rh_l)
    finish("melting", ph, pf, T, rh_l, z(), qc, qr, 1.2e-3, z(), 50e-6, qs, qg)
    # 10 supercooled drizzle under snow: cold rain-snow collisions, rain and
    # cloud freezing tables, homogeneous freezing of trace rain
    ph, pf, T = base(100000.0, 268.0, 5.5e-3, 205.0)
    P = pf / 100.0
    qc = np.where((T < 266.0) & (T > 250.0), 3e-4, 0.0)
    qc = np.where((T < 236.0) & (T > 233.0), 2e-5, qc)
    qr = np.where(T > 250.0, 2e-4, np.where((T < 235.0) & (T > 230.0), 5e-7, 0.0))
    qs = np.where((T < 262.0) & (T > 235.0), 4e-4, 0.0)
    qi = np.where((T < 255.0) & (T > 225.0), 1.5e-5, 0.0)
    rh_l = np.where(qc > 0, 1.0, 0.9)
    rh_i = np.where((qc == 0) & (qi > 0), 0.85, 0.0)                      # sublimating ice
    finish("sc_drizzle", ph, pf, T, rh_l, rh_i, qc, qr, 300e-6, qi, 100e-6, qs, z(),
           ni=np.where(qi > 0, 0.0, 0.0))
    return cols


def build_points():
    """Single-level point families of the 2026-10-19 tier-2 audit
    (thompson_microphysics/py/harness.py ``build_points``), as columns of
    independent levels (dz = 1e9 m, so no level feeds another): snow alone
    over temperature x q_s x ice saturation, and cloud ice alone over
    temperature x q_i x mean diameter x ice saturation.  Vapour is set from
    the oracle's own RSIF.  Returns (snow, ice) lists of (name, column)."""
    from legoesm import constants as C
    tc_s = [-70., -55., -45., -35., -30., -25., -20., -15., -10., -5., -2., -0.5, 0.5, 2., 4.]
    qs_l = [1e-8, 1e-6, 1e-5, 1e-4, 1e-3, 3e-3]
    si_l = [-0.4, -0.1, -0.01, 0.01, 0.05, 0.15]
    tc_i = [-50., -40., -30., -20., -10., -5.]
    qi_l = [1e-7, 1e-6, 1e-5, 1e-4]
    di_l = [30e-6, 100e-6]
    si_i = [-0.1, 0.03, 0.3]
    rho_i = 890.0

    def p_of_T(T):  # the audit's standard-lapse pressure for each temperature
        return np.maximum(1.0e5 * (T / 288.0) ** (C.g / (C.R_d * 0.0065)), 1.0e4)  # coeff-ok: std lapse

    def col(T, qv, qs, qi, ni):
        n = len(T)
        z = np.zeros(n)
        return dict(T=T, p=p_of_T(T), dz=np.full(n, 1e9), w=z, qv=qv, qc=z, qi=qi, qr=z,
                    qs=qs, qg=z, ni=ni, nr=z)

    snow, ice = [], []
    for tc in tc_s:
        rows = [(q, si) for q in qs_l for si in si_l]
        T = np.full(len(rows), C.T_freeze + tc)
        qv = np.array([1.0 + si for _, si in rows]) * _sat(p_of_T(T), T, True)
        z = np.zeros(len(rows))
        snow.append((f"snow_T{tc:+.1f}", col(T, qv, np.array([q for q, _ in rows]), z, z)))
    for tc in tc_i:
        rows = [(q, d, si) for q in qi_l for d in di_l for si in si_i]
        T = np.full(len(rows), C.T_freeze + tc)
        qv = np.array([1.0 + r[2] for r in rows]) * _sat(p_of_T(T), T, True)
        qi = np.array([r[0] for r in rows])
        lam = 4.0 / np.array([r[1] for r in rows])     # exponential PSD, mass-weighted D = 4/lam
        ice.append((f"ice_T{tc:+.1f}", col(T, qv, np.zeros(len(rows)), qi,
                                           qi * lam ** 3 / (np.pi * rho_i))))
    return snow, ice


def write_input(path, cols, dt):
    with open(path, "w") as f:
        nz = len(cols[0][1]["T"])
        f.write(f"{len(cols)} {nz} {dt!r}\n")
        for _, c in cols:
            for k in range(nz):
                f.write(" ".join(repr(float(c[v][k])) for v in FIN) + "\n")


def read_output(path, ncol, nz):
    lines = Path(path).read_text().splitlines()
    st = np.zeros((ncol, nz, len(FOUT)))
    te = np.zeros((ncol, nz, len(TEN)))
    pp = np.zeros((ncol, len(PPT)))
    i = 1
    for c in range(ncol):
        pp[c] = [float(x) for x in lines[i].split()[1:]]
        i += 1
        for k in range(nz):
            st[c, k] = [float(x) for x in lines[i].split()[1:]]
            i += 1
            if i < len(lines) and lines[i].startswith("TEN"):
                te[c, k] = [float(x) for x in lines[i].split()[2:]]
                i += 1
    return st, te, pp


def run_set(build, run, tag, cols, dts, data):
    """Run one input set through both oracle builds; store inputs and outputs."""
    for v in FIN:
        data[f"{tag}_in_{v}"] = np.stack([c[v] for _, c in cols])
    env = dict(os.environ, THOMPSON_WRITE_TABLES="1")
    ncol, nz = len(cols), len(cols[0][1]["T"])
    for dt in dts:
        key = f"{tag}_dt{int(dt)}"
        fin = run / f"in_{key}.txt"
        write_input(fin, cols, dt)
        res, txt = {}, {}
        for b in ("orig", "dump"):
            fout = run / f"out_{key}_{b}.txt"
            subprocess.run([str(Path(build) / b / "thompson_driver"), str(fin), str(fout)],
                           cwd=run, env=env, check=True)
            res[b] = read_output(fout, ncol, nz)
            # es25.16 round-trips fp64, so equal text = equal bits (incl. signed zeros)
            txt[b] = [ln for ln in fout.read_text().splitlines() if not ln.startswith("TEN")]
        if txt["orig"] != txt["dump"]:
            raise SystemExit(f"{key}: dump build output differs from the unpatched oracle")
        if not all(np.all(np.isfinite(x)) for x in res["dump"]):
            raise SystemExit(f"{key}: non-finite oracle output")
        st, te, pp = res["dump"]
        for j, v in enumerate(FOUT):
            data[f"{key}_out_{v}"] = st[..., j]
        for j, v in enumerate(TEN):
            data[f"{key}_ten_{v}"] = te[..., j]
        for j, v in enumerate(PPT):
            data[f"{key}_ppt_{v}"] = pp[:, j]


TABLES = ("tcg_racg", "tmr_racg", "tcr_gacr", "tnr_racg", "tnr_gacr",
          "tcs_racs1", "tmr_racs1", "tcs_racs2", "tmr_racs2", "tcr_sacr1",
          "tms_sacr1", "tcr_sacr2", "tms_sacr2", "tnr_racs1", "tnr_racs2",
          "tnr_sacr1", "tnr_sacr2", "tpi_qrfz", "tni_qrfz", "tpg_qrfz", "tnr_qrfz",
          "tpi_qcfz", "tni_qcfz", "tps_iaus", "tni_iaus", "tpi_ide", "t_Efrw", "t_Efsw")
N_TABLE_SAMPLES = 256
IN_INDEX = 28          # Nt_IN index the scheme reads when not aerosol-aware (xni = 1000 /m^3)


def _load_table(path):
    """A table written by thompson_driver.F90 (rank, shape, Fortran-order doubles)."""
    b = path.read_bytes()
    r = int(np.frombuffer(b[:4], np.int32)[0])
    sh = tuple(int(x) for x in np.frombuffer(b[4:4 + 4 * r], np.int32))
    return np.frombuffer(b[4 + 4 * r:], np.float64).reshape(sh, order="F")


def sample_tables(build, run, cols, data):
    """Entries of every lookup table thompson_init builds: the largest entry
    plus N_TABLE_SAMPLES fixed-seed positions among the non-zero ones
    (graupel tables at idx_bg1, freezing tables at IN_INDEX, as the scheme
    reads them).  Indices are 0-based in the port's layout."""
    tdir = run / "tables"
    tdir.mkdir(exist_ok=True)
    fin = run / "in_tables.txt"
    write_input(fin, cols[:1], DTS[0])
    env = dict(os.environ, THOMPSON_WRITE_TABLES="1")
    subprocess.run([str(Path(build) / "orig" / "thompson_driver"), str(fin),
                    str(run / "out_tables.txt"), str(tdir)], cwd=run, env=env, check=True)
    rng = np.random.default_rng(0)
    for name in TABLES:
        a = _load_table(tdir / f"{name}.bin")
        if a.ndim == 5:
            a = a[:, :, 0]                      # dimNRHG = 1 for mp_physics=8
        elif name.endswith("fz"):
            a = a[..., IN_INDEX - 1]
        nz = np.flatnonzero(a)
        pick = np.unique(np.concatenate([[np.argmax(np.abs(a))],
                                         rng.choice(nz, min(N_TABLE_SAMPLES, nz.size),
                                                    replace=False) if nz.size else []]
                                        ).astype(np.int64))
        data[f"table_{name}_idx"] = np.stack(np.unravel_index(pick, a.shape), axis=-1)
        data[f"table_{name}_val"] = a.reshape(-1)[pick]
        data[f"table_{name}_nnz"] = np.int64(nz.size)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--wrf", required=True, help="WRF checkout (v4.8.0)")
    ap.add_argument("--build", required=True, help="build dir; build.sh recompiles into it")
    ap.add_argument("--run", required=True, help="scratch dir for driver I/O")
    ap.add_argument("--out", required=True)
    a = ap.parse_args(argv)
    prov = provenance(a.wrf)
    a.build = str(Path(a.build).resolve())
    # always rebuild from the verified checkout so no stale binary supplies the oracle
    subprocess.run(["bash", str(HERE / "build.sh"), a.wrf, a.build],
                   check=True)
    run = Path(a.run).resolve()
    run.mkdir(parents=True, exist_ok=True)
    for stale in run.glob("*.dat"):   # WRF reads a cached table instead of computing it
        stale.unlink()
    cols = build_columns()
    snow, ice = build_points()
    data = {}
    sets = {"cols": (cols, DTS), "snow": (snow, DTS[:1]), "ice": (ice, DTS[:1])}
    for tag, (cs, dts) in sets.items():
        run_set(a.build, run, tag, cs, dts, data)
    sample_tables(a.build, run, cols, data)
    meta = dict(META, **prov, sets={t: dict(columns=[n for n, _ in cs], dts=list(d))
                            for t, (cs, d) in sets.items()},
                dump_matches_unpatched_bitwise=True)
    np.savez_compressed(Path(a.out).resolve(), meta=json.dumps(meta), **data)
    print("FIXTURE_OK", a.out, {t: len(cs) for t, (cs, _) in sets.items()})


if __name__ == "__main__":
    main()
