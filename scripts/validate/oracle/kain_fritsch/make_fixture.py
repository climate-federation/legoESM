"""Run the WRF KF-Eta oracle (module_cu_kfeta.F, v4.8.0 06d4240ae9) and write the fixture.

usage: python make_fixture.py <columns.npz> <build dir> <out.npz>

<columns.npz> is the tier-1/tier-2 audit column set
(oracle_audit/zm_deep_convection/inputs/columns.npz: 3 real L32 MPAS AMIP
columns and 12 constructed soundings, top-down, specific humidity ``q``).
<build dir> holds kf_driver.x and lut_dump.x from build.sh.

The oracle is fed exactly what ``kain_fritsch_convection`` feeds the port:
levels flipped to bottom-up, QV = the model's ``q`` unchanged, DZ = the
hypsometric layer depth of ``compute_column_geometry`` and RHO = dp/(G*DZ) so
the oracle's DP = RHO*G*DZ is the model's dp, W0AVG = ``w_grid``,
DX = 25 km x ``dtlcl_dx_scale`` (default 1).  Arms vary the constant set
(L = legoESM constants with constant L_v and the legoesm.thermo Tetens curve;
W = WRF share/module_model_constants.F), the winds (u20: u = 20 m/s,
v = 0; nat: the columns' own winds), the grid-scale w, the time step, the
trigger option and the hydrometeor-feedback branch.
"""
import subprocess
import sys
from pathlib import Path

import numpy as np

from legoesm import constants as C
from legoesm import thermo

CUDT = 5.0
DX = 25.0e3
LEGO = dict(G=C.g, CP=C.c_pd, R=C.R_d, EP2=C.epsilon, XLV0=C.L_v,  # latent-ok: KF argument constant XLV0
            XLV1=0.0, XLS0=C.L_v + C.L_f, XLS1=0.0,  # latent-ok: KF argument constants (XLS unused by KF_eta_PARA)
            SVP1=thermo.TETENS_E0_PA / 1000.0, SVP2=thermo.TETENS_A,
            SVP3=C.T_freeze - thermo.TETENS_B_K, SVPT0=C.T_freeze)
WRF = dict(G=9.81, CP=1004.5, R=287.0,  # const-ok: WRF share/module_model_constants.F (oracle arm)
           EP2=287.0 / 461.6,  # const-ok: WRF r_d/r_v (oracle arm)
           XLV0=3.15e6, XLV1=2370.0, XLS0=2.905e6, XLS1=259.532,
           SVP1=0.6112, SVP2=17.67, SVP3=29.65, SVPT0=273.15)  # const-ok: WRF SVP constants (oracle arm)
CONST_ORDER = ("G", "CP", "R", "EP2", "XLV0", "XLV1", "XLS0", "XLS1", "SVP1", "SVP2", "SVP3", "SVPT0")
# name: constants, winds, w [m/s], dt [s], trigger, warm_rain, f_qi, f_qs
ARMS = {
    "L_nat_w05": ("L", "nat", 0.05, 300.0, 3, 1, 0, 0),
    "L_nat_w0": ("L", "nat", 0.0, 300.0, 3, 1, 0, 0),
    "L_nat_w20": ("L", "nat", 0.20, 300.0, 3, 1, 0, 0),
    "L_u20_w05": ("L", "u20", 0.05, 300.0, 3, 1, 0, 0),
    "L_u20_w0": ("L", "u20", 0.0, 300.0, 3, 1, 0, 0),
    "L_nat_w05_dt360": ("L", "nat", 0.05, 360.0, 3, 1, 0, 0),
    "L_nat_w05_t1": ("L", "nat", 0.05, 300.0, 1, 1, 0, 0),
    "L_nat_w05_ice": ("L", "nat", 0.05, 300.0, 3, 0, 0, 0),
    "L_nat_w05_mixi": ("L", "nat", 0.05, 300.0, 3, 0, 1, 1),
    "L_nat_w05_mixs": ("L", "nat", 0.05, 300.0, 3, 0, 0, 1),
    "W_nat_w05": ("W", "nat", 0.05, 300.0, 3, 1, 0, 0),
    "W_u20_w0": ("W", "u20", 0.0, 300.0, 3, 1, 0, 0),
}
DIAG_ARMS = ("L_nat_w05", "L_u20_w05")
OUT_ARRAYS = ("dtdt", "dqdt", "dqcdt", "dqrdt", "dqidt", "dqsdt")
OUT_SCALARS = ("fired", "ishall", "nfatal", "pratec", "nca", "cutop", "cubot")
DIAG_SCALARS = ("ltop", "let", "abe", "vmflcl", "ainc", "timec", "ncount", "fabe", "tder",
                "trppt", "pptflx", "cpr", "lfs", "ldb", "wlcl", "rad", "chmin", "cldhgt",
                "lcl", "au0", "nstep", "kpbl", "lc", "fbfrc", "frc2")
DIAG_ARRAYS = ("umf", "uer", "udr", "dmf", "der", "ddr", "tu", "qu", "qliq", "qice", "detlq",
               "detic", "pptliq", "pptice", "tg", "qg", "eqfrc", "ems")


# Randomised sweep (seeded): perturbed copies of the audit columns over static
# combos x dt x DX, so branches the 12 arms never reach (closure iteration,
# moisture borrowing, grid-area cap, ...) are exercised.  Only SWEEP_PICK
# columns go into the fixture; each is re-run alone and refused if the oracle
# reads its lookup tables out of bounds (fort.98 written) or crashes.
SWEEP_SEED = 0
SWEEP_COMBOS = (("L", 3, 1, 0, 0), ("L", 1, 1, 0, 0), ("W", 3, 1, 0, 0), ("L", 3, 0, 0, 0),
                ("L", 3, 0, 1, 1), ("L", 3, 0, 0, 1))  # consts, trigger, warm_rain, f_qi, f_qs
SWEEP_DTS = (60.0, 300.0, 600.0, 900.0)
SWEEP_DXS = (3.0e3, 12.0e3, 25.0e3, 60.0e3)
SWEEP_NPER = 24
SWEEP_PICK = (  # (cell, column): mutation-detecting columns + one fired column per cell
    (0, 0), (0, 5), (0, 7), (0, 11), (0, 13), (0, 14), (1, 5), (1, 9), (2, 5), (2, 10), (2, 17),
    (2, 20), (2, 23), (3, 6), (3, 19), (4, 2), (5, 3), (5, 22), (6, 20), (7, 23), (8, 3), (9, 8),
    (10, 22), (11, 13), (11, 22), (12, 3), (12, 15), (13, 12), (13, 20), (14, 0), (14, 8),
    (15, 9), (16, 17), (16, 23), (17, 12), (18, 1), (19, 1), (20, 21), (21, 15), (22, 20),
    (23, 15), (24, 20), (25, 4), (26, 12), (27, 19), (28, 0), (28, 6), (29, 5), (30, 0), (31, 7),
    (31, 16), (32, 22), (33, 1), (34, 9), (35, 7), (36, 23), (37, 3), (38, 4), (38, 14), (39, 6),
    (40, 5), (41, 14), (42, 0), (43, 4), (44, 7), (45, 11), (46, 5), (46, 20), (47, 23),
    (48, 18), (49, 23), (50, 4), (51, 16), (52, 6), (53, 9), (54, 23), (55, 6), (56, 17),
    (57, 7), (58, 11), (59, 21), (60, 11), (61, 10), (62, 10), (63, 2), (64, 14), (65, 17),
    (66, 4), (67, 18), (68, 9), (69, 14), (70, 11), (71, 20), (72, 10), (73, 0), (74, 18),
    (75, 6), (76, 12), (77, 8), (78, 6), (79, 2), (80, 14), (81, 12), (82, 20), (83, 20),
    (84, 6), (85, 10), (86, 21), (87, 5), (88, 8), (89, 22), (90, 20), (91, 13), (92, 16),
    (93, 14), (94, 22), (95, 17),
)


def make_sweep(T0, q0, pf0, ph0, u0, v0):
    """Cells in (combo, dt, dx) order, each SWEEP_NPER perturbed columns (top-down)."""
    rng = np.random.default_rng(SWEEP_SEED)
    cells = []
    for ci in range(len(SWEEP_COMBOS)):
        for dt in SWEEP_DTS:
            for dx in SWEEP_DXS:
                n = SWEEP_NPER
                b = rng.integers(0, T0.shape[0], n)
                T, q, pf, ph = T0[b].copy(), q0[b].copy(), pf0[b].copy(), ph0[b].copy()
                ps = ph[:, -1:]
                bl = np.clip((pf - (ps - 2.0e4)) / 2.0e4, 0.0, 1.0)  # 1 at the surface, 0 200 hPa up
                up = (pf < 5.0e4).astype(float)
                T += rng.uniform(-3.0, 4.0, (n, 1)) * bl + rng.uniform(-3.0, 3.0, (n, 1)) * up
                q *= (1.0 + rng.uniform(-0.6, 0.25, (n, 1)) * bl)
                q *= np.where(up > 0, rng.uniform(0.02, 1.5, (n, 1)), 1.0)
                q = np.maximum(q, 1.0e-7)
                ush = rng.uniform(0.0, 40.0, (n, 1))
                ang = rng.uniform(0, 2 * np.pi, (n, 1))
                frac = 1.0 - pf / ps
                u = np.where(rng.uniform(size=(n, 1)) < 0.3, u0[b], ush * frac * np.cos(ang))
                v = np.where(rng.uniform(size=(n, 1)) < 0.3, v0[b], ush * frac * np.sin(ang))
                w = rng.choice([-0.3, 0.0, 0.02, 0.1, 0.5, 1.5, 4.0], (n, 1)) * np.ones_like(T)
                cells.append(dict(combo=ci, dt=dt, dx=dx, T=T, q=q, pf=pf, ph=ph, u=u, v=v, w=w))
    return cells


def run_one(build, K, dt, dx, flags, T, q, pf, dz, rhoe, u, v, w):
    """One column (top-down inputs) through the oracle in a scratch cwd.
    Returns (parsed column, out_of_bounds, crashed)."""
    import os
    import tempfile
    nlev = T.shape[0]
    lines = [f"1 {nlev}", " ".join(repr(float(K[x])) for x in CONST_ORDER)
             + f" {dt!r} {dx!r} {CUDT!r} " + " ".join(str(int(x)) for x in flags)]
    for kt in range(nlev - 1, -1, -1):
        lines.append(" ".join(repr(float(x)) for x in (
            T[kt], q[kt], pf[kt], dz[kt], rhoe[kt], u[kt], v[kt], w[kt])))
    d = tempfile.mkdtemp()
    res = subprocess.run([str(Path(build) / "kf_driver.x")], input="\n".join(lines) + "\n",
                         capture_output=True, text=True, cwd=d)
    oob = os.path.exists(os.path.join(d, "fort.98"))
    if res.returncode != 0:
        return None, oob, True
    return parse(res.stdout, 1, nlev)[0], oob, False


def geometry(T, q, pf, ph):
    import jax
    jax.config.update("jax_enable_x64", True)
    import jax.numpy as jnp
    from legoesm.atmosphere.physics.convection.mass_flux import compute_column_geometry
    dz, _, _ = compute_column_geometry(jnp.asarray(T), jnp.asarray(pf), jnp.asarray(ph),
                                       q_v=jnp.asarray(q))
    return np.asarray(dz)


def parse(text, ncol, nlev):
    cols, cur = [], None
    for line in text.splitlines():
        tok = line.split()
        if not tok:
            continue
        if tok[0] == "COL":
            cur = {}
            cols.append(cur)
        elif tok[0] == "S":
            cur[tok[1]] = float(tok[2])
        elif tok[0] == "A":
            cur[tok[1]] = np.array([float(x) for x in tok[2:]])
    assert len(cols) == ncol, (len(cols), ncol)
    return cols


def main(col_path, build, out_path):
    c = np.load(col_path)
    T, q, pf, ph = (np.asarray(c[k], np.float64) for k in ("T", "q", "p_full", "p_half"))
    ncol, nlev = T.shape
    dz = geometry(T, q, pf, ph)
    dp = ph[:, 1:] - ph[:, :-1]
    bu = lambda a: a[:, ::-1].copy()  # noqa: E731  top-down -> bottom-up
    fix = dict(names=np.array([str(n) for n in c["names"]]), T=T, q=q, p_full=pf, p_half=ph,
               u_nat=np.asarray(c["u"], np.float64), v_nat=np.asarray(c["v"], np.float64),
               dz_bu=bu(dz), const_order=np.array(CONST_ORDER),
               consts_L=np.array([LEGO[k] for k in CONST_ORDER]),
               consts_W=np.array([WRF[k] for k in CONST_ORDER]), dx=np.float64(DX),
               cudt=np.float64(CUDT))
    for cs, K in (("L", LEGO), ("W", WRF)):
        fix[f"rhoe_{cs}_bu"] = bu(dp / (K["G"] * dz))
    fix["arm_names"] = np.array(list(ARMS))
    for arm, (cs, winds, w, dt, trig, warm, fqi, fqs) in ARMS.items():
        K = LEGO if cs == "L" else WRF
        u = np.full_like(T, 20.0) if winds == "u20" else fix["u_nat"]
        v = np.zeros_like(T) if winds == "u20" else fix["v_nat"]
        lines = [f"{ncol} {nlev}",
                 " ".join(repr(float(K[x])) for x in CONST_ORDER)
                 + f" {dt!r} {DX!r} {CUDT!r} {trig} {warm} {fqi} {fqs}"]
        rhoe = fix[f"rhoe_{cs}_bu"]
        for i in range(ncol):
            for kb in range(nlev):
                kt = nlev - 1 - kb
                lines.append(" ".join(repr(float(x)) for x in (
                    T[i, kt], q[i, kt], pf[i, kt], dz[i, kt], rhoe[i, kb], u[i, kt], v[i, kt], w)))
        res = subprocess.run([str(Path(build) / "kf_driver.x")], input="\n".join(lines) + "\n",
                             capture_output=True, text=True, check=True)
        cols = parse(res.stdout, ncol, nlev)
        fix[f"{arm}_meta"] = np.array([w, dt, trig, warm, fqi, fqs, 1.0 if winds == "u20" else 0.0])
        fix[f"{arm}_consts"] = np.array([cs])
        for k in OUT_ARRAYS:
            fix[f"{arm}_{k}"] = np.stack([cc[k] for cc in cols])
        for k in OUT_SCALARS:
            fix[f"{arm}_{k}"] = np.array([cc[k] for cc in cols])
        if arm in DIAG_ARMS:
            for k in DIAG_SCALARS:
                fix[f"{arm}_{k}"] = np.array([cc.get(k, np.nan) for cc in cols])
            for k in DIAG_ARRAYS:
                fix[f"{arm}_{k}"] = np.stack([cc.get(k, np.full(nlev, np.nan)) for cc in cols])
        print(arm, "fired", int(sum(cc["fired"] for cc in cols)), "shallow",
              int(sum(cc["ishall"] == 1 for cc in cols)), "fatal",
              int(sum(cc["nfatal"] for cc in cols)), "stderr_lines", len(res.stderr.splitlines()))
    # randomised-sweep columns (see SWEEP_PICK)
    cells = make_sweep(T, q, pf, ph, fix["u_nat"], fix["v_nat"])
    sw = {k: [] for k in ("cell", "col", "combo", "dt", "dx", "T", "q", "p", "dz", "rhoe", "u", "v",
                          "w") + OUT_ARRAYS + ("fired", "ishall", "pratec", "nca")}
    for cell, col in SWEEP_PICK:
        c_ = cells[cell]
        cs, trig, warm, fqi, fqs = SWEEP_COMBOS[c_["combo"]]
        K = LEGO if cs == "L" else WRF
        T1, q1, pf1, ph1 = (c_[k][col:col + 1] for k in ("T", "q", "pf", "ph"))
        dz1 = geometry(T1, q1, pf1, ph1)
        rh1 = (ph1[:, 1:] - ph1[:, :-1]) / (K["G"] * dz1)
        cc, oob, crash = run_one(build, K, c_["dt"], c_["dx"], (trig, warm, fqi, fqs), T1[0], q1[0],
                                 pf1[0], dz1[0], rh1[0], c_["u"][col], c_["v"][col], c_["w"][col])
        assert not (oob or crash) and cc["nfatal"] == 0, (cell, col, oob, crash)
        for k, val in (("cell", cell), ("col", col), ("combo", c_["combo"]), ("dt", c_["dt"]),
                       ("dx", c_["dx"]), ("T", T1[0]), ("q", q1[0]), ("p", pf1[0]), ("dz", dz1[0]),
                       ("rhoe", rh1[0]), ("u", c_["u"][col]), ("v", c_["v"][col]),
                       ("w", c_["w"][col])):
            sw[k].append(val[::-1] if np.ndim(val) == 1 else val)  # profiles bottom-up
        for k in OUT_ARRAYS + ("fired", "ishall", "pratec", "nca"):
            sw[k].append(cc[k])
    for k, val in sw.items():
        fix[f"sweep_{k}"] = np.array(val)
    fix["sweep_combos"] = np.array([list(map(str, c_)) for c_ in SWEEP_COMBOS])
    print("sweep picks", len(SWEEP_PICK), "fired", int(sum(sw["fired"])),
          "shallow", int(sum(x == 1 for x in sw["ishall"])))
    # lookup tables of the oracle build vs the port's numpy KF_LUTAB
    from legoesm.atmosphere.physics.convection._kf_eta import kf_lookup_tables
    for cs, K in (("L", LEGO), ("W", WRF)):
        res = subprocess.run([str(Path(build) / "lut_dump.x")],
                             input=f"{K['SVP1']!r} {K['SVP2']!r} {K['SVP3']!r} {K['SVPT0']!r}\n",
                             capture_output=True, text=True, check=True)
        vals = np.array([float(x) for x in res.stdout.split()])
        tq = vals[:2 * 250 * 220].reshape(220, 250, 2)
        th = vals[2 * 250 * 220:2 * 250 * 220 + 220]
        al = vals[2 * 250 * 220 + 220:]
        tt, qs, th0, alu = kf_lookup_tables(K["SVP1"], K["SVP2"], K["SVP3"], K["SVPT0"])[:4]
        rel = lambda a, b: float(np.max(np.abs(a - b)) / np.max(np.abs(b)))  # noqa: E731
        print(f"LUT {cs}: ttab {rel(tt, tq[:, :, 0].T):.3e} qstab {rel(qs, tq[:, :, 1].T):.3e} "
              f"the0k {rel(th0, th):.3e} alu {rel(alu, al):.3e}")
    np.savez_compressed(out_path, **fix)
    print("wrote", out_path, Path(out_path).stat().st_size, "bytes")


if __name__ == "__main__":
    main(*sys.argv[1:4])
