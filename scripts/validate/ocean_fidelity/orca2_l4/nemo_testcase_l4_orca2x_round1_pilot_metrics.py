#!/usr/bin/env python3
"""ORCA2X round-1 pilot metric card (DECISION 160): candidate vs NEMO, days 1-30.

Reads the admitted NEMO record's daily restarts (instantaneous Kmm states) and
the driver's daily candidate snapshots, and evaluates the decided card with
NEMO's interior/fold-duplicate weights (``reduction_weights``).  It assigns no
bar and no verdict.  Bias = candidate - NEMO.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[4]
for package in reversed((REPO_ROOT, REPO_ROOT / "packages/core", REPO_ROOT / "packages/ocean")):
    if str(package) not in sys.path:
        sys.path.insert(0, str(package))

from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2_round1_ladder_gate as ladder,
)
from scripts.validate.ocean_fidelity.orca2_l4 import (  # noqa: E402
    nemo_testcase_l4_orca2x_round1_longrun as longrun,
)

STEPS_PER_DAY = longrun.STEPS_PER_DAY
NLEV = 30
# zdfmxl.f90:44 ``rho_c = 0.01_wp`` (density criterion for the mixed layer).
ZDFMXL_RHO_C = 0.01
BANDS_M = ((0.0, 100.0), (100.0, 1000.0), (1000.0, np.inf))
BAND_NAMES = ("0-100m", "100-1000m", ">1000m")
FIELDS = ("T", "S", "u", "v", "ssh")


def require(ok: bool, message: str) -> None:
    if not ok:
        raise RuntimeError(message)


# --------------------------------------------------------------------- readers
def read_nemo_day(root: Path, day: int) -> dict[str, np.ndarray]:
    """Two-rank daily restart (``nn_stock = 8``) on the candidate layout."""
    from netCDF4 import Dataset

    kt = STEPS_PER_DAY * day
    slabs = []
    for rank in (0, 1):
        path = root / f"ORCA2_{kt:08d}_restart_{rank:04d}.nc"
        require(path.is_file(), f"missing NEMO restart {path}")
        with Dataset(path) as ds:
            require(int(ds.variables["kt"][...]) == kt, f"{path.name}: kt mismatch")
            slab = {}
            for name, var in (("T", "tn"), ("S", "sn"), ("u", "un"), ("v", "vn")):
                values = np.asarray(ds.variables[var][0], dtype=np.float64)
                slab[name] = np.moveaxis(values, 0, -1)[..., :NLEV]
            slab["ssh"] = np.asarray(ds.variables["sshn"][0], dtype=np.float64)
        slabs.append(slab)
    out = {k: np.concatenate([slabs[0][k], slabs[1][k]], axis=1) for k in FIELDS}
    for name, values in out.items():
        require(bool(np.isfinite(values).all()), f"day {day}: non-finite NEMO {name}")
    return out


def read_candidate_day(root: Path, day: int) -> dict[str, np.ndarray]:
    path = root / f"candidate_step{STEPS_PER_DAY * day:08d}.npz"
    require(path.is_file(), f"missing candidate snapshot {path}")
    with np.load(path) as data:
        out = {k: np.asarray(data[k], dtype=np.float64) for k in FIELDS}
    for name, values in out.items():
        require(bool(np.isfinite(values).all()), f"day {day}: non-finite candidate {name}")
    return out


def restart_reader_control(root: Path, days: range) -> list[dict]:
    """Restart(kt=8d) must equal the step-entry frame of kt=8d+1 bit for bit."""
    rows = []
    for day in days:
        kt = STEPS_PER_DAY * day + 1
        if not (root / f"oracle_step_entry_kt{kt:08d}.bin").is_file():
            continue
        entry = ladder.assemble_state_fields(root, kt, stage=None)
        nemo = read_nemo_day(root, day)
        row = {"day": day, "entry_kt": kt}
        for name in FIELDS:
            require(nemo[name].shape == entry[name].shape,
                    f"{name}: restart {nemo[name].shape} vs entry {entry[name].shape}")
            # Bitwise after mapping -0.0 to +0.0 (dry cells differ only in sign of zero).
            row[name] = bool(np.array_equal((nemo[name] + 0.0).view(np.uint64),
                                            (entry[name] + 0.0).view(np.uint64)))
        rows.append(row)
    require(rows, "no restart/entry overlap available for the reader control")
    return rows


# ------------------------------------------------------------------- geometry
def static_geometry(record: Path) -> dict[str, np.ndarray]:
    from netCDF4 import Dataset

    geo = longrun.reduction_weights(record)
    with Dataset(record / "ORCA_R2_zps_domcfg.nc") as ds:
        geo["e1t"] = np.asarray(ds.variables["e1t"][:], dtype=np.float64)
        geo["e2t"] = np.asarray(ds.variables["e2t"][:], dtype=np.float64)
        geo["e1u"] = np.asarray(ds.variables["e1u"][:], dtype=np.float64)
        geo["e1v"] = np.asarray(ds.variables["e1v"][:], dtype=np.float64)
        geo["e2v"] = np.asarray(ds.variables["e2v"][:], dtype=np.float64)
        geo["e1f"] = np.asarray(ds.variables["e1f"][:], dtype=np.float64)
        geo["e2f"] = np.asarray(ds.variables["e2f"][:], dtype=np.float64)
        geo["e3t_0"] = np.moveaxis(np.asarray(ds.variables["e3t_0"][:NLEV], np.float64), 0, -1)
        geo["mbkt"] = np.maximum(np.asarray(ds.variables["bottom_level"][:], np.int64), 1)
        geo["glamt"] = np.asarray(ds.variables["glamt"][:], dtype=np.float64)
        geo["gphit"] = np.asarray(ds.variables["gphit"][:], dtype=np.float64)
    with Dataset(record / "mesh_mask_0000.nc") as ds:
        geo["gdepw_1d"] = np.asarray(ds.variables["gdepw_1d"][0], dtype=np.float64)
        geo["e3w_1d"] = np.asarray(ds.variables["e3w_1d"][0], dtype=np.float64)
    tmask, umask, vmask = geo["tmask"], geo["umask"], geo["vmask"]
    geo["ssmask"] = tmask.max(axis=2)
    # domain.f90:196-199,211-213: ht_0/hu_0/hv_0 and their masked reciprocals.
    geo["ht_0"] = (geo["e3t_0"] * tmask).sum(axis=2)
    geo["hu_0"] = (geo["e3u_0"] * umask).sum(axis=2)
    geo["hv_0"] = (geo["e3v_0"] * vmask).sum(axis=2)
    ssu, ssv = umask.max(axis=2), vmask.max(axis=2)
    geo["r1_ht_0"] = geo["ssmask"] / (geo["ht_0"] + 1.0 - geo["ssmask"])
    geo["r1_hu_0"] = ssu / (geo["hu_0"] + 1.0 - ssu)
    geo["r1_hv_0"] = ssv / (geo["hv_0"] + 1.0 - ssv)
    # domzgr.f90:384-385: nlb10 = shallowest W level below ~10 m (1-based).
    zrefdep = 10.0 - 0.1 * geo["e3w_1d"].min()
    geo["nlb10"] = int(np.argmax(geo["gdepw_1d"] > zrefdep)) + 1
    # Streamfunction F-point support: an F point with a wet U face on either side.
    us = umask[..., 0]
    geo["psi_weight"] = geo["e1f"] * geo["e2f"] * longrun.dom_uniq_mask("F") * np.maximum(
        us, np.vstack([us[1:], np.zeros((1, us.shape[1]))]))
    return geo


def live_scales(geo: dict, ssh: np.ndarray) -> dict[str, np.ndarray]:
    """domqco.f90:207-217: r3t, r3u, r3v from one model's own ssh."""
    e1e2t = geo["e1t"] * geo["e2t"]
    r3t = ssh * geo["r1_ht_0"]
    r3u = 0.5 * (e1e2t * ssh + np.roll(e1e2t * ssh, -1, axis=1)) * geo["r1_hu_0"] / (
        geo["e1u"] * geo["e2u"])
    north = np.vstack([(e1e2t * ssh)[1:], np.zeros((1, ssh.shape[1]))])
    r3v = 0.5 * (e1e2t * ssh + north) * geo["r1_hv_0"] / (geo["e1v"] * geo["e2v"])
    return {"r3t": r3t, "r3u": r3u, "r3v": r3v}


# ---------------------------------------------------------------- reductions
def bias_rms(diff: np.ndarray, weight: np.ndarray) -> dict[str, float]:
    total = float(weight.sum())
    require(total > 0.0, "empty reduction support")
    return {"bias": float((weight * diff).sum() / total),
            "rms": float(np.sqrt((weight * diff * diff).sum() / total))}


def pattern_correlation(a: np.ndarray, b: np.ndarray, weight: np.ndarray) -> float:
    w = weight / weight.sum()
    da, db = a - (w * a).sum(), b - (w * b).sum()
    return float((w * da * db).sum() / np.sqrt((w * da * da).sum() * (w * db * db).sum()))


def band_fractions(geo: dict, r3t: np.ndarray) -> list[np.ndarray]:
    """Fraction of each live partial cell inside each depth band (exact split)."""
    e3t = geo["e3t_0"] * (1.0 + r3t[..., None] * geo["tmask"])
    bottom = np.cumsum(e3t * geo["tmask"], axis=2)
    top = bottom - e3t * geo["tmask"]
    out = []
    for lo, hi in BANDS_M:
        overlap = np.clip(np.minimum(bottom, hi) - np.maximum(top, lo), 0.0, None)
        out.append(np.where(e3t > 0, overlap / e3t, 0.0) * geo["tmask"])
    return out


def surface_ke(fields: dict, geo: dict) -> np.ndarray:
    """T-point surface KE from the two adjacent wet faces, face-area weighted."""
    au = geo["e1u"] * geo["e2u"] * geo["umask"][..., 0]
    av = geo["e1v"] * geo["e2v"] * geo["vmask"][..., 0]
    u2, v2 = fields["u"][..., 0] ** 2, fields["v"][..., 0] ** 2
    au_w, u2_w = np.roll(au, 1, axis=1), np.roll(u2, 1, axis=1)
    av_s = np.vstack([np.zeros((1, av.shape[1])), av[:-1]])
    v2_s = np.vstack([np.zeros((1, v2.shape[1])), v2[:-1]])
    su, sv = au + au_w, av + av_s
    ku = np.where(su > 0, (au * u2 + au_w * u2_w) / np.where(su > 0, su, 1.0), 0.0)
    kv = np.where(sv > 0, (av * v2 + av_s * v2_s) / np.where(sv > 0, sv, 1.0), 0.0)
    return 0.5 * (ku + kv)


def u_transport_columns(fields: dict, geo: dict, r3u: np.ndarray) -> np.ndarray:
    e3u = geo["e3u_0"] * (1.0 + r3u[..., None] * geo["umask"])
    return (fields["u"] * e3u * geo["umask"]).sum(axis=2) * geo["e2u"]


def barotropic_streamfunction_sv(fields: dict, geo: dict, r3u: np.ndarray) -> np.ndarray:
    """psi at F(j,i) = -sum_{j'<=j} U-transport(j',i), zero at Antarctica [Sv]."""
    return -np.cumsum(u_transport_columns(fields, geo, r3u), axis=0) / 1e6


def drake_transport_sv(fields: dict, geo: dict, r3u: np.ndarray) -> float:
    cols = u_transport_columns(fields, geo, r3u)
    return float(cols[longrun.DRAKE_J_SLICE, longrun.DRAKE_I].sum() / 1e6)


def hmlp_nemo(rn2b: np.ndarray, geo: dict, r3t: np.ndarray, g: float, rho0: float) -> np.ndarray:
    """zdfmxl.f90:109-124 transcribed; ``rn2b[..., m]`` is NEMO W level jk = m+2."""
    zn2_c = g * ZDFMXL_RHO_C / rho0
    stretch = 1.0 + r3t
    nmln = np.full(r3t.shape, geo["nlb10"], dtype=np.int64)
    acc = np.zeros(r3t.shape)
    jpkm1 = geo["gdepw_1d"].size - 1
    for jk in range(geo["nlb10"], jpkm1 + 1):
        acc = acc + np.maximum(rn2b[..., jk - 2], 0.0) * (geo["e3w_1d"][jk - 1] * stretch)
        nmln = np.where(acc < zn2_c, np.minimum(jk, geo["mbkt"]) + 1, nmln)
    return geo["gdepw_1d"][nmln - 1] * stretch * geo["ssmask"]


class MLDEvaluator:
    """Card's own NEMO bn2 (rn2b at step entry) on an arbitrary T/S/ssh state."""

    def __init__(self, record: Path):
        import jax

        card, model, state = longrun.build_comparison(record)
        self.model, self.state = model, state
        cfg = card.recipe.model_config
        self.g = float(cfg.constants.g)
        self.rho0 = float(cfg.constants.rho_0)
        self._bundle = jax.jit(
            lambda s: model._tke_step_entry_n2_bundle(s).rn2b)

    def rn2b(self, fields: dict) -> np.ndarray:
        import jax.numpy as jnp

        st = self.state
        T = np.array(st.T.data)
        S = np.array(st.S.data)
        T[..., :NLEV], S[..., :NLEV] = fields["T"], fields["S"]
        st = st._replace(T=st.T.replace(data=jnp.asarray(T)),
                         S=st.S.replace(data=jnp.asarray(S)),
                         eta=st.eta.replace(data=jnp.asarray(fields["ssh"])))
        return np.asarray(self._bundle(st), dtype=np.float64)


# --------------------------------------------------------------------- driver
def day_metrics(cand: dict, nemo: dict, geo: dict, mld: MLDEvaluator | None,
                ice: np.ndarray) -> tuple[dict, dict]:
    live_n, live_c = live_scales(geo, nemo["ssh"]), live_scales(geo, cand["ssh"])
    area = geo["area_t"]
    m = {}
    for key, name, k in (("SST", "T", 0), ("SSS", "S", 0)):
        m[key] = bias_rms(cand[name][..., k] - nemo[name][..., k], area * geo["tmask"][..., 0])
    m["SSH"] = bias_rms(cand["ssh"] - nemo["ssh"], area)
    vol = area[..., None] * geo["e3t_0"] * (1.0 + live_n["r3t"][..., None] * geo["tmask"]) * geo["tmask"]
    fracs = band_fractions(geo, live_n["r3t"])
    for var in ("T", "S"):
        d = cand[var] - nemo[var]
        for bname, frac in zip(BAND_NAMES, fracs):
            m[f"{var}_{bname}"] = bias_rms(d, vol * frac)
        m[f"{var}_profile"] = [bias_rms(d[..., k], area * geo["tmask"][..., k])
                               if (area * geo["tmask"][..., k]).sum() > 0 else None
                               for k in range(NLEV)]
    ke_c, ke_n = surface_ke(cand, geo), surface_ke(nemo, geo)
    m["KE"] = {**bias_rms(ke_c - ke_n, area),
               "nemo_mean": float((area * ke_n).sum() / area.sum()),
               "cand_mean": float((area * ke_c).sum() / area.sum())}
    psi_c = barotropic_streamfunction_sv(cand, geo, live_c["r3u"])
    psi_n = barotropic_streamfunction_sv(nemo, geo, live_n["r3u"])
    w = geo["psi_weight"]
    m["PSI"] = {**bias_rms(psi_c - psi_n, w), "pattern_corr": pattern_correlation(psi_c, psi_n, w)}
    m["DRAKE"] = {"nemo_sv": drake_transport_sv(nemo, geo, live_n["r3u"]),
                  "cand_sv": drake_transport_sv(cand, geo, live_c["r3u"])}
    maps = {"SST": cand["T"][..., 0] - nemo["T"][..., 0],
            "SSS": cand["S"][..., 0] - nemo["S"][..., 0],
            "SSH": cand["ssh"] - nemo["ssh"], "PSI": psi_c - psi_n,
            "PSI_nemo": psi_n, "KE": ke_c - ke_n}
    if mld is not None:
        h_c = hmlp_nemo(mld.rn2b(cand), geo, live_c["r3t"], mld.g, mld.rho0)
        h_n = hmlp_nemo(mld.rn2b(nemo), geo, live_n["r3t"], mld.g, mld.rho0)
        m["MLD"] = {**bias_rms(h_c - h_n, area),
                    "nemo_mean": float((area * h_n).sum() / area.sum()),
                    "cand_mean": float((area * h_c).sum() / area.sum())}
        maps["MLD"], maps["MLD_nemo"] = h_c - h_n, h_n
    wi = area * (ice > 0.0) * geo["tmask"][..., 0]
    m["ICE_SST"] = bias_rms(cand["T"][..., 0] - nemo["T"][..., 0], wi)
    m["ICE_SSS"] = bias_rms(cand["S"][..., 0] - nemo["S"][..., 0], wi)
    m["ICE_area_km2"] = float(wi.sum() / 1e6)
    return m, maps


def planted_controls(cand: dict, nemo: dict, geo: dict, ice: np.ndarray) -> dict:
    """Known responses: +0.01 K on candidate SST, candidate u scaled by 1.01."""
    base, _ = day_metrics(cand, nemo, geo, None, ice)
    warm = dict(cand, T=cand["T"].copy())
    warm["T"][..., 0] += 0.01
    fast = dict(cand, u=cand["u"] * 1.01)
    shift = day_metrics(warm, nemo, geo, None, ice)[0]["SST"]["bias"] - base["SST"]["bias"]
    ratio = day_metrics(fast, nemo, geo, None, ice)[0]["DRAKE"]["cand_sv"] / base["DRAKE"]["cand_sv"]
    require(abs(shift - 0.01) < 1e-9, f"SST plant moved bias by {shift}")
    require(abs(ratio - 1.01) < 1e-9, f"Drake plant ratio {ratio}")
    return {"sst_plus_0.01K_bias_shift": shift, "u_times_1.01_drake_ratio": ratio}


SCALARS = ("SST", "SSS", "SSH", "T_0-100m", "T_100-1000m", "T_>1000m", "S_0-100m",
           "S_100-1000m", "S_>1000m", "KE", "PSI", "MLD", "ICE_SST", "ICE_SSS")


def monthly(rows: list[dict], maps: list[dict], geo: dict) -> dict:
    out = {}
    for key in SCALARS:
        if key not in rows[0]:
            continue
        out[key] = {s: float(np.mean([r[key][s] for r in rows])) for s in ("bias", "rms")}
        if key == "PSI":
            out[key]["pattern_corr"] = float(np.mean([r[key]["pattern_corr"] for r in rows]))
    for key, weight in (("SST", geo["area_t"]), ("SSS", geo["area_t"]),
                        ("SSH", geo["area_t"]), ("MLD", geo["area_t"]),
                        ("PSI", geo["psi_weight"])):
        if key in maps[0]:
            mean_map = np.mean([mp[key] for mp in maps], axis=0)
            out[f"{key}_monthly_mean_map"] = bias_rms(mean_map, weight)
    drake = np.array([[r["DRAKE"]["nemo_sv"], r["DRAKE"]["cand_sv"]] for r in rows])
    out["DRAKE"] = {"nemo_mean_sv": float(drake[:, 0].mean()),
                    "cand_mean_sv": float(drake[:, 1].mean()),
                    "nemo_std_sv": float(drake[:, 0].std()),
                    "cand_std_sv": float(drake[:, 1].std())}
    return out


def plots(rows: list[dict], maps: list[dict], geo: dict, out: Path) -> list[str]:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = [r["day"] for r in rows]
    keys = [k for k in SCALARS if k in rows[0]]
    fig, axes = plt.subplots(len(keys), 2, figsize=(10, 2.0 * len(keys)), sharex=True)
    for ax, key in zip(axes, keys):
        for col, stat in enumerate(("bias", "rms")):
            ax[col].plot(days, [r[key][stat] for r in rows], marker=".")
            ax[col].set_ylabel(f"{key} {stat}", fontsize=7)
            ax[col].axhline(0.0, color="0.7", lw=0.5)
    axes[-1][0].set_xlabel("day")
    axes[-1][1].set_xlabel("day")
    fig.tight_layout()
    names = ["timeseries_bias_rms.png"]
    fig.savefig(out / names[-1], dpi=110)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(6, 3))
    ax.plot(days, [r["DRAKE"]["nemo_sv"] for r in rows], label="NEMO")
    ax.plot(days, [r["DRAKE"]["cand_sv"] for r in rows], label="candidate")
    ax.set_xlabel("day"), ax.set_ylabel("Drake transport [Sv]"), ax.legend()
    fig.tight_layout()
    names.append("drake_transport.png")
    fig.savefig(out / names[-1], dpi=110)
    plt.close(fig)

    depth = np.cumsum(geo["e3w_1d"][:NLEV]) - 0.5 * geo["e3w_1d"][:NLEV]
    fig, axes = plt.subplots(1, 4, figsize=(12, 5), sharey=True)
    for col, (var, stat) in enumerate((("T", "bias"), ("T", "rms"), ("S", "bias"), ("S", "rms"))):
        for day in sorted({1, 10, 20, days[-1]} & set(days)):
            prof = rows[days.index(day)][f"{var}_profile"]
            axes[col].plot([p[stat] if p else np.nan for p in prof], depth, label=f"day {day}")
        axes[col].set_title(f"{var} {stat}")
        axes[col].set_yscale("symlog", linthresh=100)
    axes[0].invert_yaxis(), axes[0].set_ylabel("depth [m]"), axes[0].legend(fontsize=7)
    fig.tight_layout()
    names.append("profiles_T_S.png")
    fig.savefig(out / names[-1], dpi=110)
    plt.close(fig)

    keys = [k for k in ("SST", "SSS", "SSH", "MLD", "PSI", "KE") if k in maps[0]]
    fig, axes = plt.subplots(len(keys), 1, figsize=(8, 3.2 * len(keys)))
    wet = geo["tmask"][..., 0] > 0
    for ax, key in zip(axes, keys):
        mean_map = np.mean([mp[key] for mp in maps], axis=0)
        shown = np.where(wet, mean_map, np.nan)
        lim = np.nanpercentile(np.abs(shown), 99) or 1.0
        im = ax.pcolormesh(shown, cmap="RdBu_r", vmin=-lim, vmax=lim)
        fig.colorbar(im, ax=ax)
        ax.set_title(f"30-day mean {key} (candidate - NEMO), model grid")
    fig.tight_layout()
    names.append("monthly_mean_difference_maps.png")
    fig.savefig(out / names[-1], dpi=90)
    plt.close(fig)
    return names


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--record-root", type=Path, required=True)
    parser.add_argument("--candidate-root", type=Path, required=True)
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--days", type=int, default=30)
    parser.add_argument("--skip-mld", action="store_true")
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    reader = restart_reader_control(args.record_root, range(1, args.days + 1))
    require(all(all(r[f] for f in FIELDS) for r in reader),
            f"restart reader control failed: {reader}")
    geo = static_geometry(args.record_root)
    mld = None if args.skip_mld else MLDEvaluator(args.record_root)
    rows, maps = [], []
    planted = planted_controls(
        read_candidate_day(args.candidate_root, 1), read_nemo_day(args.record_root, 1), geo,
        ladder.assemble_surface_fields(args.record_root, STEPS_PER_DAY)["fr_i"])
    for day in range(1, args.days + 1):
        cand = read_candidate_day(args.candidate_root, day)
        nemo = read_nemo_day(args.record_root, day)
        ice = ladder.assemble_surface_fields(args.record_root, STEPS_PER_DAY * day)["fr_i"]
        row, mp = day_metrics(cand, nemo, geo, mld, ice)
        rows.append({"day": day, **row})
        maps.append(mp)
        print(f"day {day}: SST rms {row['SST']['rms']:.4g} SSH rms {row['SSH']['rms']:.4g}",
              flush=True)
    result = {
        "format": "orca2x-r1-pilot-metrics-v1",
        "bias_convention": "candidate - NEMO",
        "restart_reader_control": reader,
        "planted_controls": planted,
        "nlb10": geo["nlb10"],
        "mld_constants": None if mld is None else {"g": mld.g, "rho0": mld.rho0,
                                                    "rho_c": ZDFMXL_RHO_C},
        "daily": rows,
        "monthly": monthly(rows, maps, geo),
    }
    result["plots"] = plots(rows, maps, geo, args.output_root)
    (args.output_root / "pilot_metrics.json").write_text(json.dumps(result, indent=1) + "\n")
    np.savez_compressed(args.output_root / "monthly_mean_difference_maps.npz",
                        **{k: np.mean([mp[k] for mp in maps], axis=0) for k in maps[0]})
    print(json.dumps(result["monthly"], indent=1))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
