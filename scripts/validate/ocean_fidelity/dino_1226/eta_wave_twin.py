"""Sea-surface-height WAVE-FIELD twin: legoESM vs NEMO, sampled every model step.

WHY THIS PROBE EXISTS
---------------------
The DINO twin has been certified step-by-step (single-step tendency matches at
roundoff) and compared at the CLIMATE end (90-day means).  Nothing in between
has ever been looked at.  The saved twin output is DAILY, while a barotropic
gravity wave crosses the DINO basin in roughly eight hours -- so every wave in
the free-surface field has been ALIASED away by the sampling, and a mismatch
in wave speed, wave phase, or wall reflection would be invisible.

WHAT IT COMPARES
----------------
Both models are started from the SAME NEMO day-180 restart
(DINO_00005760_restart.nc) and integrated 160 steps (5 days at rn_Dt=2700 s).
Sea surface height is sampled EVERY step on both sides -- 45-minute sampling,
finer than the "hourly" the task asked for, because the NEMO time step is
2700 s and an hourly cadence is not an integer number of steps.

TIME CENTERING (the trap this probe had to get right)
-----------------------------------------------------
NEMO writes its restart in stpmlf.F90 at

    stpmlf.F90:621-624   Nrhs = Nbb ; Nbb = Nnn ; Nnn = Naa ; Naa = Nrhs   ! swap
    stpmlf.F90:634       IF( lrst_oce ) CALL rst_write( kstp, Nbb, Nnn )

i.e. the swap happens BEFORE the write, and rst_write is called WITHOUT a Kaa
argument, so the MLF branch of restart.F90 runs:

    restart.F90:191      iom_rstput( ... 'sshb', ssh(:,:,Kbb) )   ! before
    restart.F90:197      iom_rstput( ... 'sshn', ssh(:,:,Kmm) )   ! now

With the POST-swap indices, Kmm == the pre-swap Naa.  Therefore

    sshn in DINO_<kt>_restart.nc  ==  sea surface height AFTER kt steps,

and that -- not sshb -- is the field that must be lined up with legoESM's eta
after the same number of steps.  (The same argument makes 'un'/'vn' the
after-level 3-D velocities; 'ub'/'vb' are the Asselin-filtered previous level.)

The identity is checked mechanically at extraction time: the sshb of dump
kt+1 must equal a filtered combination of neighbouring levels, and -- the
cheap decisive one -- dump kt's sshn is NOT equal to dump kt+1's sshn.

MASKING
-------
NEMO fills land with exact 0.0 in the restart.  A comparison that treats those
as data reports a difference of zero over half the grid and dilutes every
normalized statistic.  The wet mask is taken from the model grid (not from
"where the field is zero", which is a structural-zero trap), and the probe
plants a synthetic violation in a dry cell to prove the mask is actually
applied -- a masked statistic that does not move when a dry cell is poisoned
is not masking anything.

NaN IS FATAL.  Nothing here uses nanmean/nanmax.

THIS PROBE PRINTS NUMBERS, NOT VERDICTS.  The interpretation lives in the
findings document, after the controls below have passed.
"""
from __future__ import annotations

import argparse
import glob
import hashlib
import json
import os
import subprocess
import sys
import time

import netCDF4
import numpy as np

# NEMO free-surface sampling: DINO runs rn_Dt = 2700 s, so one dump per step
# is a 45-minute sample.  Recorded here so the probe never has to guess it;
# it is CHECKED against the 'rdt' variable in every restart it reads.
NEMO_RDT_S = 2700.0


# --------------------------------------------------------------------------
# provenance
# --------------------------------------------------------------------------
def provenance(extra: dict) -> dict:
    """Stamp git SHA, wall clock, dtype and input identity onto every output."""
    here = os.path.dirname(os.path.abspath(__file__))
    try:
        sha = subprocess.check_output(
            ["git", "-C", here, "rev-parse", "HEAD"], text=True).strip()
        dirty = subprocess.check_output(
            ["git", "-C", here, "status", "--porcelain"], text=True).strip()
    except Exception as exc:                      # pragma: no cover - env dep
        sha, dirty = f"UNKNOWN({exc})", "UNKNOWN"
    prov = {
        "git_sha": sha,
        "git_dirty": bool(dirty),
        "utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "dtype": "float64",
        "argv": sys.argv,
    }
    prov.update(extra)
    return prov


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


# --------------------------------------------------------------------------
# NEMO side: stitch the per-rank restart tiles, one dump per step
# --------------------------------------------------------------------------
def _repo_root() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.abspath(os.path.join(here, "..", "..", "..", ".."))


def _load_rebuild():
    """Reuse the committed tile stitcher rather than re-deriving one."""
    sys.path.insert(0, os.path.join(_repo_root(), "scripts", "validate",
                                    "ocean_fidelity"))
    from rebuild_nemo_restart import rebuild  # noqa: E402,PLC0415
    return rebuild


def extract_nemo(run_dir: str, kt0: int, nsteps: int, out: str,
                 fields: tuple[str, ...] = ("sshn", "sshb")) -> dict:
    """Stitch `sshn` (and `sshb`) from every per-step NEMO restart dump.

    Returns the assembled dict; also writes `out` (.npz).

    kt0 is the restart step the run STARTED from (5760), so dump kt0+n holds
    the state after n steps.  Time coordinate t[n] = n * rdt seconds.
    """
    rebuild = _load_rebuild()
    eta = None
    kts = []
    rdt_seen = set()
    for n in range(1, nsteps + 1):
        kt = kt0 + n
        pattern = os.path.join(run_dir, f"DINO_{kt:08d}_restart_*.nc")
        tiles = sorted(glob.glob(pattern))
        if not tiles:
            raise SystemExit(f"missing NEMO dump for kt={kt}: {pattern}")
        got = rebuild(pattern, list(fields))
        for name in fields:
            if name not in got:
                raise SystemExit(f"{name} absent from NEMO dump kt={kt}")
        if eta is None:
            ny, nx = got["sshn"].shape
            eta = {name: np.empty((nsteps, ny, nx), dtype=np.float64)
                   for name in fields}
        for name in fields:
            a = got[name]
            if not np.isfinite(a).all():
                raise SystemExit(
                    f"NaN/Inf in stitched {name} at kt={kt} -- a tile did not "
                    f"cover the whole domain; refusing to continue")
            eta[name][n - 1] = a
        kts.append(kt)
        # rdt sanity: read once per file group, cheap and catches a wrong run
        if n in (1, nsteps):
            with netCDF4.Dataset(tiles[0]) as ds:
                rdt_seen.add(float(np.squeeze(ds.variables["rdt"][:])))

    if rdt_seen != {NEMO_RDT_S}:
        raise SystemExit(f"NEMO rdt {sorted(rdt_seen)} != expected "
                         f"{NEMO_RDT_S}; the run is not the certified twin")

    # geometry + the wet mask donor, taken from tile 0 of the first dump
    geom = rebuild(os.path.join(run_dir, f"DINO_{kt0+1:08d}_restart_*.nc"),
                   ["nav_lat", "nav_lon"])

    # ---- time-level control: consecutive dumps must NOT be identical, and
    # ---- dump n's sshb must equal something OTHER than its own sshn.
    d_consec = float(np.max(np.abs(eta["sshn"][1] - eta["sshn"][0])))
    d_bn = float(np.max(np.abs(eta["sshn"][0] - eta["sshb"][0])))
    if d_consec == 0.0 or d_bn == 0.0:
        raise SystemExit("time-level control FAILED: consecutive dumps or "
                         "sshn/sshb within one dump are bit-identical")

    payload = {
        "eta": eta["sshn"],
        "eta_before": eta["sshb"],
        "kt": np.asarray(kts, dtype=np.int64),
        "t_seconds": np.arange(1, nsteps + 1, dtype=np.float64) * NEMO_RDT_S,
        "nav_lat": geom["nav_lat"],
        "nav_lon": geom["nav_lon"],
    }
    prov = provenance({
        "source": "NEMO",
        "run_dir": run_dir,
        "binary_md5": md5(os.path.join(run_dir, "nemo.exe")),
        "kt0": kt0,
        "nsteps": nsteps,
        "rdt_s": NEMO_RDT_S,
        "time_level": ("sshn == ssh(Kmm) with POST-swap indices "
                       "(stpmlf.F90:621-624 swap, :634 rst_write) == state "
                       "AFTER kt steps"),
        "control_max_abs_consecutive_sshn_diff_m": d_consec,
        "control_max_abs_sshn_minus_sshb_m": d_bn,
    })
    np.savez_compressed(out, provenance=json.dumps(prov), **payload)
    print(json.dumps(prov, indent=2))
    print(f"[shape] eta {payload['eta'].shape}  "
          f"absmax {np.max(np.abs(payload['eta'])):.6e} m")
    print(f"[out] {out}")
    return payload


# --------------------------------------------------------------------------
# comparison
# --------------------------------------------------------------------------
MESH_MASK = ("/home/dbalwada/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/"
             "RUN_TRAJ/mesh_mask.nc")

# PRE-REGISTERED comparison targets, in HOURS.  With rn_Dt = 2700 s an hourly
# cadence is not an integer number of steps, so each target is realised at the
# NEAREST step and the probe reports the step's true time next to it.
TARGET_HOURS = (1.0, 2.0, 4.0, 8.0, 12.0, 24.0, 48.0, 120.0)

# PRE-REGISTERED verdict thresholds (fixed before either model was compared):
#   the per-step floor is measured IN THIS EXPERIMENT as the wet-cell max
#   |eta_lego - eta_nemo| after ONE step.  A scheme difference that does not
#   amplify accumulates at most linearly, so over n steps the difference is
#   bounded by n * floor.  Exceeding that is amplification, not transcription.
GROWTH_BAR_LINEAR = 1.0     # multiplier on (n_steps * one_step_floor)
SPECTRAL_BAND_BAR = 2.0     # max tolerated power ratio in any resolved band


def wet_mask(mesh_mask: str = MESH_MASK) -> np.ndarray:
    """Surface wet mask (ny, nx) from the model's OWN mesh, never from zeros.

    Inferring land from "where the field is exactly 0" is the structural-zero
    trap: a genuinely zero ocean value (and the equator, where several fields
    vanish by symmetry) would be silently deleted from every statistic.
    """
    with netCDF4.Dataset(mesh_mask) as ds:
        tm = np.asarray(ds.variables["tmask"][0]).squeeze()   # (nk, ny, nx)
    return tm[0] > 0.5


def load_side(path: str, name: str) -> dict:
    d = np.load(path, allow_pickle=False)
    eta = np.asarray(d["eta"], dtype=np.float64)
    if eta.ndim != 3:
        raise SystemExit(f"{name}: eta must be (nt, ny, nx), got {eta.shape}")
    if not np.isfinite(eta).all():
        raise SystemExit(f"{name}: eta contains NaN/Inf -- fatal, not masked")
    t = np.asarray(d["t_seconds"], dtype=np.float64)
    if t.shape[0] != eta.shape[0]:
        raise SystemExit(f"{name}: t_seconds {t.shape} vs eta {eta.shape}")
    return {"eta": eta, "t": t, "name": name,
            "keys": sorted(k for k in d.files)}


def masked_stats(diff: np.ndarray, wet: np.ndarray) -> dict:
    """max/rms of a difference field over WET cells only, plus its locus."""
    d = diff[wet]
    amax = float(np.max(np.abs(d)))
    rms = float(np.sqrt(np.mean(d ** 2)))
    flat = np.argmax(np.where(wet, np.abs(diff), -np.inf))
    j, i = np.unravel_index(flat, diff.shape)
    return {"max_abs_m": amax, "rms_m": rms, "argmax_j": int(j),
            "argmax_i": int(i)}


def locus_partition(wet: np.ndarray, lat: np.ndarray) -> dict:
    """Three disjoint wet regions the difference can live in.

    WALL   -- wet cells with a dry 4-neighbour, plus the outermost wet ring
              (the free-slip walls and the topographic staircase).
    EQUATOR-- |lat| <= 2 degrees, where f -> 0 and several operators have a
              structural zero.  Reported separately precisely so a normalised
              statistic there is never mistaken for a signal.
    INTERIOR - everything else.
    """
    dry = ~wet
    nb = np.zeros_like(wet)
    nb[1:, :] |= dry[:-1, :]
    nb[:-1, :] |= dry[1:, :]
    nb[:, 1:] |= dry[:, :-1]
    nb[:, :-1] |= dry[:, 1:]
    nb[0, :] = nb[-1, :] = True
    nb[:, 0] = nb[:, -1] = True
    wall = wet & nb
    eq = wet & (~wall) & (np.abs(lat) <= 2.0)
    interior = wet & (~wall) & (~eq)
    return {"wall": wall, "equator": eq, "interior": interior}


def locus_shares(diff: np.ndarray, regions: dict) -> dict:
    """Fraction of the SQUARED difference sitting in each region."""
    tot = sum(float(np.sum(diff[m] ** 2)) for m in regions.values())
    if tot <= 0.0:
        return {k: 0.0 for k in regions}
    return {k: float(np.sum(diff[m] ** 2)) / tot for k, m in regions.items()}


def plant_dry_violation(eta: np.ndarray, wet: np.ndarray) -> np.ndarray:
    """Poison every DRY cell with a huge value.

    Every masked statistic must be bit-identical on the poisoned copy.  A
    statistic that moves is not masking; it is averaging land.
    """
    if wet.all():
        raise SystemExit("no dry cells: the mask control cannot fire, so the "
                         "masking claim would be vacuous")
    out = eta.copy()
    out[..., ~wet] = 1.0e6
    return out


def spectra(series: np.ndarray, dt: float) -> tuple[np.ndarray, np.ndarray]:
    """One-sided amplitude spectrum of a de-meaned, Hann-windowed series."""
    x = series - series.mean()
    w = np.hanning(x.size)
    # correct for the window's amplitude loss so the two sides stay comparable
    amp = np.abs(np.fft.rfft(x * w)) * 2.0 / w.sum()
    freq = np.fft.rfftfreq(x.size, d=dt)
    return freq, amp


def compare(nemo_npz: str, lego_npz: str, outdir: str,
            mesh_mask: str = MESH_MASK) -> dict:
    os.makedirs(outdir, exist_ok=True)
    nemo = load_side(nemo_npz, "NEMO")
    lego = load_side(lego_npz, "legoESM")
    wet = wet_mask(mesh_mask)

    if nemo["eta"].shape != lego["eta"].shape:
        raise SystemExit(f"frame mismatch: NEMO {nemo['eta'].shape} vs "
                         f"legoESM {lego['eta'].shape}")
    if nemo["eta"].shape[1:] != wet.shape:
        raise SystemExit(f"mask {wet.shape} does not match frame "
                         f"{nemo['eta'].shape[1:]}")
    if not np.array_equal(nemo["t"], lego["t"]):
        raise SystemExit("the two time axes differ -- a wave comparison across "
                         "mismatched clocks is a confound, not a result")

    with netCDF4.Dataset(mesh_mask) as ds:
        lat = np.asarray(ds.variables["nav_lat"][:]).squeeze()
        lon = np.asarray(ds.variables["nav_lon"][:]).squeeze()

    t = nemo["t"]
    dt = float(t[1] - t[0])
    nt = t.size
    diff = lego["eta"] - nemo["eta"]

    # ---- CONTROL 1: the dry-cell violation must not move a single number ----
    poisoned = plant_dry_violation(lego["eta"], wet)
    a = masked_stats(diff[-1], wet)
    b = masked_stats((poisoned - nemo["eta"])[-1], wet)
    if a != b:
        raise SystemExit(f"MASK CONTROL FAILED: poisoning dry cells changed a "
                         f"masked statistic ({a} vs {b})")
    # ...and it must MOVE the unmasked one, or the control itself is vacuous
    if float(np.max(np.abs(diff[-1]))) == float(
            np.max(np.abs((poisoned - nemo["eta"])[-1]))):
        raise SystemExit("MASK CONTROL VACUOUS: the poison did not change even "
                         "the UNmasked statistic")

    regions = locus_partition(wet, lat)

    # ---- growth curve + locus ------------------------------------------
    growth = []
    for n in range(nt):
        st = masked_stats(diff[n], wet)
        st["share"] = locus_shares(diff[n], regions)
        st["t_hours"] = float(t[n] / 3600.0)
        st["step"] = n + 1
        growth.append(st)
    floor = growth[0]["max_abs_m"]
    bar = GROWTH_BAR_LINEAR * nt * floor

    # ---- pre-registered snapshot times ---------------------------------
    snaps = []
    for h in TARGET_HOURS:
        n = int(np.argmin(np.abs(t / 3600.0 - h)))
        s = dict(growth[n])
        s["target_hours"] = h
        snaps.append(s)

    # ---- spectra at four named probe points ----------------------------
    probes = named_probes(lat, lon, wet)
    spec = {}
    for pname, (j, i) in probes.items():
        fn, an = spectra(nemo["eta"][:, j, i], dt)
        fl, al = spectra(lego["eta"][:, j, i], dt)
        if not np.allclose(fn, fl):
            raise SystemExit("spectral frequency axes differ")
        # ratio only where NEMO carries real power, so a divide-by-a-
        # structural-zero can never manufacture a band mismatch
        ref = an.max()
        live = an > 1e-3 * ref
        ratio = np.where(live, al / np.maximum(an, 1e-30), np.nan)
        worst = int(np.nanargmax(np.where(live, np.maximum(
            ratio, 1.0 / np.maximum(ratio, 1e-30)), np.nan)))
        spec[pname] = {
            "j": int(j), "i": int(i),
            "lat": float(lat[j, i]), "lon": float(lon[j, i]),
            "peak_freq_nemo_cph": float(fn[1:][np.argmax(an[1:])] * 3600.0),
            "peak_freq_lego_cph": float(fl[1:][np.argmax(al[1:])] * 3600.0),
            "peak_amp_nemo_m": float(an[1:].max()),
            "peak_amp_lego_m": float(al[1:].max()),
            "worst_band_freq_cph": float(fn[worst] * 3600.0),
            "worst_band_period_h": float(1.0 / (fn[worst] * 3600.0))
            if fn[worst] > 0 else float("inf"),
            "worst_band_ratio_lego_over_nemo": float(ratio[worst]),
            "n_live_bands": int(live.sum()),
            "_freq_cph": fn * 3600.0, "_amp_nemo": an, "_amp_lego": al,
        }

    result = {
        "one_step_floor_max_abs_m": floor,
        "n_steps": nt,
        "dt_seconds": dt,
        "linear_bar_m": bar,
        "final_max_abs_m": growth[-1]["max_abs_m"],
        "growth_over_linear_bar": (growth[-1]["max_abs_m"] / bar
                                   if bar > 0 else float("inf")),
        "snapshots": snaps,
        "growth": [{k: v for k, v in g.items()} for g in growth],
        "spectra": {k: {kk: vv for kk, vv in v.items()
                        if not kk.startswith("_")}
                    for k, v in spec.items()},
        "provenance": provenance({
            "nemo_npz": nemo_npz, "lego_npz": lego_npz,
            "mesh_mask": mesh_mask, "wet_cells": int(wet.sum()),
            "dry_cells": int((~wet).sum()),
            "mask_control": "PASSED (dry poison inert on masked stats, "
                            "active on unmasked)",
            "growth_bar_linear": GROWTH_BAR_LINEAR,
            "spectral_band_bar": SPECTRAL_BAND_BAR,
        }),
    }
    with open(os.path.join(outdir, "eta_wave_twin.json"), "w") as fh:
        json.dump(result, fh, indent=2, default=float)
    make_figures(nemo, lego, diff, wet, lat, lon, t, snaps, growth, spec,
                 regions, outdir)
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("growth",)}, indent=2, default=float))
    print(f"[out] {outdir}")
    return result


def named_probes(lat, lon, wet) -> dict:
    """Four wet probe points, named for the physics they are meant to expose."""
    def pick(target_lat, target_lon):
        cost = np.where(wet, (lat - target_lat) ** 2 + (lon - target_lon) ** 2,
                        np.inf)
        j, i = np.unravel_index(np.argmin(cost), cost.shape)
        return int(j), int(i)
    return {
        "channel": pick(-55.0, 25.0),     # the ACC channel
        "equator": pick(0.0, 25.0),       # f -> 0, structural-zero territory
        "west_wall": pick(-30.0, 3.0),    # western boundary, reflection site
        "mid_basin": pick(30.0, 25.0),    # quiet interior reference
    }


def make_figures(nemo, lego, diff, wet, lat, lon, t, snaps, growth, spec,
                 regions, outdir) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    nan = np.where(wet, 0.0, np.nan)

    # (a) difference maps at the pre-registered hours
    fig, axes = plt.subplots(2, 4, figsize=(20, 9), constrained_layout=True)
    for ax, s in zip(axes.ravel(), snaps):
        n = s["step"] - 1
        m = np.abs(diff[n] + nan)
        v = np.nanmax(m) or 1e-30
        im = ax.pcolormesh(lon, lat, diff[n] + nan, cmap="RdBu_r",
                           vmin=-v, vmax=v, shading="auto")
        ax.set_title(f"t={s['t_hours']:.2f} h (target {s['target_hours']:g} h)\n"
                     f"max|d|={s['max_abs_m']:.2e} m", fontsize=9)
        plt.colorbar(im, ax=ax)
    fig.suptitle("eta difference  legoESM - NEMO  [m]  (dry cells blanked)")
    fig.savefig(os.path.join(outdir, "fig_diff_maps.png"), dpi=110)
    plt.close(fig)

    # (b) Hovmoller: time-longitude at the channel latitude, time-latitude at
    #     a mid-basin longitude, both sides and the difference
    jc = spec["channel"]["j"]
    ic = spec["mid_basin"]["i"]
    th = t / 3600.0
    fig, axes = plt.subplots(2, 3, figsize=(17, 10), constrained_layout=True)
    for row, (lab, sl, xax, xname) in enumerate([
            ("time-longitude @ channel lat "
             f"{lat[jc, 0]:.1f}", (slice(None), jc, slice(None)),
             lon[jc, :], "longitude"),
            ("time-latitude @ lon "
             f"{lon[0, ic]:.1f}", (slice(None), slice(None), ic),
             lat[:, ic], "latitude")]):
        wl = (wet[jc, :] if row == 0 else wet[:, ic])
        blank = np.where(wl, 0.0, np.nan)
        for col, (nm, fld) in enumerate([("NEMO", nemo["eta"]),
                                         ("legoESM", lego["eta"]),
                                         ("lego - NEMO", diff)]):
            a = fld[sl] + blank
            v = np.nanmax(np.abs(a)) or 1e-30
            ax = axes[row, col]
            im = ax.pcolormesh(xax, th, a, cmap="RdBu_r", vmin=-v, vmax=v,
                               shading="auto")
            ax.set_title(f"{nm}\n{lab}", fontsize=9)
            ax.set_xlabel(xname)
            ax.set_ylabel("hours")
            plt.colorbar(im, ax=ax)
    fig.suptitle("eta Hovmoller [m] -- same colour scale within a panel only")
    fig.savefig(os.path.join(outdir, "fig_hovmoller.png"), dpi=110)
    plt.close(fig)

    # (c) growth curve + locus shares
    fig, axes = plt.subplots(1, 2, figsize=(14, 5), constrained_layout=True)
    axes[0].semilogy(th, [g["max_abs_m"] for g in growth], label="max|d eta|")
    axes[0].semilogy(th, [g["rms_m"] for g in growth], label="rms|d eta|")
    floor = growth[0]["max_abs_m"]
    axes[0].semilogy(th, floor * np.arange(1, len(growth) + 1), "k--",
                     label="linear accumulation of the 1-step floor")
    axes[0].set_xlabel("hours")
    axes[0].set_ylabel("m")
    axes[0].legend()
    axes[0].set_title("difference growth (wet cells only)")
    for key in ("wall", "equator", "interior"):
        axes[1].plot(th, [g["share"][key] for g in growth], label=key)
    axes[1].set_xlabel("hours")
    axes[1].set_ylabel("share of squared diff")
    axes[1].set_ylim(0, 1)
    axes[1].legend()
    axes[1].set_title("locus")
    fig.savefig(os.path.join(outdir, "fig_growth_locus.png"), dpi=110)
    plt.close(fig)

    # (d) spectra at the four probe points
    fig, axes = plt.subplots(1, 4, figsize=(20, 4.5), constrained_layout=True)
    for ax, (nm, sp) in zip(axes, spec.items()):
        f = sp["_freq_cph"][1:]
        ax.loglog(f, sp["_amp_nemo"][1:], label="NEMO")
        ax.loglog(f, sp["_amp_lego"][1:], label="legoESM")
        ax.set_title(f"{nm}  lat {sp['lat']:.1f} lon {sp['lon']:.1f}",
                     fontsize=9)
        ax.set_xlabel("cycles / hour")
        ax.set_ylabel("amplitude [m]")
        ax.legend(fontsize=8)
    fig.suptitle("eta frequency spectra at the four pre-registered probes")
    fig.savefig(os.path.join(outdir, "fig_spectra.png"), dpi=110)
    plt.close(fig)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = ap.add_subparsers(dest="cmd", required=True)

    e = sub.add_parser("extract-nemo",
                       help="stitch per-step NEMO ssh into one .npz")
    e.add_argument("--run-dir", required=True)
    e.add_argument("--kt0", type=int, default=5760)
    e.add_argument("--nsteps", type=int, default=160)
    e.add_argument("--out", required=True)

    c = sub.add_parser("compare", help="the pre-registered wave comparison")
    c.add_argument("--nemo", required=True)
    c.add_argument("--lego", required=True)
    c.add_argument("--outdir", required=True)
    c.add_argument("--mesh-mask", default=MESH_MASK)

    a = ap.parse_args(argv)
    if a.cmd == "extract-nemo":
        extract_nemo(a.run_dir, a.kt0, a.nsteps, a.out)
    elif a.cmd == "compare":
        compare(a.nemo, a.lego, a.outdir, a.mesh_mask)


if __name__ == "__main__":
    main()
