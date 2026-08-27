#!/usr/bin/env python
"""#1455 per-row rescore of the circumpolar-channel transport verdict.

Pre-registration: ``PREREG_channel_rescore.md`` at e2fd09158, committed before
any statistic in this file was computed.  This probe is offline: it loads the
saved verdict360 states and never steps either model.

The loaders, exact row transport reduction, channel slice, and per-row floor
construction are imported from ``regional_audit.py``.  They are not retyped.

Usage:
  JAX_ENABLE_X64=1 channel_rescore.py --self-test
  JAX_ENABLE_X64=1 channel_rescore.py --out-dir /tmp/dino_channel_rescore
"""
from __future__ import annotations

import argparse
import hashlib
import itertools
import json
import os
import statistics
import subprocess
import sys
from pathlib import Path

import numpy as np

_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, _DIR)
import regional_audit as R  # noqa: E402,N812 -- required upstream instrument

HORIZONS = (90, 180, 270, 360)
N_MEM = R.N_MEM
CHANNEL_NAME, CHANNEL_ROWS = R.BANDS[1]
CHANNEL_INDEX = np.arange(R.A.NY)[CHANNEL_ROWS]
N_ROWS = len(CHANNEL_INDEX)

PERSIST_HI = 0.70
PERSIST_LO = 0.30
ROUGH_SMOOTH = 0.25
ROUGH_GRID = 0.60
ROW_BAR = R.V.K_PREREG
ESCALATION_BAR = R.ESCALATION_FLOORS
N_BOOT = 5000
BOOT_SEED = 1226
BONFERRONI_PREDICTORS = 3

UPSTREAM_REGIONAL_SHA = "102ef501a"
UPSTREAM_REGIONAL_SHA256 = (
    "2f2bbc03afed792ea0148002029a6ad6481c93bd2ee37679b4bf615aab392fe7"
)


def _f64(a):
    return np.asarray(a, dtype=np.float64)


def pearson(x, y):
    """Signed Pearson correlation and its named centered-energy denominator."""
    xa = _f64(x) - np.mean(x)
    ya = _f64(y) - np.mean(y)
    den = float(np.sqrt(np.sum(xa * xa) * np.sum(ya * ya)))
    if not den > 0.0:
        raise ValueError("correlation is UNMEASURABLE on a constant profile")
    return float(np.sum(xa * ya) / den), den


def _lag_corr(x, lag):
    """Lag correlation about the full profile mean, with a lagged denominator."""
    a = _f64(x) - np.mean(x)
    left, right = a[:-lag], a[lag:]
    den = float(np.sqrt(np.sum(left * left) * np.sum(right * right)))
    return float(np.sum(left * right) / den) if den > 0.0 else 0.0


def effective_rows(x, y):
    """Registered all-lag Bartlett effective row count, clipped to [3, n]."""
    n = len(x)
    if n != len(y) or n < 4:
        raise ValueError("effective_rows requires equal profiles of length >=4")
    penalty = 1.0 + 2.0 * sum(
        (1.0 - lag / n) * _lag_corr(x, lag) * _lag_corr(y, lag)
        for lag in range(1, n)
    )
    denom = max(1.0, float(penalty))
    return float(min(n, max(3.0, n / denom))), denom


def fisher_interval(r, n_eff, confidence=0.95):
    """Autocorrelation-adjusted Fisher-z interval; no precision at n_eff<=3."""
    if n_eff <= 3.0:
        return (-1.0, 1.0)
    alpha = 1.0 - confidence
    zcrit = statistics.NormalDist().inv_cdf(1.0 - alpha / 2.0)
    rr = float(np.clip(r, -1.0 + 1e-15, 1.0 - 1e-15))
    z = np.arctanh(rr)
    half = zcrit / np.sqrt(n_eff - 3.0)
    return float(np.tanh(z - half)), float(np.tanh(z + half))


def _pair_specs(kind):
    if kind == "time":
        return [((m, t1), (m, t2)) for m in range(N_MEM)
                for t1, t2 in itertools.combinations(HORIZONS, 2)]
    if kind == "member":
        return [((m1, t), (m2, t)) for t in HORIZONS
                for m1, m2 in itertools.combinations(range(N_MEM), 2)]
    raise ValueError(kind)


def _correlations(profiles, specs, indices=None):
    idx = slice(None) if indices is None else indices
    return np.asarray([pearson(profiles[a][idx], profiles[b][idx])[0]
                       for a, b in specs], dtype=np.float64)


def moving_block_indices(n, block, rng):
    starts = np.arange(n - block + 1)
    chosen = rng.choice(starts, size=int(np.ceil(n / block)), replace=True)
    return np.concatenate([np.arange(s, s + block) for s in chosen])[:n]


def aggregate_stability(profiles, kind, n_boot=N_BOOT, seed=BOOT_SEED):
    specs = _pair_specs(kind)
    corr = _correlations(profiles, specs)
    neff = np.asarray([effective_rows(profiles[a], profiles[b])[0]
                       for a, b in specs], dtype=np.float64)
    fisher = np.asarray([fisher_interval(r, n) for r, n in zip(corr, neff)])
    block = int(np.clip(np.ceil(N_ROWS / np.median(neff)), 2, 12))
    rng = np.random.default_rng(seed + (0 if kind == "time" else 1))
    boot = []
    for _ in range(n_boot):
        idx = moving_block_indices(N_ROWS, block, rng)
        try:
            boot.append(float(np.median(_correlations(profiles, specs, idx))))
        except ValueError:
            continue
    if len(boot) < n_boot // 2:
        raise RuntimeError(f"{kind} bootstrap produced too few finite draws")
    ci = tuple(float(v) for v in np.percentile(boot, [2.5, 97.5]))
    median = float(np.median(corr))
    if median >= PERSIST_HI and ci[0] > PERSIST_LO:
        status = "CONFIRMED_PERSISTENT"
    elif median <= PERSIST_LO and ci[1] < PERSIST_HI:
        status = "CONFIRMED_DECORRELATING"
    else:
        status = "UNRESOLVED"
    return {
        "median_r": median,
        "block_ci95": list(ci),
        "block_length_rows": block,
        "n_correlations": len(corr),
        "correlations": corr.tolist(),
        "n_eff": neff.tolist(),
        "n_eff_median": float(np.median(neff)),
        "n_eff_min": float(np.min(neff)),
        "fisher_ci95": fisher.tolist(),
        "status": status,
    }


def roughness(g):
    a = _f64(g)
    den = 4.0 * float(np.sum((a - np.mean(a)) ** 2))
    if not den > 0.0:
        raise ValueError("roughness is UNMEASURABLE on a constant profile")
    return float(np.sum(np.diff(a) ** 2) / den), den


def row_agreement_status(p2, p5):
    if p2 >= 0.90 and p5 <= 0.10:
        return "CONFIRMED_GENUINE"
    if p5 >= 0.25:
        return "REFUTED_GENUINE"
    return "UNRESOLVED"


def cancellation_status(median_c, jackknife, edge):
    if median_c <= 0.10 and (jackknife >= 2.0 or edge >= 2.0):
        return "CONFIRMED_SENSITIVE"
    if median_c >= 0.50 and jackknife <= 1.0 and edge <= 1.0:
        return "CONFIRMED_ROBUST"
    return "UNRESOLVED"


def _assert_close(a, b, tol=1e-12):
    if not abs(a - b) <= tol:
        raise AssertionError(f"closure failed: {a} != {b} within {tol}")


def _plant_fires(label, fn):
    try:
        fn()
    except (AssertionError, ValueError):
        print(f"  PLANT FIRED: {label}")
        return
    raise AssertionError(f"planted violation did not fire: {label}")


def self_test(run_upstream=True):
    """New arithmetic controls; every claimed guard fires on a plant."""
    print("CHANNEL RESCORE SELF-TEST")
    if run_upstream:
        R.self_test()

    rng = np.random.default_rng(9)
    u = rng.normal(size=(R.A.NY, R.A.NX, R.A.NZ))
    rows = R.row_transports(u, R.A.umask)
    band = R.band_transport(u, R.A.umask, CHANNEL_ROWS)
    _assert_close(float(np.sum(rows[CHANNEL_ROWS])), band, 1e-9)
    bad_rows = rows.copy()
    bad_rows[CHANNEL_INDEX[0]] += 1e-3
    _plant_fires("wrong row weight breaks row-to-band closure",
                 lambda: _assert_close(float(np.sum(bad_rows[CHANNEL_ROWS])),
                                       band, 1e-9))

    surface = np.zeros_like(u)
    surface[:, :, 0] = 1.0
    official = R.row_transports(surface, R.A.umask)[CHANNEL_ROWS]
    bad_layer = np.mean(np.where(R.A.umask, surface, 0.0), axis=(1, 2))[
        CHANNEL_ROWS]
    _plant_fires("layer average differs from thickness-weighted reducer",
                 lambda: np.testing.assert_allclose(bad_layer, official,
                                                    rtol=1e-12, atol=1e-12))

    block_profile = np.repeat(np.arange(7, dtype=np.float64), 5)
    neff, _ = effective_rows(block_profile, block_profile)
    if not neff < N_ROWS:
        raise AssertionError("effective-row control did not reduce n")
    _plant_fires("independent-row denominator rejected on block profile",
                 lambda: _assert_close(neff, float(N_ROWS), 1e-12))

    base = np.sin(np.linspace(0.0, 2.0 * np.pi, N_ROWS))
    stable = {(m, t): base + 0.01 * m + 0.001 * t
              for m in range(N_MEM) for t in HORIZONS}
    stable_r = float(np.median(_correlations(stable, _pair_specs("time"))))
    if stable_r < PERSIST_HI:
        raise AssertionError("known persistent profiles missed the bar")
    permuted = {}
    for m in range(N_MEM):
        for it, t in enumerate(HORIZONS):
            permuted[(m, t)] = base[rng.permutation(N_ROWS)]
    noisy_r = float(np.median(_correlations(permuted, _pair_specs("time"))))
    _plant_fires("row permutation crosses persistence bar",
                 lambda: (_ for _ in ()).throw(AssertionError())
                 if noisy_r < PERSIST_LO else None)

    alt = (-1.0) ** np.arange(N_ROWS)
    smooth = np.linspace(1.0, 2.0, N_ROWS)
    z_alt, _ = roughness(alt)
    z_smooth, _ = roughness(smooth)
    if not (z_alt >= ROUGH_GRID and z_smooth <= ROUGH_SMOOTH):
        raise AssertionError((z_alt, z_smooth))
    if cancellation_status(0.0, 3.0, 0.0) != "CONFIRMED_SENSITIVE":
        raise AssertionError("alternating cancellation plant did not classify")
    _plant_fires("alternating profile cannot pass robust-sum bars",
                 lambda: (_ for _ in ()).throw(AssertionError())
                 if cancellation_status(0.0, 3.0, 0.0)
                 != "CONFIRMED_ROBUST" else None)

    gaps = np.asarray([1.0, 5.0])
    local_floor = np.asarray([0.1, 10.0])
    local = np.abs(gaps) > ESCALATION_BAR * local_floor
    globalized = np.abs(gaps) > ESCALATION_BAR * np.mean(local_floor)
    if np.array_equal(local, globalized):
        raise AssertionError("synthetic floors do not distinguish local/global")
    _plant_fires("global floor cannot replace row-specific floors",
                 lambda: np.testing.assert_array_equal(globalized, local))
    print("CHANNEL RESCORE SELF-TEST PASSED -- 6/6 planted violations fired")


def _git_state():
    sha = subprocess.run(["git", "-C", _DIR, "rev-parse", "HEAD"],
                         check=True, capture_output=True,
                         text=True).stdout.strip()
    dirty = bool(subprocess.run(["git", "-C", _DIR, "status", "--porcelain"],
                                check=True, capture_output=True,
                                text=True).stdout.strip())
    return sha, dirty


def _verify_upstream():
    data = Path(R.__file__).read_bytes()
    got = hashlib.sha256(data).hexdigest()
    if got != UPSTREAM_REGIONAL_SHA256:
        raise SystemExit(
            f"FATAL: regional_audit.py sha256 {got} != registered "
            f"{UPSTREAM_REGIONAL_SHA256}; exact reducer reuse is not proven")
    return got


def _install_clock_helper_compat():
    """Bridge the post-audit public-to-private rename, with no logic change.

    ``git diff fidelity/dino-regional-audit -- kamm_twin_90d.py`` shows the
    helper body unchanged: only ``restart_elapsed_seconds`` became
    ``_restart_elapsed_seconds`` and its two in-file call sites moved with it.
    The imported audit still calls the former public spelling.
    """
    twin = R.X.T
    if hasattr(twin, "restart_elapsed_seconds"):
        return "native restart_elapsed_seconds"
    helper = getattr(twin, "_restart_elapsed_seconds", None)
    if helper is None:
        raise SystemExit(
            "FATAL: neither clock helper spelling exists; provenance cannot "
            "be checked")
    twin.restart_elapsed_seconds = helper
    return "compat alias restart_elapsed_seconds -> _restart_elapsed_seconds"


def load_rows():
    """Load only through the audit's stamp-refusing state loaders/reducer."""
    clock_compat = _install_clock_helper_compat()
    prov, clock = R.control_stamps()
    lego, nemo = {}, {}
    for day in HORIZONS:
        print(f"loading day {day}: 4 legoESM + 4 NEMO states", flush=True)
        lego[day] = np.asarray([
            R.row_transports(R.load_lego(m, day)["u"], R.A.umask)
            for m in range(N_MEM)
        ])
        nemo[day] = np.asarray([
            R.row_transports(R.load_nemo(m, day)["u"], R.A.umask)
            for m in range(N_MEM)
        ])
        R.control_dtype(lego[day], nemo[day])
        R.control_finite(f"row transports day {day}",
                         np.ravel([lego[day], nemo[day]]))
    return lego, nemo, prov, clock, clock_compat


def spatial_decider(mean_gap, floors, nemo_rows):
    ones = np.ones((R.A.NY, R.A.NX, R.A.NZ), dtype=np.float64)
    capacity = R.row_transports(ones, R.A.umask)[CHANNEL_ROWS]
    by_day = {}
    for day in HORIZONS:
        pred_day = {
            "transport_capacity": capacity,
            "nemo_row_transport": nemo_rows[day][0, CHANNEL_ROWS],
            "nemo_row_gradient": np.gradient(nemo_rows[day][0, CHANNEL_ROWS]),
        }
        by_day[str(day)] = {}
        for name, pred in pred_day.items():
            r, den = pearson(mean_gap[day], pred)
            neff, neff_den = effective_rows(mean_gap[day], pred)
            ci = fisher_interval(r, neff, confidence=1.0 - 0.05 / 3.0)
            by_day[str(day)][name] = {
                "r": r, "pearson_denominator": den, "n_eff": neff,
                "n_eff_denominator": neff_den, "bonferroni_ci98_33": list(ci),
            }
    d360 = by_day["360"]
    winner = max(d360, key=lambda name: abs(d360[name]["r"]))
    w = d360[winner]
    signs = [np.sign(by_day[str(day)][winner]["r"]) for day in HORIZONS]
    sign_stable = int(sum(s == signs[-1] and s != 0 for s in signs))
    ci = w["bonferroni_ci98_33"]
    effect_lower = min(abs(ci[0]), abs(ci[1])) if np.sign(ci[0]) == np.sign(ci[1]) else 0.0
    z, zden = roughness(mean_gap[360])
    eligible = ((np.abs(mean_gap[360][:-1]) > ROW_BAR * floors[360][:-1])
                & (np.abs(mean_gap[360][1:]) > ROW_BAR * floors[360][1:]))
    flips = eligible & (np.sign(mean_gap[360][:-1])
                        != np.sign(mean_gap[360][1:]))
    q = float(flips.sum() / eligible.sum()) if eligible.sum() else None
    pred_confirm = (abs(w["r"]) >= PERSIST_HI and effect_lower > PERSIST_LO
                    and sign_stable >= 3)
    if (pred_confirm or z <= ROUGH_SMOOTH) and z < ROUGH_GRID:
        status = "CONFIRMED_PHYSICAL_STRUCTURE"
    else:
        all_small = all(abs(v["r"]) <= PERSIST_LO for v in d360.values())
        all_upper = all(max(abs(x) for x in v["bonferroni_ci98_33"])
                        < PERSIST_HI for v in d360.values())
        grid = (all_small and all_upper and z >= ROUGH_GRID
                and eligible.sum() >= 6 and q is not None and q >= 2.0 / 3.0)
        status = "CONFIRMED_GRID_SCALE" if grid else "UNRESOLVED"
    return {
        "status": status, "predictors_by_day": by_day, "winner": winner,
        "r_phys": w["r"], "winner_effect_lower": effect_lower,
        "winner_sign_stable_horizons": sign_stable,
        "roughness_Z": z, "roughness_denominator": zden,
        "q_flip": q, "q_flip_numerator": int(flips.sum()),
        "q_flip_denominator": int(eligible.sum()),
    }


def evaluate(lego, nemo):
    paired = {day: lego[day][:, CHANNEL_ROWS] - nemo[day][:, CHANNEL_ROWS]
              for day in HORIZONS}
    profiles = {(m, day): paired[day][m]
                for day in HORIZONS for m in range(N_MEM)}
    floors = {day: np.sqrt(np.std(lego[day][:, CHANNEL_ROWS], axis=0, ddof=1) ** 2
                                + np.std(nemo[day][:, CHANNEL_ROWS], axis=0, ddof=1) ** 2)
              for day in HORIZONS}
    time = aggregate_stability(profiles, "time")
    member = aggregate_stability(profiles, "member")

    cross = {(i, k, day): lego[day][i, CHANNEL_ROWS] - nemo[day][k, CHANNEL_ROWS]
             for day in HORIZONS for i in range(N_MEM) for k in range(N_MEM)}
    cross_time = [pearson(cross[(i, k, a)], cross[(i, k, b)])[0]
                  for i in range(N_MEM) for k in range(N_MEM)
                  for a, b in itertools.combinations(HORIZONS, 2)]
    cross_member = [pearson(cross[(i1, k1, day)], cross[(i2, k2, day)])[0]
                    for day in HORIZONS
                    for (i1, k1), (i2, k2) in itertools.combinations(
                        itertools.product(range(N_MEM), range(N_MEM)), 2)]
    robust = {"time_median_r_all16": float(np.median(cross_time)),
              "member_median_r_all16": float(np.median(cross_member))}
    if (time["status"] == "CONFIRMED_PERSISTENT"
            and robust["time_median_r_all16"] < PERSIST_HI):
        time["status"] = "UNRESOLVED_ROBUSTNESS"
    if (member["status"] == "CONFIRMED_PERSISTENT"
            and robust["member_median_r_all16"] < PERSIST_HI):
        member["status"] = "UNRESOLVED_ROBUSTNESS"

    mean_gap = {day: np.mean(paired[day], axis=0) for day in HORIZONS}
    spatial = spatial_decider(mean_gap, floors, nemo)

    g = paired[360][0]
    f = floors[360]
    ratios = np.abs(g) / np.where(f > 0.0, f, np.inf)
    p2 = float(np.mean(ratios <= ROW_BAR))
    p5 = float(np.mean(ratios > ESCALATION_BAR))
    x = float(np.sum(np.abs(g)))
    x2 = float(np.sum(np.maximum(np.abs(g) - ROW_BAR * f, 0.0)))
    row_level = {
        "status": row_agreement_status(p2, p5),
        "P2_agree_fraction": p2, "P2_numerator": int(np.sum(ratios <= ROW_BAR)),
        "P2_denominator": N_ROWS, "P5_disagree_fraction": p5,
        "P5_numerator": int(np.sum(ratios > ESCALATION_BAR)),
        "P5_denominator": N_ROWS, "X_gross_sv": x,
        "X2_floor_excess_sv": x2, "X2_over_X": x2 / x,
        "X2_bound_status": (
            "UPPER bound on floor-excess magnitude because the day-360 "
            "floors are unsaturated"
        ),
        "row_ratios_upper_bound": ratios.tolist(),
    }

    c_all = [abs(float(np.sum(paired[day][m])))
             / float(np.sum(np.abs(paired[day][m])))
             for day in HORIZONS for m in range(N_MEM)]
    band_l = np.sum(lego[360][:, CHANNEL_ROWS], axis=1)
    band_n = np.sum(nemo[360][:, CHANNEL_ROWS], axis=1)
    band_floor = float(np.sqrt(np.std(band_l, ddof=1) ** 2
                               + np.std(band_n, ddof=1) ** 2))
    band_net = float(np.sum(g))
    jackknife = float(np.max(np.abs(g)) / band_floor)
    full_gap = lego[360][0] - nemo[360][0]
    variants = {}
    for dlo in (-1, 0, 1):
        for dhi in (-1, 0, 1):
            rows = slice(CHANNEL_ROWS.start + dlo, CHANNEL_ROWS.stop + dhi)
            val = float(np.sum(full_gap[rows]))
            variants[f"{dlo:+d}|{dhi:+d}"] = val
    edge = max(abs(v - band_net) for v in variants.values()) / band_floor
    median_c = float(np.median(c_all))
    sensitivity = {
        "status": cancellation_status(median_c, jackknife, edge),
        "C_median": median_c, "C_values": c_all,
        "C_denominator": "sum_abs_row_gap",
        "day360_control_net_sv": band_net,
        "day360_control_C": abs(band_net) / x,
        "band_floor_sv": band_floor, "J_one_row_over_band_floor": jackknife,
        "E_edge_over_band_floor": edge, "edge_variant_sums_sv": variants,
    }

    persistent = (time["status"] == "CONFIRMED_PERSISTENT"
                  and member["status"] == "CONFIRMED_PERSISTENT")
    not_grid = spatial["status"] != "CONFIRMED_GRID_SCALE"
    real = (persistent and not_grid
            and row_level["status"] == "REFUTED_GENUINE"
            and sensitivity["status"] == "CONFIRMED_SENSITIVE")
    noise = (time["status"] == "CONFIRMED_DECORRELATING"
             and member["status"] == "CONFIRMED_DECORRELATING"
             and row_level["status"] == "CONFIRMED_GENUINE"
             and sensitivity["status"] == "CONFIRMED_ROBUST")
    if real:
        final = "CONFIRMED_PERSISTENT_STRUCTURE_MASKING_REAL_PATTERN"
    elif noise:
        final = "CONFIRMED_DECORRELATING_NOISE_GENUINE_ROW_AGREEMENT"
    elif persistent:
        final = "PLAUSIBLE_PERSISTENT_STRUCTURE"
    else:
        final = "PLAUSIBLE_UNRESOLVED"
    return {
        "time_stability": time, "member_stability": member,
        "all16_robustness": robust, "spatial_structure": spatial,
        "row_level": row_level, "band_sensitivity": sensitivity,
        "final_verdict": final,
        "gaps_paired_sv": {str(day): paired[day].tolist() for day in HORIZONS},
        "row_floors_sv": {str(day): floors[day].tolist() for day in HORIZONS},
        "mean_gaps_sv": {str(day): mean_gap[day].tolist() for day in HORIZONS},
    }


def run(out_dir):
    sha_start, dirty_start = _git_state()
    if dirty_start:
        raise SystemExit(f"FATAL: producer tree {sha_start} is dirty")
    upstream_hash = _verify_upstream()
    lego, nemo, prov, clock, clock_compat = load_rows()
    result = evaluate(lego, nemo)
    sha_end, dirty_end = _git_state()
    if (sha_end, dirty_end) != (sha_start, dirty_start):
        raise SystemExit("FATAL: producer tree changed during measurement")
    result["_stamp"] = {
        "producer_sha": sha_start, "producer_tree_dirty": dirty_start,
        "probe": os.path.basename(__file__),
        "prereg": "PREREG_channel_rescore.md@e2fd09158",
        "upstream_regional_sha": UPSTREAM_REGIONAL_SHA,
        "upstream_regional_sha256": upstream_hash,
        "lego_dir": R.X.LEGO_DIR,
        "nemo_dirs": [R.X.nemo_dir(i) for i in range(N_MEM)],
        "horizons": list(HORIZONS),
        "channel_rows": [int(CHANNEL_INDEX.min()), int(CHANNEL_INDEX.max())],
        "n_rows": N_ROWS,
        "constants": {
            "PERSIST_HI": PERSIST_HI, "PERSIST_LO": PERSIST_LO,
            "ROUGH_SMOOTH": ROUGH_SMOOTH, "ROUGH_GRID": ROUGH_GRID,
            "ROW_BAR": ROW_BAR, "ESCALATION_BAR": ESCALATION_BAR,
            "N_BOOT": N_BOOT, "BOOT_SEED": BOOT_SEED,
        },
        "input_provenance": prov, "clock": clock,
        "clock_helper_compat": clock_compat,
        "floor_status": "UNSATURATED; multiples/counts are upper bounds",
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "channel_rescore.json"
    path.write_text(json.dumps(result, indent=1, default=float) + "\n")
    print(json.dumps({
        "final_verdict": result["final_verdict"],
        "time": result["time_stability"],
        "member": result["member_stability"],
        "spatial": result["spatial_structure"],
        "row_level": result["row_level"],
        "sensitivity": result["band_sensitivity"],
        "artifact": str(path), "producer_sha": sha_start,
    }, indent=1, default=float))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--self-test", action="store_true")
    parser.add_argument("--skip-upstream-self-test", action="store_true")
    parser.add_argument("--out-dir", default="/tmp/dino_channel_rescore")
    args = parser.parse_args(argv)
    self_test(run_upstream=not args.skip_upstream_self_test)
    if not args.self_test:
        run(args.out_dir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
