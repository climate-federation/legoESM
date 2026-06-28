#!/usr/bin/env python
"""WeatherBench-style scorecard for the AIMIP classical scheme sweep.

Aggregates every OAT combo's ``aimip_scorecard.json`` (written by
``run_aimip.py`` per combo) into one ranked table + a heatmap PNG, and
names the best-performing classical scheme combination.

Each combo varies ONE parameterization category (convection / turbulence /
gravity-wave drag / microphysics / cloud) off the baseline, RRTMGP fixed.
Rows are sorted by the headline metric (T@500 RMSE); the heatmap colors
each metric column by its rank (green = best) — the WeatherBench scorecard
convention.

Usage:
    python aimip_sweep_scorecard.py results/aimip_classical_sweep_stage1 \
        [config/aimip/sweep/stage1/manifest.json]
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

# Headline + secondary metrics (keys as emitted by run_aimip eval).
_METRICS = ["T", "T_sfc", "u", "v", "p_s"]


def _combo_schemes(manifest: dict) -> dict[str, dict]:
    """combo name -> its (changed) scheme overrides, from the manifest."""
    out = {}
    for c in manifest.get("combos", []):
        ov = c.get("overrides", {})
        out[c["name"]] = {k.replace("aimip_", ""): v for k, v in ov.items()}
    return out


def _read_combo_rmse(scorecard_path: Path) -> dict | None:
    """classical eval RMSE/bias dict for one combo, or None if absent/failed."""
    if not scorecard_path.exists():
        return None
    d = json.loads(scorecard_path.read_text())
    v = d.get("variants", d).get("classical")
    if not isinstance(v, dict):
        return None
    em = v.get("eval_metrics", {})
    rmse, bias = em.get("rmse", {}), em.get("bias", {})
    if "T" not in rmse:
        return None
    return {
        "rmse": {m: rmse.get(m, {}).get("mean") for m in _METRICS},
        "T_bias": bias.get("T", {}).get("mean"),
    }


def collect(sweep_dir: Path, manifest: dict, min_mtime: float = 0.0) -> list[dict]:
    """One row per combo: name, schemes, metrics, score (T + |T_bias|).

    Two freshness guards — a sweep REUSES combo dirs by name, so a prior
    sweep's results (e.g. an old gray-radiation run) can masquerade as
    current:
      * manifest filter — skip dirs not in THIS manifest (``combo_*_none``,
        ``combo_baseline_smoke`` leftovers);
      * ``min_mtime`` — skip any combo whose ``aimip_scorecard.json`` predates
        the sweep config (default = the manifest's own mtime), so a combo
        whose new run has not finished yet shows up as MISSING rather than
        contributing a stale score.
    """
    schemes = _combo_schemes(manifest)
    manifest_names = {c["name"] for c in manifest.get("combos", [])}
    rows = []
    n_stale = 0
    for combo_dir in sorted(sweep_dir.glob("combo_*")):
        if manifest_names and combo_dir.name not in manifest_names:
            continue  # stale dir from a prior sweep — not in this manifest
        sc = combo_dir / "aimip_scorecard.json"
        if min_mtime and sc.exists() and sc.stat().st_mtime < min_mtime:
            n_stale += 1
            continue  # leftover from an earlier sweep; its new run hasn't finished
        r = _read_combo_rmse(sc)
        if r is None:
            continue
        T = r["rmse"]["T"]
        tbias = r["T_bias"] or 0.0
        rows.append({
            "name": combo_dir.name,
            "schemes": schemes.get(combo_dir.name, {}),
            "rmse": r["rmse"],
            "T_bias": r["T_bias"],
            "score": (T or float("inf")) + abs(tbias),  # WB obj: rmse + |bias|
        })
    rows.sort(key=lambda x: x["score"])
    return rows


def _text_table(rows: list[dict]) -> str:
    hdr = f"{'rank':>4} {'combo':28} {'T':>7} {'T_sfc':>7} {'u':>6} {'v':>6} {'ps':>8} {'score':>7}"
    lines = ["# AIMIP classical scheme sweep — WeatherBench scorecard (RRTMGP fixed)",
             "# sorted by score = T@500 RMSE + |T bias|; lower = better", hdr,
             "-" * len(hdr)]
    for i, r in enumerate(rows):
        m = r["rmse"]
        def f(x, w=7, p=3):
            return f"{x:{w}.{p}f}" if isinstance(x, (int, float)) else f"{'--':>{w}}"
        lines.append(
            f"{i:>4} {r['name']:28} {f(m['T'])} {f(m['T_sfc'])} "
            f"{f(m['u'],6,2)} {f(m['v'],6,2)} {f(m['p_s'],8,1)} {f(r['score'])}")
    if rows:
        b = rows[0]
        lines += ["", f"BEST: {b['name']}  schemes={b['schemes']}  "
                  f"T@500={b['rmse']['T']:.3f}K  score={b['score']:.3f}"]
    return "\n".join(lines)


def _heatmap(rows: list[dict], out_png: Path) -> None:
    import numpy as np
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    names = [r["name"].replace("combo_", "") for r in rows]
    M = np.array([[r["rmse"][m] if isinstance(r["rmse"][m], (int, float)) else np.nan
                   for m in _METRICS] for r in rows], dtype=float)
    # Per-column rank-normalize (0=best..1=worst) for the color scale.
    col = np.zeros_like(M)
    for j in range(M.shape[1]):
        c = M[:, j]
        lo, hi = np.nanmin(c), np.nanmax(c)
        col[:, j] = (c - lo) / (hi - lo) if hi > lo else 0.0
    fig, ax = plt.subplots(figsize=(7, max(3, 0.32 * len(names) + 1)))
    im = ax.imshow(col, cmap="RdYlGn_r", aspect="auto", vmin=0, vmax=1)
    ax.set_xticks(range(len(_METRICS))); ax.set_xticklabels(_METRICS)
    ax.set_yticks(range(len(names))); ax.set_yticklabels(names, fontsize=7)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            if not np.isnan(M[i, j]):
                ax.text(j, i, f"{M[i,j]:.2f}", ha="center", va="center",
                        fontsize=6, color="black")
    ax.set_title("AIMIP classical sweep scorecard (RMSE; green=best per col)")
    fig.colorbar(im, ax=ax, label="per-metric rank (0=best)")
    fig.tight_layout(); fig.savefig(out_png, dpi=130); plt.close(fig)


def main(argv=None):
    argv = argv if argv is not None else sys.argv[1:]
    if not argv:
        raise SystemExit("usage: aimip_sweep_scorecard.py <sweep_dir> [manifest.json]")
    sweep_dir = Path(argv[0])
    manifest_path = Path(argv[1]) if len(argv) > 1 else (
        sweep_dir.parent.parent / "config/aimip/sweep/stage1/manifest.json")
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    # Freshness threshold = the manifest's mtime: combos whose scorecard
    # predates it are leftovers from an earlier sweep (their new run is
    # unfinished) and are excluded.
    min_mtime = manifest_path.stat().st_mtime if manifest_path.exists() else 0.0
    rows = collect(sweep_dir, manifest, min_mtime=min_mtime)
    n_expected = len(manifest.get("combos", []))
    if n_expected and len(rows) < n_expected:
        print(f"# WARNING: only {len(rows)}/{n_expected} combos are FRESH "
              f"(rest still running or stale) — ranking is PRELIMINARY.\n")
    table = _text_table(rows)
    print(table)
    (sweep_dir / "sweep_scorecard.txt").write_text(table + "\n")
    (sweep_dir / "sweep_scorecard.json").write_text(json.dumps(rows, indent=2))
    if rows:
        try:
            _heatmap(rows, sweep_dir / "sweep_scorecard.png")
            print(f"\nwrote {sweep_dir}/sweep_scorecard.{{txt,json,png}}")
        except Exception as e:  # plotting must not lose the table
            print(f"(heatmap skipped: {e}); wrote txt+json")
    return rows


if __name__ == "__main__":
    main()
