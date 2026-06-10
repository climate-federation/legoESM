"""Side-by-side comparison driver: Veros reference runs vs legoESM matrix runs.

Walks the cached Veros results under
``$LEGOESM_OCEAN_FIDELITY_CACHE/veros/`` (or the default cache) and the
legoESM matrix artifacts under ``results/ocean/`` and emits a Markdown
report enumerating both pipelines side by side.

Direct quantitative comparison is only available when the same case has
both a Veros adapter (``src/legoesm/ocean/fidelity/veros_configs/``) and
a legoESM matrix runner (``scripts/matrix/run_ocean_test_matrix.py::RUNNERS``).
The current gaps are tracked as plan tasks #11 (legoESM runners for
acc_channel / global_overturning / dino) and #12 (Veros setups for
lock_exchange / overflow / eady_uniform / dino). The script handles the
gap gracefully: cases that have only one side are reported with the
available pipeline only.

Usage::

    JAX_PLATFORMS=cpu .venv/bin/python scripts/validate/ocean_fidelity/run_comparison.py \\
        --legoesm-root results/ocean \\
        --output docs/ocean_fidelity/initial_comparison_<sha>.md

The script does NOT trigger runs — it only reads what is already on disk.
Run ``scripts/matrix/run_ocean_test_matrix.py`` and the Veros runner separately
to populate both sides first.
"""

from __future__ import annotations

import argparse
import pickle
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[2]
# Federation-aware bootstrap for a non-installed checkout: after the uv-workspace
# carve the ``legoesm`` namespace is split across packages/<member>/legoesm + the
# root src/legoesm (meta), so adding only ``<repo>/src`` no longer exposes moved
# members like ``legoesm.ocean``.  (The canonical setup is an editable install —
# ``uv sync`` / ``pip install -e`` — which makes this loop a no-op.)
for _root in [REPO_ROOT / "src", *sorted((REPO_ROOT / "packages").glob("*"))]:
    if _root.is_dir() and str(_root) not in sys.path:
        sys.path.insert(0, str(_root))

from legoesm.ocean.fidelity import artifacts as _artifacts  # noqa: E402
from legoesm.ocean.fidelity import cache as _cache  # noqa: E402


def _load_veros_results() -> dict[str, Any]:
    """Return {case_name: VerosResult} loaded from the on-disk cache.

    When a case has multiple cached runs (different runlen / kwargs), the
    one with the longest physical integration is preferred — that is the
    most physically meaningful run for like-for-like comparison.
    """
    root = _cache.sub("veros")
    out: dict[str, Any] = {}
    if not root.is_dir():
        return out
    for case_dir in root.iterdir():
        if not case_dir.is_dir():
            continue
        best = None
        for key_dir in case_dir.iterdir():
            pkl = key_dir / "result.pkl"
            if not pkl.exists():
                continue
            with open(pkl, "rb") as f:
                candidate = pickle.load(f)
            cand_runlen = float(candidate.provenance.get("runlen_s", 0.0))
            if best is None or cand_runlen > float(
                best.provenance.get("runlen_s", 0.0)
            ):
                best = candidate
        if best is not None:
            out[case_dir.name] = best
    return out


def _load_legoesm_results(root: Path) -> dict[str, list]:
    """Return {case_name: [ArtifactBundle, ...]} for every case dir found.

    Walks any depth (some cases like ``rest_state`` have a variant level
    between case and grid). ``ArtifactBundle.case`` field comes from the
    third-from-last path component, so rest_state variants surface as
    distinct ``case`` keys (``rest_state_stratified_with_land`` etc.).
    """
    out: dict[str, list] = {}
    if not root.is_dir():
        return out
    for results_path in sorted(root.rglob("results.txt")):
        case_dir = results_path.parent
        bundle = _artifacts.load(case_dir)
        out.setdefault(bundle.case, []).append(bundle)
    return out


def _field_summary(arr: np.ndarray) -> str:
    arr = np.asarray(arr)
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return "all NaN"
    return (
        f"shape={tuple(arr.shape)} min={float(finite.min()):+.3e} "
        f"max={float(finite.max()):+.3e} mean={float(finite.mean()):+.3e}"
    )


def _format_veros_section(case: str, result) -> list[str]:
    lines = [
        f"### Veros `{case}`",
        "",
        f"- runlen: {result.provenance['runlen_s'] / 86400.0:.2f} d",
        f"- wall time: {result.provenance['wall_seconds']:.2f} s",
        f"- veros version: {result.provenance['veros_version']}",
        f"- grid: nx={result.grid_metadata.get('nx')}, "
        f"ny={result.grid_metadata.get('ny')}, "
        f"nz={result.grid_metadata.get('nz')}",
        "",
        "| field | summary |",
        "| --- | --- |",
    ]
    for name in ("u", "v", "w", "temp", "salt", "rho", "psi",
                 "surface_taux", "surface_tauy"):
        if name in result.variables:
            lines.append(f"| `{name}` | {_field_summary(result.variables[name])} |")
    lines.append("")
    return lines


def _format_legoesm_section(case: str, bundles: list) -> list[str]:
    lines = [f"### legoESM `{case}`", ""]
    if not bundles:
        lines.append("_No artifacts found under results/ocean/._")
        lines.append("")
        return lines
    lines.append("| grid | resolution | status | notes |")
    lines.append("| --- | --- | --- | --- |")
    for b in bundles:
        status = b.scalars.get("status", "?")
        notes = str(b.scalars.get("notes", ""))[:100]
        lines.append(f"| {b.grid} | {b.resolution} | {status} | {notes} |")
    lines.append("")
    for b in bundles:
        if not b.timeseries:
            continue
        keys = [k for k in b.timeseries if k not in ("step", "time_days")]
        if not keys:
            continue
        lines.append(
            f"#### {b.grid}/{b.resolution} timeseries (last sample)"
        )
        lines.append("")
        lines.append("| key | value |")
        lines.append("| --- | --- |")
        for k in keys:
            v = b.timeseries[k]
            if v.size == 0 or v.dtype == object:
                continue
            lines.append(f"| `{k}` | {float(v[-1]):+.6e} |")
        lines.append("")
    # Cross-grid delta when more than one grid is present.
    grids = {b.grid: b for b in bundles}
    if len(grids) >= 2:
        lines.extend(_format_cross_grid_deltas(case, grids))
    return lines


def _veros_scalar(result, name: str) -> dict[str, float]:
    """Return {min, max, mean} of a Veros variable, with halos dropped if 2D+."""
    arr = result.variables.get(name)
    if arr is None or arr.size == 0:
        return {}
    arr = np.asarray(arr)
    if arr.ndim >= 2:
        arr = arr[2:-2, 2:-2] if arr.shape[0] > 4 and arr.shape[1] > 4 else arr
    if arr.ndim == 4:
        arr = arr[..., -1]  # latest tau
    finite = arr[np.isfinite(arr)]
    if finite.size == 0:
        return {}
    return {
        "min": float(finite.min()),
        "max": float(finite.max()),
        "mean": float(finite.mean()),
    }


def _legoesm_scalar(bundle, key: str) -> float | None:
    ts = bundle.timeseries.get(key)
    if ts is None or ts.size == 0 or ts.dtype == object:
        return None
    return float(ts[-1])


def _format_overlap_section(case: str, veros_result, legoesm_bundles: list) -> list[str]:
    """Side-by-side scalar comparison for a case with both pipelines present."""
    lines = [f"### `{case}` — Veros vs legoESM", ""]
    veros_t = _veros_scalar(veros_result, "temp")
    veros_u = _veros_scalar(veros_result, "u")
    lines.append(
        f"- Veros run: runlen={veros_result.provenance['runlen_s'] / 3600.0:.2f} h"
        f" on {veros_result.grid_metadata.get('nx')}x"
        f"{veros_result.grid_metadata.get('ny')}x"
        f"{veros_result.grid_metadata.get('nz')}"
    )
    lines.append(f"- legoESM runs: {[f'{b.grid}/{b.resolution}' for b in legoesm_bundles]}")
    lines.append("")
    lines.append("| metric | Veros | " + " | ".join(
        f"legoESM `{b.grid}/{b.resolution}`" for b in legoesm_bundles
    ) + " |")
    lines.append("| --- | --- | " + " | ".join(["---"] * len(legoesm_bundles)) + " |")

    if veros_t:
        legoesm_T_min = [_legoesm_scalar(b, "min_T") for b in legoesm_bundles]
        legoesm_T_max = [_legoesm_scalar(b, "max_T") for b in legoesm_bundles]
        legoesm_T_mean = [_legoesm_scalar(b, "mean_T") for b in legoesm_bundles]
        lines.append(_overlap_row("T min (°C)", veros_t["min"], legoesm_T_min))
        lines.append(_overlap_row("T max (°C)", veros_t["max"], legoesm_T_max))
        lines.append(_overlap_row("T mean (°C)", veros_t["mean"], legoesm_T_mean))

    if veros_u:
        # Prefer ``max_abs_u`` (full-column max — matches Veros's 3D max);
        # fall back to ``max_speed`` for legacy bundles that only stored
        # the surface-level metric.
        legoesm_u_max = []
        for b in legoesm_bundles:
            val = _legoesm_scalar(b, "max_abs_u")
            if val is None:
                val = _legoesm_scalar(b, "max_speed")
            legoesm_u_max.append(val)
        veros_u_max = max(abs(veros_u["min"]), abs(veros_u["max"]))
        lines.append(_overlap_row("max |u| (m/s)", veros_u_max, legoesm_u_max))

    legoesm_ke = [_legoesm_scalar(b, "mean_ke") for b in legoesm_bundles]
    if any(v is not None for v in legoesm_ke):
        lines.append(_overlap_row("mean KE (m²/s²)", float("nan"), legoesm_ke))

    lines.append("")
    lines.append(
        "_Note: Veros B-grid output and legoESM C-grid output have not "
        "been regridded onto a common grid, so spatial pattern correlations "
        "are not reported here. Scalar metrics above are computed on each "
        "model's native grid; their direct comparison is meaningful as a "
        "smoke check (right order of magnitude, matching sign / extrema), "
        "not as a precise pointwise fidelity score._"
    )
    lines.append("")
    return lines


def _overlap_row(label: str, veros_val: float, legoesm_vals: list) -> str:
    veros_cell = "—" if not np.isfinite(veros_val) else f"{veros_val:+.4e}"
    legoesm_cells = [
        "—" if v is None or not np.isfinite(v) else f"{v:+.4e}"
        for v in legoesm_vals
    ]
    return f"| {label} | {veros_cell} | " + " | ".join(legoesm_cells) + " |"


def _format_cross_grid_deltas(case: str, grids: dict) -> list[str]:
    """Emit a per-key |latlon - mpas| comparison of final-sample timeseries."""
    lines = [f"#### {case} cross-grid deltas (final sample)", ""]
    lines.append("| key | " + " | ".join(grids) + " | spread |")
    lines.append("| --- | " + " | ".join(["---"] * len(grids)) + " | --- |")
    finite_keys: dict[str, dict[str, float]] = {}
    for grid_name, bundle in grids.items():
        for k, v in bundle.timeseries.items():
            if k in ("step", "time_days") or v.size == 0 or v.dtype == object:
                continue
            finite_keys.setdefault(k, {})[grid_name] = float(v[-1])
    for k in sorted(finite_keys):
        per_grid = finite_keys[k]
        if len(per_grid) != len(grids):
            continue
        values = list(per_grid.values())
        spread = max(values) - min(values)
        cells = " | ".join(f"{per_grid[g]:+.4e}" for g in grids)
        lines.append(f"| `{k}` | {cells} | {spread:+.4e} |")
    lines.append("")
    return lines


def render_report(
    veros_results: dict[str, Any],
    legoesm_results: dict[str, list],
    *,
    git_sha: str,
) -> str:
    lines: list[str] = []
    lines.append(f"# Ocean fidelity initial comparison — `{git_sha}`")
    lines.append("")
    lines.append(
        "Side-by-side dump of currently cached Veros reference runs and "
        "currently emitted legoESM matrix artifacts. Generated by "
        "`scripts/validate/ocean_fidelity/run_comparison.py`."
    )
    lines.append("")
    lines.append("## Pipeline status")
    lines.append("")
    lines.append(f"- Veros cases cached: {sorted(veros_results)}")
    lines.append(f"- legoESM cases with artifacts: {sorted(legoesm_results)}")
    overlap = sorted(set(veros_results) & set(legoesm_results))
    lines.append(f"- Direct overlap: {overlap or '_none yet — see plan task #11 / #12_'}")
    lines.append("")

    if veros_results:
        lines.append("## Veros runs")
        lines.append("")
        for case, result in sorted(veros_results.items()):
            lines.extend(_format_veros_section(case, result))
    else:
        lines.append("## Veros runs")
        lines.append("")
        lines.append(
            "_No Veros results in cache. Run "
            "`from legoesm.ocean.fidelity import veros_runner; "
            "veros_runner.run_veros('acc_channel')` to populate._"
        )
        lines.append("")

    if legoesm_results:
        lines.append("## legoESM runs")
        lines.append("")
        for case, bundles in sorted(legoesm_results.items()):
            lines.extend(_format_legoesm_section(case, bundles))
    else:
        lines.append("## legoESM runs")
        lines.append("")
        lines.append(
            "_No legoESM matrix artifacts found. Run "
            "`scripts/matrix/run_ocean_test_matrix.py --grid latlon --quick "
            "--emit-fidelity-artifacts` first._"
        )
        lines.append("")

    if overlap:
        lines.append("## Like-for-like comparison")
        lines.append("")
        for case in overlap:
            lines.extend(_format_overlap_section(
                case, veros_results[case], legoesm_results[case],
            ))
    else:
        lines.append("## Like-for-like comparison")
        lines.append("")
        lines.append(
            "**Currently empty.** legoESM runners exist for "
            "`rest_state`, `barotropic_wave`, `inertia_gravity_wave`, "
            "`geostrophic_adjustment`, `phillips_two_layer`, "
            "`lock_exchange`, `overflow`, `regional_gyre`, "
            "`baroclinic_gyre`, `global_barotropic_wind`, "
            "`stommel_gyre_tracer`. Veros adapters exist for "
            "`acc_channel`, `global_overturning`. No case is in both "
            "sets. Unblock by closing plan task #11 (legoESM runners "
            "for acc_channel / global_overturning) or task #12 (Veros "
            "setups for lock_exchange / overflow / eady_uniform / dino)."
        )
        lines.append("")
    return "\n".join(lines)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        description="Compare cached Veros reference runs against legoESM matrix artifacts."
    )
    p.add_argument(
        "--legoesm-root", type=Path,
        default=Path("results/ocean"),
        help="legoESM ocean test matrix output root (default: results/ocean)",
    )
    p.add_argument(
        "--output", type=Path,
        default=Path(f"docs/ocean_fidelity/initial_comparison_{_git_sha()}.md"),
        help="markdown report path (default: hash-suffixed under "
             "docs/ocean_fidelity/)",
    )
    return p


def _git_sha() -> str:
    try:
        return subprocess.check_output(
            ["git", "-C", str(REPO_ROOT), "rev-parse", "--short", "HEAD"],
            text=True,
        ).strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def main() -> int:
    args = build_parser().parse_args()
    veros_results = _load_veros_results()
    legoesm_results = _load_legoesm_results(args.legoesm_root)
    text = render_report(veros_results, legoesm_results, git_sha=_git_sha())
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(text)
    print(f"wrote: {args.output}")
    print(
        f"veros cases: {len(veros_results)}; "
        f"legoESM cases: {len(legoesm_results)}; "
        f"overlap: {len(set(veros_results) & set(legoesm_results))}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
