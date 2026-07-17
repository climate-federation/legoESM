#!/usr/bin/env python3
"""Visual-regression gate for cubed-sphere grid artifacts (v-wind imprint).

CLAUDE.md is explicit that passing tests + improving norms are NECESSARY but NOT
SUFFICIENT for cubed-sphere ops: edge artifacts / cube imprint / grid-scale noise
are only caught by *looking* at fields (the W2 v-wind, W5 wind-speed). This script
turns that visual check into a numeric, CI-runnable regression gate:

  1. Run a short Williamson-2 (steady geostrophic) shallow-water sim on the
     cubed sphere — reusing the canonical ``tests/williamson_diagnostic.py``
     harness so the dycore / initial-condition numerics are NOT duplicated.
  2. Extract the cell-centre meridional wind ``v`` (a sensitive imprint probe).
  3. Compare it to a committed reference with three complementary metrics:
       * SSIM (structural similarity)  — global structure drift,
       * perceptual hash (Hamming)     — coarse pattern drift,
       * cube edge-artifact ratio      — boundary-vs-interior concentration
         (reused from the diagnostic harness),
     and FAIL if any exceeds tolerance. Norms can improve while the imprint
     worsens; these metrics catch that.

Usage::

    # write/refresh the committed reference (after an intentional change):
    python scripts/validate/visual_regression.py --generate
    # check the working tree against the reference (CI):
    python scripts/validate/visual_regression.py --check

The reference is a tiny numeric fixture (a (6,n,n) float array + a JSON manifest
of metadata + perceptual hash), NOT a PNG — deliberately small enough to track.

The pure metric functions (``ssim``/``phash``/``hamming``) are unit-tested in
``tests/test_visual_regression_metrics.py``; the full sim runs in a nightly CI job.
"""
from __future__ import annotations

import argparse
import json
import pathlib
import sys

import numpy as np

# --- tunables (gate tolerances; not physical constants) ---------------------
SSIM_MIN = 0.985            # below this => structural regression
HAMMING_MAX = 4             # max differing perceptual-hash bits (out of 64)
EDGE_RATIO_MAX_FACTOR = 1.25  # edge/interior ratio may not worsen by >25% vs ref
_PHASH_GRID = 8             # 8x8 -> 64-bit hash per panel
_SSIM_C1, _SSIM_C2 = 0.01, 0.03  # standard SSIM stabiliser fractions of range

_BASELINE_DIR = pathlib.Path(__file__).resolve().parents[2] / "tests" / "visual_baselines"
_FIELD_NPY = _BASELINE_DIR / "sw_cube_vwind.npy"
_MANIFEST = _BASELINE_DIR / "sw_cube_vwind.json"


# ---------------------------------------------------------------------------
# Pure metrics (unit-tested)
# ---------------------------------------------------------------------------
def _panels(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    return x.reshape(-1, x.shape[-2], x.shape[-1]) if x.ndim >= 3 else x[None]


def ssim(a: np.ndarray, b: np.ndarray) -> float:
    """Per-panel SSIM, returned as the WORST (min) panel score — so a localized
    cube-edge imprint on one panel is not diluted by averaging across panels.
    Returns 1.0 for identical inputs. Constant panels are handled explicitly
    (the SSIM luminance/contrast terms are otherwise ill-conditioned)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    data_range = float(max(a.max() - a.min(), b.max() - b.min(), 1e-12))
    c1 = (_SSIM_C1 * data_range) ** 2
    c2 = (_SSIM_C2 * data_range) ** 2
    vals = []
    for pa, pb in zip(_panels(a), _panels(b)):
        mu_x, mu_y = pa.mean(), pb.mean()
        var_x, var_y = pa.var(), pb.var()
        if var_x < 1e-20 and var_y < 1e-20:  # both ~constant: pure luminance term
            vals.append((2 * mu_x * mu_y + c1) / (mu_x ** 2 + mu_y ** 2 + c1))
            continue
        cov = ((pa - mu_x) * (pb - mu_y)).mean()
        num = (2 * mu_x * mu_y + c1) * (2 * cov + c2)
        den = (mu_x ** 2 + mu_y ** 2 + c1) * (var_x + var_y + c2)
        vals.append(num / den)
    return float(np.min(vals))  # worst panel, not the average


def phash(field: np.ndarray, grid: int = _PHASH_GRID) -> list[int]:
    """Per-panel perceptual hash (list of ints, one per panel). For each panel:
    downsample to ``grid x grid`` (block-mean), set a bit where a cell exceeds the
    panel median by a small deadband (so tiny drift near the median does not flip
    bits). Per-panel (not XOR-folded) so symmetric imprints across panels cannot
    cancel each other out."""
    out: list[int] = []
    for p in _panels(field):
        ds = _block_mean(p, grid)
        spread = float(ds.std())
        thresh = float(np.median(ds)) + 1e-3 * max(spread, 1e-12)  # deadband
        bits = (ds > thresh).astype(np.uint8).flatten()
        val = 0
        for bit in bits:
            val = (val << 1) | int(bit)
        out.append(val)
    return out


def _block_mean(p: np.ndarray, grid: int) -> np.ndarray:
    """Resize 2-D ``p`` to ``g x g`` (g = min(grid, ny, nx)) by block averaging —
    no scipy, no aliasing from grid finer than the field."""
    ny, nx = p.shape
    g = min(grid, ny, nx)
    ys = np.linspace(0, ny, g + 1).astype(int)
    xs = np.linspace(0, nx, g + 1).astype(int)
    out = np.zeros((g, g))
    for i in range(g):
        for j in range(g):
            block = p[ys[i]:ys[i + 1], xs[j]:xs[j + 1]]
            out[i, j] = block.mean() if block.size else 0.0
    return out


def hamming(h1, h2) -> int:
    """Differing bits between two perceptual hashes. Accepts single ints or
    per-panel lists (returns the WORST single-panel Hamming distance)."""
    if isinstance(h1, (list, tuple)) or isinstance(h2, (list, tuple)):
        pairs = list(zip(h1, h2))
        if len(h1) != len(h2):
            return 64  # structural mismatch -> max distance
        return max((bin(a ^ b).count("1") for a, b in pairs), default=0)
    return bin(h1 ^ h2).count("1")


# ---------------------------------------------------------------------------
# Field production (reuses the canonical diagnostic harness)
# ---------------------------------------------------------------------------
def produce_vwind_field(n_res: int = 8, n_steps: int = 5, dt: float = 600.0):
    """Run a short Williamson-2 cube SW sim and return (v_cc_field, edge_ratio)."""
    repo = pathlib.Path(__file__).resolve().parents[2]
    if str(repo) not in sys.path:
        sys.path.insert(0, str(repo))
    import jax.numpy as jnp
    from tests.williamson_diagnostic import (  # canonical harness — no dup numerics
        create_cubed_sphere,
        create_cubed_sphere_cdgrid,
        cube_edge_artifact_ratio,
        integrate,
        williamson2_ic,
    )
    from legoesm.atmosphere.dynamics.gcm.shallow_water_fv3_cdgrid import (
        CDGridShallowWaterConfig,
        CDGridShallowWaterModel,
    )
    from legoesm.core.operators_cdgrid import dgrid_to_center_vector

    grid = create_cubed_sphere(n_res)
    cdgrid = create_cubed_sphere_cdgrid(grid)
    dx_min = float(jnp.min(grid.dx))
    config = CDGridShallowWaterConfig(
        A_h=0.0,
        hyperdiff_coeff=dx_min ** 4 / (86400.0 * 30.0),
        div_damp=0.0,
        use_conservation_fixer=True,
        fix_mass=True,
    )
    model = CDGridShallowWaterModel(grid, config)
    state0 = williamson2_ic(cdgrid)
    model.set_initial_mass(state0)
    state, stable = integrate(model, state0, dt, n_steps)
    if not stable:
        raise RuntimeError("cube SW sim went unstable — cannot produce a field")
    _u_cc, v_cc = dgrid_to_center_vector(state.u_d, state.v_d)
    v = np.asarray(v_cc, dtype=np.float64)
    # ``v`` is the cube-local cell-centre meridional wind: a deterministic
    # *regression fingerprint* of the W2 flow (not analytically zero in this
    # basis). ``edge_ratio`` measures how concentrated grid-scale structure is at
    # panel boundaries vs the interior — the quantitative proxy for cube imprint.
    edge_ratio = float(cube_edge_artifact_ratio(np.abs(v), n_res))
    return v, edge_ratio


def _generate(n_res: int, n_steps: int, dt: float) -> None:
    field, edge_ratio = produce_vwind_field(n_res, n_steps, dt)
    _BASELINE_DIR.mkdir(parents=True, exist_ok=True)
    np.save(_FIELD_NPY, field.astype(np.float32))
    _MANIFEST.write_text(
        json.dumps(
            {
                "case": "williamson2_cubed_sphere_vwind",
                "n_res": n_res, "n_steps": n_steps, "dt": dt,
                "shape": list(field.shape),
                "phash": phash(field),
                "edge_ratio": edge_ratio,
                "note": "tiny numeric reference for visual_regression.py --check",
            },
            indent=2,
        )
    )
    print(f"wrote baseline {_FIELD_NPY} and {_MANIFEST} (edge_ratio={edge_ratio:.3f})")


def _check(n_res: int, n_steps: int, dt: float) -> int:
    if not _FIELD_NPY.exists() or not _MANIFEST.exists():
        print(f"ERROR: no baseline at {_BASELINE_DIR} — run with --generate first", file=sys.stderr)
        return 2
    ref = np.load(_FIELD_NPY).astype(np.float64)
    manifest = json.loads(_MANIFEST.read_text())
    # The baseline is parameter-specific — comparing against a ref built with
    # different (n_res, n_steps, dt) is meaningless. Fail clearly (config error).
    for key, val in (("n_res", n_res), ("n_steps", n_steps), ("dt", dt)):
        if key in manifest and manifest[key] != val:
            print(
                f"ERROR: --{key.replace('_', '-')}={val} != baseline {manifest[key]}; "
                f"regenerate the baseline or match its parameters.",
                file=sys.stderr,
            )
            return 2
    field, edge_ratio = produce_vwind_field(n_res, n_steps, dt)
    if field.shape != ref.shape:
        print(f"FAIL: field shape {field.shape} != reference {ref.shape}", file=sys.stderr)
        return 1
    s = ssim(field, ref)
    ref_phash = manifest["phash"]
    if not isinstance(ref_phash, list):  # tolerate an older scalar-hash baseline
        ref_phash = [ref_phash]
    ham = hamming(phash(field), ref_phash)
    ref_ratio = float(manifest.get("edge_ratio", edge_ratio))
    ratio_ok = edge_ratio <= max(ref_ratio * EDGE_RATIO_MAX_FACTOR, ref_ratio + 0.05)
    print(f"SSIM={s:.4f} (min {SSIM_MIN})  hamming={ham} (max {HAMMING_MAX})  "
          f"edge_ratio={edge_ratio:.3f} vs ref {ref_ratio:.3f}")
    fails = []
    if s < SSIM_MIN:
        fails.append(f"SSIM {s:.4f} < {SSIM_MIN}")
    if ham > HAMMING_MAX:
        fails.append(f"perceptual-hash hamming {ham} > {HAMMING_MAX}")
    if not ratio_ok:
        fails.append(f"cube edge-artifact ratio {edge_ratio:.3f} worsened vs {ref_ratio:.3f}")
    if fails:
        print("VISUAL REGRESSION: " + "; ".join(fails), file=sys.stderr)
        print("Inspect the W2 v-wind field on the cubed sphere — a grid/edge "
              "imprint likely worsened. If the change is intentional, regenerate "
              "the baseline with --generate.", file=sys.stderr)
        return 1
    print("visual-regression OK")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--generate", action="store_true", help="write/refresh the reference")
    ap.add_argument("--check", action="store_true", help="check working tree vs reference")
    ap.add_argument("--n-res", type=int, default=8)
    ap.add_argument("--n-steps", type=int, default=5)
    ap.add_argument("--dt", type=float, default=600.0)
    args = ap.parse_args(argv)
    if args.generate == args.check:
        ap.error("pass exactly one of --generate / --check")
    if args.generate:
        _generate(args.n_res, args.n_steps, args.dt)
        return 0
    return _check(args.n_res, args.n_steps, args.dt)


if __name__ == "__main__":
    raise SystemExit(main())
