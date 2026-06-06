"""Short smoke test (NOT full validation) of explicit_substep vs implicit_cn.

Runs both solvers for N steps (default 50 × 600 s = 8.3 h) from a kicked
rest state, then reports:
  - finiteness of both end-states (must hold)
  - integrated norm of each PyTree leaf (matched by path, not by index)
  - structure-equality check between the two states
  - per-leaf max-abs and RMSE divergence
  - mass-conservation drift if a layer-thickness leaf is found

Acceptance: smoke level only. Both solvers should remain finite. Norm
divergence ≤50% in 8 h means no immediate instability and rules out
gross algorithmic mismatch. Full validation (multi-day Rossby spinup,
energy/mass drift <1e-3) is OUT OF SCOPE for this script and belongs
in a longer regression suite.

Usage:
    PYTHONPATH=. .venv/bin/python scripts/validate_baro_solver.py \\
        --grid latlon --n-lat 64 --n-steps 50
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))  # repo root (scripts/validate/ -> repo)
# Reuse bench-script helpers + JAX-config order (bench_ocean_gpu_scaling -> scripts/bench/)
from scripts.bench.bench_ocean_gpu_scaling import _build_latlon, _build_mpas, _block_state  # noqa


def _path_str(path) -> str:
    """Render a jax.tree path tuple as a stable dotted-key string."""
    parts = []
    for p in path:
        # GetAttrKey, DictKey, SequenceKey, ...
        for attr in ("name", "key", "idx"):
            if hasattr(p, attr):
                parts.append(str(getattr(p, attr)))
                break
        else:
            parts.append(str(p))
    return ".".join(parts) if parts else "<root>"


def _diff(state_a, state_b) -> dict:
    """Path-keyed diff so leaves can't silently get re-ordered.

    Raises ValueError if the two states have differing tree structures.
    """
    import jax, jax.numpy as jnp
    paths_a, _ = zip(*jax.tree_util.tree_flatten_with_path(state_a)[0]) \
        if jax.tree_util.tree_flatten_with_path(state_a)[0] else ((), ())
    flat_a = jax.tree_util.tree_flatten_with_path(state_a)[0]
    flat_b = jax.tree_util.tree_flatten_with_path(state_b)[0]
    if len(flat_a) != len(flat_b):
        raise ValueError(
            f"PyTree leaf-count mismatch: a={len(flat_a)} b={len(flat_b)}"
        )
    out = {}
    for (pa, a), (pb, b) in zip(flat_a, flat_b):
        keya = _path_str(pa)
        keyb = _path_str(pb)
        if keya != keyb:
            raise ValueError(f"PyTree path mismatch: {keya!r} vs {keyb!r}")
        if not (hasattr(a, "dtype") and jnp.issubdtype(a.dtype, jnp.floating)):
            continue
        diff = jnp.abs(a - b)
        max_abs = float(jnp.max(diff))
        rmse = float(jnp.sqrt(jnp.mean(diff * diff)))
        ref_max = float(jnp.max(jnp.abs(a)) + 1e-30)
        out[keya] = (max_abs, rmse, ref_max, max_abs / ref_max)
    return out


def _ocean_mass(state, grid_or_mesh, kind: str):
    """Return ∫(H_bathy + eta) · area · land_mask  [m³].

    LL: state.H_bathy (n_lat, n_lon); grid.area (n_lat, n_lon).
    MPAS: state.bathymetry (nCells,); mesh.areaCell (nCells,).
    """
    import jax.numpy as jnp
    eta = state.eta.data
    if kind == "latlon":
        H = state.H_bathy.data
        mask = state.land_mask.data
        area = grid_or_mesh.area
        col = (H + eta) * area * mask
    else:
        H = getattr(state, "bathymetry", None)
        H = H.data if H is not None else getattr(state, "H_bathy").data
        mask = getattr(state, "land_mask", None)
        mask = mask.data if mask is not None else jnp.ones_like(eta)
        area = grid_or_mesh.areaCell
        col = (H + eta) * area * mask
    return float(jnp.sum(col))


def _step_n(model, state, n: int, dt: float):
    import jax
    @jax.jit
    def _scan_run(s):
        def _body(c, _):
            return model.step(c, dt), None
        return jax.lax.scan(_body, s, None, length=n)[0]
    out = _scan_run(state)
    _block_state(out)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--grid", choices=["latlon", "mpas"], default="latlon")
    p.add_argument("--n-lat", type=int, default=64)
    p.add_argument("--mpas-level", type=int, default=4)
    p.add_argument("--n-steps", type=int, default=50)
    p.add_argument("--dt", type=float, default=600.0)
    p.add_argument("--precision", choices=["float32", "float64"],
                   default="float64")
    p.add_argument("--eta-amp", type=float, default=0.1,
                   help="Eta perturbation amplitude [m]. Default 0.1; "
                        "sweep with multiple invocations to test stability "
                        "margin (e.g. 0.01, 0.1, 1.0, 10.0).")
    args = p.parse_args()

    # Set JAX x64 if needed before any JAX import via build helpers.
    if args.precision == "float64":
        os.environ["JAX_ENABLE_X64"] = "1"

    if args.grid == "latlon":
        key = args.n_lat
        m1, s1, _ = _build_latlon(key, args.precision == "float64",
                                  baro_solver="explicit_substep")
        m2, s2, _ = _build_latlon(key, args.precision == "float64",
                                  baro_solver="implicit_cn")
        gridtag = f"LL{args.n_lat}"
        # m1.grid / m2.grid are the LatLonCGridGeometry built inside the model
        grid_or_mesh_1 = m1.grid
        grid_or_mesh_2 = m2.grid
    else:
        key = args.mpas_level
        m1, s1, _ = _build_mpas(key, args.precision == "float64",
                                baro_solver="explicit_substep")
        m2, s2, _ = _build_mpas(key, args.precision == "float64",
                                baro_solver="implicit_cn")
        gridtag = f"I{args.mpas_level}"
        grid_or_mesh_1 = m1.mesh
        grid_or_mesh_2 = m2.mesh

    # Perturb eta. Rest state has u=v=eta=0 so the barotropic solver
    # would do nothing without a kick.
    import jax, jax.numpy as jnp
    eta = s1.eta.data
    shape = eta.shape
    bump_amp = float(args.eta_amp)  # m
    if eta.ndim == 2:
        # Lat-lon: localized Gaussian bump at mid-grid
        idx = jnp.indices(shape, dtype=eta.dtype)
        centers = [d // 2 for d in shape]
        sigma = max(1.0, min(shape) / 8.0)
        r2 = sum((idx[i] - centers[i]) ** 2 for i in range(len(shape)))
        bump = jnp.exp(-r2 / (2 * sigma * sigma)).astype(eta.dtype) * bump_amp
    else:
        # MPAS unstructured 1D (nCells,): bump a localized contiguous slice.
        bump = jnp.zeros(shape, dtype=eta.dtype)
        n = shape[0]
        i0, i1 = n // 2 - n // 20, n // 2 + n // 20
        bump = bump.at[i0:i1].set(bump_amp)
    s1 = s1._replace(eta=s1.eta.replace(data=eta + bump))
    s2 = s2._replace(eta=s2.eta.replace(data=eta + bump))
    print(f"Initial eta bump: max={float(jnp.max(bump)):.4f} m, "
          f"shape={shape}, nnz={int(jnp.sum(bump > 0))}")

    print(f"\nValidating {gridtag} {args.precision} over {args.n_steps} "
          f"steps × {args.dt}s = {args.n_steps * args.dt / 3600:.1f} h")
    # Mass before stepping (same for both — same kicked init)
    mass0 = _ocean_mass(s1, grid_or_mesh_1, args.grid)
    s1_n = _step_n(m1, s1, args.n_steps, args.dt)
    s2_n = _step_n(m2, s2, args.n_steps, args.dt)
    mass1 = _ocean_mass(s1_n, grid_or_mesh_1, args.grid)
    mass2 = _ocean_mass(s2_n, grid_or_mesh_2, args.grid)
    drift1 = abs(mass1 - mass0) / max(abs(mass0), 1.0)
    drift2 = abs(mass2 - mass0) / max(abs(mass0), 1.0)
    print(f"\nMass conservation (∫(H_bathy + eta)·area·land_mask):")
    print(f"  initial mass  = {mass0:.6e} m³")
    print(f"  explicit final= {mass1:.6e} m³  drift = {drift1:.3e}")
    print(f"  impl_cn final = {mass2:.6e} m³  drift = {drift2:.3e}")
    MASS_DRIFT_TOL = 1e-6  # 1 ppm over 50 baroclinic steps

    import jax.numpy as jnp
    diff = _diff(s1_n, s2_n)
    max_rel = max(v[3] for v in diff.values())
    print(f"{'path':>20}  {'max|Δ|':>12}  {'RMSE|Δ|':>12}  "
          f"{'max|ref|':>12}  {'max-rel':>10}")
    for k, (m, r, rf, rel) in diff.items():
        print(f"{k:>20}  {m:>12.4e}  {r:>12.4e}  {rf:>12.4e}  {rel:>10.4e}")

    # Path-keyed norm comparison.
    def _norm_per_path(state):
        flat = jax.tree_util.tree_flatten_with_path(state)[0]
        ns = {}
        for path, leaf in flat:
            if hasattr(leaf, "dtype") and jnp.issubdtype(leaf.dtype, jnp.floating):
                ns[_path_str(path)] = float(jnp.sqrt(jnp.mean(leaf * leaf)))
        return ns
    n1 = _norm_per_path(s1_n)
    n2 = _norm_per_path(s2_n)
    if set(n1.keys()) != set(n2.keys()):
        print("FAIL: PyTree path mismatch between states")
        return 2
    print()
    print(f"{'path':>20}  {'||exs||':>12}  {'||impcn||':>12}  {'rel-diff':>10}")
    NOISE_FLOOR = 1e-5
    skipped = []
    norm_ratios = []
    for k in sorted(n1):
        a, b = n1[k], n2[k]
        denom = max(a, b, 1e-30)
        rel = abs(a - b) / denom
        if max(a, b) < NOISE_FLOOR:
            skipped.append(k)
            tag = "  (noise-floor; skip)"
        else:
            tag = ""
            norm_ratios.append(rel)
        print(f"{k:>20}  {a:>12.4e}  {b:>12.4e}  {rel:>10.4e}{tag}")
    if skipped:
        print(f"Skipped {len(skipped)} noise-floor leaves: {skipped}")

    finite1 = all(jnp.all(jnp.isfinite(l)) for l in jax.tree_util.tree_leaves(s1_n)
                  if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating))
    finite2 = all(jnp.all(jnp.isfinite(l)) for l in jax.tree_util.tree_leaves(s2_n)
                  if hasattr(l, "dtype") and jnp.issubdtype(l.dtype, jnp.floating))
    max_norm_rel = max(norm_ratios) if norm_ratios else 0.0
    print(f"\nfinite-explicit:{finite1}  finite-impcn:{finite2}")
    print(f"max norm-relative difference: {max_norm_rel:.3e}")
    print(f"max pointwise-relative difference: {max_rel:.3e} "
          "(phase-shift dominated, not a correctness signal for waves)")
    if not (finite1 and finite2):
        print("FAIL: a solver produced non-finite state.")
        return 2
    if max(drift1, drift2) > MASS_DRIFT_TOL:
        print(f"WARN: mass drift > {MASS_DRIFT_TOL:.0e} (ppm); "
              f"explicit={drift1:.3e}, impl_cn={drift2:.3e}")
        return 1
    if max_norm_rel > 0.5:
        print("WARN: integrated norms diverge by >50% — investigate.")
        return 1
    print("SMOKE TEST OK: both solvers stable, no gross structural "
          "mismatch, norms within 50%. This is NOT full validation — "
          "multi-day energy/mass conservation regression is out of scope.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
