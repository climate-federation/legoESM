"""Scale-resolved twin predictability of the SWG double-gyre truth.

For fine-truth runs started from nearly identical states, the coherence
C(k, t) = sum(a*b) / sqrt(sum(a^2) sum(b^2)) over each isotropic band k of the DCT-II spectrum of
the symmetric objective's coarse (dg50) vorticity (box-averaged + Gaussian, as the training loss
sees it), control vs each perturbed twin.  The cutoff k_c(t) is the largest band with C >= threshold;
the length scale is 2*N*dx/k; bands k <= N-1 only (wavelength of DCT-II index k over N interior vertices of spacing dx).

usage: swg_twin_scale_predictability.py <control_dir> <perturbed_dir> [<perturbed_dir> ...] --out DIR
(JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 and the packages on PYTHONPATH.)
"""
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import io
import json
from pathlib import Path

import numpy as np
from scipy.fft import dctn, dstn

THRESHOLDS = (0.5, 0.8)
HERE = Path(__file__).resolve().parent


def band_index(n_y: int, n_x: int) -> np.ndarray:
    """Isotropic band of each (m, n) transform index: round(hypot(m, n))."""
    m, n = np.meshgrid(np.arange(n_y), np.arange(n_x), indexing="ij")
    return np.rint(np.hypot(m, n)).astype(int)


def band_coherence(a: np.ndarray, b: np.ndarray, basis: str = "dct") -> np.ndarray:
    """C[..., k] for fields of shape (..., ny, nx); ``dct`` = type II (the requested Neumann basis),
    ``dst`` = type I (the Dirichlet basis the model's own balance solve uses; sensitivity only)."""
    transform = {"dct": lambda x: dctn(x, type=2, axes=(-2, -1), norm="ortho"),
                 "dst": lambda x: dstn(x, type=1, axes=(-2, -1), norm="ortho")}[basis]
    band = band_index(a.shape[-2], a.shape[-1]).ravel()
    nb = int(band.max()) + 1
    fa = transform(a).reshape(-1, band.size)
    fb = transform(b).reshape(-1, band.size)

    def pooled(x):
        return np.stack([np.bincount(band, weights=row, minlength=nb) for row in x])

    cross, pa, pb = pooled(fa * fb), pooled(fa * fa), pooled(fb * fb)
    with np.errstate(invalid="ignore", divide="ignore"):
        coh = cross / np.sqrt(pa * pb)
    return coh.reshape(*a.shape[:-2], nb)


def cutoff(coherence: np.ndarray, threshold: float, contiguous: bool = False) -> np.ndarray:
    """Largest band with C >= threshold; ``contiguous``: the band before the first failure."""
    ok = coherence >= threshold
    if contiguous:
        ok = np.cumprod(ok, axis=-1).astype(bool)
    return np.where(ok, np.arange(coherence.shape[-1]), -1).max(axis=-1)


def length_km(k, n_points: int, dx_m: float):
    """Wavelength 2*N*dx/k in km (nan for k <= 0)."""
    k = np.asarray(k, dtype=float)
    with np.errstate(divide="ignore", invalid="ignore"):
        return np.where(k > 0, 2.0 * n_points * dx_m / k / 1e3, np.nan)


def _load(path: Path):
    spec = importlib.util.spec_from_file_location(path.stem, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def coarse_vorticity(root: Path, days, restrict, geometry, balanced_fields):
    """Symmetric-objective coarse vorticity on the interior F points, (T, ny, nx); sha256-checked."""
    pins = json.loads((root / "manifest.json").read_text())["daily_sha256"]
    j0, j1, i0, i1 = geometry.wet_bounds
    out = []
    for day in days:
        name = f"day{day:04d}.npz"
        payload = (root / name).read_bytes()
        if hashlib.sha256(payload).hexdigest() != pins[name]:
            raise ValueError(f"{root / name}: digest differs from manifest")
        with np.load(io.BytesIO(payload), allow_pickle=False) as z:
            if int(z["day"]) != day:
                raise ValueError(f"{name}: embedded day differs")
            c = restrict({"T": z["ssh_m"], "U": z["u_m_s"], "V": z["v_m_s"]})
        zeta = balanced_fields(c["U"], c["V"], geometry, objective="symmetric").relative_vorticity_s1
        out.append(np.asarray(zeta)[j0:j1, i0:i1])
    return np.stack(out)


def summarize(coh: np.ndarray, days, n_points: int, dx_m: float) -> dict:
    """coh: (pairs, T, K).  Mean over pairs, spread = min/max over pairs, cutoffs from the mean."""
    coh = coh[..., :n_points]  # full quarter-rings only (k <= N-1); the corner bands hold a few coefficients
    mean = np.nanmean(coh, axis=0)
    out = {"n_pairs": int(coh.shape[0]), "days": [int(d) for d in days]}
    for thr in THRESHOLDS:
        for mode, contiguous in (("largest", False), ("contiguous", True)):
            kc = cutoff(mean, thr, contiguous)
            per_pair = cutoff(coh, thr, contiguous)
            out[f"kc_{mode}_thr{thr}"] = kc.tolist()
            out[f"kc_{mode}_thr{thr}_pairs"] = per_pair.tolist()
            out[f"length_km_{mode}_thr{thr}"] = length_km(kc, n_points, dx_m).tolist()
    out["coherence_mean"] = mean.tolist()
    out["coherence_min"] = np.nanmin(coh, axis=0).tolist()
    out["coherence_max"] = np.nanmax(coh, axis=0).tolist()
    return out


def plot(summary: dict, path: Path, title: str):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    days = np.array(summary["days"])
    mean = np.array(summary["coherence_mean"])
    fig, ax = plt.subplots(figsize=(7, 4.5))
    im = ax.pcolormesh(days, np.arange(mean.shape[1]), mean.T, vmin=0, vmax=1, shading="nearest", cmap="viridis")
    for thr, colour in zip(THRESHOLDS, ("white", "red")):
        ax.plot(days, summary[f"kc_largest_thr{thr}"], colour, lw=1.5, label=f"k_c, C>={thr} (largest band)")
        ax.plot(days, summary[f"kc_contiguous_thr{thr}"], colour, lw=1.0, ls="--", label=f"C>={thr} (contiguous)")
    ax.set_xlabel("lead (days)")
    ax.set_ylabel("DCT-II isotropic band k")
    ax.set_title(title)
    ax.legend(fontsize=7, loc="upper right")
    fig.colorbar(im, label="twin coherence C(k, t)")
    fig.tight_layout()
    fig.savefig(path, dpi=150)


def main():
    import jax

    ap = argparse.ArgumentParser()
    ap.add_argument("control")
    ap.add_argument("perturbed", nargs="+")
    ap.add_argument("--out", required=True)
    ap.add_argument("--days", type=int, default=120)
    args = ap.parse_args()
    if not jax.config.jax_enable_x64:
        raise SystemExit("JAX_ENABLE_X64=1 required")
    drv = _load(HERE.parents[2] / "run" / "run_swg_online_training.py")
    ev = _load(HERE / "nemo_testcase_l1_swg_online_evaluation.py")
    from legoesm.training.swg_online import balanced_fields_from_native, build_closure_geometry

    drv.select_device("cpu", "libm", None)
    geometry = build_closure_geometry(drv.build_deck_card("dg50", "explicit"))
    restrict = ev.truth_restrictor("symmetric")
    days = range(1, args.days + 1)
    zc = coarse_vorticity(Path(args.control), days, restrict, geometry, balanced_fields_from_native)
    n_points = zc.shape[-1]
    if zc.shape[-2] != n_points:
        raise SystemExit("square interior expected")
    mean_c = zc.mean(axis=0)
    coh = {"full": [], "anom": [], "dst": []}
    for p in args.perturbed:
        zp = coarse_vorticity(Path(p), days, restrict, geometry, balanced_fields_from_native)
        coh["full"].append(band_coherence(zc, zp))
        coh["dst"].append(band_coherence(zc, zp, "dst"))
        coh["anom"].append(band_coherence(zc - mean_c, zp - mean_c))
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    np.savez(out / "coherence.npz", days=np.array(days), **{k: np.stack(v) for k, v in coh.items()})
    result = {"control": args.control, "perturbed": args.perturbed, "n_points": n_points,
              "dx_m": geometry.spacing_m, "wet_bounds": list(geometry.wet_bounds),
              "length_conversion": "lambda = 2*N*dx/k, N=n_points interior vertices (DCT-II index k)"}
    for name, arr in coh.items():
        s = summarize(np.stack(arr), days, n_points, geometry.spacing_m)
        result[name] = s
        plot(s, out / f"twin_coherence_{name}.png", f"twin coherence ({name}), N={s['n_pairs']} pairs")
    json.dump(result, open(out / "twin_scale_predictability.json", "w"))


if __name__ == "__main__":
    main()
