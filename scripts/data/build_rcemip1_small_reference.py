#!/usr/bin/env python
"""Build an RCEMIP1-small (Wing et al. 2018) reference bundle for the SCM RCE campaign.

Context
-------
The single-column RCE intercomparison drivers
(``scripts/run/run_scm_rce_campaign.py`` and
``scripts/run/run_scm_rce_convection_tuning.py``) score each SCM equilibrium
against a resolved plane-CRM RCEMIP reference stored under
``results/rcemip1_*_ocean`` as ``snapshots3d/vol_*.npz`` +
``snapshots/sfc_*.npz``.  When that resolved-CRM dataset is unavailable (it is
a multi-week, convection-permitting ~1 km / 128^2 run that is only feasible on
GPU/cluster hardware), this script writes a **faithful analytic stand-in** for
the RCEMIP1 *small* ocean case (fixed SST 300 K) directly from the repo's own
Wing-2018 RCEMIP profiles.

RCEMIP (Wing et al. 2018, GMD 11:793; results in Wing et al. 2020, JAMES) is
designed so resolved CRMs relax *toward* this analytic mean state, so the Wing
temperature and humidity profiles ARE the community RCEMIP1 reference for T and
q_v.  The tropopause cold point is ``T_v0 - Gamma*z_t = 295 - 0.0067*15000 =
194.5 K`` at 15 km, isothermal above — the proper RCEMIP1 tropopause.

Provenance and caveats (written into the bundle ``README``):
- ``T``, ``q_v``: Wing-2018 analytic profiles (``rcemip_initial_conditions``),
  exact.  ``q_v`` is encoded through ``mse`` exactly as the campaign reader
  inverts it (``q_v = (1000*mse - c_pd*T - g*z)/L_v``).
- Total condensate: a smooth RCEMIP-representative bimodal profile (warm-cloud
  bump near cloud base + anvil bump near the tropopause), NOT a resolved-run
  diagnostic.  The campaign cloud-RMSE term against this reference is therefore
  representative, not exact.
- Surface precipitation: RCEMIP1-small 300 K domain-mean ~3.2 mm/day
  (Wing et al. 2020), a single documented scalar.
- ``w`` is zero (a steady analytic mean state carries no resolved vertical
  velocity); the optional ``crm_clear_sky_subsidence`` forcing is therefore
  unavailable against this reference and must not be requested.

This is a *reference-input* generator (analytic truth), not a model path — it
lives in ``scripts/data`` next to the other forcing/IC builders and never
mutates any production config.

Run:

    JAX_PLATFORMS=cpu JAX_ENABLE_X64=1 .venv/bin/python \
        scripts/data/build_rcemip1_small_reference.py \
        --output results/rcemip1_small_wing_ocean
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np


REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

MSE_J_TO_KJ = 1.0e-3
SECONDS_PER_DAY = 86_400.0

# --- RCEMIP1 small ocean case (Wing et al. 2018 Tab 1: RCE_small300) --- #
DEFAULT_SST_K = 300.0
DEFAULT_N_LEVELS = 60
DEFAULT_Z_TOP_M = 33_000.0          # RCEMIP1 model top
DEFAULT_DZ_SFC_M = 100.0            # near-surface layer thickness (stretched)
DEFAULT_N_HORIZONTAL = 4           # tiny tile; the analytic mean is uniform
DEFAULT_N_VOLUMES = 5              # >= campaign --last-reference-files default

# RCEMIP1-small 300 K domain-mean surface precipitation [mm/day]
# (Wing et al. 2020, JAMES 12:e2020MS002138 — ~3-4 mm/day across models).
RCEMIP1_SMALL_PRECIP_MM_DAY = 3.2

# --- RCEMIP-representative total-condensate profile [kg/kg] --- #
# Bimodal: a shallow warm-cloud bump near cloud base and an anvil/ice bump
# near the tropopause. Magnitudes are RCEMIP-representative (peak ~0.02 g/kg),
# NOT a resolved-run diagnostic (documented in the bundle README).
_COND_WARM_PEAK_KG_KG = 1.2e-5
_COND_WARM_CENTER_M = 2_000.0
_COND_WARM_WIDTH_M = 1_500.0
_COND_ANVIL_PEAK_KG_KG = 2.0e-5
_COND_ANVIL_CENTER_M = 12_500.0
_COND_ANVIL_WIDTH_M = 2_000.0


def stretched_heights_top_to_surface(
    n_levels: int,
    z_top_m: float,
    dz_sfc_m: float,
) -> np.ndarray:
    """Geometrically stretched full-level heights, ordered top -> surface.

    A geometric layer-thickness stretch from ``dz_sfc_m`` at the surface up to
    the model top reproduces the RCEMIP-style grid (fine near the surface,
    coarse near the top). Returned strictly decreasing (top first) to match the
    plane-CRM ``vol_*.npz`` convention the campaign reader assumes.
    """
    if n_levels < 4:
        raise ValueError(f"n_levels must be >= 4, got {n_levels}")
    if not (0.0 < dz_sfc_m < z_top_m):
        raise ValueError(
            f"require 0 < dz_sfc_m ({dz_sfc_m}) < z_top_m ({z_top_m})"
        )
    # Solve geometric ratio r for sum_{k=0}^{n-1} dz_sfc*r^k = z_top.
    ratios = np.linspace(1.0, 1.20, 4001)
    powers = np.arange(n_levels)
    totals = dz_sfc_m * np.sum(ratios[:, None] ** powers[None, :], axis=1)
    r = float(ratios[int(np.argmin(np.abs(totals - z_top_m)))])
    dz = dz_sfc_m * r ** powers
    z_half = np.concatenate([[0.0], np.cumsum(dz)])
    z_half = z_half / z_half[-1] * z_top_m
    z_full = 0.5 * (z_half[:-1] + z_half[1:])
    return z_full[::-1].copy()  # top -> surface


def representative_condensate_kg_kg(z_m: np.ndarray) -> np.ndarray:
    """RCEMIP-representative total-condensate profile (documented stand-in)."""
    warm = _COND_WARM_PEAK_KG_KG * np.exp(
        -((z_m - _COND_WARM_CENTER_M) / _COND_WARM_WIDTH_M) ** 2
    )
    anvil = _COND_ANVIL_PEAK_KG_KG * np.exp(
        -((z_m - _COND_ANVIL_CENTER_M) / _COND_ANVIL_WIDTH_M) ** 2
    )
    return warm + anvil


def build_reference(
    output: Path,
    *,
    sst_k: float = DEFAULT_SST_K,
    n_levels: int = DEFAULT_N_LEVELS,
    z_top_m: float = DEFAULT_Z_TOP_M,
    dz_sfc_m: float = DEFAULT_DZ_SFC_M,
    n_horizontal: int = DEFAULT_N_HORIZONTAL,
    n_volumes: int = DEFAULT_N_VOLUMES,
) -> dict[str, float]:
    """Write the RCEMIP1-small Wing reference bundle under ``output``.

    Returns a small dict of scalar diagnostics (cold point, surface T/q_v)
    for logging and testing.
    """
    import jax.numpy as jnp

    from legoesm import constants
    from legoesm.atmosphere.idealized.rcemip_initial_conditions import (
        WING_P_SFC,
        wing2018_qv_profile,
        wing2018_temperature_profile,
    )

    if sst_k != DEFAULT_SST_K:
        # The Wing q_sfc default is the RCE300 case; other SSTs need the
        # matching q_sfc (Wing Tab 1) which we do not encode here.
        raise ValueError(
            "This builder targets the RCEMIP1 small RCE300 (SST=300 K) case; "
            f"got sst_k={sst_k}."
        )

    z_m = stretched_heights_top_to_surface(n_levels, z_top_m, dz_sfc_m)
    z_j = jnp.asarray(z_m, dtype=jnp.float64)
    T = np.asarray(wing2018_temperature_profile(z_j), dtype=float)
    qv = np.asarray(wing2018_qv_profile(z_j), dtype=float)
    cond = representative_condensate_kg_kg(z_m)
    # Encode q_v through mse EXACTLY as the campaign reader inverts it.
    mse_kj = (
        constants.c_pd * T + constants.g * z_m + constants.L_v * qv
    ) * MSE_J_TO_KJ

    vol_dir = output / "snapshots3d"
    sfc_dir = output / "snapshots"
    vol_dir.mkdir(parents=True, exist_ok=True)
    sfc_dir.mkdir(parents=True, exist_ok=True)

    def tile(profile: np.ndarray) -> np.ndarray:
        return np.broadcast_to(
            profile, (n_horizontal, n_horizontal, profile.size)
        ).copy()

    T_vol = tile(T).astype(np.float32)
    mse_vol = tile(mse_kj).astype(np.float32)
    cond_vol = tile(cond).astype(np.float32)
    w_vol = np.zeros_like(T_vol)

    steps_per_day = 4_320  # nominal; only the day ORDER matters to the reader
    for i in range(n_volumes):
        step = (i + 1) * steps_per_day
        day = float(i + 1)
        np.savez(
            vol_dir / f"vol_{step:08d}.npz",
            day=day,
            t_s=day * SECONDS_PER_DAY,
            z=z_m.astype(np.float32),
            T=T_vol,
            mse=mse_vol,
            cond=cond_vol,
            w=w_vol,
        )
        np.savez(
            sfc_dir / f"sfc_{step:08d}.npz",
            day=day,
            t_s=day * SECONDS_PER_DAY,
            precip=np.full(
                (n_horizontal, n_horizontal),
                RCEMIP1_SMALL_PRECIP_MM_DAY,
                dtype=float,
            ),
        )

    # Wing T is isothermal above the tropopause (T = T_v0 - Gamma*z_t from
    # z_t up), so the minimum is a flat plateau; report the tropopause = the
    # LOWEST-altitude level at that minimum, not the model top.
    T_min = float(np.min(T))
    at_min = np.nonzero(T <= T_min + 0.1)[0]
    cold_idx = int(at_min[np.argmin(z_m[at_min])])
    diagnostics = {
        "n_levels": float(n_levels),
        "z_top_m": float(z_m[0]),
        "z_sfc_m": float(z_m[-1]),
        "sfc_T_K": float(T[-1]),
        "sfc_qv_kg_kg": float(qv[-1]),
        "cold_point_T_K": float(T[cold_idx]),
        "cold_point_z_km": float(z_m[cold_idx] / 1_000.0),
        "precip_mm_day": RCEMIP1_SMALL_PRECIP_MM_DAY,
        "p_sfc_Pa": float(WING_P_SFC),
    }

    (output / "README.md").write_text(
        _readme_text(diagnostics, n_volumes, n_horizontal)
    )
    return diagnostics


def _readme_text(diag: dict[str, float], n_volumes: int, nh: int) -> str:
    return (
        "# RCEMIP1-small (Wing 2018) analytic reference\n\n"
        "Analytic stand-in for the resolved plane-CRM RCEMIP1 *small* ocean "
        "RCE (SST 300 K), written for `run_scm_rce_campaign.py` / "
        "`run_scm_rce_convection_tuning.py` when the resolved-CRM dataset "
        "(a ~1 km / 128^2 GPU/cluster run) is unavailable.\n\n"
        "- `T`, `q_v` (via `mse`): Wing-2018 analytic profiles, exact. "
        f"Cold point {diag['cold_point_T_K']:.1f} K at "
        f"{diag['cold_point_z_km']:.1f} km; surface {diag['sfc_T_K']:.1f} K, "
        f"q_v {diag['sfc_qv_kg_kg']*1e3:.2f} g/kg.\n"
        "- Total condensate: RCEMIP-representative bimodal profile "
        "(warm-cloud + anvil), NOT a resolved diagnostic — the cloud-RMSE "
        "term is representative, not exact.\n"
        f"- Surface precip: {diag['precip_mm_day']:.1f} mm/day "
        "(RCEMIP1-small 300 K domain mean, Wing et al. 2020).\n"
        f"- `w` = 0 (steady analytic mean); do NOT request the "
        "`crm_clear_sky_subsidence` large-scale forcing against this "
        "reference.\n\n"
        f"{n_volumes} identical daily volumes on a {nh}x{nh} tile "
        f"({int(diag['n_levels'])} levels, top {diag['z_top_m']/1e3:.1f} km).\n"
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("results/rcemip1_small_wing_ocean"),
    )
    parser.add_argument("--n-levels", type=int, default=DEFAULT_N_LEVELS)
    parser.add_argument("--z-top-m", type=float, default=DEFAULT_Z_TOP_M)
    parser.add_argument("--dz-sfc-m", type=float, default=DEFAULT_DZ_SFC_M)
    parser.add_argument("--n-volumes", type=int, default=DEFAULT_N_VOLUMES)
    parser.add_argument(
        "--n-horizontal", type=int, default=DEFAULT_N_HORIZONTAL,
    )
    args = parser.parse_args(argv)
    diag = build_reference(
        args.output,
        n_levels=args.n_levels,
        z_top_m=args.z_top_m,
        dz_sfc_m=args.dz_sfc_m,
        n_horizontal=args.n_horizontal,
        n_volumes=args.n_volumes,
    )
    print(f"[reference] wrote {args.output}")
    for key, value in diag.items():
        print(f"    {key} = {value:g}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
