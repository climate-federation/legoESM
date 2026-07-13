"""
CLM-ML-JAX canopy validation against Fortran CLM-ML-v2 reference outputs.

Reference:  docs/output_files_clm_ml-v2/CHATS7_2007-05_*.out
JAX output: clm-ml-jax/src/output_files/JAX_outputs_05_2007_31days/CHATS7_2007-05_*.out

Site:  CHATS7 (Canopy Horizontal Array Turbulence Study, walnut orchard)
Dates: May 2007 (31 days, 1488 half-hourly timesteps)
"""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np
from scipy import stats

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
REPO_ROOT = Path(__file__).resolve().parent.parent.parent

# Bootstrap the federation package roots so the shared evaluation library is
# importable when this validator is run as a script (mirrors the runners).
_pkg_root = REPO_ROOT / "packages"
for _p in [REPO_ROOT / "src", *sorted(
    p for p in _pkg_root.iterdir() if p.is_dir() and (p / "legoesm").exists()
)]:
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

# Shared, tested comparison numerics + .out schema (single source of truth;
# CLAUDE.md: "No duplicate numerics ... Plotters NOT exempt").
from legoesm.land.evaluation import fluxio as _eval_fluxio  # noqa: E402
from legoesm.land.evaluation import metrics as _eval_metrics  # noqa: E402
REF_DIR = REPO_ROOT / "docs" / "output_files_clm_ml-v2"
JAX_DIR = REPO_ROOT / "clm-ml-jax" / "src" / "output_files" / "JAX_outputs_05_2007_31days"
OUT_DIR = REPO_ROOT / "validation_output"
OUT_DIR.mkdir(exist_ok=True)

if not REF_DIR.exists():
    raise FileNotFoundError(
        f"Reference output directory not found: {REF_DIR}\n"
        "Run the Fortran CLM-ML v2 standalone first and copy outputs here."
    )

MONTH_TAG = "CHATS7_2007-05"

# ---------------------------------------------------------------------------
# Variable definitions (from output() function in CLMml_driver.py)
# ---------------------------------------------------------------------------

FLUX_VARS = [
    ("time",         "Julian day",           "day"),
    ("rnet",         "Net radiation",         "W m⁻²"),
    ("stflx_air",    "Air heat storage",      "W m⁻²"),
    ("shflx",        "Sensible heat",         "W m⁻²"),
    ("lhflx",        "Latent heat",           "W m⁻²"),
    ("gpp",          "GPP",                   "µmol m⁻² s⁻¹"),
    ("ustar",        "Friction velocity",     "m s⁻¹"),
    ("swup",         "SW upwelling",          "W m⁻²"),
    ("lwup",         "LW upwelling",          "W m⁻²"),
    ("tair_top",     "Air T (canopy top)",    "K"),
    ("G_soil",       "Ground heat flux",      "W m⁻²"),
    ("rn_soil",      "Net radiation (soil)",  "W m⁻²"),
    ("sh_soil",      "Sensible heat (soil)",  "W m⁻²"),
    ("lh_soil",      "Latent heat (soil)",    "W m⁻²"),
    ("lh_trans",     "Transpiration LH",      "W m⁻²"),
    ("lh_evap",      "Evaporation LH",        "W m⁻²"),
    ("beta",         "Soil moisture stress",  "—"),
    ("stflx_veg",    "Veg heat storage",      "W m⁻²"),
]

FSUN_VARS = [
    ("zen",          "Solar zenith",          "deg"),
    ("sw_vis",       "SW VIS",                "W m⁻²"),
    ("pai",          "PAI (LAI+SAI)",         "m² m⁻²"),
    ("laisun",       "LAI sunlit",            "m² m⁻²"),
    ("laisha",       "LAI shaded",            "m² m⁻²"),
    ("swveg_vis",    "SW abs veg VIS",        "W m⁻²"),
    ("swvegsun_vis", "SW abs veg VIS sun",    "W m⁻²"),
    ("swvegsha_vis", "SW abs veg VIS sha",    "W m⁻²"),
    ("gpp",          "GPP total",             "µmol m⁻² s⁻¹"),
    ("gpp_sun",      "GPP sunlit",            "µmol m⁻² s⁻¹"),
    ("gpp_sha",      "GPP shaded",            "µmol m⁻² s⁻¹"),
    ("lh_veg",       "LH veg",                "W m⁻²"),
    ("lh_sun",       "LH sunlit",             "W m⁻²"),
    ("lh_sha",       "LH shaded",             "W m⁻²"),
    ("sh_veg",       "SH veg",                "W m⁻²"),
    ("sh_sun",       "SH sunlit",             "W m⁻²"),
    ("sh_sha",       "SH shaded",             "W m⁻²"),
    ("vcmax25_veg",  "Vcmax25 veg",           "µmol m⁻² s⁻¹"),
    ("vcmax25_sun",  "Vcmax25 sunlit",        "µmol m⁻² s⁻¹"),
    ("vcmax25_sha",  "Vcmax25 shaded",        "µmol m⁻² s⁻¹"),
    ("gs_veg",       "Stomatal cond veg",     "mol m⁻² s⁻¹"),
    ("gs_sun",       "Stomatal cond sun",     "mol m⁻² s⁻¹"),
    ("gs_sha",       "Stomatal cond sha",     "mol m⁻² s⁻¹"),
    ("wind_veg",     "Wind speed veg",        "m s⁻¹"),
    ("wind_sun",     "Wind speed sun",        "m s⁻¹"),
    ("wind_sha",     "Wind speed sha",        "m s⁻¹"),
    ("tl_veg",       "Leaf T veg",            "K"),
    ("tl_sun",       "Leaf T sun",            "K"),
    ("tl_sha",       "Leaf T sha",            "K"),
    ("ta_veg",       "Air T in canopy",       "K"),
    ("ta_sun",       "Air T in canopy sun",   "K"),
    ("ta_sha",       "Air T in canopy sha",   "K"),
]

AUX_VARS = [
    ("btran",        "Soil moisture stress",  "—"),
    ("lsc_top",      "Leaf spec. conduct. top","mmol m⁻² s⁻¹ MPa⁻¹"),
    ("psis",         "Soil water potential",  "MPa"),
    ("lwp_top",      "Leaf WP (top)",         "MPa"),
    ("lwp_mid",      "Leaf WP (mid)",         "MPa"),
    ("fracminlwp",   "Frac min LWP",          "—"),
]

PROFILE_VARS = [
    (1,  "height",    "Height",               "m"),
    (2,  "fracsun",   "Sunlit fraction",       "—"),
    (3,  "lad",       "LAD",                   "m² m⁻³"),
    (25, "tair",      "Air temperature",       "K"),
    (26, "qair",      "Specific humidity",     "g kg⁻¹"),
    (27, "ra",        "Aero. resistance",      "s m⁻¹"),
    (24, "wind",      "Wind speed",            "m s⁻¹"),
    (6,  "rn_sun",    "Rn sunlit leaf",        "W m⁻²"),
    (7,  "rn_sha",    "Rn shaded leaf",        "W m⁻²"),
    (8,  "sh_sun",    "SH sunlit leaf",        "W m⁻²"),
    (9,  "sh_sha",    "SH shaded leaf",        "W m⁻²"),
    (10, "lh_sun",    "LH sunlit leaf",        "W m⁻²"),
    (11, "lh_sha",    "LH shaded leaf",        "W m⁻²"),
    (12, "anet_sun",  "Anet sunlit",           "µmol m⁻² s⁻¹"),
    (13, "anet_sha",  "Anet shaded",           "µmol m⁻² s⁻¹"),
]

SOILTEMP_LAYERS = 10  # 10 soil layers

FLUXPROFILE_VARS = [
    (1,  "height",    "Height",               "m"),
    (2,  "sh",        "SH flux profile",       "W m⁻²"),
    (3,  "lh",        "LH flux profile",       "W m⁻²"),
    (4,  "mflx",      "Momentum flux",         "N m⁻²"),
    (7,  "swdwn_vis", "SW down VIS",           "W m⁻²"),
    (8,  "swdwn_nir", "SW down NIR",           "W m⁻²"),
    (11, "lwdwn",     "LW down profile",       "W m⁻²"),
    (12, "lwup",      "LW up profile",         "W m⁻²"),
]


# ---------------------------------------------------------------------------
# Loaders
# ---------------------------------------------------------------------------

def load_out(path: Path) -> np.ndarray:
    """Load whitespace-delimited .out file → 2-D float array.

    Delegates to the shared reader
    (:func:`legoesm.land.evaluation.fluxio.load_out`).
    """
    return _eval_fluxio.load_out(path)


def load_pair(tag: str) -> tuple[np.ndarray, np.ndarray]:
    ref = load_out(REF_DIR / f"{MONTH_TAG}_{tag}.out")
    jax = load_out(JAX_DIR / f"{MONTH_TAG}_{tag}.out")
    return ref, jax


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def scalar_stats(ref: np.ndarray, jax: np.ndarray) -> dict:
    """Return a dict of comparison metrics for 1-D arrays.

    Delegates to the shared, tested implementation
    (:func:`legoesm.land.evaluation.metrics.scalar_stats`); the returned
    keys are unchanged (``n, rmse, mae, bias, r2, corr, nrmse``).
    """
    return _eval_metrics.scalar_stats(ref, jax)


def print_table(title: str, rows: list[tuple]):
    """Print a formatted statistics table."""
    print(f"\n{'='*80}")
    print(f"  {title}")
    print(f"{'='*80}")
    print(f"  {'Variable':<22} {'N':>5} {'RMSE':>9} {'MAE':>9} {'Bias':>9} {'R²':>7} {'Corr':>7} {'NRMSE':>7}")
    print(f"  {'-'*80}")
    for name, label, unit, s in rows:
        if s['n'] == 0:
            print(f"  {label:<22}  (no valid data)")
            continue
        print(
            f"  {label:<22} {s['n']:>5} {s['rmse']:>9.4f} {s['mae']:>9.4f} "
            f"{s['bias']:>9.4f} {s['r2']:>7.4f} {s['corr']:>7.4f} {s['nrmse']:>7.4f}"
            f"  [{unit}]"
        )


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

STYLE = dict(alpha=0.7, linewidth=0.8)
REF_C = "#1f77b4"
JAX_C = "#d62728"


def _savefig(name: str) -> None:
    p = OUT_DIR / name
    plt.savefig(p, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"  saved → {p.relative_to(REPO_ROOT)}")


def plot_time_series(time_ref, time_jax, ref_col, jax_col, label, unit, ax=None, title=""):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(12, 3))
    ax.plot(time_ref, ref_col, color=REF_C, label="Fortran CLM-ML-v2", **STYLE)
    ax.plot(time_jax, jax_col, color=JAX_C, label="CLM-ML-JAX", **STYLE, linestyle="--")
    ax.set_ylabel(f"{label} [{unit}]", fontsize=8)
    ax.set_xlabel("Julian day (May 2007)")
    if title:
        ax.set_title(title, fontsize=9)
    ax.legend(fontsize=7, loc="upper right")
    if standalone:
        return fig


def plot_scatter(ref_col, jax_col, label, unit, s, ax=None, title=""):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(4, 4))
    mask = np.isfinite(ref_col) & np.isfinite(jax_col) & (np.abs(ref_col) < 1e30)
    r, j = ref_col[mask], jax_col[mask]
    ax.scatter(r, j, s=2, alpha=0.3, color="#555555")
    mn, mx = min(r.min(), j.min()), max(r.max(), j.max())
    ax.plot([mn, mx], [mn, mx], "r--", linewidth=1)
    ax.set_xlabel(f"Fortran [{unit}]", fontsize=8)
    ax.set_ylabel(f"JAX [{unit}]", fontsize=8)
    info = f"r={s['corr']:.4f}\nRMSE={s['rmse']:.3f}\nbias={s['bias']:.3f}"
    ax.text(0.04, 0.96, info, transform=ax.transAxes, fontsize=7,
            va="top", ha="left", bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.7))
    if title:
        ax.set_title(title, fontsize=8)
    if standalone:
        return fig


def diurnal_cycle(time_arr, col_arr, dt_days=1 / 48):
    """Average into 48 half-hourly bins (0 = 00:00 UTC).

    Delegates to the shared composite
    (:func:`legoesm.land.evaluation.metrics.diurnal_cycle`), which uses the
    same round-and-wrap binning.
    """
    n_bins = int(round(1.0 / dt_days))
    return _eval_metrics.diurnal_cycle(
        np.asarray(time_arr), np.asarray(col_arr), n_bins=n_bins
    )


def plot_diurnal(time_ref, time_jax, ref_col, jax_col, label, unit, ax=None, title=""):
    standalone = ax is None
    if standalone:
        fig, ax = plt.subplots(figsize=(6, 3))
    hrs = np.arange(48) * 0.5
    ref_d = diurnal_cycle(time_ref, ref_col)
    jax_d = diurnal_cycle(time_jax, jax_col)
    ax.plot(hrs, ref_d, color=REF_C, label="Fortran", linewidth=1.5)
    ax.plot(hrs, jax_d, color=JAX_C, label="JAX", linewidth=1.5, linestyle="--")
    ax.set_xlabel("Hour UTC")
    ax.set_ylabel(f"{label} [{unit}]", fontsize=8)
    ax.set_xticks(np.arange(0, 25, 6))
    ax.legend(fontsize=7)
    if title:
        ax.set_title(title, fontsize=9)
    if standalone:
        return fig


# ---------------------------------------------------------------------------
# 1. FLUX file validation
# ---------------------------------------------------------------------------

def validate_flux():
    print("\n--- Validating flux.out ---")
    ref, jax = load_pair("flux")
    assert ref.shape == jax.shape, f"Shape mismatch: {ref.shape} vs {jax.shape}"
    print(f"  Shape: {ref.shape[0]} timesteps × {ref.shape[1]} columns")

    t_ref = ref[:, 0]
    t_jax = jax[:, 0]

    stats_rows = []
    for i, (name, label, unit) in enumerate(FLUX_VARS[1:], start=1):
        s = scalar_stats(ref[:, i], jax[:, i])
        stats_rows.append((name, label, unit, s))

    print_table("FLUX variables (May 2007, 31 days)", stats_rows)

    # --- multi-panel time series ---
    key_indices = [1, 3, 4, 5, 10]   # rnet, shflx, lhflx, gpp, G_soil
    fig, axes = plt.subplots(len(key_indices), 1, figsize=(14, 12), sharex=True)
    fig.suptitle("CLM-ML-JAX vs Fortran CLM-ML-v2  |  CHATS7 May 2007\nEnergy & Carbon Fluxes",
                 fontsize=11, fontweight="bold")
    for ax, ci in zip(axes, key_indices):
        name, label, unit = FLUX_VARS[ci]
        plot_time_series(t_ref, t_jax, ref[:, ci], jax[:, ci], label, unit, ax=ax)
    axes[-1].set_xlabel("Julian day (May 2007)")
    plt.tight_layout(rect=[0, 0, 1, 0.97])
    _savefig("01_flux_timeseries.png")

    # --- scatter grid for energy fluxes ---
    energy_idx = [1, 3, 4, 10, 11, 12, 13]
    nc = 4
    nr = (len(energy_idx) + nc - 1) // nc
    fig, axes = plt.subplots(nr, nc, figsize=(14, 3.5 * nr))
    fig.suptitle("Scatter plots: JAX vs Fortran — Energy fluxes (W m⁻²)", fontsize=11, fontweight="bold")
    axes = axes.flatten()
    for k, ci in enumerate(energy_idx):
        name, label, unit = FLUX_VARS[ci]
        s = stats_rows[ci - 1][3]
        plot_scatter(ref[:, ci], jax[:, ci], label, unit, s, ax=axes[k], title=label)
    for k in range(len(energy_idx), len(axes)):
        axes[k].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("02_flux_scatter.png")

    # --- diurnal cycles ---
    diurnal_idx = [1, 3, 4, 5]  # rnet, SH, LH, GPP
    fig, axes = plt.subplots(2, 2, figsize=(10, 7))
    fig.suptitle("Diurnal cycles (May 2007 mean)  —  CHATS7", fontsize=11, fontweight="bold")
    for ax, ci in zip(axes.flatten(), diurnal_idx):
        name, label, unit = FLUX_VARS[ci]
        plot_diurnal(t_ref, t_jax, ref[:, ci], jax[:, ci], label, unit, ax=ax, title=label)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("03_flux_diurnal.png")

    # --- energy balance closure check (correct form: Rnet = SH + LH + G + stflx_air + stflx_veg) ---
    # CLM-ML explicitly carries two heat storage terms:
    #   col 2: stflx_air  = canopy air heat storage [W/m²]
    #   col 17: stflx_veg = vegetation heat storage [W/m²]
    # The full closure is Rnet - SH - LH - G - stflx_air - stflx_veg ≈ 0.
    # The simpler Rnet-SH-LH-G form just measures stflx_air (~±150 W/m²)
    # and is NOT an error diagnostic — it is correct CLM-ML physics.
    fig, axes = plt.subplots(2, 2, figsize=(13, 9))
    fig.suptitle("Energy Balance Closure  |  CHATS7 May 2007\n"
                 "Full closure: Rnet − SH − LH − G − stflx_air − stflx_veg",
                 fontsize=10, fontweight="bold")
    for row, (data, tag) in enumerate([(ref, "Fortran"), (jax, "JAX")]):
        rnet      = data[:, 1]
        stflx_air = data[:, 2]
        sh        = data[:, 3]
        lh        = data[:, 4]
        G         = data[:, 10]
        stflx_veg = data[:, 17]

        # Incorrect simpler form (just shows stflx_air)
        resid_simple = rnet - sh - lh - G
        rmse_simple  = np.sqrt(np.nanmean(resid_simple ** 2))

        # Correct full form
        resid_full = rnet - sh - lh - G - stflx_air - stflx_veg
        rmse_full  = np.sqrt(np.nanmean(resid_full ** 2))

        axes[row, 0].plot(t_ref, resid_simple, linewidth=0.7, alpha=0.8, color="#555")
        axes[row, 0].axhline(0, color="r", linewidth=0.8, linestyle="--")
        axes[row, 0].set_title(f"{tag}: Rnet−SH−LH−G  (= stflx_air, not an error)\n"
                               f"RMSE={rmse_simple:.2f} W m⁻²", fontsize=8)
        axes[row, 0].set_ylabel("W m⁻²")

        axes[row, 1].plot(t_ref, resid_full, linewidth=0.7, alpha=0.8, color="#1f77b4")
        axes[row, 1].axhline(0, color="r", linewidth=0.8, linestyle="--")
        axes[row, 1].set_title(f"{tag}: Full closure residual\nRMSE={rmse_full:.4f} W m⁻²",
                               fontsize=8)
        axes[row, 1].set_ylabel("W m⁻²")

    for ax in axes[-1]:
        ax.set_xlabel("Julian day")
    plt.tight_layout(rect=[0, 0, 1, 0.93])
    _savefig("04_energy_balance_closure.png")

    return stats_rows


# ---------------------------------------------------------------------------
# 2. FSUN file validation
# ---------------------------------------------------------------------------

def validate_fsun():
    print("\n--- Validating fsun.out ---")
    ref, jax = load_pair("fsun")
    print(f"  Shape: {ref.shape[0]} timesteps × {ref.shape[1]} columns")

    t_ref = ref[:, 0] if ref.shape[1] > 32 else np.arange(ref.shape[0])
    # fsun.out does NOT have a time column; all 32 cols are data
    t_ref = np.arange(ref.shape[0]) / 48.0  # synthetic time axis (days)
    t_jax = np.arange(jax.shape[0]) / 48.0

    stats_rows = []
    for i, (name, label, unit) in enumerate(FSUN_VARS):
        s = scalar_stats(ref[:, i], jax[:, i])
        stats_rows.append((name, label, unit, s))
    print_table("FSUN variables (sun/shade decomposition)", stats_rows)

    # GPP sun/shade decomposition
    fig, axes = plt.subplots(3, 2, figsize=(13, 10))
    fig.suptitle("Sun/shade fluxes: JAX vs Fortran  |  CHATS7 May 2007", fontsize=11, fontweight="bold")
    pairs = [(8, 9, 10, "GPP"), (11, 12, 13, "LH"), (14, 15, 16, "SH")]
    for row, (tot, sun, sha, var) in enumerate(pairs):
        # scatter: total
        s_tot = stats_rows[tot][3]
        plot_scatter(ref[:, tot], jax[:, tot], f"{var} total", "—", s_tot, ax=axes[row, 0],
                     title=f"{var} total")
        # diurnal: sun vs sha
        hr = t_ref * 24 % 24
        bins = (hr / 0.5).astype(int) % 48
        for ci, style, label in [(sun, "-", "sunlit"), (sha, "--", "shaded")]:
            r_d = np.array([ref[bins == b, ci].mean() for b in range(48)])
            j_d = np.array([jax[bins == b, ci].mean() for b in range(48)])
            axes[row, 1].plot(np.arange(48) * 0.5, r_d, color=REF_C, linestyle=style,
                              label=f"Fortran {label}", linewidth=1.3)
            axes[row, 1].plot(np.arange(48) * 0.5, j_d, color=JAX_C, linestyle=style,
                              label=f"JAX {label}", linewidth=1.3, alpha=0.8)
        axes[row, 1].set_title(f"{var} diurnal — sun & shade", fontsize=9)
        axes[row, 1].set_xlabel("Hour UTC")
        axes[row, 1].legend(fontsize=7, ncol=2)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("05_fsun_decomposition.png")

    # Leaf temperature & stomatal conductance
    fig, axes = plt.subplots(2, 2, figsize=(11, 8))
    fig.suptitle("Leaf temperature & stomatal conductance — diurnal cycles", fontsize=11)
    pairs2 = [(26, "Tl_veg (K)"), (17, "Vcmax25 (µmol/m²/s)"),
              (20, "gs_veg (mol/m²/s)"), (2, "PAI (m²/m²)")]
    for ax, (ci, label) in zip(axes.flatten(), pairs2):
        r_d = diurnal_cycle(t_ref, ref[:, ci])
        j_d = diurnal_cycle(t_jax, jax[:, ci])
        ax.plot(np.arange(48) * 0.5, r_d, color=REF_C, label="Fortran", linewidth=1.5)
        ax.plot(np.arange(48) * 0.5, j_d, color=JAX_C, label="JAX", linestyle="--", linewidth=1.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel("Hour UTC")
        ax.legend(fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("06_leaf_physiology.png")

    return stats_rows


# ---------------------------------------------------------------------------
# 3. AUX file validation
# ---------------------------------------------------------------------------

def validate_aux():
    print("\n--- Validating aux.out ---")
    ref, jax = load_pair("aux")
    print(f"  Shape: {ref.shape}")
    t = np.arange(ref.shape[0]) / 48.0

    stats_rows = []
    for i, (name, label, unit) in enumerate(AUX_VARS):
        s = scalar_stats(ref[:, i], jax[:, i])
        stats_rows.append((name, label, unit, s))
    print_table("AUX variables (plant hydraulics)", stats_rows)

    fig, axes = plt.subplots(3, 2, figsize=(12, 9))
    fig.suptitle("Plant hydraulics: JAX vs Fortran  |  CHATS7 May 2007", fontsize=11)
    for ax, (i, (name, label, unit)) in zip(axes.flatten(), enumerate(AUX_VARS)):
        ax.plot(t, ref[:, i], color=REF_C, label="Fortran", **STYLE)
        ax.plot(t, jax[:, i], color=JAX_C, label="JAX", linestyle="--", **STYLE)
        s = stats_rows[i][3]
        ax.set_title(f"{label}  [RMSE={s['rmse']:.4f}  r={s['corr']:.4f}]", fontsize=8)
        ax.set_ylabel(unit, fontsize=7)
        ax.legend(fontsize=7)
    axes[-1][-1].set_xlabel("Days since May 1")
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("07_aux_hydraulics.png")

    return stats_rows


# ---------------------------------------------------------------------------
# 4. SOILTEMP validation
# ---------------------------------------------------------------------------

def validate_soiltemp():
    print("\n--- Validating soiltemp.out ---")
    ref, jax = load_pair("soiltemp")
    print(f"  Shape: {ref.shape}")
    t_ref = ref[:, 0]
    t_jax = jax[:, 0]

    # Columns: time, z1, T1, z2, T2, ... z10, T10
    depths_ref = ref[0, 1::2][:SOILTEMP_LAYERS]  # fixed depths
    temp_cols_r = ref[:, 2::2][:, :SOILTEMP_LAYERS]  # (ntim, 10)
    temp_cols_j = jax[:, 2::2][:, :SOILTEMP_LAYERS]

    stats_rows = []
    for lyr in range(SOILTEMP_LAYERS):
        s = scalar_stats(temp_cols_r[:, lyr], temp_cols_j[:, lyr])
        z = float(depths_ref[lyr])
        stats_rows.append((f"soilT_L{lyr+1}", f"Soil T layer {lyr+1} (z={z:.3f} m)", "K", s))
    print_table("SOILTEMP layers", stats_rows)

    fig, axes = plt.subplots(2, 5, figsize=(16, 6), sharex=True)
    fig.suptitle("Soil temperature profiles: JAX vs Fortran  |  CHATS7 May 2007", fontsize=11)
    for lyr, ax in enumerate(axes.flatten()):
        z = float(depths_ref[lyr])
        ax.plot(t_ref, temp_cols_r[:, lyr], color=REF_C, label="Fortran", linewidth=0.7)
        ax.plot(t_jax, temp_cols_j[:, lyr], color=JAX_C, label="JAX", linestyle="--", linewidth=0.7)
        s = stats_rows[lyr][3]
        ax.set_title(f"Layer {lyr+1}  z={z:.3f} m\nRMSE={s['rmse']:.4f} K", fontsize=8)
        ax.set_ylabel("T (K)", fontsize=7)
        if lyr == 0:
            ax.legend(fontsize=6)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("08_soiltemp_timeseries.png")

    # Soil temperature depth-time heatmap (Fortran)
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    fig.suptitle("Soil temperature depth-time heatmap  |  CHATS7 May 2007", fontsize=11)
    days = (t_ref - t_ref[0])
    for ax, (data, tag) in zip(axes, [(temp_cols_r, "Fortran"), (temp_cols_j, "JAX")]):
        im = ax.pcolormesh(days, -depths_ref, data.T, cmap="RdYlBu_r", shading="auto")
        ax.set_xlabel("Days since May 1")
        ax.set_ylabel("Depth (m, negative)")
        ax.set_title(tag, fontsize=10)
        fig.colorbar(im, ax=ax, label="T (K)", pad=0.01)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig("09_soiltemp_heatmap.png")

    return stats_rows


# ---------------------------------------------------------------------------
# 5. PROFILE validation (vertical profiles)
# ---------------------------------------------------------------------------

def validate_profile():
    print("\n--- Validating profile.out ---")
    ref, jax = load_pair("profile")
    print(f"  Shape: {ref.shape}")

    # profile.out: multiple rows per timestep (one per canopy layer)
    # Col 0: time_stamp, Col 1: height (zs)
    # Number of layers per timestep
    ntim = 1488
    nlayers = ref.shape[0] // ntim
    print(f"  Detected {nlayers} layers per timestep")

    # Reshape: (ntim, nlayers, ncols)
    ncols = ref.shape[1]
    ref_3d = ref[:ntim * nlayers].reshape(ntim, nlayers, ncols)
    jax_3d = jax[:ntim * nlayers].reshape(ntim, nlayers, ncols)

    # Mean vertical profile (all timesteps)
    ref_mean = np.nanmean(ref_3d, axis=0)  # (nlayers, ncols)
    jax_mean = np.nanmean(jax_3d, axis=0)

    heights = ref_mean[:, 1]  # height column

    fig, axes = plt.subplots(2, 4, figsize=(14, 10), sharey=True)
    fig.suptitle("Mean vertical profiles: JAX vs Fortran  |  CHATS7 May 2007", fontsize=11)
    var_subsets = [(25, "Air T (K)"), (26, "q (g/kg)"), (24, "Wind (m/s)"), (27, "Ra (s/m)"),
                   (2,  "Sunlit frac"), (3, "LAD (m²/m³)"),
                   (8,  "SH sun leaf (W/m²)"), (10, "LH sun leaf (W/m²)")]
    for ax, (ci, label) in zip(axes.flatten(), var_subsets):
        r_col = ref_mean[:, ci]
        j_col = jax_mean[:, ci]
        mask = (np.abs(r_col) < 900) & (np.abs(j_col) < 900)
        ax.plot(r_col[mask], heights[mask], color=REF_C, label="Fortran", linewidth=1.5)
        ax.plot(j_col[mask], heights[mask], color=JAX_C, label="JAX", linestyle="--", linewidth=1.5)
        ax.set_title(label, fontsize=9)
        ax.set_xlabel(label.split("(")[-1].replace(")", "") if "(" in label else "")
        ax.set_ylabel("Height (m)", fontsize=8)
        ax.legend(fontsize=7)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("10_vertical_profiles_mean.png")

    # Daytime vs nighttime vertical profiles
    t_arr = ref_3d[:, 0, 0]  # time per timestep
    frac_day = t_arr % 1.0
    # CHATS7 is in California (UTC-8 PDT / UTC-7 PST).  Local daytime
    # 06:00-18:00 PDT = 14:00-02:00+1 UTC → fraction 0.583..1.0 and 0.0..0.083.
    # Correct mask for California local daytime (not UTC daytime):
    day_mask  = (frac_day >= 0.583) | (frac_day <= 0.083)
    nite_mask = ~day_mask

    fig, axes = plt.subplots(1, 3, figsize=(12, 7), sharey=True)
    fig.suptitle("Day vs night vertical profiles  |  CHATS7 May 2007", fontsize=11)
    for ax, (ci, label, period, mask) in zip(
        axes,
        [(25, "Air T (K)", "Day",   day_mask),
         (25, "Air T (K)", "Night", nite_mask),
         (24, "Wind (m/s)", "Day",  day_mask)]
    ):
        r_m = np.nanmean(ref_3d[mask, :, ci], axis=0)
        j_m = np.nanmean(jax_3d[mask, :, ci], axis=0)
        valid = np.abs(r_m) < 900
        ax.plot(r_m[valid], heights[valid], color=REF_C, label="Fortran", linewidth=1.5)
        ax.plot(j_m[valid], heights[valid], color=JAX_C, label="JAX", linestyle="--", linewidth=1.5)
        ax.set_title(f"{label} ({period})", fontsize=10)
        ax.set_ylabel("Height (m)")
        ax.legend(fontsize=8)
    plt.tight_layout(rect=[0, 0, 1, 0.95])
    _savefig("11_profiles_day_night.png")

    # Profile-level stats for key variables
    stats_rows = []
    for ci, name, label, unit in PROFILE_VARS:
        r_flat = ref_3d[:, :, ci].ravel()
        j_flat = jax_3d[:, :, ci].ravel()
        valid = (np.abs(r_flat) < 900) & (np.abs(j_flat) < 900)
        s = scalar_stats(r_flat[valid], j_flat[valid])
        stats_rows.append((name, label, unit, s))
    print_table("PROFILE variables (layer-level, all layers)", stats_rows)

    return stats_rows


# ---------------------------------------------------------------------------
# 6. FLUXPROFILE validation
# ---------------------------------------------------------------------------

def validate_fluxprofile():
    print("\n--- Validating fluxprofile.out ---")
    ref, jax = load_pair("fluxprofile")
    print(f"  Shape: {ref.shape}")

    ntim = 1488
    nlayers = ref.shape[0] // ntim
    ncols = ref.shape[1]
    ref_3d = ref[:ntim * nlayers].reshape(ntim, nlayers, ncols)
    jax_3d = jax[:ntim * nlayers].reshape(ntim, nlayers, ncols)

    heights = np.nanmean(ref_3d[:, :, 1], axis=0)

    fig, axes = plt.subplots(2, 4, figsize=(14, 9), sharey=True)
    fig.suptitle("Mean flux profiles: JAX vs Fortran  |  CHATS7 May 2007", fontsize=11)
    vars_fp = [(2, "SH (W/m²)"), (3, "LH (W/m²)"), (4, "Momentum (N/m²)"),
               (7, "SW↓ VIS (W/m²)"), (8, "SW↓ NIR (W/m²)"),
               (11, "LW↓ (W/m²)"), (12, "LW↑ (W/m²)")]
    t_arr = ref_3d[:, 0, 0]
    frac_day = t_arr % 1.0
    # California local daytime (PDT = UTC-8): 06:00-18:00 PDT = 14:00-02:00 UTC
    day_mask = (frac_day >= 0.583) | (frac_day <= 0.083)
    for ax, (ci, label) in zip(axes.flatten(), vars_fp):
        r_d = np.nanmean(ref_3d[day_mask, :, ci], axis=0)
        j_d = np.nanmean(jax_3d[day_mask, :, ci], axis=0)
        valid = np.abs(r_d) < 1e5
        ax.plot(r_d[valid], heights[valid], color=REF_C, label="Fortran", linewidth=1.5)
        ax.plot(j_d[valid], heights[valid], color=JAX_C, label="JAX", linestyle="--", linewidth=1.5)
        ax.set_title(f"{label} (daytime)", fontsize=9)
        ax.set_ylabel("Height (m)", fontsize=8)
        ax.legend(fontsize=7)
    axes[-1][-1].set_visible(False)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    _savefig("12_fluxprofile_mean.png")

    stats_rows = []
    for ci, name, label, unit in FLUXPROFILE_VARS[1:]:
        r_flat = ref_3d[:, :, ci].ravel()
        j_flat = jax_3d[:, :, ci].ravel()
        valid = np.abs(r_flat) < 1e5
        s = scalar_stats(r_flat[valid], j_flat[valid])
        stats_rows.append((name, label, unit, s))
    print_table("FLUXPROFILE variables", stats_rows)

    return stats_rows


# ---------------------------------------------------------------------------
# 7. Taylor diagram
# ---------------------------------------------------------------------------

def taylor_diagram(stats_list: list[tuple], title: str, fname: str):
    """Plot a Taylor diagram for multiple variables."""
    fig, ax = plt.subplots(1, 1, figsize=(7, 7), subplot_kw=dict(projection="polar"))
    fig.suptitle(f"Taylor Diagram — {title}", fontsize=11, fontweight="bold")

    ax.set_theta_direction(-1)
    ax.set_theta_zero_location("N")
    ax.set_thetamin(0)
    ax.set_thetamax(90)

    # Reference standard deviation = 1 (normalized)
    max_std = 1.5
    thetas = np.linspace(0, np.pi / 2, 200)
    for r in [0.5, 1.0, 1.5]:
        ax.plot(thetas, np.full_like(thetas, r), "k-", linewidth=0.5, alpha=0.3)
    for c in [0.0, 0.2, 0.4, 0.6, 0.7, 0.8, 0.9, 0.95, 0.99]:
        th = np.arccos(c)
        ax.plot([th, th], [0, max_std], ":", color="gray", linewidth=0.5)
        ax.text(th, max_std + 0.05, f"{c}", fontsize=6, ha="center")

    colors = plt.cm.tab20(np.linspace(0, 1, len(stats_list)))
    for k, (name, label, unit, s) in enumerate(stats_list):
        if s["n"] == 0 or not np.isfinite(s["corr"]):
            continue
        theta = np.arccos(np.clip(s["corr"], -1, 1))
        std_n = s["rmse"] / max(np.sqrt(s["r2"] + 1e-10), 1e-6) if s["r2"] > 0 else 1.0
        # Use r = 1.0 (normalized reference std) and plot point at (theta, std_ratio)
        ax.scatter(theta, 1.0, s=60, color=colors[k], label=f"{label}", zorder=5, marker="o")
    ax.set_rticks([0.5, 1.0, 1.5])
    ax.legend(fontsize=6, bbox_to_anchor=(1.25, 1.0), loc="upper right")
    _savefig(fname)


# ---------------------------------------------------------------------------
# 8. Summary statistics heatmap
# ---------------------------------------------------------------------------

def summary_heatmap(all_stats: dict[str, list], fname: str):
    """Correlation matrix heatmap across all variable groups."""
    labels, corrs, rmses = [], [], []
    for group, rows in all_stats.items():
        for name, label, unit, s in rows:
            if s["n"] > 0 and np.isfinite(s["corr"]):
                labels.append(f"{group}/{label}")
                corrs.append(s["corr"])
                rmses.append(s["nrmse"])

    n = len(labels)
    if n == 0:
        return

    fig, axes = plt.subplots(1, 2, figsize=(max(6, n * 0.35 + 2), 5))
    fig.suptitle("Validation summary: CLM-ML-JAX vs Fortran CLM-ML-v2  |  CHATS7 May 2007",
                 fontsize=10, fontweight="bold")

    for ax, vals, cmap, title, vmin, vmax in [
        (axes[0], corrs, "RdYlGn", "Pearson correlation (r)", -1, 1),
        (axes[1], rmses, "RdYlGn_r", "Normalized RMSE (σ-units)", 0, 1),
    ]:
        arr = np.array(vals).reshape(1, -1)
        im = ax.imshow(arr, cmap=cmap, vmin=vmin, vmax=vmax, aspect="auto")
        ax.set_xticks(range(n))
        ax.set_xticklabels(labels, rotation=90, fontsize=6)
        ax.set_yticks([])
        ax.set_title(title, fontsize=9)
        fig.colorbar(im, ax=ax, orientation="horizontal", pad=0.25, shrink=0.8)
        for k, v in enumerate(vals):
            ax.text(k, 0, f"{v:.3f}", ha="center", va="center", fontsize=5,
                    color="black" if 0.3 < abs(v) < 0.97 else "white")

    plt.tight_layout(rect=[0, 0, 1, 0.92])
    _savefig(fname)


# ---------------------------------------------------------------------------
# 9. Residual analysis
# ---------------------------------------------------------------------------

def residual_analysis(ref, jax, t, label, unit, fname):
    """PDF of residuals and residual autocorrelation."""
    mask = np.isfinite(ref) & np.isfinite(jax) & (np.abs(ref) < 1e30)
    resid = (jax - ref)[mask]
    t_m = t[mask]

    fig, axes = plt.subplots(1, 3, figsize=(13, 4))
    fig.suptitle(f"Residual analysis: {label}  |  CHATS7 May 2007", fontsize=11)

    # PDF
    axes[0].hist(resid, bins=60, density=True, color="#4878CF", edgecolor="white", linewidth=0.3)
    mu, sig = resid.mean(), resid.std()
    x = np.linspace(resid.min(), resid.max(), 200)
    axes[0].plot(x, stats.norm.pdf(x, mu, sig), "r-", linewidth=1.5, label=f"N({mu:.3f},{sig:.3f})")
    axes[0].set_xlabel(f"JAX − Fortran  [{unit}]")
    axes[0].set_ylabel("Density")
    axes[0].set_title("Residual distribution")
    axes[0].legend(fontsize=8)

    # Time series of residuals
    axes[1].plot(t_m, resid, linewidth=0.6, color="#555555", alpha=0.8)
    axes[1].axhline(0, color="r", linewidth=1)
    axes[1].set_xlabel("Julian day")
    axes[1].set_ylabel(f"Residual [{unit}]")
    axes[1].set_title("Residuals over time")

    # Autocorrelation
    n_lag = min(100, len(resid) // 2)
    acf_vals = [np.corrcoef(resid[:-k], resid[k:])[0, 1] for k in range(1, n_lag + 1)]
    axes[2].stem(range(1, n_lag + 1), acf_vals, markerfmt="C0o", linefmt="C0-",
                 basefmt="k-")
    conf = 1.96 / np.sqrt(len(resid))
    axes[2].axhline(conf, color="r", linewidth=0.8, linestyle="--")
    axes[2].axhline(-conf, color="r", linewidth=0.8, linestyle="--")
    axes[2].set_xlabel("Lag (timesteps = 30 min)")
    axes[2].set_ylabel("ACF")
    axes[2].set_title("Residual autocorrelation")

    plt.tight_layout(rect=[0, 0, 1, 0.94])
    _savefig(fname)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    print(f"\n{'#'*80}")
    print("  CLM-ML-JAX Canopy Validation  |  CHATS7 site, May 2007 (31 days)")
    print(f"  Reference:  {REF_DIR.relative_to(REPO_ROOT)}")
    print(f"  JAX output: {JAX_DIR.relative_to(REPO_ROOT)}")
    print(f"  Plots →     {OUT_DIR.relative_to(REPO_ROOT)}/")
    print(f"{'#'*80}")

    all_stats = {}

    flux_stats  = validate_flux()
    all_stats["flux"] = flux_stats

    fsun_stats  = validate_fsun()
    all_stats["fsun"] = fsun_stats

    aux_stats   = validate_aux()
    all_stats["aux"] = aux_stats

    soiltemp_stats = validate_soiltemp()
    all_stats["soiltemp"] = soiltemp_stats

    profile_stats = validate_profile()
    all_stats["profile"] = profile_stats

    fluxprofile_stats = validate_fluxprofile()
    all_stats["fluxprofile"] = fluxprofile_stats

    # Residual analysis for key energy fluxes
    ref_f, jax_f = load_pair("flux")
    for ci, label, unit, fname in [
        (3, "Sensible Heat", "W m⁻²", "13_residuals_SH.png"),
        (4, "Latent Heat",   "W m⁻²", "14_residuals_LH.png"),
        (5, "GPP",           "µmol m⁻² s⁻¹", "15_residuals_GPP.png"),
    ]:
        residual_analysis(ref_f[:, ci], jax_f[:, ci], ref_f[:, 0],
                          label, unit, fname)

    # Summary heatmap
    summary_heatmap(all_stats, "16_summary_heatmap.png")

    # Print global pass/fail
    print(f"\n{'='*80}")
    print("  VALIDATION SUMMARY")
    print(f"{'='*80}")
    all_corrs, all_rmse_norm = [], []
    for group, rows in all_stats.items():
        for name, label, unit, s in rows:
            if s["n"] > 0 and np.isfinite(s["corr"]):
                all_corrs.append(s["corr"])
                all_rmse_norm.append(s["nrmse"])

    print(f"  Total variables assessed: {len(all_corrs)}")
    print(f"  Median Pearson r:          {np.median(all_corrs):.6f}")
    print(f"  Min    Pearson r:          {np.min(all_corrs):.6f}")
    print(f"  Median NRMSE:              {np.median(all_rmse_norm):.6f}")
    print(f"  Max    NRMSE:              {np.max(all_rmse_norm):.6f}")

    n_near_perfect = sum(1 for c in all_corrs if c > 0.9999)
    print(f"  Variables with r > 0.9999: {n_near_perfect}/{len(all_corrs)}")
    n_perfect = sum(1 for c in all_corrs if c > 0.999999)
    print(f"  Variables with r > 0.999999: {n_perfect}/{len(all_corrs)}")

    print(f"\n  Output figures saved to {OUT_DIR.relative_to(REPO_ROOT)}/")
    print(f"{'='*80}\n")


if __name__ == "__main__":
    main()
