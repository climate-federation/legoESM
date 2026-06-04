"""Column profile comparison: all legoESM microphysics schemes.

Produces three separate figures — one per column type:
  - Warm  (T_sfc=290 K, liquid only)
  - Mixed (T_sfc=270 K, spanning freeze level)
  - Cold  (T_sfc=255 K, ice dominant)

Each figure is a 5-row × 3-column grid:
  Col 1: T | dT/dt | RH | number concentrations | precip bar chart
  Col 2: q_v | q_c | q_r | q_i | q_s  (initial profiles)
  Col 3: dq_v/dt | dq_c/dt | dq_r/dt | dq_i/dt | dq_s/dt  (tendencies)

Usage
-----
    JAX_ENABLE_X64=1 python scripts/plot_microphysics_comparison.py
"""

import jax.numpy as jnp
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import numpy as np

from legoesm import constants
from legoesm.atmosphere.physics.microphysics.kessler import kessler_microphysics
from legoesm.atmosphere.physics.microphysics.sundqvist import sundqvist_microphysics
from legoesm.atmosphere.physics.microphysics.seifert_beheng import seifert_beheng_microphysics
from legoesm.atmosphere.physics.microphysics.morrison import morrison_microphysics
from legoesm.atmosphere.physics.microphysics.thompson import thompson_microphysics
from legoesm.atmosphere.physics.microphysics.p3 import p3_microphysics
from legoesm.atmosphere.physics.microphysics.config import (
    KesslerConfig, SundqvistConfig, SeifertBehengConfig,
    MorrisonConfig, ThompsonConfig, P3Config,
)
from legoesm.atmosphere.physics.microphysics.output import HydrometeorState
from legoesm.thermo import saturation_mixing_ratio


# ---------------------------------------------------------------------------
# Scheme registry: (label, fn, config, color, linestyle, linewidth)
# ---------------------------------------------------------------------------

SCHEMES = [
    ("Kessler",        kessler_microphysics,       KesslerConfig(),       "#888888", "-",  1.2),
    ("Sundqvist",      sundqvist_microphysics,      SundqvistConfig(),     "#f4a261", "--", 1.2),
    ("Seifert-Beheng", seifert_beheng_microphysics, SeifertBehengConfig(), "#00aaaa", "-",  1.2),
    ("Morrison",       morrison_microphysics,       MorrisonConfig(),      "#2166ac", "-",  2.0),
    ("Thompson",       thompson_microphysics,       ThompsonConfig(),      "#762a83", "--", 2.0),
    ("P3",             p3_microphysics,             P3Config(),            "#d6604d", "-.", 2.0),
]

ICE_SCHEMES = {"Morrison", "Thompson", "P3"}


# ---------------------------------------------------------------------------
# Column builder
# ---------------------------------------------------------------------------

def make_column(
    nlev: int = 40,
    T_sfc: float = 270.0,
    q_c_frac: float = 0.3,
    q_i_frac: float = 0.0,
    q_rim_frac: float = 0.0,
    B_rim_rho: float = 400.0,
    rh: float = 0.95,
):
    """Build a single atmospheric column (ncol=1)."""
    p_s = 1.0e5
    sigma_half = jnp.linspace(0.0, 1.0, nlev + 1)
    sigma_full = 0.5 * (sigma_half[:-1] + sigma_half[1:])
    p_half = (sigma_half * p_s)[None, :]
    p_full = (sigma_full * p_s)[None, :]

    T = T_sfc * jnp.clip(sigma_full, 0.01) ** 0.19
    T = jnp.maximum(T, 190.0)[None, :]

    rho = p_full / (constants.R_d * T)
    dp  = p_half[:, 1:] - p_half[:, :-1]
    p_mid = 0.5 * (p_half[:, :-1] + p_half[:, 1:])
    dz = jnp.abs(constants.R_d * T * dp / (constants.g * jnp.maximum(p_mid, 1.0)))

    q_sat = saturation_mixing_ratio(T, p_full)
    q_v   = rh * q_sat

    q_c = jnp.where(sigma_full[None, :] > 0.4, q_c_frac * q_sat, 0.0)

    below_freeze = T < constants.T_freeze
    q_i = jnp.where(below_freeze, q_i_frac * q_sat, 0.0)

    q_rim = jnp.where(below_freeze, q_rim_frac * q_i_frac * q_sat, 0.0)
    B_rim = jnp.where(below_freeze & (q_rim > 0), q_rim / B_rim_rho, 0.0)

    hydro_base = HydrometeorState(
        q_c=q_c, q_r=jnp.zeros_like(T), q_i=q_i,
        q_s=jnp.zeros_like(T), q_g=jnp.zeros_like(T),
        N_c=1e8 * jnp.ones_like(T),
        N_r=jnp.zeros_like(T),
        N_i=jnp.zeros_like(T),
    )
    hydro_p3 = hydro_base._replace(q_s=q_rim, q_g=B_rim)

    p_hPa = np.array(p_full[0]) / 100.0
    q_sat_np = np.array(q_sat[0])

    return T[0], q_v, p_full, p_half, rho, dz, hydro_base, hydro_p3, p_hPa, q_sat_np


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------

def run_all_schemes(T, q_v, p_full, p_half, rho, dz, hydro_base, hydro_p3, dt=300.0):
    results = {}
    T_col = T[None, :]
    for name, fn, cfg, *_ in SCHEMES:
        hydro = hydro_p3 if name == "P3" else hydro_base
        results[name] = fn(T_col, q_v, hydro, p_full, p_half, rho, dz, dt, cfg)
    return results


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _freeze_ref(ax, p_freeze):
    ax.axhline(p_freeze, color="gray", lw=0.8, ls=":", alpha=0.65)


def _symlog_x(ax, vals_list):
    """Set symlog x-scale based on the spread of values in vals_list."""
    all_vals = np.concatenate([v.ravel() for v in vals_list])
    finite = all_vals[np.isfinite(all_vals) & (all_vals != 0)]
    if len(finite):
        linthresh = max(np.percentile(np.abs(finite), 10), 1e-3)
        ax.set_xscale("symlog", linthresh=linthresh)


# ---------------------------------------------------------------------------
# Case definitions
# ---------------------------------------------------------------------------

CASES = {
    "warm": {
        "title": "Warm  (T$_{sfc}$=290 K, liquid only)",
        "kwargs": dict(T_sfc=290.0, q_c_frac=0.3, q_i_frac=0.0, q_rim_frac=0.0),
    },
    "mixed": {
        "title": "Mixed  (T$_{sfc}$=270 K, ice+liquid)",
        "kwargs": dict(T_sfc=270.0, q_c_frac=0.2, q_i_frac=0.5, q_rim_frac=0.3),
    },
    "cold": {
        "title": "Cold  (T$_{sfc}$=255 K, ice dominant)",
        "kwargs": dict(T_sfc=255.0, q_c_frac=0.05, q_i_frac=0.8, q_rim_frac=0.5),
    },
}

N_ROWS = 5
SCHEME_NAMES  = [s[0] for s in SCHEMES]
SCHEME_COLORS = [s[3] for s in SCHEMES]
SCHEME_LS     = [s[4] for s in SCHEMES]
SCHEME_LW     = [s[5] for s in SCHEMES]


# ---------------------------------------------------------------------------
# One figure per case
# ---------------------------------------------------------------------------

for case_key, case in CASES.items():
    title = case["title"]

    (T, q_v, p_full, p_half, rho, dz,
     hydro_base, hydro_p3, p_hPa, q_sat_np) = make_column(**case["kwargs"])

    results = run_all_schemes(T, q_v, p_full, p_half, rho, dz, hydro_base, hydro_p3)

    T_np   = np.array(T)
    qv_np  = np.array(q_v[0])
    RH_np  = qv_np / np.clip(q_sat_np, 1e-30, None) * 100.0

    freeze_idx = int(np.argmin(np.abs(T_np - float(constants.T_freeze))))
    p_freeze   = float(p_hPa[freeze_idx])

    fig = plt.figure(figsize=(15, 3.8 * N_ROWS))
    fig.suptitle(title, fontsize=13, fontweight="bold", y=1.005)

    gs = gridspec.GridSpec(
        N_ROWS, 3,
        figure=fig,
        hspace=0.55, wspace=0.40,
    )

    # Common kwargs for pressure-vs-x profile axes
    def _profile_ax(row, col):
        ax = fig.add_subplot(gs[row, col])
        ax.invert_yaxis()
        ax.set_ylabel("Pressure [hPa]", fontsize=8)
        ax.tick_params(labelsize=8)
        _freeze_ref(ax, p_freeze)
        return ax

    # ==========================================================
    # COLUMN 0
    # ==========================================================

    # Row 0 — Temperature
    ax = _profile_ax(0, 0)
    ax.plot(T_np, p_hPa, color="firebrick", lw=2.0)
    ax.axvline(float(constants.T_freeze), color="gray", lw=0.8, ls=":", alpha=0.65)
    ax.set_xlabel("T  [K]", fontsize=9)
    ax.set_title("Temperature", fontsize=10, fontweight="bold")

    # Row 1 — dT/dt
    ax = _profile_ax(1, 0)
    for name, _, _, color, ls, lw in SCHEMES:
        vals = np.array(results[name].dT_dt[0]) * 1e3
        ax.plot(vals, p_hPa, color=color, ls=ls, lw=lw, label=name, alpha=0.9)
    ax.axvline(0, color="black", lw=0.5, alpha=0.3)
    _symlog_x(ax, [np.array(results[n].dT_dt[0]) * 1e3 for n, *_ in SCHEMES])
    ax.set_xlabel(r"$\partial T/\partial t$  [mK s$^{-1}$]", fontsize=9)
    ax.set_title(r"$\partial T/\partial t$", fontsize=10, fontweight="bold")
    ax.legend(fontsize=6.5, loc="lower left", framealpha=0.85,
              edgecolor="lightgray", ncol=2)

    # Row 2 — Relative humidity
    ax = _profile_ax(2, 0)
    ax.plot(RH_np, p_hPa, color="#2ca02c", lw=2.0)
    ax.axvline(100.0, color="gray", lw=0.8, ls=":", alpha=0.65)
    ax.set_xlabel("RH  [%]", fontsize=9)
    ax.set_title("Relative humidity", fontsize=10, fontweight="bold")

    # Row 3 — Number concentrations (initial state)
    ax = _profile_ax(3, 0)
    N_data = [
        (np.array(hydro_base.N_c[0]), r"$N_c$",  "#4477AA"),
        (np.array(hydro_base.N_r[0]), r"$N_r$",  "#00aaaa"),
        (np.array(hydro_base.N_i[0]), r"$N_i$",  "#AAAAAA"),
    ]
    any_N = False
    for N_vals, label, color in N_data:
        if np.any(N_vals > 0):
            ax.plot(N_vals, p_hPa, color=color, lw=1.8, label=label)
            any_N = True
    if any_N:
        ax.set_xscale("log")
        ax.legend(fontsize=8, loc="lower right", framealpha=0.85, edgecolor="lightgray")
    ax.set_xlabel(r"$N$  [kg$^{-1}$]", fontsize=9)
    ax.set_title("Number concentrations", fontsize=10, fontweight="bold")

    # Row 4 — Column-integrated precipitation bar chart
    ax_bar = fig.add_subplot(gs[4, 0])
    precip_vals = [
        float(jnp.sum(results[name].precipitation)) * 1e9
        for name, *_ in SCHEMES
    ]
    bars = ax_bar.bar(
        SCHEME_NAMES, precip_vals,
        color=SCHEME_COLORS, alpha=0.85,
        hatch=[("/" if ls == "--" else ("x" if ls == "-." else ""))
               for ls in SCHEME_LS],
    )
    for bar, v in zip(bars, precip_vals):
        if v > 1e-6:
            ax_bar.text(
                bar.get_x() + bar.get_width() / 2, v,
                f"{v:.2f}", ha="center", va="bottom", fontsize=7, rotation=90,
            )
    ax_bar.set_ylabel(r"Precip  [ng kg$^{-1}$ s$^{-1}$]", fontsize=8)
    ax_bar.set_title("Column-integrated precipitation", fontsize=10, fontweight="bold")
    ax_bar.tick_params(axis="x", rotation=30, labelsize=7.5)
    ax_bar.tick_params(axis="y", labelsize=8)

    # ==========================================================
    # COLUMN 1 — Initial mixing ratios
    # ==========================================================

    mix_rows = [
        (np.array(q_v[0])          * 1e3, r"$q_v$  [g kg$^{-1}$]",   r"$q_v$",  "#4477AA", "-"),
        (np.array(hydro_base.q_c[0])* 1e3, r"$q_c$  [g kg$^{-1}$]",   r"$q_c$",  "#66CCEE", "-"),
        (np.array(hydro_base.q_r[0])* 1e3, r"$q_r$  [g kg$^{-1}$]",   r"$q_r$",  "#1f77b4", "-"),
        (np.array(hydro_base.q_i[0])* 1e3, r"$q_i$  [g kg$^{-1}$]",   r"$q_i$",  "#AAAAAA", "--"),
        # q_s row: show Morrison/Thompson (base, q_s=0) and P3 (q_rim)
        (None,                             r"$q_s$  [g kg$^{-1}$]",   None,      None,      None),
    ]

    col1_titles = [
        r"$q_v$", r"$q_c$", r"$q_r$", r"$q_i$",
        r"$q_{snow}$ (Mor/Tho) / $q_{rim}$ (P3)",
    ]

    for row, (vals, xlabel, label, color, ls) in enumerate(mix_rows):
        ax = _profile_ax(row, 1)
        ax.set_title(col1_titles[row], fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=9)

        if row < 4:
            ax.plot(vals, p_hPa, color=color, lw=1.8, ls=ls, label=label)
        else:
            # q_s: base schemes start at 0; P3 starts at q_rim
            q_s_base = np.array(hydro_base.q_s[0]) * 1e3
            q_rim_p3  = np.array(hydro_p3.q_s[0])  * 1e3
            plotted = False
            if np.any(q_s_base > 0):
                ax.plot(q_s_base, p_hPa, color="#2166ac", lw=1.8, ls="-",
                        label=r"$q_{snow}$ (Mor/Tho)")
                plotted = True
            if np.any(q_rim_p3 > 0):
                ax.plot(q_rim_p3, p_hPa, color="#d6604d", lw=1.8, ls="--",
                        label=r"$q_{rim}$ (P3)")
                plotted = True
            if plotted:
                ax.legend(fontsize=7.5, loc="lower right", framealpha=0.85,
                          edgecolor="lightgray")

    # ==========================================================
    # COLUMN 2 — Tendencies
    # ==========================================================

    tend_rows = [
        ("dq_v_dt", r"$\partial q_v/\partial t$  [μg kg$^{-1}$ s$^{-1}$]",
         r"$\partial q_v/\partial t$", 1e9, False),
        ("dq_c_dt", r"$\partial q_c/\partial t$  [μg kg$^{-1}$ s$^{-1}$]",
         r"$\partial q_c/\partial t$", 1e9, False),
        ("dq_r_dt", r"$\partial q_r/\partial t$  [μg kg$^{-1}$ s$^{-1}$]",
         r"$\partial q_r/\partial t$", 1e9, False),
        ("dq_i_dt", r"$\partial q_i/\partial t$  [μg kg$^{-1}$ s$^{-1}$]",
         r"$\partial q_i/\partial t$", 1e9, True),
        ("dq_s_dt",
         r"$\partial q_{snow}/\partial t$  (Mor/Tho)" + "\n"
         r"$\partial q_{rim}/\partial t$  (P3)  [μg kg$^{-1}$ s$^{-1}$]",
         r"$\partial q_s/\partial t$", 1e9, True),
    ]

    for row, (field, xlabel, col_title, scale, ice_only) in enumerate(tend_rows):
        ax = _profile_ax(row, 2)
        ax.set_title(col_title, fontsize=10, fontweight="bold")
        ax.set_xlabel(xlabel, fontsize=8)
        ax.axvline(0, color="black", lw=0.5, alpha=0.3)

        active = [(n, c, ls, lw) for n, _, _, c, ls, lw in SCHEMES
                  if not (ice_only and n not in ICE_SCHEMES)]
        for name, color, ls, lw in active:
            vals = np.array(getattr(results[name], field)[0]) * scale
            ax.plot(vals, p_hPa, color=color, ls=ls, lw=lw, label=name, alpha=0.9)

        _symlog_x(ax, [np.array(getattr(results[n], field)[0]) * scale
                       for n, *_ in SCHEMES
                       if not (ice_only and n not in ICE_SCHEMES)])

    # ==========================================================
    # Footer
    # ==========================================================

    fig.text(
        0.5, -0.008,
        "Gray dotted lines = T$_{freeze}$ / saturation reference.  "
        "Ice panels (dq$_i$, dq$_s$) show Morrison, Thompson, P3 only.  "
        "P3: q$_s$ slot → q$_{rim}$; q$_g$ slot → B$_{rim}$ (not shown).",
        ha="center", fontsize=8, color="dimgray",
    )

    out_path = f"microphysics_comparison_{case_key}.png"
    fig.savefig(out_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved: {out_path}")