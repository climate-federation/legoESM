"""Leaf-level comparison of legoESM's P-model / P-hydro capacities against the
original optimality codes (rpmodel, rphydro).

For every row of the reference table (scripts/validate/optimality_reference.R)
this evaluates, at the same T / VPD / PPFD / CO2 / soil psi, with leaf T = air
T (no energy balance, no canopy) and fully acclimated to the row:

  lc    least-cost lane (land/p_model.py): capacities + Medlyn g1
  ph    P-hydro ACCLIMATED optimum (land/phydro.py) at its default
        parameters (gamma = 1, as rphydro)

Each lane is reported twice: the optimum's own numbers ("c1": chi, dpsi, gs,
A, Vcmax, Jmax) and what the host FvCB kernel + Medlyn make of the injected
g1 / Vcmax25 / rjv25 ("host").  rphydro rows use the row's psi50 with
c_k = 1.5 and plant conductance = the row's ``k_scaled``; the P-hydro lanes are
given the same xylem curve and conductance, and no soil-to-root resistance.

Usage:
  python scripts/validate/optimality_gs_gap.py REF.csv OUT_DIR
"""

from __future__ import annotations

import sys
from pathlib import Path

import jax

jax.config.update("jax_enable_x64", True)

import jax.numpy as jnp
import numpy as np
import pandas as pd

from legoesm import constants
from legoesm.land.canopy.photosynthesis import (
    c3_photosynthesis, jmax_temperature_response, vcmax_temperature_response)
from legoesm.land.p_model import (
    PModelAcclimState, PModelConfig, acclimated_capacities, optimal_chi)
from legoesm.land.phydro import (
    PHydroConfig, PhydroSupply, phydro_optimum, slope_for_model)
from legoesm.land.stomata import medlyn_gs

PATM, CO2 = constants.p_atm_std, 400.0
B0_HOST = 0.01       # two_leaf_canopy._DEFAULT_B0_C3 [mol m-2 s-1]
GS_MIN = 1.0e-4      # canopy/energy_balance._GS_MIN_MOL


def host_leaf(tk, vpd_pa, ppfd, vcmax25, rjv25, g1):
    """Host FvCB + Medlyn Ci fixed point at leaf T = air T (bisection on Ci).

    Mirrors canopy/energy_balance._compute_gs_and_ci (default b0, Ci clip);
    returns (An [umol], gs_co2 [mol m-2 s-1], chi).
    """
    tgc = tk - constants.T_freeze

    def closure(ci):
        an = c3_photosynthesis(tk, ci, ppfd, vcmax25, PATM, 0.0, tgc, rjv25=rjv25)
        gsw = jnp.maximum(medlyn_gs(an, jnp.maximum(vpd_pa, 50.0) / 1e3, CO2, g1, B0_HOST), GS_MIN)
        return an, gsw, jnp.clip(CO2 - 1.6 * an / gsw, 0.5 * CO2, 0.9 * CO2)

    lo, hi = jnp.zeros_like(tk), jnp.full_like(tk, CO2)
    for _ in range(60):  # ci_new(ci) - ci is decreasing in ci
        mid = 0.5 * (lo + hi)
        up = closure(mid)[2] > mid
        lo, hi = jnp.where(up, mid, lo), jnp.where(up, hi, mid)
    an, gsw, ci = closure(0.5 * (lo + hi))
    return an, gsw / 1.6, ci / CO2


def evaluate(ref: pd.DataFrame) -> pd.DataFrame:
    pcfg = PModelConfig()
    a = lambda c: jnp.asarray(ref[c].to_numpy(), dtype=jnp.float64)
    tk, vpd, ppfd, psi_s = a("tc") + constants.T_freeze, a("vpd"), a("ppfd"), a("psi_soil")
    full = lambda v: jnp.full_like(tk, v)
    st = PModelAcclimState(iabs_mean=ppfd, t_mean_K=tk, vpd_mean_pa=vpd,
                           co2_mean_ppm=full(CO2), ps_ema=full(PATM))
    supply = PhydroSupply(rsoil_leaf=full(0.0), psi_s_mpa=psi_s)
    gstar = optimal_chi(tk, vpd, full(CO2), full(PATM), pcfg)[3]
    tg_c = tk - constants.T_freeze
    f_v, f_j = vcmax_temperature_response(tk, tg_c), jmax_temperature_response(tk, tg_c)
    xylem = dict(gplant_mol=a("k_scaled"), psi50_xylem_mpa=a("psi50"), ck_xylem=1.5)
    out = {}

    def record(lane, vc25, rjv, g1):
        out[f"{lane}_vcmax"], out[f"{lane}_jmax"] = vc25 * f_v, rjv * vc25 * f_j
        out[f"{lane}_g1"] = g1
        an, gs, chi = host_leaf(tk, vpd, ppfd, vc25, rjv, g1)
        out[f"{lane}_host_a"], out[f"{lane}_host_gs"], out[f"{lane}_host_chi"] = an, gs, chi

    lc = acclimated_capacities(st, pcfg)
    record("lc", lc.vcmax25_leaf, lc.rjv25, lc.g1_kpa)
    for lane, hcfg in (("ph", PHydroConfig(**xylem)),):
        ph = phydro_optimum(st, supply, hcfg, pcfg)
        record(lane, ph.vcmax25_leaf, ph.rjv25, slope_for_model("medlyn", ph.chi, st, gstar))
        out[f"{lane}_c1_chi"], out[f"{lane}_c1_dpsi"] = ph.chi, ph.dpsi_mpa
        out[f"{lane}_c1_psi_l"], out[f"{lane}_c1_gs"] = ph.psi_leaf_mpa, ph.gs_mol
        out[f"{lane}_c1_a"] = ph.gs_mol * CO2 * (1.0 - ph.chi)
        out[f"{lane}_converged"] = ph.converged
    return pd.concat([ref, pd.DataFrame({k: np.asarray(v) for k, v in out.items()})], axis=1)


_SERIES = {  # legend label, line style, {quantity: column}
    "rpmodel": ("k-", dict(gs="r1_gs", a="r1_a", vcmax="r1_vcmax", jmax="r1_jmax")),
    "rphydro (gamma=1, no Jmax stress)": ("C0-", dict(
        gs="r2acc_gs", a="r2acc_a", vcmax="r2acc_vcmax", jmax="r2acc_jmax", psi_l="r2acc_psi_l")),
    "code P-hydro, acclimated": ("C3:", dict(
        gs="ph_c1_gs", a="ph_c1_a", vcmax="ph_vcmax", jmax="ph_jmax", psi_l="ph_c1_psi_l")),
    "host (Medlyn g1 from chi), P-hydro": ("C3-", dict(gs="ph_host_gs", a="ph_host_a")),
    "host, least-cost": ("C1-", dict(gs="lc_host_gs", a="lc_host_a", vcmax="lc_vcmax", jmax="lc_jmax")),
}
_QUANTITIES = {"gs": "gs", "a": "An", "vcmax": "Vcmax", "jmax": "Jmax"}


def _slices(df):
    d = df[(df.tc == 20) & (df.ppfd == 1000) & (df.psi50 == -2.0)]
    return (d[d.psi_soil == 0].sort_values("vpd"), d[d.vpd == 1000].sort_values("psi_soil"),
            d[(d.psi_soil == 0) & (d.vpd == 1000)].iloc[0])


def summarise(df: pd.DataFrame) -> str:
    """Absolute baseline and normalised trends at 20 degC, PPFD 1000, psi50 -2."""
    wet, mid, base = _slices(df)
    blocks = []
    for q, name in _QUANTITIES.items():
        cols = {lab: c[q] for lab, (_, c) in _SERIES.items() if q in c}
        blocks.append(f"== {name}: baseline (wet soil, VPD 1 kPa) ==\n"
                      + pd.Series({lab: base[c] for lab, c in cols.items()}).round(3).to_string())
        for label, sl, idx in (("vs VPD [Pa], wet soil", wet, "vpd"), ("vs soil psi [MPa], VPD 1 kPa", mid, "psi_soil")):
            tab = pd.DataFrame({lab: sl[c].to_numpy() / base[c] for lab, c in cols.items()}, index=sl[idx])
            blocks.append(f"-- {name} / baseline, {label} --\n" + tab.round(2).to_string())
    blocks.append("-- leaf psi [MPa] vs soil psi, VPD 1 kPa --\n" + pd.DataFrame(
        {lab: mid[c["psi_l"]].to_numpy() for lab, (_, c) in _SERIES.items() if "psi_l" in c},
        index=mid.psi_soil).round(2).to_string())
    blocks.append(f"P-hydro acclimated optimum converged on {df.ph_converged.mean():.3f} of all rows")
    return "\n\n".join(blocks)


def plot(df: pd.DataFrame, path: Path) -> None:
    """Every curve divided by its own value at wet soil, VPD 1 kPa (top two
    rows per quantity: vs VPD, vs soil psi); last panel: leaf water potential."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    wet, mid, base = _slices(df)
    fig, ax = plt.subplots(2, 5, figsize=(24, 8.5))
    for i, (q, name) in enumerate(_QUANTITIES.items()):
        for lab, (sty, cols) in _SERIES.items():
            if q in cols:
                ax[0, i].plot(wet.vpd / 1e3, wet[cols[q]] / base[cols[q]], sty, label=lab)
                ax[1, i].plot(mid.psi_soil, mid[cols[q]] / base[cols[q]], sty, label=lab)
        ax[0, i].set(xlabel="VPD [kPa]", title=f"{name} / baseline, wet soil")
        ax[1, i].set(xlabel="soil psi [MPa]", title=f"{name} / baseline, VPD = 1 kPa")
    ax[0, 0].legend(fontsize=8)
    for lab, (sty, cols) in _SERIES.items():
        if "psi_l" in cols:
            ax[0, 4].plot(wet.vpd / 1e3, wet[cols["psi_l"]], sty, label=lab)
            ax[1, 4].plot(mid.psi_soil, mid[cols["psi_l"]], sty, label=lab)
    ax[1, 4].plot(mid.psi_soil, mid.psi_soil, "k:", label="1:1 (no draw-down)")
    ax[0, 4].set(xlabel="VPD [kPa]", title="leaf psi [MPa], wet soil")
    ax[1, 4].set(xlabel="soil psi [MPa]", title="leaf psi [MPa], VPD = 1 kPa")
    ax[1, 4].legend(fontsize=8)
    fig.suptitle("T = 20 degC, PPFD = 1000; xylem psi50 = -2 MPa, c_k = 1.5, K = 4e-3; "
                 "baseline = wet soil (psi_s = 0), VPD 1 kPa")
    fig.tight_layout()
    fig.savefig(path, dpi=120)


def main(ref_csv: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    df = evaluate(pd.read_csv(ref_csv))
    df.to_csv(out / "optimality_gs_gap.csv", index=False)
    text = summarise(df)
    (out / "optimality_gs_gap_summary.txt").write_text(text + "\n")
    print(text)
    plot(df, out / "optimality_gs_gap.png")


if __name__ == "__main__":
    main(*sys.argv[1:3])
