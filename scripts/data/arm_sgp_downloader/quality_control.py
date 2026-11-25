"""Filters for derived products only, plus scientific checks. Raw files and flags are never changed."""
import numpy as np

# Loose physical limits used only in the filtered product.
LIMITS = {"T": (-40.0, 50.0), "RH": (0.0, 105.0), "H": (-300.0, 800.0), "LE": (-200.0, 1000.0), "ustar": (0.0, 3.0),
          "SW_down": (-10.0, 1500.0), "LW_down": (100.0, 600.0)}


def apply_flags(table):
    """Blank values their file rejects, and anything outside LIMITS; suspect values stay, with their flag."""
    clean = table.copy()
    for name in table.columns:
        if name.startswith("qc_"):
            continue
        flag = "qc_fail_" + name
        if flag in table:
            clean.loc[table[flag] == 1, name] = np.nan
        low, high = LIMITS.get(name, LIMITS.get(name.split("_")[0], (-np.inf, np.inf)))
        clean.loc[(clean[name] < low) | (clean[name] > high), name] = np.nan
    return clean


def regression(a, b):
    good = np.isfinite(a) & np.isfinite(b)
    if good.sum() < 3:
        return {"n": int(good.sum())}
    slope, offset = np.polyfit(a[good], b[good], 1)
    return {"n": int(good.sum()), "slope": float(slope), "offset": float(offset),
            "r": float(np.corrcoef(a[good], b[good])[0, 1]), "mean_difference": float(np.mean(b[good] - a[good]))}


def closure(ds):
    """Regress H+LE on Rn-G with the surface ECOR fluxes; a slope of one is closed."""
    return regression((ds.Rn - ds.G).values, (ds.H_surface + ds.LE_surface).values)


def lag_of_best_match(a, b, max_lag=4):
    """Lag in 30-minute steps maximising correlation; zero supports synchronised clocks."""
    scores = {}
    for lag in range(-max_lag, max_lag + 1):
        x, y = a[max(0, -lag):len(a) - max(0, lag)], b[max(0, lag):len(b) - max(0, -lag)]
        good = np.isfinite(x) & np.isfinite(y)
        if good.sum() > 10:
            scores[lag] = float(np.corrcoef(x[good], y[good])[0, 1])
    return max(scores, key=scores.get) if scores else None


def checks(ds):
    h = ds.height.values
    result = {"missing_fraction": {v: float(ds[v].isnull().mean()) for v in ds.data_vars},
              "energy_closure": closure(ds),
              "ecor_vs_ebbr_H": regression(ds.H_surface.values, ds.H_ebbr.values),
              "ecor_vs_ebbr_LE": regression(ds.LE_surface.values, ds.LE_ebbr.values),
              "tower_4m_vs_ecor_H": regression(ds.H_surface.values, ds.H.sel(height=4).values),
              "clock_lag_ecor_ebbr_steps": lag_of_best_match(ds.H_surface.values, ds.H_ebbr.values)}
    pairs = [(int(a), int(b)) for a, b in zip(h[:-1], h[1:])]
    result["vertical_T_step_abs_p99_K"] = {
        f"{a}-{b}": float(np.nanpercentile(np.abs(ds.T.sel(height=b) - ds.T.sel(height=a)), 99))
        for a, b in pairs if ds.T.sel(height=a).notnull().any() and ds.T.sel(height=b).notnull().any()}
    return result
