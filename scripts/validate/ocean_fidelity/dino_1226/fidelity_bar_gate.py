#!/usr/bin/env python
"""Machine-enforced fidelity bar for the #1226 NEMO term sweep.

The standing user directive is corr == 1.0 AND ratio == 1.0.  Repeatedly, a
term measured at 0.99x was recorded as "MATCHED"/"FAITHFUL"/"CLOSED" and the
campaign moved on.  This gate removes the judgment call: a term is DONE only if
its measured numbers meet the bar, otherwise it is DEBT -- regardless of what
prose was written about it.

Run:  python scripts/validate/ocean_fidelity/dino_1226/fidelity_bar_gate.py
Exit 0 only when every term is at the bar.  Non-zero (and a printed table)
otherwise.  Update MEASUREMENTS as terms are re-measured; never relax BAR_*.
"""
from __future__ import annotations

import sys

# The bar.  Roundoff only -- these are NOT tolerances for physics differences.
BAR_CORR = 1.0 - 1e-9
BAR_RATIO_EPS = 1e-6

# term -> (corr, ratio, note).  corr/ratio None = never measured at all.
MEASUREMENTS: dict[str, tuple[float | None, float | None, str]] = {
    # --- sweep terms ---
    "sbc (utau/qsr/qns/sfx)":        (1.0,        1.0,        "exact"),
    "eos_rab beta":                  (1.0,        1.0,        "bit-exact"),
    "eos_rab alpha":                 (1.0,        1.000007,   "median rel 4.7e-6"),
    "bn2 (rn2b)":                    (1.0,        0.999993,   "median |rel| 6.96e-6"),
    "zdf_mxl (nmln)":                (0.99859,    None,       "12/9920 cols differ; REOPENED"),
    "ldf_slp wslpi":                 (0.999177,   1.003000,   "interior 1.0011, bottom row 1.0255; REOPENED"),
    "ldf_slp wslpj":                 (0.998875,   0.998447,   ""),
    "ldf_slp uslp":                  (0.999340,   0.998697,   ""),
    "ldf_slp vslp":                  (0.999012,   0.997594,   ""),
    "ldf_eiv kappa (aeiu)":          (1.0,        1.000608,   "ratio not 1"),
    "ldftra ahtu (Redi, nn_aht_ijk_t=20)": (1.0,  1.0,        "AT BAR: K_h_base*cos(lat_T) u-face avg is a same-row no-op -> exact vs NEMO gphiu"),
    "ldftra ahtv (Redi, nn_aht_ijk_t=20)": (1.0,  1.0000260,  "v-face interpolates avg(cos) not cos(avg): ~2.6e-5 rel, second-order in dlat"),
    "eiv transport u":               (0.998474,   0.994097,   "REOPENED"),
    "eiv transport v":               (0.995437,   0.982351,   "REOPENED - 1.8% low"),
    "traadv_fct fluxes":             (0.99994,    1.000100,   ""),
    "traadv_fct tendency (T)":       (0.994500,   None,       "after nonosc bound fix 2a73221ce"),
    "traadv_fct horizontal tend":    (0.999950,   None,       "limiter itself now correct"),
    "traadv_fct VERTICAL upstream flux": (0.910000, None,     "NEW DEFECT - w over-clip 1.7-1.9x; diapycnal-mixing candidate"),
    "traadv_fct (SALINITY)":         (None,       None,       "NEVER COMPARED"),
    "dyn_hpg":                       (1.0,        1.000045,   "ratio not 1"),
    "dyn_vor EEN u":                 (0.999896,   1.001180,   "bottom levels 1.04-1.07; REOPENED"),
    "dyn_vor EEN v":                 (0.999932,   1.000717,   "REOPENED"),
    "dyn_adv KEG":                   (1.0,        1.000000,   "byte-exact"),
    "dyn_adv ZAD":                   (0.999200,   0.995000,   "after nemo_advective fix"),
    "zdftke pdlr":                   (0.998120,   0.996800,   "REOPENED"),
    "zdftke composite avt/avm":      (0.997560,   0.996197,   "257 cells; REOPENED"),
    # --- round 2 ---
    "dyn_spg_ts pssh":               (0.999989,   0.999900,   ""),
    "dyn_spg_ts puu_b":              (0.999586,   0.987100,   "1.3% gap UNEXPLAINED"),
    "dyn_spg_ts un_adv":             (0.999777,   1.006700,   ""),
    "ATF filter u":                  (0.999969,   0.995500,   "0.45% gap"),
    "ATF filter v":                  (0.999999,   1.000400,   ""),
    "ATF filter T/S/ssh":            (1.0,        1.0,        "exact"),
    "dyn_ldf (dynldf_lev_lap) u":    (0.997900,   1.003900,   ""),
    "dyn_ldf (dynldf_lev_lap) v":    (0.999400,   1.001800,   ""),
    "ssh_nxt / div_hor":             (1.0,        0.999991,   ""),
    "dom_qco_r3c r3t":               (1.0,        0.999997,   ""),
    "dom_qco_r3c r3u/r3v":           (None,       None,       "NEVER COMPARED (reader lacks hu_0/hv_0)"),
    "mlf_baro_corr":                 (None,       None,       "algebra only; needs _step_impl hook"),
    "lbc_lnk sign":                  (None,       None,       "NEVER VERIFIED"),
    "zdf_mxl_turb":                  (None,       None,       "NEVER VERIFIED"),
    "zdf_drg_nonlin / dyn_drg_init": (None,       None,       "NEVER TERM-ISOLATED"),
    "dyn_cor_2d (69x/step)":         (None,       None,       "NEVER TERM-ISOLATED"),
}


def classify(corr: float | None, ratio: float | None) -> str:
    if corr is None or ratio is None:
        return "UNMEASURED"
    if corr >= BAR_CORR and abs(ratio - 1.0) <= BAR_RATIO_EPS:
        return "AT BAR"
    return "DEBT"


def main() -> int:
    rows = [(t, c, r, n, classify(c, r)) for t, (c, r, n) in MEASUREMENTS.items()]
    width = max(len(t) for t, *_ in rows)
    print(f"{'term':<{width}}  {'corr':>12} {'ratio':>12}  status")
    print("-" * (width + 42))
    for term, corr, ratio, note, status in rows:
        cs = "     n/a    " if corr is None else f"{corr:>12.6f}"
        rs = "     n/a    " if ratio is None else f"{ratio:>12.6f}"
        flag = "" if status == "AT BAR" else f"  <-- {status}"
        print(f"{term:<{width}}  {cs} {rs}{flag}" + (f"   ({note})" if note else ""))

    at_bar = sum(s == "AT BAR" for *_, s in rows)
    debt = sum(s == "DEBT" for *_, s in rows)
    unmeasured = sum(s == "UNMEASURED" for *_, s in rows)
    print(f"\nAT BAR {at_bar} | DEBT {debt} | UNMEASURED {unmeasured} | total {len(rows)}")
    print(f"bar: corr >= {BAR_CORR}, |ratio - 1| <= {BAR_RATIO_EPS}")
    if debt or unmeasured:
        print("\nFAIL: the sweep is NOT complete. Do not describe these as "
              "'matched', 'faithful', 'closed' or 'good enough'.")
        return 1
    print("\nPASS: every term at the bar.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
