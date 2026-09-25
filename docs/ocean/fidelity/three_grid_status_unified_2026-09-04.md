# Three-grid vs NEMO GATEWAY — UNIFIED arms (common 1-deg mask, matched days)

Arms: tripole=trp_unified180, MPAS=mpas_unified_180d, FESOM=fesom_b5_d90. All three carry the SI3 lead-heat budget (SST min -1.8 C, zero supercooled cells).
Supersedes threegrid_table_2026-09-04.md, whose tripole/MPAS arms predated that fix (SST to -10 C under ice).

## SST — global rmse / bias vs NEMO, and grid-vs-grid rmse
| day | tripole | MPAS | FESOM | trp-MPAS | trp-FESOM | MPAS-FESOM |
|---|---|---|---|---|---|---|
| 30 | 0.48 / +0.07 | 0.46 / -0.00 | 0.44 / +0.02 | 0.25 | 0.39 | 0.38 |
| 60 | 0.43 / +0.04 | 0.42 / -0.04 | 0.44 / -0.01 | 0.37 | 0.49 | 0.50 |
| 90 | 0.37 / +0.02 | 0.43 / -0.05 | 0.47 / -0.03 | 0.43 | 0.48 | 0.55 |

### SST band bias vs NEMO (day 30)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | +0.19 | +0.16 | +0.25 |
| SH mid | +0.23 | +0.11 | +0.25 |
| Tropics | +0.13 | +0.07 | +0.01 |
| NH mid | -0.41 | -0.45 | -0.44 |
| Arctic | -0.12 | -0.20 | -0.17 |

### SST band bias vs NEMO (day 90)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | -0.04 | -0.01 | -0.01 |
| SH mid | -0.03 | -0.11 | +0.01 |
| Tropics | +0.09 | +0.03 | +0.01 |
| NH mid | -0.09 | -0.22 | -0.26 |
| Arctic | +0.10 | -0.10 | -0.04 |

## SSS — global rmse / bias vs NEMO, and grid-vs-grid rmse
| day | tripole | MPAS | FESOM | trp-MPAS | trp-FESOM | MPAS-FESOM |
|---|---|---|---|---|---|---|
| 30 | 0.30 / -0.01 | 0.34 / -0.00 | 0.54 / +0.03 | 0.19 | 0.50 | 0.52 |
| 60 | 0.35 / -0.01 | 0.43 / -0.00 | 0.56 / +0.03 | 0.28 | 0.58 | 0.68 |
| 90 | 0.37 / -0.01 | 0.45 / -0.01 | 0.53 / +0.03 | 0.36 | 0.61 | 0.73 |

### SSS band bias vs NEMO (day 30)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | +0.02 | +0.03 | +0.06 |
| SH mid | +0.00 | +0.01 | +0.01 |
| Tropics | -0.02 | -0.02 | +0.00 |
| NH mid | -0.01 | +0.01 | +0.00 |
| Arctic | +0.00 | -0.02 | +0.21 |

### SSS band bias vs NEMO (day 90)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | +0.02 | +0.03 | -0.01 |
| SH mid | -0.02 | -0.01 | -0.00 |
| Tropics | -0.05 | -0.04 | +0.02 |
| NH mid | -0.02 | +0.00 | -0.01 |
| Arctic | +0.12 | +0.05 | +0.23 |

## MLD — global rmse / bias vs NEMO, and grid-vs-grid rmse
| day | tripole | MPAS | FESOM | trp-MPAS | trp-FESOM | MPAS-FESOM |
|---|---|---|---|---|---|---|
| 30 | 20.95 / +2.78 | 23.62 / +4.55 | 27.25 / +2.95 | 10.89 | 22.47 | 22.09 |
| 60 | 29.31 / +3.27 | 34.95 / +6.20 | 44.22 / +4.44 | 20.20 | 40.53 | 40.74 |
| 90 | 35.61 / +3.38 | 40.79 / +6.78 | 55.72 / +4.89 | 25.64 | 52.89 | 50.50 |

### MLD band bias vs NEMO (day 90)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | +12.8 | +16.4 | -1.2 |
| SH mid | +7.1 | +9.4 | +5.3 |
| Tropics | -4.3 | -3.7 | -2.9 |
| NH mid | -3.4 | +6.1 | +5.8 |
| Arctic | +22.6 | +31.8 | +47.8 |

## Instrument cross-check (GLM 'mask artifact' hypothesis)
Arctic (N of 45N) SST bias on the unified arms: three-way (common mask, no clamp) vs native single-arm (own mask, freeze clamp). Same snapshot, same NEMO record.
| day | tripole 3way | tripole native | MPAS 3way | MPAS native |
|---|---|---|---|---|
| 30 | -0.12 (NEMO mean 2.44) | -0.07 (NEMO mean 1.80) | -0.20 (NEMO mean 2.44) | -0.17 (NEMO mean 1.81) |
| 60 | +0.02 (NEMO mean 2.12) | — | -0.11 (NEMO mean 2.12) | -0.09 (NEMO mean 1.51) |

## Reviews (claim: "northern cold bias = pre-fix arms, not a regression")
- codex 9631633: CONFIRMED legacy lead-budget cooling, not a regression. CRITICAL: freeze clamp in
  compare_omip_nemo.py only warns (exit 0) -> proposes FAIL verdict on any supercooled cell (USER
  DECISION pending). CRITICAL: pair masks differ (42225 vs 42541 cells) -> four-way single-mask
  figures added (fourway_uni_d*). MAJOR: audit the unified arm's ice heat budget before exonerating
  other sinks.
- GLM: CONFIRMED (a)(b)(c); "mask artifact" hypothesis REFUTED by the cross-check table above
  (three-way vs native agree within 0.05 on the unified arms); hidden-sink check = ice volume.
- Ice volume vs NEMO SI3 restart at d90 (scripts/validate/ocean_fidelity/ice_volume_vs_nemo_gateway.py):
  NH volume NEMO 27.3e3 km3, unified 21.8e3, pre-fix 21.4e3; NH area 15.1 / 12.9 / 12.6 1e6 km2.
  Hidden-sink hypothesis REFUTED (our ice is 20% BELOW NEMO, not above). Pre-fix freezing
  deficit = 267 km3 ice-equivalent vs measured +365 km3 (unified - pre-fix): bookkeeping closes
  to order. OPEN LEVER: winter NH ice growth deficit (-2.2e6 km2 area at 1 April).

## 2026-09-04 afternoon: two root causes found (both reviewed by codex + GLM)
1. LEVEL-8 (29 km MPAS) blowup = the explicit vorticity biharmonic K_zeta_bih=1e14 (NEMO-match recipe,
   fixed, not dx-scaled) past its stability limit (number ~1.4 at level 7, ~11 at level 8). Arms with
   1.25e13 (dx^3) or 0 are stable. Nine other operators exonerated by one-variable arms. Also fixed
   silent MPAS flag drops (--no-gm-redi, --A-h/--B-h/--K-bih/--C-smag-lap); --pgf-scheme still inert.
2. REAL-FRESHWATER closure never dilutes the surface: tracer w is diagnosed from horizontal divergence
   only, so freshwater stretches the column uniformly (O(F/H)); brine alone acts at the poles. Day-15
   real-minus-virtual: Amazon +2.2 PSU, Weddell +0.43, Kara -0.53 (surface-trapped). Tripole has the
   NEMO-literal w (wzv_call2_evaluation) but OMIP cards run 'generic'; MPAS lacks it. Exact 2-layer
   test spec recorded. Twin divergence at d30: 0.001 C.
