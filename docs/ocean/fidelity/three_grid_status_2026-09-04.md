# Three-grid vs NEMO GATEWAY (common 1-deg mask, matched days)

Arms: tripole=trp_visc180, MPAS=mpas_fullharm_180d, FESOM=fesom_b5_d90 (config caveat: FESOM carries iwm; others predate isf/iwm/bbl).


## SST — global rmse / bias vs NEMO, and grid-vs-grid rmse
| day | tripole | MPAS | FESOM | trp-MPAS | trp-FESOM | MPAS-FESOM |
|---|---|---|---|---|---|---|
| 30 | 0.53 / +0.04 | 0.50 / -0.02 | 0.44 / +0.02 | 0.27 | 0.45 | 0.40 |
| 60 | 0.54 / -0.01 | 0.50 / -0.07 | 0.44 / -0.01 | 0.39 | 0.58 | 0.55 |
| 90 | 0.53 / -0.03 | 0.54 / -0.08 | 0.47 / -0.03 | 0.46 | 0.59 | 0.62 |

### SST band bias vs NEMO (day 90)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | -0.05 | -0.01 | -0.01 |
| SH mid | -0.03 | -0.08 | +0.01 |
| Tropics | +0.09 | +0.04 | +0.01 |
| NH mid | -0.09 | -0.22 | -0.26 |
| Arctic | -0.43 | -0.59 | -0.04 |

## SSS — global rmse / bias vs NEMO, and grid-vs-grid rmse
| day | tripole | MPAS | FESOM | trp-MPAS | trp-FESOM | MPAS-FESOM |
|---|---|---|---|---|---|---|
| 30 | 0.29 / -0.01 | 0.34 / +0.01 | 0.54 / +0.03 | 0.22 | 0.50 | 0.44 |
| 60 | 0.34 / -0.01 | 0.42 / +0.01 | 0.56 / +0.03 | 0.29 | 0.59 | 0.56 |
| 90 | 0.36 / -0.01 | 0.46 / -0.00 | 0.53 / +0.03 | 0.34 | 0.61 | 0.67 |

### SSS band bias vs NEMO (day 90)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | +0.02 | +0.03 | -0.01 |
| SH mid | -0.02 | -0.01 | -0.00 |
| Tropics | -0.05 | -0.06 | +0.02 |
| NH mid | -0.02 | -0.00 | -0.01 |
| Arctic | +0.10 | +0.21 | +0.23 |

## MLD — global rmse / bias vs NEMO, and grid-vs-grid rmse
| day | tripole | MPAS | FESOM | trp-MPAS | trp-FESOM | MPAS-FESOM |
|---|---|---|---|---|---|---|
| 30 | 20.98 / +2.80 | 23.53 / +4.48 | 27.25 / +2.95 | 10.37 | 22.44 | 23.59 |
| 60 | 29.51 / +3.31 | 34.24 / +6.06 | 44.22 / +4.44 | 17.76 | 40.48 | 41.83 |
| 90 | 35.77 / +3.41 | 40.13 / +6.67 | 55.72 / +4.89 | 22.82 | 52.83 | 51.59 |

### MLD band bias vs NEMO (day 90)
| band | tripole | MPAS | FESOM |
|---|---|---|---|
| Antarctic | +12.81 | +16.32 | -1.16 |
| SH mid | +7.12 | +9.76 | +5.34 |
| Tropics | -4.35 | -3.65 | -2.92 |
| NH mid | -3.34 | +5.22 | +5.82 |
| Arctic | +22.88 | +31.20 | +47.83 |

## Review disposition (GLM 2026-09-04) — CLAIMS RETRACTED/RELABELLED
- RETRACTED: "grid-to-grid spread ~ model-vs-NEMO error => harmonization at floor".
  MPAS-FESOM 0.62 EXCEEDS every arm's NEMO error (<=0.54); the arms are config-
  incommensurable (FESOM carries iwm + CVMix-TKE + EOS-80; tripole/MPAS arms here predate
  isf/iwm/bbl). The spread is an UPPER BOUND on the physics-card floor. Reruns: real-FW
  unified pair (jobs 9630736/9631426) + FESOM b5 give the commensurable trio.
- RELABELLED PLAUSIBLE: tripole nino3 outlier (+0.53 vs ~+0.1) "grid-intrinsic" — equatorial
  resolution ordering (FESOM 25 km < tripole 0.33 deg < MPAS 58 km) does NOT track the bias
  ordering. Counterfactual running: MPAS level-8 chain (9631490..).
- MEASURED: FESOM tidal mixing share of its Arctic MLD ~0 (b4 no-iwm +7.1 m vs b5 +7.3 m at
  d30, SST identical) — iwm exonerated; closure (CVMix-TKE) vs EOS-80 still collinear ->
  single-column twin remains the discriminator.
- BLOCKER accepted: round-off twin. d30 twin divergence measured ~1e-4 C (SHA-replication
  pairs, both lanes); d90/d180 twin launched (MPAS identical card, job 9631508).
