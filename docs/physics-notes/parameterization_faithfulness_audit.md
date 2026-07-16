# Parameterization oracle-faithfulness audit — status

Running program: for **every** physics parameterization, locate the
most-trustworthy oracle (public reference code where it exists, else the
canonical published equations), audit the JAX implementation term-by-term, add a
`Faithfulness` docstring section, and pin the closed forms to round-off
(rel `1e-9`, abs `0`) against an **independent** scalar reimplementation — every
departure carried as an explicit canary. Each scheme is driven through Codex
adversarial review to `APPROVE` before merge.

**Change type:**
- **Audit** — non-behavioral: `Faithfulness` docstring + faithfulness tests only;
  the JAX numerics are byte-identical. The scheme's over-claims (if any) are
  reframed and its forms/departures pinned.
- **Behavioral** — the numerics changed (a bug fix, a faithful form added, or an
  AD-safety fix), pinned by the same tests.

Every row reached Codex `APPROVE`. Oracle clones (IFS `ifs-source/`, gSAM
`~/Documents/Code/gSAM/…`) are gitignored and never committed — cited by path only.

## Convection

| Scheme (module) | Most-trustful oracle | Type | PR |
|---|---|---|---|
| Bechtold (`convection/bechtold.py`) | IFS / OpenIFS cy48 (`ifs-source/`) | Behavioral — turnover-τ, cloud-base qsat, RH cap, mid-level ε | #1018 |
| Tiedtke 1989 (`convection/tiedtke.py`) | Tiedtke (1989) / IFS | Audit | #1021 |
| Zhang-McFarlane (`convection/zhang_mcfarlane.py`) | Zhang & McFarlane (1995) + Raymond-Blyth dilute CAPE / E3SM | Audit | #1026 |

## Turbulence / PBL

| Scheme (module) | Most-trustful oracle | Type | PR |
|---|---|---|---|
| Louis (`turbulence/louis.py`) | Louis (1979) / Louis-Tiedtke-Geleyn (1982) | Audit | #1024 |
| YSU (`turbulence/ysu.py`) | Hong et al. (2006) / WRF | Audit | #1032 |
| Smagorinsky-Lilly (`turbulence/smagorinsky.py`) | Smagorinsky (1963) / Lilly (1962) | Audit | #1034 |
| Prognostic TKE (`turbulence/tke.py`) | Mellor & Yamada (1982) level-2.5 | Audit | #1035 |
| MYNN-2.5 (`turbulence/mynn25.py`) | Nakanishi & Niino (2009) | Audit | #1036 |
| EDMF (`turbulence/edmf.py`) | Siebesma-Soares-Teixeira (2007) | Audit | #1037 |

## Gravity-wave drag

| Scheme (module) | Most-trustful oracle | Type | PR |
|---|---|---|---|
| McFarlane orographic (`gravity_wave_drag/mcfarlane.py`) | McFarlane (1987) | Audit (+ conserves fix #1045) | #1022 |
| Hines (`gravity_wave_drag/hines.py`) | Hines (1997) | Audit | #1038 |
| Lindzen saturation (`gravity_wave_drag/lindzen.py`) | Lindzen (1981) + McFarlane (1987) launch | Audit (+ conserves fix #1046) | #1039 |
| Rayleigh friction (`gravity_wave_drag/rayleigh.py`) | Held & Suarez (1994) | Audit | #1040 |
| E3SM/CAM spectral (`gravity_wave_drag/e3sm_cam.py`) | E3SM/CAM `gw_drag` | Audit | #1041 |
| Prognostic spectral (`gravity_wave_drag/prognostic_spectral.py`) | spectral saturation (experimental) | Audit | #1042 |
| ML emulator (`gravity_wave_drag/ml_emulator.py`) | learned Equinox MLP (canary pins) | Audit | #1044 |

## Radiation

| Scheme (module) | Most-trustful oracle | Type | PR |
|---|---|---|---|
| Gray (`radiation/gray.py`) | Frierson / Isca `two_stream_gray_rad.F90` | Audit | #1023 |
| RRTMGP two-stream (`radiation/rrtmgp/…`) | RTE-RRTMGP two-stream solver | Audit | #1048 |

## Microphysics / clouds

| Scheme (module) | Most-trustful oracle | Type | PR |
|---|---|---|---|
| Sundqvist (`microphysics/sundqvist.py`) | Sundqvist (1978) / SBK89 | Audit | #1025 |
| Thompson-2008 (`microphysics/thompson.py`) | Thompson et al. (2008) / WRF | Audit | #1027 |
| Xu-Randall cloud fraction (`clouds/cloud_fraction.py`) | Xu & Randall (1996) + Sundqvist | Audit | #1030 |
| Kessler warm-rain (`microphysics/kessler.py`) | Kessler (1969) | Audit | #1031 |
| Morrison ice deposition PRD (`microphysics/morrison.py`) | Morrison et al. (2005) M2005 | Audit | #1050 |
| Seifert-Beheng warm-rain (`microphysics/seifert_beheng.py`) | Seifert & Beheng (2001) | Behavioral — faithful φ_au/φ_ac added | #1051 |
| P3 Cooper nucleation (`microphysics/p3.py`) | Morrison-Milbrandt P3 / gSAM P3 | Audit — Cooper pinned, table ice rates documented as surrogate | #1052 |
| Thompson snow fall speed (`microphysics/_thompson_snow.py`) | gSAM/WRF `module_mp_thompson.f90` + Field (2005) | Behavioral — f32 NaN-grad fix + `vts` pin | #1054 |
| ARG2000 aerosol activation (`microphysics/arg_activation.py`) | Abdul-Razzak & Ghan (2000) / gSAM M2005 | Audit | #1055 |

## Land

| Scheme (module) | Most-trustful oracle | Type | PR |
|---|---|---|---|
| Soil hydraulics (`land/soil_hydraulics.py`) | Clapp-Hornberger (1978) / gSAM SLM `soil_proc.f90` | Audit | #1056 |
| FvCB C3/C4 photosynthesis (`land/canopy/photosynthesis.py`) | Farquhar-von Caemmerer-Berry (1980) + Collatz C4 / Bernacchi | Audit | #1057 |
| Canopy shortwave two-stream RT (`land/canopy/radiative_transfer.py`) | Sellers (1985) + de Pury & Farquhar sunlit/shaded + Erbs beam split | Audit | #1058 |
| Ball-Berry / Medlyn stomata (`land/stomata.py`) | Ball et al. (1987) + Medlyn et al. (2011) | Audit | #1059 |
| Canopy MOST surface layer (`land/canopy/stability.py`) | CLM5 `FrictionVelocityMod.F90` (Businger-Dyer 4-regime ψ) | Audit — exact CLM5 ψ_m/ψ_h forms pinned; kB⁻¹=0 + bulk-Ri driver documented as departures | #1063 |
| Cosby pedotransfer (`land/pedotransfer.py`) | Cosby et al. (1984) / gSAM SLM `slm_vars.f90` | Audit — 4 Clapp-Hornberger regressions pinned; ψ_sat uncap + K_sat prefactor departures | #1065 |
| SIMTOP TOPMODEL runoff (`land/topmodel_runoff.py`) | Niu et al. (2005) SIMTOP / CLM4.5 §7 | Audit — forms + water-budget conservation pinned; factor-2 f_over decay + frozen-soil omission documented as CLM departures | #1066 |

## Sea ice

| Scheme (module) | Most-trustful oracle | Type | PR |
|---|---|---|---|
| VP/EVP/mEVP rheology (`ice/rheology.py`) | Hibler (1979) + Hunke & Dukowicz (1997) EVP + Bouillon (2013) mEVP | Audit | #1060 |
| Snow conduction + flooding (`ice/snow.py`) | Semtner (1976) / Maykut-Untersteiner (1971) + Leppäranta (1983) / Notz (2002) | Audit | #1061 |
| Lipscomb ridging (`ice/ridging.py`) | Lipscomb et al. (2007) eq. 22 | Audit — participation form + ITD-redistribution conservation invariants pinned | #1064 |
| Lipscomb ITD remap (`ice/itd.py`) | Lipscomb (2001) eqs. 14-15 | Behavioral — mean→bin-centre g(h) anchor fix (removes negative g in the central third); cutoff-support outer-third deferred (would lose mass on degenerate displaced bins) | #1067 |

## In progress / next

| Target (module) | Most-trustful oracle | Notes |
|---|---|---|
| Canopy SIF (`land/canopy/sif.py`) | van der Tol et al. (2014) SCOPE fluorescence | Check prior SIF work (PRs #845/#851) before re-pinning. |
| RRTMGP gas/cloud optics (`radiation/rrtmgp/…`) | RTE-RRTMGP reference optics | Correlated-k lookup; the two-stream solver is already pinned (#1048). |
| CLM5 snow compaction (`land/…snow…`) | Anderson (1976) / CLM4.5-5 destructive + overburden metamorphism | Published compaction/aging rates. |
| Exact Lipscomb cutoff-support (`ice/itd.py`) | Lipscomb (2001) eqs. 14-15 | Follow-up from #1067: outer-third triangle + robust delta/two-point fallback for degenerate displaced bins. |

## Notes

- The shared air-sea helper `core/bulk_flux.py` (COARE3.0, Large-Yeager) is
  already oracle-pinned by `tests/unit/test_aerobulk_oracle.py` (frozen AeroBulk
  Fortran baseline) and `tests/unit/test_sam_oceflx.py` (gSAM `oceflx.f90`); the
  untested MOST path is the CLM5 canopy copy above.
- Ocean fidelity (veros / MITgcm / Oceananigans) is covered by the separate
  mature harness under `docs/ocean/fidelity/`; this table tracks
  atmosphere / land / ice physics parameterizations.
