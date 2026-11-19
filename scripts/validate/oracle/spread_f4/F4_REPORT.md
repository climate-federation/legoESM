# F4 batch 1 — oracle round-off spread, atmosphere units > 1e-12 (main 23327ffe42)

Label: empirical round-off spread (20 seeds). Harness: oracle_audit/f4 (committed copy on branch
feat/f4-oracle-spread, scripts/validate/oracle/spread_f4). Votes: VOTES.md. Artifacts: runs/<family>/spread_<family>.json.
All five controls PASS for every family below (C1 base rerun == committed fixture bitwise).

| unit | comparisons | class | worst port rel / spread | job |
|---|---|---|---|---|
| zm_detrain (CAM6 ZM detrainment) | 5 | A 5 | port 0 at the hard limit (reaudit's 1.10e-12 is not an oracle comparison) | 11025965 |
| rad_boundary (shr_orb + ocean albedo) | 4 | A 4 | declination 5.96e-11 / 2.94e-9 | 11025966 |
| louis (OpenIFS stable + ICON Louis79 unstable) | 4 (2 Ri input pins excluded) | A 2, **C 2** | ICON unstable kh 3.01e-11 / 1.20e-15, km 2.76e-11 / 1.24e-15 | 11026047 |
| kuo_mahfouf | 12 | A 10, **C 2** | g32 dT_dt: anthes 1.33e-12 / 7.95e-13, kuo1965 1.34e-12 / 1.18e-12 | 11026099 |
| kain_fritsch (WRF KF-Eta) | 576 | A 566, UNMEASURED 10 | sweep columns fire/unfire under 1 ulp (discrete rule V7) | 11026100 |
| mynn25 | 246 | A 241, UNMEASURED 5 | covariance zero pattern flips under 1 ulp (V7); worst l65 out 8.1e-12 | 11026098 |
| mg2 prognostic N_c whole call (V9 local ref) | 9 | A 6, UNMEASURED 3 | evaporation zero patterns flip under 1 ulp (V7); dq_r 1.86e-11 | 11028600 |
| mg2 prognostic N_c act (V9 local ref) | 3 | A 2, UNMEASURED 1 | dN_c 3.28e-11; same flips | 11028601 |
| mg2 prognostic N_c sed (V9 local ref) | 6 | A 2, UNMEASURED 4 | qctend/nctend 3.28e-11; same flips | 11028602 |

Class C for the F3 forks (report only, not fixed): Louis ICON unstable km/kh; Kuo g32 dT_dt (both partitions).


| CLUBB core, l_implemented arm (5 cases incl. bomex_bigup2) | 115 | A 115 | bomex_bigup2 wp2thvp 3.81e-11 / 6.66e-6; C1-C4 PASS | 11028707 |

CLUBB harness: oracle_audit/f4/clubb_impl = committed generator + the reviewed spread adaptation of
spread/clubb_core/gen_spread.py, sharing the generator's l_implemented call (codex NO FINDINGS).

CAM6 sequencing, static audit (V19, read off code, PLAUSIBLE): CAM cam_cesm2_1_rel_60 tphysbc runs deep conv ->
shallow conv -> macmic x N (CLUBB, microp_aero, MG2) -> wetdep -> radiation; tphysac runs vertical diffusion ->
Rayleigh -> drydep -> gw_tend -> qbo -> physics_dme_adjust.  Ours (combined.py): macmic sub-cycle ported (production
deck N=3), but radiation and GWD see the PRE-physics state (documented departure), and no physics_dme_adjust
equivalent was found by name.  Verdict: C-structural (sequencing departures), closest = CAM physpkg.F90.

## Part 2 (V10-V19)
| unit | verdict | closest published reference / oracle |
|---|---|---|
| TKE (MY-2.5-inspired k-l) | C-structural | WRF MYJ module_bl_myjpbl (MY82 level 2.5) |
| clubb_lite (wp2 only) | C-structural | CLUBB release (larson-group) |
| Smagorinsky (1-D shear proxy) | C-structural | WRF module_diffusion_em km_opt=3 |
| Hines (bulk-amplitude WKB) | C-structural | ICON mo_gw_hines.f90 |
| Sundqvist condensation step | C-structural | ccpp-physics zhaocarr_gscond.f |
| mc3d Monte-Carlo transport | C-structural | I3RC community Monte Carlo model |
| Rayleigh upper sponge | C-structural | none (addition to HS94) |
| gray radiation | **A 27/27** (V20-V22) | Isca two_stream_gray_rad.F90 @ v1.0 5868cfd, FMS stubs |
| Rayleigh HS94 friction | **A 2/2** (V23) | Isca v1.0 hs_forcing.F90 (unmodified) |
| mc3d Tripleclouds overlap | C-structural (V24) | ecRad 1.7.2 radiation_tripleclouds_sw/lw.F90 |
| CAM6 sequencing | C-structural (static audit above) | CAM cam_cesm2_1_rel_60 physpkg.F90 tphysbc/tphysac |
| Beres | BLOCKED (V16) | E3SM gw_convect.F90 needs the real mfcc table (newmfspectra*.nc): not on disk |

## Part 3 (V20-V26)
| unit | comparisons | class | worst port rel / spread | job |
|---|---|---|---|---|
| gray radiation vs Isca v1.0 B_FRIERSON (E0/E1/E2, pstd_mks = p_s, 8 columns) | 27 | A 27 | E1 total heating 4.63e-14 / 6.00e-14 | 11028848 |
| HS94 Rayleigh friction vs Isca v1.0 hs_forcing (sponge off, 13 columns) | 2 | A 2 | du_dt 7.18e-16 / 3.63e-15 | 11028849 |

C1-C5 PASS both.  HS: the uncompared tdt (Newtonian + dissipative heating) changes zero pattern under 1 ulp, so the
V7 family rule fires; no class changes (every port rel <= 1e-12).  Generator self-checks (known values) pass in every
run: gray P2 insolation to 1e-14, zero TOA LW down, constant SW up, zero SW heating at atm_abs = 0; HS zero tendency at
sigma <= sigma_b and -kf (sigma - sigma_b)/(1 - sigma_b) u to 1e-14 kf|u|.  Non-vacuity: gray test fails with
linear_frac off by 1e-6 (E0/E1), HS test fails with the sponge on (job 11029100).
Tripleclouds: C-structural (V24), no harness -- ours is the closed-form grey factor 1 - fsd^2 tau/(gamma0 + tau) with
no region-resolved solver; tier-2 audit (radiation_overlap_inhomogeneity_ozone_mc3d) measured fsd 0->1 TOA SW -8.2 vs
ecRad Tripleclouds -18.2 W/m2.  The IFS ecRad 49r1 port (feat/ifs-ecrad-oracle) ports McICA only; nothing there to share.
CLUBB helpers: tests/oracle_baselines/test_spread_f4_clubb_impl.py (synthetic-anchor splice, missing-anchor refusal,
byte-identical rebuild of the committed gen_spread_impl.py, bitwise_diff/load); fails with the spread_step splice
removed and with value- instead of bitwise equality (job 11028845).
