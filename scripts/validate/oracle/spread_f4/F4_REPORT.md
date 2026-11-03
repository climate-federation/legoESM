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
| mg2 prognostic N_c (whole call, act, sed) | — | UNMEASURED (V8) | C1 FAIL: fixtures built with Spack GCC 13.4 elsewhere; gcc 8.5 differs at round-off | 11025895-97 |

Class C for the F3 forks (report only, not fixed): Louis ICON unstable km/kh; Kuo g32 dT_dt (both partitions).

Not in batch 1: CLUBB implemented arm (namelist I/O, needs its own harness), IFS CUMASTRN (done by the IFS stream: A).
Missing harnesses (part 2), not started: tke, clubb_lite, smagorinsky (no oracle named yet), gray radiation (Isca build),
cam6_physics_sequencing (tphysbc/ac driver), hines (not a port: heuristic, no oracle), rayleigh (self-reference only),
Beres (no running Fortran oracle), mc3d photon transport / Tripleclouds, Sundqvist condensation step (surrogate).
