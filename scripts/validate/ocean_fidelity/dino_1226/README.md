# DINO #1226 screen harness (investigation scratch — issue-scoped, delete when #1226 closes)

Matched-window fidelity screens for the DINO NEMO-match program (issue #1226,
PR #1208). All machine-local NEMO artifacts live on the 2xV100S box under
`~/oracle-builds/nemo5/nemo_5.0.2/cfgs/DINO/`:
- `RUN_TRAJ/mesh_mask.nc` — full-domain mesh (geometry donor for the bridge)
- `RUN_TRAJ/DINO_00000320_restart.nc` — geometry-donor restart
- `RUN_1Y/DINO_1y_00010101_00011230_grid_{T,U}.nc` — NEMO year-1 annual mean (stitched)
- `RUN_5Y/DINO_1y_00050101_00051230_grid_{T,V,U}.nc` — NEMO year-5 mean (stitched)
- `RUN_STEPDUMP/` — per-step restarts kt=5761-5790 + MY_SRC trddyn/trdtra per-term
  trend dumps (day ~180) — the instrument for the channel momentum-budget diff
- NEMO 5.0.2 source: `~/oracle-builds/nemo5/nemo_5.0.2/src/OCE/` (READ THIS, per
  the #1226 philosophy: transcribe, never diagnose from first principles)

## Screens (protocol: 1-year matched window for screening; 5-year ONLY for final certification)

    # 1-year from-rest, census-gated NEMO topo, year-1 mean accumulated:
    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python dino_year_screen.py nemo_dino_kamm_mlf out.npz
    # 90-day stability probe (day 60-90 mean):
    CUDA_VISIBLE_DEVICES=<gpu> JAX_ENABLE_X64=1 python dino_90d_screen.py nemo_dino_kamm_mlf out.npz

Both drivers refuse to run unless the legoESM 3-D wet mask equals NEMO tmask
exactly (topo census gate). ~40 min / ~12 min on one V100S.

## Metrics (year-1 reference values, 2026-07-20, branch aabadc4df)

ACC = median over lon sections of net zonal transport (e3t_1d ladder + e2u,
model's own wet cells — NO cross-model mask). SSH small-scale ratio = std of
(field - 5pt boxcar) over wet cells, lego/NEMO. Current: lego ACC 14.5 Sv vs
NEMO 69.1; SSH ratio 0.51. `dino_year5_compare.py` = the 9-panel climate
comparison (edit paths at top; written for the year-5 pair).
